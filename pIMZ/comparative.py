# general
import math
import logging
import json
import os,sys
from pIMZ.regions import SpectraRegion
import random
from collections import defaultdict, Counter
import glob
import shutil, io, base64

# general package
from natsort import natsorted
import pandas as pd

import numpy as np
from numpy.ctypeslib import ndpointer

from pyimzml.ImzMLParser import ImzMLParser, browse, getionimage
import ms_peak_picker
import regex as re


# image
import skimage
from skimage import measure as sk_measure
from adjustText import adjust_text

# processing
import ctypes
import subprocess
import dill as pickle


#vis
import dabest
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#methods
import umap
import hdbscan
import diffxpy.api as de
import anndata

from scipy import ndimage, misc, sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import squareform, pdist
import scipy.cluster as spc
import scipy as sp
import sklearn as sk

from sklearn.metrics.pairwise import cosine_similarity


#web/html
import jinja2


# applications
import progressbar


class CombinedSpectra():
    """CombinedSpectra class for a combined analysis of several spectra regions.
    """

    def __setlogger(self):
        """Sets up logging facilities for CombinedSpectra.
        """
        self.logger = logging.getLogger('CombinedSpectra')

        if len(self.logger.handlers) == 0:
            self.logger.setLevel(logging.INFO)

            consoleHandler = logging.StreamHandler()
            consoleHandler.setLevel(logging.INFO)

            self.logger.addHandler(consoleHandler)

            formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
            consoleHandler.setFormatter(formatter)

    def __init__(self, regions):
        """Initializes a CombinedSpectra object with the following attributes:
        - logger (logging.Logger): Reference to the Logger object.
        - regions (dict): A dictionary that has SpectraRegion objects names as keys and respective SpectraRegion objects as values. If a SpectraRegion object does not have a name attribute it will be named according to the region id.
        - consensus_similarity_matrix (pandas.DataFrame): Pairwise similarity matrix between consensus spectra of all combinations of regions. Initialized with None.
        - region_cluster2cluster (dict): A dictionary where every tuple (region name, region id) is mapped to its cluster id where it belongs. Initialized with None.
        - region_array_scaled (dict): A dictionary where each SpectraRegion name is mapped to the respective scaled region array either using "avg" (average) or "median" method. Initialized with an empty dict.
        - de_results_all (dict): Methods mapped to their differential analysis results (as pd.DataFrame). Initialized with an empty defaultdict.

        Args:
            regions (dict): A dictionary that maps region ids to respective SpectraRegion objects.
        """

        self.regions = {}
        self.consensus_similarity_matrix = None
        self.region_cluster2cluster = None

        self.region_array_scaled = {}
        self.de_results_all = defaultdict(lambda: dict())
        self.df_results_all = defaultdict(lambda: dict())

        self.logger = None
        self.__setlogger()

        for x in regions:
            
            addregion = regions[x]
            if addregion.name == None:
                addregion.name = x

            self.regions[addregion.name] = regions[x]


    def __get_spectra_similarity(self, vA, vB):
        """Calculates cosine similarity between two vectors of the same length.

        Args:
            vA (numpy.array/list): First vector.
            vB (numpy.array/list): Second vector.

        Returns:
            float: cosine similarity.
        """
        return np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))


    def consensus_similarity(self):        
        """
        Calculates consensus_similarity_matrix of CombinedSpectra object.
        The resulting pandas.DataFrame is a pairwise similarity matrix between consensus spectra of all combinations of regions.

        If the object was not yet scaled, it will get scaled.
        """

        self.check_scaled()

        allConsSpectra = {}

        for regionName in self.region_array_scaled:
            scaled_region = self.region_array_scaled[regionName]
            region = self.regions[regionName]

            regionCS = region.consensus_spectra(array=scaled_region, set_consensus=False)

            for clusterid in regionCS:
                allConsSpectra[(regionName, clusterid)] = regionCS[clusterid]

        allRegionClusters = sorted([x for x in allConsSpectra])

        distDF = pd.DataFrame(0.0, index=allRegionClusters, columns=allRegionClusters)

        for i in range(0, len(allRegionClusters)):
            regionI = allRegionClusters[i]
            for j in range(i, len(allRegionClusters)):

                regionJ = allRegionClusters[j]

                specSim = self.__get_spectra_similarity(allConsSpectra[regionI], allConsSpectra[regionJ])

                distDF[regionI][regionJ] = specSim
                distDF[regionJ][regionI] = specSim

        self.consensus_similarity_matrix = distDF

    def plot_consensus_similarity(self):
        """Plots the similarity matrix represented as seaborn.heatmap.
        """
        sns.heatmap(self.consensus_similarity_matrix, xticklabels=1, yticklabels=1)
        plt.show()
        plt.close()


    def cluster_concensus_spectra(self, number_of_clusters=5):
        """Performs clustering using Ward variance minimization algorithm on similarity matrix of consensus spectra and updates region_cluster2cluster with the results. region_cluster2cluster dictionary maps every tuple (region name, region id) to its cluster id where it belongs. Additionally plots the resulting dendrogram depicting relationships of regions to each other.

        Args:
            number_of_clusters (int, optional): Number of desired clusters. Defaults to 5.
        """
        df = self.consensus_similarity_matrix.copy()
        # Calculate the distance between each sample
        Z = spc.hierarchy.linkage(df.values, 'ward')

        plt.figure(figsize=(8,8))
        # Make the dendro
        spc.hierarchy.dendrogram(Z, labels=df.columns.values, leaf_rotation=0, orientation="left", color_threshold=240, above_threshold_color='grey')

        c = spc.hierarchy.fcluster(Z, t=number_of_clusters, criterion='maxclust')

        lbl2cluster = {}
        region2cluster = {}
        for lbl, clus in zip(df.columns.values, c):
            lbl2cluster[str(lbl)] = clus
            region2cluster[lbl] = clus

        # Create a color palette with 3 color for the 3 cyl possibilities
        my_palette = plt.cm.get_cmap("viridis", number_of_clusters)
                
        # Apply the right color to each label
        ax = plt.gca()
        xlbls = ax.get_ymajorticklabels()
        for lbl in xlbls:
            val=lbl2cluster[lbl.get_text()]
            #print(lbl.get_text() + " " + str(val))

            lbl.set_color(my_palette(val-1))

        plt.show()
        plt.close()

        self.region_cluster2cluster = region2cluster

    def check_scaled(self):
        """Detects not scaled region arrays and norms them using "median" method.
        """
        hasToReprocess = False
        for regionName in self.regions:
            if not regionName in self.region_array_scaled:
                hasToReprocess = True
                break

        if hasToReprocess:
            self.logger.info("Calculating internormed regions")
            self.get_internormed_regions()


    def mass_intensity(self, masses, regions=None, scaled=False, verbose=True):
        """Plots seaborn.boxplot for every selected region depicting the range of intensity values in each cluster.

        Args:
            masses (float/list/tuple/set): Desired mass(es).
            regions (list/numpy.array, optional): Desired regions where to look for mass intensities. Defaults to None meaning to consider all available regions.
            scaled (bool, optional): Whether to use intensity values of scaled region arrays. Defaults to False.
            verbose (bool, optional): Whether to add information to the logger. Defaults to True.
        """
        if not isinstance(masses, (list, tuple, set)):
            masses = [masses]

        if scaled:
            self.check_scaled()

        for regionName in self.regions:

            if not regions is None and not regionName in regions:
                continue

            cregion = self.regions[regionName]

            cluster2coords = cregion.getCoordsForSegmented()

            if not scaled:
                dataArray = cregion.region_array
            else:
                dataArray = self.region_array_scaled[regionName]

            for mass in masses:
                
                bestExMassForMass, bestExMassIdx = cregion._get_exmass_for_mass(mass)
                if verbose:
                    self.logger.info("Processing Mass {} with best existing mass {}".format(mass, bestExMassForMass))

                clusterIntensities = defaultdict(list)

                for clusterid in cluster2coords:
                    for coord in cluster2coords[clusterid]:
                        intValue = dataArray[coord[0], coord[1], bestExMassIdx]
                        clusterIntensities[clusterid].append(intValue)


                clusterVec = []
                intensityVec = []
                massVec = []
                specIdxVec = []
                for x in clusterIntensities:
                    
                    elems = clusterIntensities[x]
                    specIdxVec += [i for i in range(0, len(elems))]

                    clusterVec += ["Cluster " + str(x)] * len(elems)
                    intensityVec += elems
                    massVec += [mass] * len(elems)
                        
                dfObj = pd.DataFrame({"mass": massVec, "specidx": specIdxVec, "cluster": clusterVec, "intensity": intensityVec})
                sns.boxplot(data=dfObj, x="cluster", y="intensity")
                plt.xticks(rotation=90)
                plt.title("Intensities for Region {} ({}m/z)".format(regionName, mass))
                plt.show()
                plt.close()



    def mass_heatmap(self, masses, log=False, min_cut_off=None, plot=True, scaled=False, verbose=True, title="{mz}"):
        """Plots heatmap for every selected region depicting region_array spectra reduced to the sum of the specified masses.

        Args:
            masses (float/list/tuple/set): Desired mass(es).
            log (bool, optional): Whether to take logarithm of the output matrix. Defaults to False.
            min_cut_off (int/float, optional): Lower limit of values in the output matrix. Smaller values will be replaced with min_cut_off. Defaults to None.
            plot (bool, optional):  Whether to plot the output matrix. Defaults to True.
            scaled (bool, optional): Whether to use intensity values of scaled region arrays. Defaults to False.
            verbose (bool, optional): Whether to add information to the logger. Defaults to True.
            title (str, optional): Format string defining the plot's title.

        Returns:
            numpy.array: A matrix of the last region where each element is a sum of intensities at given masses. 
        """

        if not isinstance(masses, (list, tuple, set)):
            masses = [masses]

        if scaled:
            self.check_scaled()

        region2segments = {}
        for regionName in self.regions:

            cregion = self.regions[regionName]

            if scaled == False:
                dataArray = self.regions[regionName].region_array
            else:
                dataArray = self.region_array_scaled[regionName]

            image = np.zeros((dataArray.shape[0], dataArray.shape[1]))

            for mass in masses:
                
                bestExMassForMass, bestExMassIdx = cregion._get_exmass_for_mass(mass)
                if verbose:
                    self.logger.info("Processing Mass {} with best existing mass {}".format(mass, bestExMassForMass))

                for i in range(dataArray.shape[0]):
                    for j in range(dataArray.shape[1]):

                        image[i,j] += dataArray[i,j,bestExMassIdx]


            if log:
                image = np.log(image)

            if min_cut_off != None:
                image[image <= min_cut_off] = min_cut_off

            region2segments[regionName] = image

        if plot:


            rows = math.ceil(len(self.regions) / 2)
            fig, axes = plt.subplots(rows, 2)
            if len(axes.shape) > 1:
                axes = np.reshape(axes, (1, axes.shape[0] * axes.shape[1]))[0][:]


            allMin, allMax = 0,0

            for regionName in region2segments:
                allMin = min(allMin, np.min(region2segments[regionName]))
                allMax = max(allMax, np.max(region2segments[regionName]))

            didx = 0
            for didx, regionName in enumerate(region2segments):
                ax = axes[didx]

                heatmap = ax.matshow(region2segments[regionName], vmin=allMin, vmax=allMax)

                # We must be sure to specify the ticks matching our target names
                ax.set_title(regionName, color="w", y=0.1)

            for ddidx in range(didx+1, rows*2):
                ax = axes[ddidx]
                ax.axis('off')

            fig.colorbar(heatmap, ax=axes[-1])
            plt.suptitle(title.format(mz=";".join([str(round(x, 3)) if not type(x) in [str] else x for x in masses])))

            plt.show()
            plt.close()


        return image

    def plot_segments(self, highlight=None):
        """Plots segmented arrays of all regions as heatmaps.

        Args:
            highlight (list/tuple/set/int, optional): If cluster ids are specified here, the resulting clustering will have cluster id 2 for highlight clusters, cluster id 0 for background, and cluster id 1 for the rest. Defaults to None.
        """
        assert(not self.region_cluster2cluster is None)

        allClusters = [self.region_cluster2cluster[x] for x in self.region_cluster2cluster]
        valid_vals = sorted(set(allClusters))


        region2segments = {}
        for regionName in self.regions:
            origSegments = np.array(self.regions[regionName].segmented, copy=True)           
            region2segments[regionName] = origSegments


        if highlight != None:
            if not isinstance(highlight, (list, tuple, set)):
                highlight = [highlight]

            for regionName in region2segments:

                showcopy = np.copy(region2segments[regionName])
                
                for i in range(0, showcopy.shape[0]):
                    for j in range(0, showcopy.shape[1]):

                        if showcopy[i,j] != 0:

                            if showcopy[i,j] in highlight:
                                showcopy[i,j] = 2
                            elif showcopy[i,j] != 0:
                                showcopy[i,j] = 1

                region2segments[regionName] = showcopy

        self._plot_arrays(region2segments)


    def plot_common_segments(self, highlight=None):
        """Plots segmented arrays of every region annotating the clusters with respect to new clustering done with CombinedSpectra (saved in region_cluster2cluster).

        Args:
            highlight (list/tuple/set/int, optional):  If cluster ids are specified here, the resulting clustering will have cluster id 2 for highlight clusters, cluster id 0 for background, and cluster id 1 for the rest. Defaults to None.
        """
        assert(not self.region_cluster2cluster is None)

        allClusters = [self.region_cluster2cluster[x] for x in self.region_cluster2cluster]
        valid_vals = sorted(set(allClusters))


        region2segments = {}
        for regionName in self.regions:
            origSegments = np.array(self.regions[regionName].segmented, copy=True)

            origCluster2New = {}

            for x in self.region_cluster2cluster:
                if x[0] == regionName:
                    origCluster2New[x[1]] = self.region_cluster2cluster[x]

            newSegments = np.zeros(origSegments.shape)

            print(origCluster2New)
            for i in range(0, newSegments.shape[0]):
                for j in range(0, newSegments.shape[1]):
                    newSegments[i,j] = origCluster2New.get(origSegments[i,j], 0)
            
            region2segments[regionName] = newSegments


        if highlight != None:
            if not isinstance(highlight, (list, tuple, set)):
                highlight = [highlight]

            for regionName in region2segments:

                showcopy = np.copy(region2segments[regionName])
                
                for i in range(0, showcopy.shape[0]):
                    for j in range(0, showcopy.shape[1]):

                        if showcopy[i,j] != 0:

                            if showcopy[i,j] in highlight:
                                showcopy[i,j] = 2
                            elif showcopy[i,j] != 0:
                                showcopy[i,j] = 1

                region2segments[regionName] = showcopy

        self._plot_arrays(region2segments)

    def _plot_arrays(self, region2segments):
        """Plots heatmaps for every region given in region2segments.

        Args:
            region2segments (dict): A dictionary with region names as keys and respective segmented arrays as values.
        """
        rows = math.ceil(len(region2segments) / 2)
        fig, axes = plt.subplots(rows, 2)

        valid_vals = set()
        for regionName in region2segments:
            plotarray = region2segments[regionName]

            valid_vals = valid_vals.union(list(np.unique(plotarray)))

        valid_vals = sorted(valid_vals)
        min_ = min(valid_vals)
        max_ = max(valid_vals)

        positions = np.linspace(min_, max_, len(valid_vals))
        val_lookup = dict(zip(positions, valid_vals))
        print(val_lookup)

        def formatter_func(x, pos):
            'The two args are the value and tick position'
            val = val_lookup[x]
            return val


        if len(axes.shape) > 1:
            axes = np.reshape(axes, (1, axes.shape[0] * axes.shape[1]))[0][:]

        didx=0
        for didx, regionName in enumerate(region2segments):
            ax = axes[didx]

            im = ax.matshow(region2segments[regionName], cmap=plt.cm.get_cmap('viridis', len(valid_vals)), vmin=min_, vmax=max_)
            formatter = plt.FuncFormatter(formatter_func)

            # We must be sure to specify the ticks matching our target names
            ax.set_title(regionName, color="w", y=0.9, x=0.1)

        for ddidx in range(didx+1, rows*2):
            ax = axes[ddidx]
            ax.axis('off')

        plt.colorbar(im, ax=axes[:], ticks=positions, format=formatter, spacing='proportional')

        plt.show()
        plt.close()

    def __make_de_res_key(self, region0, clusters0, region1, clusters1):
        """Generates the storage key for two sets of clusters.

        Args:
            region0 (int): first region id.
            clusters0 (list): list of cluster ids 1.
            region1 (int): second region id.
            clusters1 (list): list of cluster ids 2.

        Returns:
            tuple: tuple (region0, sorted clusters0, region1, sorted clusters1)
        """
        return (region0, tuple(sorted(clusters0)), region1, tuple(sorted(clusters1)))

    def to_region_cluster_input(self, region_cluster_list):

        rcl0 = defaultdict(list)

        for x in region_cluster_list:
            rcl0[x[0]].append(x[1])

        rcl0 = [(x, tuple(sorted(rcl0[x]))) for x in rcl0]
        return rcl0

    def find_markers(self, region_cluster_list0, region_cluster_list1, protWeights, mz_dist=3, mz_best=False, use_methods = ["empire", "ttest", "rank"], count_scale={"ttest": 1, "rank": 1}, scaled=True, sample_max=-1):
        """Performs differential analysis to finds marker proteins for specific regions and clusters.

        Args:
            region_cluster_list0 (list/numpy.array): A list of tuples (region id, list of clusters) that will be used as the 0 conditional vector by differential analysis.
            region_cluster_list1 (list/numpy.array): A list of tuples (region id, list of clusters) that will be used as the 1 conditional vector by differential analysis.
            protWeights (ProteinWeights): ProteinWeights object for translation of masses to protein names.
            mz_dist (float/int, optional): Allowed offset for protein lookup of needed masses. Defaults to 3.
            mz_best (bool, optional): Wether to consider only the closest found protein within mz_dist (with the least absolute mass difference). Defaults to False.
            use_methods (str/list, optional): Test method(s) for differential expression. Defaults to ["empire", "ttest", "rank"].\n
                - "empire": Empirical and Replicate based statistics (EmpiRe).\n
                - "ttest": Welch’s t-test for differential expression using diffxpy.api.\n
                - "rank": Mann-Whitney rank test (Wilcoxon rank-sum test) for differential expression using diffxpy.api.\n
            count_scale (dict, optional): Count scales for different methods (relevant for empire, which can only use integer counts). Defaults to {"ttest": 1, "rank": 1}.
            scaled (bool, optional): Wether each processed region is normalized. Those which are not will be scaled with the median method. Defaults to True.
            sample_max (int, optional): Allowed number of samples (spectra of specified regions&clusters) will be used by differential analysis (will be randomly picked if there are more available than allowed). Defaults to -1 meaning all samples are used.

        Returns:
            tuple: Tuple (collections.defaultdict, pandas.core.frame.DataFrame, pandas.core.frame.DataFrame). Dictionary with test method mapped to each tuple (region, clusters) and respective results. Two further data frames with expression data and test design.
        """

        if type(region_cluster_list0) in (list, tuple):
            region_cluster_list0 = self.to_region_cluster_input(region_cluster_list0)

        if type(region_cluster_list1) in (list, tuple):
            region_cluster_list1 = self.to_region_cluster_input(region_cluster_list1)


        for pair in region_cluster_list0:
            assert(pair[0] in self.regions)
            assert([x for x in self.regions[region_cluster_list0[0][0]].idx2mass] == [x for x in self.regions[pair[0]].idx2mass])

        for pair in region_cluster_list1:
            assert(pair[0] in self.regions)
            assert([x for x in self.regions[region_cluster_list1[0][0]].idx2mass] == [x for x in self.regions[pair[0]].idx2mass])


        cluster2coords0 = {}

        for pair in region_cluster_list0:
            cluster2coords0[pair[0]] = self.regions[pair[0]].getCoordsForSegmented()
            assert(all([x in cluster2coords0[pair[0]] for x in pair[1]]))

        cluster2coords1 = {}

        for pair in region_cluster_list1:
            cluster2coords1[pair[0]] = self.regions[pair[0]].getCoordsForSegmented()
            assert(all([x in cluster2coords1[pair[0]] for x in pair[1]]))

        resKey = self.__make_de_res_key(region_cluster_list0[0][0], region_cluster_list0[0][1], region_cluster_list1[0][0], region_cluster_list1[0][1])

        if scaled:
            self.check_scaled()

        if self.de_results_all is None:
            self.de_results_all = defaultdict(lambda: dict())
        
        sampleVec = []
        conditionVec = []

        exprData = pd.DataFrame()

        for pair in region_cluster_list0:
            region0 = pair[0]
            clusters0 = pair[1]
            
            masses = [("mass_" + str(x)).replace(".", "_") for x in self.regions[region0].idx2mass]
            for clus in clusters0:

                allPixels = cluster2coords0[region0][clus]

                self.logger.info("Processing region {} cluster: {}".format(region0, clus))
                bar = progressbar.ProgressBar()

                if scaled:
                    dataArray = self.region_array_scaled[region0]
                else:
                    dataArray = self.regions[region0].region_array


                if sample_max > 0 and len(allPixels) > sample_max:
                    allPixels = random.sample(allPixels, sample_max)

                for pxl in bar(allPixels):
                    pxl_name = "{}__{}__{}".format(region0, str(len(sampleVec)), "_".join([str(x) for x in pxl]))
                    sampleVec.append(pxl_name)
                    conditionVec.append(0)

                    exprData[pxl_name] = dataArray[pxl[0], pxl[1], :]#.astype('int')

        for pair in region_cluster_list1:
            region1 = pair[0]
            clusters1 = pair[1]
            for clus in clusters1:
                self.logger.info("Processing region {} cluster: {}".format(region1, clus))

                allPixels = cluster2coords1[region1][clus]
                
                bar = progressbar.ProgressBar()

                if scaled:
                    dataArray = self.region_array_scaled[region1]
                else:
                    dataArray = self.regions[region1].region_array

                if sample_max > 0 and len(allPixels) > sample_max:
                    allPixels = random.sample(allPixels, sample_max)

                for pxl in bar(allPixels):
                    pxl_name = "{}__{}__{}".format(region1, str(len(sampleVec)), "_".join([str(x) for x in pxl]))
                    sampleVec.append(pxl_name)
                    conditionVec.append(1)

                    exprData[pxl_name] = dataArray[pxl[0], pxl[1], :]#.astype('int')


        self.logger.info("DE DataFrame ready. Shape {}".format(exprData.shape))

        pData = pd.DataFrame()

        pData["sample"] = sampleVec
        pData["condition"] = conditionVec
        pData["batch"] = 0

        self.logger.info("DE Sample DataFrame ready. Shape {}".format(pData.shape))


        diffxPyTests = set(["ttest", "rank"])
        if len(diffxPyTests.intersection(use_methods)) > 0:

            for testMethod in use_methods:

                if not testMethod in diffxPyTests:
                    continue

                self.logger.info("Performing DE-test: {}".format(testMethod))

                ttExprData = exprData.copy(deep=True)

                if count_scale != None and testMethod in count_scale:
                    ttExprData = ttExprData * count_scale[testMethod]

                    if count_scale[testMethod] > 1:
                        ttExprData = ttExprData.astype(int)

                pdat = pData.copy()
                del pdat["sample"]

                deData = anndata.AnnData(
                    X=exprData.values.transpose(),
                    var=pd.DataFrame(index=masses),
                    obs=pdat
                )

                if testMethod == "ttest":

                    test = de.test.t_test(
                        data=deData,
                        grouping="condition"
                    )

                elif testMethod == "rank":
                    test = de.test.rank_test(
                        data=deData,
                        grouping="condition"
                    )

                self.de_results_all[testMethod][resKey] = test.summary()
                self.logger.info("DE-test ({}) finished. Results available: {}".format(testMethod, resKey))

        deresDFs = defaultdict(lambda: dict())

        for test in self.de_results_all:
            for rkey in self.de_results_all[test]:

                deresDFs[test][rkey] = self.deres_to_df(self.de_results_all[test][rkey], rkey, protWeights, mz_dist=mz_dist, mz_best=mz_best, keepOnlyProteins=protWeights != None, scaled=scaled)

        self.df_results_all = deresDFs

        return deresDFs, exprData, pData


    def list_de_results(self):
        """Transforms a dictionary of the differential expression results into a list.

        Returns:
            list: A list of tuples where the first element is the name of the used method and the second - all compared sets of clusters.
        """
        allDERes = []
        for x in self.de_results_all:
            for y in self.de_results_all[x]:
                allDERes.append((x,y))

        return allDERes

    def get_spectra_matrix(self, region_array, segments, cluster2coords):
        """Returns a matrix with all spectra that correspond to the given segments.

        Args:
            region_array (numpy.array): Array of spectra.
            segments (numpy.array/list): A list of desired cluster ids.
            cluster2coords (dict): Each cluster ids mapped to the corresponding coordinates in region_array.

        Returns:
            numpy.array: An array where each element is spectrum that is part of one of the given clusters from segments.
        """
        relPixels = []
        for x in segments:
            relPixels += cluster2coords.get(x, [])

        spectraMatrix = np.zeros((len(relPixels), region_array.shape[2]))

        #print(spectraMatrix.shape)
        #print(region_array.shape)

        #print(np.min([x[0] for x in relPixels]), np.max([x[0] for x in relPixels]))
        #print(np.min([x[1] for x in relPixels]), np.max([x[1] for x in relPixels]))

        for pidx, px in enumerate(relPixels):
            spectraMatrix[pidx, :] = region_array[px[0], px[1], :]

        return spectraMatrix


    def deres_to_df(self, deResDF, resKey, protWeights, mz_dist=3, mz_best=False, keepOnlyProteins=True, inverse_fc=False, max_adj_pval=0.05, min_log2fc=0.5, scaled=True):
        """Transforms differetial expression (de) result into a DataFrame form.

        Args:
            deResDF (pandas.DataFrame): A DataFrame that summarizes de result. 
            resKey (tuple): List of regions where to look for the result.
            protWeights (ProteinWeights): ProteinWeights object for translation of masses to protein name.
            mz_dist (float/int, optional): Allowed offset for protein lookup of needed masses. Defaults to 3.
            mz_best (bool, optional): Wether to consider only the closest found protein within mz_dist (with the least absolute mass difference). Defaults to False.
            keepOnlyProteins (bool, optional): If True, differential masses without protein name will be removed. Defaults to True.
            inverse_fc (bool, optional): If True, the de result logFC will be inversed (negated). Defaults to False.
            max_adj_pval (float, optional): Threshold for maximum adjusted p-value that will be used for filtering of the de results. Defaults to 0.05.
            min_log2fc (float, optional): Threshold for minimum log2fc that will be used for filtering of the de results. Defaults to 0.5.
            scaled (bool, optional):Whether to use intensity values of scaled region arrays. Defaults to True.

        Returns:
            pandas.DataFrame: DataFrame of differetial expression (de) result.
        """

        if scaled:
            self.check_scaled()

        clusterVec = []
        geneIdentVec = []
        massVec = []
        foundProtVec = []
        lfcVec = []
        qvalVec = []
        detMassVec = []

        avgExpressionVec = []
        medianExpressionVec = []
        totalSpectraVec = []
        measuredSpectraVec = []

        avgExpressionBGVec = []
        medianExpressionBGVec = []
        totalSpectraBGVec = []
        measuredSpectraBGVec = []

        ttr = deResDF.copy(deep=True)#self.de_results[resKey]      
        self.logger.info("DE result for case {} with {} results".format(resKey, ttr.shape))
        #ttr = deRes.summary()

        log2fcCol = "log2fc"
        massCol = "gene"
        adjPvalCol = "qval"

        ttrColNames = list(ttr.columns.values)


        if log2fcCol in ttrColNames and massCol in ttrColNames and adjPvalCol in ttrColNames:
            dfColType = "diffxpy"

        else:
            #id	numFeatures	pval	abs.log2FC	log2FC	fdr	SDcorr	fc.pval	fc.fdr	nonde.fcwidth	fcCI.90.start	fcCI.90.end	fcCI.95.start	fcCI.95.end	fcCI.99.start	fcCI.99.end
            if "id" in ttrColNames and "log2FC" in ttrColNames and "fc.fdr" in ttrColNames:
                log2fcCol = "log2FC"
                massCol = "id"
                adjPvalCol = "fc.fdr"
                dfColType = "empire"


        if inverse_fc:
            self.logger.info("DE result logFC inversed")
            ttr[log2fcCol] = -ttr[log2fcCol]

        fttr = ttr[ttr[adjPvalCol].lt(max_adj_pval) & ttr[log2fcCol].abs().gt(min_log2fc)]

        self.logger.info("DE result for case {} with {} results (filtered)".format(resKey, fttr.shape))

        if scaled:
            targetDataArray = self.region_array_scaled[resKey[0]]
        else:
            targetDataArray = self.regions[resKey[0]].region_array

        targetSpectraMatrix = self.get_spectra_matrix(targetDataArray, resKey[1], self.regions[resKey[0]].getCoordsForSegmented())

        if scaled:
            bgDataArray = self.region_array_scaled[resKey[2]]
        else:
            bgDataArray = self.regions[resKey[2]].region_array

        bgSpectraMatrix = self.get_spectra_matrix(bgDataArray, resKey[3], self.regions[resKey[2]].getCoordsForSegmented())

        self.logger.info("Created matrices with shape {} and {} (target, bg)".format(targetSpectraMatrix.shape, bgSpectraMatrix.shape))

        
        for row in fttr.iterrows():
            geneIDent = row[1][massCol]
            
            ag = geneIDent.split("_")
            massValue = float("{}.{}".format(ag[1], ag[2]))

            foundProt = []
            if protWeights != None:
                foundProt = protWeights.get_protein_from_mass(massValue, maxdist=mz_dist)

                if mz_best and len(foundProt) > 0:
                    foundProt = [foundProt[0]]
                    

            if keepOnlyProteins and len(foundProt) == 0:
                continue

            lfc = row[1][log2fcCol]
            qval = row[1][adjPvalCol]

            expT, totalSpectra, measuredSpecta = self.regions[resKey[0]].get_expression_from_matrix(targetSpectraMatrix, massValue, resKey[0], ["avg", "median"])
            exprBG, totalSpectraBG, measuredSpectaBG = self.regions[resKey[2]].get_expression_from_matrix(bgSpectraMatrix, massValue, resKey[2], ["avg", "median"])

            avgExpr, medianExpr = expT
            avgExprBG, medianExprBG = exprBG

            if len(foundProt) > 0:

                for protMassTuple in foundProt:
                    
                    prot,protMass = protMassTuple
            
                    clusterVec.append("_".join([str(resKey[0])]+[str(x) for x in resKey[1]]))
                    geneIdentVec.append(geneIDent)
                    massVec.append(massValue)
                    foundProtVec.append(prot)
                    detMassVec.append(protMass)
                    lfcVec.append(lfc)
                    qvalVec.append(qval)

                    avgExpressionVec.append(avgExpr)
                    medianExpressionVec.append(medianExpr)
                    totalSpectraVec.append(totalSpectra)
                    measuredSpectraVec.append(measuredSpecta)

                    avgExpressionBGVec.append(avgExprBG)
                    medianExpressionBGVec.append(medianExprBG)
                    totalSpectraBGVec.append(totalSpectraBG)
                    measuredSpectraBGVec.append(measuredSpectaBG)

            else:
                clusterVec.append("_".join([str(resKey[0])]+[str(x) for x in resKey[1]]))
                geneIdentVec.append(geneIDent)
                massVec.append(massValue)
                foundProtVec.append("")
                detMassVec.append("-1")
                lfcVec.append(lfc)
                qvalVec.append(qval)

                avgExpressionVec.append(avgExpr)
                medianExpressionVec.append(medianExpr)
                totalSpectraVec.append(totalSpectra)
                measuredSpectraVec.append(measuredSpecta)

                avgExpressionBGVec.append(avgExprBG)
                medianExpressionBGVec.append(medianExprBG)
                totalSpectraBGVec.append(totalSpectraBG)
                measuredSpectraBGVec.append(measuredSpectaBG)


        #requiredColumns = ["gene", "clusterID", "avg_logFC", "p_val_adj", "mean", "num", "anum"]
        """
        print("clusterID", len(clusterVec))
        print("gene_ident", len(geneIdentVec))
        print("gene_mass", len(massVec))
        print("gene", len(foundProtVec))
        print("protein_mass", len(detMassVec))
        print("avg_logFC", len(lfcVec))
        print("qvalue", len(qvalVec))

        print("num", len(totalSpectraVec))
        print("anum", len(measuredSpectraVec))
        print("mean", len(avgExpressionVec))
        print("median", len(medianExpressionVec))

        print("num_bg", len(totalSpectraBGVec))
        print("anum_bg", len(measuredSpectraBGVec))
        print("mean_bg", len(avgExpressionBGVec))
        print("median_bg", len(medianExpressionBGVec))

        """

        df = pd.DataFrame()
        df["clusterID"] = clusterVec
        df["gene_ident"] = geneIdentVec
        df["gene_mass"] = massVec
        df["gene"] = foundProtVec
        df["protein_mass"] = detMassVec
        df["avg_logFC"] = lfcVec
        df["qvalue"] = qvalVec
        
        df["num"] = totalSpectraVec
        df["anum"]= measuredSpectraVec
        df["mean"] = avgExpressionVec
        df["median"] = medianExpressionVec

        df["num_bg"] = totalSpectraBGVec
        df["anum_bg"]= measuredSpectraBGVec
        df["mean_bg"] = avgExpressionBGVec
        df["median_bg"] = medianExpressionBGVec

        return df

    def plot_volcano(self, method, comparison, title, outfile=None, topn=30, masses=None, gene_names=None, only_selected=False):
        """Plots a volcano plot representing the differential analysis results of the current object.

        Args:
            method (str): Test method for differential expression analysis. “empire”, “ttest” or “rank”.
            comparison (tuple): A tuple of two tuples each consisting of cluster ids compared.
            title ((str): Title of the resulting plot.
            outfile (str, optional): The path where to save the resulting plot. Defaults to None.
            topn (int, optional): Number of the most significantly up/dowm regulated genes. Defaults to 30.
            masses (list, optional): A collection of floats that represent the desired masses to be labled. Defaults to None.
            gene_names (list, optional): A collection of strings that represent the desired gene names to be labled. Defaults to None.
            only_selected (bool, optional): Whether to plot all results and highlight the selected masses/genes (=False) or plot only selectred masses/genes (=True). Defaults to False.
        """
        dataframe = pd.merge(self.df_results_all[method][comparison], self.de_results_all[method][comparison], left_on=['gene_ident'],right_on=['gene'])
        genes = ['{:.4f}'.format(x) for x in list(dataframe['gene_mass'])]
        if masses:
            if only_selected:
                dataframe = dataframe.loc[dataframe['gene_mass'].isin(masses)]
            genes = ['{:.4f}'.format(x) for x in list(dataframe['gene_mass'])]
        if gene_names:
            if only_selected:
                dataframe = dataframe.loc[dataframe['gene_x'].isin(gene_names)]
            genes = list(dataframe['gene_x'])
        fc = list(dataframe['log2fc'])
        pval = list(dataframe['pval'])
        FcPvalGene = [(fc[i], pval[i], genes[i]) for i in range(len(genes))]
        if topn>0:
            SpectraRegion._plot_volcano(FcPvalGene, title, outfile, showGeneCount=topn, showGene=gene_names)
        else:
            SpectraRegion._plot_volcano(FcPvalGene, title, outfile, showGeneCount=len(genes), showGene=gene_names)

    def _fivenumber(self, valuelist):
        """Creates five number statistics for values in valuelist

        Args:
            valuelist (list/tuple/nupmy.array (1D)): list of values to use for statistics

        Returns:
            tuple: len, len>0, min, 25-quantile, 50-quantile, 75-quantile, max
        """

        min_ = np.min(valuelist)
        max_ = np.max(valuelist)

        (quan25_, quan50_, quan75_) = np.quantile(valuelist, [0.25, 0.5, 0.75])

        return (len(valuelist), len([x for x in valuelist if x > 0]), min_, quan25_, quan50_, quan75_, max_)


    def get_internormed_regions(self, method="median"):
        """
            Scales region arrays with either average or median of the fold changes. Updates region_array_scaled. Additionally plots the range of scaled fold changes with a boxplot.
            # TODO: specify which clusters are used for normalization! (dict: region => clusterids)

        Args:
            method (str, optional): Method that is supposed to be used for consensus spectra calculation. Either "avg" (average) or "median". Defaults to "median".
        """
        assert (method in ["avg", "median"])

        allRegionNames = [x for x in self.regions]

        # get reference background spec
        referenceMedianSpectra = self.regions[allRegionNames[0]].consensus_spectra(method="median", set_consensus=False)

        self.region_array_scaled[allRegionNames[0]] = np.copy(self.regions[allRegionNames[0]].region_array)

        print(allRegionNames)

        fcDict = {}
        bar = progressbar.ProgressBar(maxval=len(allRegionNames))
        for rIdx, regionName in bar(enumerate(allRegionNames)):

            if rIdx == 0:
                # this is the reference =)
                continue

            regionElement = self.regions[regionName]
            regionMedianSpectra = regionElement.consensus_spectra(method="median", set_consensus=False)

            scaledRegionArray = np.array(regionElement.region_array, copy=True)

            bgFoldChanges = referenceMedianSpectra[0] / regionMedianSpectra[0]

            fcDict["{}_before".format(regionName)] = bgFoldChanges

            if method == "avg":
                scaleFactor = np.mean(bgFoldChanges)
            elif method == "median":
                scaleFactor = np.median(bgFoldChanges)

            self.logger.info("FiveNumber Stats for bgFoldChanges before: {}".format(self._fivenumber(bgFoldChanges)))
            self.logger.info("scaleFactor: {}".format(scaleFactor))

            scaledRegionArray = regionElement.region_array * scaleFactor


            scaledRegionMedianSpectra = regionElement.consensus_spectra(method="median", set_consensus=False, array=scaledRegionArray)
            scaledbgFoldChanges = referenceMedianSpectra[0] / scaledRegionMedianSpectra[0]
            self.logger.info("FiveNumber Stats for scaledbgFoldChanges after: {}".format(self._fivenumber(scaledbgFoldChanges)))
            fcDict["{}_after".format(regionName)] = scaledbgFoldChanges 

            self.region_array_scaled[regionName] = scaledRegionArray

        fig, ax = plt.subplots()
        ax.boxplot(fcDict.values())
        ax.set_xticklabels(fcDict.keys())
        plt.xticks(rotation=90)
        plt.show()
        plt.close()