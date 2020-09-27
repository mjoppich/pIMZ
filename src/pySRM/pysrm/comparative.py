import numpy as np
from scipy import misc
import ctypes
import dabest
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os,sys
import imageio
from PIL import Image
from natsort import natsorted
import subprocess
from collections import defaultdict, Counter
from pyimzml.ImzMLParser import ImzMLParser, browse, getionimage
import logging
import dill as pickle
import math
import scipy.ndimage as ndimage
import diffxpy.api as de
import anndata
import progressbar
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import sparse
from scipy.sparse.linalg import spsolve

import ms_peak_picker


baseFolder = str(os.path.dirname(os.path.realpath(__file__)))
lib = ctypes.cdll.LoadLibrary(baseFolder+'/../../cppSRM/lib/libSRM.so')

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform, pdist

import scipy.cluster as spc

import scipy as sp
import sklearn as sk

import umap
import hdbscan


class CombinedSpectra():

    def __setlogger(self):
        self.logger = logging.getLogger('CombinedSpectra')

        if len(self.logger.handlers) == 0:
            self.logger.setLevel(logging.INFO)

            consoleHandler = logging.StreamHandler()
            consoleHandler.setLevel(logging.INFO)

            self.logger.addHandler(consoleHandler)

            formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
            consoleHandler.setFormatter(formatter)

    def __init__(self, regions):

        self.regions = {}
        self.consensus_similarity_matrix = None
        self.region_cluster2cluster = None

        self.logger = None
        self.__setlogger()

        for x in regions:
            
            addregion = regions[x]
            if addregion.name == None:
                addregion.name = x

            self.regions[addregion.name] = regions[x]


    def __get_spectra_similarity(self, vA, vB):
        return np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))


    def consensus_similarity(self):
        
        allConsSpectra = {}

        for regionName in self.regions:
            region = self.regions[regionName]

            regionCS = region.consensus_spectra()

            for clusterid in regionCS:
                allConsSpectra[(region.name, clusterid)] = regionCS[clusterid]

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
            sns.heatmap(self.consensus_similarity_matrix)
            plt.show()
            plt.close()


    def cluster_concensus_spectra(self, number_of_clusters=5):
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
        my_palette = plt.cm.get_cmap("Accent", 5)
        
        # transforme the 'cyl' column in a categorical variable. It will allow to put one color on each level.
        df['cat']=pd.Categorical(c)
        my_color=df['cat'].cat.codes
        
        # Apply the right color to each label
        ax = plt.gca()
        xlbls = ax.get_ymajorticklabels()
        num=-1
        for lbl in xlbls:
            num+=1
            val=lbl2cluster[lbl.get_text()]
            lbl.set_color(my_palette(val))

        plt.show()
        plt.close()

        self.region_cluster2cluster = region2cluster

    def mass_heatmap(self, masses, log=False, min_cut_off=None, plot=True):

        if not isinstance(masses, (list, tuple, set)):
            masses = [masses]

        region2segments = {}
        for regionName in self.regions:

            cregion = self.regions[regionName]

            image = np.zeros((cregion.region_array.shape[0], cregion.region_array.shape[1]))

            for mass in masses:
                
                bestExMassForMass, bestExMassIdx = cregion._get_exmass_for_mass(mass)
                self.logger.info("Processing Mass {} with best existing mass {}".format(mass, bestExMassForMass))

                for i in range(cregion.region_array.shape[0]):
                    for j in range(cregion.region_array.shape[1]):

                        image[i,j] += cregion.region_array[i,j,bestExMassIdx]


            if log:
                image = np.log(image)

            if min_cut_off != None:
                image[image <= min_cut_off] = min_cut_off

            region2segments[regionName] = image

        if plot:

            fig = plt.figure(figsize=(12, 12))

            grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                            nrows_ncols=(1,len(region2segments)),
                            axes_pad=0.15,
                            share_all=True,
                            cbar_location="right",
                            cbar_mode="single",
                            cbar_size="7%",
                            cbar_pad=0.15,
                            )

            # Add data to image grid
            axes = [ax for ax in grid]

            allMin, allMax = 0,0

            for regionName in region2segments:
                allMin = min(allMin, np.min(region2segments[regionName]))
                allMax = max(allMax, np.max(region2segments[regionName]))

            for didx, regionName in enumerate(region2segments):
                ax = axes[didx]

                heatmap = ax.matshow(region2segments[regionName], vmin=allMin, vmax=allMax)

                # We must be sure to specify the ticks matching our target names
                ax.set_title(regionName, y=0.1)


            ax.cax.colorbar(heatmap)
            ax.cax.toggle_label(True)

            plt.show()
            plt.close()


        return image



    def plot_common_segments(self, highlight=None):

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


        fig = plt.figure(figsize=(12, 12))


        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                        nrows_ncols=(1,len(region2segments)),
                        axes_pad=0.15,
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="single",
                        cbar_size="7%",
                        cbar_pad=0.15,
                        )

        # Add data to image grid
        axes = [ax for ax in grid]

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

        for didx, regionName in enumerate(region2segments):
            ax = axes[didx]

            im = ax.matshow(region2segments[regionName], cmap=plt.cm.get_cmap('viridis', len(valid_vals)), vmin=min_, vmax=max_)
            formatter = plt.FuncFormatter(formatter_func)

            # We must be sure to specify the ticks matching our target names
            ax.set_title(regionName, y=0.1)

        ax.cax.colorbar(im,ticks=positions, format=formatter, spacing='proportional')
        ax.cax.toggle_label(True)

        plt.show()
        plt.close()

    def __make_de_res_key(self, region0, clusters0, region1, clusters1):

        return (region0, tuple(sorted(clusters0)), region1, tuple(sorted(clusters1)))

    def find_markers(self, region0, clusters0, region1, clusters1, protWeights, use_methods = ["empire", "ttest", "rank"], count_scale={"ttest": 1, "rank": 1}):

        assert(region0 in self.regions)
        assert(region1 in self.regions)

        assert([x for x in self.regions[region0].idx2mass] == [x for x in self.regions[region1].idx2mass])

        if not isinstance(clusters0, (list, tuple, set)):
            clusters0 = [clusters0]

        if not isinstance(clusters1, (list, tuple, set)):
            clusters1 = [clusters1]

        assert(len(clusters0) > 0)
        assert(len(clusters1) > 0)

        cluster2coords_0 = self.regions[region0].getCoordsForSegmented()
        assert(all([x in cluster2coords_0 for x in clusters0]))

        cluster2coords_1 = self.regions[region1].getCoordsForSegmented()
        assert(all([x in cluster2coords_1 for x in clusters1]))

        self.logger.info("DE data for case: {} {}".format(region0, clusters0))
        self.logger.info("DE data for control: {} {}".format(region1, clusters1))
        print("Running {} {} against {} {}".format(region0, clusters0,region1, clusters1))

        de_results_all = defaultdict(lambda: dict())

        resKey = self.__make_de_res_key(region0, clusters0, region1, clusters1)
        
        sampleVec = []
        conditionVec = []

        exprData = pd.DataFrame()
        masses = [("mass_" + str(x)).replace(".", "_") for x in self.regions[region0].idx2mass]

        for clus in clusters0:

            allPixels = cluster2coords_0[clus]

            self.logger.info("Processing region {} cluster: {}".format(region0, clus))

            for pxl in allPixels:
                pxl_name = "{}__{}__{}".format(region0, str(len(sampleVec)), "_".join([str(x) for x in pxl]))
                sampleVec.append(pxl_name)
                conditionVec.append(0)
                exprData[pxl_name] = self.regions[region0].region_array[pxl[0], pxl[1], :]#.astype('int')


        for clus in clusters1:
            self.logger.info("Processing region {} cluster: {}".format(region1, clus))

            allPixels = cluster2coords_1[clus]
            for pxl in allPixels:
                pxl_name = "{}__{}__{}".format(region1, str(len(sampleVec)), "_".join([str(x) for x in pxl]))
                sampleVec.append(pxl_name)
                conditionVec.append(1)
                exprData[pxl_name] = self.regions[region1].region_array[pxl[0], pxl[1], :]#.astype('int')


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

                
                de_results_all[testMethod][resKey] = test.summary()
                self.logger.info("DE-test ({}) finished. Results available: {}".format(testMethod, resKey))

        deresDFs = defaultdict(lambda: dict())

        for test in de_results_all:
            for rkey in de_results_all[test]:

                deresDFs[test][rkey] = self.deres_to_df(de_results_all[test][rkey], rkey, protWeights)


        return deresDFs, exprData, pData

    def deres_to_df(self, deResDF, resKey, protWeights, keepOnlyProteins=True, inverse_fc=False, max_adj_pval=0.05, min_log2fc=0.5):

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


        targetSpectraMatrix = self.regions[resKey[0]].get_spectra_matrix(resKey[1])
        bgSpectraMatrix = self.regions[resKey[2]].get_spectra_matrix(resKey[3])

        self.logger.info("Created matrices with shape {} and {} (target, bg)".format(targetSpectraMatrix.shape, bgSpectraMatrix.shape))

        
        for row in fttr.iterrows():
            geneIDent = row[1][massCol]
            
            ag = geneIDent.split("_")
            massValue = float("{}.{}".format(ag[1], ag[2]))

            foundProt = protWeights.get_protein_from_mass(massValue, maxdist=3)

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
            
                    clusterVec.append("".join([str(x) for x in resKey[0]]))
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
                clusterVec.append("".join([str(x) for x in resKey[0]]))
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


        #requiredColumns = ["gene", "clusterID", "avg_logFC", "p_val_adj", "mean", "num", "anum"]
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
