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
import random
import ms_peak_picker


baseFolder = str(os.path.dirname(os.path.realpath(__file__)))

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform, pdist

import scipy.cluster as spc

import scipy as sp
import sklearn as sk

import umap
import hdbscan


class CombinedSpectra():
    """CombinedSpectra class for a combined analysis
    """

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

        self.region_array_scaled = {}
        self.de_results_all = defaultdict(lambda: dict())

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

    def check_scaled(self):
        hasToReprocess = False
        for regionName in self.regions:
            if not regionName in self.region_array_scaled:
                hasToReprocess = True
                break

        if hasToReprocess:
            self.logger.info("Calculating internormed regions")
            self.get_internormed_regions()


    def mass_intensity(self, masses, regions=None, scaled=False):

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
                plt.title("Intensities for Region {}".format(regionName))
                plt.show()
                plt.close()



    def mass_heatmap(self, masses, log=False, min_cut_off=None, plot=True, scaled=False):

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

            for didx, regionName in enumerate(region2segments):
                ax = axes[didx]

                heatmap = ax.matshow(region2segments[regionName], vmin=allMin, vmax=allMax)

                # We must be sure to specify the ticks matching our target names
                ax.set_title(regionName, color="w", y=0.1)

            fig.colorbar(heatmap, ax=axes[-1])

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


        rows = math.ceil(len(self.regions) / 2)
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

        for didx, regionName in enumerate(region2segments):
            ax = axes[didx]

            im = ax.matshow(region2segments[regionName], cmap=plt.cm.get_cmap('viridis', len(valid_vals)), vmin=min_, vmax=max_)
            formatter = plt.FuncFormatter(formatter_func)

            # We must be sure to specify the ticks matching our target names
            ax.set_title(regionName, color="w", y=0.1)

        plt.colorbar(im, ax=axes[-1], ticks=positions, format=formatter, spacing='proportional')

        plt.show()
        plt.close()

    def __make_de_res_key(self, region0, clusters0, region1, clusters1):

        return (region0, tuple(sorted(clusters0)), region1, tuple(sorted(clusters1)))

    def find_markers(self, region0, clusters0, region1, clusters1, protWeights, use_methods = ["empire", "ttest", "rank"], count_scale={"ttest": 1, "rank": 1}, scaled=True, sample_max=-1):

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

        if scaled:
            self.check_scaled()

        self.logger.info("DE data for case: {} {}".format(region0, clusters0))
        self.logger.info("DE data for control: {} {}".format(region1, clusters1))
        print("Running {} {} against {} {}".format(region0, clusters0,region1, clusters1))

        if self.de_results_all is None:
            self.de_results_all = defaultdict(lambda: dict())

        resKey = self.__make_de_res_key(region0, clusters0, region1, clusters1)
        
        sampleVec = []
        conditionVec = []

        exprData = pd.DataFrame()
        masses = [("mass_" + str(x)).replace(".", "_") for x in self.regions[region0].idx2mass]

        for clus in clusters0:

            allPixels = cluster2coords_0[clus]

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


        for clus in clusters1:
            self.logger.info("Processing region {} cluster: {}".format(region1, clus))

            allPixels = cluster2coords_1[clus]
            
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

                deresDFs[test][rkey] = self.deres_to_df(self.de_results_all[test][rkey], rkey, protWeights, keepOnlyProteins=protWeights != None, scaled=scaled)


        return deresDFs, exprData, pData


    def list_de_results(self):
        
        allDERes = []
        for x in self.de_results_all:
            for y in self.de_results_all[x]:
                allDERes.append((x,y))

        return allDERes

    def get_spectra_matrix(self, region_array, segments, cluster2coords):

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


    def deres_to_df(self, deResDF, resKey, protWeights, keepOnlyProteins=True, inverse_fc=False, max_adj_pval=0.05, min_log2fc=0.5, scaled=True):


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

    def _fivenumber(self, valuelist):
        """Creates five number statistics for values in valuelist

        Args:
            valuelist (list/tuple/np.array (1D)): list of values to use for statistics

        Returns:
            tuple: len, len>0, min, 25-quantile, 50-quantile, 75-quantile, max
        """

        min_ = np.min(valuelist)
        max_ = np.max(valuelist)

        (quan25_, quan50_, quan75_) = np.quantile(valuelist, [0.25, 0.5, 0.75])

        return (len(valuelist), len([x for x in valuelist if x > 0]), min_, quan25_, quan50_, quan75_, max_)


    def get_internormed_regions(self, method="median"):

        assert (method in ["avg", "median"])

        allRegionNames = [x for x in self.regions]

        # get reference background spec
        referenceMedianSpectra = self.regions[allRegionNames[0]].consensus_spectra(method="median", set_consensus=False)

        self.region_array_scaled[allRegionNames[0]] = np.copy(self.regions[allRegionNames[0]].region_array)


        fcDict = {}
        bar = progressbar.ProgressBar()
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
            if method == "median":
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