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
baseFolder = str(os.path.dirname(os.path.realpath(__file__)))

lib = ctypes.cdll.LoadLibrary(baseFolder+'/../../cppSRM/lib/libSRM.so')

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform

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


    def plot_common_segments(self):

        assert(not self.region_cluster2cluster is None)

        allClusters = [self.region_cluster2cluster[x] for x in self.region_cluster2cluster]
        valid_vals = sorted(set(allClusters))

        min_ = min(valid_vals)
        max_ = max(valid_vals)

        positions = np.linspace(min_, max_, len(valid_vals))
        val_lookup = dict(zip(positions, valid_vals))
        print(val_lookup)

        def formatter_func(x, pos):
            'The two args are the value and tick position'
            val = val_lookup[x]
            return val

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
            
                    clusterVec.append(",".join([str(x) for x in resKey[0]]))
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
                clusterVec.append(",".join([str(x) for x in resKey[0]]))
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



class SpectraRegion():

    @classmethod
    def from_pickle(cls, path):

        obj = None
        with open(path, "rb") as fin:
            obj = pickle.load(fin)

        return obj

    def to_pickle(self, path):

        with open(path, "wb") as fout:
            pickle.dump(self, fout)

    def __init__(self, region_array, idx2mass, name=None):

        assert(not region_array is None)
        assert(not idx2mass is None)

        assert(len(region_array[0,0,:]) == len(idx2mass))


        lib.StatisticalRegionMerging_New.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint8]
        lib.StatisticalRegionMerging_New.restype = ctypes.c_void_p

        lib.SRM_calc_similarity.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        lib.SRM_calc_similarity.restype = ctypes.POINTER(ctypes.c_float)

        lib.StatisticalRegionMerging_mode_dot.argtypes = [ctypes.c_void_p]
        lib.StatisticalRegionMerging_mode_dot.restype = None

        lib.StatisticalRegionMerging_mode_eucl.argtypes = [ctypes.c_void_p]
        lib.StatisticalRegionMerging_mode_eucl.restype = None

        self.logger = None
        self.__setlogger()

        self.name = None
        self.region_array = region_array
        self.idx2mass = idx2mass

        self.spectra_similarity = None
        self.dist_pixel = None

        self.idx2pixel = {}
        self.pixel2idx = {}

        self.elem_matrix = None
        self.dimred_elem_matrix = None
        self.dimred_labels = None

        self.segmented = None
        self.segmented_method = None

        self.cluster_filters = []

        self.consensus = None
        self.consensus_method = None
        self.consensus_similarity_matrix = None

        self.de_results_all = defaultdict(lambda: dict())

        for i in range(self.region_array.shape[0]*self.region_array.shape[1]):

            x,y = divmod(i, self.region_array.shape[1])

            self.idx2pixel[i] = (x,y)
            self.pixel2idx[(x,y)] = i

    def __setlogger(self):
        self.logger = logging.getLogger('SpectraRegion')

        if len(self.logger.handlers) == 0:
            self.logger.setLevel(logging.INFO)

            consoleHandler = logging.StreamHandler()
            consoleHandler.setLevel(logging.INFO)

            self.logger.addHandler(consoleHandler)

            formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
            consoleHandler.setFormatter(formatter)


    def __getstate__(self):
        return {
            "name": self.name,
            "region_array": self.region_array,
            "idx2mass": self.idx2mass,
            "spectra_similarity": self.spectra_similarity,
            "dist_pixel": self.dist_pixel,
            "idx2pixel": self.pixel2idx,
            "elem_matrix": self.elem_matrix,
            "dimred_elem_matrix": self.dimred_elem_matrix,
            "dimred_labels": self.dimred_labels,
            "segmented": self.segmented,
            "segmented_method": self.segmented_method,
            "cluster_filters": self.cluster_filters,
            "consensus": self.consensus,
            "consensus_method": self.consensus_method,
            "consensus_similarity_matrix": self.consensus_similarity_matrix,
            "de_results_all": self.de_results_all
        }

    def __setstate__(self, state):

        self.__dict__.update(state)

        """
        self.name = state.name
        self.region_array = state.region_array
        self.idx2mass = state.idx2mass
        self.spectra_similarity = state.spectra_similarity
        self.dist_pixel = state.dist_pixel
        self.pixel2idx = state.pixel2idx
        self.elem_matrix = state.elem_matrix
        self.dimred_elem_matrix = state.dimred_elem_matrix
        self.dimred_labels = state.dimred_labels
        self.segmented = state.segmented
        self.segmented_method = state.segmented_method
        self.cluster_filters = state.cluster_filters
        self.consensus = state.consensus
        self.consensus_method = state.consensus_method
        self.consensus_similarity_matrix = state.consensus_similarity_matrix
        self.de_results_all = state.de_results_all
        """

        self.logger = None
        self.__setlogger()

        self.idx2pixel = {}
        self.pixel2idx = {}

        for i in range(self.region_array.shape[0]*self.region_array.shape[1]):

            x,y = divmod(i, self.region_array.shape[1])

            self.idx2pixel[i] = (x,y)
            self.pixel2idx[(x,y)] = i


    def plot_array(self, fig, arr):


        valid_vals = np.unique(arr)
        heatmap = plt.matshow(arr, cmap=plt.cm.get_cmap('viridis', len(valid_vals)), fignum=fig.number)

        # calculate the POSITION of the tick labels
        min_ = min(valid_vals)
        max_ = max(valid_vals)

        positions = np.linspace(min_, max_, len(valid_vals))
        val_lookup = dict(zip(positions, valid_vals))

        def formatter_func(x, pos):
            'The two args are the value and tick position'
            val = val_lookup[x]
            return val

        formatter = plt.FuncFormatter(formatter_func)

        # We must be sure to specify the ticks matching our target names
        plt.colorbar(heatmap, ticks=positions, format=formatter, spacing='proportional')
        
        return fig



    def to_aorta3d(self, folder, prefix, regionID, protWeights = None, nodf=False, pathPrefix = None):

        cluster2coords = self.getCoordsForSegmented()

        os.makedirs(folder, exist_ok=True)
        segmentsPath = prefix + "." + str(regionID) + ".upgma.png"

        # plot image
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=self.segmented.min(), vmax=self.segmented.max())
        image = cmap(norm(self.segmented))
        plt.imsave(os.path.join(folder, segmentsPath), image)

        if pathPrefix != None:
            segmentsPath = os.path.join(pathPrefix, segmentsPath)

        cluster2deData = {}
        # write DE data
        if protWeights != None:

            if not nodf:
                markerGenes = self.find_all_markers(protWeights, use_methods=["ttest"], includeBackground=True)
            
            for cluster in cluster2coords:

                outputname = prefix + "." + str(regionID) + "." + str(cluster) +".tsv"
    
                if not nodf:
                    outfile = os.path.join(folder, outputname)
                    subdf = markerGenes["ttest"][markerGenes["ttest"]["clusterID"] == str(cluster)]
                    subdf.to_csv(outfile, sep="\t", index=True)


                if pathPrefix != None:
                    outputname = os.path.join(pathPrefix, outputname)

                cluster2deData[cluster] = outputname

        # write info
        regionInfos = {}
        
        
        for cluster in cluster2coords:

            clusterType = "aorta" if cluster != 0 else "background"

            regionInfo = {
                "type_det": [clusterType],
                "coordinates": [[x[1], x[0]] for x in cluster2coords[cluster]],
            }

            if cluster in cluster2deData:
                regionInfo["de_data"] = cluster2deData[cluster]

            regionInfos[str(cluster)] = regionInfo



        infoDict = {}
        infoDict = {
            "region": regionID,
            "path_upgma": segmentsPath,
            "info": regionInfos
        }


        jsElems = json.dumps([infoDict])

        # write config_file

        with open(os.path.join(folder, prefix + "." + str(regionID) + ".info"), 'w') as fout:
            print(jsElems, file=fout)
        


    def __get_exmass_for_mass(self, mass, threshold=None):
        
        dist2mass = float('inf')
        curMass = -1
        curIdx = -1

        for xidx,x in enumerate(self.idx2mass):
            dist = abs(x-mass)
            if dist < dist2mass and (threshold==None or dist < threshold):
                dist2mass = dist
                curMass = x    
                curIdx = xidx    

        return curMass, curIdx


    def mass_heatmap(self, masses, log=False, min_cut_off=None):

        if not isinstance(masses, (list, tuple, set)):
            masses = [masses]

        image = np.zeros((self.region_array.shape[0], self.region_array.shape[1]))

        for mass in masses:
            
            bestExMassForMass, bestExMassIdx = self.__get_exmass_for_mass(mass)
            self.logger.info("Processing Mass {} with best existing mass {}".format(mass, bestExMassForMass))

            for i in range(self.region_array.shape[0]):
                for j in range(self.region_array.shape[1]):

                    image[i,j] += self.region_array[i,j,bestExMassIdx]


        if log:
            image = np.log(image)

        if min_cut_off != None:
            image[image <= min_cut_off] = min_cut_off

        heatmap = plt.matshow(image)
        plt.colorbar(heatmap)
        plt.show()
        plt.close()
        #return image


    def __calc_similarity(self, inputarray):
        # load image
        dims = 1

        inputarray = inputarray.astype(np.float32)

        if len(inputarray.shape) > 2:
            dims = inputarray.shape[2]
        print(dims)
        qs = []
        qArr = (ctypes.c_float * len(qs))(*qs)

        self.logger.info("Creating C++ obj")

        # self.obj = lib.StatisticalRegionMerging_New(dims, qArr, 3)
        # print(inputarray.shape)
        # testArray = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]],[[25,26,27],[28,29,30]],[[31,32,33],[34,35,36]]], dtype=np.float32)
        # print(testArray.shape)
        # image_p = testArray.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # retValues = lib.SRM_test_matrix(self.obj, testArray.shape[0], testArray.shape[1], image_p)
        # exit()

        self.logger.info("dimensions {}".format(dims))
        self.logger.info("input dimensions {}".format(inputarray.shape))

        self.obj = lib.StatisticalRegionMerging_New(dims, qArr, len(qs))
        self.logger.info("Switching to dot mode")
        lib.StatisticalRegionMerging_mode_dot(self.obj)

        image_p = inputarray.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.logger.info("Starting calc similarity c++")
        retValues = lib.SRM_calc_similarity(self.obj, inputarray.shape[0], inputarray.shape[1], image_p)

        outclust = np.ctypeslib.as_array(retValues, shape=(inputarray.shape[0] * inputarray.shape[1], inputarray.shape[0] * inputarray.shape[1]))

        self.logger.info("outclust dimensions {}".format(outclust.shape))

        return outclust

    def calculate_similarity(self, mode="spectra", features=[], neighbors = 1):
        """

        :param mode: must be in  ["spectra", "spectra_log", "spectra_log_dist"]

        :return: spectra similarity matrix
        """

        assert(mode in ["spectra", "spectra_log", "spectra_log_dist"])

        if len(features) > 0:
            for neighbor in range(neighbors):
                features = features + [i + neighbor for i in features] + [i - neighbor for i in features]
            features = np.unique(features)
            featureIndex = [self.__get_exmass_for_mass(x) for x in features]
            featureIndex = [y for (x,y) in featureIndex if y != None]
            featureIndex = sorted(np.unique(featureIndex))
            regArray = np.zeros((self.region_array.shape[0], self.region_array.shape[1], len(featureIndex)))
            for i in range(self.region_array.shape[0]):
                for j in range(self.region_array.shape[1]):
                    extracted = [self.region_array[i,j,:][k] for k in tuple(featureIndex)]
                    regArray[i,j,:] = extracted
        else:
            regArray = np.array(self.region_array, copy=True)  

        print(regArray.shape)
        self.spectra_similarity = self.__calc_similarity(regArray)
        print(self.spectra_similarity.shape)

        if mode in ["spectra_log", "spectra_log_dist"]:
            self.logger.info("Calculating spectra similarity")
            self.spectra_similarity = np.log(self.spectra_similarity + 1)
            self.spectra_similarity = self.spectra_similarity / np.max(self.spectra_similarity)

            self.logger.info("Calculating spectra similarity done")

        if mode in ["spectra_log_dist"]:

            if self.dist_pixel == None or self.dist_pixel.shape != self.spectra_similarity.shape:
                self.dist_pixel = np.zeros((self.spectra_similarity.shape[0], self.spectra_similarity.shape[1]))

                self.logger.info("Calculating dist pixel map")

                for x in range(0, self.spectra_similarity.shape[0]):
                    coordIx, coordIy = self.idx2pixel[x]# divmod(x, self.region_array.shape[1])

                    for y in range(0, self.spectra_similarity.shape[1]):
                        coordJx, coordJy = self.idx2pixel[y] # divmod(x, self.region_array.shape[1])
                        self.dist_pixel[x,y] = np.linalg.norm((coordIx-coordJx, coordIy-coordJy))

                self.dist_pixel  = self.dist_pixel / np.max(self.dist_pixel)
                self.logger.info("Calculating dist pixel map done")


            self.spectra_similarity = 0.95 * self.spectra_similarity + 0.05 * self.dist_pixel

        return self.spectra_similarity


    def __segment__upgma(self, number_of_regions):

        ssim = 1-self.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.linkage(squareform(ssim), method='average', metric='cosine')
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')

        return c

    def __segment__wpgma(self, number_of_regions):

        ssim = 1-self.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.weighted(squareform(ssim))
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')

        return c

    def __segment__ward(self, number_of_regions):

        ssim = 1-self.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.ward(squareform(ssim))
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')
        return c

    def __segment__umap_hdbscan(self, number_of_regions):

        self.dimred_elem_matrix = np.zeros((self.region_array.shape[0]*self.region_array.shape[1], self.region_array.shape[2]))
        self.elem_matrix = np.zeros((self.region_array.shape[0]*self.region_array.shape[1], self.region_array.shape[2]))

        """
        
        ----------> spectra ids
        |
        |
        | m/z values
        |
        v
        
        """

        idx2ij = {}

        for i in range(0, self.region_array.shape[0]):
            for j in range(0, self.region_array.shape[1]):
                idx = i * self.region_array.shape[1] + j
                self.elem_matrix[idx, :] = self.region_array[i,j,:]

                idx2ij[idx] = (i,j)


        self.logger.info("UMAP reduction")
        self.dimred_elem_matrix = umap.UMAP(
            n_neighbors=10,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        ).fit_transform(self.elem_matrix)

        self.logger.info("HDBSCAN reduction")

        clusterer = hdbscan.HDBSCAN(
            min_samples=5,
            min_cluster_size=20,
        )
        self.dimred_labels = clusterer.fit_predict(self.dimred_elem_matrix) + 2

        #c = spc.hierarchy.fcluster(clusterer.single_linkage_tree_.to_numpy(), t=10, criterion='maxclust')

        return self.dimred_labels

    def vis_umap(self, legend=True):

        assert(not self.elem_matrix is None)
        assert(not self.dimred_elem_matrix is None)
        assert(not self.dimred_labels is None)


        plt.figure()
        clustered = (self.dimred_labels >= 0)
        print("Dimred Shape", self.dimred_elem_matrix.shape)
        print("Unassigned", self.dimred_elem_matrix[~clustered, ].shape)
        plt.scatter(self.dimred_elem_matrix[~clustered, 0],
                    self.dimred_elem_matrix[~clustered, 1],
                    color=(1, 0,0),
                    label="Unassigned",
                    s=2.0)

        uniqueClusters = sorted(set([x for x in self.dimred_labels if x >= 0]))

        for cidx, clusterID in enumerate(uniqueClusters):
            cmap = matplotlib.cm.get_cmap('Spectral')

            clusterColor = cmap(cidx / len(uniqueClusters))

            plt.scatter(self.dimred_elem_matrix[self.dimred_labels == clusterID, 0],
                        self.dimred_elem_matrix[self.dimred_labels == clusterID, 1],
                        color=clusterColor,
                        label=str(clusterID),
                        s=10.0)

        if legend:
            plt.legend()
        plt.show()
        plt.close()


    def plot_segments(self, highlight=None):
        assert(not self.segmented is None)

        showcopy = np.copy(self.segmented)

        if highlight != None:
            if not isinstance(highlight, (list, tuple, set)):
                highlight = [highlight]

            for i in range(0, showcopy.shape[0]):
                for j in range(0, showcopy.shape[1]):

                    if showcopy[i,j] != 0:

                        if showcopy[i,j] in highlight:
                            showcopy[i,j] = 2
                        elif showcopy[i,j] != 0:
                            showcopy[i,j] = 1


        fig = plt.figure()
        self.plot_array(fig, showcopy)
        plt.show()
        plt.close()


    def segment(self, method="UPGMA", number_of_regions=10):

        assert(not self.spectra_similarity is None)
        assert(method in ["UPGMA", "WPGMA", "WARD", "KMEANS", "UMAP_DBSCAN"])

        self.logger.info("Calculating clusters")

        c = None

        if method == "UPGMA":
            c = self.__segment__upgma(number_of_regions)

        elif method == "WPGMA":
            c = self.__segment__wpgma(number_of_regions)

        elif method == "WARD":
            c = self.__segment__ward(number_of_regions)
        elif method == "UMAP_DBSCAN":
            c = self.__segment__umap_hdbscan(number_of_regions)

        self.logger.info("Calculating clusters done")

        image_UPGMA = np.zeros(self.region_array.shape, dtype=np.int16)
        image_UPGMA = image_UPGMA[:,:,0]


        # cluster 0 has special meaning: not assigned !
        assert(not 0 in [c[x] for x in c])

        for i in range(0, image_UPGMA.shape[0]):
            for j in range(0, image_UPGMA.shape[1]):
                image_UPGMA[i,j] = c[self.pixel2idx[(i,j)]]


        self.segmented = image_UPGMA
        self.segmented_method = method
        self.cluster_filters = []

        self.logger.info("Calculating clusters saved")

        return self.segmented


    def filter_clusters(self, method='remove_singleton'):

        assert(method in ["remove_singleton", "most_similar_singleton", "merge_background", "remove_islands"])

        cluster2coords = self.getCoordsForSegmented()

        if method == "remove_islands":

            exarray = self.segmented.copy()
            exarray[exarray >= 1] = 1

            labeledArr, num_ids = ndimage.label(exarray, structure=np.ones((3,3)))

            for i in range(0, num_ids+1):

                labelCells = np.count_nonzero(labeledArr == i)

                if labelCells <= 10:
                    self.segmented[labeledArr == i] = 0



        elif method == "remove_singleton":
            for clusterID in cluster2coords:

                if clusterID == 0:
                    continue # unassigned cluster - ignore it

                clusCoords = cluster2coords[clusterID]

                if len(clusCoords) == 1:
                    self.segmented[self.segmented == clusterID] = 0



        elif method == "most_similar_singleton":
            assert(self.consensus != None)

            for clusterID in cluster2coords:

                if clusterID == 0:
                    continue # unassigned cluster - ignore it

                clusCoords = cluster2coords[clusterID]

                if len(clusCoords) == 1:

                    cons2sim = {}
                    for cid in self.consensus:

                        sim = self.__calc_direct_similarity(self.region_array[clusCoords[0]], self.consensus[cid])
                        cons2sim[cid] = sim


                    mostSimClus = sorted([(x, cons2sim[x]) for x in cons2sim], key=lambda x: x[1], reverse=True)[0][0]
                    self.segmented[self.segmented == clusterID] = mostSimClus
        elif method == "merge_background":
            
            # which clusters are in 3x3 border boxes and not in 10x10 middle box?
            borderSegments = set()

            xdim = 4
            ydim = 4

            for i in range(0, min(xdim, self.segmented.shape[0])):
                for j in range(0, min(ydim, self.segmented.shape[1])):
                    clusterID = self.segmented[i, j]
                    borderSegments.add(clusterID)

            for i in range(max(0, self.segmented.shape[0]-xdim), self.segmented.shape[0]):
                for j in range(max(0, self.segmented.shape[1]-ydim), self.segmented.shape[1]):
                    clusterID = self.segmented[i, j]
                    borderSegments.add(clusterID)

            for i in range(max(0, self.segmented.shape[0]-xdim), self.segmented.shape[0]):
                for j in range(0, min(ydim, self.segmented.shape[1])):
                    clusterID = self.segmented[i, j]
                    borderSegments.add(clusterID)

            for i in range(0, min(xdim, self.segmented.shape[0])):
                for j in range(max(0, self.segmented.shape[1]-ydim), self.segmented.shape[1]):
                    clusterID = self.segmented[i, j]
                    borderSegments.add(clusterID)

            self.logger.info("Assigning clusters to background: {}".format(borderSegments))

            for x in borderSegments:
                self.segmented[self.segmented == x] = 0
                    
        self.cluster_filters.append(method)

        return self.segmented

    def __cons_spectra__avg(self, cluster2coords):

        cons_spectra = {}
        for clusID in cluster2coords:

            spectraCoords = cluster2coords[clusID]

            if len(spectraCoords) == 1:
                coord = spectraCoords[0]
                # get spectrum, return spectrum
                avgSpectrum = self.region_array[coord[0], coord[1]]
            else:

                avgSpectrum = np.zeros((1, self.region_array.shape[2]))

                for coord in spectraCoords:
                    avgSpectrum += self.region_array[coord[0], coord[1]]

                avgSpectrum = avgSpectrum / len(spectraCoords)

            cons_spectra[clusID] = avgSpectrum[0]

        return cons_spectra

    def getCoordsForSegmented(self):
        cluster2coords = defaultdict(list)

        for i in range(0, self.segmented.shape[0]):
            for j in range(0, self.segmented.shape[1]):

                clusterID = int(self.segmented[i, j])

                #if clusterID == 0:
                #    continue # unassigned cluster

                cluster2coords[clusterID].append((i,j))

        return cluster2coords


    def consensus_spectra(self, method="avg"):

        assert(not self.segmented is None)
        assert(method in ["avg"])

        self.logger.info("Calculating consensus spectra")

        cluster2coords = self.getCoordsForSegmented()


        cons_spectra = None
        if method == "avg":
            cons_spectra = self.__cons_spectra__avg(cluster2coords)


        self.consensus = cons_spectra
        self.consensus_method = method
        self.logger.info("Calculating consensus spectra done")

        return self.consensus

    def mass_dabest(self, masses, background=0):

        assert(not self.segmented is None)

        if not isinstance(masses, (list, tuple, set)):
            masses = [masses]


        cluster2coords = self.getCoordsForSegmented()
        assert(background in cluster2coords)


        image = np.zeros((self.region_array.shape[0], self.region_array.shape[1]))

        for mass in masses:
            
            bestExMassForMass, bestExMassIdx = self.__get_exmass_for_mass(mass)
            self.logger.info("Processing Mass {} with best existing mass {}".format(mass, bestExMassForMass))

            
            clusterIntensities = defaultdict(list)

            for clusterid in cluster2coords:
                for coord in cluster2coords[clusterid]:
                    intValue = self.region_array[coord[0], coord[1], bestExMassIdx]
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
            plt.show()
            plt.close()

            dfobj_db = dfObj.pivot(index="specidx", columns='cluster', values='intensity')

            allClusterIDs = natsorted([x for x in set(clusterVec) if not " {}".format(background) in x])
            
            multi_groups = dabest.load(dfobj_db, idx=tuple(["Cluster {}".format(background)]+allClusterIDs))
            multi_groups.mean_diff.plot()

    def plot_inter_consensus_similarity(self, clusters=None):


        cluster2coords = self.getCoordsForSegmented()
        clusterLabels = sorted([x for x in cluster2coords])
        self.logger.info("Found clusterLabels {}".format(clusterLabels))

        if clusters == None:
            clusters = sorted([x for x in cluster2coords])
        
        for cluster in clusters:
            
            self.logger.info("Processing cluster {}".format(cluster))

            ownSpectra = [ self.region_array[xy[0], xy[1], :] for xy in cluster2coords[cluster] ]
            clusterSimilarities = {}

            for clusterLabel in clusterLabels:

                allSpectra = [ self.region_array[xy[0], xy[1], :] for xy in cluster2coords[clusterLabel] ]

                clusterSims = []
                for i in range(0, len(ownSpectra)):
                    for j in range(0, len(allSpectra)):
                        clusterSims.append( self.__get_spectra_similarity(ownSpectra[i], allSpectra[j]) )

                clusterSimilarities[clusterLabel] = clusterSims

            clusterVec = []
            similarityVec = []
            for x in clusterSimilarities:
                
                elems = clusterSimilarities[x]
                clusterVec += [x] * len(elems)
                similarityVec += elems
                    
            dfObj = pd.DataFrame({"cluster": clusterVec, "similarity": similarityVec})
            sns.boxplot(data=dfObj, x="cluster", y="similarity")
            plt.show()
            plt.close()



    def plot_consensus_similarity(self, mode="heatmap"):

        assert(not self.consensus_similarity_matrix is None)

        assert(mode in ["heatmap", "spectra"])

        if mode == "heatmap":
            heatmap = plt.matshow(self.consensus_similarity_matrix)
            plt.colorbar(heatmap)
            plt.show()
            plt.close()

        elif mode == "spectra":
            
            cluster2coords = self.getCoordsForSegmented()
            clusterLabels = sorted([x for x in cluster2coords])

            self.logger.info("Found clusterLabels {}".format(clusterLabels))

            clusterSimilarities = {}

            for clusterLabel in clusterLabels:

                allSpectra = [ self.region_array[xy[0], xy[1], :] for xy in cluster2coords[clusterLabel] ]

                self.logger.info("Processing clusterLabel {}".format(clusterLabel))

                clusterSims = []
                for i in range(0, len(allSpectra)):
                    for j in range(i+1, len(allSpectra)):
                        clusterSims.append( self.__get_spectra_similarity(allSpectra[i], allSpectra[j]) )

                clusterSimilarities[clusterLabel] = clusterSims

            clusterVec = []
            similarityVec = []
            for x in clusterSimilarities:
                
                elems = clusterSimilarities[x]
                clusterVec += [x] * len(elems)
                similarityVec += elems
                    
            dfObj = pd.DataFrame({"cluster": clusterVec, "similarity": similarityVec})
            sns.boxplot(data=dfObj, x="cluster", y="similarity")
            plt.show()
            plt.close()                                

    def __get_spectra_similarity(self, vA, vB):
        return np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))


    def consensus_similarity(self ):
        """
            calculates the similarity for consensus spectra
        """

        assert(not self.consensus is None)

        allLabels = sorted([x for x in self.consensus])
        specLength = len(self.consensus[allLabels[0]])
        
        # bring consensus into correct form
        consMatrix = np.zeros((len(allLabels), specLength))

        for lidx, label in enumerate(allLabels):
            consMatrix[lidx, :] = self.consensus[label]



        self.consensus_similarity_matrix = np.zeros((len(allLabels), len(allLabels)))

        for i in range(len(allLabels)):
            vA = self.consensus[allLabels[i]]
            for j in range(i, len(allLabels)):

                vB = self.consensus[allLabels[j]]

                simValue = self.__get_spectra_similarity(vA, vB)

                self.consensus_similarity_matrix[i, j] = simValue
                self.consensus_similarity_matrix[j, i] = simValue


    def __get_expression(self, massValue, segments, mode="avg"):

        assert(massValue != None)
        assert(segments != None)

        if not isinstance(mode, (list, tuple, set)):
            mode = [mode]

        if not isinstance(segments, (list, tuple, set)):
            segments = [segments]

        assert(all([x in ["avg", "median"] for x in mode]))

        cluster2coords = self.getCoordsForSegmented()

        assert(all([y in cluster2coords for y in segments]))

        # best matchng massvalue - rounding difference, etc
        massValue, massIndex = self.__get_exmass_for_mass(massValue)

        allExprValues = []
        for segment in segments:
            segmentPixels = cluster2coords[segment]

            for pixel in segmentPixels:
                exprValue = self.region_array[pixel[0], pixel[1], massIndex]
                allExprValues.append(exprValue)

        num, anum = len(allExprValues), len([x for x in allExprValues if x > 0])

        resElem = []

        for modElem in mode:

            if modElem == "avg":
                resElem.append( np.mean(allExprValues) )

            elif modElem == "median":
                resElem.append( np.median(allExprValues) )

        return tuple(resElem), num, anum

    def get_spectra_matrix(self,segments):

        cluster2coords = self.getCoordsForSegmented()

        relPixels = []
        for x in segments:
            relPixels += cluster2coords.get(x, [])

        spectraMatrix = np.zeros((len(relPixels), len(self.idx2mass)))

        for pidx, px in enumerate(relPixels):
            spectraMatrix[pidx, :] = self.region_array[px[0], px[1], :]

        return spectraMatrix


    def get_expression_from_matrix(self, matrix, massValue, segments, mode="avg"):

        assert(massValue != None)
        assert(segments != None)

        if not isinstance(mode, (list, tuple, set)):
            mode = [mode]

        if not isinstance(segments, (list, tuple, set)):
            segments = [segments]

        assert(all([x in ["avg", "median"] for x in mode]))

        # best matchng massvalue - rounding difference, etc
        massValue, massIndex = self.__get_exmass_for_mass(massValue)

        allExprValues = list(matrix[:, massIndex])

        num, anum = len(allExprValues), len([x for x in allExprValues if x > 0])
        resElem = []

        for modElem in mode:

            if modElem == "avg":
                resElem.append( np.mean(allExprValues) )

            elif modElem == "median":
                resElem.append( np.median(allExprValues) )

        return tuple(resElem), num, anum

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


        targetSpectraMatrix = self.get_spectra_matrix(resKey[0])
        bgSpectraMatrix = self.get_spectra_matrix(resKey[1])

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

            expT, totalSpectra, measuredSpecta = self.get_expression_from_matrix(targetSpectraMatrix, massValue, resKey[0], ["avg", "median"])
            exprBG, totalSpectraBG, measuredSpectaBG = self.get_expression_from_matrix(bgSpectraMatrix, massValue, resKey[0], ["avg", "median"])

            avgExpr, medianExpr = expT
            avgExprBG, medianExprBG = exprBG

            if len(foundProt) > 0:

                for protMassTuple in foundProt:
                    
                    prot,protMass = protMassTuple
            
                    clusterVec.append(",".join([str(x) for x in resKey[0]]))
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
                clusterVec.append(",".join([str(x) for x in resKey[0]]))
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


    def find_all_markers(self, protWeights, keepOnlyProteins=True, replaceExisting=False, includeBackground=True,out_prefix="nldiffreg", outdirectory=None, use_methods = ["empire", "ttest", "rank"], count_scale={"ttest": 1, "rank": 1, "empire": 10000}):
        cluster2coords = self.getCoordsForSegmented()

        dfbyMethod = defaultdict(lambda: pd.DataFrame())

        for segment in cluster2coords:

            if not includeBackground and segment == 0:
                continue

            clusters0 = [segment]
            clusters1 = [x for x in cluster2coords if not x in clusters0]

            if not includeBackground and 0 in clusters1:
                del clusters1[clusters1.index(0)]

            self.find_markers(clusters0=clusters0, clusters1=clusters1, replaceExisting=replaceExisting, outdirectory=outdirectory, out_prefix=out_prefix, use_methods=use_methods, count_scale=count_scale)

            # get result
            resKey = self.__make_de_res_key(clusters0, clusters1)

            keyResults = self.get_de_results(resKey)

            for method in keyResults:
                methodKeyDF = self.get_de_result(method, resKey)

                inverseFC = False
                if method in ["ttest", "rank"]:
                    inverseFC = True

                resDF = self.deres_to_df(methodKeyDF, resKey, protWeights, keepOnlyProteins=keepOnlyProteins, inverse_fc=inverseFC)

                dfbyMethod[method] = pd.concat([dfbyMethod[method], resDF], sort=False)           

        return dfbyMethod

                    

    def __make_de_res_key(self, clusters0, clusters1):

        return (tuple(sorted(clusters0)), tuple(sorted(clusters1)))
        

    def clear_de_results(self):
        self.de_results_all = defaultdict(lambda: dict())

    def run_nlempire(self, nlDir, pdata, pdataPath, diffOutput):
        import regex as re
        def run(cmd):
            print(" ".join(cmd))
            proc = subprocess.Popen(cmd,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.PIPE,
            )
            stdout, stderr = proc.communicate()

            print("Cmd returned with exit code", proc.returncode)
        
            return proc.returncode, stdout, stderr
        
        pysrmPath = os.path.dirname(os.path.abspath(__file__))
        tocounts = Counter([x for x in pdata["condition"]])
        mc = tocounts.most_common(1)[0]

        print("Max condition count", mc)

        if mc[1] > 100:
            code, out, err = run(["/usr/bin/java", "-Xmx24G", "-cp", pysrmPath+"/../../../tools/nlEmpiRe.jar", "nlEmpiRe.input.ExpressionSet", "-getbestsubN", "100", "-inputdir", nlDir, "-cond1", "0", "-cond2", "1", "-o", nlDir + "/exprs_ds.txt"])

            empireData=pd.read_csv(nlDir + "/exprs_ds.txt", delimiter="\t")
            del empireData["gene"]

            allreps = [x for x in list(empireData.columns.values) if not x in ["gene"]]

            cond1reps = [x for x in allreps if x.startswith("cond1")]
            cond2reps = [x for x in allreps if x.startswith("cond2")]

            newPData = pd.DataFrame()

            newPData["sample"] = cond1reps+cond2reps
            newPData["condition"] = [0] * len(cond1reps) + [1] * len(cond2reps)
            newPData["batch"] = 0

            print("Writing new Pdata")
            newPData.to_csv(nlDir + "/p_data.txt", sep="\t", index=False)

            print("Writing new exprs data")
            empireData.to_csv(nlDir + "/exprs.txt", sep="\t", index=False, header=False)

        runI = 0
        while runI < 10:
            code, out, err = run(["/usr/bin/java", "-Xmx16G", "-cp", pysrmPath+"/../../../tools/nlEmpiRe.jar", "nlEmpiRe.input.ExpressionSet", "-inputdir", nlDir, "-cond1", "0", "-cond2", "1", "-o", diffOutput])

            print(code)

            if code == 0:
                break

            errStr = err.decode()
            print(errStr)

            res = re.findall(r"replicates: \[(.*?)\]", errStr)

            if len(res) == 1:
                break

            res = res[1:]

            removeSpectra = []
            for x in res:
                ress = x.split(", ")
                for y in ress:
                    removeSpectra.append(y)


            print("Loading pdata", )
            pdata = pd.read_csv(pdataPath, delimiter="\t")

            print("Removing spectra", removeSpectra)
            pdata=pdata[~pdata['sample'].isin(removeSpectra)]
            print(pdata)
            pdata.to_csv(pdataPath, index=False, sep="\t")

            runI += 1

        return diffOutput



    def find_markers(self, clusters0, clusters1=None, out_prefix="nldiffreg", outdirectory=None, replaceExisting=False, use_methods = ["empire", "ttest", "rank"], count_scale={"ttest": 1, "rank": 1, "empire": 10000}):

        cluster2coords = self.getCoordsForSegmented()

        if not isinstance(clusters0, (list, tuple, set)):
            clusters0 = [clusters0]

        if clusters1 == None:
            clusters1 = [x for x in cluster2coords if not x in clusters0]

        assert(len(clusters1) > 0)

        assert(all([x in cluster2coords for x in clusters0]))
        assert(all([x in cluster2coords for x in clusters1]))

        self.logger.info("DE data for case: {}".format(clusters0))
        self.logger.info("DE data for control: {}".format(clusters1))
        print("Running {} against {}".format(clusters0, clusters1))

        resKey = self.__make_de_res_key(clusters0, clusters1)
        self.logger.info("DE result key: {}".format(resKey))

        if not replaceExisting:

            if all([resKey in self.de_results_all[x] for x in use_methods]):
                self.logger.info("DE result key already exists")
                return

        sampleVec = []
        conditionVec = []

        exprData = pd.DataFrame()
        masses = [("mass_" + str(x)).replace(".", "_") for x in self.idx2mass]

        for clus in clusters0:

            allPixels = cluster2coords[clus]

            self.logger.info("Processing cluster: {}".format(clus))

            for pxl in allPixels:

                pxl_name = "{}__{}".format(str(len(sampleVec)), "_".join([str(x) for x in pxl]))

                sampleVec.append(pxl_name)
                conditionVec.append(0)

                exprData[pxl_name] = self.region_array[pxl[0], pxl[1], :]#.astype('int')


        for clus in clusters1:
            self.logger.info("Processing cluster: {}".format(clus))

            allPixels = cluster2coords[clus]
            for pxl in allPixels:

                pxl_name = "{}__{}".format(str(len(sampleVec)), "_".join([str(x) for x in pxl]))

                sampleVec.append(pxl_name)
                conditionVec.append(1)

                exprData[pxl_name] = self.region_array[pxl[0], pxl[1], :]#.astype('int')


        self.logger.info("DE DataFrame ready. Shape {}".format(exprData.shape))

        pData = pd.DataFrame()

        pData["sample"] = sampleVec
        pData["condition"] = conditionVec
        pData["batch"] = 0

        self.logger.info("DE Sample DataFrame ready. Shape {}".format(pData.shape))

        fData = pd.DataFrame()
        availSamples = [x for x in exprData.columns if not x in ["mass"]]
        for sample in availSamples:
            fData[sample] = masses

            if outdirectory == None:
                #only needed for empire ...
                break

        if outdirectory != None and "empire" in use_methods:

            fillCondition = not resKey in self.de_results_all["empire"]

            if replaceExisting or fillCondition:
                self.logger.info("Starting EMPIRE; Writing Expression Files")

                empExprData = exprData.copy(deep=True)

                if count_scale != None and "empire" in count_scale:
                    empExprData = empExprData * count_scale["empire"]

                    if count_scale["empire"] > 1:
                        empExprData = empExprData.astype(int)

                empExprData.to_csv(outdirectory + "/exprs.txt", index=False,header=False, sep="\t")
                empExprData = None
                pDataOut = outdirectory+"/p_data.txt"
                pData.to_csv(pDataOut, index=False, sep="\t")
                fData.to_csv(outdirectory+"/f_data.txt", index=False, header=False, sep="\t")
                
                nlOutput = outdirectory + "/"+out_prefix+"." + "_".join([str(z) for z in resKey[0]]) +"." + "_".join([str(z) for z in resKey[1]]) + ".tsv"

                self.logger.info("Starting EMPIRE; Running nlEmpiRe")
                self.run_nlempire(outdirectory, pData, pDataOut, nlOutput)

                if os.path.isfile(nlOutput):
                    print("EMPIRE output available: {}".format(nlOutput))
                    empireData=pd.read_csv(nlOutput, delimiter="\t")
                    self.de_results_all["empire"][resKey] = empireData

            else:
                self.logger.info("Skipping empire for: {}, {}, {}".format(resKey, replaceExisting, fillCondition))


        diffxPyTests = set(["ttest", "rank"])
        if len(diffxPyTests.intersection(use_methods)) > 0:

            for testMethod in use_methods:

                if not testMethod in diffxPyTests:
                    continue

                fillCondition = not resKey in self.de_results_all[testMethod]

                if replaceExisting or fillCondition:

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
                        var=pd.DataFrame(index=[x for x in fData[availSamples[0]]]),
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

                else:
                    self.logger.info("Skipping {} for: {}, {}, {}".format(testMethod, resKey, replaceExisting, fillCondition))


        return exprData, pData, fData


    def list_de_results(self):
        
        allDERes = []
        for x in self.de_results_all:
            for y in self.de_results_all[x]:
                allDERes.append((x,y))

        return allDERes

    def find_de_results(self, keypart):

        results = []
        for method in self.de_results_all:
            for key in self.de_results_all[method]:

                if keypart == key[0] or keypart == key[1]:
                    results.append( (method, key) )

        return results



    def get_de_results(self, key):

        results = {}
        for method in self.de_results_all:
            if key in self.de_results_all[method]:
                results[method] = self.de_results_all[method][key]

        return results

    def get_de_result(self, method, key):

        rets = []
        for x in self.de_results_all:
            if x == method:
                for y in self.de_results_all[x]:
                    if y == key:
                        return self.de_results_all[x][y]
        return None



class ProteinWeights():

    def __init__(self, filename):

        self.protein2mass = {}
        self.protein_name2id = {}


        with open(filename) as fin:
            col2idx = {}
            for lidx, line in enumerate(fin):

                line = line.strip().split("\t")

                if lidx == 0:
                    for eidx, elem in enumerate(line):

                        col2idx[elem] = eidx

                    continue

                #protein_id	gene_symbol	mol_weight_kd	mol_weight

                if len(line) != 4:
                    continue

                proteinIDs = line[col2idx["protein_id"]].split(";")
                proteinNames = line[col2idx["gene_symbol"]].split(";")
                molWeight = float(line[col2idx["mol_weight"]])


                if len(proteinNames) == 0:
                    proteinNames = proteinIDs

                for proteinName in proteinNames:
                    self.protein2mass[proteinName] = molWeight
                    self.protein_name2id[proteinName] = proteinIDs

    def get_protein_from_mass(self, mass, maxdist=2):

        possibleMatches = []

        for protein in self.protein2mass:
            protMass = self.protein2mass[protein]
            if abs(mass-protMass) < maxdist:
                possibleMatches.append((protein, protMass))

        return possibleMatches



class pyIMS():

    def __init__(self):
        lib.StatisticalRegionMerging_New.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint8]
        lib.StatisticalRegionMerging_New.restype = ctypes.c_void_p

        lib.SRM_processFloat.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        lib.SRM_processFloat.restype = ctypes.POINTER(ctypes.c_uint32)

        lib.SRM_calc_similarity.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        lib.SRM_calc_similarity.restype = ctypes.POINTER(ctypes.c_float)

        
        lib.SRM_test_matrix.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        lib.SRM_test_matrix.restype = None

        lib.StatisticalRegionMerging_mode_dot.argtypes = [ctypes.c_void_p]
        lib.StatisticalRegionMerging_mode_dot.restype = None

        lib.StatisticalRegionMerging_mode_eucl.argtypes = [ctypes.c_void_p]
        lib.StatisticalRegionMerging_mode_eucl.restype = None

        self.logger = logging.getLogger('pyIMS')
        self.logger.setLevel(logging.INFO)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)

        self.logger.addHandler(consoleHandler)

        formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
        consoleHandler.setFormatter(formatter)

    def calc_similarity(self, inputarray):

        #load image
        dims = 1

        inputarray = inputarray.astype(np.float32)
        
        if len(inputarray.shape) > 2:
            dims = inputarray.shape[2]

        qs = []
        qArr = (ctypes.c_float * len(qs))(*qs)

        self.logger.info("Creating C++ obj")

        #self.obj = lib.StatisticalRegionMerging_New(dims, qArr, 3)
        #print(inputarray.shape)
        #testArray = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]],[[25,26,27],[28,29,30]],[[31,32,33],[34,35,36]]], dtype=np.float32)
        #print(testArray.shape)
        #image_p = testArray.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        #retValues = lib.SRM_test_matrix(self.obj, testArray.shape[0], testArray.shape[1], image_p)
        #exit()

        print("dimensions", dims)

        self.obj = lib.StatisticalRegionMerging_New(dims, qArr, len(qs))
        self.logger.info("Switching to dot mode")
        lib.StatisticalRegionMerging_mode_dot(self.obj)

        image_p = inputarray.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.logger.info("Starting calc similarity c++")
        retValues = lib.SRM_calc_similarity(self.obj, inputarray.shape[0], inputarray.shape[1], image_p)

        outclust = np.ctypeslib.as_array(retValues, shape=(inputarray.shape[0]*inputarray.shape[1], inputarray.shape[0]*inputarray.shape[1]))

        print(outclust.dtype)
        print(outclust.shape)

        self.logger.info("displaying matrix")
        plt.imshow(outclust)

        plt.show()

        return outclust



    def segment_array(self, inputarray, qs=[256, 0.5, 0.25], imagedim = None, dotMode = False):

        dims = 1

        if len(inputarray.shape) > 2:
            dims = inputarray.shape[2]

        qArr = (ctypes.c_float * len(qs))(*qs)

        self.logger.info("Creating SRM Object with {} dimensions".format(dims))
        self.obj = lib.StatisticalRegionMerging_New(dims, qArr, len(qs))

        if dotMode:
            self.logger.info("Switching to dot mode")
            lib.StatisticalRegionMerging_mode_dot(self.obj)
            dimage = inputarray
        else:
            dimage = (inputarray / np.max(inputarray)) * 255

        image_p = dimage.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        retValues = lib.SRM_processFloat(self.obj, dimage.shape[0], dimage.shape[1], image_p)

        outclust = np.ctypeslib.as_array(retValues, shape=(len(qs), dimage.shape[0], dimage.shape[1]))

        self.logger.debug(outclust.dtype)
        self.logger.debug(outclust.shape)

        outdict = {}

        for i,q in enumerate(qs):
            outdict[q] = outclust[i, :,:]

        if imagedim == None:
            imagedim = int(dims/3)

        image = inputarray[:,:,imagedim]
        image = image / np.max(image)

        return image, outdict



    def segment_image(self, imagepath, qs=[256, 0.5, 0.25]):

        #load image
        image = plt.imread(imagepath)
        image = image.astype(np.float32)
        image = image / np.max(image)

        print(image.shape)
        print(image.dtype)
        print(np.min(image), np.max(image))

        dims = 1

        if len(image.shape) > 2:
            dims = image.shape[2]

        qArr = (ctypes.c_float * len(qs))(*qs)

        self.obj = lib.StatisticalRegionMerging_New(dims, qArr, len(qs))
        
        dimage = image * 255
        image_p = dimage.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        retValues = lib.SRM_processFloat(self.obj, dimage.shape[0], dimage.shape[1], image_p)

        outclust = np.ctypeslib.as_array(retValues, shape=(len(qs), dimage.shape[0], dimage.shape[1]))

        print(outclust.dtype)
        print(outclust.shape)

        outdict = {}

        for i,q in enumerate(qs):
            outdict[q] = outclust[i, :,:]

        return image, outdict


class IMZMLExtract:

    def __init__(self, fname):
        #fname = "/mnt/d/dev/data/190724_AR_ZT1_Proteins/190724_AR_ZT1_Proteins_spectra.imzML"

        self.logger = logging.getLogger('IMZMLExtract')
        self.logger.setLevel(logging.INFO)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)

        self.logger.addHandler(consoleHandler)

        self.fname = fname
        self.parser = ImzMLParser(fname)
        self.dregions = None

        self.mzValues = self.parser.getspectrum(0)[0]

        self.specStart = 0

        if self.specStart != 0:
            self.mzValues = self.mzValues[self.specStart:]
            print("WARNING: SPECTRA STARTING AT POSITION", self.specStart)

        self.find_regions()

    def get_spectrum(self, specid, normalize=False):
        spectra1 = self.parser.getspectrum(specid)
        spectra1 = spectra1[1]

        if normalize:
            spectra1 = spectra1 / max(spectra1)
        
        return spectra1

    def compare_spectra(self, specid1, specid2):

        spectra1 = self.parser.getspectrum(specid1)[1]
        spectra2 = self.parser.getspectrum(specid2)[1]

        ssum = 0.0
        len1 = 0.0
        len2 = 0.0

        assert(len(spectra1) == len(spectra2))

        for i in range(0, len(spectra1)):

            ssum += spectra1[i] * spectra2[i]
            len1 += spectra1[i]*spectra1[i]
            len2 += spectra2[i]*spectra2[i]

        len1 = math.sqrt(len1)
        len2 = math.sqrt(len2)

        return ssum/(len1*len2)


    def get_mz_index(self, value, threshold=None):

        curIdxDist = 1000000
        curIdx = None

        for idx, x in enumerate(self.mzValues):
            dist = abs(x-value)

            if dist < curIdxDist and (threhsold==None or dist < threshold):
                curIdx = idx
                curIdxDist = dist
            
        return curIdx

    def get_region_indices(self, regionid):

        if not regionid in self.dregions:
            return None
        
        outindices = {}

        for coord in self.dregions[regionid]:

            spectID = self.parser.coordinates.index(coord)

            if spectID == None or spectID < 0:
                print("Invalid coordinate", coord)
                continue

            outindices[coord] = spectID

        return outindices

    def get_region_spectra(self, regionid):

        if not regionid in self.dregions:
            return None
        
        outspectra = {}

        for coord in self.dregions[regionid]:

            spectID = self.parser.coordinates.index(coord)

            if spectID == None or spectID < 0:
                print("Invalid coordinate", coord)
                continue

            cspec = self.get_spectrum( spectID )
            cspec = cspec[self.specStart:]# / 1.0
            #cspec = cspec/np.max(cspec)
            outspectra[coord] = cspec

        return outspectra


    def get_avg_region_spectrum(self, regionid):

        region_spects = self.get_region_array(regionid)

        return self.get_avg_spectrum(region_spects)

    def get_avg_spectrum(self, region_spects):

        avgarray = np.zeros((1, region_spects.shape[2]))

        for i in range(0, region_spects.shape[0]):
            for j in range(0, region_spects.shape[1]):

                avgarray[:] += region_spects[i,j,:]

        avgarray = avgarray / (region_spects.shape[0]*region_spects.shape[1])

        return avgarray[0]



    def get_region_range(self, regionid):

        allpixels = self.dregions[regionid]

        minx = min([x[0] for x in allpixels])
        maxx = max([x[0] for x in allpixels])

        miny = min([x[1] for x in allpixels])
        maxy = max([x[1] for x in allpixels])

        minz = min([x[2] for x in allpixels])
        maxz = max([x[2] for x in allpixels])

        spectraLength = 0
        for coord in self.dregions[regionid]:

            spectID = self.parser.coordinates.index(coord)

            if spectID == None or spectID < 0:
                print("Invalid coordinate", coord)
                continue

            splen = self.parser.mzLengths[spectID]-self.specStart

            spectraLength = max(spectraLength, splen)

        return (minx, maxx), (miny, maxy), (minz, maxz), spectraLength

    def get_region_shape(self, regionid):

        rr = self.get_region_range(regionid)
        xr,yr,zr,sc = rr

        imzeShape = [
            xr[1]-xr[0]+1,
            yr[1]-yr[0]+1
        ]

        if zr[1]-zr[0]+1 > 1:
            imzeShape.append( zr[1]-zr[0]+1 )

        imzeShape.append(sc)

        spectraShape = tuple(imzeShape)

        return spectraShape


    def __get_peaks(self, spectrum, window):

        peaks=set()

        for i in range(0, len(spectrum)-window):

            intens = spectrum[i:i+window]

            maxI = 0
            maxMZ = 0

            epshull = (max(intens) - min(intens)) / 2

            for mzIdx, mzVal in enumerate(intens):
                if mzVal > maxI:
                    maxI = mzVal
                    maxMZ = mzIdx

            tmp = maxMZ

            addPeak = True
            if len(peaks) > 0:

                # exist already registered peak within epsilon hull with lower intensity?
                for p in peaks:

                    if abs(p - tmp) < epshull:
                        if spectrum[p] < spectrum[tmp]:
                            peaks.remove(p)
                            peaks.add(tmp)
                            addPeak = False
                            break
                        else:

                            addPeak = False
                            break

            if addPeak:

                if maxI > 5 * np.median(intens):
                    peaks.add(tmp)

        return sorted(peaks)


    def get_peaks_fast(self, spectrum, window):

        peaks=set()

        for i in range(window, len(spectrum)-window):

            intens = spectrum[i-window:i+window]

            maxelem = np.argmax(intens)

            if maxelem == window:

                minvalue = np.min(intens)
                peakvalue = intens[window]

                assert(peakvalue == intens[maxelem])

                if peakvalue * 0.5 > minvalue:
                    assert(spectrum[i] == intens[maxelem])
                    peaks.add(i)

        return sorted(peaks)

    def to_peak_region(self, region_array, peak_window = 100):
        
        avg_spectrum = self.get_avg_spectrum(region_array)

        peaks = self.get_peaks_fast(avg_spectrum, peak_window)
        peak_region = np.zeros((region_array.shape[0], region_array.shape[1], len(peaks)))

        for i in range(0, region_array.shape[0]):
            for j in range(0, region_array.shape[1]):

                pspectrum = region_array[i,j,peaks]
                peak_region[i,j,:] = pspectrum

        return peak_region, peaks



    def normalize_spectrum(self, spectrum, normalize=None, max_region_value=None):

        assert (normalize in [None, "max_intensity_spectrum", "max_intensity_region", "max_intensity_all_regions", "vector"])

        if normalize in ["max_intensity_region", "max_intensity_all_regions"]:
            assert(max_region_value != None)

        if normalize == "max_intensity_spectrum":
            spectrum = spectrum / np.max(spectrum)
            return spectrum

        elif normalize in ["max_intensity_region", "max_intensity_all_regions"]:
            spectrum = spectrum / max_region_value
            return spectrum

        elif normalize == "vector":

            slen = np.linalg.norm(spectrum)
            spectrum = spectrum / slen

            return spectrum



    def normalize_region_array(self, region_array, normalize=None):

        assert (normalize in [None, "max_intensity_spectrum", "max_intensity_region", "max_intensity_all_regions", "vector", "inter_median", "intra_median"])


        if normalize in ["inter_median", "intra_median"]:
            
            ref_spectra = region_array[0,0,:] + 0.01

            if normalize == "intra_median":

                intra_norm = np.zeros(region_array.shape)
                for i in range(region_array.shape[0]):
                    for j in range(region_array.shape[1]):
                        median = np.median(region_array[i,j,:]/ref_spectra)

                        if median != 0:
                            intra_norm[i,j,:] = region_array[i,j, :]/median
                        else:
                            intra_norm[i,j,:] = region_array[i,j,:]

                return intra_norm

            if normalize == "inter_median":
                global_fcs = Counter()
                scalingFactor = 100000

                self.logger.info("Collecting fold changes")
                for i in range(region_array.shape[0]):
                    for j in range(region_array.shape[1]):

                        foldchanges = (scalingFactor * region_array[i][j] / ref_spectra).astype(int)
                        for fc in foldchanges:
                            global_fcs[fc] += 1


                
                totalElements = sum([global_fcs[x] for x in global_fcs])
                self.logger.info("Got a total of {} fold changes".format(totalElements))
                
                if totalElements % 2 == 1:
                    medianElements = [int(totalElements/2), int(totalElements/2)+1]
                else:
                    medianElements = [int(totalElements/2)]

                sortedFCs = sorted([x for x in global_fcs])

                self.logger.info("Median elements {}".format(medianElements))

                medians = {}

                currentCount = 0
                for i in sortedFCs:
                    fcAdd = global_fcs[i]
                    for medElem in medianElements:
                        if currentCount < medElem <= currentCount+fcAdd:
                            medians[medElem] = i

                    currentCount += fcAdd

                self.logger.info("Median elements".format(medians))

                global_median = sum([medians[x] for x in medians]) / len(medians)
                global_median = global_median / scalingFactor

                self.logger.info("Global Median".format(global_median))

                inter_norm = np.array(region_array, copy=True)

                if global_median != 0:
                    inter_norm = inter_norm / global_median

                return inter_norm


        region_dims = region_array.shape
        outarray = np.array(region_array, copy=True)

        maxInt = 0.0
        for i in range(0, region_dims[0]):
            for j in range(0, region_dims[1]):

                spectrum = region_array[i, j, :]

                if normalize in ['max_intensity_region', 'max_intensity_all_regions']:
                    maxInt = max(maxInt, np.max(spectrum))

                else:
                    spectrum = self.normalize_spectrum(spectrum, normalize=normalize)
                    outarray[i, j, :] = spectrum

        if not normalize in ['max_intensity_region', 'max_intensity_all_regions']:
            return None

        if normalize in ["max_intensity_all_regions"]:
            for idx, _ in enumerate(self.parser.coordinates):
                mzs, intensities = p.getspectrum(idx)
                maxInt = max(maxInt, np.max(intensities))


        for i in range(0, region_dims[0]):
            for j in range(0, region_dims[1]):

                spectrum = outarray[i, j, :]
                spectrum = self.normalize_spectrum(spectrum, normalize=normalize, max_region_value=maxInt)
                outarray[i, j, :] = spectrum

        return outarray

    def plot_toc(self, region_array):
        region_dims = region_array.shape
        peakplot = np.zeros((region_array.shape[0],region_array.shape[1]))
        for i in range(0, region_dims[0]):
            for j in range(0, region_dims[1]):

                spectrum = region_array[i, j, :]
                peakplot[i,j] = sum(spectrum)

        heatmap = plt.matshow(peakplot)
        plt.colorbar(heatmap)
        plt.show()
        plt.close()




    def list_highest_peaks(self, region_array, counter=False):
        region_dims = region_array.shape

        peakplot = np.zeros((region_array.shape[0],region_array.shape[1]))
        maxPeakCounter = Counter()
        allPeakIntensities = []
        for i in range(0, region_dims[0]):
            for j in range(0, region_dims[1]):

                spectrum = region_array[i, j, :]

                idx = np.argmax(spectrum, axis=None)
                mzInt = spectrum[idx]
                mzVal = self.mzValues[idx]
                peakplot[i,j] = mzVal
                allPeakIntensities.append(mzInt)

                if not counter:
                    print(i,j,mzVal)
                else:
                    maxPeakCounter[mzVal] += 1

        if counter:
            for x in sorted([x for x in maxPeakCounter]):
                print(x, maxPeakCounter[x])

        heatmap = plt.matshow(peakplot)
        plt.colorbar(heatmap)
        plt.show()
        plt.close()

        print(len(allPeakIntensities), min(allPeakIntensities), max(allPeakIntensities), sum(allPeakIntensities)/len(allPeakIntensities))

        plt.hist(allPeakIntensities, bins=len(allPeakIntensities), cumulative=True, histtype="step")
        plt.show()
        plt.close()

    def get_pixel_spectrum(self, regionid, specCoords):

        xr,yr,zr,sc = self.get_region_range(regionid)
        
        totalCoords = (specCoords[0]+xr[0], specCoords[1]+yr[0], 1)

        spectID = self.parser.coordinates.index(totalCoords)

        if spectID == None or spectID < 0:
            print("Invalid coordinate", totalCoords)
            return None

        cspec = self.get_spectrum( spectID )
        return cspec, spectID, totalCoords
        
    def get_region_index_array(self, regionid):
        xr,yr,zr,sc = self.get_region_range(regionid)
        rs = self.get_region_shape(regionid)
        self.logger.info("Found region {} with shape {}".format(regionid, rs))

        sarray = np.zeros( (rs[0], rs[1]), dtype=np.float32 )

        coord2spec = self.get_region_indices(regionid)

        for coord in coord2spec:
            xpos = coord[0]-xr[0]
            ypos = coord[1]-yr[0]

            specIdx = coord2spec[coord]

            sarray[xpos, ypos] = specIdx

        return sarray
    
    
    
    def __findBestShift( self, refspectrum, allspectra, maxshift ):
        idx2shift = {}
        idx2shifted = {}
        
        bar = progressbar.ProgressBar()
        
        for idx, aspec in enumerate(bar(allspectra)):
            
            bestsim = 0
            bestshift = -maxshift
            
            for ishift in range(-maxshift, maxshift, 1):
                shifted = aspec[maxshift+ishift:-maxshift+ishift]
                newsim = self.__cos_similarity(refspectrum, shifted)
                
                if newsim > bestsim:
                    bestsim = newsim
                    bestshift = ishift
                    
            idx2shift[idx] = bestshift
            idx2shifted[idx] = aspec[maxshift+bestshift:-maxshift+bestshift]
            
        return idx2shift, idx2shifted

    def __cos_similarity(self, vA, vB):
        assert(len(vA) == len(vB))
        return np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))


    def shift_region_array(self, reg_array, masses, maxshift, ref_coord=(0,0)):
        
        ref_spec = reg_array[ref_coord[0], ref_coord[1], maxshift:-maxshift]
        outarray = np.zeros((reg_array.shape[0], reg_array.shape[1], len(ref_spec)))
        
        idx2coord = {}
        coord2idx = {}
        specs = []
        
        for i in range(0, reg_array.shape[0]):
            for j in range(0, reg_array.shape[1]):
                
                idx2coord[len(specs)] = (i,j)
                coord2idx[(i,j)] = len(specs)
                specs.append(reg_array[i,j,:])
                
        i2s, i2sp = self.__findBestShift(ref_spec, specs, maxshift)
        
        shifts = sorted([i2s[x] for x in i2s])
        meanShift = np.mean(shifts)
        medianShift = np.median(shifts)
        
        print("Shifts: mean: {}, median: {}".format(meanShift, medianShift))
        
        for idx in i2sp:
            idxcoords = idx2coord[idx]
            shspec = i2sp[idx]
            outarray[idxcoords[0], idxcoords[1],] = shspec
            
        return outarray, masses[maxshift:-maxshift]
    

    def remove_background_spec_aligned(self, array, bgSpec, masses, maxshift):
        assert(not bgSpec is None)
        print(bgSpec.shape)
        print(array.shape)
        assert(len(bgSpec) == array.shape[2])

        outarray = np.zeros((array.shape[0], array.shape[1], array.shape[2]-2*maxshift))
        bspec = bgSpec[maxshift:-maxshift]

        

        for i in range(0, array.shape[0]):
            for j in range(0, array.shape[1]):

                aspec = array[i,j,:]
                bestsim = 0
                for ishift in range(-maxshift, maxshift, 1):
                    shifted = aspec[maxshift+ishift:-maxshift+ishift]
                    newsim = self.__cos_similarity(bspec, shifted)
                    
                    if newsim > bestsim:
                        bestsim = newsim
                        bestshift = ishift
                        
                aspec = aspec[maxshift+bestshift:-maxshift+bestshift]
                aspec = aspec - bspec
                aspec[aspec < 0.0] = 0.0

                outarray[i,j,:] = aspec

        return outarray, masses[maxshift:-maxshift]


    def remove_background_spec(self, array, bgSpec):
        assert(not bgSpec is None)
        print(bgSpec.shape)
        print(array.shape)
        assert(len(bgSpec) == array.shape[2])

        for i in range(0, array.shape[0]):
            for j in range(0, array.shape[1]):
                newspec = array[i,j,:] - bgSpec
                newspec[newspec < 0.0] = 0.0

                array[i,j,:] = newspec

        return array


    def get_region_array(self, regionid, makeNullLine=True, bgspec=None):

        xr,yr,zr,sc = self.get_region_range(regionid)
        rs = self.get_region_shape(regionid)
        self.logger.info("Found region {} with shape {}".format(regionid, rs))

        sarray = np.zeros( rs, dtype=np.float32 )

        coord2spec = self.get_region_spectra(regionid)

        for coord in coord2spec:
            xpos = coord[0]-xr[0]
            ypos = coord[1]-yr[0]

            spectra = coord2spec[coord]

            if len(spectra) < sc:
                spectra = np.pad(spectra, ((0,0),(0, sc-len(spectra) )), mode='constant', constant_values=0)

            spectra = np.array(spectra, copy=True)

            if not bgspec is None:
                spectra = spectra - bgspec

            if makeNullLine:
                spectra[spectra < 0.0] = 0.0
                spectra = spectra - np.min(spectra)

            sarray[xpos, ypos, :] = spectra

        return sarray


    def list_regions(self):

        allMaxX = 0
        allMaxY = 0

        allregionInfo = {}
        for regionid in self.dregions:

            allpixels = self.dregions[regionid]

            minx = min([x[0] for x in allpixels])
            maxx = max([x[0] for x in allpixels])

            miny = min([x[1] for x in allpixels])
            maxy = max([x[1] for x in allpixels])

            allMaxX = max(maxx, allMaxX)
            allMaxY = max(maxy, allMaxY)

            infotuple = ((minx, maxx, miny, maxy), len(allpixels))
            print(regionid, infotuple)
            allregionInfo[regionid] = infotuple

        outimg = np.zeros((allMaxY, allMaxX))

        for regionid in self.dregions:

            allpixels = self.dregions[regionid]

            minx = min([x[0] for x in allpixels])
            maxx = max([x[0] for x in allpixels])

            miny = min([x[1] for x in allpixels])
            maxy = max([x[1] for x in allpixels])

            outimg[miny:maxy, minx:maxx] = regionid+1

        heatmap = plt.matshow(outimg)
        for regionid in self.dregions:
            allpixels = self.dregions[regionid]

            minx = min([x[0] for x in allpixels])
            maxx = max([x[0] for x in allpixels])

            miny = min([x[1] for x in allpixels])
            maxy = max([x[1] for x in allpixels])

            middlex = minx + (maxx-minx)/ 2.0
            middley = miny + (maxy-miny)/ 2.0

            plt.text(middlex, middley, str(regionid))

        plt.colorbar(heatmap)
        plt.show()
        plt.close()

        return allregionInfo
        


    def find_regions(self):

        if os.path.isfile(self.fname + ".regions"):

            print("Opening regions file for", self.fname)

            with open(self.fname + ".regions", 'r') as fin:
                self.dregions = defaultdict(list)

                for line in fin:
                    line = line.strip().split("\t")

                    coords = [int(x) for x in line]

                    self.dregions[coords[3]].append( tuple(coords[0:3]) )

            for regionid in self.dregions:

                allpixels = self.dregions[regionid]

                minx = min([x[0] for x in allpixels])
                maxx = max([x[0] for x in allpixels])

                miny = min([x[1] for x in allpixels])
                maxy = max([x[1] for x in allpixels])

                #print(regionid, minx, maxx, miny, maxy)

        else:

            self.dregions = self.__detectRegions(self.parser.coordinates)

            with open(self.fname + ".regions", 'w') as outfn:

                for regionid in self.dregions:

                    for pixel in self.dregions[regionid]:

                        print("\t".join([str(x) for x in pixel]), regionid, sep="\t", file=outfn)
    
    
    def __dist(self, x,y):

        assert(len(x)==len(y))

        dist = 0
        for pidx in range(0, len(x)):

            dist += abs(x[pidx]-y[pidx])

        return dist

    def _detectRegions(self, allpixels):
        return self.__detectRegions(allpixels)

    def __detectRegions(self, allpixels):

        allregions = []

        for idx, pixel in enumerate(allpixels):

            if len(allregions) == 0:
                allregions.append([pixel])
                continue

            if idx % 1000 == 0:
                print("At pixel", idx , "of", len(allpixels), "with", len(allregions), "regions")


            accRegions = []

            for ridx, region in enumerate(allregions):

                #minx = min([x[0] for x in region])
                #maxx = max([x[0] for x in region])

                #miny = min([x[1] for x in region])
                #maxy = max([x[1] for x in region])

                #if pixel[0] - maxx > 100:
                #    continue

                #if pixel[0] - minx > 100:
                #    continue

                #if pixel[1] - maxy > 100:
                #    continue

                #if pixel[1] - miny > 100:
                #    continue

                if pixel[0] == 693 and pixel[1] == 317:
                    print("proc pixel")

                pixelAdded = False
                for coord in region:
                    if self.__dist(coord, pixel) <= 1:
                        accRegions.append(ridx)
                        pixelAdded = False
                        break

                if not pixelAdded:
                    if pixel[0] == 693 and pixel[1] == 317:
                        print("Pixel not added", pixel)

            if pixel[0] == 693 and pixel[1] == 317:
                print("Pixel acc regions", accRegions)

            if len(accRegions) == 0:
                allregions.append([pixel])

            elif len(accRegions) == 1:

                for ridx in accRegions:
                    allregions[ridx].append(pixel)

            elif len(accRegions) > 1:

                bc = len(allregions)

                totalRegion = [pixel]
                for ridx in accRegions:
                    totalRegion += allregions[ridx]

                for ridx in sorted(accRegions, reverse=True):
                    del allregions[ridx]

                allregions.append(totalRegion)

                ac = len(allregions)

                assert(ac == bc + 1 - len(accRegions))

        outregions = {}

        for i in range(0, len(allregions)):
            outregions[i] = [tuple(x) for x in allregions[i]]

        return outregions


if __name__ == '__main__':

    #img = Image.open("/Users/rita/Uni/bachelor_thesis/test2.tiff")
    #img = np.asarray(img, dtype=np.float32)
    #img = cv2.resize(img, (320, 240), interpolation = cv2.INTER_AREA) 
    #imageio.imwrite("/Users/rita/Uni/bachelor_thesis/test2_smaller.png", img)
    #seg = Segmenter()

    #imze = IMZMLExtract("/mnt/d/dev/data/msi/190724_AR_ZT1_Proteins/190724_AR_ZT1_Proteins_spectra.imzML")
    imze = IMZMLExtract("/mnt/d/dev/data/msi/slideD/181114_AT1_Slide_D_Proteins.imzML")
    spectra = imze.get_region_array(0)


    print("Got spectra", spectra.shape)
    print("mz index", imze.get_mz_index(6662))

    imze.normalize_region_array(spectra, normalize="max_intensity_region")
    spectra.spectra_log_dist()

    exit()

    seg = Segmenter()
    #seg.calc_similarity(spectra)

    #exit(0)

    #image, regions = seg.segment_image("/mnt/d/dev/data/mouse_pictures/segmented/test1_smaller.png", qs=[256, 0.5, 0.25, 0.0001, 0.00001])
    image, regions = seg.segment_array(spectra, qs=[1,0.5, 0.25, 0.1, 0.01], imagedim=imze.get_mz_index(6662), dotMode=True)

    f, axarr = plt.subplots(len(regions), 2)

    for i,q in enumerate(regions):

        curdata = regions[q]
        uniques = np.unique(curdata)
        print("Q", q, len(uniques))

        if len(uniques) < 100:
            print(uniques)
        print()

        axarr[i, 0].imshow( image )
        axarr[i, 1].imshow( curdata )

    plt.show()
    plt.close()