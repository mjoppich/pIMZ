import numpy as np
from scipy import misc
import ctypes
import dabest

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os,sys
import imageio
from PIL import Image
from natsort import natsorted

from collections import defaultdict
from pyimzml.ImzMLParser import ImzMLParser, browse, getionimage
import logging
import _pickle as pickle
import math

baseFolder = str(os.path.dirname(os.path.realpath(__file__)))

lib = ctypes.cdll.LoadLibrary(baseFolder+'/../../cppSRM/lib/libSRM.so')

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform

import scipy.cluster as spc

import scipy as sp
import sklearn as sk

import umap
import hdbscan

class SpectraRegion():

    def __init__(self, region_array, idx2mass):

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

        self.logger = logging.getLogger('SpectraRegion')
        self.logger.setLevel(logging.INFO)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)

        self.logger.addHandler(consoleHandler)

        formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
        consoleHandler.setFormatter(formatter)


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

        self.de_results = {}

        for i in range(self.region_array.shape[0]*self.region_array.shape[1]):

            x,y = divmod(i, self.region_array.shape[1])

            self.idx2pixel[i] = (x,y)
            self.pixel2idx[(x,y)] = i


    def __get_exmass_for_mass(self, mass):
        
        dist2mass = float('inf')
        curMass = -1
        curIdx = -1

        for xidx,x in enumerate(self.idx2mass):
            if abs(x-mass) < dist2mass:
                dist2mass = abs(x-mass)
                curMass = x    
                curIdx = xidx    

        return curMass, curIdx


    def mass_heatmap(self, masses, log=False):

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

        heatmap = plt.matshow(image)
        plt.colorbar(heatmap)
        plt.show()
        plt.close()


    def __calc_similarity(self, inputarray):
        # load image
        dims = 1

        inputarray = inputarray.astype(np.float32)

        if len(inputarray.shape) > 2:
            dims = inputarray.shape[2]

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

    def calculate_similarity(self, mode="spectra"):
        """

        :param mode: must be in  ["spectra", "spectra_log", "spectra_log_dist"]

        :return: spectra similarity matrix
        """

        assert(mode in ["spectra", "spectra_log", "spectra_log_dist"])

        self.spectra_similarity = self.__calc_similarity(self.region_array)

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
            n_neighbors=30,
            min_dist=0.0,
            n_components=40,
            random_state=42,
        ).fit_transform(self.elem_matrix)

        self.logger.info("HDBSCAN reduction")

        clusterer = hdbscan.HDBSCAN(
            min_samples=10,
            min_cluster_size=10,
        )
        self.dimred_labels = clusterer.fit_predict(self.dimred_elem_matrix) + 2

        #c = spc.hierarchy.fcluster(clusterer.single_linkage_tree_.to_numpy(), t=10, criterion='maxclust')

        return self.dimred_labels

    def vis_umap(self):

        assert(not self.elem_matrix is None)
        assert(not self.dimred_elem_matrix is None)
        assert(not self.dimred_labels is None)

        plt.figure()
        clustered = (self.dimred_labels >= 0)
        plt.scatter(self.dimred_elem_matrix[~clustered, 0],
                    self.dimred_elem_matrix[~clustered, 1],
                    color=(0.5, 0.5, 0.5),
                    label="Unassigned",
                    s=0.3,
                    alpha=0.5)

        uniqueClusters = sorted(set([x for x in self.dimred_labels if x >= 0]))

        for cidx, clusterID in enumerate(uniqueClusters):
            cmap = matplotlib.cm.get_cmap('Spectral')

            clusterColor = cmap(cidx / len(uniqueClusters))

            plt.scatter(self.dimred_elem_matrix[self.dimred_labels == clusterID, 0],
                        self.dimred_elem_matrix[self.dimred_labels == clusterID, 1],
                        color=clusterColor,
                        label=str(clusterID),
                        s=0.5)

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

        heatmap = plt.matshow(showcopy)
        plt.colorbar(heatmap)
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

        assert(method in ["remove_singleton", "most_similar_singleton", "merge_background"])

        cluster2coords = defaultdict(list)

        for i in range(0, self.segmented.shape[0]):
            for j in range(0, self.segmented.shape[1]):

                clusterID = self.segmented[i, j]
                cluster2coords[clusterID].append((i,j))


        if method == "remove_singleton":
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

    def __get_spectra_matrix(self,segments):

        cluster2coords = self.getCoordsForSegmented()

        relPixels = []
        for x in segments:
            relPixels += cluster2coords.get(x, [])

        spectraMatrix = np.zeros((len(relPixels), len(self.idx2mass)))

        for pidx, px in enumerate(relPixels):
            spectraMatrix[pidx, :] = self.region_array[px[0], px[1], :]

        return spectraMatrix


    def __get_expression_from_matrix(self, matrix, massValue, segments, mode="avg"):

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

    def deres_to_df(self, resKey, protWeights, keepOnlyProteins=True):

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

        deRes = self.de_results[resKey]      
        ttr = deRes.summary()

        ttr["log2fc"] = -ttr["log2fc"]
        self.logger.info("DE result for case {} with {} results".format(resKey, ttr.shape))

        fttr = ttr[ttr.qval.lt(0.05) & ttr.log2fc.abs().gt(0.5)]

        self.logger.info("DE result for case {} with {} results (filtered)".format(resKey, fttr.shape))


        targetSpectraMatrix = self.__get_spectra_matrix(resKey[0])
        bgSpectraMatrix = self.__get_spectra_matrix(resKey[1])

        self.logger.info("Created matrices with shape {} and {} (target, bg)".format(targetSpectraMatrix.shape, bgSpectraMatrix.shape))


        for row in fttr.iterrows():
            geneIDent = row[1]["gene"]
            
            ag = geneIDent.split("_")
            massValue = float("{}.{}".format(ag[1], ag[2]))

            foundProt = protWeights.get_protein_from_mass(massValue, maxdist=3)

            if keepOnlyProteins and len(foundProt) == 0:
                continue

            lfc = row[1]["log2fc"]
            qval = row[1]["qval"]

            expT, totalSpectra, measuredSpecta = self.__get_expression_from_matrix(targetSpectraMatrix, massValue, resKey[0], ["avg", "median"])
            exprBG, totalSpectraBG, measuredSpectaBG = self.__get_expression_from_matrix(bgSpectraMatrix, massValue, resKey[0], ["avg", "median"])

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


    def find_all_markers(self, protWeights, keepOnlyProteins=True, replaceExisting=False, includeBackground=True):
        cluster2coords = self.getCoordsForSegmented()

        df = pd.DataFrame()

        for segment in cluster2coords:

            clusters0 = [segment]
            clusters1 = [x for x in cluster2coords if not x in clusters0]

            if not includeBackground and 0 in clusters1:
                del clusters1[clusters1.index(0)]

            self.find_markers(clusters0=clusters0, clusters1=clusters1, replaceExisting=replaceExisting)

            # get result
            resKey = self.__make_de_res_key(clusters0, clusters1)

            resDF = self.deres_to_df(resKey, protWeights, keepOnlyProteins=keepOnlyProteins)

            df = pd.concat([df, resDF], sort=False)           

        return df

                    

    def __make_de_res_key(self, clusters0, clusters1):

        return (tuple(sorted(clusters0)), tuple(sorted(clusters1)))
        



    def find_markers(self, clusters0, clusters1=None, outdirectory=None, replaceExisting=False):

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

        resKey = self.__make_de_res_key(clusters0, clusters1)
        self.logger.info("DE result key: {}".format(resKey))

        if not replaceExisting:

            if resKey in self.de_results:
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

                exprData[pxl_name] = (10000* self.region_array[pxl[0], pxl[1], :]).astype('int')


        for clus in clusters1:
            self.logger.info("Processing cluster: {}".format(clus))

            allPixels = cluster2coords[clus]
            for pxl in allPixels:

                pxl_name = "{}__{}".format(str(len(sampleVec)), "_".join([str(x) for x in pxl]))

                sampleVec.append(pxl_name)
                conditionVec.append(1)

                exprData[pxl_name] = (10000* self.region_array[pxl[0], pxl[1], :]).astype('int')


        self.logger.info("DE DataFrame ready. Shape {}".format(exprData.shape))

        if outdirectory != None:
            exprData.to_csv(outdirectory + "/exprs.txt", index=False,header=False, sep="\t")

        pData = pd.DataFrame()

        pData["sample"] = sampleVec
        pData["condition"] = conditionVec
        pData["batch"] = 0

        self.logger.info("DE Sample DataFrame ready. Shape {}".format(pData.shape))

        if outdirectory != None:
            pData.to_csv(outdirectory+"/p_data.txt", index=False, sep="\t")


        fData = pd.DataFrame()
        availSamples = [x for x in exprData.columns if not x in ["mass"]]
        for sample in availSamples:
            fData[sample] = masses

            if outdirectory == None:
                #only needed for empire ...
                break
            
        if outdirectory != None:
            fData.to_csv(outdirectory+"/f_data.txt", index=False, header=False, sep="\t")

        if outdirectory == None:
            self.logger.info("NO outdirectory given. Performing DE-test")

            import diffxpy.api as de
            import anndata

            pdat = pData.copy()
            del pdat["sample"]

            deData = anndata.AnnData(
                X=exprData.values.transpose(),
                var=pd.DataFrame(index=[x for x in fData[availSamples[0]]]),
                obs=pdat
            )

            test = de.test.t_test(
                data=deData,
                grouping="condition"
            )


            self.de_results[ resKey ] = test
            self.logger.info("DE-test finished. Results available: {}".format(resKey))


        return exprData, pData, fData


    def list_de_results(self):
        return [x for x in self.de_results]


    def get_de_result(self, key):

        return self.de_results.get(key, None)

    def get_de_results(self, key):

        rets = []
        for x in self.de_results:
            if x[0] == key:
                rets.append(x)

        return rets



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


    def get_mz_index(self, value):

        curIdxDist = 1000000
        curIdx = 0

        for idx, x in enumerate(self.mzValues):
            dist = abs(x-value)

            if dist < curIdxDist:
                curIdx = idx
                curIdxDist = dist
            
        return curIdx

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
            cspec = cspec[self.specStart:]
            cspec = cspec/np.max(cspec)
            outspectra[coord] = cspec

        return outspectra

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

    def to_peak_region(self, region_array, peak_window = 100):

        peaks = set()

        for i in range(0, region_array.shape[0]):
            for j in range(0, region_array.shape[1]):

                spectrum = region_array[i,j,:]

                sp_peaks = self.__get_peaks(spectrum, peak_window)
                peaks = peaks.union(sp_peaks)


        peaks = sorted(peaks)

        peak_region = np.zeros((region_array.shape[0], region_array.shape[1], len(peaks)))

        for i in range(0, region_array.shape[0]):
            for j in range(0, region_array.shape[1]):

                pspectrum = region_array[i,j,peaks]
                peak_region[i,j,:] = pspectrum


        return peak_region



    def normalize_spectrum(self, spectrum, normalize=None, max_region_value=None):

        assert (normalize in [None, "max_intensity_spectrum", "max_intensity_region", "vector"])

        if normalize == "max_intensity_region":
            assert(max_region_value != None)

        if normalize == "max_intensity_spectrum":
            spectrum = spectrum / np.max(spectrum)
            return spectrum

        elif normalize == "max_intensity_region":
            spectrum = spectrum / max_region_value
            return spectrum

        elif normalize == "vector":

            slen = np.linalg.norm(spectrum)
            spectrum = spectrum / slen

            return spectrum



    def normalize_region_array(self, region_array, normalize=None):

        assert (normalize in [None, "max_intensity_spectrum", "max_intensity_region", "vector"])

        region_dims = region_array.shape

        maxInt = 0
        for i in range(0, region_dims[0]):
            for j in range(0, region_dims[1]):

                spectrum = region_array[i, j, :]

                if normalize == 'max_intensity_region':
                    maxInt = max(maxInt, np.max(spectrum))

                else:
                    spectrum = self.normalize_spectrum(spectrum, normalize=normalize)
                    region_array[i, j, :] = spectrum

        if normalize != 'max_intensity_region':
            return

        for i in range(0, region_dims[0]):
            for j in range(0, region_dims[1]):

                spectrum = region_array[i, j, :]
                spectrum = self.normalize_spectrum(spectrum, normalize=normalize, max_region_value=maxInt)
                region_array[i, j, :] = spectrum



    def get_region_array(self, regionid, makeNullLine=True):

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


            if makeNullLine:
                spectra[spectra < 0.0] = 0.0
                spectra = spectra - np.min(spectra)

            sarray[xpos, ypos, :] = spectra

        return sarray

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

                for coord in region:
                    if self.__dist(coord, pixel) <= 1:
                        accRegions.append(ridx)
                        break


            if len(accRegions) == 0:
                allregions.append([pixel])

            elif len(accRegions) == 1:

                for ridx in accRegions:
                    allregions[ridx].append(pixel)

            elif len(accRegions) > 1:

                bc = len(allregions)

                totalRegion = []
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