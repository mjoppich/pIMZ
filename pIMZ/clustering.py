# general
import logging
import json
import os
import random
import math
from collections import defaultdict, Counter
import glob
import shutil, io, base64
from typing import OrderedDict

# general package
from natsort import natsorted
import pandas as pd
import numpy as np
import regex as re
import h5py


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

from scipy import ndimage, stats
from scipy.spatial.distance import squareform, pdist
import scipy.cluster as spc
from scipy.cluster.vq import kmeans2
from sklearn import cluster, decomposition
from fcmeans import FCM

from .imzml import IMZMLExtract
from .regions import SpectraRegion, RegionClusterer

#web/html
import jinja2

# applications
import progressbar
def makeProgressBar():
    return progressbar.ProgressBar(widgets=[
        progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.AdaptiveETA()
        ])

import abc

import networkx as nx

class RegionModel:

    def __init__(self, no_relation_weight=0, bi_directional=True) -> None:

        self.no_relation_weight = no_relation_weight
        self.bi_directional = bi_directional
        self.relations = nx.DiGraph()


    def from_image(self, filepath, mapping=None, diagonal=True):

        regImg =np.load(filepath)

        if mapping is None:
            mapping = {x:x for x in np.unique(regImg)} #id

        if not set(np.unique(regImg)).issubset([x for x in mapping]):
            raise ValueError

        adjacencyCounter = Counter()

        for i in range(0, regImg.shape[0]):
            for j in range(0, regImg.shape[1]):

                curFieldRegion = regImg[i,j]

                otherRegions = []  
                #right

                if i+1 < regImg.shape[0]:
                    otherRegions.append(regImg[i+1, j])
                #bottom

                if j+1 < regImg.shape[1]:
                    otherRegions.append(regImg[i, j+1])

                if diagonal and i+1 < regImg.shape[0] and j+1 < regImg.shape[1]:
                    #diagonal
                    otherRegions.append(regImg[i+1, j+1])

                for oRegion in otherRegions:
                    adjacencyCounter[ (mapping[curFieldRegion], mapping[oRegion]) ] += 1
                    adjacencyCounter[ (mapping[oRegion],mapping[curFieldRegion]) ] += 1

        for interaction in adjacencyCounter:
            self.add_relation(interaction[0], interaction[1], weight=1)        


    def add_relation(self, src, tgt, weight=1.0):

        self.relations.add_edge(src, tgt, weight=weight)

        if self.bi_directional:
            self.relations.add_edge(tgt, src, weight=weight)

    
    def get_score(self, src, tgt):

        if (src, tgt) in self.relations.edges:
            return self.relations.edges[(src, tgt)]["weight"]

        return self.no_relation_weight

    def plot_model(self):

        plt.figure()

        labels = {n: "{} ({})".format(n, self.relations.nodes[n]['weight']) for n in self.relations.nodes}
        colors = [self.relations.nodes[n]['weight'] for n in self.relations.nodes]

        edgeColors = [self.relations.edges[n]['weight'] for n in self.relations.edges]

        nx.draw(self.relations, with_labels=True, labels=labels, node_color=colors, edge_colors=edgeColors)
        plt.show()
        plt.close()


class RegionEmbedding(metaclass=abc.ABCMeta):

    def __init__(self, region:SpectraRegion) -> None:
        self.region = region
        self.embedded_matrix = None

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'fit_transform') and callable(subclass.fit_transform) and
                hasattr(subclass, 'embedding') and callable(subclass.embedding) and
                hasattr(subclass, 'region')
                )

    def methodname(self):
        """Brief description of the specific clusterer

        """
        return self.__class__.__name__

    @abc.abstractmethod
    def embedding(self) -> np.array:
        """Returns the final embedding for given region

        Raises:
            NotImplementedError: [description]

        Returns:
            np.array: embedding
        """

        raise NotImplementedError

    @abc.abstractmethod
    def fit_transform(self, verbose:bool=False) -> np.array:
        """
        Returns the final embedding

        Args:
            num_target_clusters (int): number of target clusters
            verbose (bool, optional): Verbose output. Defaults to False.


        Raises:
            NotImplementedError: (abstract class)

        Returns:
            np.array: segmentation
        """
        raise NotImplementedError

class PCAEmbedding(RegionEmbedding):

    def __init__(self, region: SpectraRegion, dimensions: int=2) -> None:
        super().__init__(region)

        self.dimensions = dimensions
        self.idx2coord = None
        self.embedding_object = None

    def fit_transform(self, verbose: bool = False) -> np.array:
        
        elem_matrix, self.idx2coord = self.region.prepare_elem_matrix()
        #np-array dims  (n_samples, n_features)

        self.logger.info("PCA reduction")
        self.embedding_object = decomposition.PCA(
            n_components=self.dimensions,
            random_state=42,
        )

        self.logger.info("PCA fit+transform")
        self.embedded_matrix = self.embedding_object.fit_transform(elem_matrix)

    def embedding(self) -> np.array:
        
        outArray = np.zeros((self.region.region_array.shape[0], self.region.region_array.shape[1], self.dimensions))

        for idx in self.idx2coord:
            (x,y) = self.idx2coord[idx]
            outArray[x,y,:] = self.embedded_matrix[idx]
        
        return outArray

    def covariances(self):
        return self.embedding_object.get_covariance()

    def loading(self):

        # according to https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html
        computedPCs = ["PC{}".format(x) for x in range(1, self.dimensions+1)] # 1-based
        loadings = pd.DataFrame(self.embedding_object.components_.T, columns=computedPCs, index=self.region.idx2mass)

        return loadings



class UMAPEmbedding(RegionEmbedding):

    def __init__(self, region: SpectraRegion, dimensions: int=2) -> None:
        super().__init__(region)

        self.dimensions = dimensions
        self.idx2coord = None
        self.embedding_object = None

    def fit_transform(self, verbose: bool = False, densmap: bool=False, n_neighbours: int=10, min_dist: float=0) -> np.array:
        
        elem_matrix, self.idx2coord = self.region.prepare_elem_matrix()
        #np-array dims  (n_samples, n_features)

        self.logger.info("UMAP reduction")
        self.embedding_object = umap.UMAP(
            densmap=densmap,
            n_neighbors=n_neighbours,
            min_dist=min_dist,
            n_components=self.dimensions,
            random_state=42,
        )

        self.embedded_matrix = self.embedding_object.fit_transform(elem_matrix)

    def embedding(self) -> np.array:
        
        outArray = np.zeros((self.region.region_array.shape[0], self.region.region_array.shape[1], self.dimensions))

        for idx in self.idx2coord:
            (x,y) = self.idx2coord[idx]
            outArray[x,y,:] = self.embedded_matrix[idx]
        
        return outArray




class UMAP_DBSCAN_Clusterer(RegionClusterer):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

        self.matrix_mz = np.copy(self.region.idx2mass)
        self.segmented = None

    def fit(self, num_target_clusters: int, max_iterations: int = 100, verbose: bool = False):
        
        elem_matrix, idx2ij = self.region.prepare_elem_matrix()
        kmeans = cluster.KMeans(n_clusters=num_target_clusters, random_state=0).fit(elem_matrix)

        if hasattr(kmeans, 'labels_'):
            y_pred = kmeans.labels_.astype(int)
        else:
            y_pred = kmeans.predict(elem_matrix)

        clusts = np.zeros((self.region.region_array.shape[0], self.region.region_array.shape[1]))

        for idx, ypred in enumerate(y_pred):
            clusts[idx2ij[idx]] = y_pred[idx]

        self.segmented = clusts

    def transform(self, num_target_clusters: int, verbose: bool = False) -> np.array:
        return self.segmented

    def segmentation(self) -> np.array:
        return self.segmented




class FuzzyCMeansClusterer(RegionClusterer):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

        self.matrix_mz = np.copy(self.region.idx2mass)
        self.segmented = None

    def fit(self, num_target_clusters: int, max_iterations: int = 100, verbose: bool = False):
        
        elem_matrix, idx2ij = self.region.prepare_elem_matrix()
        fcm = FCM(n_clusters=num_target_clusters).fit(elem_matrix)
        fcm_labels = fcm.predict(elem_matrix)
        
        y_pred = fcm_labels

        clusts = np.zeros((self.region.region_array.shape[0], self.region.region_array.shape[1]))

        for idx, ypred in enumerate(y_pred):
            clusts[idx2ij[idx]] = y_pred[idx]

        self.segmented = clusts

    def transform(self, num_target_clusters: int, verbose: bool = False) -> np.array:
        return self.segmented

    def segmentation(self) -> np.array:
        return self.segmented

        
class KMeansClusterer(RegionClusterer):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

        self.matrix_mz = np.copy(self.region.idx2mass)
        self.segmented = None

    def fit(self, num_target_clusters: int, max_iterations: int = 100, verbose: bool = False):
        
        elem_matrix, idx2ij = self.region.prepare_elem_matrix()
        kmeans = cluster.KMeans(n_clusters=num_target_clusters, random_state=0).fit(elem_matrix)

        if hasattr(kmeans, 'labels_'):
            y_pred = kmeans.labels_.astype(int)
        else:
            y_pred = kmeans.predict(elem_matrix)

        clusts = np.zeros((self.region.region_array.shape[0], self.region.region_array.shape[1]))

        for idx, ypred in enumerate(y_pred):
            clusts[idx2ij[idx]] = y_pred[idx]

        self.segmented = clusts

    def transform(self, num_target_clusters: int, verbose: bool = False) -> np.array:
        return self.segmented

    def segmentation(self) -> np.array:
        return self.segmented

class ShrunkenCentroidClusterer(RegionClusterer):

    def __init__(self, region: SpectraRegion, delta=0.2) -> None:
        super().__init__(region)

        self.delta = delta
        self.results = None

        self.matrix_mz = np.copy(self.region.idx2mass)

    def _get_overall_centroid(self, spectra):
        #Calculate the overall centroid
        n = spectra.shape[0]*spectra.shape[1]
        return np.sum(spectra, axis=(0,1))/n

    def _get_segments(self, image):
        cluster2coords = defaultdict(list)
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):

                clusterID = int(image[i, j])
                cluster2coords[clusterID].append((i,j))

        return cluster2coords

    def _get_seg_centroids(self, segments, spectra_orig):
        #Calculate the segment centroids
        seg_cent = defaultdict(lambda : np.zeros( (spectra_orig.shape[2],)))
        for s in sorted(segments):

            allCoordIntensities = []
            
            for coord in segments[s]:
                #allCoordIntensities.append(spectra_orig[coord])
                seg_cent[s] += spectra_orig[coord]

            n = len(segments[s])
            seg_cent[s] = seg_cent[s] / n
            
        return seg_cent

    def _get_all_s_vec(self, segments, spectra_orig, seg_centroids, verbose=False):
        n = spectra_orig.shape[0]*spectra_orig.shape[1]
        K = len(segments.keys())
        seenCoords = 0

        curS = np.zeros((spectra_orig.shape[2],))
        
        for seg in segments:
            seg_centroid = seg_centroids[seg]
            coordinates = segments[seg]
            for coord in coordinates:
                seenCoords += 1
                curS += np.multiply(spectra_orig[coord]-seg_centroid, spectra_orig[coord]-seg_centroid)
                    
        curS = (1.0/(n-K)) * curS
        curS = np.sqrt(curS)

        if verbose:
            print(seenCoords, "of",n )

        return curS


    def _get_shr_centroids(self, segments, spectra_orig, seg_centroids, overall_centroid, verbose=False):
        #Calculate the segment shrunken centroids
        seg_shr_cent = dict()
        seg_tstat_cent = dict()
        n = spectra_orig.shape[0]*spectra_orig.shape[1]
        K = np.max(list(segments.keys()))
        s_list = self._get_all_s_vec(segments, spectra_orig, seg_centroids, verbose=verbose)
        s_0 = np.median(s_list)



        if verbose:
            ovrAllCentroid = np.copy(overall_centroid)
            ovrAllCentroid[ovrAllCentroid<=0] = 0
            ovrAllCentroid[ovrAllCentroid>0] = 1

            print("Selected fields OvrAll Centroid:", sum(ovrAllCentroid), "of", len(ovrAllCentroid))

            
        for seg in sorted(segments):
            seg_centroid = seg_centroids[seg]
            coordinates = segments[seg]

            if verbose:
                print("seg centroid", seg_centroid)

            m = math.sqrt((1/len(coordinates)) + (1/n))
            shr_centroid = np.zeros(seg_centroid.shape)
            tstat_centroid = np.zeros(seg_centroid.shape)



            if verbose:
                segCentroid = np.copy(seg_centroids[seg])
                segCentroid[segCentroid <= 0] = 0
                segCentroid[segCentroid > 0] = 1
                print("Selected fields Seg Centroids", seg, ":", sum(segCentroid), "of", len(segCentroid), "with s0=", s_0, "and m=", m)


            for mz in range(spectra_orig.shape[2]):

                d_ik   = (seg_centroid[mz] - overall_centroid[mz])/(m*(s_list[mz]+s_0))
                dp_ik  = np.sign(d_ik)*max(0, (abs(d_ik)-self.delta)) #where + means positive part (t+ = t if t  0 and zero otherwise)
                #only d_ik > delta will result in change!

                tstat_centroid[mz] = dp_ik
                #shr_centroid[mz] = seg_centroid[mz] + m*(s_list[mz]+s_0)*dp_ik  was used, but checking literature it should be
                shr_centroid[mz] = overall_centroid[mz] + m*(s_list[mz]+s_0)*dp_ik 
                

                if shr_centroid[mz] < 0:
                    pass
                    # it's a centroid and therefore a possible element of class spectrum!
                    #shr_centroid[mz] = 0

                #if shr_centroid[mz] < 0 and seg_centroid[mz] > 0:
                #    print(seg, mz, seg_centroid[mz], d_ik, dp_ik, shr_centroid[mz])
                
            shr_centroid = np.nan_to_num(shr_centroid, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
            seg_shr_cent[seg] = shr_centroid
            seg_tstat_cent[seg] = tstat_centroid


        allShrCentroid = np.zeros((spectra_orig.shape[2],))
        for seg in sorted(seg_shr_cent):
            allShrCentroid += seg_shr_cent[seg]

            if verbose:
                shrCentroid = np.copy(seg_shr_cent[seg])
                shrCentroid[shrCentroid <= 0] = 0
                shrCentroid[shrCentroid > 0] = 1

                print("Selected fields Shr Centroids", seg, ":", sum(shrCentroid), "of", len(shrCentroid))

                fiveNumSummary = (
                                    np.min(seg_tstat_cent[seg]),
                                    np.quantile(seg_tstat_cent[seg], 0.25),
                                    np.median(seg_tstat_cent[seg]),
                                    np.quantile(seg_tstat_cent[seg], 0.75),
                                    np.max(seg_tstat_cent[seg]),
                                    np.mean(seg_tstat_cent[seg])
                                    )
                print("t stats:", fiveNumSummary)



        if verbose:
            allShrCentroid[allShrCentroid <= 0] = 0
            allShrCentroid[allShrCentroid > 0] = 1
            print("Selected fields over all Shr Centroids", sum(allShrCentroid), "of", len(allShrCentroid))

        return seg_shr_cent, seg_tstat_cent


    def _plot_segment_centroids(matrix_shr_centroids, matrix_global_centroid, matrix_segments, matrix, matrix_mz, ylim, addSpecs=[], xlim=(-500, 1000)):
        oldFigSize = plt.rcParams["figure.figsize"]
        plt.rcParams["figure.figsize"] = (20,30)

        fig, ax = plt.subplots(1, len(matrix_shr_centroids)+1+ len(addSpecs))
        for sidx, seg in enumerate(sorted(matrix_shr_centroids)):

            usePixels = matrix_segments[seg]
            if len(usePixels) > 200:
                usePixels = random.sample(list(matrix_segments[seg]), 200)

            for px in usePixels:
                ax[sidx].plot(matrix[px], matrix_mz, alpha=0.01, color="blue")

            ax[sidx].plot(matrix_shr_centroids[seg], matrix_mz, color="black")
            ax[sidx].set_title("Segment: {}".format(seg))
            ax[sidx].set_xlim(xlim)
            ax[sidx].set_ylim(ylim)

        ax[len(matrix_shr_centroids)].plot(matrix_global_centroid,matrix_mz, color="black")
        ax[len(matrix_shr_centroids)].set_title("Segment: {}".format("global"))
        ax[len(matrix_shr_centroids)].set_xlim(xlim)
        ax[len(matrix_shr_centroids)].set_ylim(ylim)

        for asi, addSpec in enumerate(addSpecs):

            ax[len(matrix_shr_centroids)+1+asi].plot(matrix[addSpec],matrix_mz, color="black")
            ax[len(matrix_shr_centroids)+1+asi].set_title("Segment: {}".format(addSpec))
            ax[len(matrix_shr_centroids)+1+asi].set_xlim(xlim)
            ax[len(matrix_shr_centroids)+1+asi].set_ylim(ylim)

        plt.show()
        plt.close()

        plt.rcParams["figure.figsize"] = oldFigSize

    def _plotTStatistics( self, tStat, mzValues, plotRange=None ):

        plt.figure()

        for x in tStat:
            plt.plot(mzValues, tStat[x], label=str(x))

        if not plotRange is None:
            plt.xlim(plotRange)

        plt.legend()
        plt.show()
        plt.close()

    def plot_segment_centroid(self, iteration=-1, mz_range=(200,620), intensity_range=(-1,5)):
        resultDict = self._get_iteration_data(iteration)
                
        mzValues = np.copy(self.region.region_array)       
        self._plot_segment_centroids(resultDict["centroids"], resultDict["global_centroid"], resultDict["segmentation"], mzValues, self.matrix_mz, mz_range, xlim=intensity_range)

    def plot_t_statistics(self, iteration=-1, mz_range=None):
        
        resultDict = self._get_iteration_data(iteration)
        mzValues = np.copy(self.region.region_array) 
        self._plotTStatistics(resultDict["centroids_tstats"], mzValues, mz_range)


    def _distance_tibschirani(self, matrix, pxCoord, centroid, sqSStats, centroidProbability):
        specDiff = matrix[pxCoord]-centroid
        sigma = np.divide(np.multiply(specDiff, specDiff), sqSStats)
        sigma = np.nan_to_num(sigma, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

        sigma = sum(sigma)
        sigma = sigma - 2*math.log(centroidProbability)

        return sigma

    def _get_new_clusters_func(self, orig_segments, segments, spectra_orig, seg_centroids, shr_seg_centroids, print_area=5, distance_func=None, verbose=False):   

        assert(not distance_func is None)

        #Calculate the segment membership probability
        new_matrix = np.zeros((spectra_orig.shape[0], spectra_orig.shape[1]))
        n = spectra_orig.shape[0] * spectra_orig.shape[1]
        s_list = self._get_all_s_vec(segments, spectra_orig, seg_centroids)
        s_0 = np.median(s_list)
        sigmas = list()

        sSum = s_list+s_0
        sSumSq = np.multiply(sSum, sSum)
        oldSegmentCount = 0

        allMaxSigmas = []
        takenClusters = []

        printXlow = (orig_segments.shape[0]/2) - print_area
        printXhi =  (orig_segments.shape[0]/2) + print_area

        printYlow = (orig_segments.shape[1]/2)-print_area
        printYhi =  (orig_segments.shape[1]/2)+print_area

        allShrCentroid = np.zeros((spectra_orig.shape[2],))
        for seg in sorted(shr_seg_centroids):
            allShrCentroid += shr_seg_centroids[seg]

        allShrCentroid[allShrCentroid <= 0] = 0
        allShrCentroid[allShrCentroid > 0] = 1

        if verbose:
            print("Total field considered", sum(allShrCentroid), "of", len(allShrCentroid))
            for seg in sorted(segments):
                print("Segment", seg, "elements", len(segments[seg]), "of all", n, len(segments[seg])/n)
                    
        for i in range(spectra_orig.shape[0]):
            for j in range(spectra_orig.shape[1]):
                spectrum = spectra_orig[(i,j)]
                sigmas = dict()

                for seg in sorted(segments):
                    shr_seg_centroid = shr_seg_centroids[seg]
                    coordinates = segments[seg]
                    
                    sigma = distance_func(spectra_orig, (i,j), shr_seg_centroid, sSumSq, len(coordinates)/n)
                    
                    sigmas[seg] = sigma

                allMaxSigmas += [sigmas[seg] for seg in segments]
    
                #this very likely becomes 0 and SHOULD NOT be used for class assignment!
                #summed_probabilities = sum([math.exp(-0.5*sigmas[cluster]) for cluster in sorted(sigmas)])

                if verbose:
                    if (printXlow<=i<=printXhi and printYlow<=j<=printYhi) or (i,j) in [(22,26)]:# or lower_probability == 0:
                        for seg in sorted(sigmas):
                            print("[PS]", i,j, seg, sigmas[seg], math.exp(-0.5*sigmas[seg]),2*math.log(len(segments[seg])/n))
                        #print([(cluster,math.exp(-0.5*sigmas[cluster])/summed_probabilities) for cluster in sorted(sigmas)], lower_probability)

                minSigma = min(sigmas.values())
                if len(sigmas) == 0:
                    print("sigmas is empty!", sigmas, (i,j))


                minSigmaClass = [x for x in sigmas if sigmas[x] == minSigma]

                if len(minSigmaClass) == 0:
                    print("minSigmaClass Empty", i, j)
                    print(sigmas)
                    print(minSigma)

                minSigmaClass = minSigmaClass[0]

                new_matrix[i][j] = minSigmaClass
                takenClusters.append(minSigmaClass)

        if verbose:
            plt.hist(takenClusters, bins=len(set(takenClusters)))
            plt.show()
            plt.close()

        if verbose:
            print("Old segments taken:", oldSegmentCount, "of", spectra_orig.shape[0]*spectra_orig.shape[1] )

        return new_matrix, allMaxSigmas

    def fit(self, num_target_clusters: int, max_iterations: int = 100, verbose: bool = False):
        
        matrix = np.copy(self.region.region_array)
    
        shr_segmented = KMeansClusterer(self.region).fit_transform(num_target_clusters=num_target_clusters, verbose=verbose)

        iteration = 0
        self.results = OrderedDict()
        self.results[iteration] =  {'segmentation': shr_segmented, 'centroids': None, 'centroids_tstats': None, 'global_centroid': None}

        progBar = progressbar.ProgressBar(widgets=[
        progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.AdaptiveETA()
        ])

        for iteration in progBar(range(1, max_iterations+1)):
            #iteration += 1

            matrix_global_centroid = self._get_overall_centroid(matrix)
            matrix_segments = self._get_segments(shr_segmented)

            matrix_seg_centroids = self._get_seg_centroids(matrix_segments, matrix)
            
            matrix_shr_centroids, matrix_tstat_centroids = self._get_shr_centroids(matrix_segments, matrix, matrix_seg_centroids, matrix_global_centroid)

            shr_segmented, ams = self._call_new_clusters(shr_segmented, matrix_segments, matrix, matrix_seg_centroids, matrix_shr_centroids)

            if verbose:
                for x in sorted(matrix_segments):
                    print("SegLen2", x, len(matrix_segments[x]))

            self.results[iteration] = {'segmentation': shr_segmented, 'centroids': matrix_shr_centroids, 'centroids_tstats': matrix_tstat_centroids, 'global_centroid': matrix_global_centroid}

            if self._matrixEqual(self.results[iteration]['segmentation'], self.results[iteration-1]['segmentation']):
                print("Finishing iterations due to same result after", iteration, "iterations")
                break

    def _call_new_clusters(self, shr_segmented, matrix_segments, matrix, matrix_seg_centroids, matrix_shr_centroids):
        shr_segmented, ams = self._get_new_clusters_func(shr_segmented, matrix_segments, matrix, matrix_seg_centroids, matrix_shr_centroids, print_area=0, distance_func=self._distance_tibschirani)
        return shr_segmented, ams


    def _matrixEqual(self, mat1, mat2):
        comparison = mat1 == mat2
        equal_arrays = comparison.all()
        return equal_arrays

    def _get_iteration_data(self, iteration):
        resultKeys = list(self.results.keys())
        desiredKey = resultKeys[iteration]
        resultDict = self.results[desiredKey]
        return resultDict

    def segmentation(self) -> np.array:
        resDict = self._get_iteration_data(-1)
        return resDict["segmentation"]

    def transform(self, num_target_clusters: int, verbose: bool = False) -> np.array:      
        if verbose:
            print("Warning: num_target_clusters not applicable to this method")
        segResult = self.segmentation()
        return segResult
        
class SARegionClusterer(ShrunkenCentroidClusterer):

    def __init__(self, region: SpectraRegion, delta=0.2, radius=2) -> None:
        super().__init__(region, delta=delta)

        self.radius = radius

    def _distance_sa(self, matrix, pxCoord, centroid, sqSStats, centroidProbability, radius=2):

        distance = 0
        sigma = (2*radius+1)/4

        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                neighbor = (pxCoord[0]+i, pxCoord[1]+j)

                if neighbor[0] < 0 or neighbor[1] < 0:
                    #invalid coord
                    # TODO implement some working padding!
                    continue
                if neighbor[0] >= matrix.shape[0] or neighbor[1] >= matrix.shape[1]:
                    #invalid coord
                    # TODO implement some working padding!
                    continue

                weight = math.exp(-i**2-j**2)/(2*sigma**2)
                specDiff = np.linalg.norm(matrix[neighbor]-centroid) ** 2

                distance += weight * specDiff    

        return np.sqrt(distance)


    def _call_new_clusters(self, shr_segmented, matrix_segments, matrix, matrix_seg_centroids, matrix_shr_centroids):
        shr_segmented, ams = self._get_new_clusters_func(shr_segmented, matrix_segments, matrix, matrix_seg_centroids, matrix_shr_centroids, print_area=0, distance_func=lambda matrix, pxCoord, centroid, sqSStats, centroidProbability: self._distance_sa(matrix, pxCoord, 
                centroid, sqSStats, centroidProbability, radius=self.radius))

        return shr_segmented, ams


class SASARegionClusterer(ShrunkenCentroidClusterer):

    def __init__(self, region: SpectraRegion, delta=0.2, radius=2) -> None:
        super().__init__(region, delta=delta)

        self.radius = radius


    def _spectra_dist_sasa(self, spec1, spec2):
        return np.linalg.norm(spec1-spec2)

    def _distance_sasa_beta(self, xpos, npos, matrix, radius, lambda_):

        postFactor = (self._spectra_dist_sasa(matrix[npos], matrix[xpos]) / lambda_) ** 2

        return 1.0/math.exp(0.25 * postFactor)


    def _distance_sasa_alpha(self, matrix, xpos, npos, dpos, radius, lambda_):

        sigma = (2*radius+1)/4.0

        alpha_pre_top = (dpos[0]**2)-(dpos[1]**2)
        alpha_pre_bottom = (2*(sigma**2))

        if alpha_pre_top > 0:
            alpha_pre = math.exp( alpha_pre_top / alpha_pre_bottom )
        else:
            alpha_pre = 1.0 / math.exp( abs(alpha_pre_top) / alpha_pre_bottom )

        alpha_post = self._distance_sasa_beta(xpos, npos, matrix, radius, lambda_)

        return alpha_pre * alpha_post



    def _distance_sasa(self, matrix, pxCoord, centroid, sqSStats, centroidProbability, radius=2):

        allDeltas = []
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                neighbor = (pxCoord[0]+i, pxCoord[1]+j)

                if neighbor[0] < 0 or neighbor[1] < 0:
                    #invalid coord
                    # TODO implement some working padding!
                    continue
                if neighbor[0] >= matrix.shape[0] or neighbor[1] >= matrix.shape[1]:
                    #invalid coord
                    # TODO implement some working padding!
                    continue

                delta_ij_xy = self._spectra_dist_sasa(matrix[neighbor], matrix[pxCoord]) # 2-norm
                allDeltas.append(delta_ij_xy)



        minDelta = np.min(allDeltas)
        allDeltaHats = [x-minDelta for x in allDeltas]
        lambda_ = 0.5 * np.max(allDeltaHats)

        if lambda_ == 0:
            lambda_ = 1

        distance = 0

        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                neighbor = (pxCoord[0]+i, pxCoord[1]+j)
                dpos = (i,j)

                if neighbor[0] < 0 or neighbor[1] < 0:
                    #invalid coord
                    # TODO implement some working padding!
                    continue
                if neighbor[0] >= matrix.shape[0] or neighbor[1] >= matrix.shape[1]:
                    #invalid coord
                    # TODO implement some working padding!
                    continue


                specDiff = np.linalg.norm(matrix[neighbor]-centroid) ** 2 # 2-norm squared
                alpha_ij = self._distance_sasa_alpha(matrix, pxCoord, neighbor, dpos, radius, lambda_)

                distance += alpha_ij * specDiff    

        return np.sqrt(distance)


    def _call_new_clusters(self, shr_segmented, matrix_segments, matrix, matrix_seg_centroids, matrix_shr_centroids):
        shr_segmented, ams = self._get_new_clusters_func(shr_segmented, matrix_segments, matrix, matrix_seg_centroids, matrix_shr_centroids, print_area=0, distance_func=lambda matrix, pxCoord, centroid, sqSStats, centroidProbability: self._distance_sasa(matrix, pxCoord, 
                centroid, sqSStats, centroidProbability, radius=self.radius))

        return shr_segmented, ams

