# general
import logging
import json
import os
import random
import math
from collections import defaultdict, Counter
import glob
import shutil, io, base64
from collections import OrderedDict

# general package
from natsort import natsorted
from numpy.lib.recfunctions import _repack_fields_dispatcher
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
#from fcmeans import FCM

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
        self.logger = None
        self.__set_logger()

    def __set_logger(self):
        self.logger = logging.getLogger(self.methodname())
        self.logger.setLevel(logging.INFO)

        if not self.logger.hasHandlers():

            consoleHandler = logging.StreamHandler()
            consoleHandler.setLevel(logging.INFO)

            self.logger.addHandler(consoleHandler)

            formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
            consoleHandler.setFormatter(formatter)

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

    def explained_variance_ratio(self):
        return self.embedding_object.explained_variance_ratio_

    def plot_embedding(self):

        dimExplained = self.embedding_object.explained_variance_ratio_

        reductionName = "PCA"

        plt.figure(figsize=(12, 12))


        plt.scatter(self.embedded_matrix[:, 0], self.embedded_matrix[:, 1])

        plt.xlabel("{} dim1 ({:.2})".format(reductionName, dimExplained[0]))
        plt.ylabel("{} dim2 ({:.2})".format(reductionName, dimExplained[1]))

        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left", mode="expand", ncol=2)

        plt.show()
        plt.close()

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


class UMAP_WARD_Clusterer(RegionClusterer):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

        self.umapEmbedding = UMAPEmbedding(region=region, dimensions=2)
        self.pwdist = None
        self.dimred_labels = None
        self.segmented = None

    def fit(self, num_target_clusters: int, densmap: bool = False, n_neighbours: int = 10, min_dist: float = 0, verbose: bool = False):
        """Performs UMAP dimension reduction on region array followed by Euclidean pairwise distance calculation in order to do Ward's linkage.

        Args:
            num_target_clusters (int): Number of desired clusters.
            densmap (bool, optional): Whether to use densMAP (density-preserving visualization tool based on UMAP). Defaults to False. To use densMAP please use UMAP_WARD_Clusterer instead.
            n_neighbours (int, optional): The size of the local neighborhood (in terms of number of neighboring sample points) used for manifold approximation. For more information check UMAP documentation. Defaults to 10.
            min_dist (float, optional): The min_dist parameter controls how tightly UMAP is allowed to pack points together. For more information check UMAP documentation. Defaults to 0.
            verbose (bool, optional): Defaults to False.
        """
        self.umapEmbedding.fit_transform(verbose=verbose, densmap=densmap, n_neighbours=n_neighbours, min_dist=min_dist)
        dimred_elem_matrix = self.umapEmbedding.embedded_matrix

        self.pwdist = pdist(dimred_elem_matrix, metric='euclidean')

        _ = self.transform(num_target_clusters=num_target_clusters)


    def _update_segmented(self):

        image = np.zeros(self.region.region_array.shape, dtype=np.int16)
        image = image[:,:,0]

        # cluster 0 has special meaning: not assigned !
        assert(not 0 in [self.dimred_labels[x] for x in self.dimred_labels])

        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                image[i,j] = self.dimred_labels[self.region.pixel2idx[(i,j)]]

        self.segmented = image

    def transform(self, num_target_clusters: int, verbose: bool = False) -> np.array:
        """Allows to redo the WARD's clustering using the reduced data during fit operation.

        Args:
            num_target_clusters (int): Number of desired clusters.
            verbose (bool, optional): Defaults to False.

        Returns:
            np.array: Segmented array.
        """
        Z = spc.hierarchy.ward(self.pwdist)
        self.dimred_labels = spc.hierarchy.fcluster(Z, t=num_target_clusters, criterion='maxclust')
        self._update_segmented()
        return self.segmented

    def segmentation(self) -> np.array:
        return self.segmented

class DENSMAP_WARD_Clusterer(UMAP_WARD_Clusterer):
    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

    def fit(self, num_target_clusters: int, n_neighbours: int = 10, min_dist: float = 0, verbose: bool = True):
        """Uses densMAP (density-preserving visualization tool based on UMAP) dimension reduction on region array followed by Euclidean pairwise distance calculation in order to do Ward's linkage.

        Args:
            num_target_clusters (int): Number of desired clusters.
            n_neighbours (int, optional): The size of the local neighborhood (in terms of number of neighboring sample points) used for manifold approximation. For more information check UMAP documentation. Defaults to 10.
            min_dist (float, optional): The min_dist parameter controls how tightly UMAP is allowed to pack points together. For more information check UMAP documentation. Defaults to 0.
            verbose (bool, optional): Defaults to False.
        """
        return super().fit(num_target_clusters=num_target_clusters, densmap=True, n_neighbours=n_neighbours, min_dist=min_dist, verbose=verbose)

    def transform(self, num_target_clusters: int, verbose: bool = False) -> np.array:
        """Allows to redo the WARD's clustering using the reduced data during fit operation.

        Args:
            num_target_clusters (int): Number of desired clusters.
            verbose (bool, optional): Defaults to False.

        Returns:
            np.array: Segmented array.
        """
        return super().transform(num_target_clusters, verbose)

    def segmentation(self) -> np.array:
        return super().segmentation()

class UMAP_DBSCAN_Clusterer(RegionClusterer):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

        self.umapEmbedding = UMAPEmbedding(region=region, dimensions=2)
        self.dimred_labels = None
        self.dimred_elem_matrix = None
        self.segmented = None

    def fit(self, num_target_clusters: int, verbose: bool = False, densmap: bool=False, n_neighbours: int=10, min_dist: float=0, min_cluster_size: int=15, num_samples: int=10000) -> np.array:
        """Performs UMAP dimension reduction on region array followed by the HDBSCAN clustering.

        Args:
            num_target_clusters (int): Number of desired clusters.
            verbose (bool, optional): Defaults to False.
            densmap (bool, optional): Whether to use densMAP (density-preserving visualization tool based on UMAP). Defaults to False. If you want to apply densMAP please use DENSMAP_DBSCAN_Clusterer instead.
            n_neighbours (int, optional): The size of the local neighborhood (in terms of number of neighboring sample points) used for manifold approximation. For more information check UMAP documentation. Defaults to 10.
            min_dist (float, optional): The min_dist parameter controls how tightly UMAP is allowed to pack points together. For more information check UMAP documentation. Defaults to 0.
            min_cluster_size (int, optional): The minimum size of HDBSCAN clusters. Defaults to 15.
            num_samples (int, optional): Number of intensity values that will be used during HDBSCAN clustering. Defaults to 10000.
        """
        self.umapEmbedding.fit_transform(verbose=verbose, densmap=densmap, n_neighbours=n_neighbours, min_dist=min_dist)
        self.dimred_elem_matrix = self.umapEmbedding.embedded_matrix

        _ = self.transform(num_target_clusters=num_target_clusters, min_cluster_size=min_cluster_size, num_samples=num_samples)

    def transform(self, num_target_clusters: int, min_cluster_size: int = 15, num_samples: int = 10000, verbose: bool = False) -> np.array:
        """Performs HDBSCAN clustering (Hierarchical Density-Based Spatial Clustering of Applications with Noise) on the previously reduced data.

        Args:
            num_target_clusters (int): Number of desired clusters.
            min_cluster_size (int, optional): The minimum size of HDBSCAN clusters. Defaults to 15.
            num_samples (int, optional): Number of intensity values that will be used during HDBSCAN clustering. Defaults to 10000.
            verbose (bool, optional): Defaults to False.

        Returns:
            np.array: Segmented array.
        """
        self.logger.info("HDBSCAN reduction")
        if num_samples > self.dimred_elem_matrix.shape[0]:
            num_samples = self.dimred_elem_matrix.shape[0]
            self.logger.info("HDBSCAN reduction num_samples reset: {}".format(num_samples))

        if num_samples == -1 or self.dimred_elem_matrix.shape[0] < num_samples:
            selIndices = [x for x in range(0, self.dimred_elem_matrix.shape[0])]
        else:
            selIndices = random.sample([x for x in range(0, self.dimred_elem_matrix.shape[0])], num_samples)

        dr_matrix = self.dimred_elem_matrix[selIndices, :]

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True).fit(dr_matrix)
        clusterer.generate_prediction_data()
        soft_clusters = hdbscan.prediction.membership_vector(clusterer, self.dimred_elem_matrix)

        self.dimred_labels = np.array([np.argmax(x) for x in soft_clusters])+1 # +1 avoids 0

        if len(np.unique(self.dimred_labels)) > num_target_clusters:
            self.segmented = np.reshape(self.dimred_labels, (self.region.region_array.shape[0], self.region.region_array.shape[1]))

            self._reduce_clusters(num_target_clusters)
            self.dimred_labels = np.reshape(self.segmented, (self.region.region_array.shape[0] * self.region.region_array.shape[1],))

        self.dimred_labels = list(self.dimred_labels)

        self._update_segmented()
        return self.segmented

    def _reduce_clusters(self, number_of_clusters):
        """Reducing the number of clusters in segmented array by "reclustering" after the Ward's clustering on pairwise similarity matrix between consensus spectra.

        Args:
            number_of_clusters (int): Number of desired clusters.
        """
        self.logger.info("Cluster Reduction")

        self.region.segmented = self.segmented
        _ = self.region.consensus_spectra()
        self.region.consensus_similarity()

        Z = spc.hierarchy.ward(self.region.consensus_similarity_matrix)
        c = spc.hierarchy.fcluster(Z, t=number_of_clusters, criterion='maxclust')

        dimred_labels = np.reshape(self.segmented, (self.region.region_array.shape[0] * self.region.region_array.shape[1],))
        origlabels = np.array(dimred_labels, copy=True)

        for cidx, cval in enumerate(c):
            dimred_labels[origlabels == (cidx+1)] = cval

        self.segmented = np.reshape(dimred_labels, (self.region.region_array.shape[0], self.region.region_array.shape[1]))

    def _update_segmented(self):
        image= np.zeros(self.region.region_array.shape, dtype=np.int16)
        image = image[:,:,0]

        # cluster 0 has special meaning: not assigned !
        assert(not 0 in [self.dimred_labels[x] for x in self.dimred_labels])

        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                image[i,j] = self.dimred_labels[self.region.pixel2idx[(i,j)]]

        self.segmented = image

    def segmentation(self) -> np.array:
        return self.segmented

class DENSMAP_DBSCAN_Clusterer(UMAP_DBSCAN_Clusterer):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

    def fit(self, num_target_clusters: int, verbose: bool = False, n_neighbours: int = 10, min_dist: float = 0, min_cluster_size: int = 20, num_samples: int = 10000) -> np.array:
        """Uses densMAP (density-preserving visualization tool based on UMAP) followed by the HDBSCAN clustering.

        Args:
            num_target_clusters (int): Number of desired clusters.
            verbose (bool, optional): Defaults to False.
            n_neighbours (int, optional): The size of the local neighborhood (in terms of number of neighboring sample points) used for manifold approximation. For more information check UMAP documentation. Defaults to 10.
            min_dist (float, optional): The min_dist parameter controls how tightly UMAP is allowed to pack points together. For more information check UMAP documentation. Defaults to 0.
            min_cluster_size (int, optional): The minimum size of HDBSCAN clusters. Defaults to 15.
            num_samples (int, optional): Number of intensity values that will be used during HDBSCAN clustering. Defaults to 10000.
        """
        return super().fit(num_target_clusters, verbose=verbose, densmap=True, n_neighbours=n_neighbours, min_dist=min_dist, min_cluster_size=min_cluster_size, num_samples=num_samples)

    def transform(self, num_target_clusters: int, min_cluster_size: int = 15, num_samples: int = 10000, verbose: bool = False) -> np.array:
        """Performs HDBSCAN clustering (Hierarchical Density-Based Spatial Clustering of Applications with Noise) on the previously reduced data.

        Args:
            num_target_clusters (int): Number of desired clusters.
            min_cluster_size (int, optional): The minimum size of HDBSCAN clusters. Defaults to 15.
            num_samples (int, optional): Number of intensity values that will be used during HDBSCAN clustering. Defaults to 10000.
            verbose (bool, optional): Defaults to False.

        Returns:
            np.array: Segmented array.
        """
        return super().transform(num_target_clusters, min_cluster_size, num_samples, verbose)

    def segmentation(self) -> np.array:
        return super().segmentation()
'''
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
'''       
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

class ModifiedKMeansClusterer(RegionClusterer):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

        self.segmented = None

    def fit(self, num_target_clusters: int, max_iterations: int = 100, smoothing_iter: int = 2, verbose: bool = False, init_mode='random_centroids', distance='tibshiran', radius=2):
        """[summary]

        Args:
            num_target_clusters (int): Number of desired clusters.
            max_iterations (int, optional): Number of iterations of k-means algorithm. Defaults to 100.
            smoothing_iter (int, optional): Number of iterations to smooth the clustering by assigning penalties. Defaults to 2.
            verbose (bool, optional): Verbose output. Defaults to False.
            init_mode (str, optional): Type of initialisation stategy. Defaults to 'random_centroids'.\n
                - 'random': each pixel is randomly assigned to a cluster ID, centroids are computed as average of the spectra that belong to the correspondingly cluster ID\n
                - 'random_centroids': each pixel is randomly assigned to a cluster ID, each centroid is a randomly chosen spectrum that belong to the corresponding cluster ID\n
            distance (str, optional): Method to compute distance between spectra. Defaults to 'tibshirani'.\n
                - 'tibshirani': \n
                - 'squared': \n
                - 'sa': distance for spatially-aware clustering\n
                - 'sasa': distance for spatially-aware structurally-adaptive clustering\n
            radius (int, optional): neighborhood radius
        """
        assert(init_mode in ['random', 'kmeans','random_centroids', 'random_2normdist'])
        assert(distance in ['tibshirani', 'squared', 'sa', 'sasa'])

        elem_matrix, _ = self.region.prepare_elem_matrix()
        init_centroids, init_centroids2ids = self._kmeans_init_centroids(elem_matrix=elem_matrix, num_target_clusters=num_target_clusters, mode=init_mode)
        centroids, idx, centroids2ids = self._run_kmeans(elem_matrix=elem_matrix, init_centroids=init_centroids, centroids2ids=init_centroids2ids, max_iter=max_iterations, smoothing_iter=smoothing_iter, distance=distance, radius=radius)
        if centroids is None:
            return self.fit(num_target_clusters, max_iterations, smoothing_iter, verbose, init_mode, distance)
        return elem_matrix, init_centroids, init_centroids2ids, centroids, idx, centroids2ids


    def transform(self, num_target_clusters: int, verbose: bool = False) -> np.array:
        return self.segmented

    def segmentation(self) -> np.array:
        return self.segmented

    def _kmeans_init_centroids(self, elem_matrix, num_target_clusters, mode='random_centroids'):
        if mode=="random":
            random_clustering = np.random.randint(num_target_clusters, size=(elem_matrix.shape[0]))
            return self._compute_centroids(elem_matrix, random_clustering, num_target_clusters)

        elif mode=="kmeans":
            centroids = dict()
            centroids2ids = dict()
            centroid, label = kmeans2(elem_matrix, k=num_target_clusters, iter=100, minit='++')
            for i in range(num_target_clusters):
                elems = np.argwhere(label == i)
                elems = list([e[0] for e in elems])
                centroids2ids[i] = elems
                centroids[i] = centroid[i]
            self._update_segmented(elem_matrix, centroids2ids)
            return centroids, centroids2ids 


        elif mode=="random_centroids":
            # store centroids in Kxn
            centroids = dict()
            centroids2ids = dict()
            random_clustering = np.random.randint(num_target_clusters, size=(elem_matrix.shape[0]))

            for i in range(num_target_clusters):
                elems = np.argwhere(random_clustering == i)
                elems = list([e[0] for e in elems])
                centroids2ids[i] = elems
                centroids[i] = elem_matrix[elems[np.random.randint(len(elems), size=1)[0]]].reshape(elem_matrix.shape[1])
            self._update_segmented(elem_matrix, centroids2ids)
            return centroids, centroids2ids 

        elif mode=='random_2normdist':

            foundCentroidCoords = set()
            centroids2ids = dict()
            while len(foundCentroidCoords) < num_target_clusters:
                xRand = np.random.randint(0, elem_matrix.shape[0]-1)
                
                if not xRand in foundCentroidCoords:
                    foundCentroidCoords.add(xRand)
                    
            foundCentroidCoords = sorted(foundCentroidCoords)
            
            for i in range(0, elem_matrix.shape[0]):
                    
                dist2centroid = []
                for cI, coord in enumerate(foundCentroidCoords):
                    distScalar = np.linalg.norm(np.array(coord)- np.array([i]))
                    dist2centroid.append( (cI, distScalar)  )
                        
                dist2centroid = sorted(dist2centroid, key=lambda x: x[1])
                if dist2centroid[0][0] in centroids2ids:
                    centroids2ids[dist2centroid[0][0]].append(i)
                else:
                    centroids2ids[dist2centroid[0][0]]= [i]
                    
            foundCentroids = defaultdict(list)
            for cI, coord in enumerate(foundCentroidCoords):
                foundCentroids[cI] = elem_matrix[coord]
                
            return foundCentroids, centroids2ids

    def _compute_centroids(self, elem_matrix, idx, num_target_clusters):
        m = elem_matrix.shape[0]
        n = elem_matrix.shape[1]
        
        centroids = dict()
        centroids2ids = dict()
        
        # recalculate each cluster
        for i in range(num_target_clusters):
            centroids[i] = np.zeros((n))
            ids = list()
            
            num_examples = sum(idx == i) # samples per cluster
            old = centroids[i]
                
            # divide by num_examples to get mean for that centroid
            if num_examples == 0:
                centroids[i] = old
            else:
                # add up all samples in the centroid
                for j in range(m):
                    if idx[j] == i:
                        ids.append(j)
                        centroids[i] = centroids[i] + elem_matrix[j, :]
                centroids[i] = centroids[i] / num_examples
            centroids2ids[i] = ids
        self._update_segmented(elem_matrix, centroids2ids)
        return centroids, centroids2ids

    def _update_segmented(self, elem_matrix, centroids2ids):
        original = self.region.region_array
        segmented = np.zeros((elem_matrix.shape[0],))
        for i in centroids2ids.keys():
            segmented[centroids2ids[i]] = i
        self.segmented = segmented.reshape(original.shape[0], original.shape[1])

    def _run_kmeans(self, elem_matrix, init_centroids, centroids2ids, max_iter, smoothing_iter, distance, radius):
        # initialize values
        m = elem_matrix.shape[0]
        K = len(init_centroids)
        centroids = init_centroids
        idx = np.zeros((m,1)).reshape(-1,1)
        bar = makeProgressBar()
        centroids2ids_before = None
        centroids2ids_after = centroids2ids
        switcher = {
            'sa': SARegionClusterer(self.region),
            'sasa': SASARegionClusterer(self.region)
        }
        object = switcher.get(distance, None)

        # run k-means for specified iterations
        for i in bar(range(max_iter)):
            if not (centroids2ids_before == centroids2ids_after):
                centroids2ids_before = centroids2ids_after
                idx = self._find_closest_centroid(object, elem_matrix, centroids, centroids2ids_before, distance=distance, radius=radius) # get closest centroid for each pixel
                if idx is None:
                    return None, None, None
                """
                This is the modification to allow for pseudo-normalized cuts:
                
                - check surrounding assignments
                - if diff, assign penalty
                - change assignment if penalty high enough
                """
                if i == (max_iter - 1):
                    idx = self._modify_idx(idx, smoothing_iter)
                centroids, centroids2ids_after = self._compute_centroids(elem_matrix, idx, K) # calculate new centroids
            else:
                idx = self._modify_idx(idx, smoothing_iter)
                centroids, centroids2ids_after = self._compute_centroids(elem_matrix, idx, K) # calculate new centroids
                self.logger.info('Finished early, iteration '+str(i)+'.')
                break
            
        return centroids, idx, centroids2ids_after

    def _find_closest_centroid(self, object, elem_matrix, curr_centroids, centroids2ids, distance, radius):
        m = elem_matrix.shape[0]
        K = len(curr_centroids.keys())
        idx = np.zeros((m,1)).reshape(-1,1)
        n = elem_matrix.shape[0]

        if distance=='tibshirani':
            #scc = ShrunkenCentroidClusterer(self.region)

            #global sSumSq Tibshirani
            s_list = self._get_all_s_vec(elem_matrix=elem_matrix, centroids=curr_centroids, centroids2ids=centroids2ids)
            s_0 = np.median(s_list)
            sSum = s_list+s_0
            sSumSq = np.multiply(sSum, sSum)

            # go over each sample, find the closest centroid
            for i in range(m):
                dists = np.zeros((K,2)) # store squared distances
                dists[:, 1] = [j for j in range(1,K+1)] # assign K values

                for j in range(K):
                    
                    centroidProbability = len(centroids2ids[j])/n
                    if centroidProbability==0:
                        centroidProbability = 10**(-6)
                    
                    specDiff = elem_matrix[i, :] - curr_centroids[j]
                    sigma = np.divide(np.multiply(specDiff, specDiff), sSumSq)
                    sigma = np.nan_to_num(sigma, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

                    sigma = np.sum(sigma)
                    sigma = sigma - 2*math.log(centroidProbability)
                    dists[j, 0] = sigma
                    
                    #dists[j, 0] = scc._distance_tibschirani(matrix=elem_matrix, pxCoord=i,centroid=curr_centroids[j], sqSStats=sSumSq, centroidProbability=centroidProbability)
                # find closest centroid for each example
                # add centroid index to idx
                centroid_idx = np.where(dists == np.amin(dists[:, 0], axis=0))
                idx[i] = centroid_idx[0][0]


        elif distance=='squared':
            for i in range(m):
                dists = np.zeros((K,2)) 
                dists[:, 1] = [j for j in range(1,K+1)]
                
                for j in range(K):
                    dists[j, 0] = np.sum((elem_matrix[i, :] - curr_centroids[j])**2, axis=0)

                centroid_idx = np.where(dists == np.amin(dists[:, 0], axis=0))
                idx[i] = centroid_idx[0][0]

        elif distance=='sa':
            for coord_x in range(0,self.region.region_array.shape[0]):
                for coord_y in range(0,self.region.region_array.shape[1]):
                    i = coord_x * self.region.region_array.shape[1] + coord_y
                    dists = np.zeros((K,2))
                    dists[:, 1] = [j for j in range(1,K+1)]
                    
                    for j in range(K):
                        dists[j, 0] =  object._distance_sa(matrix=self.region.region_array, pxCoord=(coord_x, coord_y), centroid=curr_centroids[j], sqSStats=None, centroidProbability=None, radius=radius)
                    centroid_idx = np.where(dists == np.amin(dists[:, 0], axis=0))
                    idx[i] = centroid_idx[0][0]


        elif distance=='sasa':
            s_list = self._get_all_s_vec(elem_matrix=elem_matrix, centroids=curr_centroids, centroids2ids=centroids2ids)
            s_0 = np.median(s_list)
            sSum = s_list+s_0
            sSumSq = np.multiply(sSum, sSum)
            for coord_x in range(0,self.region.region_array.shape[0]):
                for coord_y in range(0,self.region.region_array.shape[1]):
                    i = coord_x * self.region.region_array.shape[1] + coord_y
                    dists = np.zeros((K,2))
                    dists[:, 1] = [j for j in range(1,K+1)] 
                    
                    for j in range(K):
                        centroidProbability = len(centroids2ids[j])/n
                        if centroidProbability==0:
                            centroidProbability = 0.01
                        dists[j, 0] =  object._distance_sasa(matrix=self.region.region_array, pxCoord=(coord_x, coord_y), centroid=curr_centroids[j], sqSStats=sSumSq, centroidProbability=centroidProbability, radius=radius)
                    centroid_idx = np.where(dists == np.amin(dists[:, 0], axis=0))
                    idx[i] = centroid_idx[0][0] 
                
        return idx

    def _get_all_s_vec(self, elem_matrix, centroids, centroids2ids, verbose=False):
        n = elem_matrix.shape[0]
        K = len(centroids2ids.keys())
        seenCoords = 0

        curS = np.zeros((elem_matrix.shape[1],))

        for k in centroids2ids:
            seg_centroid = centroids[k]
            coordinates = centroids2ids[k]
            for coord in coordinates:
                seenCoords += 1
                curS += np.multiply(elem_matrix[coord, :]-seg_centroid, elem_matrix[coord, :]-seg_centroid)

        curS = (1.0/(n-K)) * curS
        curS = np.sqrt(curS)

        if verbose:
            print(seenCoords, "of",n )

        return curS

    def _modify_idx(self, old_idx, smoothing_iter):
    
        idx = old_idx.reshape(self.region.region_array.shape[0], self.region.region_array.shape[1])  # reshape to 2D
        
        neighbor_idx = [(-1, -1), # top left
                            (-1, 0), # top
                            (-1, 1), # top right
                            (0, -1), # left
                            (0, 1), # right
                            (1, -1), # bottom left
                            (1, 0), # bottom
                            (1, 1), # bottom right
                        
                            (-2, -2), # begin second layer
                            (-2, -1),
                            (-2, 0),
                            (-2, 1),
                            (-2, 2),
                            (-1, -2),
                            (-1, 2),
                            (0, -2),
                            (0, 2),
                            (1, -2),
                            (1, 2),
                            (2, -2),
                            (2, -1),
                            (2, 0),
                            (2, 1),
                            (2, 2)
                            ]
        
        for m in range(smoothing_iter): # smoothing iterations
        
            # grab each pixel and compare to neighbors
            for i in range(len(idx)):
                for j in range(idx.shape[1]):
                    curr_pix = idx[i, j]

                    # get neighbors
                    immediate_neighbors = []
                    secondary_neighbors = []
                    for k in range(len(neighbor_idx)):

                        # inner layer
                        if k < 8:
                            try:
                                immediate_neighbors.append(idx[i + neighbor_idx[k][0], j + neighbor_idx[k][1]])

                            except:
                                pass

                        # outer neighbors
                        else:
                            try:
                                secondary_neighbors.append(idx[i + neighbor_idx[k][0], j + neighbor_idx[k][1]])

                            except:
                                pass


                    # penalty criteria
                    penalty = 0
                    top_color_immediate = max(set(immediate_neighbors), key=immediate_neighbors.count) # top color in immediate neighbors
                    top_color_secondary = max(set(secondary_neighbors), key=secondary_neighbors.count) # top color in outer neighbors

                    if top_color_immediate != curr_pix and immediate_neighbors.count(top_color_immediate) > 4:
                        penalty += 1

                    elif top_color_immediate != curr_pix and immediate_neighbors.count(top_color_immediate) <= 4:
                        penalty += 0.5

                        if top_color_secondary != curr_pix and secondary_neighbors.count(top_color_secondary) > 8:
                            penalty += 0.5

                    if penalty == 1:
                        idx[i, j] = top_color_immediate # reassign index color

        idx = idx.reshape(old_idx.shape[0], old_idx.shape[1])
        return idx


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

class HierarchicalClusterer(RegionClusterer, metaclass=abc.ABCMeta):
    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

        self.segmented = None

    def fit(self, num_target_clusters: int, max_iterations: int = 100, verbose: bool = False):
        return super().fit(num_target_clusters, max_iterations=max_iterations, verbose=verbose)

    def _update_segmented(self, flat_clust):
        image = np.zeros(self.region.region_array.shape, dtype=np.int16)
        image= image[:,:,0]
        
        # cluster 0 has special meaning: not assigned !
        assert(not 0 in [flat_clust[x] for x in flat_clust])

        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                image[i,j] = flat_clust[self.region.pixel2idx[(i,j)]]

        self.segmented = image

    def transform(self, num_target_clusters: int, verbose: bool = False) -> np.array:
        return self.segmented

    def segmentation(self) -> np.array:
        return self.segmented

class UPGMAClusterer(HierarchicalClusterer):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

    def fit(self, num_target_clusters: int, max_iterations: int = 100, verbose: bool = False):
        """Forms flat clusters with UPGMA clustering method (see scipy.cluster.hierarchy.linkage method='average' for more information) on the similarity matrix.

        Args:
            num_target_clusters (int): Number of desired clusters.
        """
        self.region.calculate_similarity()
        ssim = 1-self.region.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.linkage(squareform(ssim), method='average', metric='cosine')
        flat_clust = spc.hierarchy.fcluster(Z, t=num_target_clusters, criterion='maxclust')

        self._update_segmented(flat_clust)

class WPGMAClusterer(HierarchicalClusterer):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

    def fit(self, num_target_clusters: int, max_iterations: int = 100, verbose: bool = False):
        """Performs WPGMA linkage (see scipy.cluster.hierarchy.weighted for more information to the method) on the similarity matrix.

        Args:
            num_target_clusters (int): Number of desired clusters.
        """
        self.region.calculate_similarity()
        ssim = 1-self.region.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.weighted(squareform(ssim))
        flat_clust = spc.hierarchy.fcluster(Z, t=num_target_clusters, criterion='maxclust')

        self._update_segmented(flat_clust)

class WARDClusterer(HierarchicalClusterer):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

    def fit(self, num_target_clusters: int, max_iterations: int = 100, verbose: bool = False):
        """Performs Ward’s linkage (see scipy.cluster.hierarchy.ward for more information to the method) on the similarity matrix.

        Args:
            num_target_clusters (int): Number of desired clusters.
        """
        self.region.calculate_similarity()
        ssim = 1-self.region.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.ward(squareform(ssim))
        flat_clust = spc.hierarchy.fcluster(Z, t=num_target_clusters, criterion='maxclust')

        self._update_segmented(flat_clust)

class MedianClusterer(HierarchicalClusterer):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

    def fit(self, num_target_clusters: int, max_iterations: int = 100, verbose: bool = False):
        """Forms flat clusters with median clustering method (see scipy.cluster.hierarchy.linkage for more information to the method) on the similarity matrix.

        Args:
            num_target_clusters (int): Number of desired clusters.
        """
        self.region.calculate_similarity()
        ssim = 1-self.region.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.linkage(squareform(ssim), method='median', metric='cosine')
        flat_clust = spc.hierarchy.fcluster(Z, t=num_target_clusters, criterion='maxclust')

        self._update_segmented(flat_clust)

class CentroidClusterer(HierarchicalClusterer):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

    def fit(self, num_target_clusters: int, max_iterations: int = 100, verbose: bool = False):
        """Forms flat clusters with centroid clustering method (see scipy.cluster.hierarchy.linkage for more information to the method) on the similarity matrix.

        Args:
            num_target_clusters (int): Number of desired clusters.
        """
        self.region.calculate_similarity()
        ssim = 1-self.region.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.linkage(squareform(ssim), method='centroid', metric='cosine')
        flat_clust = spc.hierarchy.fcluster(Z, t=num_target_clusters, criterion='maxclust')

        self._update_segmented(flat_clust)
