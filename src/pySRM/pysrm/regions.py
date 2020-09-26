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



from numpy.ctypeslib import ndpointer

class SpectraRegion():

    @classmethod
    def from_pickle(cls, path):
        """Loads a SpectraRegion from pickle file

        Args:
            path (str): Path to pickle file to load spectra region from.

        Returns:
            SpectraRegion: SpectraRegion from pickle
        """

        obj = None
        with open(path, "rb") as fin:
            obj = pickle.load(fin)

        return obj

    def to_pickle(self, path):
        """Pickles the current object

        Args:
            path (str): Path to save the pickle file in.
        """

        with open(path, "wb") as fout:
            pickle.dump(self, fout)

    def ctypesCloseLibrary(self):
        """Unloads the C++ library
        """
        dlclose_func = ctypes.CDLL(None).dlclose
        dlclose_func.argtypes = [ctypes.c_void_p]
        dlclose_func.restype = ctypes.c_int
        dlclose_func(self.lib._handle)


    def loadLib(self):
        """Prepares everything for the usage of the C++ library
        """
        self.lib = ctypes.cdll.LoadLibrary(baseFolder+'/../../cppSRM/lib/libSRM.so')

        self.lib.StatisticalRegionMerging_New.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint8]
        self.lib.StatisticalRegionMerging_New.restype = ctypes.c_void_p

        self.lib.SRM_calc_similarity.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        self.lib.SRM_calc_similarity.restype = ctypes.POINTER(ctypes.c_float)

        self.lib.StatisticalRegionMerging_mode_dot.argtypes = [ctypes.c_void_p]
        self.lib.StatisticalRegionMerging_mode_dot.restype = None

        self.lib.StatisticalRegionMerging_mode_eucl.argtypes = [ctypes.c_void_p]
        self.lib.StatisticalRegionMerging_mode_eucl.restype = None


    def __init__(self, region_array, idx2mass, name=None):
        """Initializes a SpectraRegion

        Args:
            region_array (np.array): Array of spectra defining one region
            idx2mass (np.array): m/u Values for spectra
            name (str, optional): Name of this region (required if you want to do a comparative analysis). Defaults to None.
        """

        assert(not region_array is None)
        assert(not idx2mass is None)

        assert(len(region_array[0,0,:]) == len(idx2mass))

        self.lib = None
        self.loadLib()

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
        """Sets up logging facilities for SpectraRegion
        """

        self.logger = logging.getLogger('SpectraRegion')

        if len(self.logger.handlers) == 0:
            self.logger.setLevel(logging.INFO)

            consoleHandler = logging.StreamHandler()
            consoleHandler.setLevel(logging.INFO)

            self.logger.addHandler(consoleHandler)

            formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
            consoleHandler.setFormatter(formatter)


    def __getstate__(self):
        """Returns all data necessary to reconstruct the current state.

        Returns:
            dict: all data in one dict
        """
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
        """Reconstructs the current state from a dictionary with all data, and sets up idx2pixel.

        Args:
            state (dict of data): data required to reconstruct state
        """

        self.__dict__.update(state)

        self.logger = None
        self.__setlogger()

        self.idx2pixel = {}
        self.pixel2idx = {}

        for i in range(self.region_array.shape[0]*self.region_array.shape[1]):

            x,y = divmod(i, self.region_array.shape[1])

            self.idx2pixel[i] = (x,y)
            self.pixel2idx[(x,y)] = i


    def plot_array(self, fig, arr, discrete_legend=True):
        """Plots an array of values (e.g. segment IDs) into the given figure and adds a discrete legend.

        Args:
            fig (matplotlib figure): Figure to plot to.
            arr (array): array to visualize
            discrete_legend (bool, optional): Plot a discrete legend for values of array?. Defaults to True.

        Returns:
            [matplotlib figure]: Figure with plotted figure
        """


        valid_vals = np.unique(arr)
        heatmap = plt.matshow(arr, cmap=plt.cm.get_cmap('viridis', len(valid_vals)), fignum=fig.number)

        if discrete_legend:
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
        else:
            plt.colorbar(heatmap)
        
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
        


    def idx_for_mass(self, mass):
        """Returns the closest index for a specific mass.

        Args:
            mass (float): mass to look up index for

        Returns:
            int: index in m/z array for mass (or closest mass if not exactly found).
        """
        emass, eidx = self.__get_exmass_for_mass(mass)

        return eidx

    def __get_exmass_for_mass(self, mass, threshold=None):
        """Returns the closest mass and index for a specific mass.

        Args:
            mass (float): mass to look up index for
            threshold ([float], optional): Maximal distance from mass to contained m/z. Defaults to None.

        Returns:
            [float, int]: mass and index of closest contained m/z for mass
        """
        
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


    def mass_heatmap(self, masses, log=False, min_cut_off=None, plot=True):

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

        if plot:
            heatmap = plt.matshow(image)
            plt.colorbar(heatmap)
            plt.show()
            plt.close()

        return image
        #return image


    def calc_similarity(self, inputarray):
        # load image
        dims = 1


        if len(inputarray.shape) > 2:
            dims = inputarray.shape[2]
        print(dims)
        qs = []
        qArr = (ctypes.c_float * len(qs))(*qs)

        self.logger.info("Creating C++ obj")
        self.logger.info("{} {}".format(dims, inputarray.shape))

        # self.obj = lib.StatisticalRegionMerging_New(dims, qArr, 3)
        # print(inputarray.shape)
        # testArray = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]],[[25,26,27],[28,29,30]],[[31,32,33],[34,35,36]]], dtype=np.float32)
        # print(testArray.shape)
        # image_p = testArray.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # retValues = lib.SRM_test_matrix(self.obj, testArray.shape[0], testArray.shape[1], image_p)
        # exit()

        self.logger.info("dimensions {}".format(dims))
        self.logger.info("input dimensions {}".format(inputarray.shape))

        self.obj = self.lib.StatisticalRegionMerging_New(dims, qArr, len(qs))
        self.logger.info("Switching to dot mode")
        self.lib.StatisticalRegionMerging_mode_dot(self.obj)

        #inputarray = inputarray.astype(np.float32)
        inputarray = np.ascontiguousarray(inputarray, np.float32)
        image_p = inputarray.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.logger.info("Starting calc similarity c++")
        retValues = self.lib.SRM_calc_similarity(self.obj, inputarray.shape[0], inputarray.shape[1], image_p)

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

        self.spectra_similarity = self.calc_similarity(regArray)

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

    def __segment__centroid(self, number_of_regions):

        ssim = 1-self.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.linkage(squareform(ssim), method='centroid', metric='cosine')
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')

        return c


    def __segment__median(self, number_of_regions):

        ssim = 1-self.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.linkage(squareform(ssim), method='median', metric='cosine')
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

    def __segment__umap_ward(self, number_of_regions, dims=None, n_neighbors=10):

        self.dimred_elem_matrix = np.zeros((self.region_array.shape[0]*self.region_array.shape[1], self.region_array.shape[2]))

        ndims = self.region_array.shape[2]

        if not dims is None:
            if type(dims) == int:
                ndims = dims
            else:
                ndims = len(dims)

        self.elem_matrix = np.zeros((self.region_array.shape[0]*self.region_array.shape[1], ndims))

        print("Elem Matrix", self.elem_matrix.shape)

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

                if not dims is None:
                    self.elem_matrix[idx, :] = self.region_array[i,j,dims]
                else:
                    self.elem_matrix[idx, :] = self.region_array[i,j,:]

                idx2ij[idx] = (i,j)


        self.logger.info("UMAP reduction")
        self.dimred_elem_matrix = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        ).fit_transform(self.elem_matrix)

        self.logger.info("HDBSCAN reduction"), 

        print(self.dimred_elem_matrix.shape)

        pwdist = pdist(self.dimred_elem_matrix, metric="euclidean")

        print(pwdist.shape)

        Z = spc.hierarchy.ward(pwdist)
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')
        return c


    def __segment__umap_hdbscan(self, number_of_regions, dims=None, n_neighbors=10, min_samples=5, min_cluster_size=20):

        self.dimred_elem_matrix = np.zeros((self.region_array.shape[0]*self.region_array.shape[1], self.region_array.shape[2]))

        ndims = self.region_array.shape[2]

        if not dims is None:
            ndims = len(dims)

        self.elem_matrix = np.zeros((self.region_array.shape[0]*self.region_array.shape[1], ndims))

        print("Elem Matrix", self.elem_matrix.shape)

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

                if not dims is None:
                    self.elem_matrix[idx, :] = self.region_array[i,j,dims]
                else:
                    self.elem_matrix[idx, :] = self.region_array[i,j,:]

                idx2ij[idx] = (i,j)


        self.logger.info("UMAP reduction")
        self.dimred_elem_matrix = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        ).fit_transform(self.elem_matrix)

        self.logger.info("HDBSCAN reduction"), 

        clusterer = hdbscan.HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
        )
        self.dimred_labels = clusterer.fit_predict(self.dimred_elem_matrix)
        self.dimred_labels[self.dimred_labels >= 0] += 1

        #c = spc.hierarchy.fcluster(clusterer.single_linkage_tree_.to_numpy(), t=10, criterion='maxclust')

        return self.dimred_labels

    def vis_umap(self, legend=True):

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

    def plot_tic(self, masses=None):
        assert(not self.region_array is None)

        showcopy = np.zeros((self.region_array.shape[0], self.region_array.shape[1]))

        massIndices = [x for x in range(self.region_array.shape[2])]

        if masses != None:

            massIndices = []
            
            for mass in masses:
                mx, idx = self.__get_exmass_for_mass(mass)
                massIndices.append(idx)

        massIndices = sorted(massIndices)

        for i in range(0, showcopy.shape[0]):
            for j in range(0, showcopy.shape[1]):
                showcopy[i,j] = np.sum(self.region_array[i,j, massIndices])


        fig = plt.figure()
        self.plot_array(fig, showcopy, discrete_legend=False)
        plt.show()
        plt.close()

        return showcopy


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


    def segment(self, method="UPGMA", dims=None, number_of_regions=10, n_neighbors=10, min_samples=5, min_cluster_size=20):

        assert(not self.spectra_similarity is None)
        assert(method in ["UPGMA", "WPGMA", "WARD", "KMEANS", "UMAP_DBSCAN", "CENTROID", "MEDIAN", "UMAP_WARD"])

        self.logger.info("Calculating clusters")

        c = None

        if method == "UPGMA":
            c = self.__segment__upgma(number_of_regions)

        elif method == "WPGMA":
            c = self.__segment__wpgma(number_of_regions)

        elif method == "CENTROID":
            c = self.__segment__centroid(number_of_regions)

        elif method == "MEDIAN":
            c = self.__segment__median(number_of_regions)

        elif method == "WARD":
            c = self.__segment__ward(number_of_regions)

        elif method == "UMAP_DBSCAN":
            c = self.__segment__umap_hdbscan(number_of_regions, dims=dims, n_neighbors=n_neighbors, min_samples=min_samples, min_cluster_size=min_cluster_size)

        elif method == "UMAP_WARD":
            c = self.__segment__umap_ward(number_of_regions, dims=dims, n_neighbors=n_neighbors)

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


    def filter_clusters(self, method='remove_singleton', bg_x=4, bg_y=4):

        assert(method in ["remove_singleton", "most_similar_singleton", "merge_background", "remove_islands", "gauss"])

        cluster2coords = self.getCoordsForSegmented()

        if method == "gauss":
            result = np.zeros(self.segmented.shape)
    
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    neighbours = list()
                    
                    if i-1>=0:
                        neighbours.append(self.segmented[i-1][j])
                    if j-1>=0:
                        neighbours.append(self.segmented[i][j-1])
                    if i-1>=0 and j-1>=0:
                        neighbours.append(self.segmented[i-1][j-1])
                    if i+1<result.shape[0]:
                        neighbours.append(self.segmented[i+1][j])
                    if j+1<result.shape[1]:
                        neighbours.append(self.segmented[i][j+1])
                    if i+1<result.shape[0] and j+1<result.shape[1]:
                        neighbours.append(self.segmented[i+1][j+1])
                    
                    d = {x:neighbours.count(x) for x in neighbours}
                    key, freq = d.keys(), d.values()
                    
                    keys = np.asarray(list(key))
                    freqs = np.asarray(list(freq))
                    
                    if len(np.unique(keys))<=2:
                        result[i,j] = keys[np.argmax(freqs)]
                    else:
                        result[i,j] = self.segmented[i,j]

            self.segmented = result

        elif method == "remove_islands":

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

            xdim = bg_x
            ydim = bg_y

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

            #self.logger.info("Processing cluster: {}".format(clus))

            for pxl in allPixels:

                pxl_name = "{}__{}".format(str(len(sampleVec)), "_".join([str(x) for x in pxl]))

                sampleVec.append(pxl_name)
                conditionVec.append(0)

                exprData[pxl_name] = self.region_array[pxl[0], pxl[1], :]#.astype('int')


        for clus in clusters1:
            #self.logger.info("Processing cluster: {}".format(clus))

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


