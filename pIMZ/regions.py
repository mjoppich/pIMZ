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

import operator

import jinja2

import ms_peak_picker
import regex as re
import random

from skimage import measure as sk_measure

import glob
import shutil, io, base64


from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform, pdist

import scipy.cluster as spc

import scipy as sp
import sklearn as sk

import umap
import hdbscan



from numpy.ctypeslib import ndpointer

class SpectraRegion():
    """
    SpectraRegion class for any analysis of imzML spectra regions
    """

    @classmethod
    def from_pickle(cls, path):
        """Loads a SpectraRegion from pickle file.

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
        """Pickles the current SpectraRegion object.

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
        baseFolder = str(os.path.dirname(os.path.realpath(__file__)))

        libfile = (glob.glob(os.path.join(baseFolder, "libPIMZ*.so")) + glob.glob(os.path.join(baseFolder, "../build/lib*/pIMZ/", "libPIMZ*.so")))[0]
        self.lib = ctypes.cdll.LoadLibrary(libfile)

        self.lib.StatisticalRegionMerging_New.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint8]
        self.lib.StatisticalRegionMerging_New.restype = ctypes.c_void_p

        self.lib.SRM_calc_similarity.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        self.lib.SRM_calc_similarity.restype = ctypes.POINTER(ctypes.c_float)

        self.lib.StatisticalRegionMerging_mode_dot.argtypes = [ctypes.c_void_p]
        self.lib.StatisticalRegionMerging_mode_dot.restype = None

        self.lib.StatisticalRegionMerging_mode_eucl.argtypes = [ctypes.c_void_p]
        self.lib.StatisticalRegionMerging_mode_eucl.restype = None


    def __init__(self, region_array, idx2mass, name=None):
        """Initializes a SpectraRegion object with the following attributes:
        - lib: C++ library
        - logger (logging.Logger): Reference to the Logger object.
        - name (str): Name of the region. Defaults to None.
        - region_array (numpy.array): Array of spectra.
        - idx2mass (numpy.array): m/z values.
        - spectra_similarity (numpy.array): Pairwise similarity matrix. Initialized with None.
        - dist_pixel (numpy.array): Pairwise coordinate distance matrix (2-norm). Initialized with None.
        - idx2pixel (dict): Dictionary of enumerated pixels to their coordinates.
        - pixel2idx (dict): Inverted idx2pixel dict. Dictionary of coordinates mapped to pixel numbers. Initialized with None.
        - elem_matrix (array): A list of spectra with positional id correspond to the pixel number. Shape (n_samples, n_features). Initialized with None.
        - dimred_elem_matrix (array): Embedding of the elem_matrix in low-dimensional space. Shape (n_samples, n_components). Initialized with None.
        - dimred_labels (list): A list of HDBSCAN labels. Initialized with None.
        - segmented (numpy.array): Segmeted region_array which contains cluster ids.
        - segmented_method (str): Clustering method: "UPGMA", "WPGMA", "WARD", "KMEANS", "UMAP_DBSCAN", "CENTROID", "MEDIAN" or "UMAP_WARD". Initialized with None.
        - cluster_filters (list): A list of filters used. Can include: "remove_singleton", "most_similar_singleton", "merge_background", "remove_islands", "gauss".
        - consensus (dict): A dictionary of cluster ids mapped to their respective consensus spectra. Initialized with None.
        - consensus_method (str): Name of consensus method: "avg" or "median". Initialized with None.
        - consensus_similarity_matrix (array): Pairwise similarity matrix between consensus spectra. Initialized with None.
        - de_results_all (dict): Methods mapped to their differential analysis results. Initialized with None.
        - de_results_all (dict): Methods mapped to their differential analysis results (as pd.DataFrame). Initialized with None.

        Args:
            region_array (numpy.array): Array of spectra defining one region.
            idx2mass (numpy.array): m/z values for given spectra.
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
        self.df_results_all = defaultdict(lambda: dict())

        for i in range(self.region_array.shape[0]*self.region_array.shape[1]):

            x,y = divmod(i, self.region_array.shape[1])

            self.idx2pixel[i] = (x,y)
            self.pixel2idx[(x,y)] = i

    def __setlogger(self):
        """Sets up logging facilities for SpectraRegion.
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
            "de_results_all": self.de_results_all,
            "df_results_all": self.df_results_all
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
            fig (matplotlib.pyplot.figure): Figure to plot to.
            arr (array): array to visualize
            discrete_legend (bool, optional): Plots a discrete legend for array values. Defaults to True.

        Returns:
            matplotlib.pyplot.figure: Figure with plotted figure
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



    def to_aorta3d(self, folder, prefix, regionID, protWeights = None, nodf=False, pathPrefix = None, ctpred=None, kw2segment=None):
        """Extract eveilable data and prepares files for the 3D representation. 
        - .clustering.png: Picture of the segmented region.
        - .matrix.npy: Matrix of the segmented region.
        - .tsv: Marker Proteins Analysis findings. (Optional)
        - .info: Configuration file.

        Args:
            folder (str): Desired output folder.
            prefix (str): Desired name of the output files.
            regionID (int): Id of the desired region in the .imzML file.
            protWeights (ProteinWeights, optional): ProteinWeights object for translation of masses to protein name. Defaults to None.
            nodf (bool, optional): It set to True, do not perform differential analysis. Defaults to False.
            pathPrefix (str, optional): Desired path prefix for DE data files. Defaults to None.
            ctpred (str, optional): Path to tsv file with cluster-cell type mapping. Defaults to None.
            kw2segment (dict, optional): Dictionary keyword => segment; assigns keyword to all listed segments.
        """
        cluster2celltype = None# { str(x): str(x) for x in np.unique(self.segmented)}
        if ctpred != None:
            with open(ctpred, 'r') as fin:
                cluster2celltype = {}
                for line in fin:
                    line = line.strip().split("\t")
                    clusterID = line[0]
                    clusterType = line[1]

                    cluster2celltype[clusterID] = clusterType

            for x in cluster2celltype:
                self.logger.info("Cell-type assigned: {} -> {}".format(x, cluster2celltype[x]))


        # segments images
        cluster2coords = self.getCoordsForSegmented()
        os.makedirs(folder, exist_ok=True)
        segmentsPath = os.path.join(folder, prefix + "." + str(regionID) + ".clustering.png")
        self.logger.info("Segment Image: {}".format(segmentsPath))
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=self.segmented.min(), vmax=self.segmented.max())
        image = cmap(norm(self.segmented))
        plt.imsave(segmentsPath, image)


        # segment matrix
        
        matrixPath = os.path.abspath(os.path.join(folder, prefix + "." + str(regionID) + ".matrix.npy"))
        self.logger.info("Segment Matrix: {}".format(matrixPath))
        with open(matrixPath, "wb") as fout:
            np.save(fout, self.segmented)



        cluster2deData = {}
        # write DE data
        if protWeights != None:

            self.logger.info("Starting Marker Proteins Analysis")

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

            if cluster2celltype != None:
                if str(cluster) in cluster2celltype:
                    regionInfo["type_det"].append( cluster2celltype[str(cluster)] )
                else:
                    self.logger.info("No cell type info for cluster: '{}'".format(cluster))

            if kw2segment != None:
                for kw in kw2segment:
                    if cluster in kw2segment[kw] or str(cluster) in kw2segment[kw]:
                        regionInfo["type_det"].append( kw )

            if cluster in cluster2deData:
                regionInfo["de_data"] = cluster2deData[cluster]

            regionInfos[str(cluster)] = regionInfo



        infoDict = {}
        infoDict = {
            "region": regionID,
            "path_upgma": segmentsPath,
            "info": regionInfos,
            "segment_file": matrixPath
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
        emass, eidx = self._get_exmass_for_mass(mass)

        return eidx

    def _get_exmass_for_mass(self, mass, threshold=None):
        """Returns the closest mass and index for a specific mass.

        Args:
            mass (float): mass to look up index for
            threshold (float, optional): Maximal distance from mass to contained m/z. Defaults to None.

        Returns:
            float, int: mass and index of closest contained m/z for mass
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


    def mass_heatmap(self, masses, log=False, min_cut_off=None, max_cut_off=None, plot=True, verbose=True):
        if not isinstance(masses, (list, tuple, set)):
            masses = [masses]

        image = np.zeros((self.region_array.shape[0], self.region_array.shape[1]))

        for mass in masses:
            
            bestExMassForMass, bestExMassIdx = self._get_exmass_for_mass(mass)

            if verbose:
                self.logger.info("Processing Mass {} with best existing mass {}".format(mass, bestExMassForMass))

            for i in range(self.region_array.shape[0]):
                for j in range(self.region_array.shape[1]):

                    image[i,j] += self.region_array[i,j,bestExMassIdx]


        if log:
            image = np.log(image)

        if min_cut_off != None:
            image[image <= min_cut_off] = min_cut_off

        if max_cut_off != None:
            image[image >= max_cut_off] = max_cut_off

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
        
        self.logger.info("dimensions inputarray: {}".format(dims))
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
        """Returns similarity matrix.

        Args:
            mode (str, optional): Must be "spectra", "spectra_log" or "spectra_log_dist". Defaults to "spectra".
                - "spectra": Raw similarity matrix.
                - "spectra_log": Takes a logarithms and normalizes the similarity matrix by dividing by the
                maximum values.
                - "spectra_log_dist": Takes a logarithms, normalizes the similarity matrix by dividing by the
                maximum values and elementwise adds the distance matrix with 5% rate to the similarity matrix.
            features (list, optional): A list of desired masses. Defaults to [] meaning all masses.
            neighbors (int, optional): Number of neighboring masses to each feature to be included. Defaults to 1.

        Returns:
            numpy.array: Spectra similarity matrix
        """
        assert(mode in ["spectra", "spectra_log", "spectra_log_dist"])

        if len(features) > 0:
            for neighbor in range(neighbors):
                features = features + [i + neighbor for i in features] + [i - neighbor for i in features]
            features = np.unique(features)
            featureIndex = [self._get_exmass_for_mass(x) for x in features]
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

        self.logger.info("Ward reduction"), 

        print(self.dimred_elem_matrix.shape)

        pwdist = pdist(self.dimred_elem_matrix, metric="euclidean")

        print(pwdist.shape)

        Z = spc.hierarchy.ward(pwdist)
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')
        return c


    def __segment__umap_hdbscan(self, number_of_regions, dims=None, n_neighbors=10, min_samples=5, min_cluster_size=20, num_samples=10000):

        self.dimred_elem_matrix = np.zeros((self.region_array.shape[0]*self.region_array.shape[1], self.region_array.shape[2]))

        ndims = self.region_array.shape[2]

        if not dims is None:
            ndims = len(dims)

        self.elem_matrix = np.zeros((self.region_array.shape[0]*self.region_array.shape[1], ndims))

        self.logger.info("Elem Matrix of shape: {}".format(self.elem_matrix.shape))

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

        self.redo_hdbscan_on_dimred(number_of_regions, min_cluster_size, num_samples)

        return self.dimred_labels

    def redo_hdbscan_on_dimred(self, number_of_regions, min_cluster_size=15, num_samples=10000, set_segmented=True):

        if num_samples == -1 or self.dimred_elem_matrix.shape[0] < num_samples:
            selIndices = [x for x in range(0, self.dimred_elem_matrix.shape[0])]
        else:
            selIndices = random.sample([x for x in range(0, self.dimred_elem_matrix.shape[0])], num_samples)

        dr_matrix = self.dimred_elem_matrix[selIndices, :]

        self.logger.info("HDBSCAN reduction")
        self.logger.info("HDBSCAN Clusterer with matrix {}".format(dr_matrix.shape))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True).fit(dr_matrix)
        clusterer.generate_prediction_data()
        self.logger.info("HDBSCAN Soft Clusters with matrix {}".format(self.dimred_elem_matrix.shape))
        soft_clusters = hdbscan.prediction.membership_vector(clusterer, self.dimred_elem_matrix)
        self.logger.info("HDBSCAN Soft Clusters as output matrix {}".format(soft_clusters.shape))

        self.logger.info("HDBSCAN Soft Clusters: {}".format(soft_clusters.shape))
        print(soft_clusters)

        self.logger.info("HDBSCAN Labeling")
        self.dimred_labels = np.array([np.argmax(x) for x in soft_clusters])+1 # +1 avoids 0

        if len(np.unique(self.dimred_labels)) > number_of_regions:
            self.logger.info("Cluster Reduction for UMAP Result")
            self.segmented = np.reshape(self.dimred_labels, (self.region_array.shape[0], self.region_array.shape[1]))

            self.reduce_clusters(number_of_regions)
            self.dimred_labels = np.reshape(self.segmented, (self.region_array.shape[0] * self.region_array.shape[1],))

        self.dimred_labels = list(self.dimred_labels)


        if set_segmented:
            image_UPGMA = np.zeros(self.region_array.shape, dtype=np.int16)
            image_UPGMA = image_UPGMA[:,:,0]


            # cluster 0 has special meaning: not assigned !
            assert(not 0 in [self.dimred_labels[x] for x in self.dimred_labels])

            for i in range(0, image_UPGMA.shape[0]):
                for j in range(0, image_UPGMA.shape[1]):
                    image_UPGMA[i,j] = self.dimred_labels[self.pixel2idx[(i,j)]]


            self.segmented = image_UPGMA
            self.segmented_method = "UMAP_DBSCAN"


    def reduce_clusters(self, number_of_clusters):

        self.logger.info("Cluster Reduction")

        _ = self.consensus_spectra()
        self.consensus_similarity()

        Z = spc.hierarchy.ward(self.consensus_similarity_matrix)
        c = spc.hierarchy.fcluster(Z, t=number_of_clusters, criterion='maxclust')

        dimred_labels = np.reshape(self.segmented, (self.region_array.shape[0] * self.region_array.shape[1],))
        origlabels = np.array(dimred_labels, copy=True)

        for cidx, cval in enumerate(c):
            dimred_labels[origlabels == (cidx+1)] = cval

        self.segmented = np.reshape(dimred_labels, (self.region_array.shape[0], self.region_array.shape[1]))

        self.consensus = None
        self.consensus_similarity_matrix = None



    def vis_umap(self, legend=True):

        assert(not self.dimred_elem_matrix is None)
        assert(not self.dimred_labels is None)

        nplabels = np.array(self.dimred_labels)

        plt.figure()
        clustered = (nplabels >= 0)
        self.logger.info("Pixels    : {}".format(self.dimred_elem_matrix.shape[0]))
        self.logger.info("Unassigned: {}".format(self.dimred_elem_matrix[~clustered, ].shape[0]))
        plt.scatter(self.dimred_elem_matrix[~clustered, 0],
                    self.dimred_elem_matrix[~clustered, 1],
                    color=(1, 0,0),
                    label="Unassigned",
                    s=2.0)

        uniqueClusters = sorted(set([x for x in nplabels if x >= 0]))

        for cidx, clusterID in enumerate(uniqueClusters):
            cmap = matplotlib.cm.get_cmap('Spectral')

            clusterColor = cmap(cidx / len(uniqueClusters))

            plt.scatter(self.dimred_elem_matrix[nplabels == clusterID, 0],
                        self.dimred_elem_matrix[nplabels == clusterID, 1],
                        color=clusterColor,
                        label=str(clusterID),
                        s=10.0)

        if legend:
            plt.legend()
        plt.show()
        plt.close()

    def plot_tic(self, min_cut_off=None, max_cut_off=None, masses=None, hist=False):
        assert(not self.region_array is None)

        showcopy = np.zeros((self.region_array.shape[0], self.region_array.shape[1]))

        massIndices = [x for x in range(self.region_array.shape[2])]

        if masses != None:

            massIndices = []
            
            for mass in masses:
                mx, idx = self._get_exmass_for_mass(mass)
                massIndices.append(idx)

        massIndices = sorted(massIndices)
        allCounts = []

        for i in range(0, showcopy.shape[0]):
            for j in range(0, showcopy.shape[1]):
                pixelcount = np.sum(self.region_array[i,j, massIndices])

                showcopy[i,j] = pixelcount
                allCounts.append(pixelcount)


        if min_cut_off != None:
            showcopy[showcopy <= min_cut_off] = min_cut_off

        if max_cut_off != None:
            showcopy[showcopy >= max_cut_off] = max_cut_off


        fig = plt.figure()
        self.plot_array(fig, showcopy, discrete_legend=False)
        plt.show()
        plt.close()


        if hist:
            fig = plt.figure()
            plt.hist(allCounts, bins=len(allCounts), cumulative=True, histtype="step")
            plt.show()
            plt.close()

        return showcopy


    def set_null_spectra(self, condition):

        bar = progressbar.Bar()

        for i in bar(range(0, self.region_array.shape[0])):
            for j in range(0, self.region_array.shape[1]):
                if condition(self.region_array[i,j, :]):

                    self.region_array[i,j,:] = 0


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

    def list_segment_counts(self):

        regionCounts = Counter()

        for i in range(0, self.segmented.shape[0]):
            for j in range(0, self.segmented.shape[1]):

                regionCounts[ self.segmented[i,j] ] += 1

        for region in natsorted([x for x in regionCounts]):
            print(region, ": ", regionCounts[region])


    def segment(self, method="UPGMA", dims=None, number_of_regions=10, n_neighbors=10, min_samples=5, min_cluster_size=20, num_samples=1000):

        assert(method in ["UPGMA", "WPGMA", "WARD", "KMEANS", "UMAP_DBSCAN", "CENTROID", "MEDIAN", "UMAP_WARD"])
        if method in ["UPGMA", "WPGMA", "WARD", "KMEANS","CENTROID", "MEDIAN"]:
            assert(not self.spectra_similarity is None)

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
            c = self.__segment__umap_hdbscan(number_of_regions, dims=dims, n_neighbors=n_neighbors, min_samples=min_samples, min_cluster_size=min_cluster_size, num_samples=num_samples)

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

    def set_background(self, clusterIDs):

        if not type(clusterIDs) in [tuple, list, set]:
            clusterIDs = [clusterIDs]

        for clusterID in clusterIDs:
            self.segmented[ self.segmented == clusterID ] = 0


    def filter_clusters(self, method='remove_singleton', bg_x=4, bg_y=4, minIslandSize=10):

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

                if labelCells <= minIslandSize:
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

    def __cons_spectra__avg(self, cluster2coords, array):

        if array is None:
            array = self.region_array

        cons_spectra = {}
        for clusID in cluster2coords:

            spectraCoords = cluster2coords[clusID]

            if len(spectraCoords) == 1:
                coord = spectraCoords[0]
                # get spectrum, return spectrum
                avgSpectrum = array[coord[0], coord[1]]
            else:

                avgSpectrum = np.zeros((1, array.shape[2]))

                for coord in spectraCoords:
                    avgSpectrum += array[coord[0], coord[1]]

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


    def _get_median_spectrum(self, region_array):
        """
        Calculates the median spectrum from all spectra in region_array

        Args:
            region_array (np.array): Array of spectra
        """

        median_profile = np.array([0.0] * region_array.shape[2])

        for i in range(0, region_array.shape[2]):
            median_profile[i] = np.median(region_array[:,:,i])

        medProfAbove = [x for x in median_profile if x > 0]

        if len(medProfAbove) == 0:
            self.logger.info("Mostly Zero Median Profile!")
            startedLog = 0.0
        else:
            startedLog = np.quantile(medProfAbove, [0.05])[0]


        if startedLog == 0:
            startedLog = 0.001

        self.logger.info("Started Log Value: {}".format(startedLog))

        median_profile += startedLog

        return median_profile

    def __cons_spectra__median(self, cluster2coords, array=None):

        if array is None:
            array = self.region_array

        cons_spectra = {}
        for clusID in cluster2coords:

            spectraCoords = cluster2coords[clusID]

            if len(spectraCoords) == 1:
                coord = spectraCoords[0]
                # get spectrum, return spectrum
                medianSpectrum = array[coord[0], coord[1], :]
            else:

                clusterSpectra = np.zeros((1, len(spectraCoords), array.shape[2]))

                for cIdx, coord in enumerate(spectraCoords):
                    clusterSpectra[0, cIdx, :] = array[coord[0], coord[1], :]

                medianSpectrum = self._get_median_spectrum(clusterSpectra)

            cons_spectra[clusID] = medianSpectrum

        return cons_spectra



    def consensus_spectra(self, method="avg", set_consensus=True, array=None):

        if array is None:
            array = self.region_array
        else:
            pass#print("Using array argument")

        assert(not self.segmented is None)
        assert(method in ["avg", "median"])

        self.logger.info("Calculating consensus spectra")

        cluster2coords = self.getCoordsForSegmented()


        cons_spectra = None
        if method == "avg":
            cons_spectra = self.__cons_spectra__avg(cluster2coords, array=array)
        elif method == "median":
            cons_spectra = self.__cons_spectra__median(cluster2coords, array=array)


        if set_consensus:
            self.logger.info("Setting consensus spectra")
            self.consensus = cons_spectra
            self.consensus_method = method
        
        self.logger.info("Calculating consensus spectra done")

        return cons_spectra

    def mass_dabest(self, masses, background=0):

        assert(not self.segmented is None)

        if not isinstance(masses, (list, tuple, set)):
            masses = [masses]


        cluster2coords = self.getCoordsForSegmented()
        assert(background in cluster2coords)


        image = np.zeros((self.region_array.shape[0], self.region_array.shape[1]))

        for mass in masses:
            
            bestExMassForMass, bestExMassIdx = self._get_exmass_for_mass(mass)
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
        massValue, massIndex = self._get_exmass_for_mass(massValue)

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
        massValue, massIndex = self._get_exmass_for_mass(massValue)

        allExprValues = list(matrix[:, massIndex])

        num, anum = len(allExprValues), len([x for x in allExprValues if x > 0])
        resElem = []

        for modElem in mode:

            if modElem == "avg":
                resElem.append( np.mean(allExprValues) )

            elif modElem == "median":
                resElem.append( np.median(allExprValues) )

        return tuple(resElem), num, anum



    def _makeHTMLStringFilterTable(self, expDF):
        """Transform given pandas dataframe into HTML output

        Args:
            expDF (pd.DataFrame): Values for output

        Returns:
            htmlHead, htmlBody (str): HTML code for head and body
        """

        headpart = """
        """

        bodypart = """
        {% if title %}
        {{title}}
        {% endif %}
        
        <button id="csvButton" type="button">Save current table!</button>
        
        <table id="{{html_element_id}}" class="display" cellspacing="0" width="100%">
                <thead>
                <tr>
                {% for column in columns %}
                    <th>{{column}}</th>
                {% endfor %}
                </tr>
                </thead>

                <tbody>
                {% for key,row in rows.iterrows() %}
                <tr>
                    {% for column in columns %}
                    <td>{{ row[column] }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
                </tbody>

                <tfoot>
                <tr>
                {% for column in columns %}
                    <th>{{column}}</th>
                    {% endfor %}
                </tr>
                </tfoot>

                </table>

<script src="tablefilter/tablefilter.js"></script>

<script data-config>
    var filtersConfig = {
        base_path: 'tablefilter/',
        alternate_rows: true,
        rows_counter: true,
        btn_reset: true,
        loader: true,
        status_bar: true,
        mark_active_columns: true,
        highlight_keywords: true,
        sticky_headers: true,
        col_types: [{{coltypes}}],
        custom_options: {
            cols:[],
            texts: [],
            values: [],
            sorts: []
        },
        col_widths: [],
        extensions:[{ name: 'sort' }]
    };

    var tf = new TableFilter("{{html_element_id}}", filtersConfig);
    tf.init();

function download_csv(csv, filename) {
    var csvFile;
    var downloadLink;

    // CSV FILE
    csvFile = new Blob([csv], {type: "text/csv"});

    // Download link
    downloadLink = document.createElement("a");

    // File name
    downloadLink.download = filename;

    // We have to create a link to the file
    downloadLink.href = window.URL.createObjectURL(csvFile);

    // Make sure that the link is not displayed
    downloadLink.style.display = "none";

    // Add the link to your DOM
    document.body.appendChild(downloadLink);

    // Lanzamos
    downloadLink.click();
}

function isHidden(el) {
    var style = window.getComputedStyle(el);
    return ((style.display === 'none') || (style.visibility === 'hidden'))
}

function export_table_to_csv(html, filename) {
	var csv = [];
	var rows = document.querySelectorAll("table tr");
	
    for (var i = 0; i < rows.length; i++) {
		var row = [], cols = rows[i].querySelectorAll("td, th");

        if (!isHidden(rows[i]))
        {
            for (var j = 0; j < cols.length; j++) 
            {
                colText = ""+cols[j].innerText;
                colText = colText.replace(/(\\r\\n|\\n|\\r)/gm, ';')
                row.push(colText);

            }

            if (row.length > 0)
            {
                csv.push(row.join("\\t"));
            }		

        }
		    
	}

    // Download CSV
    download_csv(csv.join("\\n"), filename);
}

document.addEventListener('readystatechange', event => {

    if (event.target.readyState === "interactive") {      //same as:  document.addEventListener("DOMContentLoaded"...   // same as  jQuery.ready
            console.log("Ready state");

        document.getElementById("csvButton").addEventListener("click", function () {
            var html = document.getElementById("{{html_element_id}}").outerHTML;
            export_table_to_csv(html, "table.tsv");
        });

    }

    if (event.target.readyState === "complete") {
        console.log("Now external resources are loaded too, like css,src etc... ");
        
        document.getElementById("csvButton").addEventListener("click", function () {
            var html = document.getElementById("{{html_element_id}}").outerHTML;
            export_table_to_csv(html, "table.tsv");
        });
    }

});

                </script>

        """
        jsCols = []
       
        columnNames = expDF.columns.values.tolist()
        for cname in columnNames:

            if expDF[cname].dtypes in [int, float]:
                jsCols.append("\"number\"")
            else:
                jsCols.append("\"string\"")


        vHeader = [str(x) for x in columnNames]
        #print()

        self.logger.info("Got Columns: {}".format([x for x in zip(vHeader, jsCols)]))

        html_element_id= None
        if html_element_id == None:
            html_element_id = "dftable"

        jinjaTemplate = jinja2.Template(bodypart)
        output = jinjaTemplate.render(rows=expDF, columns=vHeader, title="",
                                      html_element_id=html_element_id, coltypes=", ".join(jsCols))

        return (headpart, output)


    def get_mask(self, regions):
        if not isinstance(regions, (list, tuple, set)):
            regions = [regions]

        outmask = np.zeros(self.segmented.shape)

        for region in regions:
            outmask[self.segmented == region] = 1

        return outmask


    def export_deres(self, method, resKey, outpath, title="DE Result"):
        """This methods writes out a HTMl-formatted table for all found DE results.

        Args:
            method (str): Method to export result for
            resKey (tuple): List of regions to look for result for
            outpath (str): outpath of HTML table. Required js-sources are copied into the same folder.
            title (str, optional): Title for result table
        """
        
       
        expDF = self.df_results_all[method][resKey].copy(deep=True)

        mass2image = {}
        requiredMasses = set(self.df_results_all[method][resKey]["gene_mass"].values)
        self.logger.info("Fetching Mass Heatmaps for all {} required masses".format(len(requiredMasses)))


        fgMask = self.get_mask(regions=resKey[0])
        bgMask = self.get_mask(regions=resKey[1])

        plt.imshow(fgMask)
        plt.show()
        plt.close()
        plt.imshow(bgMask)
        plt.show()
        plt.close()


        for mass in set(requiredMasses):
            mass_data = self.mass_heatmap(mass, plot=False, verbose=False)

            heatmap = plt.matshow(mass_data, fignum=100)
            plt.colorbar(heatmap)


            # Find contours at a constant value of 0.5
            contours = sk_measure.find_contours(bgMask, 0.5)
            # Display the image and plot all contours found
            for n, contour in enumerate(contours):
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color="blue")


            # Find contours at a constant value of 0.5
            contours = sk_measure.find_contours(fgMask, 0.5)

            # Display the image and plot all contours found
            for n, contour in enumerate(contours):
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color="green")



            pic_IObytes = io.BytesIO()
            plt.savefig(pic_IObytes,  format='png')
            pic_IObytes.seek(0)
            pic_hash = base64.b64encode(pic_IObytes.read()).decode()
            plt.close(100)

            imgStr = "<img src='data:image/png;base64,{}' alt='Red dot' />".format(pic_hash)

            mass2image[mass] = imgStr


        massImgValues = [mass2image.get(mass, "") for mass in expDF["gene_mass"].values]
        pos = expDF.columns.values.tolist().index("gene_mass")+1

        self.logger.info("Adding Mass Heatmap at pos {} of {} with {} entries".format(pos, len(expDF.columns.values.tolist()), len(massImgValues)))
        
        expDF.insert(loc = pos, 
          column = 'Mass Heatmap', 
          value = massImgValues) 

        (headpart, bodypart) = self._makeHTMLStringFilterTable(expDF)

        if title != None:
            bodypart = "<h1>"+title+"</h1>" + bodypart

        htmlfile="<html>\n<head>\n" + headpart + "</head>\n<body>\n" + bodypart + "</body>\n</html>"

        with open(outpath, 'w') as outHtml:
            outHtml.write(htmlfile)

        def copyFolders(root_src_dir, root_target_dir):

            for src_dir, dirs, files in os.walk(root_src_dir):
                dst_dir = src_dir.replace(root_src_dir, root_target_dir)
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)
                for file_ in files:
                    src_file = os.path.join(src_dir, file_)
                    dst_file = os.path.join(dst_dir, file_)
                    if os.path.exists(dst_file):
                        os.remove(dst_file)

                    shutil.copy(src_file, dst_dir)


        sourceDir = os.path.dirname(__file__) + "/tablefilter"
        targetDir = os.path.dirname(outpath) + "/tablefilter"

        self.logger.info("copy tablefilter files from {} to {}".format(sourceDir, targetDir))
        copyFolders(sourceDir, targetDir)




    def deres_to_df(self, method, resKey, protWeights, keepOnlyProteins=True, inverse_fc=False, max_adj_pval=0.05, min_log2fc=0.5):

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

        deResDF = self.de_results_all[method][resKey]

        ttr = deResDF.copy(deep=True)#self.de_results[resKey]      
        self.logger.info("DE result for case {} with {} results".format(resKey, ttr.shape))
        #ttr = deRes.summary()

        log2fcCol = "log2fc"
        massCol = "gene"
        adjPvalCol = "qval"

        ttrColNames = list(ttr.columns.values)
        self.logger.info("DF column names {}".format(ttrColNames))


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

            foundProt = []
            if protWeights != None:
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
            
                    clusterVec.append("_".join([str(x) for x in resKey[0]]))
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
                clusterVec.append("_".join([str(x) for x in resKey[0]]))
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

        self.df_results_all[method][resKey] = df.copy(deep=True)

        return df


    def find_all_markers(self, protWeights, keepOnlyProteins=True, replaceExisting=False, includeBackground=True,out_prefix="nldiffreg", outdirectory=None, use_methods = ["empire", "ttest", "rank"], count_scale={"ttest": 1, "rank": 1, "empire": 10000}):
        """
        Finds all marker proteins for a specific clustering.

        Args:
            protWeights (ProteinWeights): ProteinWeights object for translation of masses to protein name.
            keepOnlyProteins (bool, optional): If True, differential masses without protein name will be removed. Defaults to True.
            replaceExisting (bool, optional): If True, previously created marker-gene results will be overwritten. Defaults to False.
            includeBackground (bool, optional): If True, the cluster specific expression data are compared to all other clusters incl. background cluster. Defaults to True.
            out_prefix (str, optional): Prefix for results file. Defaults to "nldiffreg".
            outdirectory ([type], optional): Directory used for empire files. Defaults to None.
            use_methods (list, optional): Test methods for differential expression. Defaults to ["empire", "ttest", "rank"].
            count_scale (dict, optional): Count scales for different methods (relevant for empire, which can only use integer counts). Defaults to {"ttest": 1, "rank": 1, "empire": 10000}.

        Returns:
            dict of pd.dataframe: for each test conducted, one data frame with all marker masses for each cluster
        """
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

                resDF = self.deres_to_df(method, resKey, protWeights, keepOnlyProteins=keepOnlyProteins, inverse_fc=inverseFC)

                dfbyMethod[method] = pd.concat([dfbyMethod[method], resDF], sort=False)           

        return dfbyMethod

                    

    def __make_de_res_key(self, clusters0, clusters1):
        """Generates the storage key for two sets of clusters

        Args:
            clusters0 (list): list of cluster ids 1
            clusters1 (list): list of cluster ids 2

        Returns:
            tuple: tuple of both sorted cluster ids, as tuple
        """

        return (tuple(sorted(clusters0)), tuple(sorted(clusters1)))
        

    def clear_de_results(self):
        """Removes all sotred differential expression results.
        """
        self.de_results_all = defaultdict(lambda: dict())
        self.df_results_all = defaultdict(lambda: dict())

    def run_nlempire(self, nlDir, pdata, pdataPath, diffOutput):
        
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



    def find_markers(self, clusters0, clusters1=None, out_prefix="nldiffreg", outdirectory=None, replaceExisting=False, use_methods = ["empire", "ttest", "rank"], count_scale={"ttest": 1, "rank": 1, "empire": 10000}, sample_max=-1):

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
        self.logger.info("Running {} against {}".format(clusters0, clusters1))

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

            if sample_max > 0 and len(allPixels) > sample_max:
                allPixels = random.sample(allPixels, sample_max)

            bar = progressbar.ProgressBar()
            for pxl in bar(allPixels):

                pxl_name = "{}__{}".format(str(len(sampleVec)), "_".join([str(x) for x in pxl]))

                sampleVec.append(pxl_name)
                conditionVec.append(0)

                exprData[pxl_name] = self.region_array[pxl[0], pxl[1], :]#.astype('int')


        for clus in clusters1:
            #self.logger.info("Processing cluster: {}".format(clus))

            allPixels = cluster2coords[clus]

            if sample_max > 0 and len(allPixels) > sample_max:
                allPixels = random.sample(allPixels, sample_max)
            
            bar = progressbar.ProgressBar()
            for pxl in bar(allPixels):

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

                    if resKey in self.df_results_all["empire"]:
                        del self.df_results_all["empire"][resKey]

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
                    if resKey in self.df_results_all[testMethod]:
                        del self.df_results_all[testMethod][resKey]

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
    """12343 This class serves as lookup class for protein<->mass lookups for DE comparisons of IMS analyses.
    """

    def __set_logger(self):
        self.logger = logging.getLogger('ProteinWeights')
        self.logger.setLevel(logging.INFO)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)

        self.logger.addHandler(consoleHandler)

        formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
        consoleHandler.setFormatter(formatter)

    def __init__(self, filename, max_mass=-1):
        """Creates a ProteinWeights class. Requires a formatted proteinweights-file.

        Args:
            filename (str): File with at least the following columns: protein_id	gene_symbol	mol_weight_kd	mol_weight
            max_mass (float): Maximal mass to consider/include in object. -1 for no filtering. Masses above threshold will be discarded. Default is -1 .
        """

        self.__set_logger()

        self.protein2mass = defaultdict(set)
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

                if len(line) < 4:
                    continue

                proteinIDs = line[col2idx["protein_id"]].split(";")
                proteinNames = line[col2idx["gene_symbol"]].split(";")
                molWeight = float(line[col2idx["mol_weight"]])

                if max_mass >= 0 and molWeight > max_mass:
                    continue    

                if len(proteinNames) == 0:
                    proteinNames = proteinIDs

                for proteinName in proteinNames:
                    self.protein2mass[proteinName].add(molWeight)
                    self.protein_name2id[proteinName] = proteinIDs

            allMasses = self.get_all_masses()

            self.logger.info("Loaded a total of {} proteins with {} masses".format(len(self.protein2mass), len(allMasses)))

    def get_all_masses(self):
        """Returns all masses contained in the lookup-dict

        Returns:
            set: set of all masses used by this object
        """
        allMasses = set()
        for x in self.protein2mass:
            for mass in self.protein2mass[x]:
                allMasses.add(mass)

        return allMasses

    def get_mass_to_protein(self):
        """Returns a dictionary of a mass to proteins with this mass. Remember that floats can be quite exact: no direct comparisons!

        Returns:
            dict: dictionary mass => set of proteins
        """

        mass2prot = defaultdict(set)
        for x in self.protein2mass:
            for mass in self.protein2mass[x]:
                mass2prot[mass].add(x)

        return mass2prot

    def print_collisions(self, maxdist=2.0, print_proteins=False):
        """Prints number of proteins with collision, as well as mean and median number of collisions.

        For each recorded mass it is checked how many other proteins are in the (mass-maxdist, mass+maxdist) range.

        Args:
            maxdist (float, optional): Mass range in order to still accept a protein. Defaults to 2.
            print_proteins (bool, optional): If True, all collision proteins are printed. Defaults to False.
        """

        allProts = [x for x in self.protein2mass]
        mass2prot = self.get_mass_to_protein()            
        
        protsWithCollision = Counter()
        sortedMasses = sorted([x for x in mass2prot])

        for mass in sortedMasses:

            lmass, umass = mass-maxdist, mass+maxdist
            currentProts = mass2prot[mass]

            massProts = set()

            for tmass in sortedMasses:
                if lmass <= tmass <= umass:
                    massProts = massProts.union(mass2prot[tmass])

                if umass < tmass:
                    break

            if len(currentProts) == 1:
                for prot in currentProts:
                    if prot in massProts:
                        massProts.remove(prot)

            if len(massProts) > 0:
                for prot in massProts:
                    protsWithCollision[prot] += 1

        self.logger.info("         Number of total proteins: {}".format(len(self.protein2mass)))
        self.logger.info("           Number of total masses: {}".format(len(mass2prot)))
        self.logger.info("Number of proteins with collision: {}".format(len(protsWithCollision)))
        self.logger.info("        Mean Number of Collidings: {}".format(np.mean([protsWithCollision[x] for x in protsWithCollision])))
        self.logger.info("      Median Number of Collidings: {}".format(np.median([protsWithCollision[x] for x in protsWithCollision])))

        if print_proteins:
            self.logger.info("Proteins with collision: {}".format([x for x in protsWithCollision]))
        else:
            self.logger.info("Proteins with collision: {}".format(protsWithCollision.most_common(10)))
        

    def get_protein_from_mass(self, mass, maxdist=2):
        """Searches all recorded mass and proteins and reports all proteins which have at least one mass in (mass-maxdist, mass+maxdist) range.

        Args:
            mass (float): mass to search for
            maxdist (float, optional): allowed offset for lookup. Defaults to 2.

        Returns:
            list: list of all (protein, weight) tuple which have a protein in the given mass range
        """

        possibleMatches = []

        for protein in self.protein2mass:
            protMasses = self.protein2mass[protein]

            for protMass in protMasses:
                if abs(mass-protMass) < maxdist:
                    possibleMatches.append((protein, protMass))

        return possibleMatches

    def get_masses_for_protein(self, protein):
        """Returns all recorded masses for a given protein. Return None if protein not found

        Args:
            protein (str): protein to search for in database (exact matching)

        Returns:
            set: set of masses for protein
        """

        return self.protein2mass.get(protein, None)


    def compare_masses(self, pw):
        """For each protein contained in both PW objects, and for each protein mass, the distance to the best match in the other PW object is calculated.

        This is meant to provide a measure of how accuracte the theoretical calculation of m/z // Da is.

        Args:
            pw (ProteinWeights): The ProteinWeights object to compare to.
        """

        dists = []
        consideredMasses = 0

        for x in self.protein2mass:

            lookKey = x.upper()
            selfMasses = self.protein2mass[x]

            otherMasses = None
            if lookKey in pw.protein2mass:
                otherMasses = pw.protein2mass[lookKey]
            elif x in pw.protein2mass:
                otherMasses = pw.protein2mass[x]

            if otherMasses == None:
                continue

            selfMasses = sorted(selfMasses)
            otherMasses = sorted(otherMasses)

            protDiffs = []
            for smass in selfMasses:

                sMassDiffs = []
                for omass in otherMasses:
                    sMassDiffs.append(abs(smass-omass))

                selMassDiff = min(sMassDiffs)
                protDiffs.append(selMassDiff)


            selProtDiff = min(protDiffs)

            if selProtDiff > 500:
                print(x,selMassDiff, selfMasses, otherMasses)

            dists += [selProtDiff]

            if len(protDiffs) > 0:
                consideredMasses += 1

        print("Total number of considered masses: {}".format(consideredMasses))
        print("Total number of diffs > 100: {}".format(len([x for x in dists if x > 100])))
        print("Total number of diffs > 5  : {}".format(len([x for x in dists if x > 5])))
        print("Total number of diffs > 1  : {}".format(len([x for x in dists if x > 1])))
        print("Total number of diffs <= 1  : {}".format(len([x for x in dists if x <= 1])))

        print("{}\t{}\t{}\t{}\t{}\t{}".format(
            consideredMasses, len(dists), min(dists), np.median(dists), np.mean(dists), max(dists)
        ))

            


