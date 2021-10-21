# general
import logging
import json
import os
import random
import math
from collections import defaultdict, Counter
import glob
import shutil, io, base64, abc

# general package
from natsort import natsorted
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

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

from .imzml import IMZMLExtract

#web/html
import jinja2

# applications
import progressbar
def makeProgressBar():
    return progressbar.ProgressBar(widgets=[
        progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.AdaptiveETA()
        ])

class SpectraRegion():
    pass

class RegionClusterer(metaclass=abc.ABCMeta):

    def __init__(self, region:SpectraRegion) -> None:
        self.region = region
        self.logger = None
        self.__set_logger()

    def __set_logger(self):
        self.logger = logging.getLogger(self.methodname())
        self.logger.setLevel(logging.INFO)

        if not logging.getLogger().hasHandlers():

            consoleHandler = logging.StreamHandler()
            consoleHandler.setLevel(logging.INFO)

            self.logger.addHandler(consoleHandler)

            formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
            consoleHandler.setFormatter(formatter)
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'fit') and callable(subclass.fit) and
                hasattr(subclass, 'transform') and callable(subclass.transform) and
                hasattr(subclass, 'fit_transform') and callable(subclass.fit_transform) and
                hasattr(subclass, 'segmentation') and callable(subclass.segmentation) and
                hasattr(subclass, 'region')
                )

    def methodname(self):
        """Brief description of the specific clusterer

        """
        return self.__class__.__name__


    @abc.abstractmethod
    def fit(self, num_target_clusters:int, max_iterations:int=100, verbose:bool=False):
        """[summary]

        Args:
            num_target_clusters ([type]): [description]
            max_iterations (int, optional): [description]. Defaults to 100.
            verbose (bool, optional): Verbose output. Defaults to False.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, num_target_clusters: int, verbose:bool=False) -> np.array:
        """
        Returns the final segmentation

        Args:
            num_target_clusters (int): number of target clusters
            verbose (bool, optional): Verbose output. Defaults to False.


        Raises:
            NotImplementedError: (abstract class)

        Returns:
            np.array: segmentation
        """
        raise NotImplementedError

    @abc.abstractmethod
    def segmentation(self) -> np.array:
        """Returns the final segmentation for given region

        Raises:
            NotImplementedError: [description]

        Returns:
            np.array: segmentation
        """

        raise NotImplementedError

    def fit_transform(self, num_target_clusters: int, verbose:bool=False) -> np.array:
        """[summary]

        Args:
            num_target_clusters (int): number of target clusters
            verbose (bool, optional): Verbose output. Defaults to False.


        Returns:
            np.array: segmentation
        """

        self.fit(num_target_clusters=num_target_clusters, verbose=verbose)
        return self.transform(num_target_clusters=num_target_clusters, verbose=verbose)

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
            SpectraRegion: SpectraRegion object from pickle.
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
        - de_results_all (dict): Methods mapped to their differential analysis results (as pd.DataFrame). Initialized with an empty defaultdict.

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

            self.logger.info("Added new Stream Handler")


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

        if discrete_legend:
            valid_vals = sorted(np.unique(arr))

            normArray = np.zeros(arr.shape)
            
            val_lookup = {}
            positions = []
            for uIdx, uVal in enumerate(valid_vals):
                normArray[arr == uVal] = uIdx
                val_lookup[uIdx] = uVal
                positions.append(uIdx)

            heatmap = plt.matshow(normArray, cmap=plt.cm.get_cmap('viridis', len(valid_vals)), fignum=fig.number)

            # calculate the POSITION of the tick labels
            #positions = np.linspace(0, len(valid_vals), len(valid_vals))

            def formatter_func(x, pos):
                'The two args are the value and tick position'
                val = val_lookup[x]
                return val

            formatter = plt.FuncFormatter(formatter_func)

            # We must be sure to specify the ticks matching our target names
            plt.colorbar(heatmap, ticks=positions, format=formatter, spacing='proportional')
        else:
            
            heatmap = plt.matshow(arr, cmap=plt.cm.get_cmap('viridis'), fignum=fig.number)
            plt.colorbar(heatmap)
        
        return fig



    def to_aorta3d(self, folder, prefix, regionID, protWeights = None, nodf=False, pathPrefix = None, ctpred=None, kw2segment=None):
        """Extracts available data and prepares files for the 3D representation:
        - .clustering.png: Picture of the segmented region.
        - .matrix.npy: Matrix of the segmented region.
        - .tsv: Marker Proteins Analysis findings. (Optional)
        - .info: Configuration file.

        Args:
            folder (str): Desired output folder.
            prefix (str): Desired name of the output files.
            regionID (int): Id of the desired region in the .imzML file.
            protWeights (ProteinWeights, optional): ProteinWeights object for translation of masses to protein name. Defaults to None.
            nodf (bool, optional): If set to True, do not perform differential analysis. Defaults to False.
            pathPrefix (str, optional): Desired path prefix for DE data files. Defaults to None.
            ctpred (str, optional): Path to .tsv file with cluster-cell type mapping. Defaults to None.
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

        # hdf5 file with intensities

        hdf5Path = os.path.abspath(os.path.join(folder, prefix + "." + str(regionID) + ".hdf5"))

        with h5py.File(hdf5Path, "w") as data_file:
            grp = data_file.create_group("intensities")

            for mzIdx in range(0, self.region_array.shape[2]):
                mzValue = self.idx2mass[mzIdx]
                dset = grp.create_dataset( str(mzIdx) , data=self.region_array[:,:, mzIdx])
                dset.attrs["mz"] = mzValue
            data_file.close()

        cluster2deData = {}
        # write DE data
        if protWeights != None:

            self.logger.info("Starting Marker Proteins Analysis")

            if not nodf:
                markerGenes = self.find_all_markers(protWeights, use_methods=["ttest"], replaceExisting=False, includeBackground=True)
            
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
            "segment_file": matrixPath,
            "hdf5_file": hdf5Path
        }


        jsElems = json.dumps([infoDict])

        # write config_file

        with open(os.path.join(folder, prefix + "." + str(regionID) + ".info"), 'w') as fout:
            print(jsElems, file=fout)
        

    def judge_de_masses(self, filter_func):
        """Adds or edits the de_judge element of a differential analysis result dictionary of the given SpectraRegion object by applying the desired function.

        Args:
            filter_func (function): A function that is applied to every entry of a differential analysis result of every available method.
        """
        for test in self.df_results_all:

            for comp in self.df_results_all[test]:

                testDF = self.df_results_all[test][comp]

                self.logger.info("Judging results from {} and comparison {}".format(test, comp))

                dfRes = [False] * len(testDF)
                for index, row in testDF.iterrows():

                    res = filter_func(self, row)
                    dfRes[index] = res

                if "de_judge" in testDF.columns.values.tolist():
                    self.logger.info("Removing existing judge in comp: {}".format(comp))
                    del testDF["de_judge"]

                pos = testDF.columns.values.tolist().index("gene_mass")+1
                testDF.insert(loc = pos, 
                column = 'de_judge', 
                value = dfRes) 
                
                self.logger.info("Storing results from {} and comparison {} (position {})".format(test, comp,pos))
                self.df_results_all[test][comp] = testDF

        


    def idx_for_mass(self, mass):
        """Returns the closest index for a specific mass.

        Args:
            mass (float): mass to look up index for.

        Returns:
            int: index in m/z array for mass (or closest mass if not exactly found).
        """
        emass, eidx = self._get_exmass_for_mass(mass)

        return eidx

    def get_mass_from_index(self, idx):
        return self.idx2mass[idx]
 
    def _get_exmass_for_mass(self, mass, threshold=None):
        """Returns the closest mass and index in .imzML file for a specific mass.

        Args:
            mass (float): mass to look up index for.
            threshold (float, optional): Maximal distance from mass to contained m/z. Defaults to None.

        Returns:
            float, int: mass and index of closest contained m/z for mass.
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

    def _fivenumber(self, valuelist, addfuncs=None):
        """Creates five number statistics for values in valuelist.

        Args:
            valuelist (list/tuple/numpy.array (1D)): List of values to use for statistics.

        Returns:
            tuple: len, len>0, min, 25-quantile, 50-quantile, 75-quantile, max
        """

        min_ = np.min(valuelist)
        max_ = np.max(valuelist)

        (quan25_, quan50_, quan75_) = np.quantile(valuelist, [0.25, 0.5, 0.75])

        addRes = []
        if addfuncs != None:
            addRes = [fx(valuelist) for fx in addfuncs]

        return tuple([len(valuelist), len([x for x in valuelist if x > 0]), min_, quan25_, quan50_, quan75_, max_] + addRes)

    def detect_highly_variable_masses(self, topn=2000, bins=50, return_mz=False, meanThreshold=0.05):
        
        hvIndices = IMZMLExtract.detect_hv_masses(self.region, topn=topn, bins=bins, meanThreshold=meanThreshold)

        if return_mz:
            return [self.idx2mass[x] for x in hvIndices]

        return hvIndices
        

    def plot_intensity_distribution(self, mass):
        bestExMassForMass, bestExMassIdx = self._get_exmass_for_mass(mass)
        
        allMassIntensities = []
        for i in range(self.region_array.shape[0]):
            for j in range(self.region_array.shape[1]):

                allMassIntensities.append(self.region_array[i,j,bestExMassIdx])

        print("Five Number stats + mean, var", self._fivenumber(allMassIntensities, addfuncs=[np.mean, lambda x: np.var(x)/np.mean(x)]))

        plt.hist(allMassIntensities, bins=len(allMassIntensities))
        plt.title("Mass intensity histogram (m/z = {})".format(round(bestExMassForMass, 3)))
        plt.show()

    def mass_heatmap(self, masses, log=False, min_cut_off=None, max_cut_off=None, plot=True, verbose=True, pw=None, title="{mz}"):
        """Filters the region_region to the given masses and returns the matrix with summed
        representation of the gained spectra.

        Args:
            masses (array): List of masses or protein names (requires pw set).
            log (bool, optional): Whether to take logarithm of the output matrix. Defaults to False.
            min_cut_off (int/float, optional): Lower limit of values in the output matrix. Smaller values will be replaced with min_cut_off. Defaults to None.
            max_cut_off (int/float, optional): Upper limit of values in the output matrix. Greater values will be replaced with max_cut_off. Defaults to None.
            plot (bool, optional): Whether to plot the output matrix. Defaults to True.
            verbose (bool, optional): Whether to add information to the logger. Defaults to True.
            pw (ProteinWeights, optional): Allows to translate masses names to actual masses in a given ProteinWeights object. Defaults assuming the elements in masses are numeric, hence None.

        Returns:
            numpy.array: Each element is a sum of intensities at given masses.
        """
        if not isinstance(masses, (list, tuple, set)):
            masses = [masses]

        useMasses = []
        for x in masses:
            if type(x) == str:
                massprots = pw.get_masses_for_protein(x)
                useMasses += list(massprots)
            else:
                useMasses.append(x)

        image = np.zeros((self.region_array.shape[0], self.region_array.shape[1]))

        for mass in useMasses:
            
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
            plt.gca().xaxis.set_ticks_position('bottom')
            plt.title(title.format(mz=";".join([str(round(x, 3)) if not type(x) in [str] else x for x in masses])))
            plt.show()
            plt.close()

        return image


    def calc_similarity(self, inputarray):
        """Returns cosine similarity matrix which is claculated with help of C++ libarary.

        Args:
            inputarray (numpy.array): Array of spectra.

        Returns:
            numpy.array: Pairwise similarity matrix.
        """
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
            mode (str, optional): Must be "spectra", "spectra_log" or "spectra_log_dist". Defaults to "spectra".\n
                - "spectra": Raw similarity matrix.\n
                - "spectra_dist": Raw similarity matrix and elementwise adds the distance matrix with 5% rate to the similarity matrix..\n
                - "spectra_log": Takes a logarithm and normalizes the similarity matrix by dividing by the maximum values.\n
                - "spectra_log_dist": Takes a logarithm, normalizes the similarity matrix by dividing by the maximum values and elementwise adds the distance matrix with 5% rate to the similarity matrix.\n
            features (list, optional): A list of desired masses. Defaults to [] meaning all masses.
            neighbors (int, optional): Number of neighboring masses to each feature to be included. Defaults to 1.

        Returns:
            numpy.array: Spectra similarity matrix
        """
        assert(mode in ["spectra", "spectra_dist", "spectra_log", "spectra_log_dist"])

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

        if mode in ["spectra_log_dist", "spectra_dist"]:

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
        """Forms flat clusters with UPGMA clustering method (see scipy.cluster.hierarchy.linkage method='average' for more information) on the similarity matrix.

        Args:
            number_of_regions (int): Number of desired clusters.

        Returns:
            numpy.ndarray: An array where each element is the flat cluster number to which original observation belongs.
        """
        ssim = 1-self.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.linkage(squareform(ssim), method='average', metric='cosine')
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')

        return c

    def __segment__centroid(self, number_of_regions):
        """Forms flat clusters with centroid clustering method (see scipy.cluster.hierarchy.linkage for more information to the method) on the similarity matrix.

        Args:
            number_of_regions (int): Number of desired clusters.

        Returns:
            numpy.ndarray: An array where each element is the flat cluster number to which original observation belongs.
        """
        ssim = 1-self.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.linkage(squareform(ssim), method='centroid', metric='cosine')
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')

        return c


    def __segment__median(self, number_of_regions):
        """Forms flat clusters with median clustering method (see scipy.cluster.hierarchy.linkage for more information to the method) on the similarity matrix.

        Args:
            number_of_regions (int): Number of desired clusters.

        Returns:
            numpy.ndarray: An array where each element is the flat cluster number to which original observation belongs.
        """
        ssim = 1-self.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.linkage(squareform(ssim), method='median', metric='cosine')
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')

        return c

    def __segment__wpgma(self, number_of_regions):
        """Performs WPGMA linkage (see scipy.cluster.hierarchy.weighted for more information to the method) on the similarity matrix.

        Args:
            number_of_regions (int): Number of desired clusters.

        Returns:
            numpy.ndarray: An array where each element is the flat cluster number to which original observation belongs.
        """
        ssim = 1-self.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.weighted(squareform(ssim))
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')

        return c

    def __segment__ward(self, number_of_regions):
        """Performs Ward’s linkage (see scipy.cluster.hierarchy.ward for more information to the method) on the similarity matrix.

        Args:
            number_of_regions (int): Number of desired clusters.

        Returns:
            numpy.ndarray: An array where each element is the flat cluster number to which original observation belongs.
        """
        ssim = 1-self.spectra_similarity
        ssim[range(ssim.shape[0]), range(ssim.shape[1])] = 0

        Z = spc.hierarchy.ward(squareform(ssim))
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')
        return c

    def prepare_elem_matrix(self, sparse=False, dims=None):


        ndims = self.region_array.shape[2]

        if not dims is None:
            if type(dims) == int:
                ndims = dims
            else:
                ndims = len(dims)

        if not sparse:
            self.dimred_elem_matrix = np.zeros((self.region_array.shape[0]*self.region_array.shape[1], self.region_array.shape[2]))
            elem_matrix = np.zeros((self.region_array.shape[0]*self.region_array.shape[1], ndims))

        else:
            self.dimred_elem_matrix = csr_matrix((self.region_array.shape[0]*self.region_array.shape[1], self.region_array.shape[2]))
            elem_matrix = csr_matrix((self.region_array.shape[0]*self.region_array.shape[1], ndims))




        print("Elem Matrix", elem_matrix.shape)

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
                    elem_matrix[idx, :] = self.region_array[i,j,dims]
                else:
                    elem_matrix[idx, :] = self.region_array[i,j,:]

                idx2ij[idx] = (i,j)

        return elem_matrix, idx2ij


    def __segment__umap_ward(self, number_of_regions, densmap=False, dims=None, n_neighbors=10):
        """Performs UMAP dimension reduction on region array followed by Euclidean pairwise distance calculation in order to do Ward's linkage.

        Args:
            number_of_regions (int): Number of desired clusters.
            densmap (bool, optional): Whether to use densMAP (density-preserving visualization tool based on UMAP). Defaults to False.
            dims (int/list, optional): The desired amount of intensity values that will be taken into account performing dimension reduction. Defaults to None, meaning all intensities are considered.
            n_neighbors (int, optional): The size of the local neighborhood (in terms of number of neighboring sample points) used for manifold approximation. For more information check UMAP documentation. Defaults to 10.

        Returns:
            numpy.ndarray: An array where each element is the flat cluster number to which original observation belongs.
        """
        self.elem_matrix, idx2ij = self.prepare_elem_matrix(dims)


        self.logger.info("UMAP reduction")
        self.dimred_elem_matrix = umap.UMAP(
            densmap=densmap,
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
        self.dimred_labels = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')
        return self.dimred_labels


    def __segment__umap_hdbscan(self, number_of_regions, densmap=False, dims=None, n_neighbors=10, min_cluster_size=20, num_samples=10000):
        """Performs UMAP dimension reduction on region array followed by the HDBSCAN clustering.

        Args:
            number_of_regions (int): Number of desired clusters.
            densmap (bool, optional): Whether to use densMAP (density-preserving visualization tool based on UMAP). Defaults to False.
            dims (int/list, optional): The desired amount of intensity values that will be taken into account performing dimension reduction. Defaults to None, meaning all intensities are considered.
            n_neighbors (int, optional): The size of the local neighborhood (in terms of number of neighboring sample points) used for manifold approximation. For more information check UMAP documentation. Defaults to 10.
            min_cluster_size (int, optional): The minimum size of HDBSCAN clusters. Defaults to 20.
            num_samples (int, optional): Number of intensity values that will be used during HDBSCAN clustering. Defaults to 10000.

        Returns:
            list: A list of HDBSCAN labels. 
        """

        self.elem_matrix, idx2ij = self.prepare_elem_matrix(dims)

        self.logger.info("UMAP reduction")
        self.dimred_elem_matrix = umap.UMAP(
            densmap=densmap,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        ).fit_transform(self.elem_matrix)

        self.redo_hdbscan_on_dimred(number_of_regions, min_cluster_size, num_samples)

        return self.dimred_labels

    def __segment__kmeans(self, number_of_regions):
        """Forms flat clusters with k-means++ clustering method (see scipy.cluster.vq.kmeans2 for more information) on the spectra array.

        Args:
            number_of_regions (int): Number of desired clusters.

        Returns:
            numpy.ndarray: An array where each element is the flat cluster number to which original observation belongs.
        """

        all_spectra = self.region_array.reshape(-1, self.region_array.shape[2])
        centroid, label = kmeans2(all_spectra, k=number_of_regions, iter=10, minit='++')
        if 0 in label:
            label += 1
        return label

    def redo_hdbscan_on_dimred(self, number_of_regions, min_cluster_size=15, num_samples=10000, set_segmented=True):
        """Performs HDBSCAN clustering (Hierarchical Density-Based Spatial Clustering of Applications with Noise) with the additional UMAP dimension reduction in order to achieve the desired number of clusters.

        Args:
            number_of_regions (int): Number of desired clusters.
            min_cluster_size (int, optional): The minimum size of HDBSCAN clusters. Defaults to 15.
            num_samples (int, optional): Number of intensity values that will be used during HDBSCAN clustering. Defaults to 10000.
            set_segmented (bool, optional): Whether to update the segmented array of the current object. Defaults to True.
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
        """Reducing the number of clusters in segmented array by "reclustering" after the Ward's clustering on pairwise similarity matrix between consensus spectra.

        Args:
            number_of_clusters (int): Number of desired clusters.
        """
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



    def vis_umap(self, legend=True, marker_size=(2.0, 10.0)):
        """Visualises a scatterplot of the UMAP/densMAP assigned pixels.

        Args:
            legend (bool, optional): Whether to include the legend to the plot. Defaults to True.
            marker_size (tuple, optional): Tuple of preferred marker sizes for unassigned marker_size[0] and lable specific points marker_size[1]. Defaults to (2.0, 10.0).
        """
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
                    s=marker_size[0])

        uniqueClusters = sorted(set([x for x in nplabels if x >= 0]))

        for cidx, clusterID in enumerate(uniqueClusters):
            cmap=plt.cm.get_cmap('viridis', len(uniqueClusters))

            clusterColor = cmap(cidx / len(uniqueClusters))

            plt.scatter(self.dimred_elem_matrix[nplabels == clusterID, 0],
                        self.dimred_elem_matrix[nplabels == clusterID, 1],
                        color=clusterColor,
                        label=str(clusterID),
                        s=marker_size[1])

        if legend:
            plt.legend(loc="upper left",  bbox_to_anchor=(1.05, 1))
        plt.show()
        plt.close()

    def plot_tic(self, min_cut_off=None, max_cut_off=None, masses=None, hist=False, plot_log=False):
        """Displays a matrix where each pixel is the sum of intensity values over all m/z summed in the corresponding pixel in region_array.

        Args:
            min_cut_off (int/float, optional): Minimum allowed value. Smaller values will be replaced with min_cut_off value. Defaults to None.
            max_cut_off (int/float, optional): Maximum allowed value. Greater values will be replaced with max_cut_off value. Defaults to None.
            masses (numpy.array/list, optional): A list of masses to which each spectrum will be reduced. Defaults to None, meaning all masses are considered.
            hist (bool, optional): Whether to plot a cumularive histogram of values (sums) frequencies. Defaults to False.
            plot_log (bool, optional): Whether to logarithm the resulting matrix. Defaults to False.

        Returns:
            numpy.array: A matrix with summed intensities of each spectrum.
        """
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

        if plot_log:
            showcopy = np.log(showcopy)

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

    def _plot_volcano(FcPvalGene, title, outfile=None, showGeneCount=30, showGene=None):
        """Fucntion that performs plotting of the volcano plot for plot_volcano() function.

        Args:
            FcPvalGene (list): List of tuples (fold change, p-value, identification mass or name)
            title (str): Title of the plot.
            outfile (str, optional): The path where to save the resulting plot. Defaults to None.
            showGeneCount (int, optional): Number of the most significantly up/dowm regulated genes. Defaults to 30.
            showGene (list, optional): A collection of strings that represent the desired gene names to be labled. Defaults to None.
        """
        color1 = "#883656"  #"#BA507A"
        color1_nosig = "#BA507A"
        color1_nosig_less = "#d087a4"
        color2 = "#4d6841"
        color2_nosig = "#70975E"
        color2_nosig_less = "#99b78b"
        colors = {"down": (color1, color1_nosig, color1_nosig_less), "up": (color2, color2_nosig,color2_nosig_less)}
        
        with plt.style.context("default"):
            plt.figure(figsize=(16,10))
            FcPvalGene = sorted(FcPvalGene, key=lambda x: x[1])
            xydots = [(x[0], -np.log10(x[1])) for x in FcPvalGene]
            maxally = max([x[1] for x in xydots if not np.isinf(x[1])])
            xydots = [(x, y if y <= maxally else maxally) for x,y in xydots]
            dotgene = [x[2] for x in FcPvalGene]
            pvalThresh = -np.log10(0.05)
            showGeneCount_pos = showGeneCount
            showGeneCount_neg = showGeneCount
            showGenes = []
            for x in FcPvalGene:
                gene = x[2]
                geneFC = x[0]
                if showGene:
                    if gene in showGene and showGeneCount_neg > 0:
                        showGenes.append(gene)
                        showGeneCount_neg -= 1
                    if gene in showGene and showGeneCount_pos > 0:
                        showGenes.append(gene)
                        showGeneCount_pos -= 1
                else:
                    if geneFC < 0 and showGeneCount_neg > 0:
                        showGenes.append(gene)
                        showGeneCount_neg -= 1
                    if geneFC > 0 and showGeneCount_pos > 0:
                        showGenes.append(gene)
                        showGeneCount_pos -= 1
            texts = []
            sel_down_xy = []
            nosig_down_xy = []
            nosigless_down_xy = []
            sel_up_xy = []
            nosig_up_xy = []
            nosigless_up_xy = []
            upregCount = 0
            downregCount = 0
            upregSigCount = 0
            downregSigCount = 0
            unregCount = 0
            for gi, (x,y) in enumerate(xydots):
                if x < 0:
                    if y < pvalThresh or abs(x) < 1:
                        downregCount += 1
                    else:
                        downregSigCount += 1
                elif x > 0:
                    if y < pvalThresh or abs(x) < 1:
                        upregCount += 1
                    else:
                        upregSigCount += 1
                elif x == 0:
                    unregCount += 1
                if dotgene[gi] in showGenes:
                    if x < 0:
                        sel_down_xy.append((x,y))
                    else:
                        sel_up_xy.append((x,y))
                    texts.append(plt.text(x * (1 + 0.01), y * (1 + 0.01) , dotgene[gi], fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)))
                else:
                    if x < 0:
                        if y < pvalThresh or abs(x) < 1:
                            nosigless_down_xy.append((x,y))
                        else:
                            nosig_down_xy.append((x,y))
                    else:
                        if y < pvalThresh or abs(x) < 1:
                            nosigless_up_xy.append((x,y))
                        else:
                            nosig_up_xy.append((x,y))
            #print(len(sel_xy), "of", len(genes))
            ymax = max([y for x,y in xydots])
            xmin = min([x for x,y in xydots])
            xmax = max([x for x,y in xydots])
            plt.plot([x[0] for x in nosigless_up_xy], [x[1] for x in nosigless_up_xy], '.', color=colors["up"][2])
            plt.plot([x[0] for x in nosigless_down_xy], [x[1] for x in nosigless_down_xy], '.', color=colors["down"][2])
            plt.plot([x[0] for x in nosig_up_xy], [x[1] for x in nosig_up_xy], 'o', color=colors["up"][1])
            plt.plot([x[0] for x in nosig_down_xy], [x[1] for x in nosig_down_xy], 'o', color=colors["down"][1])
            plt.plot([x[0] for x in sel_up_xy], [x[1] for x in sel_up_xy], 'o', color=colors["up"][0])
            plt.plot([x[0] for x in sel_down_xy], [x[1] for x in sel_down_xy], 'o', color=colors["down"][0])

            if plt.xlim()[0]<-0.5:
                plt.hlines(y=pvalThresh, xmin=plt.xlim()[0], xmax=-0.5, linestyle="dotted")
            if plt.xlim()[1]>0.5:
                plt.hlines(y=pvalThresh, xmin=0.5, xmax=plt.xlim()[1], linestyle="dotted")

            yMaxLim = plt.ylim()[1]
            plt.vlines(x=0.5, ymin=pvalThresh, ymax=yMaxLim, linestyle="dotted")
            plt.vlines(x=-0.5, ymin=pvalThresh, ymax=yMaxLim, linestyle="dotted")
            adjust_text(texts, force_points=0.2, force_text=0.2, expand_points=(2, 2), expand_text=(1, 1), arrowprops=dict(arrowstyle="-", color='black', lw=0.5))
            #        texts.append(plt.text(x * (1 + 0.01), y * (1 + 0.01) , dotgene[gi], fontsize=12))
            plt.title(title, fontsize = 40)
            plt.xlabel("logFC", fontsize = 32)
            plt.ylabel("Neg. Log. Adj. P-Value", fontsize = 32)
            plt.xticks(fontsize=14)
            infoText = "Total Genes: {}; Up-Reg. (sig.): {} ({}); Down-Reg. (sig.): {} ({}); Un-Reg.: {}".format(
                upregCount+downregCount+upregSigCount+downregSigCount+unregCount,
                upregCount, upregSigCount,
                downregCount, downregSigCount,
                unregCount
            )
            plt.figtext(0.5, 0.01, infoText, wrap=True, horizontalalignment='center', fontsize=14)
            if outfile:
                print(outfile)
                plt.savefig(outfile, bbox_inches="tight")

        


    def set_null_spectra(self, condition):
        """Goes thought the region array and sets the intensity values to zero if the condition is met.

        Args:
            condition (function): Condition of canceling out an intensity value.
        """
       #bar = progressbar.Bar()

        for i in range(0, self.region_array.shape[0]):#bar(range(0, self.region_array.shape[0])):
            for j in range(0, self.region_array.shape[1]):
                if condition(self.region_array[i,j, :]):

                    self.region_array[i,j,:] = 0


    def plot_segments(self, highlight=None):
        """Plots the segmented array of the current SpectraRegion object.

        Args:
            highlight (list/tuple/set/int, optional): If the highlight clusters are specified, those will be assigned a cluster id 2. Otherwise 1. Background stays 0. Defaults to None.
        """
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
        self.plot_array(fig, showcopy, discrete_legend=True)
        if not highlight is None and len(highlight) > 0:
            plt.title("Highlighted (yellow) clusters: {}".format(", ".join([str(x) for x in highlight])), y=1.08)
        plt.show()
        plt.close()

    def list_segment_counts(self):
        """Prints the size of each cluster in segmenetd array.
        """
        regionCounts = Counter()

        for i in range(0, self.segmented.shape[0]):
            for j in range(0, self.segmented.shape[1]):

                regionCounts[ self.segmented[i,j] ] += 1

        for region in natsorted([x for x in regionCounts]):
            print(region, ": ", regionCounts[region])


    def segment_clusterer(self, clusterer:RegionClusterer, verbose=False):

        segmentation = clusterer.segmentation()

        # segmentation has same dimensions as original array
        assert (segmentation.shape == (self.region_array.shape[0], self.region_array.shape[1]))

        self.segmented = segmentation
        self.segmented_method = clusterer.methodname()


    def segment(self, method="UPGMA", dims=None, number_of_regions=10, n_neighbors=10, min_cluster_size=20, num_samples=1000):
        """Performs clustering on similarity matrix.

        Args:
            method (str, optional): Clustering method: "UPGMA", "WPGMA", "WARD", "KMEANS", "UMAP_DBSCAN", "CENTROID", "MEDIAN", "UMAP_WARD", "DENSMAP_WARD" or "DENSMAP_DBSCAN". Defaults to "UPGMA".\n
                - "UPGMA": Unweighted pair group method with arithmetic mean.\n
                - "WPGMA": Weighted pair group method with arithmetic mean.\n
                - "WARD": Ward variance minimization algorithm.\n
                - "KMEANS": k-means++ clustering.\n
                - "UMAP_DBSCAN": Uniform Manifold Approximation and Projection for Dimension Reduction (UMAP) followed by Density-Based Spatial Clustering of Applications with Noise (DBSCAN).\n
                - "DENSMAP_DBSCAN": densMAP performs an optimization of the low dimensional representation followed by Density-Based Spatial Clustering of Applications with Noise (DBSCAN).\n
                - "CENTROID": Unweighted pair group method with centroids (UPGMC).\n
                - "MEDIAN": Weighted pair group method with centroids (WPGMC).\n
                - "UMAP_WARD": Uniform Manifold Approximation and Projection for Dimension Reduction (UMAP) followed by Ward variance minimization algorithm (WARD).\n
                - "DENSMAP_WARD": densMAP performs an optimization of the low dimensional representation followed by Ward variance minimization algorithm (WARD).\n
            dims ([type], optional): The desired amount of intesity values that will be taken into account performing dimension reduction. Defaults to None, meaning all intesities are considered.
            number_of_regions (int, optional): Number of desired clusters. Defaults to 10.
            n_neighbors (int, optional): The size of the local neighborhood (in terms of number of neighboring sample points) used for manifold approximation. For more information check UMAP documentation. Defaults to 10.
            min_cluster_size (int, optional): The minimum size of HDBSCAN clusters. Defaults to 20.
            num_samples (int, optional): Number of intensity values that will be used during HDBSCAN clustering. Defaults to 1000.

        Returns:
            numpy.array: An array with cluster ids as elements.
        """
        assert(method in ["UPGMA", "WPGMA", "WARD", "KMEANS", "UMAP_DBSCAN", "CENTROID", "MEDIAN", "UMAP_WARD", "DENSMAP_DBSCAN", "DENSMAP_WARD"])
        if method in ["UPGMA", "WPGMA", "WARD", "CENTROID", "MEDIAN"]:
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
            c = self.__segment__umap_hdbscan(number_of_regions, dims=dims, n_neighbors=n_neighbors, min_cluster_size=min_cluster_size, num_samples=num_samples)

        elif method == "UMAP_WARD":
            c = self.__segment__umap_ward(number_of_regions, dims=dims, n_neighbors=n_neighbors)

        elif method == "DENSMAP_DBSCAN":
            c = self.__segment__umap_hdbscan(number_of_regions, densmap=True, dims=dims, n_neighbors=n_neighbors)

        elif method == "DENSMAP_WARD":
            c = self.__segment__umap_ward(number_of_regions, densmap=True, dims=dims, n_neighbors=n_neighbors)

        elif method == "KMEANS":
            c = self.__segment__kmeans(number_of_regions)

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

    def manual_segmentation(self, image_path):
        """Plots the labeled array according to the given segmentation image.

        Args:
            image_path (str/numpy.array): Either path to the image file or the numpy array of the image.
        """
        if type(image_path) in [np.array]:
            self.logger.info("Received Image as Matrix")

            img = image_path

        else:

            img = skimage.io.imread(image_path)


        labeledArr, num_ids = ndimage.label(img, structure=np.ones((3,3)))

        plt.matshow(labeledArr)
        plt.show()


    def set_background(self, clusterIDs):
        """Sets all given cluster ids to 0, meaning the background.

        Args:
            clusterIDs (tuple/list/set/int): Cluster id(s) that form the background (cluster id 0).
        """
        if not type(clusterIDs) in [tuple, list, set]:
            clusterIDs = [clusterIDs]

        for clusterID in clusterIDs:
            self.segmented[ self.segmented == clusterID ] = 0


    def filter_clusters(self, method='remove_singleton', bg_x=4, bg_y=4, minIslandSize=10):
        """Filters the segmented array. 

        Args:
            method (str, optional): Possible methods: "remove_singleton", "most_similar_singleton", "merge_background", "remove_islands", "gauss". Defaults to 'remove_singleton'.\n
                - "remove_singleton": If there are clusters that include only one pixel, they will be made a part of the background.\n
                - "most_similar_singleton": If there are clusters that include only one pixel, they will be compared to consensus spectra of all cluster and then added to the cluster with the lowest distance.\n
                - "merge_background": Collects cluster ids at the borders and assigns all findings with background id 0.\n
                - "remove_islands": Removes all pixel groups that include less then minimum allowed elements.\n
                - "gauss": In case there is only two distinguishable cluster id in 3x3 area around the cluster will be assigned the most frequent cluster id of those two.\n
            bg_x (int, optional): The x border limits whithin the clusters whould be assigned to background. Defaults to 4.
            bg_y (int, optional): The y border limits whithin the clusters whould be assigned to background. Defaults to 4.
            minIslandSize (int, optional): How many pixels an island is allowed to have. Defaults to 10.

        Returns:
            numpy.array: Array with reduced number of cluster ids.
        """
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
        if self.segmented_method in ["UPGMA", "UMAP_DBSCAN", "UMAP_WARD", "DENSMAP_WARD", "DENSMAP_DBSCAN"]:
            self.dimred_labels = self.segmented.flatten()
        return self.segmented

    def __cons_spectra__avg(self, cluster2coords, array):
        """Constructs an average spectrum for each cluster id.

        Args:
            cluster2coords (dict): A dictionary of cluster ids mapped to the corresponding coordinates.
            array (numpy.array): Array of spectra.

        Returns:
            dict: A dictionary with cluster ids mapped to the respective average spectrum.
        """
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
        """Returns a dictionary of cluster ids mapped to the corresponding coordinates.

        Returns:
            dict: Each cluster ids mapped to the corresponding coordinates.
        """
        cluster2coords = defaultdict(list)

        for i in range(0, self.segmented.shape[0]):
            for j in range(0, self.segmented.shape[1]):

                clusterID = int(self.segmented[i, j])

                #if clusterID == 0:
                #    continue # unassigned cluster

                cluster2coords[clusterID].append((i,j))

        return cluster2coords


    def _get_median_spectrum(self, region_array):
        """Calculates the median spectrum from all spectra in region_array.

        Args:
            region_array (numpy.array): Array of spectra.

        Returns:
            numpy.array: An array where each element is a median value of all spectra at each specific m/z index.
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
        """Constructs an median spectrum for each cluster id.

        Args:
            cluster2coords (dict): A dictionary of cluster ids mapped to the corresponding coordinates.
            array (numpy.array, optional): Array of spectra. Defaults to None, that means using the region_array of the object.

        Returns:
            dict: A dictionary with cluster ids mapped to the respective median spectrum.
        """
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
        """Constructs a consensus spectrum for each cluster id by using the specified method.

        Args:
            method (str, optional): Method that is supposed to be used for consensus spectra calculation. Either "avg" (average) or "median". Defaults to "avg".
            set_consensus (bool, optional): Whether to set the calculated consensus and the respective method as object attributes. Defaults to True.
            array (numpy.array, optional): Array of spectra. Defaults to None, that means using the region_array of the object.

        Returns:
            dict: A dictionary with cluster ids mapped to the respective consensus spectrum.
        """
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
        """Plots seaborn.boxplot depicting the range of intensity values of each desired mass within each cluster. Additionally, plots mean difference effect sizes with help of the DABEST package. The given cluster id is considered a control group.

        Args:
            masses (float/list/tuple/set): A desired mass or collection of masses.
            background (int, optional): Cluster id of the background. Defaults to 0.
        """
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
            plt.title("Intensities per cluster for {}m/z".format(",".join([str(x) for x in masses])))

            plt.xticks(rotation=90)
            plt.show()
            plt.close()

            dfobj_db = dfObj.pivot(index="specidx", columns='cluster', values='intensity')

            allClusterIDs = natsorted([x for x in set(clusterVec) if not " {}".format(background) in x])
            
            multi_groups = dabest.load(dfobj_db, idx=tuple(["Cluster {}".format(background)]+allClusterIDs))
            dabestFig = multi_groups.mean_diff.plot()
            dabestFig.suptitle("DABEST intensities per cluster for {}m/z".format(",".join([str(x) for x in masses])))

    def plot_inter_consensus_similarity(self, clusters=None):
        """Plots seaborn.boxplot depicting the cosine similarity distributions by comparison of spectra belonging to specified cluster ids to all available clusters.

        Args:
            clusters (numpy.array/list, optional): A list of desired cluster ids. Defaults to None, meaning to include all available clusters.
        """
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
        """Plots the similarity matrix either represented as a heatmap of similarity matrix or as seaborn.boxplot depicting similarity distributions of similarity values within the clusters.

        Args:
            mode (str, optional): Either "heatmap" or "spectra". Defaults to "heatmap".
        """
        assert(not self.consensus_similarity_matrix is None)

        assert(mode in ["heatmap", "spectra"])

        if mode == "heatmap":
            allLabels = [''] + sorted([x for x in self.consensus])
            
            heatmap = plt.matshow(self.consensus_similarity_matrix)
            plt.gca().set_xticklabels( allLabels )
            plt.gca().set_yticklabels( allLabels )

            plt.colorbar(heatmap)
            plt.show()
            plt.close()

        elif mode == "spectra":
            
            cluster2coords = self.getCoordsForSegmented()
            clusterLabels = sorted([x for x in cluster2coords])

            self.logger.info("Found clusterLabels {}".format(clusterLabels))

            clusterSimilarities = {}

            for clusterLabel in clusterLabels:
                self.logger.info("Processing clusterLabel {}".format(clusterLabel))

                clusterSims = []
                useCoords = cluster2coords[clusterLabel]
                for i in range(0, len(useCoords)):
                    for j in range(i+1, len(useCoords)):
                        iIdx = self.pixel2idx[useCoords[i]]
                        jIdx = self.pixel2idx[useCoords[j]]
                        
                        sim = self.spectra_similarity[iIdx, jIdx]
                        clusterSims.append(sim)

                clusterSimilarities[clusterLabel] = clusterSims

                #allSpectra = [ self.region_array[xy[0], xy[1], :] for xy in  cluster2coords[clusterLabel]]             
                #bar = progressbar.ProgressBar()
                #clusterSims = []
                #for i in bar(range(0, len(allSpectra))):
                #    for j in range(i+1, len(allSpectra)):
                #        clusterSims.append( self.__get_spectra_similarity(allSpectra[i], allSpectra[j]) )
                #clusterSimilarities[clusterLabel] = clusterSims

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
        """Calculates cosine similarity between two vectors of the same length.

        Args:
            vA (numpy.array/list): First vector.
            vB (numpy.array/list): Second vector.

        Returns:
            float: cosine similarity.
        """
        return np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))

    def consensus_similarity( self ):
        """
        Updates consensus_similarity_matrix attribute of SpectraRegion object. The updated matrix consists of similarity values between the spectra in the consensus dictionary.
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
        """Gives an expression (intensity values) overview of the given mass in the region.

        Args:
            massValue (float): A desired mass.
            segments (numpy.array/list/tuple/set/int): Desired cluster id(s).
            mode (numpy.array/list/tuple/set/str, optional): Whether to calculate the average and/or median value of the found expression values. Defaults to "avg".

        Returns:
            tuple: the first element consists of value(s) calculated with specified mode(s), number of found expression values, number of found expression values that differ from 0.
        """
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

    def get_spectra_matrix(self, segments):
        """Returns a matrix with all spectra in .imzML file that correspond to the given segments.

        Args:
            segments (numpy.array/list): A list of desired cluster ids.

        Returns:
            numpy.array: An array where each element is spectrum that was previously found to be part of one of the given clusters given in segments parameter.
        """
        cluster2coords = self.getCoordsForSegmented()

        relPixels = []
        for x in segments:
            relPixels += cluster2coords.get(x, [])

        spectraMatrix = np.zeros((len(relPixels), len(self.idx2mass)))

        for pidx, px in enumerate(relPixels):
            spectraMatrix[pidx, :] = self.region_array[px[0], px[1], :]

        return spectraMatrix


    def get_expression_from_matrix(self, matrix, massValue, segments, mode="avg"):
        """Gives an expression (intensity values) overview of the given matrix.

        Args:
            matrix (numpy.array): A matrix from which the intensity values will be extracted.
            massValue (float): A desired mass.
            segments (numpy.array/list/tuple/set/int): Desired cluster id(s).
            mode (numpy.array/list/tuple/set/str, optional): Whether to calculate the average and/or median value of the found expression values. "arg" (average) and/or "median". Defaults to "avg".

        Returns:
            tuple: the first element consists of value(s) calculated with specified mode(s), number of found expression values, number of found expression values that differ from 0.
        """
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

    def __check_neighbour(self, mat, x, y, background):
        """Decides whether the given pixel suppose to be a part of the background cluster parameter.

        Args:
            mat (numpy.array): CLustered image with cluster ids as elements.
            x (int): x-Coordinate.
            y (int): y-Coordinate.
            background (int): Cluster id of the cluster to be compared to.

        Returns:
            bool: Whether the pixel is neighbor of the cluster in the background parameter.
        """
        if x < mat.shape[0]-1 and mat[x+1][y] in background:
            return True
        elif x > 1 and mat[x-1][y] in background:
            return True
        elif y < mat.shape[1]-1 and mat[x][y+1] in background:
            return True
        elif y > 1 and mat[x][y-1] in background:
            return True
        elif x < mat.shape[0]-1 and y < mat.shape[1]-1 and mat[x+1][y+1] in background:
            return True
        elif x > 1 and y > 1 and mat[x-1][y-1] in background:
            return True
        elif x < mat.shape[0]-1 and y > 1 and mat[x+1][y-1] in background:
            return True
        elif y < mat.shape[1]-1 and x > 1 and mat[x-1][y+1] in background:
            return True
        else:
            return False


    def cartoonize(self, background, aorta, plaque, blur=False):
        """Simplifies the clustered image.

        Args:
            background (list/numpy.array): A list of clusters id that contains background clusters.
            aorta (list/numpy.array): A list of clusters id that contains aorta clusters.
            plaque (list/numpy.array): A list of clusters id that contains plaque clusters.
            blur (bool, optional): Whether to apply a multidimensional uniform filter to blur the image. Defaults to False.

        Returns:
            numpy.array: Simplified image with three clusters.
        """
        assert(not self.segmented is None)

        img = np.copy(self.segmented)
        cartoon_img = np.zeros((img.shape))

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j] in background:
                    cartoon_img[i,j] = 0
                elif img[i,j] in aorta or self.__check_neighbour(img, i, j, background):
                    cartoon_img[i,j] = 1
                else:
                    if not self.__check_neighbour(img, i, j, background) and self.__check_neighbour(cartoon_img, i, j, aorta) or img[i][j] in plaque:
                        cartoon_img[i,j] = 2
                    else: 
                        cartoon_img[i,j]  = 0
        if blur:
            cartoon_img = ndimage.uniform_filter(cartoon_img, size=4)
        return cartoon_img

    def __merge_clusters(self, matrix, clusters):
        """Combines the given clusters to one cluster with the id of the first element in the clusters list.

        Args:
            matrix (numpy.array): Segmented array with cluster ids as elements.
            clusters (list/numpy.array): Cluster ids to merge.

        Returns:
            numpy.array: Updated segmeted array.
        """
        merged = np.copy(self.segmented)
        for cluster in clusters:
            merged[merged == cluster] = clusters[0]
        return merged

    def cartoonize2(self, imze, background, aorta, plaque, ignore_background=True, blur=False):
        """Simplifies the segmented array by comparing median spectra of the given cluster groups to the whole spectra region.

        Args:
            imze (IMZMLExtract): IMZMLExtract object.
            background (list/numpy.array): A list of clusters id that contains background clusters.
            aorta (list/numpy.array): A list of clusters id that contains aorta clusters.
            plaque (list/numpy.array): A list of clusters id that contains plaque clusters.
            ignore_background (bool, optional): Whether to consider only aorta and plaque median spectra by "reclustering". Defaults to True.
            blur (bool, optional): Whether to apply a multidimensional uniform filter to blur the image. Defaults to False.

        Returns:
            numpy.array: Simplified segmented array with only three clusters.
        """
        assert(not self.segmented is None)

        merged = self.segmented
        if len(background)>1:
            merged = self.__merge_clusters(merged, background)
        if len(aorta)>1:
            merged = self.__merge_clusters(merged, aorta)
        if len(plaque)>1:
            merged = self.__merge_clusters(merged, plaque)
            
        background = background[0]
        aorta = aorta[0]
        plaque = plaque[0]
        
        tmp = self.segmented
        self.segmented = merged
        cons = self.consensus_spectra(method="median", set_consensus=False)
        self.segmented = tmp
        
        sim_background = np.zeros(self.segmented.shape)
        sim_aorta = np.zeros(self.segmented.shape)
        sim_plaque = np.zeros(self.segmented.shape)
        
        for i in range(sim_background.shape[0]):
            for j in range(sim_background.shape[1]):
                spectra = self.region_array[i][j]
                sim_background[i][j] = imze.compare_sequence(spectra, cons[background])
                sim_aorta[i][j] = imze.compare_sequence(spectra, cons[aorta])
                sim_plaque[i][j] = imze.compare_sequence(spectra, cons[plaque])
        
        sim_max = np.zeros((sim_background.shape[0], sim_background.shape[1]))
        for i in range(sim_background.shape[0]):
            for j in range(sim_background.shape[1]):
                if ignore_background and self.segmented[i][j] == background:
                    sim_max[i,j] = self.segmented[i][j] 
                else:
                    sim_max[i,j] = np.argmax([sim_background[i][j], sim_aorta[i][j], sim_plaque[i][j]])
        if blur:
            sim_max = ndimage.uniform_filter(sim_max, size=4)
        
        return sim_max

    def get_surroundings(self, mat, x, y):
        """Determines the cluster ids and their frequencies of the 3x3 surroundings of the given pixel.

        Args:
            mat (numpy.array): The matrix where the surrounding pixels will be computed.
            x (int): x-Coordinate of the desired pixel.
            y (int): y-Coordinate of the desired pixel.

        Returns:
            collections.Counter: Cluster ids and the respective frequencies in 3x3 window from the given pixel coordinates.
        """
        res = list()
        if x < mat.shape[0]-1:
            res.append(mat[x+1][y])
        if x > 1:
            res.append(mat[x-1][y])
        if y < mat.shape[1]-1:
            res.append(mat[x][y+1])
        if y > 1:
            res.append(mat[x][y-1])
        if x < mat.shape[0]-1 and y < mat.shape[1]-1:
            res.append(mat[x+1][y+1])
        if x > 1 and y > 1:
            res.append(mat[x-1][y-1])
        if x < mat.shape[0]-1 and y > 1:
            res.append(mat[x+1][y-1])
        if y < mat.shape[1]-1 and x > 1:
            res.append(mat[x-1][y+1])
        return Counter(res)

    def add_cellwall(self, mat, between1, between2, threshold = 2):
        """Adds the cluster for the cell wall at those pixels that have significant number of between1 and between2 assigned pixels.

        Args:
            mat (numpy.array): A segmented array with a clustered image where the cell wall cluster should be added.
            between1 (int): First cluster id of the selected cluster pair. Between those the new cluster will be added.
            between2 (int): Second cluster id of the selected cluster pair. Between those the new cluster will be added.
            threshold (int, optional): The minimal number of between1 and between2 neighboring clusters for each pixel to be considered as a cell wall component. Defaults to 2.

        Returns:
            numpy.array: Updated segmented array where the cell wall cluster has cluster id 3.
        """
        new_mat = np.copy(mat)
        new_mat = new_mat+1
        new_mat[new_mat==1] = 0
        for i in range(new_mat.shape[0]):
            for j in range(new_mat.shape[1]):
                s = self.get_surroundings(mat, i, j)
                if s[between1] > threshold and s[between2] > threshold:
                    new_mat[i][j] = 1
        return new_mat

    def plot_wireframe(self, imze, background, aorta, plaque, norm=False):
        """Plots the background, aorta, and plaque pixelwise probabilities.

        Args:
            imze (IMZMLExtract): IMZMLExtract object.
            background (list/numpy.array): A list of clusters id that contains background clusters.
            aorta (list/numpy.array): A list of clusters id that contains aorta clusters.
            plaque (list/numpy.array): A list of clusters id that contains plaque clusters.
            norm (bool, optional): Whether to divide all probabilities by global maximum probability. Defaults to False.

        Returns:
            numpy.array, numpy.array, numpy.array: Three arrays of probabilities for background, aorta, and plaque.
        """
        out = self.cartoonize(background, aorta, plaque, blur=False)
        tmp = self.segmented
        self.segmented = out
        cons = self.consensus_spectra(method="median", set_consensus=False)
        self.segmented = tmp
        sim_background = np.zeros(out.shape)
        sim_aorta = np.zeros(out.shape)
        sim_plaque = np.zeros(out.shape)

        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                spectra = self.region_array[i][j]
                sim_background[i][j] = imze.compare_sequence(spectra, cons[0])
                sim_aorta[i][j] = imze.compare_sequence(spectra, cons[1])
                sim_plaque[i][j] = imze.compare_sequence(spectra, cons[2])

        if norm:
            sim_background = np.sqrt(sim_background)
            sim_aorta = np.sqrt(sim_aorta)
            sim_plaque = np.sqrt(sim_plaque)
            
        (X, Y) = np.meshgrid(np.arange(self.segmented.shape[1]), np.arange(self.segmented.shape[0]))

        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot a basic wireframe.
        ax.plot_wireframe(X, Y, sim_background, color='green', label='Background')
        ax.plot_wireframe(X, Y, sim_aorta, color='red', label='Aorta')
        ax.plot_wireframe(X, Y, sim_plaque, label='Plaque')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('P')
        ax.legend()

        plt.show()

        return sim_background, sim_aorta, sim_plaque


    def _makeHTMLStringFilterTable(self, expDF):
        """Transform given pandas dataframe into HTML output.

        Args:
            expDF (pd.DataFrame): Values for output.

        Returns:
            htmlHead, htmlBody (str): HTML code for head and body.
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
        """Returns updated segmented shaped matrix with the desired region coordinates replaced with ones.

        Args:
            regions (list/tuple/set/int): A desired region or collection of cluster ids.

        Returns:
            numpy.array: An updated matrix with all specified cluster ids replaced with cluster id equals 1.
        """
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

        #
        # Plot segments
        #
        #
        valid_vals = np.unique(self.segmented)
        heatmap = plt.matshow(self.segmented, cmap=plt.cm.get_cmap('viridis', len(valid_vals)), fignum=100)
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

        pic_IObytes = io.BytesIO()
        plt.savefig(pic_IObytes,  format='png')
        pic_IObytes.seek(0)
        pic_hash = base64.b64encode(pic_IObytes.read()).decode()
        plt.close(100)

        imgStrSegments = "<img src='data:image/png;base64,{}' alt='Red dot' />".format(pic_hash)

        #
        # Plot segment highlights
        #
        #
        showcopy = np.copy(self.segmented)

        for i in range(0, showcopy.shape[0]):
            for j in range(0, showcopy.shape[1]):
                if showcopy[i,j] != 0:
                    if showcopy[i,j] in resKey[0]:
                        showcopy[i,j] = 2
                    elif showcopy[i,j] != 0:
                        showcopy[i,j] = 1

        valid_vals = np.unique(showcopy)
        heatmap = plt.matshow(showcopy, cmap=plt.cm.get_cmap('viridis', len(valid_vals)), fignum=100)
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

        pic_IObytes = io.BytesIO()
        plt.savefig(pic_IObytes,  format='png')
        pic_IObytes.seek(0)
        pic_hash = base64.b64encode(pic_IObytes.read()).decode()
        plt.close(100)

        imgStrSegmentHighlight = "<img src='data:image/png;base64,{}' alt='Red dot' />".format(pic_hash)

        bodypart = "<p>{}<p><p>{}<p>\n{}".format(imgStrSegments, imgStrSegmentHighlight, bodypart)

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




    def deres_to_df(self, method, resKey, protWeights, mz_dist=3, mz_best=False, keepOnlyProteins=True, inverse_fc=False, max_adj_pval=0.05, min_log2fc=0.5):
        """Transforms differetial expression (de) result in de_results_all dictionary of the SpectraRegion object into a DataFrame.

        Args:
            method (str): Test method for differential expression. "empire", "ttest" or "rank".
            resKey (tuple): List of regions where to look for the result.
            protWeights (ProteinWeights): ProteinWeights object for translation of masses to protein name.
            mz_dist (float/int, optional): Allowed offset for protein lookup of needed masses. Defaults to 3.
            mz_best (bool, optional): Wether to consider only the closest found protein within mz_dist (with the least absolute mass difference). Defaults to False.
            keepOnlyProteins (bool, optional): If True, differential masses without protein name will be removed. Defaults to True.
            inverse_fc (bool, optional): If True, the de result logFC will be inversed (negated). Defaults to False.
            max_adj_pval (float, optional): Threshold for maximum adjusted p-value that will be used for filtering of the de results. Defaults to 0.05.
            min_log2fc (float, optional): Threshold for minimum log2fc that will be used for filtering of the de results. Defaults to 0.5.

        Returns:
            pandas.DataFrame: DataFrame of differetial expression (de) result.
        """
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
                foundProt = protWeights.get_protein_from_mass(massValue, maxdist=mz_dist)

                if mz_best and len(foundProt) > 0:
                    foundProt = [foundProt[0]]

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


    def find_all_markers(self, protWeights, keepOnlyProteins=True, replaceExisting=False, includeBackground=True, mz_dist=3, mz_best=False, backgroundCluster=[0], out_prefix="nldiffreg", outdirectory=None, use_methods = ["empire", "ttest", "rank"], count_scale={"ttest": 1, "rank": 1, "empire": 10000}):
        """Finds all marker proteins for a specific clustering.

        Args:
            protWeights (ProteinWeights): ProteinWeights object for translation of masses to protein name.
            keepOnlyProteins (bool, optional): If True, differential masses without protein name will be removed. Defaults to True.
            replaceExisting (bool, optional): If True, previously created marker-gene results will be overwritten. Defaults to False.
            includeBackground (bool, optional): If True, the cluster specific expression data are compared to all other clusters incl. background cluster. Defaults to True.
            mz_dist (float/int, optional): Allowed offset for protein lookup of needed masses. Defaults to 3.
            mz_best (bool, optional): Wether to consider only the closest found protein within mz_dist (with the least absolute mass difference). Defaults to False.
            backgroundCluster ([int], optional): Clusters which are handled as background. Defaults to [0].
            out_prefix (str, optional): Prefix for results file. Defaults to "nldiffreg".
            outdirectory ([type], optional): Directory used for empire files. Defaults to None.
            use_methods (list, optional): Test methods for differential expression. Defaults to ["empire", "ttest", "rank"].\n
                - "empire": Empirical and Replicate based statistics (EmpiRe).\n
                - "ttest": Welch’s t-test for differential expression using diffxpy.api.\n
                - "rank": Mann-Whitney rank test (Wilcoxon rank-sum test) for differential expression using diffxpy.api.\n
            count_scale (dict, optional): Count scales for different methods (relevant for empire, which can only use integer counts). Defaults to {"ttest": 1, "rank": 1, "empire": 10000}.

        Returns:
            dict of pd.dataframe: for each test conducted, one data frame with all marker masses for each cluster
        """
        cluster2coords = self.getCoordsForSegmented()

        dfbyMethod = defaultdict(lambda: pd.DataFrame())

        for segment in cluster2coords:

            if not includeBackground and segment in backgroundCluster:
                continue

            clusters0 = [segment]
            clusters1 = [x for x in cluster2coords if not x in clusters0]

            if not includeBackground:
                for bgCluster in backgroundCluster:
                    if bgCluster in clusters1:
                        del clusters1[clusters1.index(bgCluster)]

            self.find_markers(clusters0=clusters0, clusters1=clusters1, replaceExisting=replaceExisting, outdirectory=outdirectory, out_prefix=out_prefix, use_methods=use_methods, count_scale=count_scale)

            # get result
            resKey = self.__make_de_res_key(clusters0, clusters1)

            keyResults = self.get_de_results(resKey)

            for method in keyResults:
                methodKeyDF = self.get_de_result(method, resKey)

                inverseFC = False
                if method in ["ttest", "rank"]:
                    inverseFC = True

                resDF = self.deres_to_df(method, resKey, protWeights, keepOnlyProteins=keepOnlyProteins, inverse_fc=inverseFC, mz_dist=mz_dist, mz_best=mz_best)

                dfbyMethod[method] = pd.concat([dfbyMethod[method], resDF], sort=False)

        return dfbyMethod

                    

    def __make_de_res_key(self, clusters0, clusters1):
        """Generates the storage key for two sets of clusters.

        Args:
            clusters0 (list): list of cluster ids 1.
            clusters1 (list): list of cluster ids 2.

        Returns:
            tuple: tuple of both sorted cluster ids, as tuple.
        """

        return (tuple(sorted(clusters0)), tuple(sorted(clusters1)))
        

    def clear_de_results(self):
        """Removes all sotred differential expression results.
        """
        self.de_results_all = defaultdict(lambda: dict())
        self.df_results_all = defaultdict(lambda: dict())

    def run_nlempire(self, nlDir, pdata, pdataPath, diffOutput):
        """Performs Empirical and Replicate based statistics (EmpiRe).

        Args:
            nlDir (str): The path to the desired output directory.
            pdata (DataFrame): Phenotype data.
            pdataPath (str): The path to the saved .tsv file with phenotype data.
            diffOutput (str): The path where to save the .tsv output file.
        """
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
        """Finds marker proteins for a specific clustering.

        Args:
            clusters0 (int/list/tuple/set): Cluster id(s) that will be labeled as condition 0.
            clusters1 (list/tuple/set, optional): Cluster id(s) that will be labeled as condition 1. Defaults to None, meaning all clusters that are not in clusters0, belong to clusters1.
            out_prefix (str, optional): If using "empire" method, the desired prefix of the newly generated files can be specified. Defaults to "nldiffreg".
            outdirectory (str, optional): If using "empire" method, the path to the desired output directory must be specified. Defaults to None.
            replaceExisting (bool, optional): If True, previously created marker-gene results will be overwritten. Defaults to False.
            use_methods (list, optional): Test methods for differential expression. Defaults to ["empire", "ttest", "rank"].\n
                - "empire": Empirical and Replicate based statistics (EmpiRe).\n
                - "ttest": Welch’s t-test for differential expression using diffxpy.api.\n
                - "rank": Mann-Whitney rank test (Wilcoxon rank-sum test) for differential expression using diffxpy.api.\n
            count_scale (dict, optional): Count scales for different methods (relevant for empire, which can only use integer counts). Defaults to {"ttest": 1, "rank": 1, "empire": 10000}.
            sample_max (int, optional): The size of the sampled list of spectra. Defaults to -1, meaning all are considered.

        Returns:
            tuple: DataFrame object containing expression data, DataFrame object containing phenotype data (condition 0/1), DataFrame object containing feature data (masses)
        """
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

            bar = makeProgressBar()
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
            
            bar = makeProgressBar()
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
        """Transforms a dictionary of the differential expression results into a list.

        Returns:
            list: A list of tuples where the first element is the name of the used method and the second - all compared sets of clusters.
        """
        allDERes = []
        for x in self.de_results_all:
            for y in self.de_results_all[x]:
                allDERes.append((x,y))

        return allDERes

    def find_de_results(self, keypart):
        """Finds all occurrences of the desired set of clusters.

        Args:
            keypart (tuple): Desired set of clusters to find.

        Returns:
            list: A list of tuples where the first element is the name of the used method followed by all occurrences that include the specified set of clusters.
        """
        results = []
        for method in self.de_results_all:
            for key in self.de_results_all[method]:

                if keypart == key[0] or keypart == key[1]:
                    results.append( (method, key) )

        return results



    def get_de_results(self, key):
        """Finds exactly those differential expression results that correspond to the given cluster sets pair.

        Args:
            key (tuple): A tuple of two tuples each consisting of cluster ids compared.

        Returns:
            dict: name of the used methods as keys mapped to the respective results.
        """
        results = {}
        for method in self.de_results_all:
            if key in self.de_results_all[method]:
                results[method] = self.de_results_all[method][key]

        return results

    def get_de_result(self, method, key):
        """Finds differential expression result of exact given cluster sets pair using the specified method.

        Args:
            method (str): Either "empire", "ttest" or "rank".
            key (tuple): A tuple of two tuples each consisting of cluster ids compared.

        Returns:
            pandas.DataFrame: differential expression data of the given case.
        """
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

        if not logging.getLogger().hasHandlers():

            consoleHandler = logging.StreamHandler()
            consoleHandler.setLevel(logging.INFO)

            self.logger.addHandler(consoleHandler)

            formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
            consoleHandler.setFormatter(formatter)

            self.logger.info("Added new Stream Handler")

    def __init__(self, filename, min_mass=-1, max_mass=-1):
        """Creates a ProteinWeights class. Requires a formatted proteinweights-file.

        Args:
            filename (str): File with at least the following columns: protein_id, gene_symbol, mol_weight_kd, mol_weight.
            max_mass (float): Maximal mass to consider/include in object. -1 for no filtering. Masses above threshold will be discarded. Default is -1.
            max_mass (float): Minimal mass to consider/include in object. -1 for no filtering. Masses below threshold will be discarded. Default is -1.
        """

        self.__set_logger()

        self.protein2mass = defaultdict(set)
        self.category2id = defaultdict(set)
        self.protein_name2id = {}

        if filename != None:

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

                    if min_mass >= 0 and molWeight < min_mass:
                        continue

                    if len(proteinNames) == 0:
                        proteinNames = proteinIDs

                    for proteinName in proteinNames:
                        self.protein2mass[proteinName].add(molWeight)
                        self.protein_name2id[proteinName] = proteinIDs

            allMasses = self.get_all_masses()
            self.logger.info("Loaded a total of {} proteins with {} masses".format(len(self.protein2mass), len(allMasses)))
    

    @classmethod
    def from_sdf(cls, filename, max_mass=-1):
        """Creates a ProteinWeights class. Requires a .sdf file.

        Args:
            filename (str): .sdf file with a starting < LM_ID > line of each new entry.
        """
        assert(filename.endswith(".sdf"))

        pw = ProteinWeights(None)

        pw.protein2mass = defaultdict(set)
        pw.protein_name2id = {}
        pw.category2id = defaultdict(set)

        sdf_dic = cls.sdf_reader(filename)

        for lm_id in sdf_dic:

            molWeight = float(sdf_dic[lm_id]["EXACT_MASS"])
            if max_mass >= 0 and molWeight > max_mass:
                continue    

            pw.protein2mass[lm_id].add(molWeight)

            if "NAME" in sdf_dic[lm_id]:
                pw.protein_name2id[sdf_dic[lm_id]["NAME"]] = lm_id
            elif "SYSTEMATIC_NAME" in sdf_dic[lm_id]:
                pw.protein_name2id[sdf_dic[lm_id]["SYSTEMATIC_NAME"]] = lm_id

            pw.category2id[sdf_dic[lm_id]["MAIN_CLASS"]].add(lm_id)#"CATEGORY"

        return pw, sdf_dic

    @classmethod
    def sdf_reader(cls, filename):
        """Reads a .sdf file into a dictionary.

        Args:
            filename (str): Path to the .sdf file.

        Returns:
            dict: Ids mapped to the respective annotation. 
        """
        res_dict = {}
        with open(filename) as fp:
            line = fp.readline()
            line_id = ""
            line_dict = {}
            while line:
                if line.startswith(">"):
                    if "LM_ID" in line:
                        if line_id:
                            res_dict[line_id] = line_dict
                            line_dict = {}
                            line_id = ""
                        line_id = fp.readline().rstrip()
                    else:
                        key = line.split("<")[1].split(">")[0]
                        line_dict[key] = fp.readline().rstrip()
                line = fp.readline()

        fp.close()
        return res_dict

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

        self.logger.info("         Number of total protein/genes: {}".format(len(self.protein2mass)))
        self.logger.info("           Number of total masses: {}".format(len(mass2prot)))
        self.logger.info("Number of proteins/genes with collision: {}".format(len(protsWithCollision)))
        self.logger.info("        Mean Number of collisions: {}".format(np.mean([protsWithCollision[x] for x in protsWithCollision])))
        self.logger.info("      Median Number of collisions: {}".format(np.median([protsWithCollision[x] for x in protsWithCollision])))

        if print_proteins:
            self.logger.info("Proteins/genes with collision: {}".format([x for x in protsWithCollision]))
        else:
            self.logger.info("Proteins/genes with collision: {}".format(protsWithCollision.most_common(10)))
        

    def get_protein_from_mass(self, mass, maxdist=2):
        """Searches all recorded mass and proteins and reports all proteins which have at least one mass in (mass-maxdist, mass+maxdist) range.

        Args:
            mass (float): mass to search for
            maxdist (float, optional): allowed offset for lookup. Defaults to 2.

        Returns:
            list: sorted list (by abs mass difference) of all (protein, weight) tuple which have a protein in the given mass range
        """

        possibleMatches = []

        for protein in self.protein2mass:
            protMasses = self.protein2mass[protein]

            for protMass in protMasses:
                if abs(mass-protMass) < maxdist:
                    protDist = abs(mass-protMass)
                    possibleMatches.append((protein, protMass, protDist))

        possibleMatches = sorted(possibleMatches, key=lambda x: x[2])
        possibleMatches = [(x[0], x[1]) for x in possibleMatches]

        return possibleMatches

    def get_masses_for_protein(self, protein):
        """Returns all recorded masses for a given protein. Return None if protein not found

        Args:
            protein (str|iterable): protein to search for in database (exact matching)

        Returns:
            set: set of masses for protein
        """

        if type(protein) == str:
            return self.protein2mass.get(protein, None)

        allMasses = set()
        for x in protein:
            protMasses = self.get_masses_for_protein(x)
            allMasses = allMasses.union(protMasses)

        return allMasses

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

            


