import numpy as np
from scipy import misc
import ctypes
import matplotlib.pyplot as plt
import os,sys
import imageio
from PIL import Image

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

    def __init__(self, region_array):
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

        for i in range(self.region_array.shape[0]*self.region_array.shape[1]):

            x,y = divmod(i, self.region_array.shape[1])

            self.idx2pixel[i] = (x,y)
            self.pixel2idx[(x,y)] = i

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

        print("dimensions", dims)

        self.obj = lib.StatisticalRegionMerging_New(dims, qArr, len(qs))
        self.logger.info("Switching to dot mode")
        lib.StatisticalRegionMerging_mode_dot(self.obj)

        image_p = inputarray.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.logger.info("Starting calc similarity c++")
        retValues = lib.SRM_calc_similarity(self.obj, inputarray.shape[0], inputarray.shape[1], image_p)

        outclust = np.ctypeslib.as_array(retValues, shape=(inputarray.shape[0] * inputarray.shape[1], inputarray.shape[0] * inputarray.shape[1]))

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
        Z = spc.hierarchy.linkage(squareform(self.spectra_similarity), method='average', metric='cosine')
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')
        return c

    def __segment__wpgma(self, number_of_regions):
        Z = spc.hierarchy.weighted(squareform(self.spectra_similarity))
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')
        return c

    def __segment__ward(self, number_of_regions):
        Z = spc.hierarchy.ward(squareform(self.spectra_similarity))
        c = spc.hierarchy.fcluster(Z, t=number_of_regions, criterion='maxclust')
        return c

    def __segment__umap_hdbscan(self, number_of_regions):

        self.dimred_elem_matrix = np.zeros((self.region_array.shape[0]*self.region_array.shape[1], self.region_array.shape[2]))

        """
        
        ----------> spectra ids
        |
        |
        | m/z values
        |
        v
        
        """

        for i in range(0, self.region_array.shape[0]):
            for j in range(0, self.region_array.shape[1]):
                self.elem_matrix[i * self.region_array.shape[1] + j, :] = self.region_array[i,j,:]

        self.dimred_elem_matrix = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=40,
            random_state=42,
        ).fit_transform(self.elem_matrix)

        self.dimred_labels = hdbscan.HDBSCAN(
            min_samples=10,
            min_cluster_size=500,
        ).fit_predict(self.dimred_elem_matrix)

        return self.dimred_labels

    def vis_umap(self):

        assert(self.elem_matrix != None)
        assert(self.dimred_labels != None)

        dimred_2d = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        ).fit_transform(self.elem_matrix)
        labels = hdbscan.HDBSCAN(
            min_samples=10,
            min_cluster_size=500,
        ).fit_predict(dimred_2d)

        plt.figure()
        clustered = (labels >= 0)
        plt.scatter(dimred_2d[~clustered, 0],
                    dimred_2d[~clustered, 1],
                    c=(0.5, 0.5, 0.5),
                    s=0.1,
                    alpha=0.5)
        plt.scatter(dimred_2d[clustered, 0],
                    dimred_2d[clustered, 1],
                    c=labels[clustered],
                    s=0.1,
                    cmap='Spectral')

        plt.show()
        plt.close()

    def segment(self, method="UPGMA", number_of_regions=10):

        assert(self.spectra_similarity != None)
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

        image_UPGMA = np.zeros(self.region_array.shape)


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

        assert(method in ["remove_singleton", "most_similar_singleton"])

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

            cons_spectra[clusID] = avgSpectrum

        return cons_spectra


    def consensus_spectra(self, method="avg"):

        assert(self.segmented != None)
        assert(method in ["avg"])

        self.logger.info("Calculating consensus spectra")

        cluster2coords = defaultdict(list)

        for i in range(0, self.segmented.shape[0]):
            for j in range(0, self.segmented.shape[1]):

                clusterID = self.segmented[i, j]

                if clusterID == 0:
                    continue # unassigned cluster

                cluster2coords[clusterID].append((i,j))


        cons_spectra = None
        if method == "avg":
            cons_spectra = self.__cons_spectra__avg(cluster2coords)


        self.consensus = cons_spectra
        self.consensus_method = method
        self.logger.info("Calculating consensus spectra done")

        return self.segmented



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



    def get_region_array(self, regionid):



        xr,yr,zr,sc = self.get_region_range(regionid)
        rs = self.get_region_shape(regionid)
        print(rs)

        sarray = np.zeros( rs, dtype=np.float32 )

        coord2spec = self.get_region_spectra(regionid)

        for coord in coord2spec:
            xpos = coord[0]-xr[0]
            ypos = coord[1]-yr[0]

            spectra = coord2spec[coord]

            if len(spectra) < sc:
                spectra = np.pad(spectra, ((0,0),(0, sc-len(spectra) )), mode='constant', constant_values=0)

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
    spectra = imze.get_region_array(0, normalize=True)


    print("Got spectra", spectra.shape)
    print("mz index", imze.get_mz_index(6662))

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