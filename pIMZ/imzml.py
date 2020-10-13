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

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform, pdist

import scipy.cluster as spc

import scipy as sp
import sklearn as sk

import umap
import hdbscan

class IMZMLExtract:
    """IMZMLExtract class is required to access and retrieve data from an imzML file.
    """

    def __init__(self, fname):
        """
        Constructs an IMZMLExtract object with the following attributes:
        -logger (logging.Logger): Reference to the Logger object.
        -fname (str): Absolute path to the .imzML file.
        -parser (pyimzml.ImzMLParser): Reference to the ImzMLParser object, which opens the two files corresponding to the file name, reads the entire .imzML file and extracts required attributes.
        -dregions (collections.defaultdict): Enumerated regions mapped to the corresponding list of pixel coordinates.
        -mzValues (numpy.array): Sequence of m/z values representing the horizontal axis of the desired mass spectrum.
        -specStart (int): Strating position of the spectra.

        Args:
            fname (str): Absolute path to the .imzML file. Must end with .imzML.
        """

        self.logger = logging.getLogger('IMZMLExtract')
        self.logger.setLevel(logging.INFO)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)

        self.logger.addHandler(consoleHandler)

        self.fname = fname
        self.parser = ImzMLParser(fname)
        self.dregions = None

        self.mzValues = self.parser.getspectrum(0)[0]
        self.coord2index = self._coord2index()

        self.specStart = 0

        if self.specStart != 0:
            self.mzValues = self.mzValues[self.specStart:]
            print("WARNING: SPECTRA STARTING AT POSITION", self.specStart)

        self.find_regions()


    def _coord2index(self):
        """Returns coordinates with their respective index.

        Returns:
            dict: tuple of 3-dimensional coordinates to int index.
        """
        retDict = {}
        for sidx, coord in enumerate(self.parser.coordinates):
            retDict[coord] = sidx
        return retDict


    def get_spectrum(self, specid, normalize=False):
        """ Reads the spectrum at the specified index and can be normalized by dividing each intensity value by the maximum value observed.

        Args:
            specid (int): Index of the desired spectrum in the .imzML file.
            normalize (bool, optional): [description]. Defaults to False.

        Returns:
            numpy.array: Sequence of intensity values corresponding to mz_array of the given specid.
        """

        spectra1 = self.parser.getspectrum(specid)
        spectra1 = spectra1[1]

        if normalize:
            spectra1 = spectra1 / max(spectra1)
        
        return spectra1

    def compare_spectra(self, specid1, specid2):
        """Calculates cosine similarity between two desired spectra.

        Args:
            specid1 (int): Index of the first desired spectrum in the .imzML file.
            specid2 (int): Index of the second desired spectrum in the .imzML file.

        Returns:
            float: Cosine similarity between two desired spectra.
        """
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
        """Returns the closest existing m/z to the given value.

        Args:
            value (float): Value for which the m/z is needed.
            threshold (float, optional): Allowed maximum distance of the discovered m/z index. Defaults to None.

        Returns:
            int: m/z index of the given value.
        """
        curIdxDist = 1000000
        curIdx = None

        for idx, x in enumerate(self.mzValues):
            dist = abs(x-value)

            if dist < curIdxDist and (threshold==None or dist < threshold):
                curIdx = idx
                curIdxDist = dist
            
        return curIdx

    def get_region_indices(self, regionid):
        """Returns a dictionary with the location of the region-specific pixels mapped to their spectral id in the .imzML file.

        Args:
            regionid (int): Id of the desired region in the .imzML file, as specified in dregions dictionary.

        Returns:
            dict: Dictionary of spatial (x, y, 1) coordinates to the index of the corresponding spectrum in the .imzML file.
        """
        if not regionid in self.dregions:
            return None
        
        outindices = {}

        for coord in self.dregions[regionid]:

            spectID = self.coord2index.get(coord, None)

            if spectID == None or spectID < 0:
                print("Invalid coordinate", coord)
                continue

            outindices[coord] = spectID

        return outindices

    def get_region_spectra(self, regionid):
        """Returns a dictionary with the location of the region-specific pixels mapped to their spectra in the .imzML file.

        Args:
            regionid (int): Id of the desired region in the .imzML file, as specified in dregions dictionary.

        Returns:
            dict: Dictionary of spatial (x, y, 1) coordinates to each corresponding spectrum in the .imzML file.
        """
        if not regionid in self.dregions:
            return None
        
        outspectra = {}

        bar = progressbar.ProgressBar()

        for coord in bar(self.dregions[regionid]):

            spectID = self.coord2index.get(coord)

            if spectID == None or spectID < 0:
                print("Invalid coordinate", coord)
                continue

            cspec = self.get_spectrum( spectID )
            cspec = cspec[self.specStart:]# / 1.0
            #cspec = cspec/np.max(cspec)
            outspectra[coord] = cspec

        return outspectra


    def get_avg_region_spectrum(self, regionid):
        """Returns an average spectrum for spectra that belong to a given region.

        Args:
            regionid (int): Id of the desired region in the .imzML file, as specified in dregions dictionary.

        Returns:
            numpy.array: Sequence of intensity values of the average spectrum.
        """
        region_spects = self.get_region_array(regionid)

        return self.get_avg_spectrum(region_spects)

    def get_avg_spectrum(self, region_spects):
        """Returns the average spectrum for an array of spectra.

        The average spectrum is the mean intensity value for all m/z values

        Args:
            region_spects (numpy.array): Three-dimensional array (x, y, s), where x and y are positional coordinates and s corresponds to the spectrum.

        Returns:
            numpy.array: Sequence of intensity values of the average spectrum.
        """

        avgarray = np.zeros((1, region_spects.shape[2]))

        for i in range(0, region_spects.shape[0]):
            for j in range(0, region_spects.shape[1]):

                avgarray[:] += region_spects[i,j,:]

        avgarray = avgarray / (region_spects.shape[0]*region_spects.shape[1])

        return avgarray[0]



    def get_region_range(self, regionid):
        """Returns the shape of the queried region id in all dimensions, x,y and spectra.

        Args:
            regionid (int): Id of the desired region in the .imzML file, as specified in dregions dictionary.

        Returns:
            [3 tuples]: x-range, y-range, z-range, and spectra dimension.
        """

        allpixels = self.dregions[regionid]

        minx = min([x[0] for x in allpixels])
        maxx = max([x[0] for x in allpixels])

        miny = min([x[1] for x in allpixels])
        maxy = max([x[1] for x in allpixels])

        minz = min([x[2] for x in allpixels])
        maxz = max([x[2] for x in allpixels])

        spectraLength = 0
        for coord in self.dregions[regionid]:

            spectID = self.coord2index.get(coord, None)

            if spectID == None or spectID < 0:
                print("Invalid coordinate", coord)
                continue

            splen = self.parser.mzLengths[spectID]-self.specStart

            spectraLength = max(spectraLength, splen)

        return (minx, maxx), (miny, maxy), (minz, maxz), spectraLength

    def get_region_shape(self, regionid):
        """Returns the shape of the queried region. The shape is always rectangular.

        Args:
            regionid (int): Id of the desired region in the .imzML file, as specified in dregions dictionary.

        Returns:
            [tuple]: (width, height, spectra length). Exclusive ends.
        """

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
        """Calculates m/z values that correspond to the peaks with at least five times higher intensity as a median value within the sliding window and extra test within an epsilon hull of the expectant.


        Args:
            spectrum (numpy.array): Sequence of intensity values of the spectrum.
            window (int): The size of the sliding windowing within the peaks should be compared.

        Returns:
            list: sorted m/z indexes that were selected as peaks.
        """
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
        """Calculates m/z values that correspond to the peaks with at least twice as high intensity as the minimum value within the sliding window.


        Args:
            spectrum (numpy.array): Sequence of intensity values of the spectrum.
            window (int): The size of the sliding windowing within the peaks should be compared.

        Returns:
            list: sorted m/z indexes that were selected as peaks.
        """
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
        """Returns the given spectra reduced to the detected peaks and the list of peaks itself.

        Args:
            region_array (numpy.array): Array of spectra.
            peak_window (int, optional): The size of the sliding windowing within the peaks should be compared. Defaults to 100.

        Returns:
            tuple: (Array of reduced spectra, peaks list)
        """
        avg_spectrum = self.get_avg_spectrum(region_array)

        peaks = self.get_peaks_fast(avg_spectrum, peak_window)
        peak_region = np.zeros((region_array.shape[0], region_array.shape[1], len(peaks)))

        for i in range(0, region_array.shape[0]):
            for j in range(0, region_array.shape[1]):

                pspectrum = region_array[i,j,peaks]
                peak_region[i,j,:] = pspectrum

        return peak_region, peaks



    def normalize_spectrum(self, spectrum, normalize=None, max_region_value=None):
        """Normalizes a single spectrum.

        Args:
            spectrum (numpy.array): Spectrum to normalize.
            normalize (str, optional): Normalization method. Must be "max_intensity_spectrum", "max_intensity_region", "vector". Defaults to None.
                - "max_intensity_spectrum": devides the spectrum by the maximum intensity value.
                - "max_intensity_region"/"max_intensity_all_regions": devides the spectrum by custom max_region_value.
                - "vector": devides the spectrum by its norm.
            max_region_value (int/float, optional): Value to normalize to for max-region-intensity norm. Defaults to None.

        Returns:
            numpy.array: Normalized spectrum.
        """

        assert (normalize in [None, "max_intensity_spectrum", "max_intensity_region", "max_intensity_all_regions", "vector"])

        retSpectrum = np.array(spectrum, copy=True)

        if normalize in ["max_intensity_region", "max_intensity_all_regions"]:
            assert(max_region_value != None)

        if normalize == "max_intensity_spectrum":
            retSpectrum = retSpectrum / np.max(retSpectrum)
            return retSpectrum

        elif normalize in ["max_intensity_region", "max_intensity_all_regions"]:
            retSpectrum = retSpectrum / max_region_value
            return retSpectrum

        elif normalize == "vector":

            slen = np.linalg.norm(retSpectrum)

            if slen < 0.01:
                retSpectrum = retSpectrum * 0
            else:
                retSpectrum = retSpectrum / slen

            return retSpectrum


    def baseline_als(self, y, lam, p, niter=10):
        """Performs Baseline Correction with Asymmetric Least Squares Smoothing by P. Eilers and H. Boelens.

        Args:
            y (numpy.array): Spectrum to correct.
            lam (int): 2nd derivative constraint.
            p (float): Weighting of positive residuals.
            niter (int, optional): Maximum number of iterations. Defaults to 10.

        Returns:
            numpy.array: Corrected spectra.
        """
        L = len(y)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z

    def _get_median_spectrum(self, region_array):
        """Calculates the median spectrum of all spectra in region_array.

        Args:
            region_array (numpy.array): Array of spectra.

        Returns:
            numpy.array: Median spectrum.
        """

        median_profile = np.array([0.0] * region_array.shape[2])

        for i in range(0, region_array.shape[2]):

            median_profile[i] = np.median(region_array[:,:,i])

        startedLog = np.quantile([x for x in median_profile if x > 0], [0.05])[0]
        if startedLog == 0:
            startedLog = 0.001

        self.logger.info("Started Log Value: {}".format(startedLog))

        median_profile += startedLog

        return median_profile


    def _fivenumber(self, valuelist):
        """Creates five number statistics for values in valuelist.

        Args:
            valuelist (list/tuple/numpy.array (1D)): List of values to use for statistics.

        Returns:
            tuple: len, len>0, min, 25-quantile, 50-quantile, 75-quantile, max
        """

        min_ = np.min(valuelist)
        max_ = np.max(valuelist)

        (quan25_, quan50_, quan75_) = np.quantile(valuelist, [0.25, 0.5, 0.75])

        return (len(valuelist), len([x for x in valuelist if x > 0]), min_, quan25_, quan50_, quan75_, max_)


    def plot_fcs(self, region_array, positions):
        """Plots the fold-changes of the spectra for each position regarding the median profile for this region.

        Args:
            region_array (numpy.array): Array of spectra.
            positions (list of tuple): 2D position to evaluate.
        """

        fig = plt.figure()

        allData = []
        allLabels = []

        refspec = self._get_median_spectrum(region_array)
        for p, position in enumerate(positions):
            rspec = region_array[position[0], position[1], :]
            res = rspec/refspec

            allData.append(res)
            allLabels.append("{}".format(position))


        bplot2 = plt.boxplot(allData,
                    notch=True,  # notch shape
                    vert=False,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    labels=allLabels)  # will be used to label x-ticks

        plt.show()
        plt.close()



    def normalize_region_array(self, region_array, normalize=None, lam=105, p = 0.01, iters = 10):
        """Returns a normalized array of spectra.

        Args:
            region_array (numpy.array): Array of spectra to normlaize.
            normalize ([type], optional): Normalization method. Must be in "max_intensity_spectrum", "max_intensity_region", "max_intensity_all_regions", "vector", "inter_median", "intra_median", "baseline_cor". Defaults to None.
                - "max_intensity_spectrum": normalizes each spectrum with "max_instensity_spectrum" method in normalize_spectrum function.\n
                - "max_intensity_region": normalizes each spectrum with "max_intensity_region" method using the maximum intensity value within the region.\n
                - "max_intensity_all_regions": normalizes each spectrum with "max_intensity_all_regions" method using the maximum intensity value within all regions.\n
                - "vector": normalizes each spectrum with "vector" method in normalize_spectrum function.\n
                - "inter_median": devides each spectrum by its median to make intensities consistent within each array.\n
                - "intra_median": devides each spectrum by the global median to achieve consistency between arrays.\n
                - "baseline_cor": Baseline Correction with Asymmetric Least Squares Smoothing by P. Eilers and H. Boelens. Requires lam, p and iters parameters.\n
            lam (int, optional): Lambda for baseline correction (if selected). Defaults to 105.
            p (float, optional): p for baseline correction (if selected). Defaults to 0.01.
            iters (int, optional): iterations for baseline correction (if selected). Defaults to 10.

        Returns:
            numpy.array: Normalized region_array.
        """
        
        assert (normalize in [None, "max_intensity_spectrum", "max_intensity_region", "max_intensity_all_regions", "vector", "inter_median", "intra_median", "baseline_cor"])

        if normalize in ["vector"]:
            outarray = np.zeros(region_array.shape)


        if normalize in ["baseline_cor"]:
            outarray = np.array([[self.baseline_als(y, lam, p, iters) for y in x] for x in region_array])
            return outarray

        if normalize in ["inter_median", "intra_median"]:
            
            ref_spectra = self._get_median_spectrum(region_array)

            if normalize == "intra_median":

                allMedians = []

                intra_norm = np.zeros(region_array.shape)
                medianPixel = 0

                bar = progressbar.ProgressBar()

                for i in bar(range(region_array.shape[0])):
                    for j in range(region_array.shape[1]):
                        res = region_array[i,j,:]/ref_spectra
                        median = np.median(res)
                        allMedians.append(median)

                        if median != 0:
                            medianPixel += 1
                            intra_norm[i,j,:] = region_array[i,j, :]/median
                        else:
                            intra_norm[i,j,:] = region_array[i,j,:]

                self.logger.info("Got {} median-enabled pixels".format(medianPixel))
                self.logger.info("5-Number stats for medians: {}".format(self._fivenumber(allMedians)))

                return intra_norm

            if normalize == "inter_median":
                global_fcs = Counter()
                scalingFactor = 100000

                bar = progressbar.ProgressBar()

                self.logger.info("Collecting fold changes")
                for i in bar(range(region_array.shape[0])):
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

                self.logger.info("Global Median {}".format(global_median))

                inter_norm = np.array(region_array, copy=True)

                if global_median != 0:
                    inter_norm = inter_norm / global_median

                return inter_norm


        region_dims = region_array.shape
        outarray = np.array(region_array, copy=True)

        maxInt = 0.0
        for i in range(0, region_dims[0]):
            for j in range(0, region_dims[1]):

                procSpectrum = region_array[i, j, :]

                if normalize in ['max_intensity_region', 'max_intensity_all_regions']:
                    maxInt = max(maxInt, np.max(procSpectrum))

                else:
                    retSpectrum = self.normalize_spectrum(procSpectrum, normalize=normalize)
                    outarray[i, j, :] = retSpectrum

        if not normalize in ['max_intensity_region', 'max_intensity_all_regions']:
            return outarray

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
        """Displays a matrix where each pixel is the sum of intensity values over all m/z summed in the corresponding pixel in region_array.

        Args:
            region_array (numpy.array): Array of spectra.
        """
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

    def plot_tnc(self, region_array):
        """Displays a matrix where each pixel is the norm count of intensity values over all m/z summed in the corresponding pixel in region_array.

        Args:
            region_array (numpy.array): Array of spectra.
        """
        region_dims = region_array.shape
        peakplot = np.zeros((region_array.shape[0],region_array.shape[1]))
        for i in range(0, region_dims[0]):
            for j in range(0, region_dims[1]):

                spectrum = region_array[i, j, :]
                peakplot[i,j] = np.linalg.norm(spectrum)

        heatmap = plt.matshow(peakplot)
        plt.colorbar(heatmap)
        plt.show()
        plt.close()


    def list_highest_peaks(self, region_array, counter=False):
        """Plots the matrix where each pixel is m/z value that corresponds to the maximum  intensity value in the corresponding pixel in region_array.

        Args:
            region_array (numpy.array): Array of spectra.
            counter (bool, optional): Prints a frequency of each m/z peak. Defaults to False.
        """
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
        """Returns the spectrum, its id and true coordinates according to the .imzML file that correspond to the given coordinates within a specific region array.

        Args:
            regionid (int): Id of the desired region in the .imzML file, as specified in dregions dictionary.
            specCoords (numpy.array): Region-specific coordinates of the desired spectrum.

        Returns:
            tuple: (spectum, spectrum id in .imzML file, global coordinates)
        """
        xr,yr,zr,sc = self.get_region_range(regionid)
        
        totalCoords = (specCoords[0]+xr[0], specCoords[1]+yr[0], 1)

        spectID = self.coord2index.get(totalCoords, None)

        if spectID == None or spectID < 0:
            print("Invalid coordinate", totalCoords)
            return None

        cspec = self.get_spectrum( spectID )
        return cspec, spectID, totalCoords
        
    def get_region_index_array(self, regionid):
        """Returns an array with spectra indexes.

        Args:
            regionid (int): Id of the desired region in the .imzML file, as specified in dregions dictionary.

        Returns:
            numpy.array: An array with the same dimensions as the specific region array and each element is an index of the spectrum that correspond to the specific coordinate.
        """
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
        """Returns the shift offers with the maximum similarity to the reference spectrum withing [-maxshift, maxshift] interval.

        Args:
            refspectrum (vector): Sequence of intensities of the desired reference spectrum.
            allspectra (numpy.array): Array of spectra to be shifted. maxshift (int): Maximum allowed shift.

        Returns:
            tuple: (dictionary of the best shifts for each spectrum in allspectra, dictionary of each shifted spectrum)
        """
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
        """Calculates cosine similarity for two vectors.

        Args:
            vA (vector): vector for cosine sim.
            vB (vector): vector for cosine sim.

        Returns:
            float: cosine similarity
        """

        assert(len(vA) == len(vB))

        vAL = np.dot(vA,vA)
        vBL = np.dot(vB,vB)

        if vAL < 0.0000001 or vBL < 0.0000001:
            return 0

        return np.dot(vA, vB) / (np.sqrt( vAL ) * np.sqrt( vBL ))


    def to_called_peaks(self, region, masses, resolution):
        """Transforms an array of spectra into an array of called peaks. The spectra resolution is changed to 1/resolution (0.25-steps for resolution == 4). Peaks are found using ms_peak_picker. If there are multiple peaks for one m/z value, the highest one is chosen.

        Args:
            region (numpy.array): region/array of spectra
            masses (numpy.array): m/z values for region
            resolution (int): Resolution to return

        Returns:
            numpy.array, numpy.array: new array of spectra, corresponding masses
        """

        assert(len(masses) == region.shape[2])

        minMZ = round(min(masses)*resolution)/resolution
        maxMZ = round(max(masses)*resolution)/resolution
        stepSize = 1/resolution
        requiredFields = int( (maxMZ-minMZ)/stepSize )

        startSteps = minMZ / stepSize

        outarray = np.zeros((region.shape[0], region.shape[1], requiredFields+1))
        outmasses = np.array([minMZ + x*stepSize for x in range(0, requiredFields+1)])

        print(min(masses), max(masses))
        print(min(outmasses), max(outmasses))

        print(outarray.shape)
        print(outmasses.shape)

        bar = progressbar.ProgressBar()
        
        px2res = {}
        for i in bar(range(0, region.shape[0])):
            for j in range(0, region.shape[1]):

                pixel = (i,j)
                intensity_array, mz_array = region[i,j,:], masses

                peak_list = ms_peak_picker.pick_peaks(mz_array, intensity_array, fit_type="quadratic")
                #retlist.append(peak_list)

                rpeak2peaks = defaultdict(list)
                for peak in peak_list:
                    if peak.area > 0.0:
                        rpeak2peaks[round(peak.mz*resolution)/resolution].append(peak)


                rpeak2peak = {}
                fpeaklist = []

                for rmz in rpeak2peaks:
                    if len(rpeak2peaks[rmz]) > 1:
                        rpeak2peak[rmz] = sorted(rpeak2peaks[rmz], key=lambda x: x.area, reverse=True)[0]
                    else:
                        rpeak2peak[rmz] = rpeak2peaks[rmz][0]

                    selPeak = rpeak2peak[rmz]

                    fpeak = ms_peak_picker.peak_set.FittedPeak(rmz, selPeak.intensity, selPeak.signal_to_noise, selPeak.peak_count, selPeak.index, selPeak.full_width_at_half_max, selPeak.area,
                        left_width=0, right_width=0)

                    fpeaklist.append(fpeak)
                
                for peak in fpeaklist:
                    idx = int((peak.mz / stepSize) - startSteps)

                    if idx >= outarray.shape[2]:
                        print(peak.mz)

                    outarray[i,j,idx] = peak.intensity

        return outarray, outmasses



    def shift_region_array(self, reg_array, masses, maxshift=20, ref_coord=(0,0)):
        """Shift spectra in reg_array such that the match best with reg_array[reg_coord, :]

        Args:
            reg_array (numpy.array): array of spectra
            masses (numpy.array): m/z values for reg_array spectra
            maxshift (int): maximal shift in each direction. Defaults to 20.
            ref_coord (tuple, optional): Coordinates of the reference spectrum within reg_array. Defaults to (0,0).

        Returns:
            (numpy.array, numpy.array): shifted reg_array, corresponding masses
        """
        
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
    

    def remove_background_spec_aligned(self, array, bgSpec, masses=None, maxshift=20):
        """Subtracts bgSpec from all spectra in array region. For each pixel, the pixel spectrum and bgSpec are aligned (maxshift).

        Args:
            array (numpy.array): Array of spectra to subtract from.
            bgSpec (numpy.array): Spectrum to subtract
            masses (numpy.array, optional): Masses of array. Defaults to None.
            maxshift (int, optional): Maximal shift in any direction for alignment. Defaults to 20.

        Returns:
            numpy.array, numpy.array: subtracted array spectra, corresponding masses
        """

        assert(not bgSpec is None)
        print(bgSpec.shape)
        print(array.shape)
        assert(len(bgSpec) == array.shape[2])

        if masses is None:
            masses = self.mzValues

        outarray = np.zeros((array.shape[0], array.shape[1], array.shape[2]-2*maxshift))
        bspec = bgSpec[maxshift:-maxshift]

        bgShifts = 0

        bar = progressbar.ProgressBar(widgets=[
        progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.AdaptiveETA()
        ])


        for i in bar(range(0, array.shape[0])):
            for j in range(0, array.shape[1]):

                aspec = array[i,j,:]
                bestsim = 0
                for ishift in range(-maxshift, maxshift, 1):
                    shifted = aspec[maxshift+ishift:-maxshift+ishift]
                    newsim = self.__cos_similarity(bspec, shifted)
                    
                    if newsim > bestsim:
                        bestsim = newsim
                        bestshift = ishift

                bgShifts += bestshift
                    
                aspec = aspec[maxshift+bestshift:-maxshift+bestshift]
                aspec = aspec - bspec
                aspec[aspec < 0.0] = 0.0

                outarray[i,j,:] = aspec

        print("avg shift", bgShifts / (array.shape[0]*array.shape[1]))
        return outarray, masses[maxshift:-maxshift]


    def remove_background_spec(self, array, bgSpec):
        """Subtracts bgSpec from all spectra in array region.

        Args:
            array (numpy.array): input array of spectra
            bgSpec (numpy.array): spectrum to subtract

        Returns:
            numpy.array: array - bgSpec
        """
        assert(not bgSpec is None)
        print(bgSpec.shape)
        print(array.shape)
        assert(len(bgSpec) == array.shape[2])

        outarray = np.zeros(array.shape)

        for i in range(0, array.shape[0]):
            for j in range(0, array.shape[1]):
                newspec = array[i,j,:] - bgSpec
                newspec[newspec < 0.0] = 0.0

                outarray[i,j,:] = newspec

        return outarray

    def integrate_masses(self, array, masses=None, window=10):
        """Returns an array of spectra with (2*window less values) where for postion p, the intensity value is the sum from p-window to p+window.

        Args:
            array (numpy.array): Input Array
            masses (numpy.array, optional): Input Masses. Defaults to None.
            window (int, optional): Window to use for integration. Defaults to 10.

        Returns:
            tuple of numpy.array: integrated input array, matching masses
        """

        if masses is None:
            masses = self.mzValues

        outarray = np.zeros((array.shape[0], array.shape[1], array.shape[2]-2*window))

        bar = progressbar.ProgressBar(widgets=[
        progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.AdaptiveETA()
        ])


        for i in bar(range(0, array.shape[0])):
            for j in range(0, array.shape[1]):
                aspec = array[i,j,:]

                initValue = 0
                for k in range(-window,window):
                    initValue += aspec[window+k]
                outarray[i,j,0] = initValue

                for p in range(window+1, array.shape[2]-window):

                    initValue += aspec[p+window]
                    initValue -= aspec[p-window-1]

                    outarray[i,j,p-window] = initValue

        return outarray, masses[window:-window]


    def get_region_array(self, regionid, makeNullLine=True, bgspec=None):
        """Returns a 2D array of spectra for the specified regionid. Subtracts the minimal intensity to accommodate for intensity shifts (due to different heights of the sample). Can subtract a background spectrum for all loaded spectra.

        Args:
            regionid (int): Id of the desired region in the .imzML file, as specified in dregions dictionary.
            makeNullLine (bool, optional): Normalizes each spectum by subtracting its minimal intensity. Defaults to True.
            bgspec (list/numpy.array, optional): spectrum to subtract from all spectra. Defaults to None.

        Returns:
            numpy.array: 2D-Array of spectra
        """

        self.logger.info("Fetching region range")
        xr,yr,zr,sc = self.get_region_range(regionid)
        self.logger.info("Fetching region shape")
        rs = self.get_region_shape(regionid)
        self.logger.info("Found region {} with shape {}".format(regionid, rs))

        sarray = np.zeros( rs, dtype=np.float32 )

        self.logger.info("Fetching region spectra")
        coord2spec = self.get_region_spectra(regionid)
        bar = progressbar.ProgressBar()


        for coord in bar(coord2spec):
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


    def list_regions(self, plot=True):
        """Lists all contained regions in .imzML file and creates a visualization of their location.

        Args:
            plot (bool, optional): Plots overview of spectra. Defaults to True.

        Returns:
            dict: Dictionary of info tuples (x range, y range, #spectra) for each region ID
        """

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

        if plot:
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
        """Prints region specific informations of the .imzML file.
        """
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

                print(regionid, minx, maxx, miny, maxy)

        else:

            self.dregions = self.__detectRegions(self.parser.coordinates)

            with open(self.fname + ".regions", 'w') as outfn:

                for regionid in self.dregions:

                    for pixel in self.dregions[regionid]:

                        print("\t".join([str(x) for x in pixel]), regionid, sep="\t", file=outfn)
    
    
    def __dist(self, x,y):
        """Calculates manhatten distance between two pixels.

        Args:
            x (tuple): Pixel x.
            y (tuple): Pixel y.

        Returns:
            float: manhatten distance between pixels
        """

        assert(len(x)==len(y))

        dist = 0
        for pidx in range(0, len(x)):

            dist += abs(x[pidx]-y[pidx])

        return dist

    def _detectRegions(self, allpixels):
        """Returns a dictionary of region coordinates and creates a visualization of their location.

        Args:
            allpixels (numpy.array): Array of allpixels.

        Returns:
            dict: region id to its coordinates
        """
        #return self.__detectRegions__old(allpixels)
        return self.__detectRegions(allpixels)

    def __detectRegions(self, allpixels):
        """Returns a dictionary of region coordinates and creates a visualization of their location.

        Returns:
            dict: region id to its coordinates
        """
        self.logger.info("Detecting Regions")

        maxX = 0
        maxY = 0

        for coord in self.parser.coordinates:
            maxX = max(maxX, coord[0])
            maxY = max(maxY, coord[1])
        
        img = np.zeros((maxX+1, maxY+1))

        for coord in self.parser.coordinates:
            img[coord[0], coord[1]] = 1
    
        labeledArr, num_ids = ndimage.label(img, structure=np.ones((3,3)))

        outregions = defaultdict(list)
        for x in range(0, labeledArr.shape[0]):
            for y in range(0, labeledArr.shape[1]):
                
                if img[x,y] == 0:
                    # background pixels ;)
                    continue

                outregions[labeledArr[x,y]].append((x,y,1))

        plt.imshow(labeledArr)
        plt.show()
        plt.close()

        self.logger.info("Detecting Regions Finished")

        return outregions


    def __detectRegions__old(self, allpixels):
        """Returns a dictionary of region coordinates and creates a visualization of their location.

        Args:
            allpixels (numpy.array): Array of allpixels.

        Returns:
            dict: region id to its coordinates
        """
        allregions = []

        bar = progressbar.ProgressBar()


        for idx, pixel in enumerate(bar(allpixels)):

            if len(allregions) == 0:
                allregions.append([pixel])
                continue

            if idx % 10000 == 0:
                print("At pixel", idx , "of", len(allpixels), "with", len(allregions), "regions")


            accRegions = []

            for ridx, region in enumerate(allregions):

                pixelAdded = False
                for coord in region:
                    if self.__dist(coord, pixel) <= 1:
                        accRegions.append(ridx)
                        pixelAdded = False
                        break

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

