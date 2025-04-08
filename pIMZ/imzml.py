# general
import math
import logging
import json
import os,sys
import random
from collections import defaultdict, Counter
import glob
import shutil, io, base64

# general package
from natsort import natsorted
import pandas as pd

import numpy as np
from numpy.ctypeslib import ndpointer

from pyimzml.ImzMLParser import ImzMLParser, browse, getionimage
import ms_peak_picker
import pybaselines
import regex as re


# image
import skimage
from skimage import measure as sk_measure

# processing
import dill as pickle

#JIT
from numba import jit, njit, prange
from typing import Tuple

#vis
import dabest
import matplotlib
import matplotlib.pyplot as plt

from scipy import ndimage, misc, sparse, signal, stats, interpolate
from scipy.sparse.linalg import spsolve
from scipy.stats import kurtosis
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull


#web/html
import jinja2


# applications
import progressbar

def makeProgressBar() -> progressbar.ProgressBar:
    return progressbar.ProgressBar(widgets=[
        progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.AdaptiveETA()
        ])


from .plotting import Plotter

@njit(parallel=True)
def normalize_tic(region_array):

    outarray = np.zeros(region_array.shape, dtype=np.float32)

    for i in prange(region_array.shape[0]):
        for j in range(0, region_array.shape[1]):
            specSum = np.sum(region_array[i,j,:])
            if specSum > 0:
                retSpectrum = (region_array[i,j,:] / specSum) * region_array.shape[2]
                outarray[i,j,:] = retSpectrum
    
    return outarray

@njit(parallel=True)
def _prepare_ppp_array(region_array, threshold):
    region_dims = (region_array.shape[0], region_array.shape[1])

    peakplot = np.zeros(region_dims, dtype=np.int32)
    for i in prange(region_dims[0]):
        for j in range(0, region_dims[1]):
            peakplot[i,j] = len([x for x in region_array[i, j, :] if x > threshold])
    return peakplot

@njit(parallel=True)
def _prepare_tic_array(region_array):
    region_dims = (region_array.shape[0], region_array.shape[1])
    peakplot = np.zeros(region_dims, dtype=np.float32)
    for i in prange(region_dims[0]):
        for j in range(0, region_dims[1]):
            peakplot[i,j] = np.sum(region_array[i, j, :])
    return peakplot

@njit(parallel=True)
def _prepare_tnc_array(region_array):

    region_dims = (region_array.shape[0], region_array.shape[1])
    peakplot = np.zeros(region_dims, dtype=np.float32)

    for i in range(0, region_dims[0]):
        for j in range(0, region_dims[1]):
            peakplot[i,j] = np.linalg.norm(region_array[i, j, :])

    return peakplot

class IMZMLExtract:
    """IMZMLExtract class is required to access and retrieve data from an imzML file.
    """

    def __init__(self, fname):
        """
        Constructs an IMZMLExtract object with the following attributes:\n
        -logger (logging.Logger): Reference to the Logger object.\n
        -fname (str): Absolute path to the .imzML file.\n
        -parser (pyimzml.ImzMLParser): Reference to the ImzMLParser object, which opens the two files corresponding to the file name, reads the entire .imzML file and extracts required attributes.\n
        -dregions (collections.defaultdict): Enumerated regions mapped to the corresponding list of pixel coordinates.\n
        -mzValues (numpy.array): Sequence of m/z values representing the horizontal axis of the desired mass spectrum.\n
        -specStart (int): Strating position of the spectra.

        Args:
            fname (str): Absolute path to the .imzML file. Must end with .imzML.
        """

        self.logger = logging.getLogger('IMZMLExtract')
        self.logger.setLevel(logging.INFO)

        #consoleHandler = logging.StreamHandler()
        #consoleHandler.setLevel(logging.INFO)
        #self.logger.addHandler(consoleHandler)

        self.fname = fname
        self.parser = ImzMLParser(fname)
        self.dregions = None

        self.mzValues = self.parser.getspectrum(0)[0]
        self.coord2index = self._coord2index()

        self.specStart = 0

        if self.specStart != 0:
            self.mzValues = self.mzValues[self.specStart:]
            self.logger.warning("WARNING: SPECTRA STARTING AT POSITION {}".format(self.specStart))


        self.find_regions()

        self.check_binned_mz()
        

    def check_binned_mz(self, spectra_to_check=1000):
        """ Checks whether all spectra have same m/z values


        Args:
            spectra_to_check (int, optional): Only checks the first spectra_to_check many spectra. Defaults to 1000.

        Returns:
            bool: True if all spectra have same m/z values
        """

        bar = makeProgressBar()


        for (ci, coord) in enumerate(bar(self.coord2index)):
            idx = self.coord2index[coord]
            
            if ci == spectra_to_check:
                self.logger.info("Checked {} spectra. All have same m/z Values.".format(ci))
                break

            if not np.array_equal(self.mzValues,self.parser.getspectrum(idx)[0]):
                self.logger.error("Not all Spectra have same m/z recording. Please bin spectra! You may use `get_region_array_for_continuous_region` to extract region array.")
                return False

        self.logger.info("All have same m/z Values.")
        return True


    def bins_per_pixel(self, regionid, plot=True):
        
        self.logger.info("Fetching region shape")
        rs = self.get_region_shape(regionid)

        self.logger.info("Fetching region range")
        xr,yr,zr,sc = self.get_region_range(regionid)

        self.logger.info("Fetching region spectra")
        coord2spec, coord2mz = self.get_continuous_region_spectra(regionid)
       
        outarray = np.zeros((rs[0], rs[1]))
        
        bar = makeProgressBar()
        for coord in bar(coord2spec):
            xpos = coord[0]-xr[0]
            ypos = coord[1]-yr[0]
            
            outarray[xpos, ypos] = len(coord2mz[coord])
            
            
        if plot:
            heatmap = plt.matshow(outarray)
            plt.colorbar(heatmap)
            plt.show()
            plt.close()
            
        return outarray


    def _coord2index(self):
        """Returns coordinates with their respective index.

        Returns:
            dict: tuple of 3-dimensional coordinates to int index.
        """
        retDict = {}
        for sidx, coord in enumerate(self.parser.coordinates):
            retDict[coord] = sidx
        return retDict


    def get_spectrum(self, specid, normalize=False, withmz=False):
        """ Reads the spectrum at the specified index and can be normalized by dividing each intensity value by the maximum value observed.

        Args:
            specid (int): Index of the desired spectrum in the .imzML file.
            normalize (bool, optional): Whether to divide the spectrum by its maximum intensity. Defaults to False.
            withmz (bool, optional): Whether to return the respective m/z values. Defaults to False.

        Returns:
            numpy.array: Sequence of intensity values corresponding to mz_array of the given specid.
        """

        spectra1 = self.parser.getspectrum(specid)
        spectra = spectra1[1]
        

        if normalize:
            spectra = spectra / max(spectra)

        if withmz:
            mz = spectra1[0]

            return spectra, mz
        
        return spectra

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

    def compare_sequence(self, spectra1, spectra2):
        """Calculates cosine similarity between two desired spectra.

        Args:
            specid1 (numpy.array/list): Intensity sequence of the first desired spectrum in the .imzML file.
            specid2 (numpy.array/list): Intensity sequence of the second desired spectrum in the .imzML file.

        Returns:
            float: Cosine similarity between two desired spectra.
        """
        return self.__cos_similarity(spectra1, spectra2)

    def get_mz_index(self, value, mzValues, threshold=None):
        """Returns the closest existing m/z to the given value.

        Args:
            value (float): Value for which the m/z is needed.
            threshold (float, optional): Allowed maximum distance of the discovered m/z index. Defaults to None.

        Returns:
            int: m/z index of the given value.
        """
        curIdxDist = 1000000
        curIdx = None

        if mzValues is None:
            mzValues = self.mzValues

        for idx, x in enumerate(mzValues):
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

        bar = makeProgressBar()

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
        bar = makeProgressBar()
        for coord in bar(self.dregions[regionid]):

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


    def _get_peaks(self, spectrum, window):
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

        peaks = self._get_peaks(avg_spectrum, peak_window)
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
            normalize (str, optional): Normalization method. Must be "max_intensity_spectrum", "max_intensity_region", "vector". Defaults to None.\n
                - "max_intensity_spectrum": divides the spectrum by the maximum intensity value.\n
                - "max_intensity_region"/"max_intensity_all_regions": divides the spectrum by custom max_region_value.\n
                - "vector": divides the spectrum by its norm.\n
                - "tic": divides the spectrum by its TIC (sum).\n
            max_region_value (int/float, optional): Value to normalize to for max-region-intensity norm. Defaults to None.

        Returns:
            numpy.array: Normalized spectrum.
        """

        assert (normalize in [None, "zscore", "tic", "max_intensity_spectrum", "max_intensity_region", "max_intensity_all_regions", "vector"])

        retSpectrum = spectrum #np.array(spectrum, copy=True)

        if normalize in ["max_intensity_region", "max_intensity_all_regions"]:
            assert(max_region_value != None)

        if normalize == "max_intensity_spectrum":
            retSpectrum = retSpectrum / np.max(retSpectrum)
            return retSpectrum

        elif normalize in ["max_intensity_region", "max_intensity_all_regions"]:
            retSpectrum = retSpectrum / max_region_value
            return retSpectrum

        elif normalize in ["tic"]:
            specSum = np.sum(retSpectrum)
            if specSum > 0:
                retSpectrum = (retSpectrum / specSum) * len(retSpectrum)
            return retSpectrum

        elif normalize in ["zscore"]:

            lspec = list(retSpectrum)
            nlspec = list(-retSpectrum)
            retSpectrum = np.array(stats.zscore( lspec + nlspec , nan_policy="omit")[:len(lspec)])
            retSpectrum = np.nan_to_num(retSpectrum)
            assert(len(retSpectrum) == len(lspec))

            return retSpectrum

        elif normalize == "vector":

            slen = np.linalg.norm(retSpectrum)

            if slen < 0.01:
                retSpectrum = retSpectrum * 0
            else:
                retSpectrum = retSpectrum / slen

            #with very small spectra it can happen that due to norm the baseline is shifted up!
            retSpectrum[retSpectrum < 0.0] = 0.0
            retSpectrum = retSpectrum - np.min(retSpectrum)

            if not np.linalg.norm(retSpectrum) <= 1.01:
                print(slen, np.linalg.norm(retSpectrum))


            return retSpectrum

    def baselines_pybaseline(self, y):

        bkg_3 = pybaselines.morphological.mor(y, half_window=30)[0]
        return y - bkg_3

    def baseline_rubberband(self, y, mzValues):
        def rubberband(x, y):
            # Find the convex hull
            v = ConvexHull(np.array([x for x in zip(x, y)])).vertices
            # Rotate convex hull vertices until they start from the lowest one
            v = np.roll(v, -v.argmin())
            # Leave only the ascending part
            v = v[:v.argmax()]
            #print(x[v])
            #print(y[v])
            print(v)

            # Create baseline using linear interpolation between vertices
            return np.interp(x, x[v], y[v])
        
        if sum(y) == 0:
            return y

        return y - rubberband(mzValues, y)

    def baseline_als(self, y, lam, p, niter=10):
        """Performs Baseline Correction with Asymmetric Least Squares Smoothing by P. Eilers and H. Boelens.
            See: https://stackoverflow.com/questions/29156532/python-baseline-correction-library
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
        D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        for i in range(niter):
            W.setdiag(w) # Do not create a new matrix, just update diagonal values
            Z = W + D
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z

    def baseline_cor(self, spectrum, division=100, simple=True):       
        nblocks = len(spectrum)//division
        ref_value = np.mean(spectrum)

        x_1 = -1
        s_int_1 = -1
        m_int_1 = -1

        m_int = 0
        x_int = 0

        m_vec = list()
        x_vec = list()

        for block in range(1,nblocks):
            interval = spectrum[1+(block-1)*division:block*division]

            if x_1 == -1 and s_int_1 == -1:
                m_int_1 = np.min(interval)
                x_1 = np.mean(self.mzValues[1+(block-1)*division:block*division])
                s_int_1 = np.std(interval)
            if kurtosis(interval) > 1 and np.mean(interval)<ref_value:
                m_int = np.min(interval)
                x_int = np.mean(self.mzValues[1+(block-1)*division:block*division])

                m_vec.append(m_int)
                x_vec.append(x_int)

        if simple:
            baseline= interp1d([self.mzValues[0],x_int,self.mzValues[self.mzValues.shape[0]-1]],[m_int_1,m_int,0])
        else:
            baseline = interp1d([self.mzValues[0]]+x_vec+[self.mzValues[self.mzValues.shape[0]-1]],[m_int_1]+m_vec+[0])
        new_base = baseline(self.mzValues)

        return spectrum-new_base

    def _get_median_spectrum(self, region_array, quantile=0.5):
        """Calculates the median spectrum of all spectra in region_array.

        Args:
            region_array (numpy.array): Array of spectra.

        Returns:
            numpy.array: Median spectrum.
        """

        median_profile = np.array([0.0] * region_array.shape[2])

        for i in range(0, region_array.shape[2]):

            median_profile[i] = np.quantile(region_array[:,:,i], [quantile])[0]

        qvec = [x for x in median_profile if x > 0]
        
        startedLog = 0
        
        if len(qvec) > 0:
            startedLog = np.quantile(qvec, [0.05])[0]
            
        if startedLog == 0:
            # TODO set this to 0.5 * smallest positive value
            startedLog = 0.001

        self.logger.info("Started Log Value: {}".format(startedLog))

        median_profile += startedLog

        return median_profile


    @classmethod
    def _fivenumber(cls, valuelist) -> Tuple[int, int,float, float, float, float, float]:
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

            self.logger.info("Pixel {}: {}".format(position, self._fivenumber(res)))


        bplot2 = plt.boxplot(allData,
                    notch=True,  # notch shape
                    vert=False,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    labels=allLabels)  # will be used to label x-ticks

        plt.xlabel("Fold Changes against median spectrum")
        plt.ylabel("Pixel Location")

        plt.show()
        plt.close()



    def smooth_spectrum(self, spectrum, method="savgol", window_length=5, polyorder=2):
        """Smoothes the given spectrum.

        Args:
            spectrum (numpy.array): Spectrum of intensities.
            method (str, optional): [Either Savitzky-Golay filter ("savgol"), 1-D Gaussian filter ("gaussian") or Kaiser filter/Finite-Impulse-Response (FIR) filter ("kaiser"). Defaults to "savgol".
            window_length (int, optional): Length of the filter window for "savgol"/"kaiser" and standard deviation for Gaussian kernel. Defaults to 5.
            polyorder (int, optional): The order of the polynomial used to fit the samples for "savgol" method. Defaults to 2.

        Returns:
            numpy.array: Smoothed spectrum.
        """
        assert (method in ["savgol", "gaussian", "kaiser"])

        if method=="savgol":
            outspectrum = signal.savgol_filter(spectrum, window_length=window_length, polyorder=polyorder, mode='nearest')
        elif method=="gaussian":
            outspectrum = ndimage.gaussian_filter1d(spectrum, sigma=window_length, mode='nearest')
        elif method=="kaiser":
            b = (np.ones(window_length))/window_length #numerator co-effs of filter transfer function
            a = np.ones(1)  #denominator co-effs of filter transfer function
            outspectrum = signal.lfilter(b,a,spectrum)

        outspectrum[outspectrum < 0] = 0

        return outspectrum


    def smooth_region_array(self, region_array, method="savgol", window_length=5, polyorder=2):
        """Performs smoothing of all spectra in the given region array.

        Args:
            region_array (numpy.array): Array of spectra to smooth.
            method (str, optional): Either Savitzky-Golay filter ("savgol"), 1-D Gaussian filter ("gaussian") or Kaiser filter/Finite-Impulse-Response (FIR) filter ("kaiser"). Defaults to "savgol".
            window_length (int, optional): Length of the filter window for "savgol"/"kaiser" and standard deviation for Gaussian kernel. Defaults to 5.
            polyorder (int, optional): The order of the polynomial used to fit the samples for "savgol" method. Defaults to 2.

        Returns:
            numpy.array: Array of smoothed spectra.
        """
        assert (method in ["savgol", "gaussian", "kaiser"])
        bar = makeProgressBar()

        outarray = np.zeros(region_array.shape)
        for i in bar(range(region_array.shape[0])):
            for j in range(region_array.shape[1]):
                outarray[i,j,:] = self.smooth_spectrum(region_array[i,j,:], method=method, window_length=window_length, polyorder=polyorder)

        return outarray


    def logarithmize_region_array(self, region_array, logfunc=lambda x: np.log1p(x)):
        """applies logfunc to region_array

        Args:
            region_array (numpy.array): The region array to manipulate/log
            logfunc (lambdax, optional): applies log1p function to region array. Defaults to lambdax:np.log1p.

        Returns:
            numpy.array: scaled/logarithmized regin array
        """


        return logfunc(region_array)
        

    def normalize_region_arrays(self, raDict, refQuantiles=None):
        
        assert (len(raDict) >= 2)
        
        if refQuantiles is None:
            refQuantiles = {}
        
        
        raDictEntries = [x for x in raDict]
        refArray = raDict[raDictEntries[0]]
        
        refConsSpec = self._get_median_spectrum(refArray, refQuantiles.get(raDictEntries[0], 0.5))
        
        def is_not_number(x):
        
            if(math.isinf(x) and x > 0):
                return True
            elif(math.isinf(x) and x < 0):
                return True
            elif(math.isnan(x)):
                return True
            else:
                return False
        
        outArrays = {}
        outArrays[raDictEntries[0]] = refArray.copy()
        
        for i in raDictEntries[1:]:
            
            oArray = raDict[i].copy()
            oQuantile = refQuantiles.get(i, 0.5)
            
            oConsSpec = self._get_median_spectrum(oArray, oQuantile)
            
            allfcs = oConsSpec / refConsSpec
            allfcs = [x for x in allfcs if not is_not_number(x)]
            
            scaleFactor = np.median(allfcs)
            
            print("Region", i, "median FC", scaleFactor)
            
            oArray = oArray * (1/scaleFactor)
            
            outArrays[i] = oArray
            
        return outArrays
            


    def normalize_region_array(self, region_array, normalize=None, lam=105, p = 0.01, iters = 10, division=100, simple=True, median_quantile=0.5, shape=None):
        """Returns a normalized array of spectra.

        Args:
            region_array (numpy.array): Array of spectra to normlaize.
            normalize (str, optional): Normalization method. Must be in "max_intensity_spectrum", "max_intensity_region", "max_intensity_all_regions", "vector", "inter_median", "intra_median", "baseline_cor", "baseline_cor_local". Defaults to None.\n
                - "max_intensity_spectrum": normalizes each spectrum with "max_instensity_spectrum" method in normalize_spectrum function.\n
                - "max_intensity_region": normalizes each spectrum with "max_intensity_region" method using the maximum intensity value within the region.\n
                - "max_intensity_all_regions": normalizes each spectrum with "max_intensity_all_regions" method using the maximum intensity value within all regions.\n
                - "vector": normalizes each spectrum with "vector" method in normalize_spectrum function.\n
                - "inter_median": divides each spectrum by its median to make intensities consistent within each array.\n
                - "intra_median": divides each spectrum by the global median to achieve consistency between arrays.\n
                - "baseline_cor": Baseline Correction with Asymmetric Least Squares Smoothing by P. Eilers and H. Boelens. Requires lam, p and iters parameters.\n
                - "baseline_cor_local": Subtraction of a baseline, estimated after the elimination of the most significant peaks in the mass spectrum.\n
                - "tic": normalizes each spectrum by its TIC (total ion current)
            lam (int, optional): Lambda for baseline correction (if selected). Defaults to 105.
            p (float, optional): p for baseline correction (if selected). Defaults to 0.01.
            iters (int, optional): iterations for baseline correction (if selected). Defaults to 10.
            division (int, optional): number of separate blocks to consider by baseline_cor_local, if selected. Defaults to 100.
            simple (bool, optional): Whether to use three or more values for interpolation by baseline_cor_local, if selected. Defaults to True, three values only. 

        Returns:
            numpy.array: Normalized region_array.
        """
        
        assert (normalize in [None, "zscore", "baselines_pybaseline", "baseline_rubberband", "tic", "max_intensity_spectrum",
                              "max_intensity_region", "max_intensity_all_regions", "vector",
                              "inter_median", "intra_median",
                              "baseline_cor", "baseline_cor_local"])


        if normalize in ["vector"]:
            outarray = np.zeros(region_array.shape)


        if normalize in ["baseline_rubberband"]:

            outarray = np.zeros(region_array.shape)
            bar = makeProgressBar()
            xvalues = np.array([x for x in range(region_array.shape[2])])
            for i in bar(range(region_array.shape[0])):
                for j in range(region_array.shape[1]):
                    outarray[i,j,:] = self.baseline_rubberband(region_array[i,j,:], xvalues)

            return outarray

        if normalize in ["baselines_pybaseline"]:

            outarray = np.zeros(region_array.shape)
            bar = makeProgressBar()
            xvalues = np.array([x for x in range(region_array.shape[2])])
            for i in bar(range(region_array.shape[0])):
                for j in range(region_array.shape[1]):
                    outarray[i,j,:] = self.baselines_pybaseline(region_array[i,j,:])

            return outarray
            

        if normalize in ["baseline_cor"]:

            outarray = np.zeros(region_array.shape)
            bar = makeProgressBar()
            for i in bar(range(region_array.shape[0])):
                for j in range(region_array.shape[1]):
                    outarray[i,j,:] = self.baseline_als(region_array[i,j,:], lam, p, iters)

            return outarray

        if normalize in ["baseline_cor_local"]:

            outarray = np.zeros(region_array.shape)
            bar = makeProgressBar()
            for i in bar(range(region_array.shape[0])):
                for j in range(region_array.shape[1]):
                    outarray[i,j,:] = self.baseline_cor(spectrum=region_array[i,j,:], division=division, simple=simple)

            return outarray

        if normalize in ["inter_median", "intra_median", "intra_fc"]:
            
            ref_spectra = self._get_median_spectrum(region_array, quantile=median_quantile)

            if normalize == "intra_median":

                allMedians = []

                intra_norm = np.zeros(region_array.shape)
                medianPixel = 0

                bar = makeProgressBar()

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

            elif normalize == "inter_median":
                global_fcs = Counter()
                scalingFactor = 100000

                bar = makeProgressBar()

                self.logger.info("Collecting fold changes")
                for i in bar(range(region_array.shape[0])):
                    for j in range(region_array.shape[1]):
                        
                        if not shape is None:
                            if not shape[i,j]:
                                continue

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

                self.logger.info("Median elements: {}".format(len(medians)))

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
        bar = makeProgressBar()
        for i in bar(range(0, region_dims[0])):
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

        bar = makeProgressBar()
        for i in bar(range(0, region_dims[0])):
            for j in range(0, region_dims[1]):
                outarray[i, j, :] = self.normalize_spectrum(outarray[i, j, :], normalize=normalize, max_region_value=maxInt)

        return outarray

    def get_tic_array(self, region_array):
        tic_array = _prepare_tic_array(region_array)
        return tic_array

    def plot_tic(self, region_array):
        """Displays a matrix where each pixel is the sum of intensity values over all m/z summed in the corresponding pixel in region_array.

        Args:
            region_array (numpy.array): Array of spectra.
        """
        peakplot = self.get_tic_array(region_array)

        fig, _ = plt.subplots()
        Plotter.plot_array_scatter(peakplot, fig=fig, discrete_legend=False)
        plt.title("TIC (total summed intensity per pixel)", y=1.08)
        plt.show()
        plt.close()

    def plot_tnc(self, region_array):
        """Displays a matrix where each pixel is the norm count of intensity values over all m/z summed in the corresponding pixel in region_array.

        Args:
            region_array (numpy.array): Array of spectra.
        """
        peakplot = _prepare_tnc_array(region_array)

        fig, _ = plt.subplots()
        Plotter.plot_array_scatter(peakplot, fig=fig, discrete_legend=False)
        plt.title("TNC (total normed intensity per pixel)", y=1.08)
        plt.show()
        plt.close()


    def plot_ppp(self, region_array, file=None, threshold=0):
        """Displays a matrix where each pixel shows the number of peaks at that location.

        Args:
            region_array (numpy.array): Array of spectra.
        """
        peakplot = _prepare_ppp_array(region_array, threshold)

        fig, _ = plt.subplots()
        Plotter.plot_array_scatter(peakplot, fig=fig, discrete_legend=False)
        plt.title("Peaks Per Pixel", y=1.08)
        if not file is None:
            plt.savefig(file, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_pixel_intensities(self, region_array, pixel):
        """plot m/z intensity distribution for pixel in region_array

        Args:
            region_array ([type]): [description]
            pixel ([type]): [description]
        """

        pixelSpec = region_array[pixel[0], pixel[1], :]

        plt.hist(pixelSpec, cumulative=True, histtype="step")
        plt.title("m/z intensity histogram for pixel at position {}".format(pixel))
        plt.show()
        plt.close()

    def plot_mz_intensities(self, region_array, mz_array, mz):
        """plot m/z intensity distribution for pixel in region_array

        Args:
            region_array ([type]): [description]
            pixel ([type]): [description]
        """
        mzIndex = self.get_mz_index(mz, mz_array)
        mzSpec = np.array(region_array[:,:,mzIndex], copy=True)
        mzSpec = np.reshape(mzSpec, (region_array.shape[0]*region_array.shape[1], 1))
        

        plt.hist(mzSpec, cumulative=True, histtype="step")
        plt.title("intensity histogram for m/z-value {}".format(mz_array[mzIndex]))
        plt.show()
        plt.close()

    def list_highest_peaks(self, region_array, counter=False, verbose=False):
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

                if not counter and verbose:
                    print(i,j,mzVal)
                else:
                    maxPeakCounter[mzVal] += 1

        if counter:
            for x in sorted([x for x in maxPeakCounter]):
                print(x, maxPeakCounter[x])

        heatmap = plt.matshow(peakplot)
        plt.xlabel("m/z value of maximum intensity")
        plt.colorbar(heatmap)
        plt.show()
        plt.close()

        print(len(allPeakIntensities), min(allPeakIntensities), max(allPeakIntensities), sum(allPeakIntensities)/len(allPeakIntensities))

        plt.hist(allPeakIntensities, bins=len(allPeakIntensities), cumulative=True, histtype="step")
        plt.title("Cumulative Histogram of maximum peak intensities")
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
        
        bar = makeProgressBar()
        
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


    
    @classmethod
    def detect_hv_masses(cls, region, topn=2000, bins=50, meanThreshold=0.05):    
        """
        https://www.nature.com/articles/nbt.3192#Sec27 / Seurat
        We calculated the mean and a dispersion measure (variance/mean) for each gene across all single cells,
        
        and placed genes into 20 bins based on their average expression. Within each bin
        
        we then z-normalized the dispersion measure of all genes within the bin to identify genes whose expression values were highly variable even when compared to genes with similar average expression.
        
        We used a z-score cutoff of 2 to identify 160 significantly variable genes, after excluding genes with very low average expression
        
        As expected, our highly variable genes consisted primarily of developmental and spatially regulated factors whose expression levels are expected to vary across the dissociated cells.

        The more bins, the longer in takes. The more bins, the less likely are larger z-scores.


        Args:
            region (np.array): input region array
            topn (int, optional): highly variable masses to report. Defaults to 2000.
            bins (int, optional): number of bins to use. Defaults to 50.
            meanThreshold (float, optional): minimal average expression considered for HV-detection. Defaults to 0.05.

        Returns:
            list: vector of indices to keep
        """           

        allMassMeanDisp = []  
        bar = makeProgressBar()

        for k in bar(range(0, region.shape[2])):

            #allMassIntensities = []
            #for i in range(region.shape[0]):
            #    for j in range(region.shape[1]):
            #        allMassIntensities.append(region[i,j,k])
            allMassIntensities = np.ravel(region[:,:,k])

            massMean = np.mean(allMassIntensities)
            massVar = np.var(allMassIntensities)

            allMassMeanDisp.append((k, massMean, massMean/massVar))

        minmax = min([x[1] for x in allMassMeanDisp]), max([x[1] for x in allMassMeanDisp])
        binSpace = np.linspace(0, minmax[1], bins)
        binned = np.digitize([x[1] for x in allMassMeanDisp], binSpace)
        bin2elem = defaultdict(list)

        for idx, binID in enumerate(binned):
            bin2elem[binID].append( allMassMeanDisp[idx] )

        allZElems = []
        for binID in sorted([x for x in bin2elem]):

            binDisps = [x[2] for x in bin2elem[binID] if x[1] > meanThreshold]
            binZs = stats.zscore( binDisps , nan_policy="omit")

            binElems = []
            for disp, z in zip(bin2elem[binID], binZs):
                binElems.append((disp[0], disp[1], z)) #massIndex, mass mean, zvalue

            allZElems += binElems

        allZElems = sorted(allZElems, key=lambda x: x[2], reverse=True)
        quartile = np.quantile([x[1] for x in allZElems], [0.25])[0]

        allZElems_selected = [x for x in allZElems if x[1] > quartile]

        hvMasses = []
        for idx, massMean, massDisp in allZElems_selected:
            hvMasses.append(idx)
            if len(hvMasses) >= topn:
                break

        print("Returning {} highly-variable masses with z > {}".format(len(hvMasses), quartile))
        print(cls._fivenumber([x[1] for x in allZElems]))

        return sorted(hvMasses)


    def to_reduced_masses(self, region, topn=2000, bins=50, return_indices=False):
        """Detects HV (highly variable) masses and reduces the spectra array accordingly.

        Args:
            region (numpy.array): region/array of spectra
            topn (int, optional): Top HV indices. Defaults to 2000.
            bins (int, optional): Number of bins for sorting based on average expression. Defaults to 50.
            return_indices (bool, optional): Whether to include the HV indices as a further return value. Defaults to False.

        Returns:
            numpy.array: new array of spectra
        """

        hvIndices = IMZMLExtract.detect_hv_masses(region, topn=topn, bins=bins)

        print("Identified", len(hvIndices), "HV indices")

        if return_indices:
            return hvIndices

        retArray = np.copy(region)
        retArray[:,:,:] = 0
        retArray[:,:,hvIndices] = region[:,:,hvIndices]

        return retArray

    def reduce_region_to_hv(self, region, region_mz, topn=2000, bins=50, meanThreshold=0.05):
        """Detects HV (highly variable) masses and reduces the spectra array accordingly.

        Args:
            region (numpy.array): region/array of spectra
            region (numpy.array): list/array of spectra mz
            topn (int, optional): Top HV indices. Defaults to 2000.
            bins (int, optional): Number of bins for sorting based on average expression. Defaults to 50.
            meanThreshold (float, optional): minimal average expression considered for HV-detection. Defaults to 0.05.

        Returns:
            numpy.array: new array of spectra
            numpy.array: new array of mz values
        """
        
        hvIndices = IMZMLExtract.detect_hv_masses(region, topn=topn, bins=bins, meanThreshold=meanThreshold)

        print("Identified", len(hvIndices), "HV indices")

        return region[:,:,hvIndices], np.array(region_mz)[hvIndices]


    def interpolate_data(self, region, masses, resolution=0.1, method="akima"):
        """Interpolate all spectra within region to the specified resolution

        Args:
            region (numpy.array): array of spectra
            masses (list): list of corresponding m/z values (same length as spectra)
            resolution (float, optional): step-size for interpolated spectra. Defaults to 0.1
            method (str, optional): Method to use to interpolate the spectra: "akima", "interp1d", "CubicSpline", "Pchip" or "Barycentric". Defaults to "akima".

        Returns:
            numpy.array, numpy.array: array of spectra, corresponding masses
        """
        assert(len(masses) == region.shape[2])

        massesNew = [x for x in np.arange(min(masses), max(masses), resolution)]


        outarray = np.zeros((region.shape[0], region.shape[1], len(massesNew)))
        bar = makeProgressBar()


        for i in bar(range(0, region.shape[0])):
            for j in range(0, region.shape[1]):

                ijSpec = region[i,j]

                specNew = self.interpolate_spectrum(ijSpec, masses, massesNew, method=method)

                outarray[i,j] = specNew

        outarray[ outarray < 0] = 0


        return outarray, massesNew

    def interpolate_spectrum(self, spec, masses, masses_new, method="Pchip"):
        """_summary_

        Args:
            spec (list/numpy.array, optional): spectrum
            masses (list): list of corresponding m/z values (same length as spectra)
            masses_new (list): list of m/z values
            method (str, optional):  Method to use to interpolate the spectra: "akima", "interp1d", "CubicSpline", "Pchip" or "Barycentric". Defaults to "Pchip".

        Returns:
            lisr: updated spectrum
        """
        if method == "akima":
            f = interpolate.Akima1DInterpolator(masses, spec)
            specNew = f(masses_new)
        elif method == "interp1d":
            f = interpolate.interp1d(masses, spec)
            specNew = f(masses_new)
        elif method == "CubicSpline":
            f = interpolate.CubicSpline(masses, spec)
            specNew = f(masses_new)
        elif method == "Pchip":
            f = interpolate.PchipInterpolator(masses, spec)
            specNew = f(masses_new)
        elif method == "Barycentric":
            f = interpolate.BarycentricInterpolator(masses, spec)
            specNew = f(masses_new)
        else:
            raise Exception("Unknown interpolation method")

        #if sum(np.isnan(specNew)) > 0:
        #    raise Exception("Interpolation raised NaNs")

        return specNew

    def to_called_peaks(self, region, masses, resolution=0.1, reduce_peaks=False, picking_method="quadratic"):
        """Transforms an array of spectra into an array of called peaks. The spectra resolution is changed to 1/resolution (0.25-steps for resolution == 4). Peaks are found using ms_peak_picker. If there are multiple peaks for one m/z value, the highest one is chosen.

        Args:
            region (numpy.array): region/array of spectra
            masses (numpy.array): m/z values for region
            resolution (int, optional): Resolution to return. Defaults to 0.1
            reduce_peaks (bool, optional): if true, return only outarray of useful peaks. Defaults to False.
            picking_method (str, optional): The name of the peak model to use. One of "quadratic", "gaussian", "lorentzian", or "apex" (see ms_peak_picker for details). Defaults to "quadratic".
        Returns:
            numpy.array, numpy.array: new array of spectra, corresponding masses
        """

        assert(len(masses) == region.shape[2])

        minMZ = math.floor(min(masses)/resolution)*resolution
        maxMZ = math.ceil(max(masses)/resolution)*resolution

        print("minMZ:", minMZ, min(masses))
        print("maxMZ:", maxMZ, max(masses))

        requiredFields = int( (maxMZ-minMZ)/resolution )

        currentRes = masses[-1]-masses[-2]

        if resolution < currentRes:
            print("WARNING: Selected steps ({}) are smaller than current step size ({})".format(resolution, currentRes))

        startSteps = round(minMZ / resolution)

        outarray = np.zeros((region.shape[0], region.shape[1], requiredFields+1), dtype=np.float32)
        outmasses = np.array([minMZ + x*resolution for x in range(0, requiredFields+1)])

        print(min(masses), max(masses))
        print(min(outmasses), max(outmasses))

        print(outarray.shape)
        print(outmasses.shape)

        bar = makeProgressBar()
        
        
        peakIdx = set()

        for i in bar(range(0, region.shape[0])):
            for j in range(0, region.shape[1]):

                if np.sum(region[i,j,:]) == 0:
                    continue

                peak_list = ms_peak_picker.pick_peaks(region[i,j,:], masses, fit_type=picking_method)
                #retlist.append(peak_list)
                #allPeaksMZ = set([x.mz for x in peak_list])
                #print((i,j), min(allPeaksMZ), max(allPeaksMZ))

                rpeak2peaks = defaultdict(list)
                for peak in peak_list:
                    if peak.mz < minMZ or peak.mz > maxMZ:
                        continue

                    if peak.area > 0.0:
                        resampledPeakMZ = round(peak.mz/resolution)*resolution
                        rpeak2peaks[resampledPeakMZ].append(peak)


                rpeak2peak = {}

                for rmz in rpeak2peaks:
                    if len(rpeak2peaks[rmz]) > 1:
                        rpeak2peak[rmz] = sorted(rpeak2peaks[rmz], key=lambda x: x.area, reverse=True)[0]
                    else:
                        rpeak2peak[rmz] = rpeak2peaks[rmz][0]

                    selPeak = rpeak2peak[rmz]

                    idx = round(selPeak.mz / resolution) - startSteps

                    if idx >= outarray.shape[2] or idx < 0:
                        print("invalid idx for mz", selPeak.mz, "(",rmz,")", ":", idx, "with startsteps", startSteps)
                        print(min(masses), max(masses))
                        return None

                    peakIdx.add(idx)
                    outarray[i,j,idx] = selPeak.intensity

        print("Identified peaks for", len(peakIdx), "of", outarray.shape[2], "fields")

        if reduce_peaks:
            peakIdx = sorted(peakIdx)
            outarray = outarray[:,:,peakIdx]
            outmasses = outmasses[peakIdx]

            return outarray, outmasses

        print("Returning Peaks")
        return outarray, outmasses


    def to_peaks(self, region, masses, resolution=None, reduce_peaks=False, min_peak_prominence=0.5, min_peak_width=0.5, background_quantile=0.5):
        """Transforms an array of spectra into an array of called peaks. The spectra resolution is changed to 1/resolution (0.25-steps for resolution == 4). Peaks are found using ms_peak_picker. If there are multiple peaks for one m/z value, the highest one is chosen.

        Args:
            region (numpy.array): region/array of spectra
            masses (numpy.array): m/z values for region
            resolution (int, optional): Resolution to return. Defaults to None.
            reduce_peaks (bool, optional): if true, return only outarray of useful peaks. Defaults to False.
            min_peak_prominence (float, optional): Required prominence of peaks. Defaults to 0.5
            min_peak_width (float, optional): Required width of peaks in samples. Defaults to 0.5
            background_quantile (float, optional): Required height of peaks. Defalus to 0.5

        Returns:
            numpy.array, numpy.array: new array of spectra, corresponding masses
        """

        assert(len(masses) == region.shape[2])

        if resolution is None:
            resolution = masses[1]-masses[0]

        minMZ = math.floor(min(masses)/resolution)*resolution
        maxMZ = math.ceil(max(masses)/resolution)*resolution

        print("resolution:", resolution)
        print("minMZ:", minMZ, min(masses))
        print("maxMZ:", maxMZ, max(masses))

        requiredFields = int( (maxMZ-minMZ)/resolution )

        currentRes = masses[-1]-masses[-2]

        if resolution < currentRes:
            print("WARNING: Selected steps ({}) are smaller than current step size ({})".format(resolution, currentRes))

        startSteps = round(minMZ / resolution)

        outarray = np.zeros((region.shape[0], region.shape[1], requiredFields+1), dtype=np.float32)
        outmasses = np.array([minMZ + x*resolution for x in range(0, requiredFields+1)])

        print(min(masses), max(masses))
        print(min(outmasses), max(outmasses))

        print(outarray.shape)
        print(outmasses.shape)

        bar = makeProgressBar()
        
        
        peakIdx = set()

        quantile_height = np.quantile(region, [background_quantile])[0]
        print("background intensity:", quantile_height, np.min(region), np.max(region))
        allPixelPeaks = []

        for i in bar(range(0, region.shape[0])):
            for j in range(0, region.shape[1]):

                if np.sum(region[i,j,:]) == 0:
                    continue

                peaks, properties = signal.find_peaks(region[i,j,:], width=min_peak_width, prominence=min_peak_prominence, rel_height=0.5, height=quantile_height)

                rpeak2peaks = defaultdict(list)
                for p in range(0, len(peaks)):
                    
                    peak_mz = masses[peaks[p]]
                    peak_intensity = region[i,j,peaks[p]]                  
                    peak_half_width = properties["width_heights"][p]

                    resampledPeakMZ = round(peak_mz/resolution)*resolution
                    rpeak2peaks[resampledPeakMZ].append({"mz": resampledPeakMZ, "intensity": peak_intensity})


                pixelPeaks = 0

                rpeak2peak = {}
                for rmz in rpeak2peaks:
                    if len(rpeak2peaks[rmz]) > 1:
                        rpeak2peak[rmz] = sorted(rpeak2peaks[rmz], key=lambda x: x["intensity"], reverse=True)[0]
                    else:
                        rpeak2peak[rmz] = rpeak2peaks[rmz][0]

                    selPeak = rpeak2peak[rmz]

                    idx = round(selPeak["mz"]/ resolution) - startSteps

                    if idx >= outarray.shape[2] or idx < 0:
                        print("invalid idx for mz", selPeak, "(",rmz,")", ":", idx, "with startsteps", startSteps)
                        print(min(masses), max(masses))
                        return None

                    peakIdx.add(idx)
                    outarray[i,j,idx] = selPeak["intensity"]
                    pixelPeaks += 1

                allPixelPeaks.append(pixelPeaks)

        print("Pixel Peaks Summary", self._fivenumber(allPixelPeaks))
                

        print("Identified peaks for", len(peakIdx), "of", outarray.shape[2], "fields")
        
        if reduce_peaks:
            peakIdx = sorted(peakIdx)
            outarray = outarray[:,:,peakIdx]
            outmasses = outmasses[peakIdx]

            return outarray, outmasses
        
        print("Returning Peaks")
        return outarray, outmasses





    def shift_region_array(self, reg_array, masses, maxshift=20, ref_coord=(0,0)):
        """Shift spectra in reg_array such that the match best with reg_array[reg_coord, :]

        Args:
            reg_array (numpy.array): array of spectra
            masses (numpy.array): m/z values for reg_array spectra
            maxshift (int, optional): maximal shift in each direction. Defaults to 20.
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

        bar = makeProgressBar()

        for i in bar(range(0, array.shape[0])):
            for j in range(0, array.shape[1]):

                aspec = array[i,j,:]
                bestsim = 0
                bestshift = 0
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

        bar = makeProgressBar()


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
        bar = makeProgressBar()

        paddedSpectra = 0
        for coord in bar(coord2spec):
            xpos = coord[0]-xr[0]
            ypos = coord[1]-yr[0]

            spectra = coord2spec[coord]

            if len(spectra) < sc:
                spectra = np.pad(spectra, (0, sc-len(spectra)), mode='constant', constant_values=0)
                paddedSpectra += 1

            spectra = np.array(spectra, copy=True)

            if not bgspec is None:
                spectra = spectra - bgspec

            if makeNullLine:
                spectra[spectra < 0.0] = 0.0
                spectra = spectra - np.min(spectra)

            sarray[xpos, ypos, :] = spectra

        self.logger.info("Finished region {} with shape {} ({} padded pixels)".format(regionid, rs, paddedSpectra))


        return sarray, self.mzValues


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

                middlex = minx + (maxx-minx)/ 3.0
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





    """

    CONTINUOUS SPECTRA SUPPORT


    """
    def get_continuous_region_spectra(self, regionid):
        """Returns a dictionary with the location of the region-specific pixels mapped to their spectra in the .imzML file, and a dict with mz values

        Args:
            regionid (int): Id of the desired region in the .imzML file, as specified in dregions dictionary.

        Returns:
            dict: Dictionary of spatial (x, y, 1) coordinates to each corresponding spectrum in the .imzML file.
        """
        if not regionid in self.dregions:
            return None
        
        outspectra = {}
        outmzvalues = {}

        bar = makeProgressBar()

        for coord in bar(self.dregions[regionid]):

            spectID = self.coord2index.get(coord)

            if spectID == None or spectID < 0:
                print("Invalid coordinate", coord)
                continue

            cspec, cspec_mz = self.get_spectrum( spectID, withmz=True )
            cspec = cspec[self.specStart:]# / 1.0
            cspec_mz = cspec_mz[self.specStart:]# / 1.0
            outspectra[coord] = cspec
            outmzvalues[coord] = cspec_mz

        return outspectra, outmzvalues

    def check_avg_mz_distance(self, regionid):
        _, c2mz = self.get_continuous_region_spectra(regionid)

        alldiffs = []
        #bar = makeProgressBar()

        for x in c2mz:
            mzVec = c2mz[x]
            if len(mzVec) == 0:
                continue
            diffs = list(np.abs(mzVec[:-1]-mzVec[1:]))
            alldiffs += diffs

        diffMean = np.mean(alldiffs)

        return diffMean


    def get_region_array_for_continuous_region(self, regionid, resolution=0.1, method="akima", new_masses=None, makeNullLine=False, check_strictly_increasing=False):

        self.logger.info("Fetching region range")
        xr,yr,zr,sc = self.get_region_range(regionid)

        self.logger.info("Fetching region spectra")
        coord2spec, coord2mz = self.get_continuous_region_spectra(regionid)
       
        self.logger.info("Identifying mz-range")

        discr_coord2spec = {}

        if new_masses is None:
            mz_range = (np.inf, -np.inf)
            bar = makeProgressBar()
            for x in bar(coord2mz):
                min_max = (min(coord2mz[x]), max(coord2mz[x]))
                mz_range = (min([min_max[0], mz_range[0]]), max([min_max[1], mz_range[1]]))

            self.logger.info("Identified mz-range: {}".format(mz_range))
            masses_new = [x for x in np.arange(mz_range[0], mz_range[1], resolution)]
        else:
            mz_range = (np.min(new_masses), np.max(new_masses))
            self.logger.info("Identified mz-range: {}".format(mz_range))
            masses_new = new_masses
        
        
        rs = self.get_region_shape(regionid)
        rs = (rs[0], rs[1], len(masses_new))
        self.logger.info("Identified region {} as shape {}".format(regionid, rs))
        sarray = np.zeros( rs, dtype=np.float32 )

        self.logger.info("Creating spectra")

        bar = makeProgressBar()
        for coord in bar(coord2spec):
            
            coordSpecInts = np.array(coord2spec[coord], copy=True)
            coordSpecMZs = np.array(coord2mz[coord], copy=True)
                
            diffs = np.diff(coordSpecMZs)
            if check_strictly_increasing:
                if not np.all(diffs > 0):
                    # mz-values not strictly increasing!
                    sortedIdx = np.argsort(coordSpecMZs)
                    
                    coordSpecInts = coordSpecInts[sortedIdx]
                    coordSpecMZs = coordSpecMZs[sortedIdx]
                    
                    use_idx = []
                    for i in range(len(coordSpecMZs)-1):
                        if coordSpecMZs[i] ==  coordSpecMZs[i+1]:
                            continue
                        use_idx.append(i)
                    use_idx = np.array(use_idx)
                    
                    coordSpecInts = coordSpecInts[use_idx]
                    coordSpecMZs = coordSpecMZs[use_idx]

            discr_spec = self.interpolate_spectrum(coordSpecInts, coordSpecMZs, masses_new, method=method)
            assert(len(discr_spec) == rs[2])
            
            xpos = coord[0]-xr[0]
            ypos = coord[1]-yr[0]

            if makeNullLine:
                discr_spec[np.isnan(discr_spec)] = 0
                discr_spec[discr_spec < 0] = 0

            sarray[xpos, ypos, :] = discr_spec

        self.logger.info("Forming region array from spectra")


        bar = makeProgressBar()
        for coord in bar(discr_coord2spec):
            xpos = coord[0]-xr[0]
            ypos = coord[1]-yr[0]

            spectra = np.array(discr_coord2spec[coord], copy=True)

            if makeNullLine:
                spectra[np.isnan(spectra)] = 0
                spectra[spectra < 0] = 0

            sarray[xpos, ypos, :] = spectra


        self.logger.info("Finished region {} with shape {}".format(regionid, rs))


        return sarray, masses_new
    def plot_spectra(self, region_array, coords, valRange, xvals=None, stems=False, label="", start_plot=True, end_plot=True):
        """plots selected spectra in a given range for region_array

        Args:
            region_array (np.array): region_array (3D spectra array)
            coords (list): list of coordinate tuples [(x,y)]
            valRange (list, tuple): range to plot the spectra. Defaults to full range.
            xvals (np.array, list, optional): mz-Values for the entries in region_array 3rd dim. Defaults to None.
            stems (bool, optional): Whether to plot lines or stems for all intensities. Defaults to False.
        """
        
        stemMarkers = ["D", "o", "O"]
        stemColors = ["r", "g", "b"]
        stemFormats = []
        for x in stemMarkers:
            for y in stemColors:
                stemFormats.append(y+x)

        if xvals is None:
            if region_array.shape[2] == len(self.mzValues):
                xvals = self.mzValues

        if start_plot:
            plt.figure()
        
        for xi, x in enumerate(coords):
            labelstr = "{} {}".format(label, str(x))
            if not stems:
                plt.plot(xvals, region_array[x], label=labelstr)
            else:
                markerline, stemlines, baseline = plt.stem(xvals, region_array[x], label=labelstr, markerfmt=stemFormats[xi])
                plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
                plt.setp(stemlines, 'linestyle', 'dotted')
            
        if end_plot:
            plt.xlim(valRange)
            #plt.gca().relim()
            #plt.gca().autoscale_view()
            plt.legend()
            plt.show()
            plt.close()
    