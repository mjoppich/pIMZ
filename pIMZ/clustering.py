# general
import logging
import json
import os
import random
import math
from collections import defaultdict, Counter
import glob
import shutil, io, base64

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
from sklearn import cluster

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



        
class KMeansClusterer(RegionClusterer):

    def __init__(self, region: SpectraRegion, delta=0.2) -> None:
        super().__init__(region)

        self.delta = delta
        self.results = None

        self.matrix_mz = np.copy(self.region.idx2mass)
        self.segmented = None

    def fit(self, num_target_clusters: int, max_iterations: int = 100, verbose: bool = False):
        
        elem_matrix, idx2ij = self.region.prepare_elem_matrix()
        kmeans = cluster.KMeans(n_clusters=15, random_state=0).fit(elem_matrix)

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

        ovrAllCentroid = np.copy(overall_centroid)
        ovrAllCentroid[ovrAllCentroid<=0] = 0
        ovrAllCentroid[ovrAllCentroid>0] = 1

        if verbose:
            print("Selected fields OvrAll Centroid:", sum(ovrAllCentroid), "of", len(ovrAllCentroid))

            
        for seg in sorted(segments):
            seg_centroid = seg_centroids[seg]
            coordinates = segments[seg]

            print("seg centroid", seg_centroid)

            m = math.sqrt((1/len(coordinates)) + (1/n))
            shr_centroid = np.zeros(seg_centroid.shape)
            tstat_centroid = np.zeros(seg_centroid.shape)

            segCentroid = np.copy(seg_centroids[seg])
            segCentroid[segCentroid <= 0] = 0
            segCentroid[segCentroid > 0] = 1

            if verbose:
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

            shrCentroid = np.copy(seg_shr_cent[seg])
            shrCentroid[shrCentroid <= 0] = 0
            shrCentroid[shrCentroid > 0] = 1

            if verbose:
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

        allShrCentroid[allShrCentroid <= 0] = 0
        allShrCentroid[allShrCentroid > 0] = 1

        if verbose:
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

    def _get_iteration_data(self, iteration):
        resultKeys = list(self.results.keys())
        resultDict = resultKeys[iteration]

        return resultDict

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

    def fit(self, num_target_clusters: int, max_iterations: int = 100, verbose: bool = False):
        
        matrix = np.copy(self.region.region_array)
    
        shr_segmented = KMeansClusterer(self.region).fit_transform(num_target_clusters=num_target_clusters, verbose=verbose)

        iteration = 0
        self.results = {}
        self.results[iteration] =  {'segmentation': shr_segmented, 'centroids': None, 'centroids_tstats': None, 'global_centroid': None}

        while iteration < max_iterations:
            iteration += 1

            matrix_global_centroid = self._get_overall_centroid(matrix)
            matrix_segments = self._get_segments(shr_segmented)

            matrix_seg_centroids = self._get_seg_centroids(matrix_segments, matrix)
            
            matrix_shr_centroids, matrix_tstat_centroids = self._get_shr_centroids(matrix_segments, matrix, matrix_seg_centroids, matrix_global_centroid, delta=20)

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

    def transform(self, num_target_clusters: int, verbose: bool = False) -> np.array:
        
        if verbose:
            print("Warning: num_target_clusters not applicable to this method")

        resultDict = self._get_iteration_data(-1)
        
        return resultDict["segmentation"]
        


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

                delta_ij_xy = self._spectra_dist_sasa(matrix[neighbor], matrix[pxCoord])
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


                specDiff = np.linalg.norm(matrix[neighbor]-centroid)
                alpha_ij = self._distance_sasa_alpha(matrix, pxCoord, neighbor, dpos, radius, lambda_)

                distance += alpha_ij * specDiff    

        return distance


    def _call_new_clusters(self, shr_segmented, matrix_segments, matrix, matrix_seg_centroids, matrix_shr_centroids):
        shr_segmented, ams = self._get_new_clusters_func(shr_segmented, matrix_segments, matrix, matrix_seg_centroids, matrix_shr_centroids, print_area=0, distance_func=lambda matrix, pxCoord, centroid, sqSStats, centroidProbability: self._distance_sasa(matrix, pxCoord, 
                centroid, sqSStats, centroidProbability, radius=self.radius))

        return shr_segmented, ams

