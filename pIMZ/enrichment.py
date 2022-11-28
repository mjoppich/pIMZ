# general
import logging
import json
import os
import random
import math
from collections import defaultdict, Counter
import glob
import shutil, io, base64, abc
from typing import Dict, Any, Iterable, Union

# general package
from natsort import natsorted
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

import regex as re
import h5py
from intervaltree import Interval, IntervalTree

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
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats.multitest import multipletests
from statistics import NormalDist


from .imzml import IMZMLExtract
from .plotting import Plotter
from .regions import SpectraRegion
from .annotations import ProteinWeights

import abc

# applications
import progressbar
def makeProgressBar() -> progressbar.ProgressBar:
    return progressbar.ProgressBar(widgets=[
        progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.AdaptiveETA()
        ])

from itertools import chain
from statsmodels.stats.multitest import multipletests
from scipy.stats import hypergeom




class OverrepresentationAnalysis:
    
    def __init__(self, spec:SpectraRegion, ann:ProteinWeights):
        
        self.spec = spec
        self.ann = ann
    
    
    def perform_analysis(self, pws, inputDF, match_name="name", featureColumn="feature", numberDetectedFeatures=None, pvalCutOff=0.05, minAbsLog2FC= 0.25, onlyPos=False, verbose_prints=None):
        
        
        #
        ## First determine how many peaks could there be => at most region-array entries
        #
        measuredGenes = self.spec.region_array.shape[2]
        print("Measured Features", measuredGenes)
        
        
        sigDF = inputDF.loc[lambda x: x["#matches"]!= 0]
        sigDF = sigDF.loc[lambda x: x["p_value_adj"]<pvalCutOff].loc[lambda x: abs(x["log2FC"])> minAbsLog2FC]
        
        if onlyPos:
            sigDF = sigDF.loc[lambda x: x["log2FC"] > 0]         
            
        setToResult = {}
        
        #
        ## Number of significant Peaks
        #
        significantFeatures = set(sigDF[featureColumn])
        print("Significant Feature Counts: {}".format(len(significantFeatures)))
        print(list(significantFeatures)[:5])
        
        bar = makeProgressBar()
        for pathwayID in bar(pws):

            pathwayName, pathwayFeatures = pws[pathwayID]
            setFeatures = self.ann.get_closest_mz_for_proteins(pathwayFeatures, match_name=match_name, mz_bst=self.spec.mz_bst)
            #print(list(setFeatures)[:5])
            sampleSize = len(setFeatures)
            
            
            if sampleSize == 0:
                continue
            
            successIntersection = significantFeatures.intersection(setFeatures)
            drawnSuccesses = len(successIntersection)
            
            if drawnSuccesses == 0:
                continue
            
            pval = hypergeom.sf(drawnSuccesses - 1, measuredGenes, len(significantFeatures), sampleSize)
            fractionOfHitSamples = drawnSuccesses / sampleSize if sampleSize > 0 else 0
            
            resultObj = {
                'elem_id': pathwayID,
                'population_size': measuredGenes,
                'success_population': len(significantFeatures),
                'sample_size': sampleSize,
                'success_samples': drawnSuccesses,
                'pval': pval,
                'sample_success_fraction': fractionOfHitSamples,
                "features": successIntersection
            }
            
            if np.isnan(pval):
                print(pathwayID)
                print(resultObj)
                return None
            
            if pathwayName in setToResult:
                print("Double Pathway Name", pathwayName)

            setToResult[pathwayName] = resultObj

        sortedElems = [x for x in setToResult]
        elemPvals = [setToResult[x]["pval"] for x in sortedElems]
        
        print("Prelim Results", len(setToResult), len(sortedElems), len(set(sortedElems)))
        
        if len(setToResult) == 0:
            return None

        rej, elemAdjPvals, _, _ = multipletests(elemPvals, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)

        for eidx, elem in enumerate(sortedElems):

            assert (setToResult[elem]['pval'] == elemPvals[eidx])
            setToResult[elem]['adj_pval'] = elemAdjPvals[eidx]

        return setToResult
            
            