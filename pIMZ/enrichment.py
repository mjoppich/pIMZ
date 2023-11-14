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


class EnrichmentAnalysis(metaclass=abc.ABCMeta):
    
    def __init__(self):
        pass
    
    def custom_div_cmap(self, numcolors=11, name='custom_div_cmap',
                        mincol='blue', midcol='white', maxcol='red'):
        """ Create a custom diverging colormap with three colors
        
        Default is blue to white to red with 11 colors.  Colors can be specified
        in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
        """

        from matplotlib.colors import LinearSegmentedColormap 
        
        cmap = LinearSegmentedColormap.from_list(name=name, 
                                                colors = [x for x in [mincol, midcol, maxcol] if not x is None],
                                                N=numcolors)
        return cmap

    def plotORAresult( self, dfin, title, numResults=10, figsize=(10,10)):
        #https://www.programmersought.com/article/8628812000/
        
        def makeTitle(colDescr, colSize, sucSize, setSize):
            out = []
            for x,z,s,p in zip(colDescr, colSize, sucSize, setSize):
                out.append("{} (cov={:.3f}={}/{})".format(x, z, s, p))

            return out


        df_raw = dfin.copy()

        # Prepare Data
        #determine plot type

        termIDColumn = "elem_id"
        termNameColumn = "elem_name"
        df = df_raw.copy()


        rvb = None

        color1 = "#883656"
        color2 = "#4d6841"
        color3 = "#afafaf"

        if "sample_success_fraction" in df_raw.columns:
            #ORA
            rvb = self.custom_div_cmap(150, mincol=color2, maxcol=color3, midcol=None)
            colorValues = [rvb(x/df.sample_success_fraction.max()) for x in df.sample_success_fraction]

        else:
            raise ValueError()

        df['termtitle'] = makeTitle(df_raw[termNameColumn], df["sample_success_fraction"], df["success_samples"], df["sample_size"])


        df.reset_index()
        df = df[:numResults]
        colorValues = colorValues[:numResults]
        
        df = df.iloc[::-1]
        colorValues = colorValues[::-1]

        print(df.shape)

        if df.shape[0] == 0:
            return
        
        maxNLog = max(-np.log(df.adj_pval))
        maxLine = ((maxNLog// 10)+1)*10       
        
        # Draw plot
        fig, ax = plt.subplots(figsize=figsize, dpi= 80)
        ax.hlines(y=df.termtitle, xmin=0, xmax=maxLine, color='gray', alpha=0.7, linewidth=1, linestyles='dashdot')
        ax.vlines(x=-np.log(0.05), ymin=0, ymax=numResults, color='red', alpha=0.7, linewidth=1, linestyles='dashdot')
        
        sizeFactor = 10    
        scatter = ax.scatter(y=df.termtitle, x=-np.log(df.adj_pval), s=df.sample_size*sizeFactor, c=colorValues, alpha=0.7, )

        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, func=lambda x: x/sizeFactor)
        labels = [x for x in labels]

        # Title, Label, Ticks and Ylim
        ax.set_title(title, fontdict={'size':12})
        ax.set_xlabel('Neg. Log. Adj. p-Value')
        ax.set_yticks(df.termtitle)
        ax.set_yticklabels(df.termtitle, fontdict={'horizontalalignment': 'right'})
        plt.grid(False)
        plt.tight_layout()
        plt.yticks(fontsize=16)
        plt.show()



class OverrepresentationAnalysis(EnrichmentAnalysis):
    
    def __init__(self, ann:ProteinWeights):
        super().__init__()
        
        self.ann = ann
    
    
    def perform_analysis(self, pws, inputDF, mz_bst, numFeatures=None, match_name="name", featureColumn="feature", numberDetectedFeatures=None, pvalCutOff=0.05, minAbsLog2FC= 0.25, onlyPos=False, verbose_prints=None):
        
        
        #
        ## First determine how many peaks could there be => at most region-array entries
        #
        if numFeatures is None:
            numFeatures = len(set(inputDF["feature"]))
        
        measuredGenes = numFeatures
        print("Measured Features", numFeatures)
        
        
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
            setFeatures = self.ann.get_closest_mz_for_proteins(pathwayFeatures, match_name=match_name, mz_bst=mz_bst)
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
                'elem_name': pathwayName,
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
        
        print("Prelim Results", len(setToResult))
        
        if len(setToResult) == 0:
            return None

        rej, elemAdjPvals, _, _ = multipletests(elemPvals, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)

        for eidx, elem in enumerate(sortedElems):

            assert (setToResult[elem]['pval'] == elemPvals[eidx])
            setToResult[elem]['adj_pval'] = elemAdjPvals[eidx]

        return setToResult
            
            