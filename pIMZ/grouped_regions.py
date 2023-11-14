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
from natsort import natsorted, natsort_keygen
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Any, Iterable, Union

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
from .plotting import Plotter

#web/html
import jinja2

# applications
import progressbar
def makeProgressBar() -> progressbar.ProgressBar:
    return progressbar.ProgressBar(widgets=[
        progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.AdaptiveETA()
        ])

import sys


class GroupedRegions:
    
    def __init__(self, regions:dict, regions2groups:dict, grouporder=None) -> None:
        
        self.regions = regions
        
        for rn in self.regions:
            self.regions[rn].name = rn
            
        self.regions2groups = regions2groups
        self.region_names = [x for x in self.regions]
        
        allgroups = set()
        for x in self.regions2groups:
            allgroups.add(self.regions2groups[x])
        
        if len(allgroups) != 2:
            print("GroupedRegions currently only supports exactly 2 groups", file=sys.stderr)
            assert(len(allgroups) == 2)
        
        self.group2regions = defaultdict(list)
        
        for x in self.regions2groups:
            self.group2regions[self.regions2groups[x]].append(x)        
        
        self.groups = [x for x in self.group2regions]
        
        if not grouporder is None:
            grouporder = list(grouporder)
            self.groups = sorted(self.groups, key=lambda x: grouporder.index(x))
        
        self.__set_logger()

        
    def __set_logger(self):
        self.logger = logging.getLogger("GroupedRegions")
        self.logger.setLevel(logging.INFO)

        if not self.logger.hasHandlers():

            consoleHandler = logging.StreamHandler()
            consoleHandler.setLevel(logging.INFO)

            self.logger.addHandler(consoleHandler)

            formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
            consoleHandler.setFormatter(formatter)

            self.logger.info("Added new Stream Handler")     
        
    def plot_function_grouped(self, plotfunc, discrete_legend=False, log2=False):
        
        plotElems = dict()
        for x in self.regions:
            if log2:
                plotElems[x] = np.log2(plotfunc(self.regions[x]))
            else:
                plotElems[x] = plotfunc(self.regions[x])
            
        Plotter._plot_arrays_grouped(plotElems, self.group2regions[self.groups[0]], self.group2regions[self.groups[1]], discrete_legend=discrete_legend)
            

    def __iter__(self):
        return iter(self.regions)


    def __getitem__(self, index):
        return self.regions.__getitem__(index)
    
    def __setitem__(self, index, val):
        return self.regions.__setitem__(index, val)
    
    def get_index(self, index:int):
        return self.regions.get(self.region_names[index], None)
    
    
    def create_input_masks(self,group1: Dict[Any, Iterable], group2: Dict[Any, Iterable], grouping:str, verbose:bool=True):
        input_masks_group1 = {}
        input_masks_group2 = {}

        pixels_group1 = 0
        pixels_group2 = 0

        self.logger.info("Preparing input masks")

        for spec_name in self.regions:
            spec=self.regions[spec_name]
            input_masks_group1[spec_name] = np.zeros( (spec.region_array.shape[0], spec.region_array.shape[1]), dtype=np.uint8 )
            input_masks_group2[spec_name] = np.zeros( (spec.region_array.shape[0], spec.region_array.shape[1]), dtype=np.uint8 )

            spec_groups1 = group1.get(spec_name, [])
            for spec_group in spec_groups1:
                input_masks_group1[spec_name][np.where(spec.meta[grouping] == spec_group)] = 1

            pixels_group1 += np.sum(input_masks_group1[spec_name].flatten())

            spec_groups2 = group2.get(spec_name, [])
            for spec_group in spec_groups2:
                input_masks_group2[spec_name][np.where(spec.meta[grouping] == spec_group)] = 1
            
            pixels_group2 += np.sum(input_masks_group2[spec_name].flatten())


        for x in input_masks_group1:
            if verbose:
                self.logger.info("For region {rid} and group {gid} identified {pxi} of {pxa} pixels for group1".format(rid=x, gid=group1.get(x, []), pxi=np.sum(np.ravel(input_masks_group1[x])), pxa=np.multiply(*input_masks_group1[x].shape)))

            input_masks_group1[x] = input_masks_group1[x] == 1

        for x in input_masks_group2:
            if verbose:
                self.logger.info("For region {rid} and group {gid} identified {pxi} of {pxa} pixels for group2".format(rid=x, gid=group2.get(x, []), pxi=np.sum(np.ravel(input_masks_group2[x])), pxa=np.multiply(*input_masks_group2[x].shape)))

            input_masks_group2[x] = input_masks_group2[x] == 1

        if verbose:
            self.logger.info("Got all input masks")


        return input_masks_group1, input_masks_group2, pixels_group1, pixels_group2

    
    def boxplot_feature_expression(self, features, selected_groups, accept_threshold=0.2,
                                   verbose=False, grouping="segmented", log=False,
                                   pw=None, pw_match_name="name"):
        
        if not isinstance(features, (list, tuple, set)):
            features = [features]
        #
        ## Fetching required m/z values
        #      
        
        group1 = {x: selected_groups[x] for x in self.group2regions[self.groups[0]]}
        group2 = {x: selected_groups[x] for x in self.group2regions[self.groups[1]]}
        
        input_masks_group1, input_masks_group2, _, _ = self.create_input_masks(group1, group2, grouping, verbose=verbose) 


        def process_feature(features):
            group1_values = dict()
            group2_values = dict()

            for spec_name in self.regions:

                if not spec_name in input_masks_group1 and not spec_name in input_masks_group2:
                    continue
                
                spec = self.regions[spec_name]
                
                mzFeatures = set()
                for x in features:
                    if type(x) == str and not pw is None:

                        possibleMZs = pw.get_closest_mz_for_protein(x, match_name=pw_match_name, mz_bst=spec.mz_bst)
                        mzFeatures.update(possibleMZs)
                        
                    else:
                        
                        bestExMassForMass, bestExMassIdx = spec._get_exmass_for_mass(x)    
                        
                        if bestExMassIdx < 0:
                            continue
                        
                        if verbose:
                            self.logger.info("Processing mz {} with best existing mz {}".format(x, bestExMassForMass))
                                    
                        mzFeatures.add( bestExMassForMass)
                
                
                fIdxs = set()
                for mz in mzFeatures:
                    _, mzIdx = spec.mz_bst.findClosestValueTo(mz)
                    fIdxs.add(mzIdx)
                
                fIdxs = np.array(sorted(fIdxs))
                
                #fIdx = self.specs[spec_name]._get_exmass_for_mass(feature)[1]

                if spec_name in input_masks_group1 :
                    spec_values = np.ravel(spec.region_array[:,:,fIdxs][input_masks_group1[spec_name]])
                    spec_values = spec_values[spec_values > accept_threshold]

                    if len(spec_values) > 0:
                        group1_values[spec_name] = spec_values

                if spec_name in input_masks_group2:

                    spec_values = np.ravel(spec.region_array[:,:,fIdxs][input_masks_group2[spec_name]])
                    spec_values = spec_values[spec_values > accept_threshold]
                    
                    if len(spec_values) > 0:
                        group2_values[spec_name] = spec_values
                    

            return group1_values, group2_values

        g1Values, g2Values = process_feature(features)
        
        allvalues = []
        
        for s in g1Values:
            for val in g1Values[s]:
                allvalues.append(("group1", s, val))
                allvalues.append(("group1", ("group1", 1), val))
        for s in g2Values:
            for val in g2Values[s]:
                allvalues.append(("group2", s, val))
                allvalues.append(("group2", ("group2", 1), val))

        exprdf = pd.DataFrame.from_records(allvalues, columns=["group", "slide", "value"])
        
        #print(set(exprdf["slide"]))
       
        xorder = [x for x in g1Values] + [("group1", 1), ("group2", 1)] + [x for x in g2Values]
        #print(xorder)
        
        chart = sns.violinplot(x ="slide", y = "value", data = exprdf, order=xorder, log_scale=log)
        
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.show()
