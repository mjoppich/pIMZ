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

import abc

# applications
import progressbar
def makeProgressBar():
    return progressbar.ProgressBar(widgets=[
        progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.AdaptiveETA()
        ])

class SpectraRegion():
    pass


class DifferentialTest(metaclass=abc.ABCMeta):

    def __set_logger(self):
        self.logger = logging.getLogger(self.testname)
        self.logger.setLevel(logging.INFO)

        if not self.logger.hasHandlers():

            consoleHandler = logging.StreamHandler()
            consoleHandler.setLevel(logging.INFO)

            self.logger.addHandler(consoleHandler)

            formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
            consoleHandler.setFormatter(formatter)

            self.logger.info("Added new Stream Handler")

    def __init__(self, specs: Union[SpectraRegion, Dict[Any,SpectraRegion]], testname="Differential") -> None:

        
        if isinstance(specs, SpectraRegion):
            self.specs = {specs.name: specs}
        else:
            self.specs = specs
        
        self.testname = testname
        self.pseudo_count = 1e-9
        self.threshold = 0.2
        #logger
        self.__set_logger()


    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'perform_de_analysis') and callable(subclass.perform_de_analysis) and
                hasattr(subclass, 'spec')
                )

    
    def perform_de_analysis(self, group1: Union[Iterable, Dict[Any, Iterable]], group2: Union[Iterable, Dict[Any, Iterable]], grouping:str) -> pd.DataFrame:
        
        for xname in self.specs:
            assert grouping in self.specs[xname].meta

        if isinstance(group1, Iterable):
            assert( len(self.specs) == 1 )
            specname = [x for x in self.specs][0]
            group1 = {specname: group1}

        if isinstance(group2, Iterable):
            assert( len(self.specs) == 1 )
            specname = [x for x in self.specs][0]
            group2 = {specname: group2}

        
        for xname in group1:
            assert(xname in self.specs)

            for xclus in group1[xname]:
                assert xclus in np.unique(self.specs[xname].meta[grouping])

        for xname in group2:
            assert(xname in self.specs)

            for xclus in group2[xname]:
                assert xclus in np.unique(self.specs[xname].meta[grouping])

        return self.do_de_analysis(group1, group2,grouping)



    def __make_de_res_key(self, group1: Dict[Any, Iterable], group2: Dict[Any, Iterable]) -> Iterable:
        """Generates the storage key for two sets of clusters.

        Args:
            clusters0 (list): list of cluster ids 1.
            clusters1 (list): list of cluster ids 2.

        Returns:
            tuple: tuple of both sorted cluster ids, as tuple.
        """
        group1_clusters = [(x, tuple(group1[x])) for x in group1]
        group2_clusters = [(x, tuple(group2[x])) for x in group2]

        return (tuple(sorted(group1_clusters)), tuple(sorted(group2_clusters)))
          
    def do_de_analysis(self, group1: Dict[Any, Iterable], group2: Dict[Any, Iterable], grouping:str) -> pd.DataFrame:


        self.logger.info("Preparing common features")
        all_spec_names = [x for x in self.specs]
        common_features = list(self.specs[all_spec_names[0]].idx2mass)
        for x in all_spec_names:
            common_features = set.intersection(common_features, list(self.specs[all_spec_names[0]].idx2mass))

        self.logger.info("Identified {} common features".format(len(common_features)))

        self.logger.info("Preparing input masks")
        input_masks_group1 = {}
        input_masks_group2 = {}

        pixels_group1 = 0
        pixels_group2 = 0
        
        for spec_name in all_spec_names:
            input_masks_group1[spec_name] = np.zeros( (self.specs[spec_name].region_array.shape[0], self.specs[spec_name].region_array.shape[1]) )
            input_masks_group2[spec_name] = np.zeros( (self.specs[spec_name].region_array.shape[0], self.specs[spec_name].region_array.shape[1]) )

            spec_groups1 = group1.get(spec_name, [])
            for spec_group in spec_groups1:
                input_masks_group1[spec_name][np.where(self.specs[spec_name].meta[self.grouping] == spec_group)] = 1

            pixels_group1 += np.sum(input_masks_group1[spec_name].flatten())

            spec_groups2 = group1.get(spec_name, [])
            for spec_group in spec_groups2:
                input_masks_group2[spec_name][np.where(self.specs[spec_name].meta[self.grouping] == spec_group)] = 1
            
            pixels_group2 += np.sum(input_masks_group2[spec_name].flatten())



        for x in input_masks_group1:
            self.logger.info("For region {} identified {} of {} pixels for group1".format(x, np.sum(input_masks_group1[x].flatten())), np.mul(input_masks_group1[x].shape))

        for x in input_masks_group2:
            self.logger.info("For region {} identified {} of {} pixels for group2".format(x, np.sum(input_masks_group2[x].flatten())), np.mul(input_masks_group2[x].shape))

        self.logger.info("Got all input masks")

        dfDict = defaultdict(list)

        bar = makeProgressBar()
        for feature in bar(common_features):

            group1_values = []
            group2_values = []

            for spec_name in all_spec_names:

                if not spec_name in input_masks_group1 and not spec_name in input_masks_group2:
                    continue
                
                fIdx = self.specs[spec_name]._get_exmass_for_mass(feature)
                if spec_name in input_masks_group1:

                    spec_values = self.specs[spec_name].region_array[:,:,fIdx][input_masks_group1[spec_name]].flatten()
                    group1_values += spec_values

                if spec_name in input_masks_group2:

                    spec_values = self.specs[spec_name].region_array[:,:,fIdx][input_masks_group2[spec_name]].flatten()
                    group2_values += spec_values


            aboveThreshold_group1 = np.array([x for x in group1_values if x > self.threshold])
            aboveThreshold_group2 = np.array([x for x in group2_values if x > self.threshold])

            mean_group1 = np.mean(aboveThreshold_group1)
            mean_group2 = np.mean(aboveThreshold_group2)

            dfDict["feature"].append(feature)
            dfDict["pct.1"].append(len(aboveThreshold_group1) / pixels_group1)
            dfDict["pct.2"].append(len(aboveThreshold_group2) / pixels_group2)
            dfDict["mean.1"].append(mean_group1)
            dfDict["mean.2"].append(mean_group2)

            dfDict["log2FC"].append(np.log2( (mean_group1+ self.pseudo_count)/(mean_group2+self.pseudo_count)  ))
            dfDict["p_value"].append(self.compare_groups(group1_values, group2_values))


        de_df = pd.DataFrame.from_dict(dfDict)
        pvals = de_df["p_value"]

        if self.corr_method == 'benjamini-hochberg':
            from statsmodels.stats.multitest import multipletests
            
            pvals[np.isnan(pvals)] = 1
            _, pvals_adj, _, _ = multipletests(
                pvals, alpha=0.05, method='fdr_bh',returnsorted=False, is_sorted=False
            )
        elif self.corr_method == 'bonferroni':
            pvals_adj = np.minimum(pvals * len(common_features), 1.0)

        de_df["p_value_adj"] = pvals_adj

        return de_df


   
    @abc.abstractmethod
    def compare_groups(self, group1_values:Iterable, group2_values:Iterable) -> float:
        pass

class DifferentialTTest(DifferentialTest):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)


    def compare_groups(self, group1_values: Iterable, group2_values: Iterable) -> float:
        
        with np.errstate(invalid="ignore"):
            scores, pvals = stats.ttest_ind(
                group1_values, group2_values,
                equal_var=False,  # Welch's
            )

            return pvals
            
        return 1.0


class DifferentialWilcoxonRankSumTest(DifferentialTest):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)


    def compare_groups(self, group1_values: Iterable, group2_values: Iterable) -> float:
        
        with np.errstate(invalid="ignore"):
            scores, pvals = stats.ranksums(
                group1_values, group2_values
            )

            return pvals
        
        return 1.0



