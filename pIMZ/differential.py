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
from statsmodels.distributions.empirical_distribution import ECDF
from statistics import NormalDist


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
          
    def create_input_masks(self,group1: Dict[Any, Iterable], group2: Dict[Any, Iterable], grouping:str):
        input_masks_group1 = {}
        input_masks_group2 = {}

        pixels_group1 = 0
        pixels_group2 = 0

        self.logger.info("Preparing input masks")

        for spec_name in self.specs:
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


        return input_masks_group1, input_masks_group2, pixels_group1, pixels_group2


    def create_common_features(self):
        self.logger.info("Preparing common features")
        all_spec_names = [x for x in self.specs]
        common_features = list(self.specs[all_spec_names[0]].idx2mass)
        for x in all_spec_names:
            common_features = set.intersection(common_features, list(self.specs[all_spec_names[0]].idx2mass))

        self.logger.info("Identified {} common features".format(len(common_features)))
        
        return common_features


    def do_de_analysis(self, group1: Dict[Any, Iterable], group2: Dict[Any, Iterable], grouping:str) -> pd.DataFrame:

        common_features = self.create_common_features()

        input_masks_group1, input_masks_group2, pixels_group1, pixels_group2 = self.create_input_masks(group1, group2, grouping) 


        dfDict = defaultdict(list)

        bar = makeProgressBar()
        for feature in bar(common_features):

            group1_values = []
            group2_values = []

            for spec_name in self.specs:

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

            dfDict["log2FC"].append(self.logFC(mean_group1, mean_group2))
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

    def logFC(self, mean_group1, mean_group2):
        return np.log2( (mean_group1+ self.pseudo_count)/(mean_group2+self.pseudo_count)  )
   
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



class DifferentialEmpireTest(DifferentialTest):

    def __init__(self, region: SpectraRegion) -> None:
        super().__init__(region)

        binDF1, cBins1, bin2ecdf1 = self.makeBins(deDF, group1)
        binDF2, cBins2, bin2ecdf2 = self.makeBins(deDF, group2)

    def makeBins(self, deDF, replicates):

        repDF = deDF[replicates].copy()
        repDF["mean"] = repDF.mean(axis=1)
        repDF["std"] = repDF.std(axis=1)

        repDF["bins"], createdBins = pd.cut(repDF["mean"], 100, retbins=True)
        l2i = {}
        for i,x in enumerate(repDF["bins"].cat.categories):
            l2i[x.left] = i

        repDF["ibin"] = [l2i[x.left] for x in repDF["bins"]]

        allBinIDs = sorted(set(repDF["ibin"]))
        binID2ECDF = {}
        for binID in allBinIDs:
            selBinDF = repDF[repDF["ibin"] == binID]
            allFCs = []
            
            for r1 in replicates:
                for r2 in replicates:
                    if r1==r2:
                        continue
                    allFCs += [np.log2(x) for x in (selBinDF[r1] / selBinDF[r2]) if ~np.isnan(x) and ~np.isinf(x) and not x==0]

            binECDF = ECDF(allFCs)
            binID2ECDF[binID] = binECDF

        return repDF, createdBins, binID2ECDF


    def prepare_bins(self, group1: Dict[Any, Iterable], group2: Dict[Any, Iterable], grouping:str):
        common_features = self.create_common_features()

        input_masks_group1, input_masks_group2, _, _ = self.create_input_masks(group1, group2, grouping)

        all_means_grp1 = []
        all_means_grp2 = []


        #
        ##
        ### First pass - identify maximal means
        ##
        #

        bar = makeProgressBar()
        for feature in bar(common_features):

            group1_values = []
            group2_values = []

            for spec_name in self.specs:

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

            mean_grp1 = np.mean(group1_values)
            mean_grp2 = np.mean(group2_values)

            all_means_grp1.append(mean_grp1)
            all_means_grp2.append(mean_grp2)

        createdBins_grp1 = np.linspace(np.min(all_means_grp1), np.max(all_means_grp2), 100)
        bins_grp1 = np.digitize(all_means_grp1, bins=createdBins_grp1)

        createdBins_grp2 = np.linspace(np.min(all_means_grp2), np.max(all_means_grp2), 100)
        bins_grp2 = np.digitize(all_means_grp2, bins=createdBins_grp2)


        #
        ##
        ### Second pass: create ECDFs for each bin
        ##
        #

        bin2logfcs1 = defaultdict(list)
        bin2logfcs2 = defaultdict(list)

        bar = makeProgressBar()
        for fi, feature in bar(enumerate(common_features)):

            group1_values = []
            group2_values = []

            for spec_name in self.specs:

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

            binID1 = bins_grp1[fi]
            binID2 = bins_grp2[fi]

            bin2logfcs1[binID1] += create_pairwise_foldchanges(aboveThreshold_group1)
            bin2logfcs2[binID2] += create_pairwise_foldchanges(aboveThreshold_group2)

        bin2ecdf1 = {}
        bin2ecdf2 = {}

        for x in bin2logfcs1:
            bin2ecdf1[x] = ECDF(bin2logfcs1[x])
        for x in bin2logfcs2:
            bin2ecdf2[x] = ECDF(bin2logfcs2[x])



        #
        ##
        ### Third pass: gene zscore
        ##
        #
        allZVals = []
        bar = makeProgressBar()
        for fi, feature in bar(enumerate(common_features)):

            group1_values = []
            group2_values = []

            for spec_name in self.specs:

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

            Zsum, fcValues = self.calculateZ(aboveThreshold_group1, aboveThreshold_group2, bin2ecdf1, bin2ecdf2)

            medianFC = np.median(fcValues)

            if len(fcValues) > 0 and medianFC != 0:
                Znormed, _ = self.calculateZ(aboveThreshold_group1, aboveThreshold_group2, bin2ecdf1, bin2ecdf2, median=medianFC)

            else:
                Znormed = 0
                Zsum = 0

            if Zsum > 0:
                geneZ = max(Zsum - abs(Znormed), 0)
            else:
                geneZ = min(Zsum + abs(Znormed), 0)

            allZVals.append(geneZ)

        xECDF = ECDF(allZVals)

        #
        ##
        ### Fourth pass: gene pvalue
        ##
        #
        allZVals = []
        nd=NormalDist()

        bar = makeProgressBar()
        for fi, feature in bar(enumerate(common_features)):

            group1_values = []
            group2_values = []

            for spec_name in self.specs:

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


            genePzdist = xECDF(geneZ)

            if genePzdist < 10 ** -10:
                genePzdist += 10 ** -10
            if (1-genePzdist) < 10 ** -10:
                genePzdist -= 10 ** -10

            geneZzdist = nd.inv_cdf(genePzdist)
            geneP = 2*(1-nd.cdf(abs(geneZzdist)))




    def compare_groups(self, group1_values: Iterable, group2_values: Iterable) -> float:

        mean_grp1 = np.mean(group1_values)
        mean_grp2 = np.mean(group2_values)

        std_grp1 = np.std(group1_values)
        std_grp2 = np.std(group2_values)


    def calculateZ(row, group1, group2, bin2ecdf1, bin2ecdf2, median=None):
        nd=NormalDist()
        geneFCs = []
        geneZ = 0

        zeroValues = set()

        for c1r in group1:
            for c2r in group2:
                expr1 = row[c1r]
                expr2 = row[c2r]

                if not median is None:
                    expr2 *= median

                if expr2 < 10 ** -10:
                    #zeroValues.add(expr2)
                    continue

                fc = expr1/expr2

                if np.isnan(fc):
                    print(median, expr1, expr2)

                geneFCs.append(fc)

                if expr1 < 10 ** -10:
                    continue

                lFC = np.log2(fc)

                if ~np.isinf(lFC):
                    
                    group1Bin = row["bin1"]
                    group2Bin = row["bin2"]
                    
                    group1P = bin2ecdf1[group1Bin](lFC)
                    group2P = bin2ecdf2[group2Bin](lFC)

                    if group1P < 10 ** -10:
                        group1P += 10 ** -10
                    if group2P < 10 ** -10:
                        group2P += 10 ** -10

                    if (1-group1P) < 10 ** -10:
                        group1P -= 10 ** -10
                    if (1-group2P) < 10 ** -10:
                        group2P -= 10 ** -10

                    #print(group1P, group2P)
                    group1Z = nd.inv_cdf(group1P)
                    group2Z = nd.inv_cdf(group2P)

                    geneZ += group1Z+group2Z
                else:
                    print(fc, lFC)

        #print("Observed 0-Values", zeroValues)
        return geneZ, geneFCs