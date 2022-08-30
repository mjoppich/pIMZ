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

import abc

# applications
import progressbar
def makeProgressBar() -> progressbar.ProgressBar:
    return progressbar.ProgressBar(widgets=[
        progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.AdaptiveETA()
        ])


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

    def __init__(self, specs: Union[SpectraRegion, Dict[Any,SpectraRegion]], corr_method="benjamini_hochberg", testname="Differential") -> None:

        
        if isinstance(specs, SpectraRegion):
            self.specs = {specs.name: specs}
        else:
            self.specs = specs
        
        self.testname = testname
        self.pseudo_count = 1e-9
        self.threshold = 0.2
        self.corr_method = corr_method
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

        return self.do_de_analysis(group1, group2, grouping)



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
            input_masks_group1[spec_name] = np.zeros( (self.specs[spec_name].region_array.shape[0], self.specs[spec_name].region_array.shape[1]), dtype=np.uint8 )
            input_masks_group2[spec_name] = np.zeros( (self.specs[spec_name].region_array.shape[0], self.specs[spec_name].region_array.shape[1]), dtype=np.uint8 )

            spec_groups1 = group1.get(spec_name, [])
            for spec_group in spec_groups1:
                input_masks_group1[spec_name][np.where(self.specs[spec_name].meta[grouping] == spec_group)] = 1

            pixels_group1 += np.sum(input_masks_group1[spec_name].flatten())

            spec_groups2 = group2.get(spec_name, [])
            for spec_group in spec_groups2:
                input_masks_group2[spec_name][np.where(self.specs[spec_name].meta[grouping] == spec_group)] = 1
            
            pixels_group2 += np.sum(input_masks_group2[spec_name].flatten())


        for x in input_masks_group1:
            self.logger.info("For region {rid} identified {pxi} of {pxa} pixels for group1".format(rid=x, pxi=np.sum(np.ravel(input_masks_group1[x])), pxa=np.multiply(*input_masks_group1[x].shape)))

            input_masks_group1[x] = input_masks_group1[x] == 1

        for x in input_masks_group2:
            self.logger.info("For region {} identified {} of {} pixels for group2".format(x, np.sum(np.ravel(input_masks_group2[x])), np.multiply(*input_masks_group2[x].shape)))

            input_masks_group2[x] = input_masks_group2[x] == 1

        self.logger.info("Got all input masks")


        return input_masks_group1, input_masks_group2, pixels_group1, pixels_group2


    def create_common_features(self):
        self.logger.info("Preparing common features")
        all_spec_names = [x for x in self.specs]
        common_features = set(self.specs[all_spec_names[0]].idx2mass)
        for x in all_spec_names:
            common_features = set.intersection(common_features, set(self.specs[all_spec_names[0]].idx2mass))

        self.logger.info("Identified {} common features".format(len(common_features)))
        
        return common_features


    def do_de_analysis(self, group1: Dict[Any, Iterable], group2: Dict[Any, Iterable], grouping:str) -> pd.DataFrame:

        common_features = self.create_common_features()

        input_masks_group1, input_masks_group2, pixels_group1, pixels_group2 = self.create_input_masks(group1, group2, grouping) 


        
        def process_feature(feature):
            group1_values = np.empty(0)
            group2_values = np.empty(0)

            for spec_name in self.specs:

                if not spec_name in input_masks_group1 and not spec_name in input_masks_group2:
                    continue
                
                fIdx = self.specs[spec_name]._get_exmass_for_mass(feature)[1]

                if spec_name in input_masks_group1:
                    spec_values = np.ravel(self.specs[spec_name].region_array[:,:,fIdx][input_masks_group1[spec_name]])
                    spec_values = spec_values[spec_values > self.threshold]
                    group1_values = np.concatenate((group1_values, spec_values))

                if spec_name in input_masks_group2:

                    spec_values = np.ravel(self.specs[spec_name].region_array[:,:,fIdx][input_masks_group2[spec_name]])
                    spec_values = spec_values[spec_values > self.threshold]
                    group2_values = np.concatenate((group2_values, spec_values))


            mean_group1 = np.mean(group1_values)
            mean_group2 = np.mean(group2_values)

            #print(len(group1_values), len(group2_values), pixels_group1, pixels_group2)

            returnDict = {}
            returnDict["feature"] =feature
            returnDict["pct.1"] = len(group1_values) / pixels_group1
            returnDict["pct.2"] = len(group2_values) / pixels_group2
            returnDict["mean.1"] = mean_group1
            returnDict["mean.2"] = mean_group2
            returnDict["log2FC"] = self.logFC(mean_group1, mean_group2)
            returnDict["p_value"] = self.compare_groups(group1_values, group2_values)

            return returnDict

        print("Starting analysis")
        #results = list(map(process_feature, common_features))
        bar = makeProgressBar()
        results = [process_feature(x) for x in bar(common_features)]
  
        print("Finishing analysis")

        de_df = pd.DataFrame.from_records(results)
        pvals = de_df["p_value"]

        if self.corr_method == 'benjamini_hochberg':
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


        input_masks_group1, input_masks_group2, pixels_group1, pixels_group2 = self.create_input_masks(group1, group2, grouping) 


    def create_ecdf(self, input_masks, bins):

        selCoords = defaultdict(list)

        for spec_name in self.specs:
            if not spec_name in input_masks:
                continue

            inMask = input_masks.get(spec_name, None)

            for i in range(inMask.shape[0]):
                for j in range(inMask.shape[1]):
                    if inMask[i,j] == 1:
                        selCoords[spec_name].append((i,j))

        binID2ECDF = {}
        for binID in bins:

            binMatrix = self.specs.region_array[:,:, bins == binID]

            allFCs = []
            for coord1 in selCoords:
                for coord2 in selCoords:

                    if coord1 == coord2:
                        continue

                    coord1Vec = binMatrix[coord1]
                    coord2Vec = binMatrix[coord2]

                    useIndices = coord1Vec > self.threshold & coord2Vec > self.threshold

                    allFCs += [np.log2(x) for x in (coord1Vec[useIndices] / coord2Vec[useIndices]) if ~np.isnan(x) and ~np.isinf(x) and not x==0]
        
            binECDF = ECDF(allFCs)
            binID2ECDF[binID] = binECDF

        return binID2ECDF


    def makeBins(self, input_mask, binCount=100):

        
        maskMin = np.min(np.ravel(self.specs.region_array[input_mask]))
        maskMax = np.max(np.ravel(self.specs.region_array[input_mask]))

        bins = np.linspace(maskMin, maskMax, binCount+1)
        allIntervals = []
        allBinIDs = []
        for i in range(len(bins)-1):
            binInt = Interval(bins[i], bins[i+1], data=i)
            allIntervals.append(binInt)
            allBinIDs.append(i)

        t = IntervalTree(allIntervals)

        featureBins = np.array([-1] * self.specs.region_array.shape[2])

        for i in range(self.specs.region_array.shape[2]):

            featureValues = np.mean(np.ravel(self.specs.region_array[:,:,i][input_mask]))
            binID = list(t[featureValues])[0].data
            featureBins[i] = binID

        selCoords = []
        for i in range(input_mask.shape[0]):
            for j in range(input_mask.shape[1]):
                if input_mask[i,j] == 1:
                    selCoords.append((i,j))
        
        
        binID2ECDF = {}
        for binID in allBinIDs:

            binMatrix = self.specs.region_array[:,:, featureBins == binID]

            allFCs = []
            for coord1 in selCoords:
                for coord2 in selCoords:

                    if coord1 == coord2:
                        continue

                    allFCs += [np.log2(x) for x in (binMatrix[coord1] / binMatrix[coord2]) if ~np.isnan(x) and ~np.isinf(x) and not x==0]
        
            binECDF = ECDF(allFCs)
            binID2ECDF[binID] = binECDF
        

        return binID2ECDF, featureBins


    def prepare_bins(self, group1: Dict[Any, Iterable], group2: Dict[Any, Iterable], grouping:str, binCount:int=100):
        common_features = self.create_common_features()

        input_masks_group1, input_masks_group2, pixels_group1, pixels_group2 = self.create_input_masks(group1, group2, grouping)

        all_means_grp1 = []
        all_means_grp2 = []


        #
        ##
        ### First pass - identify maximal means
        ##
        #

        self.logger.info("First pass - identify maximal means")

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
                    group1_values += list(spec_values[spec_values > self.threshold])

                if spec_name in input_masks_group2:

                    spec_values = self.specs[spec_name].region_array[:,:,fIdx][input_masks_group2[spec_name]].flatten()
                    group2_values += list(spec_values[spec_values > self.threshold])


            mean_grp1 = np.mean(group1_values)
            mean_grp2 = np.mean(group2_values)

            all_means_grp1.append(mean_grp1)
            all_means_grp2.append(mean_grp2)

        all_means_grp1 = np.array(all_means_grp1)
        all_means_grp2 = np.array(all_means_grp2)

        createdBins_grp1 = np.linspace(np.min(all_means_grp1), np.max(all_means_grp2), binCount)
        bins_grp1 = np.digitize(all_means_grp1, bins=createdBins_grp1)

        createdBins_grp2 = np.linspace(np.min(all_means_grp2), np.max(all_means_grp2), binCount)
        bins_grp2 = np.digitize(all_means_grp2, bins=createdBins_grp2)


        #
        ##
        ### Second pass: create ECDFs for each bin
        ##
        #
        self.logger.info("Second pass - create ECDF for each bin")

        
        bin2ecdf1 = self.create_ecdf(input_masks_group1, bins_grp1)
        bin2ecdf2 = self.create_ecdf(input_masks_group2, bins_grp2)


        #
        ##
        ### Third pass: gene zscore
        ##
        #
        self.logger.info("Third pass - calculate gene z-scores and p-values")

        featureResults = defaultdict(lambda: dict())

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

                    spec_values = np.ravel(self.specs[spec_name].region_array[:,:,fIdx][input_masks_group1[spec_name]])
                    group1_values += list(spec_values[spec_values > self.threshold])

                if spec_name in input_masks_group2:

                    spec_values = np.ravel(self.specs[spec_name].region_array[:,:,fIdx][input_masks_group2[spec_name]])
                    group2_values += list(spec_values[spec_values > self.threshold])


            Zsum, medianFC = self.calculateZ(group1_values, group2_values, bin2ecdf1, bin2ecdf2)

            if medianFC != 0:
                Znormed, _ = self.calculateZ(group1_values, group2_values, bin2ecdf1, bin2ecdf2, median=medianFC)
            else:
                Znormed = 0
                Zsum = 0

            if Zsum > 0:
                geneZ = max(Zsum - abs(Znormed), 0)
            else:
                geneZ = min(Zsum + abs(Znormed), 0)

            allZVals.append(geneZ)

            featureResults[feature]["feature"] = feature
            featureResults[feature]["pct.1"] = len(group1_values) / pixels_group1
            featureResults[feature]["pct.2"] = len(group2_values) / pixels_group2
            featureResults[feature]["mean.1"] = np.mean(group1_values)
            featureResults[feature]["mean.2"] = np.mean(group2_values)
            featureResults[feature]["logFC"] = medianFC


            featureResults[feature]["Zsum"] = Zsum
            featureResults[feature]["Znormed"] = Znormed
            featureResults[feature]["geneZ"] = geneZ
            featureResults[feature]["considered"] = geneZ != 0


        xECDF = ECDF(allZVals)

        #
        ##
        ### Fourth pass: gene pvalue
        ##
        #
        self.logger.info("Fourth pass - calculate adjusted gene p-value")

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

                    spec_values = np.ravel(self.specs[spec_name].region_array[:,:,fIdx][input_masks_group1[spec_name]])
                    group1_values += list(spec_values[spec_values > self.threshold])

                if spec_name in input_masks_group2:

                    spec_values = np.ravel(self.specs[spec_name].region_array[:,:,fIdx][input_masks_group2[spec_name]])
                    group2_values += list(spec_values[spec_values > self.threshold])


            genePzdist = xECDF(geneZ)

            if genePzdist < 10 ** -10:
                genePzdist += 10 ** -10
            if (1-genePzdist) < 10 ** -10:
                genePzdist -= 10 ** -10

            geneZzdist = nd.inv_cdf(genePzdist)
            geneP = 2*(1-nd.cdf(abs(geneZzdist)))

            featureResults[feature]["p_value"] = geneP

        
        resultDF = pd.DataFrame.from_dict(featureResults, orient='index')

        _, usedAdjPvals, _, _ = multipletests(list(resultDF["p_value"]), alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
        resultDF["p_value_adj"] = usedAdjPvals

        return resultDF

    def compare_groups(self, group1_values: Iterable, group2_values: Iterable) -> float:
        return 0


    def calculateZ(group1, group2, group1Bin, group2Bin, bin2ecdf1, bin2ecdf2, median=None, samples=100):
        nd=NormalDist()
        geneFCs = []
        geneZ = 0

        if not samples is None:
            group1 = random.choices(group1, k=min(samples, len(group2)))
            group2 = random.choices(group2, k=min(samples, len(group2)))

        for expr1 in group1:
            for expr2 in group2:

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
        medianFC = 0
        if len(geneFCs) > 0:
            medianFC = np.median(geneFCs)

        return geneZ, medianFC