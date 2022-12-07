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

# additional imports for merits analysis 
from upsetplot import from_contents, plot
import math
#import regions
from skimage.filters import threshold_multiotsu

class KsSimilarity:

    def __init__(self):
        
        pass

    def perform_analysis(self, spec_mz_values:SpectraRegion, clustering=np.empty([0, 0]), weight_ks=1, weight_p=1, weight_c=1):
        """ This function finds masses that represent a certein cluster 
        
            input:  mendatory:  - spec_mz_values: mz values in SpectraRegion format
                    optional:   - clustering: array, default is spec_mz_values.meta["segmented"]
                                - weights: floats 

            return: dataframe with all information for masses per cluster such as 
                    - ks statistics
                    - p-values
                    - relevance score
        """
        
        mz_region_array = spec_mz_values.region_array
        print(mz_region_array.shape)

        if clustering.shape != mz_region_array.shape:
            clustering = spec_mz_values.meta["segmented"]
            #clustering = clustering_multi_otsu(clustering, cluster_amount=5)[0]
        
        clusters=np.unique(clustering)

        considered_masses = list(self.excluding_outliers(mz_region_array).keys())
        ks_test_unclustered = self.find_relevant_masses_by_ks(mz_region_array, considered_masses, clustering, for_clustered=False)
        # mass, cluster, ks, p, p_mult
        ms_df = self.dict_to_dataframe(ks_test_unclustered) 
        # multiple testing correction
        rej, elemAdjPvals, _, _ = multipletests(ms_df["p_value"], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
        ms_df["p_value_mult"] = elemAdjPvals
        # dict: mass is key -> needed lateron for relevence ranking
        ks_p_dict = self.dataframe_to_mass_dict(ms_df)
        # dict: cluster is key
        rel_masses_for_cluster = self.dataframe_to_cluster_dict(ms_df)
        # add column: is unique?
        ms_df = self.find_unique_masses_for_cluster(rel_masses_for_cluster, ms_df)
        # add score for relevance
        ms_df = self.relevance_ranking(ks_test=ks_p_dict, considered_masses=rel_masses_for_cluster, clusters=clusters, weight_ks=weight_ks, weight_p=weight_p, weight_c=weight_c, cluster_amount=len(clusters), df_ms=ms_df)
        
        return ms_df

    def excluding_outliers(self, ms_region_array:np.array):
        """input: three dimensional array with mass spec data
            returns: dictionary, mass:threshold, all masses where more than 5% of the pixels are expressed"""
        ms_shape = ms_region_array.shape
        considered_masses = {}
        for mass in range(ms_shape[2]):
            list_mass = ms_region_array[:,:,mass].flatten()
            #plt.imshow(region_array[:,:,0])
            list_mass = sorted(list_mass)
            # end index is the intensity above which points will be considered as outliers and therefore ignored
            len(list_mass)
            end_index=int(math.floor(len(list_mass)*0.95))
            th_outliers = list_mass[end_index]
            if th_outliers > 0:
                considered_masses[mass] = th_outliers
        return considered_masses


    def dataframe_to_mass_dict(self, ms_df):
        """input: dataframe
            return: dict: {mass: {c1:(ks,p),c2:(ks,p) } }"""
        ks_p_dict = {}
        for i in ms_df.index:
            c = ms_df["cluster_name"][i]
            mass = ms_df["mass_name"][i]
            ks = ms_df["ks_value"][i]
            p = ms_df["p_value_mult"][i]
            if mass in ks_p_dict.keys():
                ks_p_dict[mass].update({c:(ks,p)})
            else:
                ks_p_dict[mass]={c:(ks,p)}
        return ks_p_dict


    def dataframe_to_cluster_dict(self, mf_df):
        """input: dataframe
            return: dict: cluster: all masses """
        relevant_masses_by_cluster_raw = defaultdict(list)
        for i in mf_df.index:
                c = mf_df["cluster_name"][i]
                mass = mf_df["mass_name"][i]
                relevant_masses_by_cluster_raw[c].append(mass)
        relevant_masses_by_cluster_raw= dict(sorted(relevant_masses_by_cluster_raw.items()))
        return relevant_masses_by_cluster_raw


    def find_unique_masses_for_cluster(self, rel_masses_for_cluster, df_ms=pd.DataFrame([])):
        """input: dict: cluster=masses 
            return: dataframe, unique added OR dict cluster=unique_masses"""
        df_contents = from_contents(rel_masses_for_cluster)

        if not df_ms.shape==(0,0):
            df_ms['unique_for_cluster']=False
            l = len(df_contents['id'].axes[0][0]) # weird way to get the cluster amount
            for mass in df_contents['id'].items():
                for c in range(l):
                    if mass[0][c] == True and any(mass[0][0:c]) == False and any(mass[0][(c+1):l]) == False:
                        index_mass = df_ms.index[(df_ms['mass_name'] == mass[1]) & (df_ms['cluster_name'] == c)].tolist()
                        if len(index_mass)==0:
                            print(f"mass: {mass[1]}, cluster: {c} not found in df")
                            break
                        elif len(index_mass)>1:
                            print(f"mass: {mass[1]}, cluster: {c} multilpe times in df: index_mass")
                            break
                        else:
                            df_ms.at[index_mass[0], "unique_for_cluster"] = True
                        break
            return df_ms

        else:
            unique = defaultdict(list)
            l = len(df_contents['id'].axes[0][0]) # weird way to get the cluster amount
            for mass in df_contents['id'].items():
                for c in range(l):
                    if mass[0][c] == True and any(mass[0][0:c]) == False and any(mass[0][(c+1):l]) == False:
                        unique[c].append(mass[1])
                        break
            return unique


    def p_value_normalization(self, p):
        # p is between 1 and 0, everything below 10**(-50) is set to 1
        if p<10**(-50):
            return 1
        elif p==1:
            return 0
        else:
            return -math.log(p)/ -math.log(10**(-50))
            

    def cluster_am_normalization(self, c, cluster_amount):
        # c is the amount of other clusters, that share this mass 
        # cluster_amount: how many clusters are we using?
        return 1 - (c / (cluster_amount-1))


    def relevance_ranking(self, ks_test:dict, considered_masses:defaultdict(list), clusters:list, weight_ks:float, weight_p: float, weight_c:float, cluster_amount:int, df_ms=pd.DataFrame([])):
        """weight_p: 1/p * weight_p, weight_c: is mass only relevant in one cluster? 1/amount of occurence * weight_c  
        Einzelne Terme normalisieren mega relevant ist 1 unrelevant ist 0
        considered_masses[c] = mass1, mass2, ...
        ks_test: {mass:{cluster:(ks,p), cluster:(ks,p)}}
        return: dict: c-> sorted list of relevant masses, or add column in dataframe, if given"""
        
        if df_ms.shape==(0,0):
            sorted_relevance_by_cluster = {}
            
            for c in clusters:
                cluster_dict_relevance = {}
                for mass in considered_masses[c]:
                    ks_vlaue = ks_test[mass][c][0]
                    # here, padj needs to be used, removing multiple testing error
                    p_value = self.p_value_normalization(ks_test[mass][c][1])
                    amount_of_occurence = self.cluster_am_normalization(len(ks_test[mass]), cluster_amount=cluster_amount)
                    score = ks_vlaue * weight_ks + p_value * weight_p + amount_of_occurence * weight_c
                    score = score/(weight_ks + weight_p + weight_c)
                    cluster_dict_relevance[mass] = score
                for mass in sorted(cluster_dict_relevance, key=cluster_dict_relevance.get, reverse=True):
                    #print(mass, cluster_dict[mass]) 
                    if c in sorted_relevance_by_cluster.keys():
                        sorted_relevance_by_cluster[c].update({mass:cluster_dict_relevance[mass]})
                    else:   
                        sorted_relevance_by_cluster[c]=({mass:cluster_dict_relevance[mass]})

            return sorted_relevance_by_cluster
        
        else:
            df_ms["relevance_score"]=0
            for c in clusters:
                if c not in considered_masses.keys():
                    print(c)
                    break
                for mass in considered_masses[c]:
                    ks_vlaue = ks_test[mass][c][0]
                    # here, padj needs to be used, removing multiple testing error
                    p_value = self.p_value_normalization(ks_test[mass][c][1])
                    amount_of_occurence = self.cluster_am_normalization(len(ks_test[mass]), cluster_amount=cluster_amount)
                    score = ks_vlaue * weight_ks + p_value * weight_p + amount_of_occurence * weight_c
                    score = score/(weight_ks + weight_p + weight_c)
                    index_mass = df_ms.index[(df_ms['mass_name'] ==mass) & (df_ms['cluster_name'] == c)].tolist()[0]
                    df_ms.at[index_mass, "relevance_score"] = score
            return df_ms

    def clustering_multi_otsu(self, img_unclustered: np.ndarray, cluster_amount: int, values_for_thresholding=[], additional_th=[]):
        # returns: np.ndarray containing clustered img
        if type(values_for_thresholding)!=np.ndarray:
            """print("using img as values for thresholding")"""
            vft = img_unclustered
        else:
            vft = values_for_thresholding

        thresholds = threshold_multiotsu(vft, classes=cluster_amount)
        ths = []
        for t in additional_th:
            ths.append(t)
        for i in range(cluster_amount-1):
            ths.append(thresholds[i])
        """print(thresholds)"""
        ths = sorted(ths)
        #ths.append(254.5)
        #ths.append(255)
        #print(ths)
        img_he_clustered = np.digitize(img_unclustered, bins=ths, right=True)
        return img_he_clustered, ths

    def KS_test(self, region_array:np.array, mass:int, he_clustered_ms_shaped:np.array, cluster_range=range(5), for_clustered=True, for_unclustered=True):
        """ returns: dict: key = cluster, value = clustered statistics, clustered p value, unclutered statistics, unclustered p value"""
        # classes_clustered saves all clustered intensity values for HE categories
        classes_clustered = defaultdict(list) # mass_clustered: 0 is background

        # classes_unclusteres saves all unclustered intensities values for HE categories
        classes_unclustered = defaultdict(list) # mass_array: 0 is background
        classes_he = defaultdict(list)

        # get data for specific mass
        mass_array = region_array[:,:,mass]
        # -----do pre clustering
        background_mass = mass_array[10,10]
        values_for_thresholding = []
        for i in range(mass_array.shape[0]):
            for j in range(mass_array.shape[1]):
                pixel = mass_array[i,j]
                if pixel!=background_mass:
                    values_for_thresholding.append(pixel)
                else:
                    # value is equal to background (=0)
                    mass_array[i,j]=-1
        # -----end pre clustering
        if for_clustered:
            mass_clustered = self.clustering_multi_otsu(img_unclustered=mass_array, cluster_amount=4, values_for_thresholding=np.array(values_for_thresholding), additional_th=[0])[0]
            #  plt.imshow(mass_clustered)
            if for_unclustered:
                for row in range(mass_clustered.shape[0]):
                    for col in range(mass_clustered.shape[1]):
                        he_class = he_clustered_ms_shaped[row][col]
                        classes_he[he_class].append((row,col))
                        if not mass_clustered[row][col] == 0:
                            classes_clustered[he_class].append(mass_clustered[row][col])
                        if not mass_array[row][col] == 0:
                            classes_unclustered[he_class].append(mass_array[row][col])
                
                unclassifierd_clustered = defaultdict(list)
                unclassifierd_unclustered = defaultdict(list)
                # combine the remaining clusteres to one
                for i in cluster_range:
                    other_indices = list(cluster_range)
                    other_indices.remove(i)
                    for j in other_indices:
                        unclassifierd_clustered[i] += classes_clustered[j]
                        unclassifierd_unclustered[i] += classes_unclustered[j]

                # KS test
                ks_per_cluster = {}
                for c in cluster_range:
                    KS_test_clustered = stats.kstest(classes_clustered[c], unclassifierd_clustered[c], alternative="less")
                    KS_test_unclustered = stats.kstest(classes_unclustered[c], unclassifierd_unclustered[c], alternative="less")
                    ks_per_cluster[c] = KS_test_clustered[0], KS_test_clustered[1], KS_test_unclustered[0], KS_test_unclustered[1]
                return ks_per_cluster

            else:
                for row in range(mass_clustered.shape[0]):
                    for col in range(mass_clustered.shape[1]):
                        he_class = he_clustered_ms_shaped[row][col]
                        classes_he[he_class].append((row,col))
                        if not mass_clustered[row][col] == 0:
                            classes_clustered[he_class].append(mass_clustered[row][col])

                unclassifierd_clustered = defaultdict(list)
                # combine the remaining clusteres to one
                for i in cluster_range:
                    other_indices = list(cluster_range)
                    other_indices.remove(i)
                    for j in other_indices:
                        unclassifierd_clustered[i] += classes_clustered[j]

                # KS test
                ks_per_cluster = {}
                for c in cluster_range:
                    KS_test_clustered = stats.kstest(classes_clustered[c], unclassifierd_clustered[c], alternative="less")
                    ks_per_cluster[c] = KS_test_clustered[0], KS_test_clustered[1]
                return ks_per_cluster

        elif for_unclustered:
            for row in range(mass_clustered.shape[0]):
                for col in range(mass_clustered.shape[1]):
                    he_class = he_clustered_ms_shaped[row][col]
                    classes_he[he_class].append((row,col))
                    if not mass_array[row][col] == 0:
                        classes_unclustered[he_class].append(mass_array[row][col])
            
            unclassifierd_unclustered = defaultdict(list)
            # combine the remaining clusteres to one
            for i in cluster_range:
                other_indices = list(cluster_range)
                other_indices.remove(i)
                for j in other_indices:
                    unclassifierd_unclustered[i] += classes_unclustered[j]

            # KS test
            ks_per_cluster = {}
            for c in cluster_range:
                KS_test_unclustered = stats.kstest(classes_unclustered[c], unclassifierd_unclustered[c], alternative="less")
                ks_per_cluster[c] = KS_test_unclustered[0], KS_test_unclustered[1]
            return ks_per_cluster
            
        
    def find_relevant_masses_by_ks(self, region_array, masses, he_clustered_ms_shaped, cluster_range=range(5), a=0.05, th_ks=0.1, for_clustered=True, for_unclustered=True):
        """returns two dictionarys: clustered and unclustered save all masses that have a p-value<a and ks >0.1 with results"""
        if for_clustered and for_unclustered:
            ks_test_clustered = {}
            ks_test_unclustered = {}
            for mass in masses:
                ks_of_mass = self.KS_test(region_array, mass, he_clustered_ms_shaped=he_clustered_ms_shaped, cluster_range=cluster_range)
                for c in cluster_range:
                    if ks_of_mass[c][1]< a and ks_of_mass[c][0] > th_ks:
                        ks_test_clustered[(mass, c)] = (ks_of_mass[c][0], ks_of_mass[c][1])
                    if ks_of_mass[c][3]< a and ks_of_mass[c][2] > th_ks:
                        ks_test_unclustered[(mass, c)] = (ks_of_mass[c][2], ks_of_mass[c][3])
            return ks_test_clustered, ks_test_unclustered

        elif for_unclustered:
            ks_test_unclustered = {}
            for mass in masses:
                # modify: so that only unclustered is calculated
                ks_of_mass = self.KS_test(region_array, mass, he_clustered_ms_shaped=he_clustered_ms_shaped, cluster_range=cluster_range)
                for c in cluster_range:
                    if ks_of_mass[c][1]< a and ks_of_mass[c][0] > th_ks:
                        ks_test_unclustered[(mass, c)] = (ks_of_mass[c][2], ks_of_mass[c][3])
            return ks_test_unclustered

        elif for_clustered:
            ks_test_clustered = {}
            for mass in masses:
                # modify: so that only clustered is calculated
                ks_of_mass = self.KS_test(region_array, mass, he_clustered_ms_shaped=he_clustered_ms_shaped, cluster_range=cluster_range)
                for c in cluster_range:
                    if ks_of_mass[c][1]< a and ks_of_mass[c][0] > th_ks:
                        ks_test_clustered[(mass, c)] = (ks_of_mass[c][0], ks_of_mass[c][1])
            return ks_test_clustered, ks_test_unclustered
        
        return None


    def dict_to_dataframe(self, ks_dict):
        my_data = defaultdict(list)

        for key in ks_dict:
            my_data["mass_name"].append(key[0])
            my_data["cluster_name"].append(key[1])
            my_data["ks_value"].append(ks_dict[key][0])
            my_data["p_value"].append(ks_dict[key][1])
            my_data["p_value_mult"].append(ks_dict[key][1]*5*16000)

        ks_dataframe = pd.DataFrame.from_dict(my_data)
        return ks_dataframe


def clustering_multi_otsu(img_unclustered: np.ndarray, cluster_amount: int, values_for_thresholding=[]):
    # returns: np.ndarray containing clustered img
    if type(values_for_thresholding)!=np.ndarray:
        """print("using img as values for thresholding")"""
        values_for_thresholding = img_unclustered
    thresholds = threshold_multiotsu(values_for_thresholding, classes=cluster_amount)
    ths = []
    for i in range(cluster_amount-1):
        ths.append(thresholds[i])
    """print(thresholds)"""
    ths.append(254.5)
    ths.append(255)
    print(ths)
    img_he_clustered = np.digitize(img_unclustered, bins=ths, right=True)
    return img_he_clustered, ths
