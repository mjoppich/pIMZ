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
from skimage import filters
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
from matplotlib.collections import LineCollection

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
from statistics import mean


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
import random
from sklearn.cluster import KMeans
from scipy.spatial import distance


#TODO:



def KS_test( mass_array:np.array, clustering:np.array, cluster_considered=None, background_mass=0, general_background=False, mass=0, method=stats.kstest, th_t=10, a=0.05, th_ks=0.3, th_ws=10):
    """ returns: dict: key = cluster, value = clustered statistics, clustered p value, unclutered statistics, unclustered p value

        ttest wrong implemented... -> results are empty/ no masses meet conditions:
        add_mass = [c for idx,(ks, p) in enumerate(ks_per_cluster[c]) if (abs(ks)< stats.t.ppf(1.0 - a, deg_free[c][idx]) and p < a)] #for ttest----
        lines outcommented:  ttest----   
    
    """
    
    # classes_clustered saves all clustered intensity values for HE categories
    classes_clustered = defaultdict(list) # mass_clustered: 0 is background

    # classes_unclusteres saves all unclustered intensities values for HE categories
    classes_unclustered = defaultdict(list) # mass_array: 0 is background
    classes_he = defaultdict(list)

    cluster_range = np.unique(clustering)
    
    if cluster_considered is None:
        cluster_considered = cluster_range
    
    # -----do pre clustering
    values_for_thresholding = mass_array[clustering != background_mass]
    for c in cluster_range:
        classes_he[c] = [tuple(x) for x in np.argwhere(clustering == c)]
        
        relevant_intensities = mass_array[ clustering == c ]
        classes_unclustered[c] = relevant_intensities[relevant_intensities != 0]
    unclassifierd_unclustered = defaultdict(list)
    #print(f'cluster 1:{classes_unclustered[1]}')

    # KS test nur gegen einen cluster: cluster considered
    if type(cluster_considered)!=type(cluster_range):
        # compare expression in one cluster to every other seperatly-> intersection
        ks_per_cluster=defaultdict(list)
        # foreground cluster is consdered cluster
        foreground_intensities = classes_unclustered[cluster_considered]
        for c in cluster_range:
            #background cluster is c
            background_intensities = classes_unclustered[c]
            if len(foreground_intensities) == 0:
                pass
                #ks_per_cluster[c] = None, 1.0
            elif len(background_intensities)==0:
                # no intensities in background
                ks_per_cluster[cluster_considered].append((1, 0))

            else:
                #KS_test_unclustered = stats.kstest(foreground_intensities, background_intensities, alternative="less")
                KS_test_unclustered = stats.kstest(foreground_intensities, background_intensities, alternative="less")
                ks_per_cluster[cluster_considered].append((KS_test_unclustered[0], KS_test_unclustered[1]))

        return ks_per_cluster

    # KS test
    if general_background:
        ks_per_cluster = {}
        background_intensities = np.concatenate( [classes_unclustered[x] for x in cluster_range ] ) #if x != c
        
        # compare expression in one custer compared to rest
        for c in cluster_considered:
            foreground_intensities = classes_unclustered[c]
            
            if len(foreground_intensities) == 0 or len(background_intensities) == 0:
                ks_per_cluster[c] = None, 1.0
            else:
                #KS_test_unclustered = method(foreground_intensities, background_intensities, alternative="less")
                KS_test_unclustered = method(foreground_intensities, background_intensities, alternative="less")
                ks_per_cluster[c] = KS_test_unclustered[0], KS_test_unclustered[1]

        return ks_per_cluster
    
    else:
        # compare expression in one cluster to every other seperatly-> intersection
        #background_intensities = np.concatenate( [classes_unclustered[x] for x in cluster_range ] ) #if x != c
        ks_per_cluster=dict()

        if method==stats.wilcoxon:
            # find samplesize of smallest cluster
            amount_classes_unclustered = {}
            for c in classes_unclustered.keys():
                amount_classes_unclustered[c] = len(classes_unclustered[c])
            smallest_cluster_size = min(amount_classes_unclustered.values())
            # random sampling to get equal sizes
            wilcoxon_random_draw = {}
            for c in classes_unclustered.keys():
                print("random sampling")
                print(type(classes_unclustered[c].tolist()))
                print(smallest_cluster_size)
                random_draw = random.sample(classes_unclustered[c].tolist(), smallest_cluster_size)
                wilcoxon_random_draw[c] = random_draw

            for c in cluster_considered:
                ks_per_cluster[c] = [(0, 1)] * len(cluster_considered)

                foreground_intensities = wilcoxon_random_draw[c]
                for bi, b in enumerate(cluster_considered):
                    if c == b:
                        continue
                    background_intensities = wilcoxon_random_draw[b]
                    
                    if len(foreground_intensities) == 0 or len(background_intensities) == 0:
                        pass
                    else:
                        KS_test_unclustered = method(foreground_intensities, background_intensities, alternative="greater")# wilcoxon----
                        ks_per_cluster[c][bi] = (KS_test_unclustered[0], KS_test_unclustered[1])
            

        else:
            for c in cluster_considered:
                ks_per_cluster[c] = [(0, 1)] * len(cluster_considered)

                foreground_intensities = classes_unclustered[c]
                for bi, b in enumerate(cluster_considered):
                    if c == b:
                        continue
                    background_intensities = classes_unclustered[b]
                    
                    if len(foreground_intensities) == 0 :
                        pass
                    elif len(background_intensities) == 0:
                        ks_per_cluster[c][bi] = (th_t+1, 0)
                    else:
                        if method==stats.kstest:
                            KS_test_unclustered = method(foreground_intensities, background_intensities, alternative="less")
                        elif method==stats.ttest_ind:
                            KS_test_unclustered = method(foreground_intensities, background_intensities, alternative="greater")# ttest----
                        ks_per_cluster[c][bi] = (KS_test_unclustered[0], KS_test_unclustered[1])
                        #print(f'ks,p: {KS_test_unclustered[0], KS_test_unclustered[1]}')
            
        #print(f"mass past else: {mass}")
        #th_ks = 0.3
        #th_ws = 0.3
        #th_t = 20
        #a = 0.001
        #a = 0.05
        background_amount= len(cluster_considered)-1
        ks_per_cluster_sig=defaultdict(list)
        
        for c in ks_per_cluster.keys():
            #add_mass = [c for (ks, p) in ks_per_cluster[c] if (ks > 0 and p < 1)] 
            if method==stats.kstest:
                add_mass = [c for (ks, p) in ks_per_cluster[c] if (ks > th_ks and p < a)]
            elif method==stats.ttest_ind:
                add_mass = [c for (ks, p) in ks_per_cluster[c] if (abs(ks) > th_t  and p < a)]
            elif method==stats.wilcoxon:
                add_mass = [c for (ks, p) in ks_per_cluster[c] if (abs(ks) > th_ws  and p < a)]
            if len(add_mass)==background_amount:
                #print(f"{mass}: true")
                # add the mean 
                ks_stat_mean = mean([i[0] for i in ks_per_cluster[c] if i[0]!=0])
                p_value_mean = mean([i[1] for i in ks_per_cluster[c] if i[1]!=1])
                ks_per_cluster_sig[c].append((ks_stat_mean, p_value_mean))
            elif len(add_mass)>(background_amount*0.85):
                pass
            
        #print(f"mass end: {mass}") 

        return ks_per_cluster_sig
    



class KsSimilarity:

    def __init__(self, spec_mz_values:SpectraRegion, initial_cl_metaseg=False):
        self.spec = spec_mz_values
        # to do mz_region_array = self.spec.region_array
        self.region_array = np.copy(self.spec.region_array)
        if initial_cl_metaseg:
            self.initial_clustering = spec_mz_values.meta['segmented'].copy()
            self.initial_clustering = self.initial_clustering +1
        else:
            self.initial_clustering = None
        self.core_pixel = None
        self.clusterwise_core_pixel = defaultdict(list) # for each cluster, append all tuples (r,c) that are core
        self.clustering = None
        self.core_similarity_clustering = None
        self.core_similarity_dict=None
        self.filled_clustering = None
        self.quality_masses = None
        self.quality_pixel = None
        self.count=0
        self.data_summary = None # find relevant masses
        self.analysis_results = None # save perform_analysis data here
        self.cluster_distances = None

    def perform_analysis(self, passed_clustering=np.empty([0, 0]), weight_ks=1, weight_p=1, weight_c=1, features=[], iterative=False, th_t=10, a=0.05, method=stats.ttest_ind, raw=False):
        """ This function finds masses that represent a certein cluster 
        
            input:  mendatory:  - spec_mz_values: mz values in SpectraRegion format
                    optional:   - clustering: array, default is spec_mz_values.meta["segmented"]
                                - weights: floats 
                                - features: indices of masses that analysis should be done on, default all
                                - iterative: Use itaerative process when performing analysis 
                                    to find core pixels(=most homogene tissue) and relevant masses per cluster

            return: dataframe with all information for masses per cluster such as 
                    - ks statistics
                    - p-values
                    - relevance score
        """
        self.initial_clustering = passed_clustering

        if passed_clustering.shape != self.region_array.shape[0:2]:
            print("Taking spec clustering")
            self.clustering = self.spec.meta["segmented"].copy() # initial clustering for iterative process
            self.clustering = self.clustering + 1

            self.clustering = self.clustering.astype(int)
        else:
            print("Saving new clustering")
            self.clustering = passed_clustering

        clusters=np.unique(self.clustering)


        if features == []:
            features = range(self.region_array.shape[2])

        considered_masses = self.excluding_outliers(self.region_array, features)
        print("Identified {} of {} masses for processing.".format(len(considered_masses), self.region_array.shape[2]))
        
        ks_test_unclustered = self.find_relevant_masses_by_ks(self.region_array, considered_masses, self.clustering, cluster_range=clusters, for_clustered=False, iterative=iterative, th_t=th_t, a=a, method=method)
        # mass, cluster, ks, p, p_mult
        # TODO only regard pixels != 0 in clustering
        if iterative:
            ms_df = self.dict_to_dataframe(ks_test_unclustered[0]) 
            core_clustering = ks_test_unclustered[1]
            print(f"ks_test_unclustered[1].shape: {core_clustering.shape}")
            print(f"region.array:{self.region_array.shape}")
            plt.imshow(core_clustering)
            # ---------self.clustering: replacing unused pixels by -1. Actually: replace the self.clustering completely
            for r in range(self.clustering.shape[0]):
                for c in range(self.clustering.shape[1]):
                    if core_clustering[r,c] == 0:
                        self.clustering[r,c] = -1
            # ---------
            if raw:
                return ms_df
        else:
            ms_df = self.dict_to_dataframe(ks_test_unclustered) 

        if not 'p_value' in ms_df.columns:
            print("not 'p_value' in ms_df.columns")
            return ms_df

        # multiple testing correction
        rej, elemAdjPvals, _, _ = multipletests(ms_df["p_value"], alpha=a, method='fdr_bh', is_sorted=False, returnsorted=False)
        ms_df["p_value_adj"] = elemAdjPvals
        # dict: mass is key -> needed lateron for relevence ranking
        ks_p_dict = self.dataframe_to_mass_dict(ms_df)
        # dict: cluster is key
        rel_masses_for_cluster = self.dataframe_to_cluster_dict(ms_df)
        # add column: is unique?
        #ms_df = self.find_unique_masses_for_cluster(rel_masses_for_cluster, ms_df)
        # add score for relevance
        self.analysis_results = ms_df 
        ms_df = self.relevance_ranking(ks_test=ks_p_dict, considered_masses=rel_masses_for_cluster, clusters=clusters, weight_ks=weight_ks, weight_p=weight_p, weight_c=weight_c, cluster_amount=len(clusters), df_ms=ms_df)
        self.analysis_results = ms_df 
        return ms_df

    def excluding_outliers(self, ms_region_array:np.array, masses):
        """input: three dimensional array with mass spec data
            returns: dictionary, mass:threshold, all masses where more than 5% of the pixels are expressed"""
        considered_masses = set()
        
        print("Excluding Outliers")
        bar = makeProgressBar()
                
        for mass in bar(masses):
            list_mass = ms_region_array[:,:,mass].flatten()
            
            expr_pixels = np.sum(list_mass>0)
            if expr_pixels > len(list_mass)*0.05:
                considered_masses.add(mass)
                
            continue
            
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
            p = ms_df["p_value_adj"][i]
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
        """ NOT REALLY WORKING
        input: dict: cluster=masses 
            return: dataframe, unique added OR dict cluster=unique_masses"""

        clusters = [int(item) for item in list(rel_masses_for_cluster.keys())]

        df_contents = from_contents(rel_masses_for_cluster)# changing mass names to 0, 1, 2...
        if not df_ms.shape==(0,0):
            df_ms['unique_for_cluster']=False
            if type(df_contents['id'].axes[0][0])==tuple:
                l = len(df_contents['id'].axes[0][0]) # weird way to get the cluster amount
                for mass in df_contents['id'].items():
                    for c in range(l):
                        if mass[0][c] == True and any(mass[0][0:c]) == False and any(mass[0][(c+1):l]) == False:
                            
                            index_mass = df_ms.index[(df_ms['mass_name'] == mass[1]) & (df_ms['cluster_name'] == c)].tolist()
                            if len(index_mass)==0:
                                #print(f"mass: {mass[1]}, cluster: {c} not found in df")
                                break
                            elif len(index_mass)>1:
                                #print(f"mass: {mass[1]}, cluster: {c} multilpe times in df: index_mass")
                                break
                            else:
                                df_ms.at[index_mass[0], "unique_for_cluster"] = True
                            break
            else:
                l = 1
                for mass in df_contents['id'].items():
                    if mass[0][1] == True:
                        index_mass = df_ms.index[(df_ms['mass_name'] == mass[1]) & (df_ms['cluster_name'] == c)].tolist()
                        if len(index_mass)==0:
                            break
                        elif len(index_mass)>1:
                            break
                        else:
                            df_ms.at[index_mass[0], "unique_for_cluster"] = True
                        break
                        
            return df_ms

        else:
            unique = defaultdict(list)
            #l = len(df_contents['level_0'].axes[0][0]) # weird way to get the cluster amount
            #for mass in df_contents['level_0'].items():
            l = len(df_contents['id'].axes[0][0]) # weird way to get the cluster amount
            for mass in df_contents['id'].items():
                for c in range(l):
                    cluster_number = clusters[c]
                    if mass[0][c] == True and any(mass[0][0:c]) == False and any(mass[0][(c+1):l]) == False:
                        unique[cluster_number].append(mass[1])
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


    def ks_value_normalization(self, ks):
        if ks<0:
            return 0
        max_ks = max(self.analysis_results["ks_value"])
        return ks/max_ks
    
    
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
                    ks_value = ks_test[mass][c][0]
                    ks_value = self.ks_value_normalization(ks_value)
                    # here, padj needs to be used, removing multiple testing error
                    idx = self.analysis_results.index[(self.analysis_results['mass_name']==mass)] & self.analysis_results.index[(self.analysis_results['cluster_name']==c)]
                    idx = list(idx)[0]
                    p_value_adj = self.analysis_results["p_value_adj"][idx]
                    p_value_adj = self.p_value_normalization(p_value_adj)
                    amount_of_occurence = self.cluster_am_normalization(len(ks_test[mass]), cluster_amount=cluster_amount)
                    score = ks_value * weight_ks + p_value_adj * weight_p + amount_of_occurence * weight_c
                    score = score/(weight_ks + weight_p + weight_c)
                    cluster_dict_relevance[mass] = score
                for mass in sorted(cluster_dict_relevance, key=cluster_dict_relevance.get, reverse=True):
                    if c in sorted_relevance_by_cluster.keys():
                        sorted_relevance_by_cluster[c].update({mass:cluster_dict_relevance[mass]})
                    else:   
                        sorted_relevance_by_cluster[c]=({mass:cluster_dict_relevance[mass]})

            return sorted_relevance_by_cluster
        
        else:
            df_ms["relevance_score"]=0
            for c in clusters:
                if c not in considered_masses.keys():
                    continue
                for mass in considered_masses[c]:
                    ks_value = ks_test[mass][c][0]
                    ks_value = self.ks_value_normalization(ks_value)
                    # here, padj needs to be used, removing multiple testing error
                    idx = self.analysis_results.index[(self.analysis_results['mass_name']==mass)] & self.analysis_results.index[(self.analysis_results['cluster_name']==c)]
                    idx = list(idx)[0]
                    p_value_adj = self.analysis_results["p_value_adj"][idx]
                    p_value_adj = self.p_value_normalization(p_value_adj)

                    #p_value_adj = self.p_value_normalization(ks_test[mass][c][1])
                    amount_of_occurence = self.cluster_am_normalization(len(ks_test[mass]), cluster_amount=cluster_amount)
                    score = ks_value * weight_ks + p_value_adj * weight_p + amount_of_occurence * weight_c
                    score = score/(weight_ks + weight_p + weight_c)
                    index_mass = df_ms.index[(df_ms['mass_name'] ==mass) & (df_ms['cluster_name'] == c)].tolist()[0]
                    df_ms.at[index_mass, "relevance_score"] = score
            return df_ms


    def clustering_multi_otsu(self, img_unclustered: np.ndarray, cluster_amount: int, values_for_thresholding=[], additional_th=[]):
        # returns: np.ndarray containing clustered img
        if type(values_for_thresholding)!=np.ndarray:
            vft = img_unclustered
        else:
            vft = values_for_thresholding

        thresholds = threshold_multiotsu(vft, classes=cluster_amount)
        ths = []
        for t in additional_th:
            ths.append(t)
        for i in range(cluster_amount-1):
            ths.append(thresholds[i])
        ths = sorted(ths)
        img_he_clustered = np.digitize(img_unclustered, bins=ths, right=True)
        return img_he_clustered, ths
    
            
        
    def find_relevant_masses_by_ks(self, region_array, masses, clustering, cluster_range=range(5), a=0.05, th_t=10, th_ks=0.5, for_clustered=True, for_unclustered=True, iterative=False, method=stats.ttest_ind, parallel=False):
        """returns two dictionarys: clustered and unclustered save all masses that have a p-value<a and ks >0.1 with results"""
        if for_clustered and for_unclustered:
            return None
        elif for_unclustered:
                
            ks_test_unclustered = {}
            
            masses = list(masses)
            #method=stats.ttest_ind
            #method=stats.wilcoxon
            print(f"method: {method}")
            if parallel:

                from joblib import Parallel, delayed, parallel_backend
                print("in loky")
                with parallel_backend('loky', n_jobs=8):
                    results = Parallel()(delayed(KS_test)(region_array[:,:, mass], clustering=clustering, general_background=False, mass=mass, method=method, th_t=th_t, a=a) for mass in masses)
    
                    background_amount= len(np.unique(clustering))-1
                    for idx, res in enumerate(results): #???
                    #for res in results:
                        for c in res:
                            # ks_test_unclustered[(res, c)] = (res[c][0], res[c][1])
                            # ks_test_unclustered[(masses[idx], c)] = (res[c][0], res[c][1]) #???
                            ks_test_unclustered[(masses[idx], c)] = (res[c][0][0], res[c][0][1]) #???

                self.data_summary = ks_test_unclustered

                return ks_test_unclustered
            else: 
        
                bar = makeProgressBar()
                results = (KS_test(region_array[:,:, mass], clustering=clustering, general_background=False, mass=mass, method=method, th_t=th_t, a=a) for mass in bar(masses))

                background_amount= len(np.unique(clustering))-1
                for idx, res in enumerate(results): #???
                #for res in results:
                    for c in res:
                        # ks_test_unclustered[(res, c)] = (res[c][0], res[c][1])
                        # ks_test_unclustered[(masses[idx], c)] = (res[c][0], res[c][1]) #???
                        ks_test_unclustered[(masses[idx], c)] = (res[c][0][0], res[c][0][1]) #???

                self.data_summary = ks_test_unclustered

                return ks_test_unclustered

        elif for_clustered:
            return None
        
        return None


    def iterative_cluster_assignment(self, clustering, th_t, a, method=stats.ttest_ind, max_iteration=5, raw=False):
        
        current_clustering = clustering.copy()
        if 0 in current_clustering.flatten():
            print('current_clustering = current_clustering+1,  as clustering contains 0')
            current_clustering = current_clustering+1
        last_clustering = np.zeros( current_clustering.shape )
        
        pixels = clustering.size
                     
        current_iteration = 0

        quanlity_analysis_pixel = defaultdict(list)# defaultdicts 1-10
        quanlity_analysis_masses = defaultdict(list)# defaultdicts 1-10
            
              
        while current_iteration < max_iteration and ( pixels-np.sum(current_clustering == last_clustering) ) > 10:
            
            # find cluster-specific masses
            #ms_df = self.perform_analysis()
            
            print("Current Iteration", current_iteration)
            
            availClusters = np.unique(current_clustering)            
            cluster2masses = defaultdict(list)
            
            last_clustering = np.copy(current_clustering)
            
            print("Performing KS Analysis for iteration {}".format(current_iteration))
            #TODO i think this is needed!
            #use_clustering = (current_clustering-1)
            #use_clustering[use_clustering < 0] = 0
            ms_df = self.perform_analysis(passed_clustering=np.copy(current_clustering),th_t=th_t, a=a, method=method, raw=raw)
            
            if not 'p_value' in ms_df.columns:
                print("not 'p_value' in ms_df.columns")
                break
            
            # quality statistics
            for c in np.unique(current_clustering):
                m = np.unique(ms_df.loc[(ms_df['cluster_name'] == c)]["mass_name"])
                quanlity_analysis_masses[c].append(m)
                p = np.sum(current_clustering == c)
                quanlity_analysis_pixel[c].append(p)

            for clusterID in availClusters:
                sub_df = ms_df[(ms_df.cluster_name == clusterID)].sort_values("relevance_score", ascending=False).head(100)
                cluster2masses[clusterID] = list(sub_df.mass_name)
                
                
            current_clustering = np.zeros(current_clustering.shape)
            for clusterID in cluster2masses:
                
                pixel2high_expressed = np.zeros( last_clustering.shape )
                
                for mass in cluster2masses[clusterID]:
                    mass_array = np.copy(self.region_array[:,:,mass])
                    
                    #intensity threshold
                    intensityThreshold = np.quantile(mass_array, 0.5) #[ current_clustering == clusterID ]
                    pixel2high_expressed += (mass_array>intensityThreshold)
                    
                massThreshold = np.floor( 0.9 * len(cluster2masses[clusterID]) )
                
                current_clustering[(last_clustering == clusterID) & (pixel2high_expressed >= massThreshold)] = clusterID
  
            
            current_iteration += 1  

            plt.imshow(current_clustering)
            plt.show()
            plt.close()
            
            print("Equal pixels", np.sum(current_clustering == last_clustering))
            print("Equal pixels %", np.sum(current_clustering == last_clustering)/pixels)
            self.core_pixel = np.sum(current_clustering == self.initial_clustering)
            # add clusterwise corepixel
            self.find_clusterwise_core_pixel()

        for c in np.unique(current_clustering):
            p = np.sum(current_clustering == c)
            quanlity_analysis_pixel[c].append(p)
        
        self.quality_masses = quanlity_analysis_masses
        self.quality_pixel = quanlity_analysis_pixel
        self.clustering = current_clustering
                
        return current_clustering
        

    def dict_to_dataframe(self, ks_dict):
        my_data = defaultdict(list)

        m2c = defaultdict(set)
        for key in ks_dict:
            mass = key[0]
            cluster = key[1]
            m2c[mass].add(cluster)

        for key in ks_dict:
            my_data["mass_name"].append(key[0])
            my_data["cluster_name"].append(key[1])
            my_data["ks_value"].append(ks_dict[key][0])
            my_data["p_value"].append(ks_dict[key][1])
            #my_data["p_value_adj"].append(ks_dict[key][1]*5*16000)
            my_data["rel_in_clusters"].append(tuple(m2c[key[0]]))
            my_data["unique_for_cluster"].append(len(m2c[key[0]])==1)

        ks_dataframe = pd.DataFrame.from_dict(my_data)
        return ks_dataframe

    
    def remove_similar_masses(self, mass_window_range=20):
        """in a window range looking for the mass with the highest score, only keeping this one
            changing dataframe self.analysis_reslts (removing rows)"""
        
        masses = self.analysis_results['mass_name']
        close_masses = []
        for m in range(masses[0], masses[len(masses)-1]+mass_window_range, mass_window_range):
            close_masses.append([mass for mass in masses if (mass<m+mass_window_range/2 and mass>=m-mass_window_range/2)])

        close_masses = [x for x in close_masses if x != []]
        
        keep_masses = []
        for close_m in close_masses:
            scores= list(self.analysis_results[self.analysis_results['mass_name'].isin(close_m)]["relevance_score"])
            max_score = max(scores)
            max_index = scores.index(max_score)
            mass = list(self.analysis_results[self.analysis_results['mass_name'].isin(close_m)]["mass_name"])[max_index]
            keep_masses.append(mass)#masses with highest score

        self.analysis_results = self.analysis_results[self.analysis_results['mass_name'].isin(keep_masses)]

    
    
    def statistics_of_mass(self, masses_w_clusters=(), percentile=75, add_to_df=True, for_all=True):
        """(concerning one cluster use intensity threshhold)
            masses_w_clusters: list of tuples
            percentile: percentile mz value for mass-> = threhhold
            add_to_df=False: returns dict: (mass,cluster) = (sensitivity,specificity) 
            add_to_df=True: adds to self.analysis_results
                            masses that are not in analysis_results are just skipped"""
        
        highest_cluster = max(np.unique(self.clustering))
        lowest_cluster = min(np.unique(self.clustering))
        
        if add_to_df:
            print("adding statistics for masses to results df")
            if not 'sensitivity' in self.analysis_results.columns:
                self.analysis_results['sensitivity'] = "NaN"
            if not 'specificity' in self.analysis_results.columns:
                self.analysis_results['specificity'] = "NaN"
            if not 'false_neg_rate' in self.analysis_results.columns:
                self.analysis_results['false_neg_rate'] = "NaN"
            if not 'false_pos_rate' in self.analysis_results.columns:
                self.analysis_results['false_pos_rate'] = "NaN"


            if for_all:
                for ind in self.analysis_results.index:
                    mass = self.analysis_results["mass_name"][ind]
                    cluster = self.analysis_results["cluster_name"][ind]

                    mass_int = self.region_array[:,:,mass]
                    i_th = np.percentile(mass_int, percentile)

                    # binary expression
                    #binary_mass = np.full(mass_int.shape, lowest_cluster-10) #both no signal
                    #binary_mass[mass_int > i_th] = cluster #signal only in mass
                    #binary_mass[self.clustering == binary_mass] = 2*highest_cluster #both signal
                    #binary_mass[(self.clustering == cluster) & (binary_mass==lowest_cluster-10)] = 3*highest_cluster #singnal only in cluster

                    binary_mass = np.zeros(mass_int.shape)
                    binary_mass[mass_int > i_th] = 1
                    
                    #plt.imshow(binary_mass)

                    both_signal = np.sum(binary_mass == 2*highest_cluster)
                    both_no_signal = np.sum(binary_mass == lowest_cluster-10)
                    cluster_signal = np.sum(self.clustering == cluster)
                    cluster_no_signal = np.sum(self.clustering != cluster)

                    sensitivity = both_signal/cluster_signal
                    if mass==7828:
                        print(both_signal, cluster_signal)
                    specificity = both_no_signal/cluster_no_signal

                    sensitivity = np.sum( (binary_mass == 1) & (self.clustering == cluster) ) / np.sum(self.clustering == cluster)
                    specificity = np.sum( (binary_mass == 0) & (self.clustering != cluster) ) / np.sum(self.clustering != cluster)
                    false_pos_rate = np.sum( (binary_mass == 1) & (self.clustering != cluster) ) / np.sum(self.clustering != cluster)
                    false_neg_rate = np.sum( (binary_mass == 0) & (self.clustering == cluster) ) / np.sum(self.clustering == cluster)

                    self.analysis_results['sensitivity'][ind] = sensitivity
                    self.analysis_results['specificity'][ind] = specificity
                    self.analysis_results['false_pos_rate'][ind] = false_pos_rate
                    self.analysis_results['false_neg_rate'][ind] = false_neg_rate


            else: # add to df, only specific
                for mass, cluster in masses_w_clusters:
                    # find indexes:
                    ind = self.analysis_results.index[self.analysis_results['mass_name']==mass and self.analysis_results['cluster_name']==cluster].tolist()
                    if len(ind)==0:
                        continue
                    
                    mass_int = self.region_array[:,:,mass]
                    i_th = np.percentile(mass_int, percentile)


                    # binary expression
                    binary_mass = np.full(mass_int.shape, lowest_cluster-10) #both no signal
                    binary_mass[mass_int > i_th] = cluster #signal only in mass
                    binary_mass[self.clustering == binary_mass] = 2*highest_cluster #both signal
                    binary_mass[(self.clustering == cluster) & (binary_mass==lowest_cluster-10)] = 3*highest_cluster #singnal only in cluster
                    
                    plt.imshow(binary_mass)

                    both_signal = np.sum(binary_mass == 2*highest_cluster)
                    both_no_signal = np.sum(binary_mass == lowest_cluster-10)
                    cluster_signal = np.sum(self.clustering == cluster)
                    cluster_no_signal = np.sum(self.clustering != cluster)

                    sensitivity = both_signal/cluster_signal
                    specificity = both_no_signal/cluster_no_signal

                    self.analysis_results['sensitivity',ind[0]] = sensitivity
                    self.analysis_results['specificity',ind[0]] = specificity

            
        else:#return
            mass_stat = {}
            for mass, cluster in masses_w_clusters:
                mass_int = self.region_array[:,:,mass]
                i_th = np.percentile(mass_int, percentile)

                # binary expression
                binary_mass = np.full(mass_int.shape, lowest_cluster-10) #both no signal
                binary_mass[mass_int > i_th] = cluster #signal only in mass
                binary_mass[self.clustering == binary_mass] = 2*highest_cluster #both signal
                binary_mass[(self.clustering == cluster) & (binary_mass==lowest_cluster-10)] = 3*highest_cluster #singnal only in cluster
                
                plt.imshow(binary_mass)

                both_signal = np.sum(binary_mass == 2*highest_cluster)
                both_no_signal = np.sum(binary_mass == lowest_cluster-10)
                cluster_signal = np.sum(self.clustering == cluster)
                cluster_no_signal = np.sum(self.clustering != cluster)

                sensitivity = both_signal/cluster_signal
                specificity = both_no_signal/cluster_no_signal

                mass_stat[(mass, cluster)] = (sensitivity, specificity)

            return mass_stat


    def statistics_of_cluster(self):
        """returns dict: cluster-> (sens, spez)
                            sens, spez avg over all found masses for that cluster"""
        cluster_to_masses = self.found_masses_for_cluster()
        cluster_to_statistics = {}
        for c in cluster_to_masses.keys():
            masses = cluster_to_masses[c][1]
            sens = 0
            spez = 0
            for idx in masses.index:
                sens += self.analysis_results['sensitivity'][idx]
                spez += self.analysis_results['specificity'][idx]
            sens = sens/cluster_to_masses[c][0]
            spez = spez/cluster_to_masses[c][0]
            cluster_to_statistics[c] = {sens, spez}
        return cluster_to_statistics


    def found_masses_for_cluster(self, len_only=False):
        """returns dict: cluster-> ( len(masses), (index value) )
        (index value) is a pandas series. get indices by using cluster_to_masses[c][1].index 
                                            get values by unsing cluster_to_masses[c][1].values"""
        cluster_to_masses = {}
        
        if len_only:
            for c in np.unique(self.analysis_results["cluster_name"]):
                masses = self.analysis_results[self.analysis_results["cluster_name"]==c]["mass_name"]
                l = len(masses)
                cluster_to_masses[c] = l
        else:       
            for c in np.unique(self.analysis_results["cluster_name"]):
                masses = self.analysis_results[self.analysis_results["cluster_name"]==c]["mass_name"]
                l = len(masses)
                cluster_to_masses[c] = (l,masses)
        return cluster_to_masses

    def show_one_cluster(self, cl_to_show, use_ititial=True, show=True):
        if use_ititial:
            to_show = np.copy(self.initial_clustering)
        else:
            to_show = np.copy(self.clustering)
        to_show[to_show!=cl_to_show] = 0
        if show:
            plt.imshow(to_show)
        return to_show


    def connected_cluster_size(self,test_clustering="initial"):
        """ calculates sizes of connected clusters
            parameter:  test_initial_clustering==True -> cluster that was used to initialize iterative cluster asignment
                        test_initial_clustering==False-> calculated cluster
            returns:    avg size of connected clusters
                        dict: occuring sizes -> count of occurence"""
        if test_clustering== "initial":
            tested_clustering = self.initial_clustering
        elif test_clustering== "core":
            tested_clustering = self.clustering
        elif  test_clustering== "filled":
            tested_clustering = self.filled_clustering
        else: 
            return None
        connected_clusters = sk_measure.label(tested_clustering)
        cc_size = {}
        for cc in np.unique(connected_clusters):
            count = np.count_nonzero(connected_clusters == cc)
            cc_size[cc] = count

        avg_cluster_size = mean(list(cc_size.values()))
    
        size_count = {}
        for size in np.unique(list(cc_size.values())):
            size_count[size] = 0
            

        for cluster in cc_size.keys():
            size = cc_size[cluster]
            size_count[size] += 1

        return len(np.unique(connected_clusters)), avg_cluster_size, size_count
    

    def density(self, radius:int, circular=False, test_clustering="initial", plotting=False):
        """computes average share of surrounding area for pixels of each cluster.
        
        Parameters: - clustering: np.ndarray, the clustering that is tested for cluster density
                    - radius: the radius that determines the size of the search area around each pixel
                    - circular: if true: compute euklidean dist to determine wether in area
                                else: take rectangular area
                    
        returns:    - dict: cluster 
                        -> avg share of same cluster pixels over all pixels in that cluster in given radius."""
        
        clustering_dict = {'initial':self.initial_clustering, 'core':self.clustering, 'filled':self.filled_clustering}
        tested_clustering = clustering_dict[test_clustering]

        density_dict = defaultdict(list)
        for row in range(tested_clustering.shape[0]):
            for col in range(tested_clustering.shape[1]):
                cl = tested_clustering[row, col]
                surrounding_clusters = get_surrounding_area(tested_clustering, (row,col), radius=radius, circular=circular)
                same_cl = sum(c == cl for c in surrounding_clusters)
                share = same_cl/len(surrounding_clusters)
                density_dict[cl].append(share)
        
        avg_density_dict = {}
        for cl in density_dict.keys():
            avg_density_dict[cl] = mean(density_dict[cl])

        if plotting:
            df = pd.DataFrame([(k,i) for k in density_dict.keys() for i in density_dict[k]], 
                              columns=['cluster','density'])
            sns.violinplot(data=df, x="cluster", y="density", linewidth=0.5, width=0.95)

        return avg_density_dict
    
    
    def core_pixel_per_cluster(self):
        """returns: dict: cluster-> (amount_core_pixel, share_core_pixel)"""
        self.initial_clustering
        self.clustering
        cl_to_core = {}
        for cl in np.unique(self.initial_clustering):#for each cluster
            total_pixels = np.sum(self.initial_clustering == cl)
            core_pixels = np.sum(self.clustering == cl)
            cl_to_core[cl] = (core_pixels, core_pixels/total_pixels)
        return cl_to_core
    

    def find_clusterwise_core_pixel(self):
        """fills default dict clusterID-> list of all core pixels"""
        
        for cl in np.unique(self.initial_clustering):#for each cluster
            core_indices = list(zip(*np.where(self.clustering==cl)))
            for indices in core_indices:
                self.clusterwise_core_pixel[cl].append(indices)
    
    

    def plot_mass_with_contours(self, mass, cluster, initial_clustering=True, clustering = None):
        """plots mass heatmap and cluster contours"""
        if initial_clustering:
            clustering = self.initial_clustering
        cluster_binary = show_one_cluster(clustering, cluster)
        plt.imshow(self.region_array[:,:,mass])
        plot_outlines(cluster_binary.T, lw=1, color='r')


    def fill_core_pixels(self):
        """fill core pixel clustering up again"""
        filled_clustering = self.clustering.copy()
        for r in range(self.clustering.shape[0]):
            for c in range(self.clustering.shape[1]):
                if self.clustering[r,c]==0:
                    # non core pixel: 
                    filled_clustering[r,c]=find_best_fitting_cluster(self.region_array, (r,c), self.clustering)
        self.filled_clustering = filled_clustering


    def compute_core_pixel_similarity_clustering(self, sampling=True, samplesize=100):
        """compute similatity of all pixels to all core pixels of each cluster """
        #core_similarity = np.zeros((self.region_array.shape[0],self.region_array.shape[1], int(max(self.clusterwise_core_pixel.keys()))+1))
        core_similarity = defaultdict(lambda: defaultdict(list))
        self.core_similarity_clustering = np.zeros((self.region_array.shape[0],self.region_array.shape[1]))
        
        bar = makeProgressBar()
        for r in bar(range(self.region_array.shape[0])):
            for c in range(self.region_array.shape[1]):
                this_spectrum = self.region_array[r,c,:]
                dot_products = []
                for cl in self.clusterwise_core_pixel.keys():
                    if sampling:
                        # draw 10 random core pixel:
                        sample_core = random.sample(self.clusterwise_core_pixel[cl], samplesize)
                    else:
                        sample_core = self.clusterwise_core_pixel[cl]
                    for pixel in sample_core:
                        core_spectrum = self.region_array[pixel[0],pixel[1],:]
                        dot_products.append(np.dot(this_spectrum, core_spectrum))
                    core_similarity[r,c][int(cl)].append(mean(dot_products))
                self.core_similarity_clustering[r,c] = max(core_similarity[r,c], key=core_similarity[r,c].get)
        self.core_similarity_dict = core_similarity


    def check_mass_for_cluster(self, mass, cluster):
        if len(self.analysis_results[(self.analysis_results["mass_name"]==mass) & (self.analysis_results["cluster_name"]==cluster)])>0 :
            df_match = self.analysis_results[(self.analysis_results["mass_name"]==mass) & (self.analysis_results["cluster_name"]==cluster)]
            dict_match = df_match.to_dict()
            idx = list(dict_match.values())[0].keys()
            for k in dict_match.keys():
                dict_match[k] = dict_match[k][list(idx)[0]]
            print(f'--- SUMMARY FOR mass {dict_match["mass_name"]} and cluster {dict_match["cluster_name"]} -----------')
            print("")
            print('--- UNIQUENES & RELEVANCE --------------------------')
            if dict_match["unique_for_cluster"]==True:
                print('this mass is unique for this cluster')
            else:
                other_clusters = list(dict_match["rel_in_clusters"])
                other_clusters.remove(dict_match["cluster_name"])
                print(f'other clusters, mass is relevant in: {other_clusters}') 
            print(f'relevance score: {round(dict_match["relevance_score"],4)}') 
            print("")
            print('--- TEST RESULTS -----------------------------------')
            print(f'statistical value: {round(dict_match["ks_value"],4)}') 
            print(f'p value: {round(dict_match["p_value"],4)}') 
            print(f'adjusted p value: {round(dict_match["p_value_adj"],4)}') 
            print("")
            print('--- STATISTCAL EVALUATION --------------------------')
            print(f'sensitivity: {round(dict_match["sensitivity"],4)}') 
            print(f'specificity: {round(dict_match["specificity"],4)}') 
            print(f'false negative rate: {round(dict_match["false_neg_rate"],4)}') 
            print(f'false positive rate: {round(dict_match["false_pos_rate"],4)}') 
            print("")
            print('----------------------------------------------------')
        else: 
            print(f"mass {mass} was not identified as characteristic for cluster {cluster}")
        self.plot_mass_with_contours(mass, cluster)
        if len(self.analysis_results[(self.analysis_results["mass_name"]==mass) & (self.analysis_results["cluster_name"]==cluster)])>0 :
           return dict_match 
        else:
            return None
  

def connected_cluster_size(tested_clustering):
    """ calculates sizes of connected clusters
        parameter:  test_initial_clustering==True -> cluster that was used to initialize iterative cluster asignment
                    test_initial_clustering==False-> calculated cluster
        returns:    amount of connected_clusters
                    avg size of connected clusters
                    dict: occuring sizes -> count of occurence"""
    connected_clusters = sk_measure.label(tested_clustering)
    cc_size = {}
    for cc in np.unique(connected_clusters):
        count = np.count_nonzero(connected_clusters == cc)
        cc_size[cc] = count

    avg_cluster_size = mean(list(cc_size.values()))

    size_count = {}
    for size in np.unique(list(cc_size.values())):
        size_count[size] = 0
        

    for cluster in cc_size.keys():
        size = cc_size[cluster]
        size_count[size] += 1

    return len(np.unique(connected_clusters)), avg_cluster_size, size_count


def clustering_multi_otsu(img_unclustered: np.ndarray, cluster_amount: int, values_for_thresholding=[]):
    # returns: np.ndarray containing clustered img
    if type(values_for_thresholding)!=np.ndarray:
        values_for_thresholding = img_unclustered
    thresholds = threshold_multiotsu(values_for_thresholding, classes=cluster_amount)
    ths = []
    for i in range(cluster_amount-1):
        ths.append(thresholds[i])
    ths.append(254.5)
    ths.append(255)
    img_he_clustered = np.digitize(img_unclustered, bins=ths, right=True)
    return img_he_clustered, ths


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    indices = np.unravel_index(indices, ary.shape)
    #my code
    new_indices=[]
    for i in range(len(indices[0])):
        new_indices.append((indices[0][i], indices[1][i]))
    return new_indices


def fractional_intersection(c2m2p: dict, fraction = 0.9):
    """c2m2p = cluster -> mass -> pixels
        returns: c2p = cluster -> pixels, where 90% of the masses that are rel for the cluster, are expressed"""
    c2p = {}
    c2p2c = {} # cluster -> pixels -> count
    for cluster in c2m2p.keys():
        for mass in c2m2p[cluster]:
            break
    pass


def remove_similar_masses(ms_df:pd.DataFrame, mass_window_range=20):
    import numpy as np
    import pandas as pd

    masses = ms_df['mass_name']

    close_masses = []
    for m in range(masses[0], masses[len(masses)-1]+mass_window_range, mass_window_range):
        close_masses.append([mass for mass in masses if (mass<m+mass_window_range/2 and mass>=m-mass_window_range/2)])

    close_masses = [x for x in close_masses if x != []]
    #print(close_masses)

    keep_masses = []
    for close_m in close_masses:
        scores= list(ms_df[ms_df['mass_name'].isin(close_m)]["relevance_score"])
        max_score = max(scores)
        max_index = scores.index(max_score)
        mass = list(ms_df[ms_df['mass_name'].isin(close_m)]["mass_name"])[max_index]
        keep_masses.append(mass)#masses with highest score

    #print(keep_masses)

    keep_df = ms_df[ms_df['mass_name'].isin(keep_masses)]
    #print(keep_df)
    return(keep_df)



def show_one_cluster(clustering, cl_to_show, show=False):
    to_show = np.copy(clustering)
    to_show[to_show!=cl_to_show] = 0
    if show:
        plt.imshow(to_show)
    return to_show



def get_all_edges(bool_img):
    """
    Get a list of all edges (where the value changes from True to False) in the 2D boolean image.
    The returned array edges has he dimension (n, 2, 2).
    Edge i connects the pixels edges[i, 0, :] and edges[i, 1, :].
    Note that the indices of a pixel also denote the coordinates of its lower left corner.
    """
    edges = []
    ii, jj = np.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == bool_img.shape[1]-1 or not bool_img[i, j+1]:
            edges.append(np.array([[i, j+1],
                                   [i+1, j+1]]))
        # East
        if i == bool_img.shape[0]-1 or not bool_img[i+1, j]:
            edges.append(np.array([[i+1, j],
                                   [i+1, j+1]]))
        # South
        if j == 0 or not bool_img[i, j-1]:
            edges.append(np.array([[i, j],
                                   [i+1, j]]))
        # West
        if i == 0 or not bool_img[i-1, j]:
            edges.append(np.array([[i, j],
                                   [i, j+1]]))

    if not edges:
        return np.zeros((0, 2, 2))
    else:
        return np.array(edges)


def close_loop_edges(edges):
    """
    Combine thee edges defined by 'get_all_edges' to closed loops around objects.
    If there are multiple disconnected objects a list of closed loops is returned.
    Note that it's expected that all the edges are part of exactly one loop (but not necessarily the same one).
    """

    loop_list = []
    while edges.size != 0:

        loop = [edges[0, 0], edges[0, 1]]  # Start with first edge
        edges = np.delete(edges, 0, axis=0)

        while edges.size != 0:
            # Get next edge (=edge with common node)
            ij = np.nonzero((edges == loop[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                loop.append(loop[0])
                # Uncomment to to make the start of the loop invisible when plotting
                # loop.append(loop[1])
                break

            loop.append(edges[i, (j + 1) % 2, :])
            edges = np.delete(edges, i, axis=0)

        loop_list.append(np.array(loop))

    return loop_list


def plot_outlines(bool_img, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    edges = get_all_edges(bool_img=bool_img)
    edges = edges - 0.5  # convert indices to coordinates; TODO adjust according to image extent
    outlines = close_loop_edges(edges=edges)
    cl = LineCollection(outlines, **kwargs)
    ax.add_collection(cl)


def find_best_fitting_cluster(region_array, non_core_pixel, clustering):
    r = non_core_pixel[0]
    c = non_core_pixel[1]
    
    this_spectrum = region_array[r,c,:]
    # iterate over all clusters
    # maximise avg dotproduct
    similarity_to_cluster = {}
    for cl in np.unique(clustering):
        if cl==0:
            continue

        dot_products = []
        for row in range(clustering.shape[0]):
            for col in range(clustering.shape[1]):
                if clustering[row,col] == cl:
                    cl_spectrum = region_array[row,col,:]
                    dot_products.append(np.dot(this_spectrum, cl_spectrum))

        similarity_to_cluster[cl] = np.mean(dot_products)
    
    assigned_cluster = max(similarity_to_cluster, key=similarity_to_cluster.get)
    return assigned_cluster


def compare_clusterings_for_data(dict_of_KsSimilarity:dict):
    names=[]
    amount_found_masses=[]
    clusterID=[]
    amount_connencted_cluster=[]
    avg_size_cc=[]
    share_core_pixel=[]
    density=[]

    for cl in dict_of_KsSimilarity.keys():
        ksSimilarity = dict_of_KsSimilarity[cl]
        found_masses_for_cluster = ksSimilarity.found_masses_for_cluster(len_only=True)
        amount_cc, avg_cl_size, _ = ksSimilarity.connected_cluster_size()
        cl_density = ksSimilarity.density(radius=1, circular=False)
        for c in list(found_masses_for_cluster.keys()):
            names.append(cl)
            amount_found_masses.append(found_masses_for_cluster[c])
            clusterID.append(c)
            amount_connencted_cluster.append(amount_cc)
            avg_size_cc.append(avg_cl_size)
            share_core_pixel.append(len(ksSimilarity.clusterwise_core_pixel[c]))
            density.append(cl_density[c])
    

    data_summary = pd.DataFrame({"cluster_KsSimilarity": names,
                                 "clusterID": clusterID,
                                 "amount_found_masses": amount_found_masses,
                                 "amount_connencted_cluster": amount_connencted_cluster,
                                 "avg_size_cc": avg_size_cc,
                                 "share_core_pixel": share_core_pixel,
                                 "density":density})
    return data_summary

def homogeneity(clustering:np.array, ignore_zero=True):
    """computes distribution of euklidean distance """
    clustering_dict = matrix_to_dict(clustering)
    distances = defaultdict(list)
    indices = defaultdict(list)
    summary = {}
    for c in clustering_dict.keys():
        if ignore_zero and c==0:
            continue
        center = tuple(KMeans(n_clusters=1).fit(clustering_dict[c]).cluster_centers_[0])
        for elem in clustering_dict[c]:
            euc_dist = distance.euclidean(elem, center)
            #cluster dispersion
            distances[c].append(euc_dist)
            indices[c].append(elem)
        cl_std = np.std(distances[c])
        cl_mean = mean(distances[c])
        # share of own cluster within 3 std 
        """own_counter = 0
        other_counter = 0
        r_range = range(max(math.floor(center[0]-cl_std),0), min(math.ceil(center[0]+cl_std), clustering.shape[0]))
        c_range = range(max(math.floor(center[1]-cl_std),0), min(math.ceil(center[1]+cl_std), clustering.shape[1]))
        for row in r_range:
            for col in c_range:
                if distance.euclidean((row,col), center)<cl_std:
                    if ignore_zero:
                        if clustering[row,col] == c:
                            own_counter+=1
                        elif clustering[row,col] != 0:
                            other_counter+=1
                    else:
                        if clustering[row,col] == c:
                            own_counter+=1
                        else:
                            other_counter+=1"""
        if ignore_zero:
            A = np.count_nonzero(clustering)
        else:
            A = clustering.size
        n = len(clustering_dict[c])
        l = n/A
        K = l**(-1)*sum(d<3*cl_std for d in distances[c])/n
        L = (K/math.pi)**0.5
        
        summary[c] = {"centroid":center, "mean":cl_mean, "std":cl_std, "L(3*std)":L}
        
    return summary, distances


def get_surrounding_area(matrix:np.ndarray,p,radius=1, circular=False):
    """return list of all values in surrounding area
    Parameters: - circular """
    r = matrix.shape[0]
    c = matrix.shape[1]
    search_area = matrix[max(p[0]-radius,0) : min(p[0]+radius+1,r),  max(p[1]-radius,0) : min(p[1]+radius+1,c)]
    if not circular:
        return search_area.flatten().tolist()
    
    seach_area_circular = []
    for row in range(max(p[0]-radius,0), min(p[0]+radius+1,r)):
        for col in range( max(p[1]-radius,0), min(p[1]+radius+1,c)):
            if row == p[0] and col == p[1]:
                continue
            eukl_dist = math.sqrt((row - p[0])*(row - p[0]) + (col - p[1])*(col - p[1]))
            if eukl_dist <= radius:
                seach_area_circular.append(matrix[row,col])

    return seach_area_circular




def matrix_to_dict(matrix_2d:np.array):
    cluster_dict = defaultdict(list)
    for r in range(matrix_2d.shape[0]):
        for c in range(matrix_2d.shape[1]):
            cluster = matrix_2d[r,c]
            cluster_dict[cluster].append((r,c))
    return cluster_dict


    
    
