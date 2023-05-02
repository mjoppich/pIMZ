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




def KS_test( mass_array:np.array, clustering:np.array, clusters_considered=None, background_mass=0):
    """ returns: dict: key = cluster, value = clustered statistics, clustered p value, unclutered statistics, unclustered p value"""
    # classes_clustered saves all clustered intensity values for HE categories
    classes_clustered = defaultdict(list) # mass_clustered: 0 is background

    # classes_unclusteres saves all unclustered intensities values for HE categories
    classes_unclustered = defaultdict(list) # mass_array: 0 is background
    classes_he = defaultdict(list)

    cluster_range = np.unique(clustering)
    
    if clusters_considered is None:
        clusters_considered = cluster_range
    
    # -----do pre clustering
    values_for_thresholding = mass_array[clustering != background_mass]

    for c in cluster_range:
        classes_he[c] = [tuple(x) for x in np.argwhere(clustering == c)]
        
        relevant_intensities = mass_array[ clustering == c ]
        classes_unclustered[c] = relevant_intensities[relevant_intensities != 0]
            
    unclassifierd_unclustered = defaultdict(list)

    # KS test
    ks_per_cluster = {}
    
    background_intensities = np.concatenate( [classes_unclustered[x] for x in cluster_range ] ) #if x != c
    
    for c in clusters_considered:
        #print(f"pint classes unclustered for cluster {c}: {classes_unclustered[c]}")
        
        foreground_intensities = classes_unclustered[c]
        
        if len(foreground_intensities) == 0 or len(background_intensities) == 0:
            ks_per_cluster[c] = None, 1.0
        else:
            KS_test_unclustered = stats.kstest(foreground_intensities, background_intensities, alternative="less")
            ks_per_cluster[c] = KS_test_unclustered[0], KS_test_unclustered[1]
            
    return ks_per_cluster



class KsSimilarity:

    def __init__(self, spec_mz_values:SpectraRegion):
        self.spec = spec_mz_values
        # to do mz_region_array = self.spec.region_array
        self.region_array = np.copy(self.spec.region_array)
        self.clustering = None

    def perform_analysis(self, passed_clustering=np.empty([0, 0]), weight_ks=1, weight_p=1, weight_c=1, features=[], iterative=False):
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
        
        ks_test_unclustered = self.find_relevant_masses_by_ks(self.region_array, considered_masses, self.clustering, cluster_range=clusters, for_clustered=False, iterative=iterative)
        # mass, cluster, ks, p, p_mult
        # TODO only regard pixels != 0 in clustering
        if iterative:
            ms_df = self.dict_to_dataframe(ks_test_unclustered[0]) 
            core_clustering = ks_test_unclustered[1]
            print(f"ks_test_unclustered[1].shape: {core_clustering.shape}")
            print(f"region.array:{self.region_array.shape}")
            plt.imshow(core_clustering)
            # ---------self.clustering: eplacing unused pixels by -1. Actually: replace the self.clustering completely
            for r in range(self.clustering.shape[0]):
                for c in range(self.clustering.shape[1]):
                    if core_clustering[r,c] == 0:
                        self.clustering[r,c] = -1
            # ---------
        else:
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

        clusters = [int(item) for item in list(rel_masses_for_cluster.keys())]

        print(type(clusters), clusters)
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



    
            
        
    def find_relevant_masses_by_ks(self, region_array, masses, clustering, cluster_range=range(5), a=0.05, th_ks=0.1, for_clustered=True, for_unclustered=True, iterative=False):
        """returns two dictionarys: clustered and unclustered save all masses that have a p-value<a and ks >0.1 with results"""
        if for_clustered and for_unclustered:
            return None
            ks_test_clustered = {}
            ks_test_unclustered = {}
            bar=makeProgressBar()
            for mass in bar(masses):
                ks_of_mass = self.KS_test(region_array, mass, he_clustered_ms_shaped=clustering, cluster_range=cluster_range, clusters_considered=cluster_range)
                
                for c in cluster_range:
                    if ks_of_mass[c][1]< a and ks_of_mass[c][0] > th_ks:
                        ks_test_clustered[(mass, c)] = (ks_of_mass[c][0], ks_of_mass[c][1])
                    if ks_of_mass[c][3]< a and ks_of_mass[c][2] > th_ks:
                        ks_test_unclustered[(mass, c)] = (ks_of_mass[c][2], ks_of_mass[c][3])
            return ks_test_clustered, ks_test_unclustered

        elif for_unclustered:
            print("For unclustered part")
 
            ks_test_unclustered = {}
            
            masses = list(masses)
            from joblib import Parallel, delayed, parallel_backend
            
            with parallel_backend('loky', n_jobs=8):
                results = Parallel()(delayed(KS_test)(region_array[:,:, mass], clustering=clustering) for mass in masses)

                for idx, res in enumerate(results):
                    for c in res:
                        if res[c][1]< a and res[c][0] > th_ks:
                            ks_test_unclustered[(masses[idx], c)] = (res[c][0], res[c][1])
            
            return ks_test_unclustered

        elif for_clustered:
            
            print("For clustered part")
            ks_test_clustered = {}
            
            loop = asyncio.get_event_loop() 
            looper = asyncio.gather(*[KS_test(region_array[:,:, mass], clustering=clustering) for mass in masses]) 
            results = loop.run_until_complete(looper)
            
            for res in results:
                for c in res:
                    if ks_of_mass[c][1]< a and ks_of_mass[c][0] > th_ks:
                        ks_test_clustered[(mass, c)] = (ks_of_mass[c][0], ks_of_mass[c][1])
            
            return ks_test_clustered
        
        return None


    def iterative_cluster_assignment(self, clustering):
        
        current_clustering = clustering.copy()
        current_clustering = current_clustering+1
        last_clustering = np.zeros( current_clustering.shape )
        
        pixels = clustering.size
                     
        current_iteration = 0
        max_iteration = 5
              
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
            ms_df = self.perform_analysis(passed_clustering=np.copy(current_clustering))
            
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
            
                
        return current_clustering
        
        fraction_of_masses_th = 0.5# should be 0.9
        highly_exp_th = 0.1#10%
        max_iterations = 5
        # adapt the pixels that are considered in the region array-> goal: only homogen tissue!
        # initial clustering: spec.meta["segmented"]-> 10 clusters
        
        current_clustering = clustering.copy() # clustering
        print(f"shape of clustering: {current_clustering.shape}")
        updated_clustering=np.zeros((clustering.shape)) # new clustering  2D
        last_amount_pixels = 2*clustering.size
        updated_amount_pixels = clustering.size
        iterations = 0
        # while loop should break if too many iterations or if no change in core pixels

        # i need to do independent iterative processes for each cluster -> KS_test(cluster_range=c), c is single cluster
        
        while iterations < max_iterations and (last_amount_pixels-updated_amount_pixels)/last_amount_pixels>0.02:
            print(iterations)
            iterations += 1
            last_amount_pixels = updated_amount_pixels
            updated_amount_pixels = 0

            cluster_range = np.unique(current_clustering)
            cluster_range = cluster_range.tolist()
            if 0 in cluster_range:
                cluster_range.remove(0)

            ks_test_unclustered = self.find_relevant_masses_by_ks(self.region_array, masses, current_clustering, cluster_range=cluster_range, for_clustered=False, iterative=False)
            ms_df = self.dict_to_dataframe(ks_test_unclustered) 
            c2m = self.dataframe_to_cluster_dict(ms_df)
            if c2m=={}:
                print("c2m is empty")
                ks_test_unclustered = last_ks_test_unclustered
                print(last_ks_test_unclustered.keys(), np.unique(current_clustering))# both are empty
                break
            
            uc2m = self.find_unique_masses_for_cluster(c2m)
            c2m2p = {}# cluster -> mass -> pixel
            c2p=defaultdict(list)
            bar = makeProgressBar()
            for cluster in bar(uc2m.keys()): 
                m2p = {}
                # uc2m[cluster] is the list of masses relevant for a cluster
                for mass in uc2m[cluster]:
                    mass_rel_pixels = largest_indices(self.region_array[:,:,mass], int(self.region_array[:,:,mass].size*highly_exp_th))
                    m2p[mass] = mass_rel_pixels
                c2m2p[cluster] = m2p
                p2m=defaultdict(list)# counts pixels
                for mass in m2p:
                    for pixel_tupel in m2p[mass]:
                        p2m[pixel_tupel].append(mass)
                # add pixels, if they are expressed in at least "fraction_of_masses_th" masses
                total_masses = len(m2p)
                th = total_masses*fraction_of_masses_th
                for pixel_tupel in p2m:
                    if len(p2m[pixel_tupel])>= th:
                        c2p[cluster].append(pixel_tupel)
                
                # convert dict values from list to tuple (list not hashable)
                #c2m2p[cluster] = {key:tuple(lst) for key, lst in c2m2p[cluster].items()}
                #c2p[cluster] = set(c2m2p[cluster].values())
            current_clustering = np.zeros((clustering.shape))

            for cluster in c2p:


                for r,c in c2p[cluster]:
                    current_clustering[r,c] = cluster
                #current_clustering[c2up[cluster]] = cluster
                updated_amount_pixels += len(c2p[cluster])

                print(current_clustering)

            last_ks_test_unclustered = ks_test_unclustered.copy()

            # end of loop:

                
        return ks_test_unclustered, current_clustering




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
            my_data["p_value_mult"].append(ks_dict[key][1]*5*16000)
            my_data["rel_in_clusters"].append( tuple(m2c[key[0]]) )
            my_data["unique_for_cluster2"].append( len(m2c[key[0]])==1 )

        ks_dataframe = pd.DataFrame.from_dict(my_data)
        return ks_dataframe


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
    #print(ths)
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