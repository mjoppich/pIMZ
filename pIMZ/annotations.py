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

#web/html
import jinja2

# applications
import progressbar
def makeProgressBar():
    return progressbar.ProgressBar(widgets=[
        progressbar.Bar(), ' ', progressbar.Percentage(), ' ', progressbar.AdaptiveETA()
        ])


class ProteinWeights():
    """12343 This class serves as lookup class for protein<->mass lookups for DE comparisons of IMS analyses.
    """

    def __set_logger(self):
        self.logger = logging.getLogger('ProteinWeights')
        self.logger.setLevel(logging.INFO)

        if not logging.getLogger().hasHandlers():

            consoleHandler = logging.StreamHandler()
            consoleHandler.setLevel(logging.INFO)

            self.logger.addHandler(consoleHandler)

            formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
            consoleHandler.setFormatter(formatter)

            self.logger.info("Added new Stream Handler")

    def __init__(self, filename, min_mass=-1, max_mass=-1):
        """Creates a ProteinWeights class. Requires a formatted proteinweights-file.

        Args:
            filename (str): File with at least the following columns: protein_id, gene_symbol, mol_weight_kd, mol_weight.
            max_mass (float): Maximal mass to consider/include in object. -1 for no filtering. Masses above threshold will be discarded. Default is -1.
            max_mass (float): Minimal mass to consider/include in object. -1 for no filtering. Masses below threshold will be discarded. Default is -1.
        """

        self.__set_logger()

        self.protein2mass = defaultdict(set)
        self.category2id = defaultdict(set)
        self.protein_name2id = {}

        if filename != None:

            with open(filename) as fin:
                col2idx = {}
                for lidx, line in enumerate(fin):

                    line = line.strip().split("\t")

                    if lidx == 0:
                        for eidx, elem in enumerate(line):

                            col2idx[elem] = eidx

                        continue

                    #protein_id	gene_symbol	mol_weight_kd	mol_weight

                    if len(line) < 4:
                        continue

                    proteinIDs = line[col2idx["protein_id"]].split(";")
                    proteinNames = line[col2idx["gene_symbol"]].split(";")
                    molWeight = float(line[col2idx["mol_weight"]])

                    if max_mass >= 0 and molWeight > max_mass:
                        continue    

                    if min_mass >= 0 and molWeight < min_mass:
                        continue

                    if len(proteinNames) == 0:
                        proteinNames = proteinIDs

                    for proteinName in proteinNames:
                        self.protein2mass[proteinName].add(molWeight)
                        self.protein_name2id[proteinName] = proteinIDs

            allMasses = self.get_all_masses()
            self.logger.info("Loaded a total of {} proteins with {} masses".format(len(self.protein2mass), len(allMasses)))
    


    def get_all_masses(self):
        """Returns all masses contained in the lookup-dict

        Returns:
            set: set of all masses used by this object
        """
        allMasses = set()
        for x in self.protein2mass:
            for mass in self.protein2mass[x]:
                allMasses.add(mass)

        return allMasses

    def get_mass_to_protein(self):
        """Returns a dictionary of a mass to proteins with this mass. Remember that floats can be quite exact: no direct comparisons!

        Returns:
            dict: dictionary mass => set of proteins
        """

        mass2prot = defaultdict(set)
        for x in self.protein2mass:
            for mass in self.protein2mass[x]:
                mass2prot[mass].add(x)

        return mass2prot

    def get_ppm(self, mz, ppm):
        return mz * (ppm / 1000000)


    def print_collisions(self, maxdist=2.0, ppm=None, print_proteins=False):
        """Prints number of proteins with collision, as well as mean and median number of collisions.

        For each recorded mass it is checked how many other proteins are in the (mass-maxdist, mass+maxdist) range.

        Args:
            maxdist (float, optional): Mass range in order to still accept a protein. Defaults to 2.
            print_proteins (bool, optional): If True, all collision proteins are printed. Defaults to False.
        """

        if (maxdist is None and ppm is None) or (not maxdist is None and not ppm is None):
            raise ValueError("either maxdist or ppm must be None")


        self.logger.info("Ambiguity: there exists other protein with conflicting mass")
        self.logger.info("Collision: there exists other protein with conflicting mass due to distance/error/ppm.")
        self.logger.info("Frequency: for how many masses does this occur?")
        self.logger.info("Count: with how many proteins does it happen?")

        allProts = set([x for x in self.protein2mass])
        allProtMasses = Counter([len(self.protein2mass[x]) for x in self.protein2mass])
        mass2prot = self.get_mass_to_protein()            
        
        collisionFrequency = Counter()
        ambiguityFrequency = Counter()

        collisionCount = Counter()
        ambiguityCount = Counter()
        
        sortedMasses = sorted([x for x in mass2prot])

        massAmbiguities = [len(mass2prot[x]) for x in mass2prot]
        plt.hist(massAmbiguities, density=False, cumulative=True, histtype="step")
        plt.yscale("log")
        plt.show()
        plt.close()

        for mass in sortedMasses:

            if not maxdist is None:
                lmass, umass = mass-maxdist, mass+maxdist
            else:
                ppmdist = self.get_ppm(mass, ppm)
                lmass, umass = mass-ppmdist, mass+ppmdist
            
            currentProts = mass2prot[mass]

            massProts = set()

            for tmass in sortedMasses:
                if lmass <= tmass <= umass:
                    massProts = massProts.union(mass2prot[tmass])

                if umass < tmass:
                    break

            #massprots contains all protein names, which match current mass
            assert(currentProts.intersection(massProts) == currentProts)

            #this is all ambiguities
            if len(massProts) > 0:
                for prot in massProts:
                    ambiguityFrequency[prot] += 1
                    ambiguityCount[prot] += len(massProts)

            # there are proteins for this current, which we are not interested in for conflicts
            if len(currentProts) >= 1:
                for prot in currentProts:
                    if prot in massProts:
                        massProts.remove(prot)

            #this is ambiguity due to size error
            #massProts are other proteins!
            if len(massProts) > 0:
                for prot in massProts:
                    collisionFrequency[prot] += 1
                    collisionCount[prot] += len(massProts)

        ambiguityFreqCounter = Counter()
        for x in ambiguityFrequency:
            ambiguities = ambiguityFrequency[x]
            ambiguityFreqCounter[ambiguities] += 1

        collisionFreqCounter = Counter()
        for x in collisionFrequency:
            collisions = collisionFrequency[x]
            collisionFreqCounter[collisions] += 1

        if not maxdist is None:
            self.logger.info("Using uncertainty of {}m/z".format(maxdist))
        else:
            self.logger.info("Using uncertainty of {}ppm".format(ppm))





        self.logger.info("             Number of total protein/genes: {}".format(len(allProts)))
        self.logger.info("                   Number of unique masses: {}".format(len(mass2prot)))
        self.logger.info("")
        self.logger.info("             Proteins/genes with ambiguity: {}".format(len(ambiguityFrequency)))
        self.logger.info("             Mean Frequency of ambiguities: {}".format(np.mean([ambiguityFrequency[x] for x in ambiguityFrequency])))
        self.logger.info("           Median Frequency of ambiguities: {}".format(np.median([ambiguityFrequency[x] for x in ambiguityFrequency])))
        self.logger.info("")
        self.logger.info("             Proteins/genes with collision: {}".format(len(collisionFrequency)))
        self.logger.info("              Mean Frequency of collisions: {}".format(np.mean([collisionFrequency[x] for x in collisionFrequency])))
        self.logger.info("            Median Frequency of collisions: {}".format(np.median([collisionFrequency[x] for x in collisionFrequency])))
        self.logger.info("")
        self.logger.info("             Proteins/genes with ambiguity: {}".format(len(ambiguityCount)))
        self.logger.info("                Mean Number of ambiguities: {}".format(np.mean([ambiguityCount[x] for x in ambiguityCount])))
        self.logger.info("              Median Number of ambiguities: {}".format(np.median([ambiguityCount[x] for x in ambiguityCount])))
        self.logger.info("")
        self.logger.info("             Proteins/genes with collision: {}".format(len(collisionCount)))
        self.logger.info("                 Mean Number of collisions: {}".format(np.mean([collisionCount[x] for x in collisionCount])))
        self.logger.info("               Median Number of collisions: {}".format(np.median([collisionCount[x] for x in collisionCount])))

        self.logger.info("")
        if print_proteins:
            self.logger.info("Proteins/genes with ambiguity: {}".format([x for x in ambiguityCount]))
        else:
            self.logger.info("    Proteins/genes with ambiguity count: {}".format(ambiguityCount.most_common(10)))

        if print_proteins:
            self.logger.info("Proteins/genes with collision: {}".format([x for x in collisionCount]))
        else:
            self.logger.info("    Proteins/genes with collision count: {}".format(collisionCount.most_common(10)))
        
        self.logger.info("")
        for x in ambiguityFreqCounter:
            self.logger.info("ambiguity count; proteins with {} other matching proteins: {}".format(x, ambiguityFreqCounter[x]))
        self.logger.info("collision count; proteins with other matching proteins: {}".format(sum([ambiguityFreqCounter[x] for x in ambiguityFreqCounter])))
        self.logger.info("")
        for x in collisionFreqCounter:
            self.logger.info("collision count; proteins with {} other matching proteins: {}".format(x, collisionFreqCounter[x]))
        self.logger.info("collision count; proteins with other matching proteins: {}".format(sum([collisionFreqCounter[x] for x in collisionFreqCounter])))


    def get_protein_from_mass(self, mass, maxdist=2, ppm=None):
        """Searches all recorded mass and proteins and reports all proteins which have at least one mass in (mass-maxdist, mass+maxdist) range.

        Args:
            mass (float): mass to search for
            maxdist (float, optional): allowed offset for lookup. Defaults to 2.
            ppm (float, optional): allowed relative offset for lookup. Defaults to 2.

        Returns:
            list: sorted list (by abs mass difference) of all (protein, weight) tuple which have a protein in the given mass range
        """

        possibleMatches = []

        for protein in self.protein2mass:
            protMasses = self.protein2mass[protein]

            for protMass in protMasses:

                if not maxdist is None:
                    if abs(mass-protMass) < maxdist:
                        protDist = abs(mass-protMass)
                        possibleMatches.append((protein, protMass, protDist))
                elif not ppm is None:
                    
                    ppmDist = mass * ppm / 1000000
                    if abs(mass-protMass) < ppmDist:
                        protDist = abs(mass-protMass)
                        possibleMatches.append((protein, protMass, protDist))

                else:
                    raise ValueError()

        possibleMatches = sorted(possibleMatches, key=lambda x: x[2])
        possibleMatches = [(x[0], x[1]) for x in possibleMatches]

        return possibleMatches

    def get_protein_from_mz(self, mzval, maxdist=None, ppm=None, mzoffset=1):
        """Searches all recorded mass and proteins and reports all proteins which have at least one mass in (mass-maxdist, mass+maxdist) range.

        Args:
            mass (float): mass to search for
            maxdist (float, optional): allowed offset for lookup. Defaults to 2.
            ppm (float, optional): allowed relative offset for lookup. Defaults to 2.
            mzoffset (float): m/z to mass offset; m/z-mzoffset = mass

        Returns:
            list: sorted list (by abs mass difference) of all (protein, weight) tuple which have a protein in the given mass range
        """

        return self.get_protein_from_mass( mzval-mzoffset, maxdist=maxdist, ppm=ppm )


    def get_masses_for_protein(self, protein):
        """Returns all recorded masses for a given protein. Return None if protein not found

        Args:
            protein (str|iterable): protein to search for in database (exact matching)

        Returns:
            set: set of masses for protein
        """

        if type(protein) == str:
            return self.protein2mass.get(protein, None)

        allMasses = set()
        for x in protein:
            protMasses = self.get_masses_for_protein(x)
            allMasses = allMasses.union(protMasses)

        return allMasses

    def compare_masses(self, pw):
        """For each protein contained in both PW objects, and for each protein mass, the distance to the best match in the other PW object is calculated.

        This is meant to provide a measure of how accuracte the theoretical calculation of m/z // Da is.

        Args:
            pw (ProteinWeights): The ProteinWeights object to compare to.
        """

        dists = []
        consideredMasses = 0

        for x in self.protein2mass:

            lookKey = x.upper()
            selfMasses = self.protein2mass[x]

            otherMasses = None
            if lookKey in pw.protein2mass:
                otherMasses = pw.protein2mass[lookKey]
            elif x in pw.protein2mass:
                otherMasses = pw.protein2mass[x]

            if otherMasses == None:
                continue

            selfMasses = sorted(selfMasses)
            otherMasses = sorted(otherMasses)

            protDiffs = []
            for smass in selfMasses:

                sMassDiffs = []
                for omass in otherMasses:
                    sMassDiffs.append(abs(smass-omass))

                selMassDiff = min(sMassDiffs)
                protDiffs.append(selMassDiff)


            selProtDiff = min(protDiffs)

            if selProtDiff > 500:
                print(x,selMassDiff, selfMasses, otherMasses)

            dists += [selProtDiff]

            if len(protDiffs) > 0:
                consideredMasses += 1

        print("Total number of considered masses: {}".format(consideredMasses))
        print("Total number of diffs > 100: {}".format(len([x for x in dists if x > 100])))
        print("Total number of diffs > 5  : {}".format(len([x for x in dists if x > 5])))
        print("Total number of diffs > 1  : {}".format(len([x for x in dists if x > 1])))
        print("Total number of diffs <= 1  : {}".format(len([x for x in dists if x <= 1])))

        print("{}\t{}\t{}\t{}\t{}\t{}".format(
            consideredMasses, len(dists), min(dists), np.median(dists), np.mean(dists), max(dists)
        ))

            

class AnnotatedProteinWeights(ProteinWeights):

    def __init__(self, filename, min_mass=-1, max_mass=-1):
        super().__init__(filename, min_mass=min_mass, max_mass=max_mass)


    def get_annotated_info(self, mzsearch):
        raise NotImplementedError()


class SDFProteinWeights(AnnotatedProteinWeights):
    
    def __init__(self, filename, min_mass=-1, max_mass=-1):
        super().__init__(None, min_mass=min_mass, max_mass=max_mass)

        assert(filename.endswith(".sdf"))

        self.protein2mass = defaultdict(set)
        self.protein_name2id = {}
        self.category2id = defaultdict(set)

        self.sdf_dic = self.sdf_reader(filename)

        for lm_id in self.sdf_dic:

            molWeight = float(self.sdf_dic[lm_id]["EXACT_MASS"])
            if max_mass >= 0 and molWeight > max_mass:
                continue    

            self.protein2mass[lm_id].add(molWeight)

            if "NAME" in self.sdf_dic[lm_id]:
                self.protein_name2id[self.sdf_dic[lm_id]["NAME"]] = lm_id
            elif "SYSTEMATIC_NAME" in self.sdf_dic[lm_id]:
                self.protein_name2id[self.sdf_dic[lm_id]["SYSTEMATIC_NAME"]] = lm_id

            self.category2id[self.sdf_dic[lm_id]["MAIN_CLASS"]].add(lm_id)#"CATEGORY"




    @classmethod
    def sdf_reader(cls, filename):
        """Reads a .sdf file into a dictionary.

        Args:
            filename (str): Path to the .sdf file.

        Returns:
            dict: Ids mapped to the respective annotation. 
        """
        res_dict = {}
        with open(filename) as fp:
            line = fp.readline()
            line_id = ""
            line_dict = {}
            while line:
                if line.startswith(">"):
                    if "LM_ID" in line:
                        if line_id:
                            res_dict[line_id] = line_dict
                            line_dict = {}
                            line_id = ""
                        line_id = fp.readline().rstrip()
                    else:
                        key = line.split("<")[1].split(">")[0]
                        line_dict[key] = fp.readline().rstrip()
                line = fp.readline()

        fp.close()
        return res_dict


    def get_annotated_info(self, mzsearch, annotation_key="CATEGORY"):

        classCounter = Counter()

        for res in mzsearch:
            resAnnot = self.sdf_dic[res[0]]
            classCounter[resAnnot[annotation_key]] += 1
            
        return classCounter
        
