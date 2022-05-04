# general
import logging
import json
import os
import random
import math
from collections import defaultdict, Counter
import glob
from re import M
import shutil, io, base64, abc

# general package
from natsort import natsorted
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from intervaltree import IntervalTree

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
    """
    This class serves as lookup class for protein<->mass lookups for DE comparisons of IMS analyses.
    """

    def __set_logger(self):
        self.logger = logging.getLogger('ProteinWeights')
        self.logger.setLevel(logging.INFO)

        if not self.logger.hasHandlers():

            consoleHandler = logging.StreamHandler()
            consoleHandler.setLevel(logging.INFO)

            self.logger.addHandler(consoleHandler)

            formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
            consoleHandler.setFormatter(formatter)

            self.logger.info("Added new Stream Handler")

    def __init__(self, filename, massMode=+1, ppm=5, min_mass=-1, max_mass=-1):
        """Creates a ProteinWeights class. Requires a formatted proteinweights-file.

        Args:
            filename (str): File with at least the following columns: protein_id, gene_symbol, mol_weight_kd, mol_weight.
            massMode (int): +1 if mz-Values were captured in M+H mode, -1 if mZ-Values were captured in M-H mode (protonated or deprotonated), https://github.com/rformassspectrometry/CompoundDb/issues/38, https://www.uni-saarland.de/fileadmin/upload/lehrstuhl/jauch/An04_Massenspektroskopie_Skript_Volmer.pdf
            ppm (int, optional): ppm (parts per million) error. Default is 5.
            max_mass (float, optional): Maximal mass to consider/include in object. -1 for no filtering. Masses above threshold will be discarded. Default is -1.
            max_mass (float, optional): Minimal mass to consider/include in object. -1 for no filtering. Masses below threshold will be discarded. Default is -1.
        """

        self.__set_logger()

        self.min_mass = min_mass
        self.max_mass = max_mass
        self.ppm = ppm
        self.massMode = massMode

        #self.protein2mass = defaultdict(set)
        self.mz_tree = IntervalTree()
        #self.category2id = defaultdict(set)
        #self.protein_name2id = {}

        if filename != None:

            self.__load_file()


    def __load_file(self, filename):


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
                massWeight = float(line[col2idx["mol_weight"]])
                mzWeight = self.get_mz_from_mass(massWeight)

                if self.max_mass >= 0 and massWeight > self.max_mass:
                    continue    

                if self.min_mass >= 0 and massWeight < self.min_mass:
                    continue

                if len(proteinNames) == 0:
                    proteinNames = proteinIDs

                for proteinName in proteinNames:
                    #self.protein2mass[proteinName].add(mzWeight)
                    ppmDist = self.get_ppm(mzWeight, self.ppm)
                    # this adds MZ values to the tree!
                    self.mz_tree.addi(mzWeight - ppmDist, mzWeight + ppmDist, {"name": proteinName, "mass": massWeight, "mzWeight": mzWeight})

        allMasses = self.get_all_masses()
        self.logger.info("Loaded a total of {} proteins with {} masses".format(len(self.mz_tree), len(allMasses)))

    def get_mz_from_mass(self, mass):
        return mass + self.massMode

    def get_mass_from_mz(self, mz):
        return mz - self.massMode

    def get_all_masses(self):
        """Returns all masses contained in the lookup-intervaltree

        Returns:
            set: set of all masses used by this object
        """
        allMasses = set()
        for interval in self.mz_tree:
            allMasses.add( interval.data["mass"] )

        return allMasses

    def get_mass_to_protein(self):
        """Returns a dictionary of a mass to proteins with this mass. Remember that floats can be quite exact: no direct comparisons!

        Returns:
            dict: dictionary mass => set of proteins
        """

        mass2prot = defaultdict(set)
        for interval in self.mz_tree:
            mass2prot[ interval.data["mass"] ].add(interval.data["name"])

        return mass2prot

    def get_ppm(self, mz, ppm):
        return mz * (ppm / 1000000)


    def annotate_df(self, df, ppm=5, mass_column="gene_mass"):
        
        uniqmasses = [0] * df.shape[0]
        hitProteins = [[]] * df.shape[0]

        for ri, (_, row) in enumerate(df.iterrows()):

            curmass = float(row[mass_column])
            allHitMasses = self.get_protein_from_mz(curmass, ppm=ppm)

            if row["gene_ident"] == "mass_553_6274052499639":
                print(ri, curmass, allHitMasses)

            uniqmasses[ri] = len(allHitMasses)
            hitProteins[ri] = [x[0] for x in allHitMasses]

        df["#matches"] = uniqmasses
        df["matches"] = hitProteins

        return df


    def get_protein_from_mz(self, mz, ppm=5):
        """Searches all recorded mass and proteins and reports all proteins which have at least one mass in (mass-maxdist, mass+maxdist) range.

        Args:
            mass (float): mass to search for
            ppm (int, optional): allowed relative offset for lookup. Defaults to 5.

        Returns:
            list: sorted list (by abs mass difference) of all (protein, weight) tuple which have a protein in the given mass range
        """

        possibleMatches = []

        ppmDist = self.get_ppm(mz, ppm)
        overlaps = self.mz_tree.overlap(mz-ppmDist, mz+ppmDist)
        for overlap in overlaps:
            
            protMass = overlap[2]["mzWeight"]
            protDist = abs(mz-protMass)
            possibleMatches.append((overlap[2]["name"], protMass, protDist))

        possibleMatches = sorted(possibleMatches, key=lambda x: x[2])
        possibleMatches = [(x[0], x[1]) for x in possibleMatches]

        return possibleMatches

    def get_protein_from_mass(self, mass, ppm=5):
        mz = self.get_mz_from_mass(mass)
        return self.get_protein_from_mz(mz, ppm=ppm)

    


    def get_mass_for_protein(self, protein):
        return self.get_data_for_protein(protein, data_slot="mzWeight")
    
    def get_mz_for_protein(self, protein):
        return self.get_data_for_protein(protein, data_slot="mass")

    def get_data_for_protein(self, protein, data_slot="mass"):
        """Returns all recorded values in data_slot for given proteins. Return None if protein not found

        Args:
            protein (str|iterable): protein to search for in database (exact matching)
            data_slot (str): data slot in interval to report

        Returns:
            set: set of masses for protein
        """

        if type(protein) == str:
            retData = set()
            for interval in self.mz_tree:
                if interval.data["name"] == protein:
                    retData.add(interval.data[data_slot])
            return retData        

        allData = set()
        for x in protein:
            protData = self.get_masses_for_protein(x)
            allData = allData.union(protData)

        return allData


    def get_data_for_proteins(self, proteins, data_slot="mass"):

        dataCounter = Counter()

        for x in proteins:
            datas = self.get_data_for_protein(x[0], data_slot=data_slot)
            
            for y in datas:
                dataCounter[y] += 1

        return dataCounter


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
          

class MaxquantPeptides(ProteinWeights):

    def __init__(self, filename, massMode=+1, ppm=5, min_mass=-1, max_mass=-1, name_column="Gene names", name_function=lambda x: x.strip().split(";"), encoding='Windows-1252', error_bad_lines=False):
        super().__init__(None, massMode=massMode, ppm=ppm, min_mass=min_mass, max_mass=max_mass)

        self.__load_file(filename, name_column=name_column, name_function=name_function, encoding=encoding, error_bad_lines=error_bad_lines)

    def __load_file(self, filename, name_column="Gene names", name_function=lambda x: x.strip().split(";"), encoding='Windows-1252', error_bad_lines=False):

        inputfile = io.StringIO()
        with open(filename, encoding=encoding, errors="replace") as fin:
            for line in fin:
                fixedLine = line.replace('\ufffd', ' ')
                fixedLine = line.replace(chr(25), "\t")
                #if line != fixedLine:
                #    print(line)
                #    print(len(line.split("\t")))
                #    print(fixedLine)
                #    print(len(fixedLine.split("\t")))
                inputfile.write(fixedLine)
        inputfile.seek(0)

        indf = pd.read_csv(inputfile, sep="\t", encoding=encoding, error_bad_lines=error_bad_lines, dtype="O")

        for ri, row in indf.iterrows():

            try:
                aaSequence = row["Sequence"]
                molWeight = float(row["Mass"])
                mzWeight = self.get_mz_from_mass(molWeight)
            except ValueError:
                self.logger.error("Unable to process line {}: {}".format(ri, row))
                continue
                


            if not pd.isna(row[name_column]):
                proteinNames = name_function(row[name_column])
            else:
                proteinNames = ["{}_{}".format(aaSequence, mzWeight)]


            if self.max_mass >= 0 and mzWeight > self.max_mass:
                continue    

            if self.min_mass >= 0 and mzWeight < self.min_mass:
                continue

            if len(proteinNames) == 0:
                self.logger.warn("Skipping row for no gene name {}".format(aaSequence))
                continue

            for proteinName in proteinNames:
                ppmDist = self.get_ppm(mzWeight,self.ppm)
                self.mz_tree.addi(mzWeight - ppmDist, mzWeight + ppmDist, {"name": proteinName, "mass": molWeight, "mzWeight": mzWeight})

        allMasses = self.get_all_masses()
        self.logger.info("Loaded a total of {} proteins with {} masses".format(len(self.mz_tree), len(allMasses)))

class AnnotatedProteinWeights(ProteinWeights):

    def __init__(self, filename, min_mass=-1, max_mass=-1):
        super().__init__(filename, min_mass=min_mass, max_mass=max_mass)


    def get_annotated_info(self, mzsearch):
        raise NotImplementedError()


class SDFProteinWeights(AnnotatedProteinWeights):
    
    def __init__(self, filename, min_mass=-1, max_mass=-1):
        super().__init__(None, min_mass=min_mass, max_mass=max_mass)

        assert(filename.endswith(".sdf"))

        self.__load_file(filename)

    def __load_file(self, filename):

        self.sdf_dic = self.sdf_reader(filename)

        for lm_id in self.sdf_dic:


            molWeight = float(self.sdf_dic[lm_id]["EXACT_MASS"])
            mzWeight = self.get_mz_from_mass(molWeight)

            if "NAME" in self.sdf_dic[lm_id]:
                proteinName = self.sdf_dic[lm_id]["NAME"]
            elif "SYSTEMATIC_NAME" in self.sdf_dic[lm_id]:
                proteinName = self.sdf_dic[lm_id]["SYSTEMATIC_NAME"]

            if self.max_mass >= 0 and mzWeight > self.max_mass:
                continue    

            if self.min_mass >= 0 and mzWeight < self.min_mass:
                continue

            ppmDist = self.get_ppm(mzWeight,self.ppm)
            self.mz_tree.addi(mzWeight - ppmDist, mzWeight + ppmDist, {"name": proteinName, "mass": molWeight, "mzWeight": mzWeight, "category": self.sdf_dic[lm_id]["MAIN_CLASS"]})


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

