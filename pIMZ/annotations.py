# general
from email.policy import default
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
import pronto
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
from .plotting import Plotter

#web/html
import jinja2

from matplotlib.cm import ScalarMappable
from statsmodels.stats.multitest import multipletests
from scipy.stats import hypergeom

# applications
import progressbar
def makeProgressBar(total=None) -> progressbar.ProgressBar:
    return progressbar.ProgressBar(maxval=total, widgets=[
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

    def __init__(self, filename, massMode=+1, ppm=5, mz_ppm=1, min_mass=-1, max_mass=-1):
        """Creates a ProteinWeights class. Requires a formatted proteinweights-file.

        Args:
            filename (str): File with at least the following columns: protein_id, gene_symbol, mol_weight_kd, mol_weight.
            massMode (int): +1 if mz-Values were captured in M+H mode, -1 if mZ-Values were captured in M-H mode (protonated or deprotonated), https://github.com/rformassspectrometry/CompoundDb/issues/38, https://www.uni-saarland.de/fileadmin/upload/lehrstuhl/jauch/An04_Massenspektroskopie_Skript_Volmer.pdf
            ppm (int, optional): ppm (parts per million) error used for lookup. Default is 5.
            mz_ppm (int, optional): ppm error for storing masses. Small for exact masses, larger for experimentally derived masses. Default is 1.
            max_mass (float, optional): Maximal mass to consider/include in object. -1 for no filtering. Masses above threshold will be discarded. Default is -1.
            max_mass (float, optional): Minimal mass to consider/include in object. -1 for no filtering. Masses below threshold will be discarded. Default is -1.
        """

        self.__set_logger()

        self.min_mass = min_mass
        self.max_mass = max_mass
        self.ppm = ppm
        self.mz_ppm = mz_ppm
        self.massMode = massMode
        
        self.mz_tree = IntervalTree()
        self.data2slot = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

        if filename != None:
            self.__load_file(filename)


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
                    ppmDist = self.get_ppm(mzWeight, self.mz_ppm)
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
            mass2prot[ interval.data["mass"] ].update(interval.data["name"])

        return mass2prot

    def get_ppm(self, mz, ppm=None):
        
        if ppm is None:
            ppm = self.ppm
        
        return mz * (ppm / 1000000)

    def get_ppm_range(self, mz):
        ppmError = self.get_ppm(mz, self.ppm)
        return (mz-ppmError, mz+ppmError)

    def annotate_dfs(self, dfs, ppm=5, mass_column="gene_mass", add_matches=True):

        for x in dfs:
            dfs[x] = self.annotate_df(dfs[x], ppm=ppm, mass_column=mass_column, add_matches=add_matches)


    def annotate_df(self, df, ppm=None, mass_column="gene_mass", add_matches=True):
        
        uniqmasses = [0] * df.shape[0]
        hitProteins = [[]] * df.shape[0]

        for ri, (_, row) in enumerate(df.iterrows()):

            curmass = float(row[mass_column])
            allHitMasses = self.get_protein_from_mz(curmass, ppm=ppm)

            uniqmasses[ri] = len(allHitMasses)
            hitProteins[ri] = [x[0] for x in allHitMasses]

        if add_matches:
            df["#matches"] = uniqmasses

        df["matches"] = hitProteins

        return df

    def to_single_match_df(self, df):
        return df.explode('matches')


    def get_potential_celltypes(self, df, organs=["Smooth muscle", "Vasculature", "Connective tissue", "Immune system", "Heart", "Epithelium"], script_url="https://raw.githubusercontent.com/mjoppich/scrnaseq_celltype_prediction/master/analyseMarkers.py", threshold=0.5, force_update=False):

        #download script

        if not os.path.exists("analyseMarkers.py") or force_update:
            self.logger.info("Downloading analyseMarkers.py")
            import requests
            r = requests.get(script_url, allow_redirects=True)
            open("analyseMarkers.py", 'wb').write(r.content)

        
        df.explode("matches").to_csv("allmarkers.tsv", sep="\t", index=None)
        
        callArgs = ["python3", "analyseMarkers.py", "-i", "allmarkers.tsv", "--cluster", "cluster", "--gene", "matches", "--expr-mean", "mean.1", "--expressing-cell-count", "anum.1", "--cluster-cell-count", "num.1", "--logfc", "log2FC", "--pvaladj", "p_value_adj", "-n", "10", "--organs"] + ["'{}'".format(x) for x in organs]
        

        self.logger.info("Calling analyseMarkers.py")
        output = subprocess.check_output(" ".join(callArgs), shell=True).decode()

        self.logger.info("Processing output")
        from io import StringIO
        infile = StringIO(output)

        cluster2celltype = list()
        for line in infile:

            line = line.strip().split("\t")
            clusterID = eval(line[0])
            cellType = line[1]
            cellTypeScore = float(line[2])

            if cellTypeScore > threshold:
                cluster2celltype.append((clusterID, cellType, cellTypeScore))

        return pd.DataFrame.from_records(cluster2celltype, columns=["Cluster", "CellType", "Score"])

    def get_closest_mz_for_protein(self, prot_name, match_name, mz_bst, ppm=None):
        
        
        #print(prot_name, match_name)
        
        protMZValues = self.get_data_for_protein(prot_name, pw_match_name=match_name, data_slot="mzWeight")
        acceptedMZ = set()
        acceptedIndices = set()
        #print("protMZValues", protMZValues)
        for protMZ in protMZValues:
            
            closestMZ, closestIdx = mz_bst.findClosestValueTo(protMZ)
            #print("ClosestMZ", protMZ, "-->", closestMZ)
            
            if closestMZ is None:
                print("Could not find", protMZ)
                continue
                        
            possibleFeatures = self.get_protein_from_mz(closestMZ, match_name=match_name, ppm=ppm)
            possibleFeatures = [x[0] for x in possibleFeatures]
            
            #print("possible Features", possibleFeatures)
            #print("Contained", prot_name in possibleFeatures)
            
            if prot_name in possibleFeatures:
                acceptedMZ.add(closestMZ)
                
        return acceptedMZ
            
    def get_closest_mz_for_proteins(self, prot_name, match_name, mz_bst, ppm=None):
        retSet = set()
        
        for x in prot_name:
            foundSet = self.get_closest_mz_for_protein(x, match_name, mz_bst, ppm=ppm)
            retSet.update(foundSet)
            
        return retSet


    def get_protein_from_mz(self, mz, ppm=None, match_name="name", data_slot="mzWeight"):
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
            
            protMass = overlap[2][data_slot]
            protDist = abs(mz-protMass)
            
            for name in overlap[2][match_name]:
                possibleMatches.append((name, protMass, protDist))

        possibleMatches = sorted(possibleMatches, key=lambda x: x[2])
        possibleMatches = [(x[0], x[1]) for x in possibleMatches]

        return possibleMatches

    def get_protein_from_mass(self, mass, ppm=5):
        mz = self.get_mz_from_mass(mass)
        return self.get_protein_from_mz(mz, ppm=ppm)

    def get_mass_for_protein(self, protein):
        return self.get_data_for_protein(protein, data_slot="mass")
    
    def get_mz_for_protein(self, protein):
        return self.get_data_for_protein(protein, data_slot="mzWeight")

    def get_data_for_protein(self, protein, data_slot="mzWeight", pw_match_name="name"):
        """Returns all recorded values in data_slot for given proteins. Return None if protein not found

        Args:
            protein (str|iterable): protein to search for in database (exact matching)
            data_slot (str): data slot in interval to report

        Returns:
            set: set of masses for protein
        """
        
        if not pw_match_name in self.data2slot or not data_slot in self.data2slot[pw_match_name]:
            self.logger.info("Creating data2slot for data={} and slot={}".format(pw_match_name, data_slot))
            for interval in self.mz_tree:
                for elem in interval.data[ pw_match_name ]:
                    self.data2slot[pw_match_name][data_slot][elem].add(interval.data[data_slot])

        if type(protein) == str:
            return self.data2slot.get(pw_match_name, {}).get(data_slot, {}).get(protein, [])

        allData = set()
        for x in protein:
            protData = self.get_data_for_protein(x)
            allData = allData.union(protData)

        return allData


    def get_data_for_proteins(self, proteins, pw_match_name="name", data_slot="mass"):

        dataCounter = Counter()

        for x in proteins:
            datas = self.get_data_for_protein(x, data_slot=data_slot, pw_match_name=pw_match_name)
            
            for y in datas:
                dataCounter[y] += 1

        return set(dataCounter)


    def print_collisions(self):
        """Prints number of proteins with collision, as well as mean and median number of collisions.

        For each recorded mass it is checked how many other proteins are in the (mass-maxdist, mass+maxdist) range.

        Args:
            maxdist (float, optional): Mass range in order to still accept a protein. Defaults to 2.
            print_proteins (bool, optional): If True, all collision proteins are printed. Defaults to False.
        """


          

class MaxquantPeptides(ProteinWeights):

    def __init__(self, filename, massMode=+1, ppm=5, mz_ppm=5, min_mass=-1, max_mass=-1, name_column="Gene names", name_function=lambda x: x.strip().split(";"), encoding='Windows-1252', error_bad_lines=False, warn_bad_lines=False,carbamidomethylation=False, propionamido=False):
        super().__init__(None, massMode=massMode, ppm=ppm, min_mass=min_mass, max_mass=max_mass, mz_ppm=mz_ppm)

        self.__load_file(filename, name_column=name_column, name_function=name_function, encoding=encoding, error_bad_lines=error_bad_lines, warn_bad_lines=warn_bad_lines, carbamidomethylation=carbamidomethylation, propionamido=propionamido)

    @classmethod
    def split(string, delimiters):
        import re
        regex_pattern = '|'.join(map(re.escape, delimiters))
        return re.split(regex_pattern, string, 0)

    def __load_file(self, filename, name_column="Gene names", name_function=lambda x: x.strip().replace("_HUMAN", "").split(";"), encoding='Windows-1252', warn_bad_lines=False, error_bad_lines=False, carbamidomethylation=False, propionamido=False):

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

        indf = pd.read_csv(inputfile, sep="\t", encoding=encoding, error_bad_lines=error_bad_lines, warn_bad_lines=warn_bad_lines, dtype="O")

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
            
            if carbamidomethylation or propionamido:
                numCysteins = Counter(aaSequence)["C"]
                #https://bioportal.bioontology.org/ontologies/PSIMOD?p=classes&conceptid=MOD:00969
                if carbamidomethylation:
                    molWeight = molWeight - (numCysteins * 57.02)
                    
                if propionamido:
                    molWeight = molWeight - (numCysteins * 71)

            for proteinName in proteinNames:
                ppmDist = self.get_ppm(mzWeight, self.mz_ppm)
                self.mz_tree.addi(mzWeight - ppmDist, mzWeight + ppmDist, {"name": [proteinName], "mass": molWeight, "mzWeight": mzWeight})

        allMasses = self.get_all_masses()
        self.logger.info("Loaded a total of {} proteins with {} masses".format(len(self.mz_tree), len(allMasses)))

class AnnotatedProteinWeights(ProteinWeights):

    def __init__(self, filename, massMode=+1, ppm=5, mz_ppm=1, min_mass=-1, max_mass=-1):
        super().__init__(filename, min_mass=min_mass, max_mass=max_mass, massMode=massMode, ppm=ppm, mz_ppm=mz_ppm)


    def get_annotated_info(self, mzsearch):
        raise NotImplementedError()


class SDFProteinWeights(AnnotatedProteinWeights):
    """ Used for HMDB metabolite structures

    Args:
        AnnotatedProteinWeights (_type_): _description_
    """
    
    def __init__(self, filename, massMode=+1, ppm=5, mz_ppm=1, min_mass=-1, max_mass=-1):
        super().__init__(filename, min_mass=min_mass, max_mass=max_mass, massMode=massMode, ppm=ppm, mz_ppm=mz_ppm)

        assert(filename.endswith(".sdf"))

        self.__load_file(filename)

    def __load_file(self, filename):

        self.sdf_dic = self.sdf_reader(filename)
        self.category2element = defaultdict(set)

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
        
            ppmDist = self.get_ppm(mzWeight, self.mz_ppm)
            
            intervalData = {"name": [proteinName],
                            "mass": molWeight,
                            "mzWeight": mzWeight,
                            "category": [ self.sdf_dic[lm_id]["MAIN_CLASS"] ]}
            
            self.mz_tree.addi(mzWeight - ppmDist, mzWeight + ppmDist, intervalData)
            
            for cat in intervalData["category"]:
                self.category2element[cat].add(proteinName)

    def get_enrichment_data(self):
        
        enrichmentData = {}
        
        for x in self.category2element:
            enrichmentData[ x ] = ("{}".format(x), set(self.category2element[x]))
            
        return enrichmentData

    @classmethod
    def sdf_reader(cls, filename, dbIdentifier = "LM_ID"):
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
                    if dbIdentifier in line:
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






class SLProteinWeights(AnnotatedProteinWeights):
    """Swiss Lipids Annotation

    Args:
        AnnotatedProteinWeights (_type_): _description_
    """
    
    def __init__(self, filename, massMode=+1, ppm=5, mz_ppm=1, min_mass=-1, max_mass=-1, massColumn="Exact m/z of [M+H]+", nameColumn="metabolite name",categoryColumn = "lipid class"):
        super().__init__(None, min_mass=min_mass, max_mass=max_mass, massMode=massMode, ppm=ppm, mz_ppm=mz_ppm)

        assert(filename.endswith(".tsv"))

        self.__load_file(filename, massColumn=massColumn, nameColumn=nameColumn, categoryColumn=categoryColumn)

    def __load_file(self, filename, massColumn, nameColumn, categoryColumn):

        self.sl_df = pd.read_csv(filename, sep="\t")

        
        self.category2name = defaultdict(set)

        for lineIndex, row in self.sl_df.iterrows():

            molWeight = float(row[massColumn])
            mzWeight = self.get_mz_from_mass(molWeight)
            
            proteinName = row[nameColumn]

            if np.isnan(molWeight) or np.isnan(mzWeight):
                continue

            if self.max_mass >= 0 and mzWeight > self.max_mass:
                continue    

            if self.min_mass >= 0 and mzWeight < self.min_mass:
                continue
            
            categories = row[categoryColumn].split(" | ")

            ppmDist = self.get_ppm(mzWeight, self.mz_ppm)
            self.mz_tree.addi(mzWeight - ppmDist, mzWeight + ppmDist, {"name": [proteinName],
                                                                       "mass": molWeight,
                                                                       "mzWeight": mzWeight,
                                                                       "category": categories})
            
            for cat in categories:
                self.category2name[cat].add(proteinName)

    def get_enrichment_data(self):
        
        enrichmentData = {}
        
        for x in self.category2name:
            enrichmentData[ x ] = ("{}".format(x), set(self.category2name[x]))
            
        return enrichmentData


class PBProteinWeights(AnnotatedProteinWeights):
    """PBProteinWeights uses information from PathBank to acquire information about metabolites/small molecules and their corresponding pathways.
    
    You have to download the structures (as one file) and pathway annotations from http://pathbank.org/downloads .

    Args:
        AnnotatedProteinWeights (_type_): Objeckt from AnnotatedProteinWeights to query relevant masses.
    """
        
    def __init__(self, filenameSDF, filenameMetabolites, massMode=+1, ppm=5,mz_ppm=1, min_mass=-1, max_mass=-1, massColumn="Exact m/z of [M+H]+", nameColumn="metabolite name",categoryColumn = "lipid class", organisms=['Homo sapiens', 'Escherichia coli', 'Mus musculus',
       'Arabidopsis thaliana', 'Saccharomyces cerevisiae', 'Bos taurus',
       'Caenorhabditis elegans', 'Rattus norvegicus',
       'Drosophila melanogaster', 'Pseudomonas aeruginosa']):
        super().__init__(None, min_mass=min_mass, max_mass=max_mass, massMode=massMode, ppm=ppm, mz_ppm=mz_ppm)

        self.logger.info("Reading Pathways")
        self.pb_df = pd.read_csv(filenameMetabolites, sep=",")       
        self.category2name = defaultdict(set)
        self.metabolite2pathway = defaultdict(set)
        self.pathway2metabolite = defaultdict(set)
        self.pathway2name = {}
                
        
        ignoredPathways = set()
        self.logger.info("Processing Pathways")
        bar = makeProgressBar(total=self.pb_df.shape[0])
        for lineIndex, row in bar(self.pb_df.iterrows()):
            
            pbID = row["PathBank ID"]
            pbName = row["Pathway Name"]
            metaboliteID = row["Metabolite ID"]
            
            if not row["Species"] in organisms:
                ignoredPathways.add(pbID)
                continue
            
            if not pbID in self.pathway2name:
                self.pathway2name[pbID] = pbName
            
            self.metabolite2pathway[metaboliteID].add( pbID )
            self.pathway2metabolite[pbID].add( metaboliteID )

        self.logger.info("Ignored {} of {} pathways".format(len(ignoredPathways), len(ignoredPathways)+len(self.pathway2metabolite)))


        self.logger.info("Reading Structures")
        self.sdf_dic = SDFProteinWeights.sdf_reader(filenameSDF, dbIdentifier="DATABASE_ID")
        
        self.logger.info("Processing Structures")
        bar = makeProgressBar()
        for lm_id in bar(self.sdf_dic):
            
            if not "MOLECULAR_WEIGHT" in self.sdf_dic[lm_id]:
                print(self.sdf_dic[lm_id])
                continue

            molWeight = float(self.sdf_dic[lm_id]["MOLECULAR_WEIGHT"])
            mzWeight = self.get_mz_from_mass(molWeight)
            metaboliteID = lm_id
            
            if self.max_mass >= 0 and mzWeight > self.max_mass:
                continue    

            if self.min_mass >= 0 and mzWeight < self.min_mass:
                continue

            ppmDist = self.get_ppm(mzWeight, self.mz_ppm)
            
            self.mz_tree.addi(mzWeight - ppmDist, mzWeight + ppmDist, {"name": [metaboliteID],
                                                                       "mass": molWeight,
                                                                       "mzWeight": mzWeight,
                                                                       "pathway": self.metabolite2pathway.get(metaboliteID, []),
                                                                       "pathway_name": [self.pathway2name[x] for x in self.metabolite2pathway.get(metaboliteID, [])],
                                                                       "pathway_pretty_name": ["{} ({})".format(self.pathway2name[x], x) for x in self.metabolite2pathway.get(metaboliteID, [])]
                                                                       })
            



    def get_mz_for_category(self, category):
        
        if not category in self.category2name:
            self.logger.error("Not a valid category: " + str(category))
            return None
        
        return self.get_data_for_proteins(self.category2name[category], data_slot="mzWeight")

    
    def get_enrichment_data(self):
        
        enrichmentData = {}
        
        for x in self.pathway2metabolite:
            enrichmentData[ x ] = ("{} ({})".format(self.pathway2name[x], x), set(self.pathway2metabolite[x]))
            
        return enrichmentData
    
    
class ChebiProteinWeights(AnnotatedProteinWeights):
    """PBProteinWeights uses information from PathBank to acquire information about metabolites/small molecules and their corresponding pathways.
    
    You have to download the structures (as one file) and pathway annotations from http://pathbank.org/downloads .

    Args:
        AnnotatedProteinWeights (_type_): Objeckt from AnnotatedProteinWeights to query relevant masses.
    """
    
    def __init__(self, chebiOboCorePath, massMode=+1, ppm=5, mz_ppm=1, min_mass=-1, max_mass=-1):
        super().__init__(None, min_mass=min_mass, max_mass=max_mass, massMode=massMode, ppm=ppm, mz_ppm=mz_ppm)

        self.massMode = 0

        self.logger.info("Reading Pathways")
        self.gr = pronto.Ontology(chebiOboCorePath)
        
        def get_mass_property( term ):
            
            if isinstance(term, pronto.relationship.Relationship):
                return None
            
            for x in term.annotations:
                if x.property == "http://purl.obolibrary.org/obo/chebi/mass":
                    return float(x.literal)
            
            return None

        self.chebiSets = defaultdict(set)
        self.metabolite2set = defaultdict(set)
        self.chebiID2Name = {}
        
        bar = makeProgressBar()

        for x in bar(self.gr):
            xterm = self.gr[x]
            
            if xterm.obsolete:
                continue
            
            if isinstance(xterm, pronto.relationship.Relationship):
                continue
            
            molWeight = get_mass_property(self.gr[x])
                        
            self.chebiID2Name[xterm.id] = xterm.name

            if molWeight is None:
                # not an entity
                for subElem in xterm.subclasses(None):
                    molWeight = get_mass_property(subElem)
                    
                    if molWeight is None:
                        continue
                    self.chebiSets[xterm.id].add(subElem.id)
                    
        for setName in self.chebiSets:
            for metabolite in self.chebiSets[setName]:
                self.metabolite2set[metabolite].add(setName)


        self.logger.info("Reading Metabolites")
        bar = makeProgressBar()
        
        for x in bar(self.gr):
            xterm = self.gr[x]
            
            if xterm.obsolete:
                continue
            
            if isinstance(xterm, pronto.relationship.Relationship):
                continue
            
            molWeight = get_mass_property(self.gr[x])
            
            if not molWeight is None:
                #it is an entity
                
                if np.isnan(molWeight) or molWeight <= 0:
                    continue
                
                mzWeight = self.get_mz_from_mass(molWeight)
                metaboliteID = xterm.id
                
                if self.max_mass >= 0 and mzWeight > self.max_mass:
                    continue    

                if self.min_mass >= 0 and mzWeight < self.min_mass:
                    continue

                ppmDist = self.get_ppm(mzWeight, self.mz_ppm)
                
                
                self.mz_tree.addi(mzWeight - ppmDist, mzWeight + ppmDist, {"name": [metaboliteID], "mass": molWeight,
                                                                           "mzWeight": mzWeight,
                                                                           "pathway": self.metabolite2set.get(metaboliteID, []),
                                                                           "pathway_name": [self.chebiID2Name[x] for x in self.metabolite2set.get(metaboliteID, [])]})
                

    def get_mz_for_category(self, category):
        
        if not category in self.category2name:
            self.logger.error("Not a valid category: " + str(category))
            return None
        
        return self.get_data_for_proteins(self.category2name[category], data_slot="mzWeight")


    def get_enrichment_data(self):
        
        enrichmentData = {}
        
        for x in self.chebiSets:
            enrichmentData[ x ] = ("{} ({})".format(self.chebiID2Name[x], x), set(self.chebiSets[x]))
            
        return enrichmentData
    

class PPIAnalysis:

    def __init__(self, load_shao=False, load_omnipath=False, load_jin=False, geneInfo="uniprot_human_go_kegg_pro_pfam.tab"):

        print("Loading gene information")
        geneInfo, protInfo = self.loadGeneInfo(geneInfo, to_upper=True)

        loadedPPIs = []

        if load_omnipath:
            print("Loading PPI (human)")
            ppiInfo_omnipath = self.load_OmnipathPPI(prot_info=protInfo)
            loadedPPIs.append(ppiInfo_omnipath)

        if load_jin:
            print("Loading LR (mouse)")
            ppiInfo_jin = self.load_RL_mouse_jin()
            loadedPPIs.append(ppiInfo_jin)


        if load_shao:
            print("Loading LR (human)")
            ppiInfo_shao = self.load_LR_human_Shao()
            loadedPPIs.append(ppiInfo_shao)

        print("Combining PPI")
        self.ppiInfo = self.combinePPI(loadedPPIs)


    def get_neighbouring_clusters(self, spec, grouping):

        metaArray = spec.meta[grouping]

        allNeighbours = set()

        for i in range(1, metaArray.shape[0]-1):
            for j in range(1, metaArray.shape[1]-1):

                for k in range(-1, 2):
                    for l in range(-1, 2):

                        if metaArray[i,j] == metaArray[i+k, j+l]:
                            continue

                        allNeighbours.add( (metaArray[i,j], metaArray[i+k, j+l]) )
                        allNeighbours.add( (metaArray[i+k, j+l], metaArray[i,j]) )

        
        return [("mean.{}".format(x), "mean.{}".format(y)) for x,y in allNeighbours]


    def analyse(self, meanExprDF, outputName, outputFolder, index_col="mass", clusters=None, title="", sel_source=None, sel_target=None, min_comm=0, selfCommunication=False, neighbours=None):

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder, exist_ok=True)


        clusterNames = [x for x in meanExprDF.columns if x.startswith("mean")]
        clusterNames = natsorted(clusterNames)

        if not clusters is None and len(clusters) > 0:

            useClusterNames = []
            for x in clusters:

                if x in meanExprDF.columns:
                    useClusterNames.append(x)
                elif "mean."+x in meanExprDF.columns:
                    useClusterNames.append("mean."+x)
                elif not x in meanExprDF.columns:
                    print(x, "is not a valid column of", meanExprDF.columns)
                    exit(-1)
        else:
            useClusterNames = clusterNames

        scExpr = meanExprDF[ [index_col] + useClusterNames] 


        titleDescr = title
        cluster2color = Plotter.getClusterColors(clusterNames)

        srcHighlights = None
        tgtHighlights = None


        if not sel_source is None:
            allCols = clusterNames
            srcHighlights = set()
            for x in allCols:
                for y in sel_source:
                    if y in x:
                        srcHighlights.add(x)

        if not sel_target is None:
            allCols = clusterNames
            tgtHighlights = set()
            for x in allCols:
                for y in sel_target:
                    if y in x:
                        tgtHighlights.add(x)

        print("Selected sources:", srcHighlights)
        print("Selected targets:", tgtHighlights)

        sdf = self.plot_interactome_chord_matrix(scExpr.copy(), outputFolder, outputName, titleDescr, srcHighlights, tgtHighlights, None, cluster2color, index_col, min_comm, selfCommunication=selfCommunication, neighbours=neighbours)

        return sdf        



    def communication_score(self, expr1, expr2):
        return expr1*expr2

    def make_space_above(self, ax, topmargin=1):
        """ increase figure size to make topmargin (in inches) space for 
            titles, without changing the axes sizes"""
        fig = ax.figure
        s = fig.subplotpars
        w, h = fig.get_size_inches()

        figh = h - (1-s.top)*h  + topmargin
        fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
        fig.set_figheight(figh)



    def plot_interactome_chord_matrix(self, scExpr, outputFolder, outputName, titleDescr, sourceElems, targetElems, highlightTuple, cluster2color, index_col, min_comm, selfCommunication, neighbours):

        scExpr = scExpr.set_index(index_col)
        scExpr = scExpr.apply(pd.to_numeric, errors='ignore')

        clusterNames = natsorted([x for x in scExpr.columns])

        allLigands = list(self.ppiInfo.src_gene)
        allReceptors = list(self.ppiInfo.tgt_gene)

        containedLigs = natsorted([ x for x in scExpr.index if x.upper() in allLigands ])
        containedRecs = natsorted([ x for x in scExpr.index if x.upper() in allReceptors ])

        print("Contained Ligs", len(containedLigs))
        print("Contained Recs", len(containedRecs))

        csScores = defaultdict(lambda : Counter())
        csScoresij = defaultdict(list)

        allCsScores = []

        lr_interactions = []
        for rownum, row in self.ppiInfo.iterrows():
            lr_interactions.append( (row["src_gene"], row["tgt_gene"], row["direction"]) )


        cslrijCount = 0
        cslrij_mean_sum = 0
        cslrij_taken_mean_sum = 0
        for ligand, receptor, inttype in set(lr_interactions):

            if not ligand in containedLigs:
                continue
            if not receptor in containedRecs:
                continue

            for clusterI in clusterNames:
                for clusterJ in clusterNames:

                    if not selfCommunication:
                        if clusterI == clusterJ:
                            continue
                    
                    if not clusterI == clusterJ:
                        if not neighbours is None:
                            if not (clusterI, clusterJ) in neighbours:
                                continue

                    exprLigand = scExpr.loc[ligand, clusterI]
                    exprRecept = scExpr.loc[receptor, clusterJ]

                    cslrijCount += 1

                    if not isinstance(exprLigand, pd.Series):
                        exprLigand = [exprLigand]
                    if not isinstance(exprRecept, pd.Series):
                        exprRecept = [exprRecept]

                    for eL in exprLigand:
                        for eR in exprRecept:
                            cslrij = self.communication_score(eL, eR)

                            cslrij_mean_sum += cslrij

                            if cslrij < min_comm:
                                continue

                            cslrij_taken_mean_sum += cslrij

                            csScores[(ligand, receptor)][(clusterI, clusterJ)] += cslrij
                            csScoresij[(clusterI, clusterJ)].append((ligand, receptor, cslrij))
                            allCsScores.append((ligand, receptor, clusterI, clusterJ, cslrij))


        if len(allCsScores) == 0:
            print("total com scores", len(allCsScores))
            return

        print("total com scores", len(allCsScores))
        print("mean score calculated", cslrij_mean_sum/cslrijCount)
        print("mean score taken", cslrij_taken_mean_sum/len(allCsScores))

        df = pd.DataFrame(allCsScores, columns=["ligand", "receptor", "clusterI", "clusterJ", "score"])

        df.to_excel(os.path.join(outputFolder, "scores.{}.xlsx".format( outputName )))

        sumDF = df.groupby(["clusterI", "clusterJ"]).agg(score=("score", "sum"), count=("score", "count"))
        sumDF.to_excel(os.path.join(outputFolder, "agg.{}.xlsx".format( outputName )))
        selDF = sumDF

        allElems = set()
        for idx, row in selDF.iterrows():

            allElems.add(idx[0])
            allElems.add(idx[1])

        allElems = natsorted(allElems)

        predefinedElems = []
        if not sourceElems is None:
            predefinedElems += [x for x in sourceElems]
        if not targetElems is None:
            predefinedElems += [x for x in targetElems]

        allElems = predefinedElems + [x for x in allElems if not x in predefinedElems]

        flux = np.zeros((len(allElems), len(allElems)))

        for idx, row in selDF.iterrows():
            iElem = idx[0]
            jElem = idx[1]

            flux[allElems.index(iElem), allElems.index(jElem)] += row["score"]
            #flux[allElems.index(jElem), allElems.index(iElem)] += row["score"]

        clusterIDs = [x.split(".")[1] for x in allElems]
        allNames = ["Cluster " + x for x in clusterIDs]
        cmap = matplotlib.cm.get_cmap('viridis')
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=len(allNames))

        redsMap =  matplotlib.cm.get_cmap('Reds')

        allColors = []
        redI = 0
        for i, clusterID in enumerate(clusterIDs):
            clusterColor = cluster2color[clusterID]
            allColors.append(clusterColor)

        normReds = matplotlib.colors.Normalize(vmin=0, vmax=1)

        grpDF = sumDF.groupby("clusterI").agg("sum")
        print(grpDF)

        def getArcColor(i, j, n):

            if sourceElems is None:
                return allColors[i]

            print(allElems[i], "-->", allElems[j], allElems[i] in sourceElems)

            if allElems[i] in sourceElems:
                return allColors[i]

            elif allElems[j] in sourceElems:
                return allColors[j]

            else:
                return "#e6e6e6"


        fig, ax = plt.subplots(figsize=(10,10))

        plt.rcParams["image.composite_image"] = False

        targetScore = 0
        for x in predefinedElems:
            targetScore += grpDF.loc[x].score
        print("TargetScore for", len(predefinedElems), "Elements:", targetScore)

        startDegree = 90 - (((targetScore / grpDF.score.sum()) * 360) / 2)
        delayedArcs = [i for i,_ in enumerate(predefinedElems)]
        Plotter.chord_diagram(flux, allNames, delayedArcs=delayedArcs, startdegree=startDegree, ax=ax, gap=0.03, use_gradient=True, colors=allColors, sort="size", rotate_names=True, fontcolor="grey", arccolors=getArcColor)

        if not sourceElems is None or not targetElems is None:
            cmappable = ScalarMappable(norm=normReds, cmap=redsMap)
            cbar = fig.colorbar(cmappable, ax=ax, location = 'right', pad = 0.4)

            cbar.ax.set_ylabel("Cluster->Cluster Communication Score")
        
        self.make_space_above(ax, topmargin=2)   

        plt.suptitle("Interaction Map ({})".format(titleDescr), fontsize =20)
        plt.margins(1)
        plt.savefig(os.path.join(outputFolder, "chord_plot.{}.png".format( outputName )), bbox_inches='tight', dpi=500)
        plt.savefig(os.path.join(outputFolder, "chord_plot.{}.svg".format( outputName )), bbox_inches='tight')
        plt.savefig(os.path.join(outputFolder, "chord_plot.{}.pdf".format( outputName )), bbox_inches='tight')

        print(os.path.join(outputFolder, "chord_plot.{}.png".format( outputName )))
        sumDF["experiment"] = outputName

        return sumDF


    def to_list(self, input, delim=" "):
        if input == None or pd.isna(input) or input == "nan" or input == "None":
            return []

        if not type(input) == str:
            print(input)

        return [x for x in input.split(delim) if len(x) > 0]

    def loadGeneInfo(self, path, to_upper=True):
        uniprotDF = pd.read_csv(path, sep="\t", header=0)
        protInfo = {}
        geneInfo = {}

        for entry in uniprotDF.iterrows():

            entryInfo = entry[1].to_dict()

            if entryInfo["Entry"] in protInfo:
                print("doublet", entryInfo["Entry"])
                exit()


            entryInfo["Gene names"] = self.to_list(entryInfo["Gene names"])
            if to_upper:
                entryInfo["Gene names"] = [x.upper() for x in entryInfo["Gene names"]]

            entryInfo["GO"] = self.to_list(entryInfo["Gene ontology IDs"], "; ")
            entryInfo["InterPro"] = self.to_list(entryInfo["Cross-reference (InterPro)"], ";")
            entryInfo["PRO"] = self.to_list(entryInfo["Cross-reference (PRO)"], ";")
            entryInfo["KEGG"] = self.to_list(entryInfo["Cross-reference (KEGG)"], ";")
            entryInfo["PFAM"] = self.to_list(entryInfo["Cross-reference (Pfam)"], ";")
            entryInfo["SUBLOCATION"] = self.to_list(entryInfo["Subcellular location [CC]"], ";")

            del entryInfo["Gene ontology IDs"]
            del entryInfo["Cross-reference (InterPro)"]
            del entryInfo["Cross-reference (PRO)"]
            del entryInfo["Cross-reference (KEGG)"]
            del entryInfo["Cross-reference (Pfam)"]
            del entryInfo["Subcellular location [CC]"]

            if len(entryInfo["Gene names"]) == 0:
                continue
            geneName = entryInfo["Gene names"][0]

            if to_upper:
                geneName = geneName.upper()


            geneInfo[geneName] = entryInfo
            protInfo[entryInfo["Entry"]] = entryInfo

        return geneInfo, protInfo

    def load_OmnipathPPI(self, prot_info, path="OmniPathPPIs.tsv"):

        #TODO if file not there => download!
        ppiDF = pd.read_csv(path, sep="\t", header=0)

        ppiInfo = {}
        noConsensusInfo = 0
        noProtInfo = 0

        #interpro2ppi = defaultdict(list)
        #go2ppi = defaultdict(list)

        for entry in ppiDF.iterrows():

            entryInfo = entry[1].to_dict()

            consensusEntryInfo = {}
            
            for x in ["source","target"]:#,"consensus_direction","consensus_stimulation","consensus_inhibition"]:
                consensusEntryInfo[x] = entryInfo[x]

            if entryInfo["consensus_direction"] != 1:
                noConsensusInfo += 1
                consensusEntryInfo["direction"] = 1
            else:
                consensusEntryInfo["direction"] = entryInfo["consensus_direction"]

            (src, tgt) = consensusEntryInfo["source"], consensusEntryInfo["target"]

            if "COMPLEX" in src or "COMPLEX" in tgt:
                continue

            if (src, tgt) in ppiInfo:
                print("doublet", (src, tgt))
                exit()

            if not src in prot_info:
                print("no src info", src)
                noProtInfo += 1
                continue

            if not tgt in prot_info:
                print("no src info", tgt)
                noProtInfo += 1
                continue

            srcInfo = prot_info[src]
            tgtInfo = prot_info[tgt]

            try:
                consensusEntryInfo["src_gene"] = srcInfo["Gene names"][0]
                consensusEntryInfo["tgt_gene"] = tgtInfo["Gene names"][0]
            except:
                print(srcInfo)
                print(tgtInfo)

                break

            ppiInfo[(consensusEntryInfo["src_gene"], consensusEntryInfo["tgt_gene"])] = consensusEntryInfo

        print("sorted out", noConsensusInfo, "of", len(ppiDF), "PPIs for no consensus direction")
        print("sorted out", noProtInfo, "of", len(ppiDF), "PPIs for no protein info")

        return ppiInfo

    def load_RL_mouse_jin(self, path="Mouse-2020-Jin-LR-pairs.csv"):

        #TODO if file not there => download!

        ppiDF = pd.read_csv(path, header=0)

        ppiInfo = {}
        noConsensusInfo = 0
        noProtInfo = 0


        for entry in ppiDF.iterrows():

            entryInfo = entry[1].to_dict()
            consensusEntryInfo = {}
            consensusEntryInfo["direction"] = 1

            (src, tgt) = entryInfo["ligand_symbol"], entryInfo["receptor_symbol"]
            src = src.upper()
            tgt = tgt.upper()

            consensusEntryInfo["source"] = src
            consensusEntryInfo["target"] = tgt

            srcGenes = src.split("&")
            tgtGenes = tgt.split("&")

            for srcGene in srcGenes:
                for tgtGene in tgtGenes:

                    consensusEntryInfo["src_gene"] = srcGene
                    consensusEntryInfo["tgt_gene"] = tgtGene

                    ppiInfo[(srcGene, tgtGene)] = dict(consensusEntryInfo)

        print("sorted out", noConsensusInfo, "of", len(ppiDF), "PPIs for no consensus direction")
        print("sorted out", noProtInfo, "of", len(ppiDF), "PPIs for no protein info")
        print("Total Interactions", len(ppiInfo))
        return ppiInfo


    def load_LR_human_Shao(self, path="Human-2020-Shao-LR-pairs.txt"):

        #TODO if file not there => download!

        ppiDF = pd.read_csv(path, sep="\t", header=0)

        ppiInfo = {}
        noConsensusInfo = 0
        noProtInfo = 0


        for entry in ppiDF.iterrows():

            entryInfo = entry[1].to_dict()
            consensusEntryInfo = {}
            consensusEntryInfo["direction"] = 1

            (src, tgt) = entryInfo["ligand_gene_symbol"], entryInfo["receptor_gene_symbol"]
            src = src.upper()
            tgt = tgt.upper()

            consensusEntryInfo["source"] = src
            consensusEntryInfo["target"] = tgt
            consensusEntryInfo["src_gene"] = src
            consensusEntryInfo["tgt_gene"] = tgt

            ppiInfo[(src, tgt)] = dict(consensusEntryInfo)

        print("sorted out", noConsensusInfo, "of", len(ppiDF), "PPIs for no consensus direction")
        print("sorted out", noProtInfo, "of", len(ppiDF), "PPIs for no protein info")
        print("Total Interactions", len(ppiInfo))
        return ppiInfo


    def combinePPI( self, ppis ):

        ppiInfo = {}

        for ppiInput in ppis:
            
            inputNewEntries = 0
            incompatiblePPIs = 0
            for x in ppiInput:

                if not x in ppiInfo:
                    ppiInfo[x] = ppiInput[x]
                    inputNewEntries += 1
                else:

                    if ppiInfo[x]["direction"] != ppiInput[x]["direction"]:
                        incompatiblePPIs += 1
                    
                    continue

            print("Inserted new entries {} of {} (incompatible entries not inserted {})".format(inputNewEntries, len(ppiInput), incompatiblePPIs))

        ggiDF = pd.DataFrame.from_dict(ppiInfo).T
        ggiDF = ggiDF.reset_index(drop=True)

        return ggiDF


class FlowAnalysis:

    def __init__(self):
        pass

    
    def create_flows(self, meanExprDF, symbol_column, seriesOrder):

        clusterNames = [x for x in meanExprDF.columns if x.startswith("mean")]

        takeClusterNames = []
        for x in clusterNames:
            if not x.split(".")[1] in seriesOrder:
                continue
            takeClusterNames.append(x)

        clusterNames = natsorted(takeClusterNames)


        maxValue = max(meanExprDF[takeClusterNames].max())
        minValue = min(meanExprDF[takeClusterNames].min())
        binSpace = np.linspace(minValue, maxValue, 5, True)

        exprDF = pd.DataFrame()
        for cname in takeClusterNames:
            
            cnum = cname.split(".")[1]

            bins = np.digitize(np.array(meanExprDF[cname]), binSpace)
            binName = "bins.{}".format(cnum)

            print(min(bins), max(bins))

            exprDF[cname] = meanExprDF[cname]
            exprDF[binName] = bins

        exprDF[symbol_column] = meanExprDF[symbol_column]
        print(exprDF.head())

        return exprDF



    def analyse_flows(self, exprDF, seriesOrder, series2name, levelOrder, symbol_column="matches"):


        #seriesOrder = ["WT", "KO"]
        #series2name = OrderedDict([("WT", "Wildtype"), ("KO", "Knockout")])
        #levelOrder = OrderedDict([(-2, 'LOW'), (-1, 'low'), (0, 'UnReg'), (1, 'high'), (2, 'HIGH')])

        flows = self.create_flows(exprDF, symbol_column, seriesOrder)

        for x in seriesOrder:
            assert( "bins.{}".format(x) in flows.columns)

        binColumns = [x for x in flows.columns if x.startswith("bins.")]

        flows['flowgroups'] = flows.apply( lambda row: tuple( [row[x] for x in binColumns] ), axis=1)
        allgroups = list(set(flows["flowgroups"]))
        flows['flowgroupid'] = flows.apply( lambda row: allgroups.index(row["flowgroups"]), axis=1)

        flowgroup_flow = defaultdict(lambda: 0)
        flowgroup_route = {}
        flowgroup_genes = defaultdict(set)

        for ri, row in flows.iterrows():
            fgid = row["flowgroupid"]
            fgflow = 1
            fgroute = row["flowgroups"]
            fggene = row[symbol_column]

            flowgroup_route[fgid] = [(x,y) for x,y in zip(seriesOrder, fgroute)]
            flowgroup_flow[fgid] += fgflow
            flowgroup_genes[fgid].add(fggene)

        weightSequence = []

        for fgid in flowgroup_route:
            weightSequence.append( tuple(flowgroup_route[fgid] + [flowgroup_flow.get(fgid, 0)]) )


        self._make_plot(weightSequence, series2name, levelOrder, transformCounts=lambda x: np.sqrt(x))


        rp = self.read_gmt_file("ReactomePathways.gmt")
        allDFs = []

        for fgid in flowgroup_genes:

            fg_genes = [str(x) for x in flowgroup_genes[fgid]]

            df = self.analyse_genes_for_genesets(rp, fg_genes)
            df["fgid"] = fgid

            print(fgid)
            print(flowgroup_route[fgid], len(fg_genes))
            #print(df[df["pwsize"] > 1].sort_values(["pval"], ascending=True).head(3))

            allDFs.append(df)

        allFGDFs = pd.concat(allDFs, axis=0)

        _ , elemAdjPvals, _, _ = multipletests(allFGDFs["pval"], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
        allFGDFs["adj_pval"] =elemAdjPvals

        return flows, allFGDFs


    def _make_plot( self, nodeWeigthSequence, s2n, lo, transformCounts = lambda x: x):

        series2index = [x for x in s2n]
        nodePositions = {}

        for nodeName in s2n:
            for nodeLevel in lo:

                nodePositions[ (nodeName, nodeLevel) ] = (series2index.index(nodeName), 2*nodeLevel)

        minNodeLevel = min([nodePositions[x][1] for x in nodePositions])
        maxNodeLevel = max([nodePositions[x][1] for x in nodePositions])

        #nodeWeigthSequence = [ (("WT", 2), ("KO", 0), 1), (("WT", 2), ("KO", -2), 1), (("WT", -1), ("KO", -2), 1) ]


        nodeOffsets = defaultdict(lambda: 0)

        colours = Plotter.generate_colormap(len(nodeWeigthSequence))

        maxFlowPerNode = defaultdict(lambda: 0)

        for si, nws in enumerate(nodeWeigthSequence):
            weight = transformCounts(nws[-1])
            nodes = nws[0:-1]

            for node in nodes:
                maxFlowPerNode[node] += weight

        maxFlowInAllNodes = max([x for x in maxFlowPerNode.values()])

        print(maxFlowInAllNodes)

        fig, ax = plt.subplots(figsize=(20,10))
        ax.axis('off')
        plt.title("")

        for si, nws in enumerate(nodeWeigthSequence):

            weight = transformCounts(nws[-1]) / maxFlowInAllNodes
            nodes = nws[0:-1]

            for i in range(1, len(nodes)):

                src = nodes[i-1]
                tgt = nodes[i]

                p1 = nodePositions[src]
                p2 = nodePositions[tgt]

                p1 = p1[0], p1[1] - nodeOffsets[src] + maxFlowPerNode[src]/maxFlowInAllNodes/2.0
                p2 = p2[0], p2[1] - nodeOffsets[tgt] + maxFlowPerNode[tgt]/maxFlowInAllNodes/2.0

                if tgt == ("KO", 0):
                    print(p2)

                xs, ys1, ys2 = Plotter.sigmoid_arc(p1, weight, p2, resolution=0.1, smooth=0, ax=ax)

                nodeOffsets[src] += weight
                #nodeOffsets[tgt] += weight

                c = colours[si % len(colours)]
                plt.fill_between(x=xs, y1=ys1, y2=ys2, alpha=0.5, color=c, axes=ax)



        props = dict(boxstyle='round', facecolor='grey', alpha=1.0, pad=1)

        for nn in nodePositions:

            nodeStr = "{lvl}".format(cond=s2n[nn[0]], lvl=lo[nn[1]])
            nodePosition = nodePositions[nn]

            ax.text(nodePosition[0], nodePosition[1], nodeStr, transform=ax.transData, fontsize=14,rotation=90,
                verticalalignment='center', ha='center', va='center', bbox=props)

        # place a text box in upper left in axes coords

        for si, series in enumerate(s2n):

            ax.text(si, minNodeLevel-1, s2n[series], transform=ax.transData, fontsize=14,rotation=0,
                verticalalignment='center', ha='center', va='center', bbox=props)


        plt.ylim((minNodeLevel-1.5, maxNodeLevel+0.5))

        plt.tight_layout()
        plt.show()

    def read_gmt_file(self, filepath):

        geneset2genes = {}

        with open(filepath) as fin:

            for line in fin:

                line = line.strip().split("\t")

                #print(line)
                pwName = line[0]
                pwID = line[1]
                pwGenes = line[2:]

                #print(pwName)
                #print(pwID)
                #print(pwGenes)

                geneset2genes[pwID] = (pwName, pwGenes)

        return geneset2genes


    def analyse_genes_for_genesets(self, pathways, genes, populationSize=None):
        """
        
        pathways: pathway object
        genes: genes of interest

        populationSize: size of universe. if None, all genes in pathways will be chosen
        
        """

        allGenes = list()
        for x in pathways:
            allGenes += pathways[x][1]

        allGenes = set(allGenes)

        if populationSize is None:
            populationSize = len(allGenes)

        geneSet = set([x.upper() for x in genes])
        numSuccInPopulation = len(geneSet.intersection(allGenes))

        
        outData = defaultdict(list)

        for pwID in pathways:

            pwName, pwGenes = pathways[pwID]

            sampleSize = len(pwGenes)
            successIntersection = geneSet.intersection(pwGenes)
            drawnSuccesses = len(successIntersection)

            pval = hypergeom.sf(max([0, drawnSuccesses - 1]), populationSize, numSuccInPopulation, sampleSize)

            if drawnSuccesses == 0:
                pval = 1.0

            # population: all genes
            # condition: genes
            # subset: pathway

            #print 'total number in population: ' + sys.argv[1] -> populationSize
            #print 'total number with condition in population: ' + sys.argv[2] -> numSuccInPopulation
            #print 'number in subset: ' + sys.argv[3] -> len(pwGenes)
            #print 'number with condition in subset: ' + sys.argv[4] -> drawnSuccesses
            #print 'p-value >= ' + sys.argv[4] + ': ' + str(stats.hypergeom.sf(drawnSuccesses - 1,populationSize,numSuccInPopulation,sampleSize))

            genes_coverage = drawnSuccesses / len(geneSet) if len(geneSet) > 0 else 0
            pathway_coverage = drawnSuccesses / sampleSize if sampleSize > 0 else 0

            outData["pwid"].append(pwID)
            outData["pwname"].append(pwName)
            outData["pwsize"].append(sampleSize)
            outData["genesize"].append(len(genes))
            outData["pw_gene_intersection"].append(drawnSuccesses)
            outData["pw_coverage"].append(pathway_coverage)
            outData["genes_coverage"].append(genes_coverage)
            outData["succes_genes"].append(successIntersection)
            outData["pval"].append(pval)
            outData["mean_coverage"].append(pathway_coverage*genes_coverage)

        return pd.DataFrame.from_dict(outData)