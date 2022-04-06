import gffutils
import os, sys
from Bio.Seq import Seq
import csv

from Bio.SeqUtils import molecular_weight
from rpg.digest import digest_one_sequence
from rpg.sequence import Sequence
from rpg.enzyme import Enzyme
from rpg import core

from trypsin import get_trypsin

baseDir = "/usr/local/hdd/rita/hiwi/Homo_sapiens/"

in_file = "Homo_sapiens.GRCh38.105.gtf"
myFasta = baseDir + "Homo_sapiens.GRCh38.dna.toplevel.fa"

dbFile = baseDir + in_file + ".db"

if os.path.isfile(dbFile):
    print("Loading existing database", file=sys.stderr)
    db = gffutils.FeatureDB(dbFile, keep_order=True)
    print("Loading existing database done", file=sys.stderr)

else:

    db = gffutils.create_db(baseDir + in_file, dbfn=dbFile, id_spec={'gene': 'gene_id', 'transcript': "transcript_id"}, disable_infer_transcripts=True, disable_infer_genes=True, merge_strategy="create_unique", keep_order=True)


def getAttribute(attr, aname, dval):

    if aname in attr:
        return attr[aname][0]
    
    return dval

trypsin_enzyme = get_trypsin()
f = open(baseDir + in_file + ".tsv", 'w')
writer = csv.writer(f, delimiter = "\t")
writer.writerow(["protein_id", "gene_symbol", "peptide_seq", "mol_weight_kd", "mol_weight", None])

for t in db.features_of_type('transcript', order_by='start'): # or mRNA depending on the gff
    

    geneID = getAttribute(t.attributes, "gene_id", None)
    geneName = getAttribute(t.attributes, "gene_name", None)
    transcriptID = getAttribute(t.attributes, "transcript_id", None)

    transcriptType = getAttribute(t.attributes, "transcript_biotype", None)
    transcriptTag = getAttribute(t.attributes, "tag", None)

    if not transcriptType in ["protein_coding"]:
        continue

    if geneID == None:
        continue

        
    seq_combined = ''

    stopCodons = []
    for i in db.children(t, featuretype="stop_codon"):
        stopCodons.append(i)

    startCodons = []
    for i in db.children(t, featuretype="start_codon"):
        startCodons.append(i)

    if len(stopCodons) > 0:

        #print(t.id, len(startCodons), len(stopCodons))
        allExons = [i for i in db.children(t, featuretype='CDS')]
        allStartCodons = [i for i in db.children(t, featuretype='start_codon')]
        allExons = sorted(allExons, key=lambda x: x.start)

        numStartCodons = sum([1 for x in allStartCodons])

        if numStartCodons != 1:
            #print(transcriptID, geneName, numStartCodons, transcriptTag, "no start", file=sys.stderr)
            continue

        seq = ""
        for exon in allExons:
            useq = exon.sequence(myFasta, use_strand=False)
            seq += useq

        seq = seq.upper()

        if t.strand == "-":
            seq = Seq(seq).reverse_complement()

        else:
            seq = Seq(seq)


        try:
            aaSeq = seq.translate(stop_symbol="")
        except:
            print("error", file=sys.stderr)
            print(transcriptID, geneName, file=sys.stderr)
            print(seq, file=sys.stderr)
            print(aaSeq, file=sys.stderr)
            continue

        aaSeq2digest = Sequence(header='', sequence=str(aaSeq))

        
        result_digestion = digest_one_sequence(seq=aaSeq2digest, enz=[trypsin_enzyme], mode='sequential', aa_pka=core.AA_PKA_IPC)
        for peptide in result_digestion[0].peptides:
            peptideWeight = molecular_weight(Seq(peptide.sequence), seq_type="protein")
            peptideWeightMono = molecular_weight(Seq(peptide.sequence), seq_type="protein", monoisotopic=True)
            if geneName:
                # protein_id	gene_symbol	mol_weight_kd	mol_weight  --- transcript_id + where it was found
                writer.writerow([geneName.upper(), geneName, peptide.sequence, peptideWeight/1000, peptideWeight, "normal"])
                writer.writerow([geneName.upper(), geneName, peptide.sequence, peptideWeightMono/1000, peptideWeightMono, "monoisotopic"])

f.close()




        
    