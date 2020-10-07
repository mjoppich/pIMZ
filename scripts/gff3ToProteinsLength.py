

import gffutils
import os, sys
from Bio.Seq import Seq

from Bio.SeqUtils import molecular_weight


baseDir = "/mnt/f/dev/data/genomes/"

#in_file = "hg38_primary_assembly_and_lncRNA.gtf"
#myFasta = "GRCh38.p12.genome.fa"

#in_file = "mm10_primary_assembly_and_lncRNA.gtf"
#myFasta = "GRCm38.p6.genome.fa"


in_file = "Mus_musculus.GRCm38.101.gtf"
myFasta = baseDir + "/" + "Mus_musculus.GRCm38.dna.toplevel.fa"

dbFile = baseDir + "/" + in_file + ".db"

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
            aaWeight = molecular_weight(aaSeq, seq_type="protein")
            aaWeightMono = molecular_weight(aaSeq, seq_type="protein", monoisotopic=True)

            # protein_id	gene_symbol	mol_weight_kd	mol_weight
            print("{}\t{}\t{}\t{}\tnormal".format(geneName.upper(), geneName, aaWeight/1000, aaWeight), file=sys.stdout)
            print("{}\t{}\t{}\t{}\tmonoisotopic".format(geneName.upper(), geneName, aaWeightMono/1000, aaWeightMono), file=sys.stdout)
        except:
            print("error", file=sys.stderr)
            print(transcriptID, geneName, file=sys.stderr)
            print(seq, file=sys.stderr)
            print(aaSeq, file=sys.stderr)
            continue








        
    