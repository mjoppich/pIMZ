import pandas as pd
import numpy as np
from intervaltree import IntervalTree
import time

t0 = time.time()
path = '/usr/local/hdd/rita/hiwi/Homo_sapiens/Homo_sapiens.GRCh38.105.gtf.tsv'
x = 1
ppm = 5

def add(x, y):
    return x + y

df_raw = pd.read_csv(path, sep='\t')
df_raw.columns = ['protein_id', 'gene_symbol', 'peptide_seq', 'mol_weight_kd', 'mol_weight', 'type']

df = df_raw.drop_duplicates()
print("Found "+str(len(df_raw)-len(df))+" duplicates")
df_monoisotopic = df[df['type']=='monoisotopic']
df_monoisotopic = df_monoisotopic[df_monoisotopic['peptide_seq'].str.len()>5]

print('#calculate ppm intervals')
df_monoisotopic.loc[:,"range-"] = df_monoisotopic["mol_weight"] - (df_monoisotopic["mol_weight"] * ppm)/10**6
df_monoisotopic.loc[:,"range+"] = df_monoisotopic["mol_weight"] + (df_monoisotopic["mol_weight"] * ppm)/10**6
print('#reduce data frame ')
df_reduced = df_monoisotopic[['protein_id','range-','range+']].copy()
print('#drop_duplicates to make the table even smaller. Save masses result into the same intervals (important to keep if different the protein ids!)')
df_reduced = df_reduced.drop_duplicates()

range_minus = list(df_reduced['range-'])
range_plus = list(df_reduced['range+'])
data = list(df_reduced['protein_id'])

tree = IntervalTree()
print('Built a tree')
for i in range(len(data)):
    tree.addi(range_minus[i], range_plus[i], [data[i]])

print('Added '+str(len(data))+' elements, #nodes='+str(len(tree)))

tree.merge_equals(data_reducer=add)

#tree.merge_overlaps(data_reducer=add)
print('Delete equals, #nodes='+str(len(tree)))

df_reduced['hits'] = df_reduced.apply(lambda row: np.unique(np.concatenate([ [x_ for x_ in x[2] if not x_==row['protein_id']] for x in tree.overlap(row['range-'], row['range+'])]).ravel()), axis=1)
df_reduced['found'] = df_reduced.apply(lambda row: len(row['hits']), axis=1)
#df_reduced['found'] = df_reduced.apply(lambda row: sum([len(x[2])-x[2].count(row['protein_id']) for x in tree.overlap(row['range-'], row['range+'])]), axis=1)
#df_reduced['found'] = df_reduced.apply(lambda row: len([x for x in tree.overlap(row['range-'], row['range+']) if not row['protein_id'] in x[2]]), axis=1)
#df_reduced['found'] = df_reduced.apply(lambda row: len(np.unique(tree.overlap(row['range-'], row['range+']).pop()[2])), axis=1)
df_merged_back = pd.merge(df_reduced, df_monoisotopic, on = ['protein_id', 'range-', 'range+'])
print('#filter out those peptides that were "found"')
filter = df_merged_back[df_merged_back['found']==0]
print('#final nice dataframe')
final = filter[['protein_id', 'gene_symbol', 'peptide_seq', 'mol_weight_kd', 'mol_weight', 'type']].copy()
final.to_csv('peptides'+str(ppm)+'ppm.tsv', sep='\t')
#df_merged_back.to_csv('all_peptides+hits_'+str(ppm)+'ppm.tsv', sep='\t')
t1 = time.time()-t0
print(str(t1)+" time")
