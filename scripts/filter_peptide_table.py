import pandas as pd
import numpy as np

path = '/usr/local/hdd/rita/hiwi/Homo_sapiens/Homo_sapiens.GRCh38.105.gtf.tsv'
x = 1
ppm = 5

df_raw = pd.read_csv(path, sep='\t')
df_raw.columns = ['protein_id', 'gene_symbol', 'peptide_seq', 'mol_weight_kd', 'mol_weight', 'type']
#filter peptides present in more than x genes
df = df_raw.drop_duplicates()
print("Found "+str(len(df_raw)-len(df))+" duplicates")
df_monoisotopic = df[df['type']=='monoisotopic']
df_grouped = df_monoisotopic.groupby(by='peptide_seq')
unique_peptides = df_grouped.nunique()[df_grouped['protein_id'].nunique() <= x].index
df_unique = df_monoisotopic[df_monoisotopic['peptide_seq'].isin(unique_peptides)]
print("Filtered " + str(len(df_monoisotopic.index)) + " peptides that are present in more than "+str(x)+" genes -> " + str(len(df_unique.index)) + ' -> ' +str(len(df_unique.index)/len(df_monoisotopic.index)))

#filter 2
print('#calculate ppm intervals')
df_monoisotopic["range-"] = df_monoisotopic["mol_weight"] - (df_monoisotopic["mol_weight"] * ppm)/10**6
df_monoisotopic["range+"] = df_monoisotopic["mol_weight"] + (df_monoisotopic["mol_weight"] * ppm)/10**6
print('#reduce data frame ')
df_reduced = df_monoisotopic[['protein_id', 'mol_weight','range-','range+']].copy()
print('#drop_duplicates to make the table even smaller. Save masses result into the same intervals (important to keep if different the protein ids!)')
df_reduced = df_reduced.drop_duplicates()
print('#found_left means  |------|========|-------| row')
#count all rows that have different protein_ids, and see sheme
df_reduced['found_left'] = df_reduced.apply(lambda row: ( (np.greater_equal(row['range+'], df_reduced['range+']) & (np.greater_equal(df_reduced['range+'], row['range-'])) & ~(row['protein_id']==df_reduced['protein_id']) )).sum(), axis=1)
print('#found_right means  row |------|========|-------|')
#count all rows that have different protein_ids, and see sheme
df_reduced['found_right'] = df_reduced.apply(lambda row: ((np.greater_equal(row['range+'], df_reduced['range-']) & (np.greater_equal(df_reduced['range-'], row['range-'])) & ~(row['protein_id']==df_reduced['protein_id']) )).sum(), axis=1)
print('#merge back so that we have peptides again')
df_merged_back = pd.merge(df_reduced, df_monoisotopic, on = ['mol_weight', 'protein_id', 'range-', 'range+'])
print('#filter out those peptides that were "found"')
filter = df_merged_back[(df_merged_back['found_left']==0) & (df_merged_back['found_right']==0)]
print('#final nice dataframe')
final = filter[['protein_id', 'gene_symbol', 'peptide_seq', 'mol_weight_kd', 'mol_weight', 'type']].copy()
final.to_csv('all_peptides'+str(ppm)+'ppm.tsv', sep='\t')
'''
df_sorted = df_unique.sort_values('mol_weight')
df_ppm = df_sorted.groupby(pd.cut(df_sorted["mol_weight"], np.arange(list(df_sorted['mol_weight'])[0], list(df_sorted['mol_weight'])[-1], rate)))
print("Grouped according to the mol_weight...")
df_final = df_ppm.filter(lambda x: x['protein_id'].nunique() == 1)

df_final.to_csv('monoisotopic_peptides'+str(rate)+'.tsv', sep='\t')
print("Filtered according to the ppm=" +str(rate)+" -> " + str(len(df_final.index)) + ' -> ' +str(len(df_final.index)/len(df_unique.index)))
df_to_original = df[df['peptide_seq'].isin(df_final['peptide_seq'])]
print("Writing final table...")
df_to_original.to_csv('all_peptides'+str(rate)+'.tsv', sep='\t')
'''