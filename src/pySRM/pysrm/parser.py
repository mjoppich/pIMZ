import pandas as pd
import numpy as np

def extract_diff_masses():

    weights_data = pd.read_csv("/usr/local/hdd/rita/hiwi/pyIMS/src/pySRM/pysrm/protein_weights.tsv", sep="\t")
    panglao_data = pd.read_csv("/usr/local/hdd/rita/hiwi/pyIMS/src/pySRM/pysrm/panglao.tsv", sep="\t")

    weights_data = weights_data[weights_data['gene_symbol'].notna()]

    weights_data_reduced = pd.DataFrame(columns=['gene_symbol', 'mol_weight'])
    for index, row in weights_data.iterrows():
        if not ";" in row['gene_symbol']:
            weights_data_reduced = weights_data_reduced.append({'gene_symbol': row['gene_symbol'], 'mol_weight': row['mol_weight']}, ignore_index=True)
        else:
            symbols = row['gene_symbol']
            symbols = symbols.split(";")
            for i in range(len(symbols)):
                weights_data_reduced = weights_data_reduced.append({'gene_symbol': symbols[i], 'mol_weight': row['mol_weight']}, ignore_index=True)

    panglao_data = panglao_data.rename(columns={"official gene symbol": "gene_symbol"})

    panglao_data['gene_symbol'] = panglao_data['gene_symbol'].str.upper() 
    weights_data_reduced['gene_symbol'] = weights_data_reduced['gene_symbol'].str.upper() 

    result = pd.merge(weights_data_reduced, panglao_data, on='gene_symbol') #Inner join be default

    mz_diff = result["mol_weight"].array
    mz_diff = np.array(mz_diff)
    #for me
    #result.to_csv("test.tsv", sep='\t')
    return np.unique(mz_diff)

if __name__ == "__main__":
    # execute only if run as a script
    extract_diff_masses()