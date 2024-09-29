import pandas as pd
import scanpy as sc
import numpy as np
from anndata import read_csv

np.set_printoptions(edgeitems=10)
sc.settings.n_jobs = 32

def loader(path):
    df = pd.read_csv(path)

    # print(df.values)

    data = []

    for row in df.values:
        row_split = row[0].split('\t')
        gene = row_split[1]
        sequence = row_split[2]

        print(f'"{gene}","{sequence}"')
        data.append([gene, sequence])

    final_df = pd.DataFrame(data, columns=["Gene", "Sequence"])
    final_df.to_csv("gene_sequences_final.csv", index=False)

if __name__ == "__main__":
    loader("/home/arvin/PycharmProjects/ExpFromSeq/processed_data.csv")
