import pandas as pd
import scanpy as sc
import numpy as np

np.set_printoptions(edgeitems=10)
sc.settings.n_jobs = 32

def load_process_expression (exp_path, lookup_path):
    # Load the reference data (already annotated)
    # ref_centroids_unprocessed = pd.read_csv("/inkwell05/arvin/Hongqui_Centroids/single_nucleus_centroids.csv")
    ref_centroids_unprocessed = pd.read_csv(exp_path)

    ref_centroids_unprocessed = ref_centroids_unprocessed.T
    # print(ref_centroids_unprocessed)
    # print(ref_centroids_unprocessed.index.to_list()[1:])
    # print(ref_centroids_unprocessed.iloc[0].to_list())
    # print(ref_centroids_unprocessed.values[1:])
    ref_centroids = pd.DataFrame(index=ref_centroids_unprocessed.index.to_list()[1:],
                                 columns=ref_centroids_unprocessed.iloc[0].to_list(),
                                 data=ref_centroids_unprocessed.values[1:])


    # lookup_df = pd.read_csv("/inkwell05/arvin/Hongqui_Centroids/mouse_symbol_to_ENSEMBL_ID.csv", header=None)
    # lookup_df = pd.read_csv(lookup_path, header=None)
    # ref_centroids = update_ref_centroids(ref_centroids, lookup_df)

    return ref_centroids.T

if __name__ == "__main__":
    load_process_expression("/inkwell05/arvin/Hongqui_Centroids/single_nucleus_centroids.csv",
                            "/inkwell05/arvin/Hongqui_Centroids/mouse_symbol_to_ENSEMBL_ID.csv")
