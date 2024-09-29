import requests
from ExpPreparer import load_process_expression
import pandas as pd

def get_mouse_gene_sequence(ensembl_gene_id):
    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{ensembl_gene_id}?species=mus_musculus&content-type=text/x-fasta"
    response = requests.get(server + ext, headers={"Content-Type": "text/x-fasta"})
    if not response.ok:
        response.raise_for_status()
        return None
    return response.text
# Example Ensembl Gene ID for mouse (replace with your gene ID)
ensembl_gene_ids = load_process_expression("/home/arvin/PycharmProjects/ExpFromSeq/single_nucleus_centroids.csv"
                                           ,"").T.columns.values
# print(ensembl_gene_ids)
id_to_seq = {}
# Get gene sequence for GRCm39 mouse genome
for id in ensembl_gene_ids:

    print(id)
    try:
        id_to_seq[id] = "".join(get_mouse_gene_sequence(id).split("\n")[1:])
    except:
        continue
    print(id_to_seq[id][:30])
    print()
df = pd.DataFrame(list(id_to_seq.items()), columns=['ID', 'Seq'])
print(df)
df.to_csv('processed_data', sep='\t')