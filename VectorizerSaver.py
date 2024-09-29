from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from ExpPreparer import load_process_expression
import pickle

# Convert gene sequences to k-mers
def kmerize(sequence, k=6):
    return ' '.join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])

def train_model():
    # expression_df = load_process_expression("/home/arvin/PycharmProjects/ExpFromSeq/single_nucleus_centroids.csv","")
    expression_df = load_process_expression("/home/arvin/PycharmProjects/ExpFromSeq/single_nucleus_centroids.csv"
                                            ,"")
    # mask = np.ones(len(expression_df), dtype=bool)
    # mask[4000:4200] = False
    # expression_df = expression_df[mask]
    sequences_df = pd.read_csv("/home/arvin/PycharmProjects/ExpFromSeq/gene_sequences_final.csv")

    sequences_df = sequences_df[:8000]
    expression_df = expression_df[:8000]

    genes_df1 = set(expression_df.index)
    genes_df2 = set(sequences_df['Gene'])

    common_genes = genes_df1.intersection(genes_df2)

    sequences_df = sequences_df[sequences_df['Gene'].isin(common_genes)]

    sequences = sequences_df.iloc[:, 1].to_list()

    # Parallelize the kmerization process using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        sequences_kmers = list(executor.map(kmerize, sequences))
    # Vectorize the k-mers
    # vectorizer = CountVectorizer(max_features=10000, min_df=5, max_df=0.8)
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(sequences_kmers)

    with open('/home/arvin/PycharmProjects/ExpFromSeq/rna_vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)