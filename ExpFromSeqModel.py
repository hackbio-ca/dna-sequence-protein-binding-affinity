import time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from ExpPreparer import load_process_expression
import pickle
import numpy as np

# Convert gene sequences to k-mers
def kmerize(sequence, k=6):
    return ' '.join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])

def train_model():
    # expression_df = load_process_expression("/home/arvin/PycharmProjects/ExpFromSeq/single_nucleus_centroids.csv","")
    expression_df = load_process_expression("./single_nucleus_centroids.csv"
                                            ,"")
    mask = np.ones(len(expression_df), dtype=bool)
    mask[4000:4200] = False
    expression_df = expression_df[mask]

    sequences_df = pd.read_csv("./gene_sequences_final.csv")

    sequences_df = sequences_df[:8000]
    expression_df = expression_df[:8000]

    # genes_to_keep = set(expression_df.index.to_list())
    # indices_to_keep = [expression_df.index[i] == sequences_df.iloc[i, 0]
    #             for i in range(len(expression_df))]
    # sequences_df = sequences_df[indices_to_keep]
    # expression_df = expression_df[indices_to_keep]

    # break_value = 0
    # for idx, gene in enumerate(sequences_df.iloc[:, 0].values[:3000]):
    #     if not gene == expression_df.index.values[idx]:
    #         print(idx)
    #         break_value = idx
    #         print(f"{sequences_df.iloc[break_value - 1, 0]}      {expression_df.index.values[break_value - 1]}")
    #         print(f"{sequences_df.iloc[break_value, 0]}      {expression_df.index.values[break_value]}")
    #         print(f"{sequences_df.iloc[break_value + 1, 0]}      {expression_df.index.values[break_value + 1]}")
    #         print(f"{sequences_df.iloc[break_value + 2, 0]}      {expression_df.index.values[break_value + 2]}")
    #         if sequences_df.iloc[break_value, 0] == expression_df.index.values[break_value + 1]:
    #             expression_df = expression_df.drop(expression_df.index.values[break_value])
    #         elif sequences_df.iloc[break_value + 1, 0] == expression_df.index.values[break_value]:
    #             sequences_df = sequences_df.drop(break_value)
    #

    genes_df1 = set(expression_df.index)
    genes_df2 = set(sequences_df['Gene'])

    common_genes = genes_df1.intersection(genes_df2)

    expression_df = expression_df.loc[expression_df.index.isin(common_genes)]
    sequences_df = sequences_df[sequences_df['Gene'].isin(common_genes)]

    print(expression_df)
    print(sequences_df)


    # for idx, gene in enumerate(sequences_df.iloc[:, 0].values):
    #     if not gene == expression_df.index.values[idx]:
    #         print(idx)
    #         break_value = idx
    #         break

    sequences = sequences_df.iloc[:, 1].to_list()
    expression_averages = expression_df

    print("Data Loaded and Matched")

    # # Feature: gene sequences, Target: expression averages
    # sequences = data['Sequence']
    # expression_averages = data.drop(['Gene', 'Sequence'], axis=1)

    # k = 6
    # sequences_kmers = sequences.apply(lambda x: kmerize(x, k))
    #
    # # Vectorize the k-mers
    # vectorizer = CountVectorizer()
    # X = vectorizer.fit_transform(sequences_kmers)
    # y = expression_averages.values

    time1 = time.time()

    # Define k
    k = 6
    # Parallelize the kmerization process using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        sequences_kmers = list(executor.map(kmerize, sequences))
    # Vectorize the k-mers
    vectorizer = CountVectorizer(max_features=10000, min_df=5, max_df=0.8)
    # vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sequences_kmers)
    # Target values
    y = expression_averages.values

    vec_file = 'vectorizer.pkl'
    pickle.dump(vectorizer, open(vec_file, 'wb'))

    print("Kmers Vectorized")
    print(f"Took: {time.time() - time1}s")
    print()

    # print(X)
    # print()
    # print()
    # print(y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    print("Data Split. Training....")
    print()

    time1 = time.time()

    model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=32)
    model.fit(X_train, y_train)

    print(f"Took: {time.time() - time1}s")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_pretrained_model(path):
    with open(path, 'rb') as f:
        loaded_model = pickle.load(f)

    return loaded_model

def predict_expression(model, vectorizer, new_sequence="ATGCGTAGCTACGTGATCGTGTAC", k=6):
    new_kmers = kmerize(new_sequence, k)
    new_X = vectorizer.transform([new_kmers])
    predicted_expression = model.predict(new_X)

    return predicted_expression[0]

def pull_unused_data(expression_df):

    mask = np.zeros(len(expression_df), dtype=bool)
    mask[4000:4200] = True
    expression_df = expression_df[mask]

    return expression_df

def load_vectorizer(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def permute_rows(df):
    df_permuted = df.copy()
    n_rows = len(df_permuted)
    permuted_index = np.random.permutation(n_rows)
    df_permuted = df_permuted.iloc[permuted_index]

    return df_permuted

if __name__ == "__main__":
    train_model()
