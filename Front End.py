import scipy
from ExpPreparer import load_process_expression
from ExpFromSeqModel import load_pretrained_model
from ExpFromSeqModel import predict_expression
from ExpFromSeqModel import pull_unused_data
from ExpFromSeqModel import load_vectorizer
from ExpFromSeqModel import permute_rows
import pandas as pd

def compare_ind_transcriptions(generated: list[float], real: list[float]):

    # Implemented with RMSE for now

    residual_sum = 0

    for i in range(len(real)) :
        residual_sum += (real[i] - generated[i])^2

    return (residual_sum / len(real))**0.5

def correlate_gene_transcription(generated: list[float], real: list[float]):

    # print(len(generated))
    # print(len(real))

    return scipy.stats.pearsonr(generated, real)[0]


if __name__ == '__main__':

    #Load reference centroids, model and do overall analysis between them

    ref_centroids = load_process_expression("./single_nucleus_centroids.csv", "")


    unused = pull_unused_data(ref_centroids)
    vectorizer = load_vectorizer("./vectorizer.pkl")
    model = load_pretrained_model("./random_forest_model.pkl")
    sequences_df = pd.read_csv("./gene_sequences_final.csv")

    # Get the full average correlation between with the real data
    total_rs = 0
    for i in unused.index.to_list():
        real_vector = unused.loc[i].values
        transcript = sequences_df.loc[sequences_df.iloc[:, 0] == i, sequences_df.columns[1]].values[0]
        predicted = predict_expression(model, vectorizer, transcript)

        total_rs += correlate_gene_transcription(real_vector, predicted)

    average = total_rs / len(unused)

    print(f"Total R value average for data set: {average}")

    print("FULL MODEL ANALYSES")

    # # Pass the same thing to the random permutation one to get biological significance
    # unused = permute_rows(unused)
    #
    # total_rs = 0
    # for i in unused.index.to_list():
    #     real_vector = unused.loc[i].values
    #     transcript = sequences_df.loc[sequences_df.iloc[:, 0] == i, sequences_df.columns[1]].values[0]
    #     predicted = predict_expression(model, vectorizer, transcript)
    #
    #     total_rs += correlate_gene_transcription(real_vector, predicted)
    #
    # average = total_rs / len(unused)
    #
    # print(f"Total R value average for permuted data set: {average}")

    print("INDIVIDUAL GENE ANALYSIS")

    ended = False
    while not ended:
        input_sequence = input("Please enter the mRNA sequence to analyze:")

        # Do the analysis with the input sequence to return the list of transcription values

        predicted = predict_expression(model, vectorizer, input_sequence)

        for idx, cell_type in enumerate(ref_centroids.index.to_list()):
            try:
                print(f"{cell_type}     {predicted[idx]}")
            except:
                break
        # generate real values to compare with



        restart = input("Would you like to analyze another mRNA sequence? Type \'Y\' to confirm and anything else to deny.")

        if restart != "Y":
            ended = True


    print("Thanks for using the program.")
