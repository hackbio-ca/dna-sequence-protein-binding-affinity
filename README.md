# RNA-expression-from-sequences

## 🥇 Winner of the First Place award!

Design a model to predict RNA expression for mice brains from random RNA sequences using ENSEMBL GRCm39 mouse brain genome.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Abstract

Design a model that produces the closest mouse brain-cell type match given an RNA sequence given a random RNA sequence from the mouse brain genome GRCm39. Submitted as part of the [Toronto Bioinformatics Hackathon 2024](https://hackbio.ca/)


## Background
mRNA is the most crucial aspect of any protein synthesis - after being transcribed from DNA with a myriad of enzymes, the same RNA sequence can code for multiple proteins. Given that one RNA code for many proteins, how do we know which one to select? Our project takes a computational approach towards predicting the closest cell type match given an RNA sequence. While the selection of RNA to produce a protein happens with a mixture of multiple enzymes that involve codon usage (i.e tRNA adaptation, mRNA secondary structure, untranslated regions (UTRs), etc.) under physiological conditions, we utilize AI to perform these predictions on a computation level. Using the randomForestRegressor model in Scikit Learn, our program trains the model on single-nucleus RNA sequencing data in mice brains ([Langlieb _et al._, 2023](https://www.google.com/url?q=https://pubmed.ncbi.nlm.nih.gov/36945580/&sa=D&source=docs&ust=1727629784546695&usg=AOvVaw3Kkx8Eat2nMI4SE3Nr84OU)), and uses this data to predict the level of expression of a random input RNA in each cell type. 

## Workflow

**Phase 1: Raw Data Processing**

The raw data collected was organized as follows:
- Columns: specific mRNA gene (expressed as an ENSEMBL code from the GRCm39 mouse brain genome (citation))
- Rows: percentage of expression of mRNA gene in a specific cell type (e.g: Ex_Lhx2_Col5a2_1 ) [total data used was about 5030 rows * 20984 columns]

The data received was initially stored in a Pandas Dataframe, then transposed to a dictionary with two columns - ‘Gene’ and ‘Sequence’, where column ‘Gene’ corresponds to all the specific ENSEMBL genes and ‘Sequence’ refers to the corresponding mRNA sequence (this was found by querying rest.ensembl.org. The data was stored in a .csv file

**Phase 2: Model Training**

Using the final data stored in the .csv, the randomForestRegressor model in Scikit learn was trained, producing a vectorizer and the trained model was stored. Any subsequent entries to the program would be using this trained model to make predictions.

** Phase 3: Evaluating Predictions**

We developed a front-end terminal interface for users to interact with the model - when entered a random RNA sequence, the query would be sent to the trained model and an output of the predicted percentage expression in each cell type was received. To evaluate the accuracy of the model, we conducted a correlation between the predicted values and the actual values produced by the model.

Check out our [Devpost](https://devpost.com/submit-to/22491-toronto-bioinformatics-hackathon/manage/submissions/556398-predicting-mrna-expression-from-mrna-sequences/project_details/edit) link for more details on our submission for the Toronto Bioinformatics Hackathon 2024. 


## Installation

Download all project files in a directory.

## Quick Start

Run Front_End.py, enter mRNA sequence into console and wait for the output! 

```python
>>> Please enter the mRNA sequence to analyze: "ACGTAGCTAGCTGATCGTAGCTGTGCTATGTCGTGTCGATCGTAC"

                   ENSMUSG00000051951  ... ENSMUSG00000075046
Ex_Lhx2_Col5a2_1             9.725944  ...                0.0
Inh_Dmrt3_Prdm6_2            9.033217  ...                0.0
Inh_Pax2_Plscr2_1           10.158686  ...                0.0
Inh_Tfap2b_Slc22a7           9.883856  ...                0.0
Chol_Isl2_Gata6_1           10.036545  ...                0.0
...                               ...  ...                ...
Inh_Six6_Vipr2_9             9.372059  ...                0.0
Inh_Six6_Vipr2_8              6.04083  ...                0.0
Inh_Six6_Vipr2_4             4.973628  ...                0.0
Inh_Six6_Vipr2_1             5.822238  ...                0.0
Inh_Six6_Vipr2_3              7.43262  ...                0.0

Mean correlation between predicted and actual values of model: 20
```


## Contribute

Contributions are welcome! If you'd like to contribute, please open an issue or submit a pull request. See the [contribution guidelines](CONTRIBUTING.md) for more information.

## Support

If you have any issues or need help, please open an [issue](https://github.com/hackbio-ca/dna-sequence-protein-binding-affinity/issues) or contact the project maintainers.

## License

This project is licensed under the [MIT License](LICENSE).
