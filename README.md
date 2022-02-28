# graph-genetics
<div id="top"></div>

## Description

This is the code repository for the paper entitled "Graph classification for phenotype prediction in neurodegenerative diseases". The repository follows the methodology and results presented in the abovementioned work. 

![Image](figure1.png)

* [genes_of_interest](genes_of_interest) obtain Gene-Disease Associations from DisGeNET using an R script.
* [networks](networks) contains several Python scripts for building different networks (PPIs from different sources, random networks).
* [data_preprocessing](data_preprocessing) several scripts for obtaining genetic data from the different cohorts employed.
* [create_datasets](create_datasets) Python scripts for building different datasets for supervised classification models.
  * [create_nx_datasets.py](create_datasets/create_nx_datasets.py) is the one for building graph-datasets for Graph Neural Networks (GNNs)
* [ml_models](ml_models) Machine learning models for comparing with GNNs.
* [data](data) contains several data files used in this work. Please note genetic data coming from the cohorts employed is not available due to privacy reasons.
* [results](results) contains several files with the results presented in this work

## Implementation

The code in this work was built using:

* [disgenet2r](https://nextjs.org/) for obtaining GDAs from DisGeNET.
* [biomaRt](https://reactjs.org/) for obtaining genomic coordinates of the genes of interest.
* [VCFTools](https://vuejs.org/) and [Ensemble's Variant Effect Predictor (VEP)](https://angular.io/) for extracting and annotating missense variants.
* [NetworkX]() for networks' manipulation and building graph datasets.
* [GraphGym]() for evaluating and testing GNN models on graph datasets.
* [Scikit-Learn]() for building non-GNN models.
* [SciPy]() for statistical analyses.

## Contact
Please refer any questions to:
Laura Hernández-Lorenzo - [GitHub](https://github.com/laurahdezlorenzo) - [email](laurahl@ucm.es)
