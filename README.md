# graph_genetics
<div id="top"></div>

## Description

This is the code repository for the paper entitled *Benchmarking Graph Neural Networks for phenotype prediction through genotype in Alzheimer’s Disease*. The repository follows the methodology and results presented in the abovementioned work. 

![Image](figures/figure1a.png)
![Image](figures/figure1b.png)

The results obtained for the manuscript are organized in the following notebooks:

* [1_ADNI_GNNs_networks](1_ADNI_GNNs_networks.ipynb) - for Section "3.1. Comparing results using different input networks"
* [2_ADNI_GNNs_vs_nonGNNs](2_ADNI_GNNs_vs_nonGNNs.ipynb)  - for Section "3.2. Benchmarking GNNs performance vs. other non-GNN models"
* [3_ADNI_GNNs_random_networks](3_ADNI_GNNs_random_networks.ipynb)  - for Section "3.3. Using random networks as input"
* [4_LOAD_GNNs](4_LOAD_GNNs.ipynb)  - for Section "3.4. Using randomized networks as input"


These notebooks use information from several scripts, organized as follows:

* [data_preprocessing](data_preprocessing) contains an R script for obtaining Gene-Disease Associations from DisGeNET and several scripts for obtaining genetic data from the different cohorts employed.
* [networks](networks) contains several Python scripts for obtaining biological networks from different sources and build random networks.
* [create_datasets](create_datasets) Python scripts for building different datasets for supervised classification models.
  * [create_nx_datasets.py](create_datasets/create_nx_datasets.py) is the one for building graph-datasets for Graph Neural Networks (GNNs)
* [ml_models](ml_models) different functions for using with other non-GNN models.

Other subdirectories present in this repository:

* [data](data) contains several data files used in this work. Please note genetic data coming from the cohorts employed is not available due to privacy reasons.
* [results](results) CSV files with the results presented in this work.
* [figures](figures)

## Implementation

The code in this work was built using:

* [disgenet2r](https://www.disgenet.org/disgenet2r) for obtaining GDAs from DisGeNET.
* [biomaRt](https://bioconductor.org/packages/release/bioc/html/biomaRt.html) for obtaining genomic coordinates of the genes of interest.
* [VCFTools](http://vcftools.sourceforge.net/) and [Ensemble's Variant Effect Predictor (VEP)](https://www.ensembl.org/info/docs/tools/vep/index.html) for extracting and annotating missense variants.
* [NetworkX](https://networkx.org/) for networks' manipulation and building graph datasets.
* [GraphGym](https://github.com/snap-stanford/GraphGym) for evaluating and testing GNN models on graph datasets.
* [Scikit-Learn](https://scikit-learn.org/stable/) for building non-GNN models.
* [SciPy](https://scipy.org/) for statistical analyses.

## Contact
Please refer any questions to:
Laura Hernández-Lorenzo - [GitHub](https://github.com/laurahdezlorenzo) - [email](laurahl@ucm.es)
