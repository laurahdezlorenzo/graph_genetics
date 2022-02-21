#!/bin/bash

echo 'LOAD Phenotype'
echo '--------------------------------------------------------------------------'
echo
python ppa-graphs/ML_models/machine_learning_models.py AD PPI /home/laura/Documents/DATASETS/table_datasets Phenotype
python ppa-graphs/ML_models/machine_learning_models.py ND PPI /home/laura/Documents/DATASETS/table_datasets Phenotype
echo ''