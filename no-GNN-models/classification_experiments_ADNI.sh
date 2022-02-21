#!/bin/bash

# echo 'PET'
echo '--------------------------------------------------------------------------'
echo
python no-GNN-models/machine_learning_models.py AD PPI data PET
python no-GNN-models/machine_learning_models.py ND PPI data PET
echo ''

echo 'PET&DX'
echo '--------------------------------------------------------------------------'
echo
python no-GNN-models/machine_learning_models.py AD PPI data/table_datasets PETandDX
python no-GNN-models/machine_learning_models.py ND PPI data/table_datasets PETandDX
echo ''


echo 'PET'
echo '--------------------------------------------------------------------------'
echo
python no-GNN-models/machine_learning_models.py AD PPI data/table_datasets/no_APOE PET
python no-GNN-models/machine_learning_models.py ND PPI data/table_datasets/no_APOE PET
echo ''

echo 'PET&DX'
echo '--------------------------------------------------------------------------'
echo
python no-GNN-models/machine_learning_models.py AD PPI data/table_datasets/no_APOE PETandDX
python no-GNN-models/machine_learning_models.py ND PPI data/table_datasets/no_APOE PETandDX
echo ''