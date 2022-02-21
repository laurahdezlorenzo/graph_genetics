#!/bin/bash

echo `date`
echo ''

echo 'PET CLASS AD NETWORK'
echo '--------------------------------------------------------------------------'

echo "Creating BioGRID dataset ..."
python3 ppa-graphs/create_nx_datasets.py AD PPI biogrid PET
echo ''

echo "Creating HuRI dataset ..."
python3 ppa-graphs/create_nx_datasets.py AD PPI huri PET
echo ''

echo "Creating SNAP-Brain dataset ..."
python3 ppa-graphs/create_nx_datasets.py AD PPI snap_brain PET
echo ''

echo "Creating GIANT-Brain dataset ..."
python3 ppa-graphs/create_nx_datasets.py AD PPI giant_brain PET
echo ''


echo 'PET&DX CLASS AD NETWORK'
echo '--------------------------------------------------------------------------'

echo "Creating BioGRID dataset ..."
python3 ppa-graphs/create_nx_datasets.py AD PPI biogrid PETandDX
echo ''

echo "Creating HuRI dataset ..."
python3 ppa-graphs/create_nx_datasets.py AD PPI huri PETandDX
echo ''

echo "Creating SNAP-Brain dataset ..."
python3 ppa-graphs/create_nx_datasets.py AD PPI snap_brain PETandDX
echo ''

echo "Creating GIANT-Brain dataset ..."
python3 ppa-graphs/create_nx_datasets.py AD PPI giant_brain PETandDX
echo ''
