#!/bin/bash

echo `date`
echo ''

echo 'AD PPI'
echo '--------------------------------------------------------------------------'
echo

for i in {1..100}; do
    echo "Creating random dataset #$i..."
    python ppa-graphs/create_nx_datasets.py AD PPI "rand$i" PET
done
echo ''


echo 'ND PPI'
echo '--------------------------------------------------------------------------'
echo

for i in {1..100}; do
    echo "Creating random dataset #$i..."
    python ppa-graphs/create_nx_datasets.py ND PPI "rand$i" PET
done
echo ''

echo 'AD PPI'
echo '--------------------------------------------------------------------------'
echo

for i in {1..100}; do
    echo "Creating random dataset #$i..."
    python ppa-graphs/create_nx_datasets.py AD PPI "rand$i" PETandDX
done
echo ''


echo 'ND PPI'
echo '--------------------------------------------------------------------------'
echo

for i in {1..100}; do
    echo "Creating random dataset #$i..."
    python ppa-graphs/create_nx_datasets.py ND PPI "rand$i" PETandDX
done
echo ''