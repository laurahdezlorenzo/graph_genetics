#!/bin/bash

echo `date`
echo ''

echo 'AD PPI'
echo '--------------------------------------------------------------------------'
echo
for i in {1..101}; do
    python ppa-graphs/randomize_network.py AD PPI $i
done
echo ''


echo 'ND PPI'
echo '--------------------------------------------------------------------------'
echo
for i in {1..101}; do
    python ppa-graphs/randomize_network.py ND PPI $i
done
echo ''

echo 'Finish!'
echo `date`
echo ''