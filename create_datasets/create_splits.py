# import torch
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle as pkl

def create_folds_stratified_cv(target, n):

    print(target)

    with open(f'data/graph_datasets/{target}/AD_PPI_string_missense.pkl', 'rb') as f: # it doesn't matter the network we use because samples are the same
        list_of_dicts = pkl.load(f)
    
    graphs = []
    for i in list_of_dicts:
        graph = nx.Graph(i)
        sample = graph.graph['sampleID']
        label  = graph.graph['graph_label'].item()
        graphs.append([sample, label])

    
    df = pd.DataFrame(graphs, columns=['sample', 'y'])
    df['trick'] = df['y']
    df.set_index('sample', inplace=True)
    y = df['y']

    skf = StratifiedKFold(n_splits=n)
    k = 1
    for train_idx, test_idx in skf.split(df, y):
        print(f'Fold -  {k}   |   train -  {np.bincount(y[train_idx])}   |   test -  {np.bincount(y[test_idx])}')

        train_idx_samples = df.iloc[train_idx].index.values
        test_idx_samples = df.iloc[test_idx].index.values

        split_dict = {}
        split_dict['train'] = list(train_idx_samples)
        split_dict['valid'] = list(test_idx_samples)

        f = open(f'data/splits/{n}Fold_CV_{target}/k{k}_{target}.pkl', 'wb')
        pkl.dump(split_dict, f)
        f.close()

        k += 1
    
    print()


if __name__ == '__main__':

    create_folds_stratified_cv('PET', 10)
    create_folds_stratified_cv('PETandDX', 10)
    # create_folds_stratified_cv('LOAD', 10)