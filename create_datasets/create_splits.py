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

          # 22/04/2022 - This part is only to search for differences among splits        
#         val_samples = list(df.iloc[test_idx].index)
#         data = pd.read_csv('data/ADNI/ADNIMERGE_metadata.csv', index_col=0)
#         tmp = data.loc[val_samples]
#         tmp.to_csv(f'data/splits/10Fold_CV_{target}/fold{k}_val_samples.csv')


        split_dict = {}
        split_dict['train'] = list(train_idx)
        split_dict['valid'] = list(test_idx)

        # print(split_dict['valid'])

        f = open(f'data/splits/{n}Fold_CV_{target}/k{k}_{target}.pkl', 'wb')
        pkl.dump(split_dict, f)
        f.close()

        k += 1
    
    print()


if __name__ == '__main__':

    # create_folds_stratified_cv('PET', 10)
    create_folds_stratified_cv('PETandDX', 10)
    # create_folds_stratified_cv('LOAD', 10)