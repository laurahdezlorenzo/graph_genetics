'''
Create datasets for Spektral.
'''
# import torch
import networkx as nx
import numpy as np
import pickle
from spektral.data import Dataset, Graph
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):

    '''
    ADNI genetics cohort dataset. Graphs are different PPI neworks. Node attri-
    butes are the number of missense or different types of mutations found in 
    each protein.
    '''

    def __init__(self, infile, **kwargs):
        # Store some custom parameters of the dataset
        self.infile = infile

        super().__init__(**kwargs)

    def read(self):
        # Return a list of Spektral Graph objects
        output = []
        with open(self.infile, 'rb') as file:
                graphs = pickle.load(file)

        for g in graphs:
            
            # Make numpy array with node features
            x = []
            nodes_list = g.nodes(data=True)
            for node in nodes_list:
                node_attr = node[1]['node_feature']
                node_attr = node_attr.numpy()
                x.append(node_attr)
            
            x = np.array(x)
            
            # Obtain adjacency matrix 
            a = nx.to_numpy_array(g)

            # Make numpy array with graph features
            s = g.graph['sampleID']
            y = g.graph['graph_label'].numpy().astype(np.float64)

            # Create Spektral's Graph object
            graph = Graph(x=x, a=a, y=y)
            output.append(graph)

        
        return output#, sample_ids

def stratified_split(dataset, target, mode):

    print()
    print('Train-Val-Test splits')
    print('=============================================================')

    # da_counter = 0
    y_all = []
    for g in dataset:
        y_all.append(g.y)
        # if g.y == 1:
        #     da_counter += 1

    # Create train (80%) and val-test (20%) datasets from original dataset
    tr_idx, tmp_idx = train_test_split(np.arange(len(dataset)),
                                            stratify=y_all,
                                            test_size=0.2,
                                            random_state=1)

    tr = dataset[tr_idx]
    # tr_counter = 0
    # for g in tr:
    #     if g.y == 1:
    #         tr_counter +=1

    tmp = dataset[tmp_idx]
    y_tmp = []
    for g in tmp:
        y_tmp.append(g.y)

    # Create validation (50%) and test (50%) datasets from tmp_dataset
    te_idx, va_idx = train_test_split(tmp_idx,
                                            stratify=y_tmp,
                                            test_size=0.5,
                                            random_state=1)

    va = dataset[va_idx]
    te = dataset[te_idx]

    # TODO: Arreglar esta cosa!!
    # va_counter = 0
    # for g in va:
    #     if g.y == [[0. 0. 1.]]:
    #         va_counter +=1

    # te_counter = 0
    # for g in te:
    #     if g.y == 1:
    #         te_counter +=1

    # print()
    # print(f'Positives in da: {da_counter} out of {len(dataset)} ({ da_counter/len(dataset)*100 }%)' )  
    # print(f'Positives in tr: {tr_counter} out of {len(tr_idx)} ({ tr_counter/len(tr_idx)*100 }%)' )
    # print(f'Positives in te:  {te_counter} out of {len(te_idx)}  ({te_counter/len(te_idx)*100}%)')
    # print(f'Positives in va:  {va_counter} out of {len(va_idx)}  ({va_counter/len(va_idx)*100}%)')

    split_dict = {}

    if mode == 'ML': # for machine learning models with sk-learn
        split_dict['train'] = list(tr_idx) + list(va_idx)
        split_dict['test'] = list(te_idx)

<<<<<<< HEAD
        #f = open(f'ppa-graphs/Spektral_models/split_{target}.pkl', 'wb')
        #pickle.dump(split_dict, f)
        #f.close()
=======
        # f = open(f'ppa-graphs/Spektral_models/split_{target}.pkl', 'wb')
        # pickle.dump(split_dict, f)
        # f.close()
>>>>>>> 2a0bc69f902268accd117eb85efd0644cf4ecaf3
    
    if mode == 'GG': # for GNN models with GraphGym
        split_dict['train'] = list(tr_idx)
        split_dict['valid'] = list(va_idx)
        split_dict['test'] = list(te_idx)

<<<<<<< HEAD
        #f = open(f'ppa-graphs/Spektral_models/split_{target}_GG.pkl', 'wb')
        #pickle.dump(split_dict, f)
        #f.close()
=======
        # f = open(f'ppa-graphs/Spektral_models/split_{target}_GG.pkl', 'wb')
        # pickle.dump(split_dict, f)
        # f.close()
>>>>>>> 2a0bc69f902268accd117eb85efd0644cf4ecaf3

    return tr, va, te


if __name__ == '__main__':

    target = 'DX'

    dataset = MyDataset(f'/home/laura/Documents/DATASETS/graph_datasets/{target}/AD_PPI_variants.pkl')
    print(dataset.n_labels)
