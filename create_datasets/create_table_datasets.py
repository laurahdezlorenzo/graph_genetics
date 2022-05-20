'''
Description
'''
import os, io, datetime
import numpy as np
import pandas as pd
import networkx as nx

def adni_data(mode, best_network, data_file, metadata_file):

    data = pd.read_csv(data_file, index_col=0)
    metadata = pd.read_csv(metadata_file, index_col=0)

    G = nx.read_edgelist(best_network)
    nodes = G.nodes()

    data = data.T

    data = data[data.columns.intersection(nodes)]
    # print(data)

    # For missense
    if mode == 'missense':
        result = pd.concat([data, metadata], axis=1)
        return result

def load_data(mode, disease, data_file, metadata_file):

    data = pd.read_csv(data_file, index_col=0)
    metadata = pd.read_csv(metadata_file, index_col='IID', sep='\t')

    data = data.T

    # For missense
    if mode == 'missense':
        result = pd.concat([data, metadata], axis=1)
        return result

    return result
