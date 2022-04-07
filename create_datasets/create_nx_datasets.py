'''
Description
'''
import os, io, datetime
import pickle
import torch
import numpy as np
import pandas as pd
import networkx as nx


def process_variants_adni(infile, header):

    '''
    Description.
    '''

    # Add header
    col_file = open(header, 'r')
    col_names = col_file.read().split('\n')
    col_file.close()

    [col.upper() for col in col_names]

    # Load file
    variants_matrix = pd.read_csv(infile, sep='\t', names=col_names, low_memory=False)

    # Delete non-useful columns
    variants_matrix.drop(variants_matrix.columns[9:46], axis=1, inplace=True)
    variants_matrix.drop(variants_matrix.columns[11:42], axis=1, inplace=True)
    variants_matrix.drop(columns=['Allele', 'IMPACT'], axis=1, inplace=True)

    # Replace genotypes with a numeric value (NaN: miss, 1: presence, 0:absence)
    variants_matrix.replace({'./.':np.NaN, '0/0':0}, inplace=True)
    variants_matrix.replace(['0/1', '1/0', '1/1'], 1, inplace=True)

    return variants_matrix

def process_variants_load(infile, header, disease):

    '''
    Description.
    '''

    # Add header
    col_file = open(header, 'r')
    col_names = col_file.read().split('\n')
    col_file.close()

    [col.upper() for col in col_names]

    # Load file
    variants_matrix = pd.read_csv(infile, sep='\t', names=col_names)
    print(variants_matrix.shape)


    if disease == 'AD':

        # Delete non-useful columns
        variants_matrix.drop(variants_matrix.columns[9:46], axis=1, inplace=True)
        variants_matrix.drop(variants_matrix.columns[11:42], axis=1, inplace=True)
        variants_matrix.drop(columns=['Allele', 'IMPACT'], axis=1, inplace=True)
    
    elif disease == 'ND':
        # Delete non-useful columns
        variants_matrix.drop(variants_matrix.columns[9:47], axis=1, inplace=True)
        variants_matrix.drop(variants_matrix.columns[11:37], axis=1, inplace=True)
        variants_matrix.drop(columns=['Allele', 'IMPACT'], axis=1, inplace=True)

    # Replace genotypes with a numeric value (NaN: miss, 1: presence, 0:absence)
    variants_matrix.replace({'./.':np.NaN, '0/0':0}, inplace=True)
    variants_matrix.replace(['0/1', '1/0', '1/1'], 1, inplace=True)

    print(variants_matrix.shape)

    return variants_matrix


def same_lists(mode, genes_variants, genes_graph):
    
    '''
    Make `genes_variants` dataframe and `genes_graph` list coincide. First, 
    deletes genes from `genes_variants` dataframe that are not in the list of 
    genes / proteins (nodes) of the PPI graph. Second, add as new rows (filled
    with zeroes) to `genes_variants` dataframe genes that are in the PPI.
    '''

    # First
    for gene in list(genes_variants.index):
        if not gene in genes_graph:
            genes_variants.drop(gene, inplace=True)

    # Second
    for node in genes_graph:
        if not node in genes_variants.index:
            if mode == 'variants':
                genes_variants.loc[node] = [[0]*19] * genes_variants.shape[1]
            else:
                genes_variants.loc[node] = [0] * genes_variants.shape[1]
    print(genes_variants.shape)
    return genes_variants

def per_node(mode, df_original, nodes):

    '''
    Compute several metrics per node in the PPI graph.
    '''

    df = df_original.drop(columns=['CHROM', 'POS', 'ID', 'REF', 'ALT'])
    
    if mode == 'missense': # count number of missense variants per node

        missense = df.drop(columns=['CONSEQUENCE'])
        missense_sum = missense.groupby('SYMBOL').sum()
        missense_sum = same_lists(mode, missense_sum, nodes)

        return missense_sum

def add_graph_features(g, s, label, labels_df):

    g.graph['sampleID'] = s
  
    if label == 'PET': # Dataset with PET+ vs PET- subjects
        av45 = labels_df.loc[s]['AV45']
        pib = labels_df.loc[s]['PIB']
        diagnosis = labels_df.loc[s]['DX']

        if av45 >= 1.11:
            g.graph['graph_label'] = torch.tensor([1])
            return g

        elif av45 < 1.11:
            g.graph['graph_label'] = torch.tensor([0])
            return g

        elif np.isnan(av45):
            if pib >= 1.27:# and diagnosis == 'Dementia':
                g.graph['graph_label'] = torch.tensor([1])
                # print('PIB+ Dem', s)
                return g

            elif pib < 1.27:
                g.graph['graph_label'] = torch.tensor([0])
                return g
    
    elif label == 'PETandDX': # Dataset with PET+ Dementia vs PET- CN subjects
        av45 = labels_df.loc[s]['AV45']
        pib = labels_df.loc[s]['PIB']
        diagnosis = labels_df.loc[s]['DX']

        if av45 >= 1.11 and diagnosis == 'Dementia':
            g.graph['graph_label'] = torch.tensor([1])
            return g

        elif av45 < 1.11 and diagnosis == 'CN':
            g.graph['graph_label'] = torch.tensor([0])
            return g

        elif np.isnan(av45):
            if pib >= 1.27 and diagnosis == 'Dementia':
                g.graph['graph_label'] = torch.tensor([1])
                return g

            elif pib < 1.27 and diagnosis == 'CN':
                g.graph['graph_label'] = torch.tensor([0])
                return g
    
    elif label == 'LOAD': # Dataset with subjects stratified by LOAD diagnosis
        if s != '':
            diag = labels_df.loc[s]['Phenotype']

            if diag == 1:
                g.graph['graph_label'] = torch.tensor([0])
                return g

            elif diag == 2:
                g.graph['graph_label'] = torch.tensor([1])
                return g

def create_samples_graphs(mode, nodes_matrix, edges_matrix, original_graph, diagnosis, target):

    '''
    Create a graph for each sample in the dataset, using nodes and edges 
    attributes obtained previously from genetic variants information.
    '''

    samples = iter(list(nodes_matrix.columns))
    nodes = list(original_graph)
    edges = original_graph.edges

    print('Creating samples graphs...')

    graphs_list = []
    counter = 0

    for sample in samples:

        sample_graph = original_graph.copy()

        # Add graph features
        sample_graph = add_graph_features(sample_graph, sample, target, diagnosis)

        if sample_graph == None:
            continue
        
        if sample_graph.graph['graph_label'] == torch.tensor([1]):
            counter += 1
        
        # Add node and edge features (depending on the mode)
        if mode == 'missense':
            for n in nodes:
                sample_graph.nodes[n]['node_attr'] = torch.tensor([nodes_matrix.loc[n][sample]]) # missense

        graphs_list.append(sample_graph)
        # print('Sample graph used:', '# nodes =', nx.number_of_nodes(sample_graph), '# edges =', nx.number_of_edges(sample_graph))

    print(f'Class: {target}. Found {counter} positive subjects out of {len(graphs_list)}')

    return graphs_list

def delete_small_components(graphs, thres_nodes):

    '''
    Delete components with less than 5 nodes
    '''

    for G in graphs:

        for component in list(nx.connected_components(G)):

            if len(component) <= thres_nodes:

                for node in component:
                    
                    G.remove_node(node)

    return graphs

def main(indir, dataset, target, disease, network, mode, number):


    '''
    1. Select the scaffold network to use and load network data
        - original: PPI from STRING
        - noAPOE: PPI from STRING without APOE gene
        - biogrid: PPI from BioGRID
        - huri: PPI from HuRI
        - snap_brain: brain-specific PPI from PPT-Ohmnet
        - giant_brain: brain-specific functional network from GIANT
    '''

    if network == 'string':
        ppin_file_path      = f'{indir}/{disease}_STRING_PPI_edgelist.txt'
        print(ppin_file_path)

    elif network == 'snap_brain_noAPOE':
        ppin_file_path      = f'{indir}/other_networks/{disease}_SNAP_PPI_brain_noAPOE.edgelist'
    
    elif network == 'biogrid':
        ppin_file_path      = f'{indir}/other_networks/{disease}_BioGrid_PPI.edgelist'

    elif network == 'huri':
        ppin_file_path      = f'{indir}/other_networks/{disease}_HuRI_PPI.edgelist'

    elif network == 'snap_brain':
        ppin_file_path      = f'{indir}/other_networks/{disease}_SNAP_PPI_brain.edgelist'

    elif network == 'giant_brain':
        ppin_file_path      = f'{indir}/other_networks/{disease}_GIANT_brain.edgelist' # it is not simply a PPI

    elif network == 'shuffled':
        ppin_file_path      = f'{indir}/random_networks/shuffled/{disease}_PPI_rand{number}_edgelist.txt'
    
    elif network == 'rewired':
        ppin_file_path      = f'{indir}/random_networks/rewired/{disease}_PPI_rand{number}_edgelist.txt'
    
    ppi_graph = nx.read_edgelist(ppin_file_path)
    nodes = list(ppi_graph)
    edges = ppi_graph.edges

    print('Network used:', disease, network)
    print('# nodes =', nx.number_of_nodes(ppi_graph))
    print('# edges =', nx.number_of_edges(ppi_graph))
    print()


    '''
    2. Select the dataset to use and load variants data
        - ADNI: genetic cohort from Alzheimer's Disease Neuroimaging Initiative (ADNI)
        - LOAD: GWAS data from T Gen II dataset from NIAGDS
    '''

    if dataset == 'ADNI': # Use ADNI data
        print('Dataset used: ADNI')
        header = f'{indir}/ADNI/field_names.txt'
        missense_file_path  = f'{indir}/ADNI/{disease}_PPI_worst_missense.tsv'
        diagnosis_file_path = f'{indir}/ADNI/ADNIMERGE_metadata.csv'
        diagnosis = pd.read_csv(diagnosis_file_path, index_col=0)
        missense = process_variants_adni(missense_file_path, header)

    elif dataset == 'LOAD': # Use LOAD data
        print('Dataset used: LOAD')
        header = f'{indir}/LOAD/field_names_{disease}.txt'
        missense_file_path  = f'{indir}/LOAD/{disease}_PPI_worst_missense.tsv'
        diagnosis_file_path = f'{indir}/LOAD/LOAD_metadata.tsv'
        diagnosis = pd.read_csv(diagnosis_file_path, sep='\t', index_col=1)
        missense = process_variants_load(missense_file_path, header, disease)
    
    missense.columns = map(str.upper, missense.columns)

    '''
    3. Select the way of building graph datasets
        - missense: number of missense variants per gene (as node attributes)
        - TO-DO missense_pathogenic: number of missense pathogenic variants per gene (as node attributes) 
        - TO-DO variants: number of different types of variants per gene (as node attributes)
        - TO-DO variants_int: number of different types of variants per gene (as node attributes) & type of edge (as edge attributes)
    '''

    print(mode)
    if mode == 'missense':
            
        nodes_attr = per_node(mode, missense, nodes)
        nodes_attr.to_csv(f'data/table_datasets/{disease}_{network}_{mode}_{dataset}.csv')
        edges_attr = None

        result_graphs = create_samples_graphs('missense', nodes_attr, None, ppi_graph, diagnosis, target)
        result_graphs = delete_small_components(result_graphs, 4)

        print('Sample graph used:', '# nodes =', nx.number_of_nodes(result_graphs[0]), '# edges =', nx.number_of_edges(result_graphs[0]))
        print('Density =', nx.density(result_graphs[0]), 'Diameter =', nx.diameter(result_graphs[0]))

        return result_graphs
    


if __name__ == "__main__":

    dataset = 'LOAD'
    target  = 'LOAD'
    disease = 'ND'
    network = 'original'
    
    indir = 'data'
    outdir = f'data/graph_datasets/{target}'
    print('Input directory:', indir)
    print('Output directory:', outdir)
    print()

    start_time = datetime.datetime.now()
    print()

    result_nodes = main(indir, dataset, target, disease, network, 'missense', None)
    print('Coding: number of missense variants per node')

    outfile = f'{outdir}/{disease}_PPI_missense.pkl'
    print('Resulting dataset saved at:', outfile)
    print()

    with open(outfile, 'wb') as f:
        pickle.dump(result_nodes, f)

    result_nodes_time = datetime.datetime.now()
    print('Processing time:', result_nodes_time - start_time)
    print('\n\n')