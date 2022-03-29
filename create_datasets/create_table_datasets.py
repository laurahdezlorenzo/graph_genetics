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

    # # For variants
    # elif mode == 'variants':

    #     variant_types = ['missense_variant',
	# 			'synonymous_variant',
	# 			'stop_gained',
	# 			'frameshift_variant', 
	# 			'inframe_deletion',
	# 			'inframe_insertion',
	# 			'stop_retained_variant', 
	# 			'coding_sequence_variant',
	# 			'start_lost',
	# 			'stop_lost',
	# 			'protein_altering_variant',
	# 			'3_prime_utr', '5_prime_utr',
	# 			'non_coding_transcript_exon_variant',
	# 			'non_coding_transcript_variant',
	# 			'upstream_gene_variant',
	# 			'downstream_gene_variant',
	# 			'intron_variant',
	# 			'splice_region_variant']
        
    #     gene_dfs = []
    #     for col in data.columns:
    #         gene_data = pd.DataFrame(data[col].str.split(', ', 19).tolist(),
    #                                     columns = variant_types,
    #                                     index=data.index)
            
            
    #         gene_data['missense_variant'].replace('\[', '', regex=True, inplace=True)
    #         gene_data['splice_region_variant'].replace('\]', '', regex=True, inplace=True)
    #         gene_data = gene_data.add_prefix(f'{col}_')
    #         gene_dfs.append(gene_data)

    #     new_data = pd.concat(gene_dfs, axis=1)
    #     result = pd.concat([new_data, metadata], axis=1)

    #     return result

def load_data(mode, disease, data_file, metadata_file):

    data = pd.read_csv(data_file, index_col=0)
    metadata = pd.read_csv(metadata_file, index_col='IID', sep='\t')

    data = data.T

    if disease == 'AD':
        cols_to_drop = ['SLC30A4', 'SLC30A6', 'INPP5D', 'CD2AP', 'CHRNB2', 'CHRNA7']
        data.drop(columns=cols_to_drop, inplace=True)
    
    elif disease == 'ND':
        cols_to_drop = ['PRPH', 'NEFH', 'SLC30A4', 'SLC30A6', 'CD33', 'GSTP1',
                        'TH', 'SLC6A3', 'DDC', 'SLC18A2', 'CHRNB2', 'CHRNA7']
        data.drop(columns=cols_to_drop, inplace=True)

    # For missense
    if mode == 'missense':
        result = pd.concat([data, metadata], axis=1)
        return result

    return result