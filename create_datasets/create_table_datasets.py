'''
Description
'''
import os, io, datetime
import numpy as np
import pandas as pd

def ADNI_data(mode, data_file, metadata_file):

    data = pd.read_csv(data_file, index_col=0)
    metadata = pd.read_csv(metadata_file, index_col=0)
    print(metadata.shape)

    data = data.T

    if DIS == 'AD' and NET == 'PPI':
        cols_to_drop = ['SLC30A4', 'SLC30A6', 'INPP5D', 'CD2AP', 'CHRNB2', 'CHRNA7']
        data.drop(columns=cols_to_drop, inplace=True)
    
    elif DIS == 'ND' and NET == 'PPI':
        cols_to_drop = ['PRPH', 'NEFH', 'SLC30A4', 'SLC30A6', 'CD33', 'GSTP1',
                        'TH', 'SLC6A3', 'DDC', 'SLC18A2', 'CHRNB2', 'CHRNA7']
        data.drop(columns=cols_to_drop, inplace=True)

    # For missense
    if mode == 'missense':
        result = pd.concat([data, metadata], axis=1)
        return result

    # For variants
    elif mode == 'variants':

        variant_types = ['missense_variant',
				'synonymous_variant',
				'stop_gained',
				'frameshift_variant', 
				'inframe_deletion',
				'inframe_insertion',
				'stop_retained_variant', 
				'coding_sequence_variant',
				'start_lost',
				'stop_lost',
				'protein_altering_variant',
				'3_prime_utr', '5_prime_utr',
				'non_coding_transcript_exon_variant',
				'non_coding_transcript_variant',
				'upstream_gene_variant',
				'downstream_gene_variant',
				'intron_variant',
				'splice_region_variant']
        
        gene_dfs = []
        for col in data.columns:
            gene_data = pd.DataFrame(data[col].str.split(', ', 19).tolist(),
                                        columns = variant_types,
                                        index=data.index)
            
            
            gene_data['missense_variant'].replace('\[', '', regex=True, inplace=True)
            gene_data['splice_region_variant'].replace('\]', '', regex=True, inplace=True)
            gene_data = gene_data.add_prefix(f'{col}_')
            gene_dfs.append(gene_data)

        new_data = pd.concat(gene_dfs, axis=1)

        result = pd.concat([new_data, metadata], axis=1)
        return result

def LOAD_data(mode, data_file, metadata_file):

    data = pd.read_csv(data_file, index_col=0)
    metadata = pd.read_csv(metadata_file, index_col='IID', sep='\t')
    # print(metadata)
    print(metadata.shape)

    data = data.T
    # print(data)

    if DIS == 'AD' and NET == 'PPI':
        cols_to_drop = ['SLC30A4', 'SLC30A6', 'INPP5D', 'CD2AP', 'CHRNB2', 'CHRNA7']
        data.drop(columns=cols_to_drop, inplace=True)
    
    elif DIS == 'ND' and NET == 'PPI':
        cols_to_drop = ['PRPH', 'NEFH', 'SLC30A4', 'SLC30A6', 'CD33', 'GSTP1',
                        'TH', 'SLC6A3', 'DDC', 'SLC18A2', 'CHRNB2', 'CHRNA7']
        data.drop(columns=cols_to_drop, inplace=True)

    # For missense
    if mode == 'missense':
        result = pd.concat([data, metadata], axis=1)
        result = result[:-1]
        return result

    return result

if __name__ == '__main__':

    DIS = 'ND'  # or ND
    NET = 'PPI' # or Multi
    DIR = '/home/laura/Documents/DATASETS/table_datasets'

    os.chdir('ppa-graphs')

    ############################################################################
    # Table datasets for ADNI data

    # missense_file = f'/home/laura/Documentos/DATASETS/table_datasets/{DIS}_{NET}_missense.csv'
    # variants_file = f'/home/laura/Documentos/DATASETS/table_datasets/{DIS}_{NET}_variants.csv'
    # metadata_file = 'data/ADNI/ADNIMERGE_genetics_biomarkers.csv'

    # missense_dataset = ADNI_data('missense', missense_file, metadata_file)
    # missense_dataset.to_csv(f'{DIR}/{DIS}_{NET}_missense_with_biomarkers.csv')
    # print(missense_dataset)

    # variants_dataset = ADNI_data('variants', variants_file, metadata_file)
    # variants_dataset.to_csv(f'{DIR}/{DIS}_{NET}_variants_with_biomarkers.csv')
    # print(variants_dataset)

    ###########################################################################
    # Table datasets for LOAD data
    
    missense_file = f'/home/laura/Documents/DATASETS/table_datasets/{DIS}_{NET}_missense_LOAD.csv'
    metadata_file = '/home/laura/Documents/DATASETS/PPI_networks/LOAD/LOAD_metadata.tsv'

    missense_dataset = LOAD_data('missense', missense_file, metadata_file)
    missense_dataset.to_csv(f'{DIR}/{DIS}_{NET}_missense_with_biomarkers_LOAD.csv')
    print(missense_dataset)