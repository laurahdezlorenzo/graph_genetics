'''
Description
'''

import os, io
import requests
import numpy as np
import pandas as pd


def load_GDAs(input_file):

    '''
    Load GDAs downloaded from DisGeNET
    '''

    gdas = pd.read_csv(input_file, sep='\t')
    gene_symbols = list(set(gdas['gene_symbol']))

    print('Unique genes from DisGeNET:', len(gene_symbols))

    return gene_symbols

def get_string_ids(genes, species, identity):

    '''
    For a given list of proteins the script resolves them (if possible) to the best 
    matching STRING identifier and prints out the mapping on screen in the TSV format
    '''

    string_api_url = "https://string-db.org/api"
    output_format = "tsv-no-header"
    method = "get_string_ids"

    # Set parameters
    params = {
        'identifiers' : "\r".join(genes),
        'species' : species, 
        'limit' : 1, 
        'echo_query' : 1,
        'caller_identity' : 'lhlorenzo'
    }

    # Construct URL
    request_url = "/".join([string_api_url, output_format, method])

    # Call STRING
    results = requests.post(request_url, data=params)

    # Read and parse results
    string_identifiers = []
    for line in results.text.strip().split("\n"):
        l = line.split("\t")
        string_identifiers.append(l[2])

    print('Genes in STRING:', len(string_identifiers))
    
    return string_identifiers

def get_PPI_network(genes, species, identity, disease):
        
    '''
    For the given list of proteins print out only the interactions between these
    protein which have medium or higher confidence experimental score.
    '''

    string_api_url = "https://string-db.org/api"
    output_format = "tsv-no-header"
    method = "network"

    # Construct URL
    request_url = "/".join([string_api_url, output_format, method])

    # Set parameters
    params = {
        "identifiers" : "%0d".join(genes),
        "species" : species,
        "network_type" : "physical",
        "caller_identity" : identity
    }

    # Call STRING
    response = requests.post(request_url, data=params)

    fields = ['stringId_A', 'stringId_B', 'preferredName_A', 'preferredName_B',
                'ncbiTaxonId', 'score', 'nscore', 'fscore', 'pscore', 'ascore',
                'escore', 'dscore', 'tscore']

    string_ppis = pd.read_csv(io.StringIO(response.content.decode('utf-8')),
                                sep='\t',
                                names=fields)

    print('Interactions between these genes:', string_ppis.shape[0])

    get_interacting_proteins(string_ppis, disease)
    make_edgelist(string_ppis, disease)

def get_interacting_proteins(ppin, disease):

    uniques = np.unique(ppin[['preferredName_A', 'preferredName_B']].values)
    interacting_proteins = list(uniques)

    with open(f'data/{disease}_STRING_PPI_intprots.txt', 'w') as f:
        for item in interacting_proteins:
            f.write("%s\n" % item)

    print('Proteins interacting at least with another one:', len(interacting_proteins))

def make_edgelist(ppin, disease):

    '''
    Format PPIN data to obtain a list of protein interactions identified by their
    gene symbol.
    '''

    cols_to_drop = ['stringId_A', 'stringId_B', 'ncbiTaxonId', 'score', 'nscore',
                     'fscore', 'pscore', 'ascore', 'escore', 'dscore', 'tscore']

    ppin.drop(columns=cols_to_drop, axis=1, inplace=True)

    ppin.to_csv(f'data/{disease}_STRING_PPI_edgelist.txt',
                sep=' ',
                header=False,
                index=False)

