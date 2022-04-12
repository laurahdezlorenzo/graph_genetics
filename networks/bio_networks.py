import requests
import json
import networkx as nx
import mygene
import os, io
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
  
    return string_identifiers

def get_string(input_file):
        
    '''
    For the given list of proteins print out only the interactions between these
    protein which have medium or higher confidence experimental score.
    '''

    species = 9606
    identity = 'lhlorenzo'
    disease = 'AD'

    genes_of_interest = load_GDAs(input_file)
    genes = get_string_ids(genes_of_interest, species, identity)

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

    string_ppis = pd.read_csv(io.StringIO(response.content.decode('utf-8')), sep='\t', names=fields)

    uniques = np.unique(string_ppis[['preferredName_A', 'preferredName_B']].values)
    interacting_proteins = list(uniques)
    with open(f'data/{disease}_STRING_PPI_intprots.txt', 'w') as f:
        for item in interacting_proteins:
            f.write("%s\n" % item)

    # Format PPIN data to obtain a list of protein interactions identified by their gene symbol.
    cols_to_drop = ['stringId_A', 'stringId_B', 'ncbiTaxonId', 'score', 'nscore',
                     'fscore', 'pscore', 'ascore', 'escore', 'dscore', 'tscore']
    edgelist = string_ppis.drop(columns=cols_to_drop, axis=1)
    edgelist.to_csv('data/other_networks/AD_STRING_PPI.edgelist', sep=' ', header=False, index=False)

    # Load network
    G_frozen = nx.read_edgelist('data/other_networks/AD_STRING_PPI.edgelist')  
    G = nx.Graph(G_frozen)
    
    original = G.number_of_nodes()

    # Delete nodes from components with less than 5 nodes
    for component in list(nx.connected_components(G)):
        if len(component)<5:
            for node in component:
                G.remove_node(node)

    largest = G.number_of_nodes()
    lost = original - largest
    lost_percent = round((lost/original), 4)

    print('Whole network:', original, 'nodes')
    print('Biggest connected component:', largest, 'nodes')
    print('Percentage of lost genes/nodes:', lost, f'({lost_percent*100}%)')

    return edgelist


def get_biogrid(genes_file):

    request_url = "https://webservice.thebiogrid.org" + "/interactions"
    access_key = "6a15ac18eadaa8786bc1a88d0ae00171"

    infile = open(genes_file, "r")
    geneList = infile.read().split("\n")
    infile.close()

    # These parameters can be modified to match any search criteria following
    # the rules outlined in the Wiki: https://wiki.thebiogrid.org/doku.php/biogridrest
    params = {
        "accesskey": access_key,
        "format": "json",  # Return results in TAB2 format
        "geneList": "|".join(geneList),  # Must be | separated
        "searchNames": "true",  # Search against official names
        'interSpeciesExcluded': 'true', # interactions w/ different species are excluded
        'throughputTag': 'high', 
        'includeHeader': 'true',
        "includeInteractors": "false",  # set to false to get interactions between genes
        "taxId": 9606  # Limit to Homo sapiens
    }

    r = requests.get(request_url, params=params)
    interactions = r.json()

    # Create a hash of results by interaction identifier
    data = {}
    for interaction_id, interaction in interactions.items():
        data[interaction_id] = interaction
        # Add the interaction ID to the interaction record, so we can reference it easier
        data[interaction_id]["INTERACTION_ID"] = interaction_id

    # Load the data into a pandas dataframe
    dataset = pd.DataFrame.from_dict(data, orient="index")

    # Re-order the columns and select only the columns we want to see

    columns = [
        "INTERACTION_ID",
        "ENTREZ_GENE_A",
        "ENTREZ_GENE_B",
        "OFFICIAL_SYMBOL_A",
        "OFFICIAL_SYMBOL_B",
        "EXPERIMENTAL_SYSTEM",
        "PUBMED_ID",
        "PUBMED_AUTHOR",
        "THROUGHPUT",
        "QUALIFICATIONS",
    ]
    dataset = dataset[columns]

    edgelist = dataset[['OFFICIAL_SYMBOL_A', 'OFFICIAL_SYMBOL_B']]
    edgelist = edgelist[edgelist['OFFICIAL_SYMBOL_A'] != edgelist['OFFICIAL_SYMBOL_B']] # remove self loops
    edgelist.to_csv('data/other_networks/AD_BioGrid_PPI.edgelist', sep='\t', index=False, header=None)

    G_frozen = nx.read_edgelist("data/other_networks/AD_BioGrid_PPI.edgelist")  
    G = nx.Graph(G_frozen)
    
    original = G.number_of_nodes()

    # Delete nodes from components with less than 5 nodes
    nodes_to_remove = []
    for component in list(nx.connected_components(G)):
        if len(component)<5:
            for node in component:
                G.remove_node(node)

    largest = G.number_of_nodes()
    lost = original - largest
    lost_percent = round((lost/original), 4)

    print('Whole network:', original, 'nodes')
    print('Biggest connected component:', largest, 'nodes')
    print('Percentage of lost genes/nodes:', lost, f'({lost_percent*100}%)')

    return edgelist

def get_huri(downloaded_file):

    interactions = pd.read_csv(downloaded_file, comment='#')
    edgelist = interactions[['Interactor A Gene Name', 'Interactor B Gene Name']]
    edgelist = edgelist[edgelist['Interactor A Gene Name'] != edgelist['Interactor B Gene Name']] # remove self loops
    edgelist.to_csv('data/other_networks/AD_HuRI_PPI.edgelist', sep='\t', index=None, header=False)

    G_frozen = nx.read_edgelist("data/other_networks/AD_HuRI_PPI.edgelist")  
    G = nx.Graph(G_frozen)
    
    original = G.number_of_nodes()

    # Delete nodes from components with less than 5 nodes
    nodes_to_remove = []
    for component in list(nx.connected_components(G)):
        if len(component)<5:
            for node in component:
                G.remove_node(node)

    largest = G.number_of_nodes()
    lost = original - largest
    lost_percent = round((lost/original), 4)

    print('Whole network:', original, 'nodes')
    print('Biggest connected component:', largest, 'nodes')
    print('Percentage of lost genes/nodes:', lost, f'({lost_percent*100}%)')

    return edgelist

def get_snap(genes_file):

    G = nx.read_edgelist('data/other_networks/PPT-Ohmnet_tissues-combined.edgelist', nodetype=int, data=(('tissue', str),))

    tissues_edgelist = pd.read_csv('data/other_networks/PPT-Ohmnet_tissues-combined.edgelist', sep='\t')
    brain_specific = tissues_edgelist[tissues_edgelist['tissue'] == 'brain']

    brain_specific.to_csv('data/other_networks/PPT-Ohmnet_tissues-brain.edgelist', sep='\t', index=False)
    G_brain = nx.read_edgelist('data/other_networks/PPT-Ohmnet_tissues-brain.edgelist', nodetype=int, data=(('tissue', str),))

    # List of genes to search for
    infile = open(genes_file, "r")
    genes = infile.read().split("\n")
    infile.close()
    len(genes)

    # Genes in PPT-Ohmnet are Entrez IDs, it is necessary to convert them to gene Symbols.
    mg = mygene.MyGeneInfo()
    out = mg.querymany(genes, scopes='symbol', fields='entrezgene', species='human')

    entrezgenes = []
    mapping = {}
    for o in out:
        entrezgenes.append(int(o['entrezgene']))
        mapping[int(o['entrezgene'])] = o['query']

    A_brain_frozen = G_brain.subgraph(entrezgenes)
    A_brain = nx.Graph(A_brain_frozen)
    original = A_brain.number_of_nodes()

    # Delete nodes from components with less than 5 nodes
    nodes_to_remove = []
    for component in list(nx.connected_components(A_brain)):
        if len(component)<5:
            for node in component:
                A_brain.remove_node(node)

    # Remove self-loops
    A_brain.remove_edges_from(list(nx.selfloop_edges(A_brain)))

    largest = A_brain.number_of_nodes()
    lost = original - largest
    lost_percent = round((lost/original), 4)

    print('Whole network:', original, 'nodes')
    print('Biggest connected component:', largest, 'nodes')
    print('Percentage of lost genes/nodes:', lost, f'({lost_percent*100}%)')

    A_brain_relabeled = nx.relabel_nodes(A_brain, mapping)
    nx.write_edgelist(A_brain_relabeled, 'data/other_networks/AD_SNAP_PPI_brain.edgelist')

    return A_brain_relabeled

def get_giant(genes_file):

    G_brain = nx.read_edgelist('data/other_networks/brain_C1.dat', nodetype=int, data=(('code', int),))

    # List of genes to search for
    infile = open(genes_file, "r")
    genes = infile.read().split("\n")
    infile.close()

    # Genes in GIANT are Entrez IDs, it is necessary to convert them to gene Symbols.
    mg = mygene.MyGeneInfo()
    out = mg.querymany(genes, scopes='symbol', fields='entrezgene', species='human')

    entrezgenes = []
    mapping = {}
    for o in out:
        entrezgenes.append(int(o['entrezgene']))
        mapping[int(o['entrezgene'])] = o['query']

    A_brain_frozen = G_brain.subgraph(entrezgenes)
    A_brain = nx.Graph(A_brain_frozen)
    original = A_brain.number_of_nodes()

    # Delete nodes from components with less than 5 nodes
    nodes_to_remove = []
    for component in list(nx.connected_components(A_brain)):
        if len(component)<5:
            for node in component:
                A_brain.remove_node(node)

    # Remove self-loops
    A_brain.remove_edges_from(list(nx.selfloop_edges(A_brain)))
  
    largest = A_brain.number_of_nodes()
    lost = original - largest
    lost_percent = round((lost/original), 4)

    print('Whole network:', original, 'nodes')
    print('Biggest connected component:', largest, 'nodes')
    print('Percentage of lost genes/nodes:', lost, f'({lost_percent*100}%)')

    A_brain_relabeled = nx.relabel_nodes(A_brain, mapping)
    nx.write_edgelist(A_brain_relabeled, 'data/other_networks/AD_GIANT_brain.edgelist')

    return A_brain_relabeled
