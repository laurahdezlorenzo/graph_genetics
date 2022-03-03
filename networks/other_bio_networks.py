import requests
import json
import pandas as pd
import networkx as nx
import mygene

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
    for o in out:
        entrezgenes.append(int(o['entrezgene']))

    A_brain_frozen = G_brain.subgraph(entrezgenes)
    A_brain = nx.Graph(A_brain_frozen)

    # Delete nodes from components with less than 5 nodes
    nodes_to_remove = []
    for component in list(nx.connected_components(A_brain)):
        if len(component)<5:
            for node in component:
                A_brain.remove_node(node)

    # Remove self-loops
    A_brain.remove_edges_from(list(nx.selfloop_edges(A_brain)))

    # Write edgelist
    nx.write_edgelist(A_brain, 'data/other_networks/AD_SNAP_PPI_brain.edgelist')

    return A_brain

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
    for o in out:
        entrezgenes.append(int(o['entrezgene']))

    A_brain_frozen = G_brain.subgraph(entrezgenes)
    A_brain = nx.Graph(A_brain_frozen)
    len(A_brain.nodes)

    # Delete nodes from components with less than 5 nodes
    nodes_to_remove = []
    for component in list(nx.connected_components(A_brain)):
        if len(component)<5:
            for node in component:
                A_brain.remove_node(node)

    # Remove self-loops
    A_brain.remove_edges_from(list(nx.selfloop_edges(A_brain)))

    nx.write_edgelist(A_brain, 'data/other_networks/AD_GIANT_brain.edgelist')

    return A_brain
