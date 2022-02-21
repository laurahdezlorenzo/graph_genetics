'''Code for randomize a graph structure with Networkx'''

import os, sys
import networkx as nx
import graph_tool.all as gt
import pandas as pd
import random
import matplotlib.pyplot as plt

def plot_graph(G):

    '''
    Plot graph.
    '''

    plt.figure(figsize=(12,12))
    nx.draw(G, with_labels=True)
    plt.show()

def load_graph(edgelist):

    graph = nx.read_edgelist(edgelist)

    print('Original graph')
    print('Number of nodes:', graph.number_of_nodes())
    print('Number of edges:', graph.number_of_edges())

    return graph

def shuffle_nodes(G):

    # Create a random mapping: old label -> new label
    node_mapping = dict(zip(G.nodes(),
                        sorted(G.nodes(),
                        key=lambda k: random.random())))
    
    # Build a new graph
    G_shuffled = nx.relabel_nodes(G, node_mapping)

    print('Sluffled graph')
    print('Number of nodes:', G_shuffled.number_of_nodes())
    print('Number of edges:', G_shuffled.number_of_edges())

    return G_shuffled

def expected_degree(G):

    # print(index)
    # print(degrees)
    # print(relabel_dict)
    
    # Rewire edges
    degrees = [deg for (_, deg) in G.degree()]
    G_rewired = nx.expected_degree_graph(degrees, selfloops=False)

    # Relabel nodes
    relabel_dict = {}
    index = 0
    for (node, deg) in G.degree():
        relabel_dict[index] = node
        index += 1

    G_rewired = nx.relabel_nodes(G_rewired, relabel_dict)

    print('Expected rewired graph')
    print('Number of nodes:', G_rewired.number_of_nodes())
    print('Number of edges:', G_rewired.number_of_edges())

    return G_rewired


def generate_RDPN(graph_file):

    '''
    Generates random GGI network where nodes keep the exactly same degrees.
    '''

    edges = pd.read_csv(graph_file, sep = " ",header = None)
    edges.drop_duplicates(inplace=True)
    edge_list = edges.values.tolist()
    nodes = list(set([item for sublist in edge_list for item in sublist]))
    int_nodes = {nodes[i]:i for i in range(len(nodes))}
    edges = edges.replace(int_nodes)
    edge_list_int = edges.values.tolist()
    
    GT = gt.Graph(directed=False)
    GT.add_edge_list(edge_list_int)

    gt.random_rewire(GT, model = "constrained-configuration", n_iter = 100, edge_sweep = True)

    edges_new = list(GT.get_edges())
    nodes_rev = {v:k for k,v in int_nodes.items()}
    edges_new = [(nodes_rev[x[0]], nodes_rev[x[1]]) for x in edges_new]

    # print(edges_new)

    G_rewired = nx.Graph()
    for e in edges_new:
        G_rewired.add_edge(e[0], e[1])

    print('Rewired graph')
    print('Number of nodes:', G_rewired.number_of_nodes())
    print('Number of edges:', G_rewired.number_of_edges())

    return(G_rewired)


if __name__ == '__main__':

    # DIS = 'ND'  # or ND
    # NET = 'PPI' # or Multi
    # REP = 1

    DIS = str(sys.argv[1]) 
    NET = str(sys.argv[2])
    REP = str(sys.argv[3])
    DIR = '/home/laura/Documents/DATASETS/PPI_networks'

    os.chdir('ppa-graphs')

    infile  = f'{DIR}/{DIS}_{NET}_edgelist_biggest.txt'
    outfile = f'{DIR}/REWIRED/{DIS}_{NET}_random_edgelists/{DIS}_{NET}_rand{REP}_edgelist.txt'

    original = load_graph(infile)

    # shuffled = shuffle_nodes(original)
    # exp_rewired = expected_degree(G)
    rewired  = generate_RDPN(infile) 

    # plot_graph(original)
    # plot_graph(shuffled)
    # plot_graph(rewired)

    # Save new random network as an edgelist file
    # nx.write_edgelist(shuffled, outfile)
    nx.write_edgelist(rewired, outfile)
    print()