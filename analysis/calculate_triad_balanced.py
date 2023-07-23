import networkx as nx
import itertools
from itertools import combinations
import os
import csv
import time

import sys
sys.path.append('../')


## counting the number of instances in a list
def count_lists(mylist):
    new_dict = {}
    for i in mylist:
        if i[1] not in new_dict:
            new_dict[i[1]] = 1
        else:
            new_dict[i[1]] += 1
    return (new_dict)


## Get all triples in triads with respect to their census and edgelists (in edge_atts)
def get_directed_triads(triads):
    # Get all triplets of edges
    for candidate_edges in combinations(triads.items(), 3):
        # Get edges between unique pair of nodes
        unique_edges = set([tuple(sorted(k)) for k, v in candidate_edges])
        # Only consider triad in which the tree edges use a unique pair of nodes
        if len(unique_edges) == 3:
            yield dict(candidate_edges)


## searching through traids
def search_triangles(G, nodes=None):
    if nodes is None:
        nodes_nbrs = G.adj.items()
    else:
        nodes_nbrs = ((n, G[n]) for n in G.nbunch_iter(nodes))
    for v, v_nbrs in nodes_nbrs:
        vs = set(v_nbrs) - {v}
        for w in vs:
            # print(w)
            xx = vs & (set(G[w]) - {w})
            yield [set(x) for x in list(zip(itertools.repeat(v), itertools.repeat(w), list(xx)))]


def calculate_connections_interaction(triads):

    source = []
    target = []
    for s, t in triads.keys():
        if s not in source:
            source.append(s)
        else:
            node_i = s
            source.remove(s)

        if t not in target:
            target.append(t)
        else:
            node_k = t
    node_j = source[0]

    return triads[(node_i, node_j)] * triads[(node_j, node_k)] + triads[(node_i, node_k)]


# Calculate balance in traids (main function)
# https://github.com/saref/multilevel-balance/tree/master
def calculate_traid_balance(G):

    G_new = G.copy()

    # remove isolate and pendants from the dataframe
    G_new.remove_nodes_from(nx.isolates(G_new))
    remove = [node for node, degree in G_new.degree() if degree == 1]
    G_new.remove_nodes_from(remove)
    # remove self loop
    G_new.remove_edges_from(nx.selfloop_edges(G_new))
    # print('triadic_census: ', triadic_census(G_new))

    triad_dict = {}
    triad_class = {}
    all_triads = []
    ## there are only 4 transistive census: 030T, 120D, 120U, and 300
    non_transitive_census = ['003', '012', '102', '021D', '021C', '021U', '021', '111U', '111D', '201', '030C', '120C',
                             '210']

    iter_g = search_triangles(G_new)

    for iter_t in iter_g:
        for ta in list(iter_t):
            tt = ",".join([str(x) for x in sorted(set(ta))])
            triad_dict[tt] = True

    for val in triad_dict.keys():
        nodes = [int(x) for x in val.split(",")]
        census = [k for k, v in nx.triads.triadic_census(G_new.subgraph(nodes)).items() if v][0]
        if census not in non_transitive_census:
            sign = nx.get_edge_attributes(G_new.subgraph(nodes), 'weight')
            triad_class[val] = [census, sign]

    for key, value in triad_class.items():
        all_directed_triads = list(get_directed_triads(value[1]))
        all_triads.append([all_directed_triads, value[0]])


    B_030T = []
    Ub_030T = []
    B_120D = []
    Ub_120D = []
    B_120U = []
    Ub_120U = []
    B_300 = []
    Ub_300 = []

    for items in all_triads:

        balance_list = []

        ## removing two cycles from 300 and then calculate balance
        if items[1] == '300':
            for triangle in items[0]:
                node = []
                for edge in triangle:
                    if edge[0] not in node:
                        node.append(edge[0])
                if len(node) != 3:
                    balance = 1
                    for edge in triangle:
                        balance *= triangle[edge]
                    balance_list.append(balance)
                    if balance > 0:
                        B_300.append(triangle)
                    else:
                        Ub_300.append(triangle)
        elif items[1] == '030T':
            for item in items[0]:
                balance = 1
                for edge in item:
                    balance *= item[edge]
                balance_list.append(balance)
                if balance > 0:
                    B_030T.append(item)
                else:
                    Ub_030T.append(item)
        elif items[1] == '120D':
            for item in items[0]:
                balance = 1
                for edge in item:
                    balance *= item[edge]
                balance_list.append(balance)
                if balance > 0:
                    B_120D.append(item)
                else:
                    Ub_120D.append(item)
        elif items[1] == '120U':
            for item in items[0]:
                balance = 1
                for edge in item:
                    balance *= item[edge]
                balance_list.append(balance)
                if balance > 0:
                    B_120U.append(item)
                else:
                    Ub_120U.append(item)

    if len(B_300) + len(Ub_300) != 0:
        BR_300 = len(B_300) / (len(B_300) + len(Ub_300))
    else:
        BR_300 = 0

    if len(B_030T) + len(Ub_030T) != 0:
        BR_030T = len(B_030T) / (len(B_030T) + len(Ub_030T))
    else:
        BR_030T = 0

    if len(B_120D) + len(Ub_120D) != 0:
        BR_120D = len(B_120D) / (len(B_120D) + len(Ub_120D))
    else:
        BR_120D = 0

    if len(B_120U) + len(Ub_120U) != 0:
        BR_120U = len(B_120U) / (len(B_120U) + len(Ub_120U))
    else:
        BR_120U = 0

    k = 0
    Balance_ratio = 0
    for j in [BR_300, BR_030T, BR_120D, BR_120U]:
        if j != 0:
            Balance_ratio += j
            k += 1

    print('Triad Level Balance: ', Balance_ratio / k)

    return Balance_ratio / k
    # return balanced_motifs, unbalanced_motifs


if __name__ == '__main__':

    DATAPATH = os.path.join('../results', 'MNIST')
    root_dir = 'pruning_n_150_p_0.05'
    model_dir = os.path.join(DATAPATH, root_dir)
    csv_file = os.path.join('edge_list_weighted_200.csv')

    Graph_list = []
    path = os.path.join(model_dir, csv_file)
    with open(path, newline='') as f:
        edgelist = []
        reader = csv.reader(f)
        edgelist = list(reader)

    # remove the header
    edgelist = edgelist[1:]

    Graph_list.append(nx.DiGraph())
    for [id, u, v, w] in edgelist:
        Graph_list[0].add_edge(u, v, weight=float(w))

    mapping = dict(zip(Graph_list[0].nodes(), range(len(Graph_list[0].nodes()))))
    Graph_list[0] = nx.relabel_nodes(Graph_list[0], mapping)
    nx.relabel_nodes(Graph_list[0], mapping)
    print('-------------------------Triadic balance-----------------------------')
    t_start = time.time()
    calculate_traid_balance(Graph_list[0])





