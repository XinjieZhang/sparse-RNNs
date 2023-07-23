import networkx as nx
import itertools
from itertools import combinations
import os
import csv
from networkx import triadic_census
import matplotlib.pyplot as plt

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
def calculate_traid_balance(G):

    G_new = G.copy()

    # remove isolate and pendants from the dataframe
    G_new.remove_nodes_from(nx.isolates(G_new))
    remove = [node for node, degree in G_new.degree() if degree == 1]
    G_new.remove_nodes_from(remove)
    # remove self loop
    G_new.remove_edges_from(nx.selfloop_edges(G_new))
    print('triadic_census: ', triadic_census(G_new))

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

    ## getting the balance vs. imbalance triads
    balances = []
    imbalances = []
    balanced_motifs = []
    unbalanced_motifs = []
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
                        balanced_motifs.append(calculate_connections_interaction(triangle))
                    else:
                        unbalanced_motifs.append(calculate_connections_interaction(triangle))
        else:
            for item in items[0]:
                balance = 1
                for edge in item:
                    balance *= item[edge]
                balance_list.append(balance)
                if balance > 0:
                    balanced_motifs.append(calculate_connections_interaction(item))
                else:
                    unbalanced_motifs.append(calculate_connections_interaction(item))

        neg = []
        for n in balance_list:
            if n <= 0:
                neg.append(n)
        if neg:
            imbalances.append(items)
        else:
            balances.append(items)

    print('Triad Level Balance: ', (len(balances) / (len(balances) + len(imbalances))))
    print('Number of balance and transitive triads: ', len(balances))
    print('Number of imbalance and transitive triads: ', len(imbalances))

    print('Number of balance triads in each census', count_lists(balances))
    print('Number of imbalance triads in each census', count_lists(imbalances))

    return balanced_motifs, unbalanced_motifs


if __name__ == '__main__':

    results = []

    DATAPATH = os.path.join(os.getcwd(), '../results', 'MNIST')
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
    print('-------------------------Triadic balance-----------------------------')
    balanced_motifs, unbalanced_motifs = calculate_traid_balance(Graph_list[0])
    results.append([balanced_motifs, unbalanced_motifs])

    plt.figure()
    plt.scatter(range(len(balanced_motifs)), balanced_motifs, marker='o', label='Balanced', s=3.5)
    plt.scatter(range(len(unbalanced_motifs)), unbalanced_motifs, marker='*', label='Unbalanced', s=3.5)
    plt.xlabel('motifs', fontsize=12)
    plt.ylabel('$w_{ij} \dot w_{jk} + w_{ik}$', fontsize=12)
    plt.legend()
    plt.show()