import networkx as nx
import itertools
from itertools import combinations
import os
import csv
import numpy as np
import seaborn as sns
from networkx import triadic_census
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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


# # https://github.com/saref/multilevel-balance/tree/master
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
                        balanced_motifs.append(triangle)
                    else:
                        unbalanced_motifs.append(triangle)
        else:
            for item in items[0]:
                balance = 1
                for edge in item:
                    balance *= item[edge]
                balance_list.append(balance)
                if balance > 0:
                    balanced_motifs.append(item)
                else:
                    unbalanced_motifs.append(item)

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


def get_edgelist(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        edgelist = list(reader)

    # remove the header
    edgelist = edgelist[1:]

    edges = dict()
    for _, s, t, w in edgelist:
        edges[(int(s), int(t))] = float(w)

    return edges


if __name__ == '__main__':

    balanced_res = []
    unbalanced_res = []
    difference_res = []
    x = []

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

    # weight distribution visualization
    links_to_balanced = []
    weights_balanced = []
    for i in range(len(balanced_motifs)):
        for item in balanced_motifs[i].keys():
            if item not in links_to_balanced:
                links_to_balanced.append(item)
                weights_balanced.append(balanced_motifs[i][item])

    links_to_unbalanced = []
    weights_unbalanced = []
    for i in range(len(unbalanced_motifs)):
        for item in unbalanced_motifs[i].keys():
            if item not in links_to_unbalanced:
                links_to_unbalanced.append(item)
                weights_unbalanced.append(unbalanced_motifs[i][item])

    bins_num = 75
    # lower = min(min(weights_balanced), min(weights_unbalanced))
    # upper = max(max(weights_balanced), max(weights_unbalanced))

    bins = np.linspace(-0.7, 0.7, bins_num)

    # hist1, _ = np.histogram(weights_balanced, bins)
    hist1, _ = np.histogram(weights_balanced, bins=bins, density=False)
    hist2, _ = np.histogram(weights_unbalanced, bins=bins, density=False)
    hist1 = hist1 / sum(hist1)
    hist2 = hist2 / sum(hist2)
    hist_diff = (hist1 - hist2)
    balanced_res += list(hist1)
    unbalanced_res += list(hist2)
    difference_res += list(hist_diff)
    x += list(bins[:-1])

    fig = plt.figure(figsize=(5, 4))
    fig.add_axes([0.2, 0.2, 0.75, 0.7])

    ax = sns.lineplot(x=x, y=balanced_res, label='balanced motifs')
    ax1 = sns.lineplot(x=x, y=unbalanced_res, label="unbalanced motifs", color='orange')
    ax2 = sns.lineplot(x=x, y=difference_res, label='difference', color='grey')
    # plt.xlim(-0.3, 0.3)
    # plt.ylim(0, 48)
    # plt.ylim(-0.02, 0.07)
    plt.xlabel('weights', fontsize=12)  # 设置x轴标签
    plt.ylabel('frequency', fontsize=12)  # 设置y轴标签
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=7, loc='upper right')
    plt.show()
