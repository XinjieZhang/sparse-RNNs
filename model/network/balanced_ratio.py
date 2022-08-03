# coding utf-8

import numpy as np
import networkx as nx


def motif_type(G):
    n = len(G.nodes)
    degree_sequence = np.zeros([2, n])  # degree_sequence[[indegree], [outdegree]]
    degree_sorted = np.zeros([2, n])  # degree_sorted[[indegree], [outdegree]]
    for i in range(n):
        degree_sequence[0, i] = list(G.in_degree)[i][1]
        degree_sequence[1, i] = list(G.out_degree)[i][1]

    index = np.argsort(degree_sequence[0, :])
    for i in range(n):
        degree_sorted[:, i] = degree_sequence[:, index[i]]
    if list(degree_sorted[0, :]) == [0, 1, 2] and list(degree_sorted[1, :]) == [2, 1, 0]:
        return 38
    elif list(degree_sorted[0, :]) == [1, 1, 2] and list(degree_sorted[1, :]) == [2, 2, 0]:
        return 46
    elif list(degree_sorted[0, :]) == [0, 2, 2] and list(degree_sorted[1, :]) == [2, 1, 1]:
        return 166
    elif list(degree_sorted[0, :]) == [2, 2, 2] and list(degree_sorted[1, :]) == [2, 2, 2]:
        return 238
    else:
        return None


def compute_balanced_number(edge, A, B=None):

    if B is None:
        B = list([0, 0]) # [Balanced num, Imbalanced num]

    sign = 1
    for (s, t) in edge:
        sign *= np.sign(A[s, t]) # A is adjacent matrix (weighted)

    if sign > 0:
        B[0] += 1
    else:
        B[1] += 1

    return B


def find_semicycle(G):
    semicycle = list()
    for (s, t) in G.edges:
        copyG = G.copy()
        if (t, s) in G.edges:
            copyG.remove_edges_from([(s, t)])
            semicycle.append(copyG.edges)
    return semicycle


def compute_balance_ratio(type, G1, weight):

    if type == 38:
        B = compute_balanced_number(G1.edges, weight)
        return B

    if type == 46 or type == 166:
        semicycle = find_semicycle(G1)
        B = list([0, 0])
        for i in range(len(semicycle)):
            B = compute_balanced_number(semicycle[i], weight, B)
        return B

    if type == 238:
        nodes = list(G1.nodes)
        edges_remove = list()
        edges_remove.append([(nodes[0], nodes[1]), (nodes[0], nodes[2]), (nodes[1], nodes[2])])
        edges_remove.append([(nodes[0], nodes[1]), (nodes[0], nodes[2]), (nodes[2], nodes[1])])
        edges_remove.append([(nodes[1], nodes[0]), (nodes[2], nodes[0]), (nodes[1], nodes[2])])
        edges_remove.append([(nodes[1], nodes[0]), (nodes[2], nodes[0]), (nodes[2], nodes[1])])
        edges_remove.append([(nodes[0], nodes[1]), (nodes[2], nodes[0]), (nodes[2], nodes[1])])
        edges_remove.append([(nodes[1], nodes[0]), (nodes[0], nodes[2]), (nodes[1], nodes[2])])

        B = list([0, 0])
        for semicycle in edges_remove:
            copyG1 = G1.copy()
            copyG1.remove_edges_from(semicycle)
            B = compute_balanced_number(copyG1.edges, weight, B)

        return B 


def find_neighbor(node_i, edges):

    neighbor = list()
    for s, t in edges:
        if s == node_i:
            if t not in neighbor:
                neighbor.append(t)
            continue
        elif t == node_i:
            if s not in neighbor:
                neighbor.append(s)

    return sorted(neighbor)


"""
def motif_correlated_with_edge(source, s_neighbor, target, t_neighbor):

    motif = list()
    i = source
    j = target
    i_neighbor = s_neighbor
    j_neighbor = t_neighbor
    if len(s_neighbor) < len(t_neighbor):
        i = target
        j = source
        i_neighbor = t_neighbor
        j_neighbor = s_neighbor

    for m in range(len(i_neighbor)-1):
        for n in range(m+1, len(i_neighbor)):
            motif.append(sorted([i, i_neighbor[m], i_neighbor[n]]))

    for m in range(len(j_neighbor)-1):
        for n in range(m+1, len(j_neighbor)):
            if sorted([j, j_neighbor[m], j_neighbor[n]]) not in motif:
                motif.append(sorted([j, j_neighbor[m], j_neighbor[n]]))

    return motif
"""


def motif_correlated_with_edge(source, s_neighbor, target, t_neighbor):

    motif = list()
    i = source
    j = target
    i_neighbor = s_neighbor
    j_neighbor = t_neighbor
    if len(s_neighbor) < len(t_neighbor):
        i = target
        j = source
        i_neighbor = t_neighbor
        j_neighbor = s_neighbor

    i_neighbor.remove(j)
    j_neighbor.remove(i)
    for m in range(len(i_neighbor)):
        motif.append(sorted([i, j, i_neighbor[m]]))

    for n in range(len(j_neighbor)):
        if sorted([j, i, j_neighbor[n]]) not in motif:
            motif.append(sorted([j, i, j_neighbor[n]]))

    return motif


def compute_local_balanced_ratio(source, target, edge_list, weight):
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    i = source
    j = target

    i_neighbor = find_neighbor(i, edge_list)
    j_neighbor = find_neighbor(j, edge_list)

    subgraphs = motif_correlated_with_edge(i, i_neighbor, j, j_neighbor)

    B = [0, 0]
    for motif in subgraphs:
        G1 = G.subgraph(motif)
        type = motif_type(G1)
        if type is not None:
            B += np.array(compute_balance_ratio(type=type, G1=G1, weight=weight))

    return B


if __name__ == '__main__':

    n = 5 # number of nodes
    p = 0.5  # sparsity
    G = nx.erdos_renyi_graph(n=n, p=p, seed=0, directed=True)

    w = np.random.uniform(low=-0.5, high=0.5, size=[n, n])
    w_sign = np.zeros(shape=[n, n])
    for s, t in G.edges:
        w_sign[s, t] = np.sign(w[s, t])

    print(compute_local_balanced_ratio(0, 4, G.edges, w_sign))

