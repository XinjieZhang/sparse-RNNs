# coding utf-8

import numpy as np
import random
import time
import os
import networkx as nx
from balanced_ratio import compute_local_balanced_ratio


def select_edges(edges, weight):
    m = len(edges)
    l = 0
    edges_copy = edges.copy()
    weight_copy = weight.copy()
    while l < 1:
        index = np.random.choice(range(m), size=2, replace=False)
        l1 = edges[index[0]]
        l2 = edges[index[1]]
        if (l1[0] != l2[0]) and (l1[1] != l2[1]):
            l11 = (l1[0], l2[1])
            l22 = (l2[0], l1[1])
            w11 = weight[l1[0], l1[1]]
            w22 = weight[l2[0], l2[1]]
            if (l11 not in edges) and (l22 not in edges):
                edges_copy[index[0]] = l11
                edges_copy[index[1]] = l22
                weight_copy[l1[0], l1[1]] = 0
                weight_copy[l2[0], l2[1]] = 0
                weight_copy[l1[0], l2[1]] = w11
                weight_copy[l2[0], l1[1]] = w22
                l += 1

    return l1, l2, edges_copy, weight_copy


def local_balanced_ratio(l1, l2, edges, weight):
    B1 = compute_local_balanced_ratio(l1[0], l1[1], edges, weight)
    B2 = compute_local_balanced_ratio(l2[0], l2[1], edges, weight)
    B = B1 + B2
    if max(B) == 0:
        return 0
    else:
        return B[0] / (B[0] + B[1])


def p_min(delta, T):
    probability = np.exp(-delta / T)
    return probability


def p_max(delta, T):
    probability = np.exp(delta / T)
    return probability


def deal_min(x1, x2, w1, w2, delta, T):
    if delta < 0:
        return x2, w2
    else:
        p = p_min(delta, T)
        if p > random.random():
            return x2, w2
        else:
            return x1, w1


def deal_max(x1, x2, w1, w2, delta, T):
    if delta > 0:
        return x2, w2
    else:
        p = p_max(delta, T)
        if p > random.random():
            return x2, w2
        else:
            return x1, w1


def main(n, p, seed=1):
    # generate ER graphs
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed, directed=True)
    rng = np.random.RandomState(seed=seed)
    w = rng.uniform(low=-0.5, high=0.5, size=[n, n])
    edge_list = G.edges()
    w_sign = np.zeros(shape=[n, n])

    path = os.path.join('generated', 'trial')

    if not os.path.exists(path):
        os.makedirs(path)

    f = open(os.path.join(path, 'ER_n_'+str(n)+'_p_'+str(p)+'.txt'), 'w')
    for s, t in G.edges:
        w_sign[s, t] = np.sign(w[s, t])
        if np.sign(w[s, t]) > 0:
            f.write(str(s) + '\t' + str(t) + '\t' + str(1) + '\n')
        else:
            f.write(str(s) + '\t' + str(t) + '\t' + str(2) + '\n')
    f.close()

    # hyperparameters for simulated annealing
    T_max = 100
    T_min = 1e-1
    rate = 0.98
    tab = 'max'
    iterMax = 10

    # simulated annealing
    edges_1 = list(edge_list)
    weight_1 = w_sign
    T = T_max
    t_start = time.time()
    while T >= T_min:
        for i in range(iterMax):
            # Randomly select two edges to be exchanged
            l1, l2, edges_2, weight_2 = select_edges(edges_1, weight_1)
            C1 = local_balanced_ratio(l1, l2, edges_1, weight_1)
            C2 = local_balanced_ratio((l1[0], l2[1]), (l2[0], l1[1]), edges_2, weight_2)

            dC = C2 - C1

            if tab == 'max':
                edges_1, weight_1 = deal_max(edges_1, edges_2, weight_1, weight_2, dC, T)
            else:
                edges_1, weight_1 = deal_min(edges_1, edges_2, weight_1, weight_2, dC, T)
        T *= rate

    print('finished!')
    print('Time: {:0.2f}'.format(time.time() - t_start))

    fw = open(os.path.join(path, 'BN_n_'+str(n)+'_p_'+str(p)+'.txt'), 'w')
    for s, t in sorted(edges_1):
        if weight_1[s, t] > 0:
            fw.write(str(s) + '\t' + str(t) + '\t' + str(1) + '\n')
        else:
            fw.write(str(s) + '\t' + str(t) + '\t' + str(2) + '\n')
    fw.close()


if __name__ == '__main__':
    n = 150  # number of nodes
    p = 0.05  # sparsity
    main(n=n, p=p)

