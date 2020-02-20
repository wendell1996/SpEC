import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import random
import time
import pickle as pkl
import networkx as nx
import sys
import os
import operator
import sklearn.metrics as Metrics
import pandas as pd
import argparse
import torch


def _f1(a, b):
    eps = 1e-30
    p = len(a & b) / (len(a) + eps)
    r = len(a & b) / (len(b) + eps)
    return 2 * p * r / (p + r + eps)


def max_f1(a, b):
    return max([_f1(a[i], b) for i in range(len(a))])


def f1(pre, true):
    eps = 1e-30
    sum_true = sum([max_f1(pre, true[i]) for i in range(len(true))])
    sum_pre = sum([max_f1(true, pre[i]) for i in range(len(pre))])
    f1 = sum_true / 2 / len(true) + sum_pre / 2 / (len(pre) + eps)
    return f1


def get_community(probability_matrix):
    s_mode = sp.issparse(probability_matrix)
    community_list = []
    for community in probability_matrix.transpose():
        if s_mode:
            community = set(community.nonzero()[1])
        else:
            community = set(np.nonzero(community)[0])
        community_list.append(community)
    return community_list


def _jaccard(a, b):
    return len(a & b) / len(a | b)


def max_jaccard(a, b):
    return max([_jaccard(a[i], b) for i in range(len(a))])


def jaccard(pre, true):
    sum_true = sum([max_jaccard(pre, true[i]) for i in range(len(true))])
    sum_pre = sum([max_jaccard(true, pre[i]) for i in range(len(pre))])
    jac = sum_true / 2 / len(true) + sum_pre / 2 / len(pre)
    return jac


def get_index_map(idx_list):
    index_map = {}
    for i, index in enumerate(idx_list):
        index_map[index] = i
    return index_map


def transformed_index_labels(idx_list, labels_list):
    index_map = get_index_map(idx_list)
    new_labels_list = []
    for community in labels_list:
        new_community = set([index_map[x] for x in community])
        new_labels_list.append(new_community)
    return new_labels_list


def draw(adj, pos=None, figsize=(10, 10), dpi=100, iterations=10):
    g = nx.from_numpy_matrix(adj)
    fig = plt.figure(figsize=figsize, facecolor=None, dpi=dpi)
    plt.axis("off")
    if pos is None:
        pos = nx.spring_layout(
            g, k=1 / np.sqrt(g.number_of_nodes()), iterations=iterations)
    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=50,
        node_color='#336699',
        alpha=1,
        linewidths=0,
        font_size=0)
    nx.draw_networkx_edges(g, pos, alpha=0.5, width=1, edge_color='#000000')
    plt.show()
    print(nx.info(g))
    return pos


def aggregator(adj, feats, mode='sum'):
    s_mode = sp.issparse(adj)
    if s_mode:
        feats_agg = sp.csr_matrix(feats.shape)
    else:
        feats_agg = np.zeros_like(feats)
    if mode == 'sum':
        feats_agg = adj.dot(feats)
        if s_mode:
            feats_agg.eliminate_zeros()
        return feats_agg
    for i, neiv in enumerate(adj):
        if s_mode:
            nei_idx = neiv.nonzero()[1]
        else:
            nei_idx = np.nonzero(neiv)[0]
        if not nei_idx.size:
            continue
        nei = feats[nei_idx]
        if mode == 'max':
            neif = np.max(nei, axis=0)
        if mode == 'mean':
            neif = np.mean(nei, axis=0)
        else:
            raise ValueError("aggregator mode: max|mean|sum")
        feats_agg[i] = neif
        if s_mode:
            feats_agg.eliminate_zeros()
    return feats_agg


def NMI(true,pred):
    return Metrics.normalized_mutual_info_score(true,pred)

def matched(true,pred):
    max_idx = max(max(true),max(pred))
    cm = Metrics.confusion_matrix(true,pred,labels=np.arange(0,max_idx+1))
    shifted_mat = np.zeros((cm.shape[0]*2,cm.shape[0] * 2))
    shifted_mat[:cm.shape[0],cm.shape[0]:] = cm
    g = nx.from_numpy_matrix(shifted_mat)
    match = nx.max_weight_matching(g)
    unmatched = set(np.arange(0,cm.shape[0]))
    label_map = {}
    for m in match:
        p,t = max(m),min(m)
        unmatched.remove(t)
        label_map[p] = t
    unmatched = list(unmatched)
    for i in range(cm.shape[0],cm.shape[0]*2):
        if not i in label_map:
            label_map[i] = unmatched[-1]
            unmatched.pop()
   
    for i in range(len(pred)):
        pred[i] = label_map[pred[i]+cm.shape[0]]
    return pred

def matched_cm(true,pred):
    max_idx = max(max(true),max(pred))
    pred = matched(true,pred,labels=np.arange(0,max_idx+1))
    cm = Metrics.confusion_matrix(true,pred)
    return cm

def matched_ac(true,pred):
    pred = matched(true,pred)
    ac = Metrics.accuracy_score(true,pred)
    return ac

def score(true,pred):
    nmi = NMI(true,pred)
    max_ac = matched_ac(true,pred)
    return max_ac,nmi

def time_log_wrapper(logging):
    def _time_log(func):
        def inner(*args, **kwargs):
            start = time.time()
            func(*args, **kwargs)
            print(logging + "({0:.6f}s)".format(time.time() - start))

        return inner

    return _time_log


def time_log(logging):
    def _time_log(func, *args, **kwargs):
        start = time.time()
        ans = func(*args, **kwargs)
        print(logging + "({0:.6f}s)".format(time.time() - start))
        return ans

    return _time_log


def to_one_hot(labels, num_categories=None, min=True):
    if num_categories is None:
        num_categories = max(labels) + 1
    one_hot_labels = np.zeros((labels.shape[0], num_categories))
    if min == True:
        one_hot_labels[np.arange(labels.shape[0]), labels - 1] = 1
    else:
        one_hot_labels[np.arange(labels.shape[0]), labels] = 1
    return one_hot_labels


def normalized_laplacian_sparse(adj, axis=0):
    epsilon = 1e-30
    D = degree_matrix_sparse(adj, axis=axis)
    data_tmp = np.array([0 if abs(x) < epsilon else 1 / np.sqrt(x) for x in D.data])
    D.data = data_tmp
    I = sp.eye(adj.shape[0])
    return I - D.dot(adj).dot(D)


def laplacian_sparse(adj, axis=0):
    D = sp.lil_matrix(np.diag(np.array(adj.sum(axis=axis)).flatten()))
    return D - adj


def svd(matrix, k):
    u, d, v = linalg.svds(matrix.asfptype(), k=k)
    u = sp.csr_matrix(u)
    row = np.arange(d.shape[0])
    d = sp.csr_matrix((d, (row, row)))
    v = sp.csr_matrix(v)
    return u * d.sqrt(), d.sqrt() * v


def svd_non_negative(matrix, k):
    if sp.issparse(matrix):
        u, v = svd(matrix, k)
    else:
        u, v = svd(sp.csr_matrix(matrix), k)
    u[u < 0] = 0
    v[v < 0] = 0
    u.eliminate_zeros()
    v.eliminate_zeros()
    if not sp.issparse(matrix):
        u = u.toarray()
        v = v.toarray()
    return u, v


def normalized_laplacian(adj, axis=0):
    D_sqrt_inv = np.diag(1 / (np.sqrt(np.array(np.sum(adj, axis=axis))).flatten()))
    I = np.eye(adj.shape[0])
    return I - D_sqrt_inv.dot(adj).dot(D_sqrt_inv)


def normalized_adjacency(adj, axis=0):
    D_sqrt_inv = np.diag(1 / (np.sqrt(np.array(np.sum(adj, axis=axis))).flatten()))
    return D_sqrt_inv.dot(adj).dot(D_sqrt_inv)


def laplacian(adj, axis=0):
    D = np.diag(np.array(adj.sum(axis=axis)).flatten())
    return D - adj


def degree_matrix(adj, axis=0):
    return np.diag(np.array(adj.sum(axis=axis)).flatten())


def degree_matrix_sparse(adj, axis=0):
    sum = np.array(adj.sum(axis=axis)).flatten()
    row = np.arange(sum.shape[0])
    diag = sp.csr_matrix((sum, (row, row)))
    return diag


def norm_sparse(mat, transpose=False, printStatus=False, printStep=10000):
    """normalize by column"""
    if transpose:
        mat = mat.transpose()
    mat = mat.tocsc()
    for i in range(mat.shape[1]):
        if printStatus and not i % printStep:
            print(i)
        n = sp.linalg.norm(mat[:, i])
        if abs(n) > 1e-14:
            n = 1 / n
        mat[:, i] *= n
    if transpose:
        mat = mat.transpose()
    return mat.tocsr()


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def get_graph(adj):
    '''
    get a graph from zero-padded adj
    :param adj:
    :return:
    '''
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G


# draw a list of graphs [G]
def draw_graph_list(G_list, row, col, fname='figures/test', layout='spring', is_single=False, k=1, node_size=55,
                    alpha=1, width=1.3):
    # # draw graph view
    # from pylab import rcParams
    # rcParams['figure.figsize'] = 12,3
    plt.switch_backend('agg')
    for i, G in enumerate(G_list):
        plt.subplot(row, col, i + 1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                            wspace=0, hspace=0)
        plt.axis("off")
        if layout == 'spring':
            pos = nx.spring_layout(G, k=k / np.sqrt(G.number_of_nodes()), iterations=100)
            # pos = nx.spring_layout(G)

        elif layout == 'spectral':
            pos = nx.spectral_layout(G)

        if is_single:
            # node_size default 60, edge_width default 1.5
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#336699', alpha=1, linewidths=0,
                                   font_size=0)
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes(G, pos, node_size=1.5, node_color='#336699', alpha=1, linewidths=0.2, font_size=1.5)
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

    plt.tight_layout()
    plt.savefig(fname + '.pdf')
    plt.close()


# save a list of graphs
def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pkl.dump(G_list, f)


def save_indices_list(indices_list, fname):
    with open(fname, "wb") as f:
        pkl.dump(indices_list, f)


# load a list of graphs
def load_graph_list(fname):
    with open(fname, "rb") as f:
        graph_list = pkl.load(f)
    for i in range(len(graph_list)):
        #######list
        edges_with_selfloops = list(graph_list[i].selfloop_edges())
        if len(edges_with_selfloops) > 0:
            graph_list[i].remove_edges_from(edges_with_selfloops)
    return graph_list


def load_index_list(fname):
    with open(fname, "rb") as f:
        index_list = pkl.load(f)
    return index_list


if __name__ == '__main__':
    pass
