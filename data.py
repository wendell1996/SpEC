from torch.utils.data import Dataset as dt
import numpy as np
import networkx as nx
import torch


#bfs_seq from GraphRNN
def bfs_seq(G, start_id):
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next = next + neighbor
        output = output + next
        start = next
    return output


def encode_adj(adj, max_prev_node=10, is_full=False):
    '''
    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0] - 1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n - 1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i, :] = adj_output[i, :][::-1]  # reverse order

    return adj_output


def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i, ::-1][output_start:output_end]  # reverse order
    adj_full = np.zeros((adj_output.shape[0] + 1, adj_output.shape[0] + 1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n - 1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


def encode_adj_flexible(adj):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n - 1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end - len(adj_slice) + np.amin(non_zero)

    return adj_output


class Dataset(dt):
    def __init__(self, data, max_num_node=None, max_prev_node=None, iteration=10):
        '''
        ['adjacency_matrix', 'features_all', 'labels_all', 'labels_train', 'labels_validation',
                          'labels_test', 'mask_train', 'mask_validation', 'mask_test']
        '''

        self.data = data
        self.adj_all = []
        self.len_all = []
        G_list = []
        G_list.append(data['adjacency_matrix'].toarray())
        for G in G_list:
            self.adj_all.append(G)
            self.len_all.append(G.shape[0])
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            print('calculating max previous node, total iteration: {}'.format(iteration))
            self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            print('max previous node: {}'.format(self.max_prev_node))
        else:
            self.max_prev_node = max_prev_node

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        sparse = 5
        adj_copy = self.adj_all[idx].copy()
        feats_copy = (self.data['features_all'].toarray()).copy()
        x_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        x_batch[0, :] = 1  # the first input token is all ones
        y_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        feats_batch = np.zeros((self.n, feats_copy.shape[1]))  # here zeros are padded for small graph
        # generate input x, y pairs
        idx = np.arange(0, adj_copy.shape[0])
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_matrix(adj_copy_matrix)
        idx = idx[x_idx]
        feats_copy = feats_copy[x_idx]
        # then do bfs in the permuted G
        len_batch = 0
        while len_batch < sparse:
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            len_batch = len(x_idx)
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        feats_copy = feats_copy[x_idx]
        adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)
        idx = idx[x_idx]
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        feats_batch[0:feats_copy.shape[0], :] = feats_copy
        return {'x': x_batch, 'y': y_batch, 'len': len_batch, 'features': feats_batch, 'indices': idx}

    def calc_max_prev_node(self, iter=10, topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print('iter {} times'.format(i))
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-1 * topk:]
        return max_prev_node


class Dataset_test(dt):
    def __init__(self, data, max_num_node=None, max_prev_node=None):
        '''
        ['adjacency_matrix', 'features_all', 'labels_all', 'labels_train', 'labels_validation',
                          'labels_test', 'mask_train', 'mask_validation', 'mask_test']
        '''

        self.data = data
        self.adj_all = []
        self.len_all = []
        G_list = []
        G_list.append(data['adjacency_matrix'].toarray())
        for G in G_list:
            self.adj_all.append(G)
            self.len_all.append(G.shape[0])
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            raise ValueError('max_prev_node is None')
        else:
            self.max_prev_node = max_prev_node

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        feats_copy = (self.data['features_all'].toarray()).copy()
        feats_batch = np.zeros((self.n, feats_copy.shape[1]))  # here zeros are padded for small graph
        # generate input x, y pairs
        idx = np.arange(0, adj_copy.shape[0])
        x_idx = np.random.permutation(adj_copy.shape[0])
        idx = idx[x_idx]
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        start_idx = np.random.randint(adj_copy.shape[0])
        x_idx = np.array(bfs_seq(G, start_idx))
        idx_new = np.arange(0, adj_copy.shape[0])
        oth_idx = np.array([x for x in idx_new if x not in x_idx])
        if (oth_idx != []):
            oth_idx = idx[oth_idx]
        idx = idx[x_idx]
        idx = np.concatenate((idx, oth_idx), axis=0)
        len_batch = len(idx)
        idx = idx.astype(np.int64)
        feats_copy = feats_copy[idx]
        # get x and y and adj
        # for small graph the rest are zero padded
        feats_batch[0:feats_copy.shape[0], :] = feats_copy
        return {'len': len_batch, 'features': feats_batch, 'indices': idx}
