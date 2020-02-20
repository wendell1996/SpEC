from utils import *
from config import *
from embedding_model import *
import data
from augment import *
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import sys
import scipy.sparse as sp
import scipy.io as scio
from sklearn.decomposition import NMF
from sklearn.metrics import jaccard_similarity_score
import os

parser = argparse.ArgumentParser()
parser.add_argument('--generate', '-g', action='store_false', default=True, dest='generate', help='Generate graph')
parser.add_argument('--generate_iter', action='store_false', default=True, dest='generate_iter',
                    help='Generate graph every 10 epochs')
parser.add_argument('--cpu', action='store_true', default=False, dest='cpu', help='Train on GPU')
parser.add_argument('-t', action='store', default=2, dest='diffusion', help='Hyperparameter \'t\' of node augmentation',
                    type=int)
parser.add_argument('--dataset_name_list', '-d', action='store', nargs='+', default=[],
                    dest='dataset_name_list', help='Dataset name', type=str)
parser.add_argument('--s_store', action='store_true', default=False, dest='s_store')
parser.add_argument('--epoch', '-e', action='store', default=50, dest='epoch', help='Train epoch', type=int)
parser.add_argument('--sparse', '-s', action='store', default=5, dest='sparse',
                    help='Node number of a sparse subgraph', type=int)
parser.add_argument('--pendant', '-p', action='store', default=1, dest='pendant', help='Node degree of graph', type=int)
parser.add_argument('--custom', action='store_true', default=False, dest='custom', help='Custom initial')
parser.add_argument('--beta', '-b', action='store', default=1, dest='beta', help='Beta', type=float)
parser.add_argument('--alpha', '-a', action='store', default=0.5, dest='alpha', help='Alpha', type=float)
parser.add_argument('--iteration', '-i', action='store', default=100, dest='iteration', help='Max iteration', type=int)
parser.add_argument('--overlapping', '-o', action='store_true', default=False, dest='overlapping',
                    help='Overlapping community detection')
parser.add_argument('--show', action='store_true', default=False, dest='show', help='Show images and reports')
parser.add_argument('--show_iter', action='store_true', default=False, dest='show_iter',
                    help='Show images and reports every iteration')
parser.add_argument('--log_iter', action='store', default=50, dest='log_iter', help='log iterations', type=int)
parser.add_argument('--root', '-r', action='store', default='./',
                    dest='root', help='Root path')
parser.add_argument('--repeat', action='store', default=10, dest='repeat', help='Repeated times', type=int)

argument = parser.parse_args()
generate = argument.generate
generate_iter = argument.generate_iter
gpu = not argument.cpu
if gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
diffusion = argument.diffusion # t=0 for website network, 1 for social network, 2 for citation network
dataset_name_list = argument.dataset_name_list
s_store = argument.s_store
epoch = argument.epoch
sparse = argument.sparse
pendant = argument.pendant
custom = argument.custom
beta = argument.beta
alpha = argument.alpha
iteration = argument.iteration
overlapping = argument.overlapping
show = argument.show
root_path = argument.root
show_iter = argument.show_iter
log_iter = argument.log_iter
repeat = argument.repeat

config = Config()

config.gpu = gpu
config.update_path(root_path)

for dataset_name in dataset_name_list:
    print(dataset_name)
    dataset_loader = config.data_configure(dataset_name, data.Dataset, data.Dataset_test)
    dataset = dataset_loader['train'].dataset.data
    adjacency_matrix = dataset['adjacency_matrix']
    features = dataset['features_all']
    if not s_store:
        adjacency_matrix = adjacency_matrix.toarray()
        features = features.toarray()
    if not overlapping:
        labels_all = np.argmax(dataset['labels_all'], axis=1)
    else:
        labels_all = dataset['labels_all']
    augmented_adjacency_matrix = augmented_graph(config, dataset_loader, epoch=epoch, sparse=sparse,
                                                     pendant=pendant, generate=generate, test=generate_iter,
                                                     draw_graph=show)
    augmented_adjacency_matrix_identity, features = augmented_attributes(augmented_adjacency_matrix, features,
                                                                             diffusion=diffusion)
    A = augmented_adjacency_matrix_identity
    X = features

    if not overlapping:
        ac_list = []
        nmi_list = []
    else:
        jaccard_list = []
        f1_list = []

    for i in range(repeat):
        if show_iter:
            print('Initilizing...')
        if custom:
            U, C = svd_non_negative(X, config.number_categories)
            nmf = NMF(n_components=config.number_categories, init='custom', max_iter=200, tol=1e-7)
            if s_store:
                U = U.toarray()
                C = C.toarray()
            U = nmf.fit_transform(X, W=U.astype(np.float64), H=C.astype(np.float64))
        else:
            nmf = NMF(n_components=config.number_categories, max_iter=200, tol=1e-7)
            U = nmf.fit_transform(X)
        C = nmf.components_
        if s_store:
            U, C = sp.csr_matrix(U), sp.csr_matrix(C)
        if show_iter:
            print('Factorizing...')
        U, C = sparse_embedding_iteration(A, X, U, C, beta, alpha, iteration, log=show_iter, log_iter=log_iter)
        if not overlapping:
            predicted_labels = non_overlapping_detection(U)
            ac, nmi = score(labels_all, predicted_labels)
            ac_list.append(ac)
            nmi_list.append(nmi)
        else:
            predicted_matrix = overlapping_detection(U, 0.1)
            predicted_community = get_community(predicted_matrix)
            expected_community = get_community(labels_all)
            f1_score = f1(predicted_community, expected_community)
            jaccard_score = jaccard(predicted_community, expected_community)
            f1_list.append(f1_score)
            jaccard_list.append(jaccard_score)

    if not overlapping:
        print("AVG_AC:{} AVG_NMI:{}".format(np.array(ac_list).mean(), np.array(nmi_list).mean()))
    else:
        print("AVG_F1:{} AVG_Jac:{}".format(np.array(f1_list).mean(), np.array(jaccard_list).mean()))
