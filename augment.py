from config import *
import data
from copy import deepcopy
from train import *


def graph_gen(config, dataset_loader):
    if not os.path.isdir(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.isdir(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)
    if not os.path.isdir(config.graph_path):
        os.makedirs(config.graph_path)

    gen = config.agrnn_configure(AGRNN, gpu=config.gpu, features=True)
    train_features(config, gen, dataset_loader, gpu=config.gpu)


def augmented_graph(config, dataset_loader, epoch=50, sparse=2, pendant=0, generate=True, test=False,
                    draw_graph=False):
    config.epochs = epoch
    if not test:
        config.test_epochs = epoch
    if generate:
        graph_gen(config, dataset_loader)
    dataset = dataset_loader['train'].dataset.data
    original_adjacency_matrix = dataset['adjacency_matrix'].toarray().astype(np.float64)
    original_graph = nx.from_numpy_matrix(original_adjacency_matrix)
    synthetic_graph = read_synthetic_graph(config, epoch)
    synthetic_adjacency_matrix = nx.to_numpy_array(synthetic_graph)
    sparse_index = []
    check = np.zeros(config.max_num_nodes)
    for idx in range(config.max_num_nodes):
        if check[idx] == 1:
            continue
        bfs_idx = np.array(data.bfs_seq(original_graph, idx))
        check[bfs_idx] = 1
        if bfs_idx.shape[0] <= sparse:
            sparse_index.extend(bfs_idx)
    if pendant > 0:
        degree_vector = original_adjacency_matrix.sum(axis=1)
        dependent_index = np.where(degree_vector <= pendant)[0]
        sparse_index.extend(dependent_index)
        sparse_index = np.unique(sparse_index)
    matrix_index = np.ix_(sparse_index, sparse_index)
    augmented_adjacency_matrix = deepcopy(original_adjacency_matrix)
    # augmented_adjacency_matrix[matrix_index] = 0
    zero_index = np.where(augmented_adjacency_matrix[matrix_index] == 0)
    augmented_slice = augmented_adjacency_matrix[matrix_index]
    augmented_slice[zero_index] += synthetic_adjacency_matrix[matrix_index][zero_index]
    augmented_adjacency_matrix[matrix_index] = augmented_slice
    if draw_graph:
        pos = draw(augmented_adjacency_matrix)
        draw(synthetic_adjacency_matrix, pos)
        draw(original_adjacency_matrix, pos)
    return augmented_adjacency_matrix


def augmented_attributes(adjacency_matrix, features, diffusion, mode='sum', normalize=False):
    if sp.issparse(adjacency_matrix):
        adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])
    else:
        adjacency_matrix = adjacency_matrix + np.eye(adjacency_matrix.shape[0])
    for i in range(diffusion):
        features += aggregator(adjacency_matrix, features, mode)
    if normalize:
        features = normalize(features, axis=1)
    return adjacency_matrix, features


def read_synthetic_graph(config, epoch):
    graph_file_name = "{}/{}_feat_{}_{}.dat".format(config.graph_path, config.dataset, config.max_previous_nodes, epoch)
    index_file_name = "{}/{}_idx_{}_{}.dat".format(config.graph_path, config.dataset, config.max_previous_nodes, epoch)
    graphs = load_graph_list(graph_file_name)
    indices = load_index_list(index_file_name)
    idx = 0
    adj = recover_adjacency_matrix(nx.to_numpy_array(graphs[idx]), indices[idx])
    return nx.from_numpy_matrix(adj)


def recover_adjacency_matrix(adj, idx):
    idx_mask = np.bincount(idx)
    ridx = np.nonzero(idx_mask)[0]
    return adj[np.ix_(ridx, ridx)]


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # torch.manual_seed(123)
    # random.seed(123)
    # np.random.seed(123)

    config = Config()
    dataset_name = 'cornell'
    dataset_loader = config.data_configure(dataset_name, data.Dataset, data.Dataset_test)
    # graph_gen(config, dataset_loader)
    augmented_graph(config, dataset_loader, epoch=50, sparse=5, pendant=1, generate=False, test=False,
                    draw_graph=True)
    pass
