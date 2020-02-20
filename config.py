import sys
import os
import scipy.io as scio
from utils import *


class Config(object):
    def __init__(self):
        # self.gpu = True
        self.gpu = False

        self.dataset = None
        self.features_dimension = None
        self.number_categories = None

        self.max_previous_nodes = None
        self.max_num_nodes = None
        self.num_layers = None

        self.epochs = 1000
        self.batch_size = 3
        self.num_batches = 1
        self.epochs_log = 1
        self.batches_log = self.num_batches
        self.lr = 0.03
        self.milestones = [400, 1000]
        self.lr_rate = 0.3

        self.test_batch_size = 1
        self.test_epochs = 10
        self.epochs_test_start = 0
        self.test_total_size = 1

        self.root_path = './'
        self.update_path()

    def update_path(self, root_path=None):
        if not (root_path is None):
            self.root_path = root_path
        self.log_path = os.path.join(self.root_path, 'log')
        self.graph_path = os.path.join(self.root_path, 'graphs')
        self.checkpoint_path = os.path.join(self.root_path, 'checkpoint')
        self.data_path = os.path.join(self.root_path, 'data')

    def data_configure(self, dataset, Dataset, Dataset_test, iteration=10, **kwargs):
        self.dataset = dataset
        data = {}
        self.num_layers = 4
        self.max_previous_nodes = 100
        
        self.data_path = os.path.join(self.root_path,'data')
        dataset_file = os.path.join(self.data_path,self.dataset.title())
        data_raw = scio.loadmat(dataset_file)
        data = data_raw
        self.features_dimension = int(data['feat_dim'])
        self.number_categories = int(data['num_cate']) 
        self.max_num_nodes = int(data['num_node'])
        data_set = Dataset(data, max_num_node=self.max_num_nodes, max_prev_node=self.max_previous_nodes, iteration=10,
                           **kwargs)
        data_set_test = Dataset_test(data, max_num_node=self.max_num_nodes, max_prev_node=self.max_previous_nodes)
        sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
            [1.0 / len(data_set) for i in range(len(data_set))], num_samples=self.batch_size * self.num_batches,
            replacement=True)
        dataset_loader = torch.utils.data.DataLoader(data_set, batch_size=self.batch_size, num_workers=4,
                                                     sampler=sample_strategy)
        dataset_loader_test = torch.utils.data.DataLoader(data_set_test, batch_size=self.batch_size, num_workers=4,
                                                          sampler=sample_strategy)
        return {'train': dataset_loader, 'test': dataset_loader_test}

    def agrnn_configure(self, AGRNN, gpu=True, features=True):
        inputs_dim = self.max_previous_nodes
        inputs_dim = self.features_dimension
        agrnn = AGRNN(edge_inputs_dim=inputs_dim, edge_embedded_dim=128, edge_outputs_dim=16,
                            edge_hidden_dim=64, edge_num_layers=self.num_layers, node_inputs_dim=1,
                            node_embedded_dim=16,
                            node_num_layers=self.num_layers, gpu=gpu)
        if gpu:
            agrnn = agrnn.cuda()
        return agrnn

