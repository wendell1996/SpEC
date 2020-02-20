import torch


class AGRNNCell(torch.nn.Module):
    def __init__(self, inputs_dim, embedded_dim, outputs_dim, hidden_dim, num_layers):
        super(AGRNNCell, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(inputs_dim, embedded_dim),
            torch.nn.ReLU()
        )
        self.rnn = torch.nn.GRU(embedded_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, embedded_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedded_dim, outputs_dim)
        )

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param, gain=torch.nn.init.calculate_gain('sigmoid'))
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = torch.nn.init.xavier_uniform_(module.weight.data,
                                                                   gain=torch.nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size, gpu=False):
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        if gpu:
            self.hidden = self.hidden.cuda()

    def forward(self, inputs, pack=True, sequence_len=None):
        outputs_emb = self.linear1(inputs)
        if pack:
            if sequence_len is None:
                raise ValueError("sequence_len is None")
            outputs_emb = torch.nn.utils.rnn.pack_padded_sequence(outputs_emb, sequence_len, batch_first=True)
        if self.hidden is None:
            raise ValueError("hidden is not initialized. Invoke init_hidden(batch_size).")
        outputs_raw, self.hidden = self.rnn(outputs_emb, self.hidden)
        if pack:
            outputs_raw, sequence_len = torch.nn.utils.rnn.pad_packed_sequence(outputs_raw, batch_first=True)
        outputs = self.linear2(outputs_raw)
        return outputs


class AGRNN(torch.nn.Module):
    def __init__(self, edge_inputs_dim, edge_embedded_dim, edge_outputs_dim, edge_hidden_dim, edge_num_layers,
                 node_inputs_dim, node_embedded_dim, node_num_layers, node_hidden_dim=None, node_outputs_dim=None,
                 gpu=False):
        super(AGRNN, self).__init__()
        self.gpu = gpu
        if node_hidden_dim is None:
            node_hidden_dim = edge_outputs_dim
        if node_outputs_dim is None:
            node_outputs_dim = node_inputs_dim
        self.node_num_layers = node_num_layers
        ####edge-level
        self.edge = AGRNNCell(edge_inputs_dim, edge_embedded_dim, edge_outputs_dim, edge_hidden_dim, edge_num_layers)

        ####node-level
        self.node = AGRNNCell(node_inputs_dim, node_embedded_dim, node_outputs_dim, node_hidden_dim, node_num_layers)

    def forward(self, inputs_features, inputs_sequences=None, pack=True, sequence_len=None):
        ####edge-level
        outputs_edge = self.edge(inputs_features, pack, sequence_len)

        ####node-level
        sequences_sequence_len = []
        sequence_len_bin = torch.bincount(torch.tensor(sequence_len))
        max_num_previous_nodes = inputs_sequences.size(1)
        for i in range(len(sequence_len_bin) - 1, 0, -1):
            count_temp = torch.sum(sequence_len_bin[i:]).numpy()  # count how many y_len is above i
            sequences_sequence_len.extend([min(i, max_num_previous_nodes)] * int(count_temp))

        outputs_edge = torch.nn.utils.rnn.pack_padded_sequence(outputs_edge, sequence_len, batch_first=True).data
        # reverse for aligning
        indices = [i for i in range(outputs_edge.size(0) - 1, -1, -1)]
        indices = torch.LongTensor(indices)
        if self.gpu:
            indices = indices.cuda()
        outputs_edge = outputs_edge.index_select(0, indices)
        hidden_null = torch.zeros(self.node_num_layers - 1, outputs_edge.size(0), outputs_edge.size(1))
        if self.gpu:
            hidden_null = hidden_null.cuda()
        self.node.hidden = torch.cat((outputs_edge.unsqueeze(0), hidden_null), dim=0)
        outputs = self.node(inputs_sequences, pack, sequences_sequence_len)
        # clean
        outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, sequences_sequence_len, batch_first=True)
        outputs = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)[0]

        return outputs, sequences_sequence_len


def sample_sigmoid(y, sample, thresh=0.5, sample_time=2, gpu=False):
    '''
    form graphrnn
        do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y = torch.sigmoid(y)
    # do sampling
    if sample:
        if sample_time > 1:
            y_result = torch.rand(y.size(0), y.size(1), y.size(2))
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = torch.rand(y.size(1), y.size(2))
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data > 0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = torch.rand(y.size(0), y.size(1), y.size(2))
            if gpu:
                y_thresh = y_thresh.cuda()
            y_result = torch.gt(y, y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = torch.ones(y.size(0), y.size(1), y.size(2)) * thresh
        y_result = torch.gt(y, y_thresh).float()
    return y_result


if __name__ == "__main__":
    pass
