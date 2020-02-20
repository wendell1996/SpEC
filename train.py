from model import *
from data import *
from utils import *


def train_epoch_features(epoch, config, model, dataset_loader, optimizer, scheduler, gpu=True):
    model.train()
    logging_train = ''
    for i, data in enumerate(dataset_loader):
        model.zero_grad()
        x = data['x'].float()
        y = data['y'].float()
        feats = data['features'].float()
        seq_len = data['len']
        seq_len, sort_index = torch.sort(seq_len, 0, descending=True)
        seq_len = seq_len.numpy().tolist()
        x = torch.index_select(x, 0, sort_index)
        y = torch.index_select(y, 0, sort_index)
        feats = torch.index_select(feats, 0, sort_index)
        if gpu:
            x = x.cuda()
            y = y.cuda()
            feats = feats.cuda()

        # reverse for aligning
        y = torch.nn.utils.rnn.pack_padded_sequence(y, seq_len, batch_first=True).data
        idx = [i for i in range(y.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        if gpu:
            idx = idx.cuda()
        y_reshape = y.index_select(0, idx)
        y_reshape = torch.unsqueeze(y_reshape, 2)
        ones = torch.ones(y_reshape.size(0), 1, 1)
        if gpu:
            ones = ones.cuda()
        x2 = torch.cat((ones, y_reshape[:, 0:-1, 0:1]), dim=1)

        model.edge.init_hidden(x.size(0), gpu)
        outs, seq_seq_len = model(feats, x2, sequence_len=seq_len)

        y_reshape = torch.nn.utils.rnn.pack_padded_sequence(y_reshape, seq_seq_len, batch_first=True)
        y_reshape = torch.nn.utils.rnn.pad_packed_sequence(y_reshape, batch_first=True)[0]
        outputs_p = torch.nn.functional.sigmoid(outs)
        loss = torch.nn.functional.binary_cross_entropy(outputs_p, y_reshape)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if not epoch % config.epochs_log and not i % config.batches_log:
            logging = 'epoch:{} batch:{} batch size:{}*{} loss:{}'.format(epoch, i, config.batch_size,
                                                                          config.num_batches, loss)
            print(logging)
            logging_train = logging
    return logging_train


def test_epoch_features(epoch, config, model, dataset_loader, test_batch_size, gpu):
    it = iter(dataset_loader)
    data = it.__next__()
    model.eval()
    y_pred_long = torch.zeros(test_batch_size, config.max_num_nodes, config.max_previous_nodes)  # discrete prediction
    feats = data['features'].float()
    indices = data['indices']
    seq_len = data['len']
    seq_len, sort_index = torch.sort(seq_len, 0, descending=True)
    seq_len = seq_len.numpy().tolist()
    feats = torch.index_select(feats, 0, sort_index)
    indices = torch.index_select(indices, 0, sort_index)
    model.edge.init_hidden(feats.size(0))
    if gpu:
        feats = feats.cuda()
        model.edge.init_hidden(feats.size(0), gpu)
    h = model.edge(feats, pack=True, sequence_len=seq_len)  # batch_size,seq_len,out_dim

    sequences_sequence_len = []
    sequence_len_bin = torch.bincount(torch.tensor(seq_len))
    max_num_previous_nodes = config.max_previous_nodes
    for i in range(len(sequence_len_bin) - 1, 0, -1):
        count_temp = torch.sum(sequence_len_bin[i:]).numpy()  # count how many y_len is above i
        sequences_sequence_len.extend([min(i, max_num_previous_nodes)] * int(count_temp))

    h = torch.nn.utils.rnn.pack_padded_sequence(h, seq_len, batch_first=True).data
    # reverse for aligning
    idx = [i for i in range(h.size(0) - 1, -1, -1)]
    idx = torch.LongTensor(idx)
    if gpu:
        idx = idx.cuda()
    h = h.index_select(0, idx)
    hidden_null = torch.zeros(config.num_layers - 1, h.size(0), h.size(1))
    if gpu:
        hidden_null = hidden_null.cuda()
    model.node.hidden = torch.cat((h.unsqueeze(0), hidden_null), dim=0)

    x_step = torch.zeros(h.size(0), config.max_previous_nodes, 1)
    output_x_step = torch.ones(h.size(0), 1, 1)
    if gpu:
        x_step = x_step.cuda()
        output_x_step = output_x_step.cuda()
    for j in range(config.max_previous_nodes):
        output_y_pred_step = model.node(output_x_step, pack=True, sequence_len=torch.ones(h.size(0), ).long())
        output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1, gpu=gpu)
        x_step[:, j:j + 1, :] = output_x_step
        # output.hidden = output.hidden.data
    y_pred_long = x_step
    y_pred_long = torch.nn.utils.rnn.pack_padded_sequence(y_pred_long, sequences_sequence_len, batch_first=True)
    y_pred_long = torch.nn.utils.rnn.pad_packed_sequence(y_pred_long, batch_first=True)[0]
    y_pred_long = torch.squeeze(y_pred_long, 2)
    # rnn.hidden = Variable(rnn.hidden.data)

    tmp_len = seq_len.copy()
    recover_len = []
    tmp_batch = len(tmp_len)
    for i in range(len(tmp_len) - 1, -1, -1):
        recover_len += [tmp_batch] * int(tmp_len[i])
        tmp_batch -= 1
        tmp_len = [x - tmp_len[i] for x in tmp_len]

    y_pred_long = torch.nn.utils.rnn.PackedSequence(y_pred_long, torch.LongTensor(recover_len))
    y_pred_long = torch.nn.utils.rnn.pad_packed_sequence(y_pred_long, batch_first=True)[0]

    for i, length in enumerate(seq_len):
        idx = [i for i in range(length - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        if gpu:
            idx = idx.cuda()
        y_pred_long[i, :] = y_pred_long[i, :].index_select(0, idx)
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    G_indices = []
    # adj = dataset_loader.dataset.data['adjacency_matrix'].toarray()
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i][1:, :].cpu().numpy())
        G_pred = nx.from_numpy_matrix(adj_pred)  # get a graph from zero-padded adj
        G_pred_list.append(G_pred)
        G_indices.append(indices[i])
    return G_pred_list, G_indices


def train_features(config, model, dataset_loader, gpu=True):
    '''
    x(batch_size,sequence_len,max_num_previous_nodes)
    x2(sequence_len_sum,max_num_previous_nodes,1)
    '''
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    optimizer = torch.optim.Adam(list(model.parameters()), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.lr_rate)
    model.train()

    for epoch in range(config.epochs + 1):
        start = time.time()
        logging_train = train_epoch_features(epoch, config, model, dataset_loader['train'], optimizer, scheduler, gpu)
        logging = "epoch:{}/{} train time:{:.6f}sec".format(epoch, config.epochs, time.time() - start)
        log_time = open("{}/log_time_{}_{}.txt".format(config.log_path, config.dataset, localtime), 'a')
        log_time.write(logging_train + '\n')
        log_time.write(logging + '\n')
        log_time.close()
        print(logging)
        if not epoch % config.test_epochs and epoch >= config.epochs_test_start:
            G_pred = []
            G_idx = []
            start = time.time()
            while len(G_pred) < config.test_total_size:
                G_pred_step, indices = test_epoch_features(epoch, config, model, dataset_loader['test'],
                                                           config.test_batch_size,
                                                           gpu)
                G_pred.extend(G_pred_step)
                G_idx.extend(indices)
            # save graphs
            fname = "{}/{}_feat_{}_{}.dat".format(config.graph_path, config.dataset, config.max_previous_nodes, epoch)
            save_graph_list(G_pred, fname)
            fname_idx = "{}/{}_idx_{}_{}.dat".format(config.graph_path, config.dataset, config.max_previous_nodes,
                                                     epoch)
            save_indices_list(G_idx, fname_idx)
            logging_test = 'test done, graphs saved in {},cost {:6f} seconds'.format(fname, time.time() - start)
            log_time = open("{}/log_time_{}_{}.txt".format(config.log_path, config.dataset, localtime), 'a')
            log_time.write(logging_test + '\n')
            log_time.close()
            print(logging_test)
