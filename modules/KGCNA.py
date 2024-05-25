'''
Created on dec 15, 2022
PyTorch Implementation of KGCA
@author: Heguangliang
'''
__author__ = "Heguangliang "

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from utils.data_loader import read_cf
from utils.helper import ensureDir


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """

    def __init__(self, n_item, channel=64):
        super(Aggregator, self).__init__()
        self.n_item = n_item
        self.emb_size = channel
        initializer = nn.init.xavier_uniform_
        user_attend_linear = nn.Linear(self.emb_size, self.emb_size)

        initializer(user_attend_linear.weight)

        item_kg_gate_linear = nn.Linear(self.emb_size, self.emb_size)
        item_neibor_gate_linear = nn.Linear(self.emb_size, self.emb_size)
        initializer(item_kg_gate_linear.weight)
        initializer(item_neibor_gate_linear.weight)
        user_item_gate_linear = nn.Linear(self.emb_size, self.emb_size)
        user_neibor_gate_linear = nn.Linear(self.emb_size, self.emb_size)
        initializer(user_item_gate_linear.weight)
        initializer(user_neibor_gate_linear.weight)

        self.user_attend_linear = user_attend_linear
        self.item_kg_gate_linear = item_kg_gate_linear
        self.item_neibor_gate_linear = item_neibor_gate_linear
        self.user_item_gate_linear = user_item_gate_linear
        self.user_neibor_gate_linear = user_neibor_gate_linear
        self.user_activate = nn.Tanh()
        self.item_gate_func = nn.Sigmoid()
        self.user_gate_func = nn.Sigmoid()

    def forward(self, entity_emb, user_emb, edge_index,
                edge_type, weight, norm_item_user_adj, norm_user_neibor, norm_item_neibor, sample_user_item
                ):
        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        kg_entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        item_emb = entity_emb[:self.n_item, :]
        """colaborater neibor aggregate"""
        Aggregator_item_neibor = torch.sparse.mm(norm_item_neibor, item_emb)
        """item_base gate aggregate"""

        item_score = self.item_gate_func(
            self.item_kg_gate_linear(kg_entity_agg[:self.n_item, :]) + self.item_neibor_gate_linear(
                Aggregator_item_neibor))
        item_agg = item_score * kg_entity_agg[:self.n_item, :] + (1 - item_score) * Aggregator_item_neibor
        entity_agg_emb = torch.cat([item_agg, kg_entity_agg[self.n_item:, :]], dim=0)
        """ user aggregate"""

        user_item_emb = item_emb[sample_user_item]  # [n_users, n_sample_item_size, channel]
        user_item_attend_emb = self.user_activate(
            self.user_attend_linear(user_item_emb))  # [n_users, n_sample_item_size, channel]
        user_score = torch.bmm(user_item_attend_emb, user_emb.unsqueeze(dim=-1)).squeeze(
            dim=-1)  # [n_users, n_sample_item_size]
        norm_score = torch.softmax(user_score, dim=1)

        norm_score = norm_score.unsqueeze(dim=1)  # [n_users, 1,n_sample_item_size]
        user_item_agg = torch.bmm(norm_score, user_item_emb).squeeze(dim=1)  # [n_users, channel]
        user_collaborate_emb = torch.sparse.mm(norm_user_neibor, user_emb)
        user_gate_score = self.user_gate_func(
            self.user_item_gate_linear(user_item_agg) + self.user_neibor_gate_linear(user_collaborate_emb))
        user_agg_emb = user_gate_score * user_item_agg + (1 - user_gate_score) * user_collaborate_emb
        # user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg  # [n_users, channel]
        regularizer_loss = torch.norm(self.user_attend_linear.weight) ** 2 + torch.norm(
            self.user_attend_linear.bias) ** 2 + torch.norm(self.item_kg_gate_linear.weight) ** 2 + torch.norm(
            self.item_kg_gate_linear.bias) ** 2 + torch.norm(self.item_neibor_gate_linear.weight) ** 2 + torch.norm(
            self.item_neibor_gate_linear.bias) ** 2 + torch.norm(self.user_item_gate_linear.weight) ** 2 + torch.norm(
            self.user_item_gate_linear.bias) ** 2 + torch.norm(self.user_neibor_gate_linear.weight) ** 2 + torch.norm(
            self.user_neibor_gate_linear.bias) ** 2
        return entity_agg_emb, user_agg_emb, regularizer_loss


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_users, n_items,
                 n_relations, norm_user_neibor, norm_item_neibor, node_dropout_rate=0.5,
                 mess_dropout_rate=0.1):  # channel enbeding size
        super(GraphConv, self).__init__()
        self.convs = nn.ModuleList()
        self.norm_user_neibor = norm_user_neibor
        self.norm_item_neibor = norm_item_neibor
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items

        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]#weight 关系的embeding

        for i in range(n_hops):
            self.convs.append(
                Aggregator(channel=channel, n_item=n_items))
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * (1 - rate)), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    # ?
    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_emb, entity_emb, edge_index, edge_type,

                norm_item_user_adj, sample_user_item, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            norm_item_user_adj = self._sparse_dropout(norm_item_user_adj, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        totall_regularizer_loss = 0
        for i in range(len(self.convs)):
            entity_emb, user_emb, regularizer_loss = self.convs[i](entity_emb, user_emb, edge_index, edge_type,
                                                                   self.weight,
                                                                   norm_item_user_adj, self.norm_user_neibor,
                                                                   self.norm_item_neibor,
                                                                   sample_user_item)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
            totall_regularizer_loss += regularizer_loss

        totall_regularizer_loss += torch.norm(self.weight) ** 2

        return entity_res_emb, user_res_emb, totall_regularizer_loss


class Recommender(nn.Module):
    def __init__(self, data_config, user_dict, args_config, graph, norm_item_user_adj,
                 norm_user_neibor, norm_item_neibor):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.test_user_dict = user_dict['test_user_set']
        self.train_user_dict = user_dict['train_user_set']
        self.n_train = data_config['n_train']
        self.n_test = data_config['n_test']
        self.path = args_config.data_path + args_config.dataset
        self.fold = args_config.fold
        self.pretrain_data = data_config['pretrain_data']

        self.norm_item_user_adj = norm_item_user_adj
        self.norm_user_neibor = norm_user_neibor
        self.norm_item_neibor = norm_item_neibor

        self.decay = args_config.l2

        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops

        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate

        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.graph = graph

        self.edge_index, self.edge_type = self._get_edges(graph)  # slow

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        if self.pretrain_data is None:
            self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        else:
            user_emb = torch.tensor(self.pretrain_data['user_embed'], requires_grad=True)
            item_emb = torch.tensor(self.pretrain_data['item_embed'], requires_grad=True)
            other_embed = initializer(torch.empty(self.n_nodes - self.n_users - self.n_items, self.emb_size))
            self.all_embed = torch.cat((user_emb, item_emb, other_embed), dim=0)
        # [n_users, n_entities]
        self.norm_item_user_adj = self._convert_sp_mat_to_sp_tensor(self.norm_item_user_adj).to(self.device)
        self.norm_user_neibor = self._convert_sp_mat_to_sp_tensor(self.norm_user_neibor).to(self.device)
        self.norm_item_neibor = self._convert_sp_mat_to_sp_tensor(self.norm_item_neibor).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_items=self.n_items,
                         n_relations=self.n_relations,
                         norm_user_neibor=self.norm_user_neibor,
                         norm_item_neibor=self.norm_item_neibor,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate
                         )

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph_tensor):
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, batch=None, train_sample_user_item=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        entity_gcn_emb, user_gcn_emb, totall_regularizer_loss = self.gcn(user_emb,
                                                                         item_emb,
                                                                         self.edge_index,
                                                                         self.edge_type,
                                                                         norm_item_user_adj=self.norm_item_user_adj,
                                                                         sample_user_item=train_sample_user_item,
                                                                         mess_dropout=self.mess_dropout,
                                                                         node_dropout=self.node_dropout
                                                                         )

        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]

        return self.create_bpr_loss(u_e, pos_e, neg_e, totall_regularizer_loss)

    def generate(self, test_sample_user_item):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb, totall_regularizer_loss = self.gcn(user_emb,
                                                                         item_emb,
                                                                         self.edge_index,
                                                                         self.edge_type,
                                                                         norm_item_user_adj=self.norm_item_user_adj,
                                                                         sample_user_item=test_sample_user_item,
                                                                         mess_dropout=False,
                                                                         node_dropout=False)

        return user_gcn_emb, entity_gcn_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items, totall_regularizer_loss):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (totall_regularizer_loss + torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')
            f.close()
        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_user_dict.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_user_dict[uid]
            test_iids = self.test_user_dict[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            elif idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state

    def get_popularity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/' + str(self.fold) + '_user_subset' + '/popularity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get {}_user_subset popularity split.'.format(self.fold))

        except Exception:
            split_uids, split_state = self.create_popularity_split()
            path = self.path + '/' + str(self.fold) + '_user_subset' + '/popularity.split'
            ensureDir(path)
            f = open(path, 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create {}_user_subset popularity split.'.format(self.fold))
            f.close()

        return split_uids, split_state

    def create_popularity_split(self):
        def statistic_popularity(users_list):
            item_list = []
            for uid in users_list:
                item_list = np.concatenate((item_list, self.train_user_dict[uid]))
            item_list = list(item_list.astype(np.int32))
            average_popularity = int(np.mean(item_popularity[item_list]))
            max_popularity = np.max(item_popularity[item_list])
            min_popularity = np.min(item_popularity[item_list])
            return average_popularity, max_popularity, min_popularity

        all_users_to_test = list(self.test_user_dict.keys())
        user_n_iid = dict()
        train_cf = read_cf(self.path + '/train.txt')

        vals = [1.] * len(train_cf)
        adj = sp.coo_matrix((vals, (train_cf[:, 0], train_cf[:, 1])), shape=(self.n_users, self.n_items))
        item_popularity = np.array(adj.sum(0)).squeeze(axis=0)

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_user_dict[uid]
            # test_iids = self.test_user_dict[uid]

            average_popularity = int(np.mean(item_popularity[train_iids]))

            if average_popularity not in user_n_iid.keys():
                user_n_iid[average_popularity] = [uid]
            else:
                user_n_iid[average_popularity].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []

        fold = self.fold
        rate = 1 / fold
        n_count = self.n_users
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += len(user_n_iid[n_iids])
            n_count -= len(user_n_iid[n_iids])

            if n_rates >= rate * (self.n_users):
                split_uids.append(temp)
                average_popularity, max_popularity, min_popularity = statistic_popularity(temp)
                state = '# average popularity of items: [%d], min popularity of items: [%d], max popularity of items: ' \
                        '[%d], #users=[%d]' % (
                            average_popularity, min_popularity, max_popularity, n_rates)
                split_state.append(state)
                print(state)
                temp = []
                n_rates = 0
            elif idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)
                average_popularity, max_popularity, min_popularity = statistic_popularity(temp)
                state = '# average popularity of items: [%d], min popularity of items: [%d], max popularity of items: ' \
                        '[%d], #users=[%d]' % (
                            average_popularity, min_popularity, max_popularity, n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state
