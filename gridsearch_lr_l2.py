'''
Created on July 1, 2022
@author: heguangliang ( heguangliang@JiNan.edu.cn)
'''
__author__ = " heguangliang"

import random
from collections import defaultdict

import torch
import numpy as np

from time import time
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import sample_item, gridsearch_build_sparse_neibor_matrix, gridsearch_load_data
from modules.KGCNA import Recommender
from utils.evaluate import test

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_dict(train_entity_pairs, start, end, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    # args.epoch = 20
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    directory = args.data_path + args.dataset + '/'
    gridsearch_lr = eval(args.gridsearch_lr)
    gridsearch_l2 = eval(args.gridsearch_l2)
    user_neibor_size = eval(args.gridsearch_user_neibor_size)
    item_neibor_size = eval(args.gridsearch_item_neibor_size)
    user_sample_size = eval(args.gridsearch_user_sample_size)
    context = eval(args.gridsearch_context)
    super_param = defaultdict(list)
    start_time = time()
    train_cf, test_cf, user_dict, n_params, graph, norm_item_user_adj = gridsearch_load_data(args)

    test_sample_user_item = torch.LongTensor(
        sample_item(user_dict['train_user_set'], args.test_sample_item_size)).to(
        device)
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    n_params['test_sample_user_item'] = test_sample_user_item
    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    graph = torch.LongTensor(graph)
    for ct in context:
        for user_neibor in user_neibor_size:
            for item_neibor in item_neibor_size:
                """build sparse neibor matrix"""
                norm_user_neibor, norm_item_neibor = gridsearch_build_sparse_neibor_matrix(train_cf,
                                                                                           user_neibor,
                                                                                           item_neibor,
                                                                                           directory)
                for sample_size in user_sample_size:
                    for lr in gridsearch_lr:
                        for l2 in gridsearch_l2:

                            # n_params['test_sample_user_item'] = torch.LongTensor(
                            #     sample_item(user_dict['train_user_set'], sample_size)).to(
                            #     device)

                            # torch.cuda.empty_cache()
                            train_grid_start = time()
                            args.train_sample_item_size = sample_size
                            args.user_neibor_size = user_neibor
                            args.item_neibor_size = item_neibor
                            args.context_hops = ct
                            args.lr = lr
                            args.l2 = l2
                            print(
                                'super_param  context_hops: {}, user_neibor: {}, item_neibor: {}, sample_size: {}, '
                                'lr: {}, l2: {} '
                                    .format(ct, user_neibor, item_neibor, sample_size, lr, l2))

                            train_sample_item_size = args.train_sample_item_size
                            """define model"""
                            model = Recommender(n_params, user_dict, args, graph, norm_item_user_adj, norm_user_neibor,
                                                norm_item_neibor).to(device)
                            """define optimizer"""
                            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                            cur_best_pre_0 = 0
                            stopping_step = 0
                            should_stop = False
                            print("start training ...")
                            gridsearch_train_cf_pairs = train_cf_pairs.clone()

                            for epoch in range(args.epoch):
                                """training CF"""
                                # shuffle training data
                                index = np.arange(len(train_cf))
                                np.random.shuffle(index)
                                gridsearch_train_cf_pairs = gridsearch_train_cf_pairs[index]

                                """training"""
                                loss, s = 0, 0
                                totall_regularizer = 0
                                train_s_t = time()
                                train_sample_user_item = torch.LongTensor(
                                    sample_item(user_dict['train_user_set'], train_sample_item_size)).to(
                                    device)
                                while s + args.batch_size <= len(train_cf):
                                    batch = get_feed_dict(gridsearch_train_cf_pairs,
                                                          s, s + args.batch_size,
                                                          user_dict['train_user_set'])

                                    batch_loss, _, regularizer = model(batch, train_sample_user_item)
                                    with torch.autograd.set_detect_anomaly(True):
                                        optimizer.zero_grad()
                                        batch_loss.backward()
                                        optimizer.step()
                                    loss += batch_loss
                                    totall_regularizer += regularizer
                                    s += args.batch_size

                                train_e_t = time()
                                print('using time %.4f, training loss at epoch %d: %.4f, regularizer: %.6f' % (
                                    train_e_t - train_s_t, epoch, loss.item(), totall_regularizer.item()))

                            """testing"""

                            test_s_t = time()
                            ret = test(model, list(model.test_user_dict.keys()), user_dict, n_params)
                            test_e_t = time()
                            train_grid_end = time()
                            train_res = PrettyTable()
                            train_res.field_names = ["super_param: context_hops user_neibor item_neibor sample_size lr l2 ",
                                                     "training time", "tesing time", "Loss", "recall", "ndcg",
                                                     "precision", "hit_ratio", "mean Average Precision", "auc", "f1"]
                            train_res.add_row(
                                [(ct, user_neibor, item_neibor, sample_size, lr, l2), train_grid_end - train_grid_start,
                                 test_e_t - test_s_t, loss.item(), ret['recall'],
                                 ret['ndcg'],
                                 ret['precision'], ret['hit_ratio'], ret['map'], ret['auc'], ret['f1']]
                            )
                            print(train_res)
                            super_param[(ct, user_neibor, item_neibor, sample_size, lr, l2)] = [ret['recall'],
                                                                                        ret['ndcg'],
                                                                                        ret['precision'],
                                                                                        ret['hit_ratio'],
                                                                                        ret['map'], ret['auc'],
                                                                                        ret['f1']]
    super_param = sorted(super_param.items(), key=lambda x: x[1][0][0], reverse=True)
    gridsearch_res = PrettyTable()
    gridsearch_res.field_names = ["super_param: context_hops user_neibor item_neibor sample_size lr l2",
                                  "recall", "ndcg",
                                  "precision", "hit_ratio", "mean Average Precision", "auc", "f1"]
    for item in super_param:
        gridsearch_res.add_row(
            [(item[0][0], item[0][1], item[0][2], item[0][3], item[0][4], item[0][5]), item[1][0], item[1][1],
             item[1][2], item[1][3], item[1][4], item[1][5], item[1][6]]
        )
    print(gridsearch_res)
    end_time = time()
    print('gridsearch finished totall time %.4f' % (end_time - start_time))
