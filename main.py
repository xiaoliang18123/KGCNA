'''
Created on July 1, 2020
@author: HeGuangLing (HeGuangLing.he@jnu.edu.cn)
'''
__author__ = "HeGuangLing"

import random

import torch
import numpy as np

from time import time
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data, sample_item
from modules.KGCNA import Recommender
from utils.evaluate import test
from utils.helper import early_stopping, ensureDir

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
    loss_list=[]
    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    norm_item_user_adj, norm_user_neibor, norm_item_neibor = mat_list
    test_sample_user_item = torch.LongTensor(sample_item(user_dict['train_user_set'], args.test_sample_item_size)).to(
        device)
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    n_params['test_sample_user_item'] = test_sample_user_item
    train_sample_item_size = args.train_sample_item_size
    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    graph = torch.LongTensor(graph)

    """define model"""
    model = Recommender(n_params, user_dict, args, graph, norm_item_user_adj, norm_user_neibor,
                        norm_item_neibor).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.save:
        # save_path = args.out_dir + args.model_type + '/' + args.dataset + '.ckpt'
        save_path = '{}{}/{}_u{}_i{}_train_sp{}_test_sp{}_lr{}_l2{}_layer{}.ckpt'.format(args.out_dir, args.model_type,
                                                                                         args.dataset,
                                                                                         args.user_neibor_size,
                                                                                         args.item_neibor_size,
                                                                                         args.train_sample_item_size,
                                                                                         args.test_sample_item_size,
                                                                                         args.lr,
                                                                                         args.l2,
                                                                                         args.context_hops)
        ensureDir(save_path)

    stopping_step = 0
    if args.pretrain == 1:
        model.load_state_dict(torch.load(save_path))
        if args.report == 1:
            print("report differ sparsity level performance")
            users_to_test_list, split_state = model.get_sparsity_split()

            users_to_test_list.append(list(model.test_user_dict.keys()))
            split_state.append('all')

            save_path = '%sreport/%s/%s.sparsity_result' % (args.proj_path, args.dataset, args.model_type)
            ensureDir(save_path)
            f = open(save_path, 'w')
            f.write(
                'embed_size=%d, lr=%.4f, l2_weight=%.5f, \n' % (args.dim, args.lr, args.l2))
            for i, users_to_test in enumerate(users_to_test_list):
                ret = test(model, users_to_test, user_dict, n_params)

                final_perf = "recall=[%s], ndcg=[%s], precision=[%s], hit=[%s]" % \
                             ('\t'.join(['%.5f' % r for r in ret['recall']]),
                              '\t'.join(['%.5f' % r for r in ret['ndcg']]),
                              '\t'.join(['%.5f' % r for r in ret['precision']]),
                              '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                              )
                print(final_perf)

                f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
            f.close()
            print("report differ popularity level performance")
            users_to_test_list, popularity_state = model.get_popularity_split()

            save_path = '%sreport/%s/%s.popularity_result' % (args.proj_path, args.dataset, args.model_type)
            ensureDir(save_path)
            f = open(save_path, 'w')
            f.write(
                'embed_size=%d, lr=%.4f, l2_weight=%.5f, \n' % (args.dim, args.lr, args.l2))
            for i, users_to_test in enumerate(users_to_test_list):
                ret = test(model, users_to_test, user_dict, n_params)

                final_perf = "recall=[%s], ndcg=[%s], precision=[%s], hit=[%s]" % \
                             ('\t'.join(['%.5f' % r for r in ret['recall']]),
                              '\t'.join(['%.5f' % r for r in ret['ndcg']]),
                              '\t'.join(['%.5f' % r for r in ret['precision']]),
                              '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                              )
                print(final_perf)

                f.write('\t%s\n\t%s\n' % (popularity_state[i], final_perf))
            f.close()
            exit()
        else:
            users_to_test = list(model.test_user_dict.keys())

            ret = test(model, users_to_test, user_dict, n_params)
            cur_best_pre_0 = ret['recall'][0]
            pretrain_ret = 'pretrained model recall=[%.5f, %.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f, ' \
                           '%.5f], precision=[%.5f, %.5f, %.5f, ' \
                           '%.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f, %.5f], auc=[%.5f]' \
                           % \
                           (
                               ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['recall'][3],
                               ret['recall'][4],
                               ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][3], ret['ndcg'][4],
                               ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][3],
                               ret['precision'][4],
                               ret['hit_ratio'][0], ret['hit_ratio'][1], ret['hit_ratio'][2], ret['hit_ratio'][3],
                               ret['hit_ratio'][4],
                               ret['auc'])
            print(pretrain_ret)

    else:
        cur_best_pre_0 = 0
    print("start training ...")
    for epoch in range(args.epoch):
        """training CF"""
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]

        """training"""
        totall_regularizer = 0
        loss, s = 0, 0
        train_s_t = time()
        train_sample_user_item = torch.LongTensor(sample_item(user_dict['train_user_set'], train_sample_item_size)).to(
            device)
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_pairs,
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
        loss_list.append(loss.item())
        if epoch % 10 == 9 or epoch == 1:
            """testing"""
            test_s_t = time()
            ret = test(model, list(model.test_user_dict.keys()), user_dict, n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision",
                                     "hit_ratio", "mean Average Precision", "auc", "f1"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'],
                 ret['precision'], ret['hit_ratio'], ret['map'], ret['auc'], ret['f1']]
            )
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][0] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), save_path)
                print('save the weights in path: ', save_path)

        else:

            print('using time %.4f, training loss at epoch %d: %.4f, regularizer: %.6f' % (
                train_e_t - train_s_t, epoch, loss.item(), totall_regularizer.item()))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
    loss_list=np.array(loss_list)
    np.savetxt('data/' + args.dataset+'/loss.txt', loss_list,fmt='%.4f')


