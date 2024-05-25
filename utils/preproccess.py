import os

import argparse
import numpy as np
import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import heapq
import random
from time import time
from collections import defaultdict
import torch
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)

def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)
def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))

def build_sparse_neibor_matrix(train_data,user_top_k_neibor,item_top_k_neibor):
    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()
    print("Begin to build sparse adj matrix ...")
    np_mat = train_data
    cf = np_mat.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    vals = [1.] * len(cf)
    adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_users, n_users+n_items))
    # interaction: user->item, [n_users, n_items]
    adj=adj.tocsr()[:,n_users:]
    item_user_adj=adj.T
    print(" build sparse item_user_adj  matrix ...")
    norm_item_user_adj=_si_norm_lap(item_user_adj)
    print(" build sparse user_neibor  matrix ...")
    user_neibor=user_adj(adj,user_top_k_neibor)
    norm_user_neibor=_si_norm_lap(user_neibor)
    print(" build sparse item_neibor  matrix ...")
    item_neibor=item_adj(adj,item_top_k_neibor)
    norm_item_neibor=_si_norm_lap(item_neibor)
    return norm_item_user_adj, norm_user_neibor, norm_item_neibor
def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)
# def sample_item(train_user,nums=16):
#     uset_history_item=np.zeros((n_users,nums))
#     for u_id in train_user:
#         if len(train_user[u_id])>=nums:
#             uset_history_item[u_id,:]=np.random.choice(train_user[u_id],nums,replace=False)
#         else:
#             uset_history_item[u_id,:]=np.random.choice(train_user[u_id],nums,replace=True)
#     return uset_history_item
def sample_item(dirname, train_user,nums=16):
    dir_path = dirname + 'prepreccess/'
    ensureDir(dir_path)
    file_name = dir_path +'user_sample_{}_items.txt'.format(nums)
    if os.path.isfile(file_name):
        can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
        return np.array(can_triplets_np)
    else:
        uset_history_item=np.zeros((n_users,nums))
        for u_id in train_user:
            if len(train_user[u_id])>=nums:
                uset_history_item[u_id,:]=np.random.choice(train_user[u_id],nums,replace=False)
            else:
                uset_history_item[u_id,:]=np.random.choice(train_user[u_id],nums,replace=True)
        np.savetxt(file_name,uset_history_item,fmt='%d')
        return uset_history_item
# def user_adj(adj,top_K):
#     """
#
#     :param adj: sparse.csr.csr_matrix size=(user_nuber,item_number)
#     :param top_K: neibor
#     :return: sp.coo_matrix top_k neibor matrix size=(user_nuber,user_nuber)
#     """
#     nums=adj.shape[0]
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     index = list()
#
#     data = []
#
#     adj=adj.dot(adj.T)
#     adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
#     row,col=adj.nonzero()
#
#     score= {}
#
#     k=0
#     def addneibor():
#         K_max_item_score = heapq.nlargest(top_K+1, score, key=score.get)
#         if len(K_max_item_score) == 1:
#             index.append([k, k])
#             data.append(1.)
#         else:
#             for i in K_max_item_score:
#                 if i != k:
#                     index.append([k, i])
#                     data.append(score[i])
#     for i in range(len(row)):
#        if row[i]==k:
#             score[col[i]]=adj[row[i],col[i]]
#        else:
#            addneibor()
#            score.clear()
#            while k<row[i]:
#                 k+=1
#            score[col[i]]=adj[row[i],col[i]]
#     addneibor()
#     index=np.array(index)
#     return sp.coo_matrix((data,(index[:,0],index[:,1])),shape=(nums,nums))
# def item_adj(adj,top_K):
#     """
#
#     :param adj: sparse.csr.csr_matrix size=(user_nuber,item_number)
#     :param top_K: neibor number
#     :return: sp.coo_matrix top_k neibor matrix size=(item_number,item_number)
#     """
#   return user_adj(adj.T,top_K)
def user_adj(dirname , adj, top_K, user_neibor=True):
    """
    :param dirname: dir eg.data/last-fm
    :param adj: sparse.csr.csr_matrix size=(user_nuber,item_number)
    :param top_K: neibor
    :return: sp.coo_matrix top_k neibor matrix size=(user_nuber,user_nuber)
    """
    nums=adj.shape[0]
    if user_neibor:
        dir_path = dirname + 'prepreccess/user_{}_neibor/'.format(top_K)
    else:
        dir_path = dirname + 'prepreccess/item_{}_neibor/'.format(top_K)
    if os.path.exists(dir_path):
        index_file_name = dir_path + 'index.txt'
        data_file_name = dir_path + 'data.txt'
        index = np.loadtxt(index_file_name, dtype=np.int32)
        data = np.loadtxt(data_file_name, dtype=float)
        adj = sp.coo_matrix((data, (index[:, 0], index[:, 1])), shape=(nums, nums))
        return adj
    else:
        ensureDir(dir_path)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        index = list()

        data = []

        adj = adj.dot(adj.T)
        adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        row, col = adj.nonzero()

        score = {}

        k = 0

        def addneibor():
            K_max_item_score = heapq.nlargest(top_K + 1, score, key=score.get)
            if len(K_max_item_score) == 1:
                index.append([k, k])
                data.append(1.)
            else:
                for i in K_max_item_score:
                    if i != k:
                        index.append([k, i])
                        data.append(score[i])

        for i in range(len(row)):
            if row[i] == k:
                score[col[i]] = adj[row[i], col[i]]
            else:
                addneibor()
                score.clear()
                while k < row[i]:
                    k += 1
                score[col[i]] = adj[row[i], col[i]]
        addneibor()
        index = np.array(index)
        data = np.array(data)
        index_file_name = dir_path + 'index.txt'
        data_file_name = dir_path + 'data.txt'
        np.savetxt(index_file_name, index, fmt='%d')
        np.savetxt(data_file_name, data, fmt='%.3f')

        adj = sp.coo_matrix((data, (index[:, 0], index[:, 1])), shape=(nums, nums))
        return adj

def item_adj(dirname, adj,top_K):
    """

    :param adj: sparse.csr.csr_matrix size=(user_nuber,item_number)
    :param top_K: neibor number
    :return: sp.coo_matrix top_k neibor matrix size=(item_number,item_number)
    """
    return user_adj(dirname, adj.T,top_K, user_neibor=False)
def build_adj(train_data):
    np_mat = train_data
    cf = np_mat.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    vals = [1.] * len(cf)
    adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_users, n_users + n_items))
    # interaction: user->item, [n_users, n_items]
    adj = adj.tocsr()[:, n_users:]
    return adj
# supdir ='../testdata/'
# subdir = ['last-fm/', 'amazon-book/', 'alibaba-fashion/']
# sample_size = [16,32,64,128,256]
# user_neibor_number = [4,8,16,32,64,128]
# item_neibor_number = [4,8,16,32,64,128]
# dirname = supdir +subdir[0]
# train_data = read_cf(dirname + 'train.txt')
# test_data = read_cf(dirname + 'test.txt')
# remap_item(train_data, test_data)
# user_sample_item = sample_item(dirname,train_user_set,nums=sample_size[1])
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# adj = build_adj(train_data)
# user_neibor = user_adj(dirname,adj, user_neibor_number[1], user_neibor =True)
# item_neibor = item_adj(dirname, adj, item_neibor_number[1])
# print(user_neibor.todense())
# print(item_neibor.todense())
#
# print(item_neibor.todense())
def preproccess(supdir ='../testdata/',subdir = ['last-fm/', 'amazon-book/', 'alibaba-fashion/'],
                sample_size = [16,32,64,128,256], user_neibor_number = [4,8,16,32,64,128], item_neibor_number= [4,8,16,32,64,128]):
    print('begin preproccess==========================================================================================')
    for sd in subdir:
        dirname = supdir + sd
        train_data = read_cf(dirname + 'train.txt')
        test_data = read_cf(dirname + 'test.txt')
        train_user_set.clear()
        test_user_set.clear()
        remap_item(train_data, test_data)
        startsb=time()
        for size in sample_size:
            start_sample=time()
            print(sd[:-1] + ' start_sample {}_item at:'.format(size),start_sample)
            user_sample_item = sample_item(dirname,train_user_set,nums=size)
            end_sample = time()
            print(sd[:-1] + ' end_sample {}_item at:'.format(size), end_sample)
            print(sd[:-1] +' totall time :',end_sample-start_sample)
            print(user_sample_item.shape)
        adj = build_adj(train_data)
        for u_number in user_neibor_number:
            start_build_user_neibor=time()
            print(sd[:-1] +' start_build_user_{}_neibor at:'.format(u_number), start_build_user_neibor)
            user_neibor = user_adj(dirname, adj, u_number, user_neibor=True)
            end_build_user_neibor = time()
            print(sd[:-1] +' end_build_user_{}_neibor at:'.format(u_number),  end_build_user_neibor)
            print(sd[:-1] +' totall time :', end_build_user_neibor - start_build_user_neibor)
            print(sd[:-1] +' user {}_neibor preroccess finish'.format(u_number))
            print(user_neibor.shape)
        for item_number in item_neibor_number:
            start_build_item_neibor = time()
            print(sd[:-1] +' start_build_item_{}_neibor at:'.format(item_number), start_build_item_neibor)
            item_neibor = item_adj(dirname, adj, item_number)
            end_build_item_neibor = time()
            print(sd[:-1] +' end_build_item_{}_neibor at:'.format(item_number), end_build_item_neibor)
            print(sd[:-1] + ' totall time :', end_build_item_neibor - start_build_item_neibor)
            print(sd[:-1] + ' item {}_neibor preroccess finish'.format(item_number))
            print(item_neibor.shape)
        endsb = time()
        print(sd[:-1] + " preroccess finish" + " use totall time:", endsb-startsb)
        print(sd[:-1] + " preroccess finish")
    print('preroccess finish==========================================================================================')

parser = argparse.ArgumentParser(description="preproccess")
parser.add_argument("--supdir", nargs="?", default="../data/", help="Choose a dataset:[last-fm,amazon-book,alibaba]")
args =parser.parse_args()
preproccess(supdir =args.supdir,subdir = ['last-fm/', 'amazon-book/', 'alibaba-fashion/'],
                sample_size = [16,32,64,128,256], user_neibor_number = [4,8,16,32,64,128], item_neibor_number= [4,8,16,32,64,128])
print(args.supdir)
#1901