import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import os
from collections import defaultdict

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0


def remap_item(train_data, eval_data, test_data):
    global n_users, n_items

    train_user_set = defaultdict(list)
    train_item_set = defaultdict(list)
    eval_user_set = defaultdict(list)
    test_user_set = defaultdict(list)

    train_data = train_data[train_data[:, 2] == 1][:, :2]
    eval_data = eval_data[eval_data[:, 2] == 1][:, :2]
    test_data = test_data[test_data[:, 2] == 1][:, :2]

    n_users = max(max(train_data[:, 0]), max(eval_data[:, 0]),  max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(eval_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
        train_item_set[int(i_id)].append(int(u_id))
    for u_id, i_id in eval_data:
        eval_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    return train_user_set, eval_user_set, test_user_set, train_item_set


def read_triplets(file_name):
    global n_entities, n_relations, n_nodes
    can_triplets_np = np.loadtxt(file_name + 'kg_final.txt', dtype=np.int32)
    relation_dict = np.zeros((len(set(can_triplets_np[:, 1])), 1), dtype='<U44')

    can_triplets_np = np.unique(can_triplets_np, axis=0)
    if args.inverse_r:
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def build_graph(triplets):
    knowledge_graph = nx.MultiDiGraph()
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        knowledge_graph.add_edge(h_id, t_id, key=r_id)
    return knowledge_graph


def build_sparse_relational_graph(train_data):
    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    train_data = train_data[train_data[:, 2] == 1][:, :2]
    relation_dict = defaultdict(list)
    for u_id, i_id in tqdm(train_data, ascii=True):
        relation_dict[0].append([u_id, i_id])
    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    return mean_mat_list[0]


def read_interact(directory):
    rating_file = directory + 'ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    """6:2:2"""
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * test_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    left = set(left) - set(test_indices)
    train_indices = list(left)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data

def load_data(model_args):
    global args
    args = model_args

    directory = args.data_path + args.dataset + '/'
    print('reading user item interact data ...')
    train_cf, eval_cf, test_cf = read_interact(directory)

    print('interaction count: train %d, eval %d, test %d' % (train_cf.shape[0], eval_cf.shape[0], test_cf.shape[0]))
    train_user_set, eval_user_set, test_user_set, train_item_set = remap_item(train_cf, eval_cf, test_cf)

    print('read knowledge graph triples ...')
    triplets = read_triplets(directory)

    print('building the graph ...')
    knowledge_graph = build_graph(triplets)

    print('building the user_item_graph ...')
    user_item_graph = build_sparse_relational_graph(train_cf)

    dataset_info = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations),
    }
    dataset_dict = {
        'train_user_set': train_user_set,
        'eval_user_set': eval_user_set,
        'test_user_set': test_user_set,
        'train_cf': train_cf,
        'eval_cf': eval_cf,
        'test_cf': test_cf,
        'triplets': triplets,
    }

    return dataset_dict, dataset_info, knowledge_graph, user_item_graph

