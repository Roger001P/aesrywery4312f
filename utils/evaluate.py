from .metrics import *
from functools import partial
import torch
import numpy as np
import multiprocessing
import heapq
from sklearn.metrics import f1_score, roc_auc_score

cores = multiprocessing.cpu_count() // 2

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r

def get_performance(user_pos_test, r, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}


def test_one_user(x, train_user_set, test_user_set, n_items, args):
    Ks = eval(args.Ks)
    rating = x[0]
    u = x[1]
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = test_user_set[u]
    if len(user_pos_test) == 0:
        return 0
    all_items = set(range(0, n_items))

    test_items = list(all_items - set(training_items))

    r = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)

    ret = get_performance(user_pos_test, r, Ks)
    return ret


def test(model, dataset_dict, dataset_info, flag):
    args = dataset_dict['args']
    Ks = eval(args.Ks)
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    BATCH_SIZE = args.test_batch_size

    result = {'pre': np.zeros(len(Ks)),
              'rec': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.,
              'f1': 0.}

    n_items = dataset_info['n_items']
    n_users = dataset_info['n_users']

    train_user_set = dataset_dict['train_user_set']
    if flag == "eval":
        test_cf = dataset_dict['eval_cf']
        test_user_set = dataset_dict['eval_user_set']

    else:
        test_cf = dataset_dict['test_cf']
        test_user_set = dataset_dict['test_user_set']

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    i_e, u_e = model.generate()

    count = 0
    auc = f1 = 0
    start = 0
    if test_cf.shape[1] == 3:
        auc_list = []
        f1_list = []
        while start < test_cf.shape[0]:

            test_data = test_cf[start:start + BATCH_SIZE]
            user_index = test_data[:, 0]
            item_index = test_data[:, 1]
            labels = test_data[:, 2]
            user_emb = u_e[torch.tensor(user_index, device=device)]
            item_emb = i_e[torch.tensor(item_index, device=device)]
            pre_scores = torch.sigmoid(torch.sum(torch.mul(user_emb, item_emb), axis=1))
            pre_scores = np.array(pre_scores.detach().cpu())
            pre_labels = [1 if i >= 0.5 else 0 for i in pre_scores]
            auc_batch = roc_auc_score(y_true=labels, y_score=pre_scores)
            f1_batch = f1_score(y_true=labels, y_pred=pre_labels)

            auc_list.append(auc_batch)
            f1_list.append(f1_batch)
            start += BATCH_SIZE
        auc = np.mean(auc_list)
        f1 = np.mean(f1_list)

    pool = multiprocessing.Pool(cores)
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_list_batch = test_users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = u_e[user_batch]
        n_item_batchs = n_items // i_batch_size + 1
        rate_batch = np.zeros(shape=(len(user_batch), n_items))
        i_count = 0
        for i_batch_id in range(n_item_batchs):
            i_start = i_batch_id * i_batch_size
            i_end = min((i_batch_id + 1) * i_batch_size, n_items)
            item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)
            i_g_embeddings = i_e[item_batch]
            i_rate_batch = torch.sigmoid(torch.matmul(u_g_embeddings, i_g_embeddings.t())).detach().cpu()
            rate_batch[:, i_start: i_end] = i_rate_batch
            i_count += i_rate_batch.shape[1]
        assert i_count == n_items

        user_batch_rating_uid = zip(rate_batch, user_list_batch)
        test_one_user_expand = partial(test_one_user, train_user_set=train_user_set, test_user_set=test_user_set, n_items=n_items, args=args)
        batch_result = pool.map(test_one_user_expand, user_batch_rating_uid)
        count += len(batch_result)
        for re in batch_result:
            if re == 0:
                n_test_users -= 1
                continue
            result['pre'] += re['precision']
            result['rec'] += re['recall']
            result['ndcg'] += re['ndcg']
            result['hit_ratio'] += re['hit_ratio']

    result['pre'] = result['pre'] / n_test_users
    result['rec'] = result['rec'] / n_test_users
    result['ndcg'] = result['ndcg'] / n_test_users
    result['hit_ratio'] = result['hit_ratio'] / n_test_users
    pool.close()

    result['auc'] = auc
    result['f1'] = f1

    return result