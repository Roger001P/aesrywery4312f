import numpy as np
from tqdm import tqdm
import pickle
from collections import Counter

def construct_item_entity_set(triplets, n_items):


    triplets = triplets[triplets[:, 2] >= n_items]
    triplets = triplets[triplets[:, 0] < n_items]
    triplets = triplets[np.argsort(triplets[:, 0])]

    item_attribute_entity_sets = {}
    for r in set(triplets[:, 1]):
        triplets_r = triplets[triplets[:, 1] == r]
        """Remove attribute entities with overly high occurrence to improve discriminability"""
        counter = Counter(list(triplets_r[:, 2]))
        sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(sorted_counter)):
            if sorted_counter[i][1] >= (n_items * 0.5):
                triplets_r = triplets_r[triplets_r[:, 2] != sorted_counter[i][0]]
            else:
                break

        if triplets_r.shape[0] == 0:
            item_attribute_entity_sets[r] = {}
            continue
        attributes = {}
        for i in range(triplets_r.shape[0]):
            if not attributes.get(triplets_r[i, 0]):
                attributes[triplets_r[i, 0]] = [triplets_r[i, 2]]
            else:
                attributes[triplets_r[i, 0]].append(triplets_r[i, 2])
        if len(attributes) <= 100:
            continue
        else:
            item_attribute_entity_sets[r] = attributes
    return item_attribute_entity_sets

def construct_user_interest_graph(train_user_set, n_users, item_attribute_entity_sets, T):
    top_n = 10
    user_interest_graph = []  # 用户兴趣图
    user_interests_set = {}
    interest_relation_list = []
    entity_interest_set = {}

    for r in tqdm(item_attribute_entity_sets.keys(), ascii=True):
        r_i = np.zeros((0, 2), dtype=int)
        for user in range(n_users):
            if not train_user_set.get(user):
                continue
            attribute = []
            interests = []

            for item in train_user_set[user]:
                if item not in item_attribute_entity_sets[r]:
                    continue
                else:
                    attribute.extend(item_attribute_entity_sets[r][item])
            if len(attribute) == 0:
                continue
            counter = Counter(attribute)
            sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)

            for i in range(top_n):
                if i >= len(sorted_counter):
                    break
                if sorted_counter[i][1] >= T:
                    interests.append(sorted_counter[i][0])
                else:
                    break
            user_np = np.full(len(interests), user, dtype=int)
            u_i = np.column_stack((user_np, np.array(interests, dtype=int)))
            r_i = np.concatenate((r_i, u_i), axis=0)
        if len(r_i) <= 0:
            continue
        """Remove interest entities with overly high occurrence to improve discriminability"""
        counter = Counter(list(r_i[:, 1]))
        sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(sorted_counter)):
            if sorted_counter[i][1] >= n_users * 0.5:
                r_i = r_i[r_i[:, 0] != sorted_counter[i][0]]
        interest_relation_list.append(r)
        user_interest_graph.append(r_i)

        u_set = {}
        e_set = {}
        for u in set(r_i[:, 0]):
            u_set[u] = set(r_i[r_i[:, 0] == u][:, 1])
        for e in set(r_i[:, 1]):
            e_set[e] = set(r_i[r_i[:, 1] == e][:, 0])
        user_interests_set[r] = u_set
        entity_interest_set[r] = e_set
    return interest_relation_list, user_interest_graph, entity_interest_set

def construct_interest_aware_sample(n_users, n_items, train_user_set, item_attribute_entity_sets, interest_relation_list, entity_interest_set, args):

    directory = args.data_path + args.dataset + '/'
    C = args.C
    user_interested_item_num = {}
    user_interested_item_set = {}
    if args.dataset == 'movie':
        with open(f'{directory}user_interested_item_num_2.pkl', 'rb') as f:
            user_interested_item_num = pickle.load(f)

        # for r in interest_relation_list:
        #     for user in tqdm(range(n_users), ascii=True):
        #         if not user_interests_set[r].get(user):
        #             continue
        #         if not user_interested_item_num.get(user):
        #             user_interested_item_num[user] = {}
        #
        #         for item in range(n_items):
        #             if not item_attribute_entity_sets[r].get(item):
        #                 continue
        #             num = len(set(user_interests_set[r][user]) & set(item_attribute_entity_sets[r][item]))
        #             if num == 0:
        #                 continue
        #
        #             if not user_interested_item_num[user].get(item):
        #                 user_interested_item_num[user][item] = num
        #             else:
        #                 user_interested_item_num[user][item] += num

        # for r_index in range(len(interest_relation_list)):
        #     r = interest_relation_list[r_index]
        #
        #     rr[r] = 0
        #
        #     for i in tqdm(range(n_items), ascii=True):
        #         if not item_attribute_entity_sets[r].get(i):
        #             continue
        #         attribute_set = set(item_attribute_entity_sets[r][i])
        #         user_list = []
        #         for a in attribute_set:
        #             if not entity_interest_set[r].get(a):
        #                 continue
        #
        #             for uu in entity_interest_set[r][a]:
        #                 if not train_user_set.get(uu):
        #                     continue
        #                 if uu in user_list:
        #                     continue
        #                 else:
        #                     user_list.append(uu)
        #                 rr[r] += 1
        #                 # if i in train_user_set[uu]:
        #                 if not user_interested_item_num.get(uu):
        #                     user_interested_item_num[uu] = {}
        #                     user_interested_item_num[uu][i] = 1
        #                 else:
        #                     if not user_interested_item_num[uu].get(i):
        #                         user_interested_item_num[uu][i] = 1
        #                     else:
        #                         user_interested_item_num[uu][i] += 1
    else:
        for r_index in range(len(interest_relation_list)):
            r = interest_relation_list[r_index]
            for i in tqdm(range(n_items), ascii=True):
                if not item_attribute_entity_sets[r].get(i):
                    continue
                attribute_set = set(item_attribute_entity_sets[r][i])
                user_list = []
                for a in attribute_set:
                    if not entity_interest_set[r].get(a):
                        continue

                    for uu in entity_interest_set[r][a]:
                        if not train_user_set.get(uu):
                            continue
                        if uu in user_list:
                            continue
                        else:
                            user_list.append(uu)
                        if not user_interested_item_num.get(uu):
                            user_interested_item_num[uu] = {}
                            user_interested_item_num[uu][i] = 1
                        else:
                            if not user_interested_item_num[uu].get(i):
                                user_interested_item_num[uu][i] = 1
                            else:
                                user_interested_item_num[uu][i] += 1

    for uu in user_interested_item_num.keys():
        for ii in user_interested_item_num[uu].keys():
            if user_interested_item_num[uu][ii] >= C:  # 阈值C
                if not user_interested_item_set.get(uu):
                    user_interested_item_set[uu] = [ii]
                else:
                    user_interested_item_set[uu].append(ii)

    user_interested_item = np.zeros((0, 3), dtype=int)
    for u in tqdm(range(n_users), ascii=True):
        if not user_interested_item_set.get(u) or not train_user_set.get(u):
            continue
        n = len(user_interested_item_set[u])
        cf_1 = np.full(n, u, dtype=int)
        cf_2 = np.array(user_interested_item_set[u], dtype=int)
        cf_3 = np.ones(n, dtype=int)
        user_interested_item = np.concatenate((user_interested_item, np.vstack((cf_1, cf_2, cf_3)).T), axis=0)
    return user_interested_item, user_interested_item_set
def construct_graph(dataset_dict, dataset_info, args):
    n_users = dataset_info['n_users']
    n_items = dataset_info['n_items']
    triplets = dataset_dict['triplets']
    train_user_set = dataset_dict['train_user_set']
    T = args.T

    print("Start constructing the item attribute entity set")
    item_attribute_entity_sets = construct_item_entity_set(triplets, n_items)
    print("Start constructing the user interest entity set")
    interest_relation_list, user_interest_graph, entity_interest_set = construct_user_interest_graph(train_user_set, n_users, item_attribute_entity_sets, T)
    print("Start constructing the interest_aware_loss_sample")
    user_interested_item, user_interested_item_set = construct_interest_aware_sample(n_users, n_items, train_user_set, item_attribute_entity_sets, interest_relation_list, entity_interest_set, args)

    return user_interest_graph, user_interested_item, user_interested_item_set
