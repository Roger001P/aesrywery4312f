import random
from tqdm import tqdm
import torch
import numpy as np
import time
from prettytable import PrettyTable
from utils.parser import parse_args_kgsr
from utils.data_loader import load_data
from utils.graph_construction import construct_graph
from modules.Ours import Ours
from utils.evaluate import test
from utils.sampler import UniformSampler

seed = 2024
sampling = UniformSampler(seed)

def neg_sampling(dataset_dict, n_items):
    user_interested_item = dataset_dict['user_interested_item']
    train_cf = dataset_dict['train_cf']
    user_interested_item_set = dataset_dict['user_interested_item_set']
    train_user_set = dataset_dict['train_user_set']

    interest_task_negs = sampling.sample_negative_interest(user_interested_item[:, 0], n_items, user_interested_item_set, train_user_set, 1)
    labels = np.zeros(user_interested_item.shape[0], dtype=int)
    interest_task_samples = np.concatenate([user_interested_item[:, 0].reshape(-1, 1), interest_task_negs, labels.reshape(-1, 1)], axis=1)
    interest_task_samples = np.concatenate([user_interested_item, interest_task_samples], axis=0)

    interact_task_samples = train_cf[train_cf[:, 2] == 1][:, :2]
    interact_task_negs = sampling.sample_negative_train(interact_task_samples[:, 0], n_items, train_user_set, 1)
    interact_task_samples = np.concatenate((interact_task_samples, interact_task_negs), axis=1)
    return interact_task_samples, interest_task_samples

def get_batch(interact_task_samples, interest_task_samples, i, batch_size):
    batch = {}
    interact_task_samples = torch.from_numpy(interact_task_samples[i * batch_size:(i + 1) * batch_size]).to(device).long()
    batch['interact_task_samples'] = interact_task_samples
    batch['interest_task_samples'] = torch.tensor(interest_task_samples, device=device)
    return batch

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']
    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

if __name__ == '__main__':

    """fix the random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    """read args"""
    args = parse_args_kgsr()
    test_batch = args.test_batch
    batch_size = args.batch_size
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
    """print time"""
    start_date = time.strftime("%m_%d")
    start_time = time.strftime("%m-%d-%Hh%Mm")
    print(time.strftime(f"{args.dataset}-%m-%d-%Hh%Mm-"))
    """build dataset"""
    dataset_dict, dataset_info, knowledge_graph, user_item_graph = load_data(args)
    dataset_dict['args'] = args
    """Graph Construction"""
    user_interest_graph, user_interested_item, user_interested_item_set = construct_graph(dataset_dict, dataset_info, args)
    dataset_dict['user_interested_item'] = user_interested_item
    dataset_dict['user_interested_item_set'] = user_interested_item_set
    """define model"""
    model = Ours(dataset_info, args, knowledge_graph, user_item_graph, user_interest_graph).to(device)
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    start_epoch = 0
    cur_stopping_step = 0
    stop_epoch = 0
    should_stop = False

    print("start training ...")
    for epoch in range(start_epoch, args.epoch):
        interact_task_samples, interest_task_samples = neg_sampling(dataset_dict, dataset_info['n_items'])

        """training"""
        model.train()
        all_loss = 0
        train_s_t = time.time()
        n = len(interact_task_samples) // batch_size + 1
        for i in tqdm(range(n), ascii=True):
            batch = get_batch(interact_task_samples, interest_task_samples, i, batch_size)
            batch_loss = model(batch)
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()
            all_loss += batch_loss.item()
        train_e_t = time.time()

        """testing"""
        if epoch % test_batch == 0 and epoch >= 10:
            test_s_t = time.time()
            model.eval()
            with torch.no_grad():
                eval_ret = test(model, dataset_dict, dataset_info, "eval")
                test_ret = test(model, dataset_dict, dataset_info, "test")
            test_e_t = time.time()
            train_res = PrettyTable()
            train_res.float_format = '.6f'
            train_res.field_names = ["Epoch", "testing time", "AUC", "F1", "Rec", "Pre", "ndcg"]
            train_res.add_row([epoch, test_e_t - test_s_t, eval_ret['auc'], eval_ret['f1'], eval_ret['rec'], eval_ret['pre'], eval_ret['ndcg']])
            train_res.add_row([epoch, test_e_t - test_s_t, test_ret['auc'], test_ret['f1'], test_ret['rec'], test_ret['pre'], test_ret['ndcg']])
            print(train_res)

            cur_pre = eval_ret['auc']
            cur_best_pre, best_test_auc = 0, 0
            cur_best_pre, cur_stopping_step, should_stop = early_stopping(cur_pre, cur_best_pre, cur_stopping_step, expected_order='acc', flag_step=10)
            stop_epoch = epoch - 10 * test_batch

            if cur_stopping_step == 0:
                print("### Find better!")
                best_test_auc = test_ret['auc']
            if should_stop:
                print('early stopping at %d, eval_AUC:%.4f, test_AUC:%.4f' % (stop_epoch, cur_best_pre, best_test_auc))
                break
        print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, all_loss))

