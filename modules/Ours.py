import torch
import torch.nn as nn
from .contrast import Contrast_user
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_add


class UserItemGcn(nn.Module):

    def __init__(self, n_hops, n_users, n_items, act, device,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(UserItemGcn, self).__init__()
        self.device = device
        self.n_users = n_users
        self.n_items = n_items
        self.n_hops = n_hops
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        if act == 'tanh':
            self.act = torch.tanh
        elif act == 'relu':
            self.act = torch.relu
        elif act == 'sigmoid':
            self.act = torch.sigmoid
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_emb, item_emb, interact_mat, mess_dropout=True, node_dropout=False):
        """node dropout"""
        if node_dropout:
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        interact_indices = interact_mat.coalesce().indices()
        user_indices = interact_indices[0]
        item_indices = interact_indices[1]
        user_agg = user_emb
        item_agg = item_emb
        user_ui_emb = torch.zeros(size=user_emb.size(), device='cuda:0')
        item_ui_emb = torch.zeros(size=item_emb.size(), device='cuda:0')
        for i in range(self.n_hops):

            user_agg_temp = user_agg
            item_agg_temp = item_agg

            user_agg = item_agg_temp[item_indices]
            item_agg = user_agg_temp[user_indices]

            user_agg = scatter_add(src=user_agg, index=user_indices, dim_size=self.n_users, dim=0)
            item_agg = scatter_add(src=item_agg, index=item_indices, dim_size=self.n_items, dim=0)
            if mess_dropout:
                item_agg = self.dropout(item_agg)
                user_agg = self.dropout(user_agg)
            item_agg = F.normalize(item_agg)
            user_agg = F.normalize(user_agg)

            user_ui_emb = user_agg + user_ui_emb
            item_ui_emb = item_agg + item_ui_emb
        item_ui_emb = item_ui_emb + item_emb
        user_ui_emb = user_ui_emb + user_emb
        return item_ui_emb, user_ui_emb


class RCIMGcn(nn.Module):

    def __init__(self, n_hops, n_users, n_entities, user_interests,
                 device, emb_size, tau, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(RCIMGcn, self).__init__()
        self.n_hops = n_hops
        self.n_users = n_users
        self.n_entities = n_entities
        self.user_interests = user_interests
        self.emb_size = emb_size
        self.tau = tau
        self.device = device
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.dropout = nn.Dropout(p=mess_dropout_rate)
        self.contrast_user = Contrast_user(self.emb_size, tau=self.tau)

    def forward(self, user_emb, user_ui_emb, entity_emb, mess_dropout):
        att_weight = torch.zeros((self.n_users, 0), device=self.device)
        user_int_emb = torch.zeros((self.n_users, 0, self.emb_size), device=self.device)
        for i in range(len(self.user_interests)):

            user_int_emb_r = torch.zeros(user_emb.size(), device=self.device)
            user_agg_emb = user_emb
            entity_int_emb = entity_emb
            user_index = self.user_interests[i][:, 0]
            entity_index = self.user_interests[i][:, 1]

            for hop in range(self.n_hops):
                user_agg_emb_temp = user_agg_emb
                user_agg_emb = entity_int_emb[entity_index]
                entity_int_emb = user_agg_emb_temp[user_index]

                user_agg_emb = scatter_add(src=user_agg_emb, index=user_index, dim=0, dim_size=self.n_users)
                entity_int_emb = scatter_add(src=entity_int_emb, index=entity_index, dim=0, dim_size=self.n_entities)
                if mess_dropout:
                    user_agg_emb = self.dropout(user_agg_emb)
                    entity_int_emb = self.dropout(entity_int_emb)
                user_agg_emb = F.normalize(user_agg_emb)
                entity_int_emb = F.normalize(entity_int_emb)
                user_int_emb_r += user_agg_emb

            user_int_emb = torch.cat((user_int_emb, user_int_emb_r.unsqueeze(1)), dim=1)
            att_weight = torch.cat((att_weight, torch.sum(user_int_emb_r * user_ui_emb, dim=1).unsqueeze(1)), dim=1)

        att_weight = torch.softmax(att_weight, dim=1)
        user_int_emb = user_int_emb * att_weight.unsqueeze(-1)
        user_int_emb = torch.sum(user_int_emb, dim=1)
        user_int_emb = user_int_emb + user_emb

        user_cl_loss = self.contrast_user(user_ui_emb, user_int_emb)
        return user_int_emb, user_cl_loss


class KgAttention(nn.Module):

    def __init__(self, n_hops, n_entities, emb_size, device, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(KgAttention, self).__init__()
        self.n_hops = n_hops
        self.n_entities = n_entities
        self.emb_size = emb_size
        self.device = device
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.dropout = nn.Dropout(p=mess_dropout_rate)
        self.ne_sim = 0

        initializer = nn.init.xavier_uniform_
        self.q_w = initializer(torch.empty(self.emb_size, self.emb_size))
        self.k_w = initializer(torch.empty(self.emb_size, self.emb_size))
        self.q_w = nn.Parameter(self.q_w)
        self.k_w = nn.Parameter(self.k_w)

    def calculate_att_weights(self, entity_emb_head, entity_emb_tail, relation_emb):
        left = torch.matmul(entity_emb_head, self.q_w)
        right = torch.tanh(torch.matmul((relation_emb * entity_emb_tail), self.k_w))
        att_weights = torch.sum(left * right, dim=1)
        return att_weights

    def forward(self, entity_emb, edge_index, edge_type, edge_emb, mess_dropout):

        head, tail = edge_index
        relation_emb = edge_emb[edge_type]
        entity_kg_emb = torch.zeros(entity_emb.size(), device=self.device)
        agg_emb = entity_emb
        for i in range(self.n_hops):
            att_weight = self.calculate_att_weights(agg_emb[head], agg_emb[tail], relation_emb)
            att_weight_norm = scatter_softmax(att_weight, head, dim_size=self.n_entities, dim=0)

            agg_emb = att_weight_norm.unsqueeze(1) * relation_emb * agg_emb[tail]

            agg_emb = scatter_mean(src=agg_emb, index=head, dim_size=self.n_entities, dim=0)
            if mess_dropout:
                agg_emb = self.dropout(agg_emb)
            agg_emb = F.normalize(agg_emb)

            entity_kg_emb = entity_kg_emb + agg_emb
            entity_kg_emb = entity_kg_emb + entity_emb
        return entity_kg_emb


class Ours(nn.Module):
    def __init__(self, dataset_info, args_config, knowledge_graph, user_item_graph, user_interest_graph):
        super(Ours, self).__init__()

        self.n_users = dataset_info['n_users']
        self.n_items = dataset_info['n_items']
        self.n_relations = dataset_info['n_relations']
        self.n_entities = dataset_info['n_entities']  # include items
        self.n_nodes = dataset_info['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.tau = args_config.tau
        self.cl_rate = args_config.cl_rate
        self.il_rate = args_config.il_rate
        self.emb_size = args_config.dim
        self.user_interest_graph_hop = args_config.user_interest_graph_hop
        self.knowledge_graph_hop = args_config.knowledge_graph_hop
        self.user_item_graph_hop = args_config.user_item_graph_hop
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate

        act = args_config.activate
        if act == 'tanh':
            self.act = torch.tanh
        elif act == 'relu':
            self.act = torch.relu
        elif act == 'sigmoid':
            self.act = torch.sigmoid

        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.adj_mat = user_item_graph
        self.edge_index, self.edge_type = self._get_edges(knowledge_graph)
        self.user_interests = []
        for interests in user_interest_graph:
            self.user_interests.append(torch.tensor(interests, device=self.device, dtype=int))
        self.user_item_graph_encoder, self.KG_encoder, self.Rcim = self.module_init()

        self._init_weight()
        self.all_emb = nn.Parameter(self.all_emb)
        self.edge_emb = nn.Parameter(self.edge_emb)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_emb = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.edge_emb = initializer(torch.empty(self.n_relations, self.emb_size))
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def module_init(self):
        user_item_graph_encoder = UserItemGcn(n_hops=self.user_item_graph_hop,
                                              n_users=self.n_users,
                                              n_items=self.n_items,
                                              act=self.act,
                                              device=self.device,
                                              node_dropout_rate=self.node_dropout_rate,
                                              mess_dropout_rate=self.mess_dropout_rate)
        KG_encoder = KgAttention(n_hops=self.knowledge_graph_hop,
                                 n_entities=self.n_entities,
                                 emb_size=self.emb_size,
                                 device=self.device,
                                 node_dropout_rate=self.node_dropout_rate,
                                 mess_dropout_rate=self.mess_dropout_rate)
        Rcim = RCIMGcn(n_hops=self.user_interest_graph_hop,
                       n_users=self.n_users,
                       n_entities=self.n_entities,
                       user_interests=self.user_interests,
                       emb_size=self.emb_size,
                       tau=self.tau,
                       device=self.device,
                       node_dropout_rate=self.node_dropout_rate,
                       mess_dropout_rate=self.mess_dropout_rate)
        return user_item_graph_encoder, KG_encoder, Rcim

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch=None):
        interact_task_samples = batch['interact_task_samples']
        interest_task_samples = batch['interest_task_samples']

        user_emb = self.all_emb[:self.n_users, :]
        entity_emb = self.all_emb[self.n_users:, :]
        item_emb = entity_emb[:self.n_items]

        item_ui_emb, user_ui_emb = self.user_item_graph_encoder(user_emb,
                                                                item_emb,
                                                                self.interact_mat,
                                                                mess_dropout=self.mess_dropout,
                                                                node_dropout=self.node_dropout)

        entity_kg_emb = self.KG_encoder(entity_emb,
                                        self.edge_index,
                                        self.edge_type,
                                        self.edge_emb,
                                        self.mess_dropout)

        item_kg_emb = entity_kg_emb[:self.n_items]
        user_int_emb, user_cl_loss = self.Rcim(user_emb,
                                               user_ui_emb,
                                               entity_kg_emb + entity_emb,
                                               self.mess_dropout)

        user_final_emb = torch.cat((user_ui_emb, user_int_emb), dim=-1)
        item_final_emb = torch.cat((item_ui_emb, item_kg_emb), dim=-1)

        int_loss = self.create_interest_loss(user_final_emb, item_final_emb, interest_task_samples)

        bpr_loss, emb_loss = self.create_interact_loss(user_final_emb, item_final_emb,interact_task_samples)

        all_loss = bpr_loss + self.cl_rate * user_cl_loss + self.il_rate * int_loss + self.decay * emb_loss

        return all_loss

    def generate(self):
        user_emb = self.all_emb[:self.n_users, :]
        entity_emb = self.all_emb[self.n_users:, :]
        item_emb = entity_emb[:self.n_items]
        """interest"""
        item_ui_emb, user_ui_emb = self.user_item_graph_encoder(user_emb,
                                                                item_emb,
                                                                self.interact_mat,
                                                                mess_dropout=False, node_dropout=False)
        entity_kg_emb = self.KG_encoder(entity_emb,
                                        self.edge_index,
                                        self.edge_type,
                                        self.edge_emb,
                                        mess_dropout=False)

        item_kg_emb = entity_kg_emb[:self.n_items]
        user_int_emb, _ = self.Rcim(user_emb,
                                    user_ui_emb,
                                    entity_kg_emb + entity_emb,
                                    mess_dropout=False)
        user_int_emb = user_int_emb + user_emb
        item_kg_emb = item_kg_emb + item_emb

        item_ui_emb = item_ui_emb + item_emb
        user_ui_emb = user_ui_emb + user_emb

        user_final_emb = torch.cat((user_ui_emb, user_int_emb), dim=-1)
        item_final_emb = torch.cat((item_ui_emb, item_kg_emb), dim=-1)

        return item_final_emb, user_final_emb

    def create_interact_loss(self, user_final_emb, item_final_emb, interact_task_samples):

        users = user_final_emb[interact_task_samples[:, 0]]
        pos_items = item_final_emb[interact_task_samples[:, 1]]
        neg_items = item_final_emb[interact_task_samples[:, 2]]


        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        bpr_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        emb_loss = ((torch.norm(users) ** 2
                        + torch.norm(pos_items) ** 2
                        + torch.norm(neg_items) ** 2) / batch_size
                       ) / 2
        return bpr_loss, emb_loss

    def create_interest_loss(self, user_final_emb, item_final_emb, int_task_samples):
        users = user_final_emb[int_task_samples[:, 0]]
        items = item_final_emb[int_task_samples[:, 1]]
        labels = int_task_samples[:, 2]

        scores = torch.sigmoid(torch.sum(torch.mul(users, items), dim=1))
        loss_fun = nn.BCELoss()
        int_loss = loss_fun(scores, labels.float())
        return int_loss
