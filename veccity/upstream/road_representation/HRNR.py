import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch.nn import Module
from torch.nn.parameter import Parameter
from logging import getLogger
from sklearn.metrics import roc_auc_score

from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel
import pdb


class HRNR(AbstractReprLearningModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get("device", torch.device("cpu"))
        self.special_spmm = SpecialSpmm()
        self.dataloader = data_feature.get('dataloader')
        self._logger = getLogger()
        self.model = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.dataset = config.get('dataset', '')
        self.output_dim = config.get('output_dim', 128)
        self.label_num = data_feature.get('label_class')

        self.struct_assign = data_feature.get("struct_assign")
        self.fnc_assign = data_feature.get("fnc_assign")
        adj = data_feature.get("adj_mx")
        self.adj = get_sparse_adj(adj, self.device)
        self.lane_feature=data_feature.get("lane_feature")
        self.type_feature=data_feature.get("type_feature")
        self.length_feature=data_feature.get("length_feature")
        self.node_feature=data_feature.get("node_feature")
        self.hidden_dims=config.get("hidden_dims")
        hparams=dict_to_object(config.config)

        hparams.lane_num=data_feature.get("lane_num")
        hparams.length_num=data_feature.get("length_num")
        hparams.type_num=data_feature.get("type_num")
        hparams.node_num=data_feature.get("num_nodes")

        edge = self.adj.indices()
        edge_e = torch.ones(edge.shape[1], dtype=torch.float).to(self.device)
        struct_inter = self.special_spmm(edge, edge_e, torch.Size([self.adj.shape[0], self.adj.shape[1]]),self.struct_assign)  # N*N   N*C
        struct_adj = torch.mm(self.struct_assign.t(), struct_inter)  # get struct_adj

        self.graph_enc = GraphEncoderTL(hparams, self.struct_assign, self.fnc_assign, struct_adj, self.device)

        self.linear = torch.nn.Linear(self.hidden_dims * 2, self.label_num)

        self.node_emb, self.init_emb = None, None

        self.model_cache_file = './veccity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.road_embedding_path = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)

    def forward(self, x):
        self.node_emb = self.graph_enc(self.node_feature, self.type_feature, self.length_feature, self.lane_feature, self.adj)
        self.init_emb = self.graph_enc.init_feat
        output_state = torch.cat((self.node_emb[x], self.init_emb[x]), 1)
        pred_tra = self.linear(output_state)
        return pred_tra
        
    def run(self, train_dataloader, eval_dataloader):
        self._logger.info("Starting training...")
        hparams = dict_to_object(self.config.config)
        ce_criterion = torch.nn.CrossEntropyLoss()
        max_f1 = 0
        max_auc = 0
        count = 0
        model_optimizer = torch.optim.Adam(self.parameters(), lr=hparams.lp_learning_rate)
        eval_dataloader_iter = iter(eval_dataloader)
        patience = 50
        for i in range(hparams.max_epoch):
            self._logger.info("epoch " + str(i) + ", processed " + str(count))
            for step, (train_set, train_label) in enumerate(train_dataloader):
                model_optimizer.zero_grad()
                train_set = train_set.clone().detach()
                train_label = train_label.clone().detach()
                pred = self(train_set)
                loss = ce_criterion(pred, train_label)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.parameters(), hparams.lp_clip)
                model_optimizer.step()
                if count % 20 == 0:
                    eval_data = get_next(eval_dataloader_iter)
                    if eval_data is None:
                        eval_dataloader_iter = iter(eval_dataloader)
                        eval_data = get_next(eval_dataloader_iter)
                    test_set, test_label = eval_data
                    precision, recall, f1, auc = self.test_label_pred(test_set, test_label, self.device)
                    if auc > max_auc:
                        max_auc = auc
                        node_embedding = self.graph_enc.node_emb_layer.weight.data.cpu().numpy()
                        np.save(self.road_embedding_path,node_embedding)
                    if f1 > max_f1:
                        max_f1 = f1
                    
                    if auc >= max_auc and f1 >= max_f1:
                        patience = 50
                    else:
                        patience -= 1
                        if patience == 0:
                            self._logger.info("early stop")
                            self._logger.info("max_auc: " + str(max_auc))
                            self._logger.info("max_f1: " + str(max_f1))
                            self._logger.info("step " + str(count))
                            self._logger.info(loss.item())
                            return
                    self._logger.info("max_auc: " + str(max_auc))
                    self._logger.info("max_f1: " + str(max_f1))
                    self._logger.info("step " + str(count))
                    self._logger.info(loss.item())
                count += 1
                
        # node_embedding = self.graph_enc.node_emb_layer.weight.data.cpu().numpy()
        # np.save(self.road_embedding_path,node_embedding)
        # 在pipeline 会save
        # torch.save((self.state_dict(), self.optimizer.state_dict()), self.model_cache_file)
    
    def test_label_pred(self,  test_set, test_label, device):
        right = 0
        sum_num = 0
        test_set = test_set.clone().detach()
        pred = self(test_set)
        pred_prob = F.softmax(pred, -1)
        pred_scores = pred_prob[:, 1]
        auc = roc_auc_score(np.array(test_label), np.array(pred_scores.tolist()))
        self._logger.info("auc: " + str(auc))

        pred_loc = torch.argmax(pred, 1).tolist()
        right_pos = 0
        right_neg = 0
        wrong_pos = 0
        wrong_neg = 0
        for item1, item2 in zip(pred_loc, test_label):
            if item1 == item2:
                right += 1
                if item2 == 1:
                    right_pos += 1
                else:
                    right_neg += 1
            else:
                if item2 == 1:
                    wrong_pos += 1
                else:
                    wrong_neg += 1
            sum_num += 1
        recall_sum = right_pos + wrong_pos
        precision_sum = wrong_neg + right_pos
        if recall_sum == 0:
            recall_sum += 1
        if precision_sum == 0:
            precision_sum += 1
        recall = float(right_pos) / recall_sum
        precision = float(right_pos) / precision_sum
        if recall == 0 or precision == 0:
            self._logger.info("p/r/f:0/0/0")
            return 0.0, 0.0, 0.0, 0.0
        f1 = 2 * recall * precision / (precision + recall)
        self._logger.info("label prediction @acc @p/r/f: " + str(float(right) / sum_num) + " " + str(precision) +
                          " " + str(recall) + " " + str(f1))
        return precision, recall, f1, auc

class GraphEncoderTL(Module):
    def __init__(self, hparams, struct_assign, fnc_assign, struct_adj, device):
        super(GraphEncoderTL, self).__init__()
        self.hparams = hparams
        self.device = device
        self.struct_assign = struct_assign
        self.fnc_assign = fnc_assign
        self.struct_adj = struct_adj

        self.node_emb_layer = nn.Embedding(hparams.node_num, hparams.node_dims).to(self.device)
        self.type_emb_layer = nn.Embedding(hparams.type_num, hparams.type_dims).to(self.device)
        self.length_emb_layer = nn.Embedding(hparams.length_num, hparams.length_dims).to(self.device)
        self.lane_emb_layer = nn.Embedding(hparams.lane_num, hparams.lane_dims).to(self.device)

        self.tl_layer_1 = GraphEncoderTLCore(hparams, self.struct_assign, self.fnc_assign, self.device)
        self.tl_layer_2 = GraphEncoderTLCore(hparams, self.struct_assign, self.fnc_assign, self.device)
        self.tl_layer_3 = GraphEncoderTLCore(hparams, self.struct_assign, self.fnc_assign, self.device)

        self.init_feat = None

    def forward(self, node_feature, type_feature, length_feature, lane_feature, adj):
        node_emb = self.node_emb_layer(node_feature)
        type_emb = self.type_emb_layer(type_feature)
        length_emb = self.length_emb_layer(length_feature)
        lane_emb = self.lane_emb_layer(lane_feature)
        raw_feat = torch.cat([lane_emb, type_emb, length_emb, node_emb], 1)
        self.init_feat = raw_feat

        raw_feat = self.tl_layer_1(self.struct_adj, raw_feat, adj)
        raw_feat = self.tl_layer_2(self.struct_adj, raw_feat, adj)
        return raw_feat


def get_sparse_adj(adj, device):
    self_loop = np.eye(len(adj))
    adj = np.array(adj) + self_loop
    adj = sparse.coo_matrix(adj)

    adj_indices = torch.tensor(np.concatenate([adj.row[:, np.newaxis], adj.col[:, np.newaxis]], 1),
                               dtype=torch.long, device=device).t()
    adj_values = torch.tensor(adj.data, dtype=torch.float, device=device)
    adj_shape = adj.shape
    adj = torch.sparse.FloatTensor(adj_indices, adj_values, adj_shape).to(device)
    return adj.coalesce()


class GraphEncoderTLCore(Module):
    def __init__(self, hparams, struct_assign, fnc_assign, device):
        super(GraphEncoderTLCore, self).__init__()
        self.device = device
        self.struct_assign = struct_assign
        self.fnc_assign = fnc_assign

        self.fnc_gcn = GraphConvolution(
            in_features=hparams.hidden_dims,
            out_features=hparams.hidden_dims,
            device=device).to(self.device)

        self.struct_gcn = GraphConvolution(
            in_features=hparams.hidden_dims,
            out_features=hparams.hidden_dims,
            device=self.device).to(self.device)

        self.node_gat = SPGAT(
            in_features=hparams.hidden_dims,
            out_features=hparams.hidden_dims,
            alpha=hparams.alpha, dropout=hparams.dropout).to(self.device)

        self.l_c = torch.nn.Linear(hparams.hidden_dims * 2, 1).to(self.device)

        self.l_s = torch.nn.Linear(hparams.hidden_dims * 2, 1).to(self.device)

        self.sigmoid = nn.Sigmoid()

    def forward(self, struct_adj, raw_feat, raw_adj):
        # forward
        self.raw_struct_assign = self.struct_assign
        self.raw_fnc_assign = self.fnc_assign

        self.struct_assign = self.struct_assign / (F.relu(torch.sum(self.struct_assign, 0) - 1.0) + 1.0)
        self.fnc_assign = self.fnc_assign / (F.relu(torch.sum(self.fnc_assign, 0) - 1.0) + 1.0)

        self.struct_emb = torch.mm(self.struct_assign.t(), raw_feat)
        self.fnc_emb = torch.mm(self.fnc_assign.t(), self.struct_emb)

        # backward
        ## F2F
        self.fnc_adj = torch.sigmoid(torch.mm(self.fnc_emb, self.fnc_emb.t()))  # n_f * n_f
        self.fnc_adj = self.fnc_adj + torch.eye(self.fnc_adj.shape[0]).to(self.device) * 1.0
        self.fnc_emb = self.fnc_gcn(self.fnc_emb.unsqueeze(0), self.fnc_adj.unsqueeze(0)).squeeze()

        ## F2C
        fnc_message = torch.div(torch.mm(self.raw_fnc_assign, self.fnc_emb),
                                (F.relu(torch.sum(self.fnc_assign, 1) - 1.0) + 1.0).unsqueeze(1))

        self.r_f = self.sigmoid(self.l_c(torch.cat((self.struct_emb, fnc_message), 1)))
        self.struct_emb = self.struct_emb + 0.15 * fnc_message  # magic number: 0.15

        ## C2C
        struct_adj = F.relu(struct_adj - torch.eye(struct_adj.shape[1]).to(self.device) * 10000.0) + torch.eye(
            struct_adj.shape[1]).to(self.device) * 1.0
        self.struct_emb = self.struct_gcn(self.struct_emb.unsqueeze(0), struct_adj.unsqueeze(0)).squeeze()

        ## C2N
        struct_message = torch.mm(self.raw_struct_assign, self.struct_emb)
        self.r_s = self.sigmoid(self.l_s(torch.cat((raw_feat, struct_message), 1)))
        raw_feat = raw_feat + 0.5 * struct_message

        ## N2N
        raw_feat = self.node_gat(raw_feat, raw_adj)
        return raw_feat


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape,
                b):  # indices, value and shape define a sparse tensor, it will do mm() operation with b
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SPGAT(Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SPGAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, inputs, adj):
        inputs = inputs.squeeze()
        dv = 'cuda' if inputs.is_cuda else 'cpu'
        N = inputs.size()[0]
        edge_index = adj.indices()

        h = torch.mm(inputs, self.W)
        # h: N x out
        edge_h = torch.cat((h[edge_index[0, :], :], h[edge_index[1, :], :]), dim=1).t()  # 2*D x E
        values = self.a.mm(edge_h).squeeze()
        edge_value_a = self.leakyrelu(values)

        # softmax
        edge_value = torch.exp(edge_value_a - torch.max(edge_value_a))  # E
        e_rowsum = self.special_spmm(edge_index, edge_value, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1
        edge_value = self.dropout(edge_value)
        # edge_value: E
        h_prime = self.special_spmm(edge_index, edge_value, torch.Size([N, N]), h)
        # h_prime: N x out
        epsilon = 1e-15
        h_prime = h_prime.div(e_rowsum + torch.tensor([epsilon], device=dv))
        # h_prime: N x out
        if self.concat:  # if this layer is not last layer,
            return F.elu(h_prime)
        else:  # if this layer is last layer
            return h_prime


class GraphConvolution(Module):
    """
      Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, device, bias=True):
        super(GraphConvolution, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.5
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def norm(self, adj):
        node_num = adj.shape[-1]
        # add remaining self-loops
        self_loop = torch.eye(node_num, dtype=torch.float).to(self.device)
        self_loop = self_loop.reshape((1, node_num, node_num))
        self_loop = self_loop.repeat(adj.shape[0], 1, 1)
        adj_post = adj + self_loop
        # signed adjacent matrix
        deg_abs = torch.sum(torch.abs(adj_post), dim=-1)
        deg_abs_sqrt = deg_abs.pow(-0.5)
        diag_deg = torch.diag_embed(deg_abs_sqrt, dim1=-2, dim2=-1)

        norm_adj = torch.matmul(torch.matmul(diag_deg, adj_post), diag_deg)
        return norm_adj

    def forward(self, inputs, adj):
        support = torch.matmul(inputs, self.weight)
        adj_norm = self.norm(adj)
        output = torch.matmul(support.transpose(1, 2), adj_norm.transpose(1, 2))
        output = output.transpose(1, 2)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst

def get_next(it):
    res = None
    try:
        res = next(it)
    except StopIteration:
        pass
    return res