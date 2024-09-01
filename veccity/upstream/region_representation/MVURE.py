import math
from logging import getLogger
import numpy as np
import torch
import torch.nn as nn
import time
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import torch.optim as optim
from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel
from veccity.utils import need_train

#[2020-IJCAI Multi-View Joint Graph Representation Learning for Urban Region Embedding]
class MVURE(AbstractReprLearningModel):
    def __init__(self,config,data_feature):
        super().__init__(config,data_feature)
        self.mob_adj = data_feature.get("mob_adj")
        self.s_adj_sp = data_feature.get("s_adj_sp")
        self.t_adj_sp = data_feature.get("t_adj_sp")
        self.poi_adj = data_feature.get("poi_adj")
        self.poi_adj_sp = data_feature.get("poi_adj_sp")
        self.feature = data_feature.get("feature")
        self.num_nodes = data_feature.get("num_nodes")
        self.geo_to_ind = data_feature.get('geo_to_ind', None)
        self.ind_to_geo = data_feature.get('ind_to_geo', None)
        self._logger = getLogger()
        self.device = config.get('device')
        self.output_dim = config.get('output_dim', 96)
        self.is_directed = config.get('is_directed', True)
        self.dataset = config.get('dataset', '')
        self.iter = config.get('max_epoch', 2000)
        self.model = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.weight_dacay = config.get('weight_dacay', 1e-3)
        self.learning_rate = config.get('learning_rate', 0.005)
        self.early_stopping = config.get('early_stopping',10)
        self.txt_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './veccity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.mvure_model = MVURE_Layer(self.mob_adj, self.s_adj_sp, self.t_adj_sp, self.poi_adj, self.feature[0],
                                 self.feature.shape[2], self.output_dim, self.device)
        
    def run(self, train_dataloader=None, eval_dataloader=None):
        if not need_train(self.config):
            return
        start_time = time.time()
        self.feature = self.preprocess_features(self.feature)
        self.mvure_model.to(self.device)
        self._logger.info(self.mvure_model)
        optimizer = optim.Adam(self.mvure_model.parameters(),lr=self.learning_rate, weight_decay=self.weight_dacay)
        item_num, _ = self.mob_adj.shape
        self._logger.info("start training,lr={},weight_dacay={}".format(self.learning_rate,self.weight_dacay))
        outs = None
        for epoch in range(self.iter):
            self.mvure_model.train()
            outs, loss_embedding = self.mvure_model()
            loss = self.mvure_model.calculate_loss(loss_embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._logger.info("Epoch {}, Loss {}".format(epoch, loss.item()))
        t1 = time.time()-start_time
        self._logger.info('cost time is '+str(t1/self.iter))
        total_num = sum([param.nelement() for param in self.mvure_model.parameters()])
        total_num += outs.view(-1).shape[0]
        total_num += loss_embedding.view(-1).shape[0]
        self._logger.info('Total parameter numbers: {}'.format(total_num))
        node_embedding = outs
        node_embedding = node_embedding.detach().cpu().numpy()
        np.save(self.npy_cache_file, node_embedding)
        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：(' + str(len(node_embedding)) + ',' + str(len(node_embedding[0])) + ')')

    def preprocess_features(self,feature):
        """Row-normalize feature matrix and convert to tuple representation"""
        feature_new = feature[0]
        colvar = np.var(feature_new, axis=1, keepdims=True)
        colmean = np.mean(feature_new, axis=1, keepdims=True)
        c_inv = np.power(colvar, -0.5)
        c_inv[np.isinf(c_inv)] = 0.
        feature_new = np.multiply((feature_new - colmean), c_inv)
        feature_new = feature_new[np.newaxis]
        return feature_new

    def adj_to_bias(self,adj, sizes, nhood=1):
        adj = adj[np.newaxis]
        nb_graphs = adj.shape[0]  # num_graph个图
        mt = np.empty(adj.shape)  # 输出矩阵的形状和adj相同
        # 图g的转换
        for g in range(nb_graphs):
            mt[g] = np.eye(adj.shape[1])  # 与g形状相同的对角矩阵
            for _ in range(nhood):  # 通过self-loop构建K阶邻接矩阵，即A^(K),这里K=1
                mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
            # 大于0的置1，小于等于0的保持不变
            for i in range(sizes[g]):
                for j in range(sizes[g]):
                    if mt[g][i][j] > 0.0:
                        mt[g][i][j] = 1.0
        # mt中1的位置为0，位置为0的返回很小的负数-1e9
        return -1e9 * (1.0 - mt)

class self_attn(nn.Module):
    def __init__(self,hidden_dim, device):
        #这里的input*dim和hidden_dim是对于每一个节点展平后的维度，因此传入前要乘上num_nodes
        super(self_attn,self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        # self.Q_linear = nn.Linear(input_dim,hidden_dim)
        # self.K_linear = nn.Linear(input_dim,hidden_dim)

    def forward(self,inputs):
        """
        :param inputs: [num_views,num_nodes,embedding_dim]
        :return:[num_views,num_nodes,embedding_dim],每一个视图做一遍注意力机制
        """
        num_views,num_nodes,embedding_dim = inputs.shape
        inputs_3dim = inputs
        result = torch.zeros([num_views , num_nodes, embedding_dim], dtype=torch.float32).to(self.device)
        self.input_dim = embedding_dim
        Q_linear = nn.Linear(self.input_dim, self.hidden_dim, device=self.device)
        K_linear = nn.Linear(self.input_dim, self.hidden_dim, device=self.device)
        Q = Q_linear(inputs)
        Q = Q.reshape([num_views,num_nodes*self.hidden_dim])
        Q = torch.unsqueeze(Q,-1)
        K = K_linear(inputs)
        K = K.reshape([num_views, num_nodes * self.hidden_dim])
        K = torch.unsqueeze(K,-1)
        d_k = math.sqrt(self.hidden_dim*num_nodes)
        attn = torch.bmm(Q.transpose(-2,-1),K)/d_k
        attn = torch.squeeze(attn)
        attn = F.softmax(attn)
        for i in range(num_views):
            result[i] += (attn[i]*inputs_3dim[i])
        return result

class mv_attn(nn.Module):
    def __init__(self, device):
        super(mv_attn,self).__init__()
        #输入为单视图表征，输出权值
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self,inputs):
        """
        :param inputs: [num_views,num_nodes,embedding_dim]
        :return:
        """
        num_views,num_nodes,embedding_dim = inputs.shape
        inputs_3dim = inputs
        inputs = inputs.view(num_views, num_nodes * embedding_dim)
        self.mlp = nn.Linear(num_nodes * embedding_dim, 1, device=self.device)
        omega = self.mlp(inputs)
        omega = self.sigmoid(omega)
        omega = torch.squeeze(omega)
        result = torch.zeros([num_nodes, embedding_dim], dtype=torch.float32).to(self.device)
        for i in range(num_views):
            result += (omega[i]*inputs_3dim[i])
        return result

class MVURE_Layer(nn.Module):
    def __init__(self,mob_adj,s_graph,t_graph,poi_graph,feature,input_dim,output_dim, device):
        super(MVURE_Layer,self).__init__()
        self.device = device
        self.mob_adj = torch.tensor(mob_adj).to(torch.float32).to(self.device)
        self.inputs = torch.from_numpy(feature).to(torch.float32).to(device)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = 8
        out_feat_num = self.output_dim//self.num_heads
        self.s_gat = GATConv(in_feats=self.inputs.shape[-1],out_feats=out_feat_num,num_heads=self.num_heads,attn_drop=0.2,activation=F.relu)
        self.t_gat = GATConv(in_feats=self.inputs.shape[-1],out_feats=out_feat_num,num_heads=self.num_heads,attn_drop=0.2,activation=F.relu)
        self.poi_gat = GATConv(in_feats=self.inputs.shape[-1],out_feats=out_feat_num,num_heads=self.num_heads,attn_drop=0.2,activation=F.relu)
        self.num_nodes = feature.shape[-2]
        self.fused_layer = self_attn(out_feat_num, self.device)
        self.fused_layer.to(self.device)
        self.mv_layer = mv_attn(self.device)
        self.mv_layer.to(self.device)
        self.alpha = 0.8
        self.beta = 0.5
        self.s_graph = self.construct_dgl_graph(s_graph)
        self.t_graph = self.construct_dgl_graph(t_graph)
        self.poi_graph_org=torch.tensor(poi_graph).to(torch.float32).to(self.device)
        self.poi_graph = self.construct_dgl_graph(poi_graph)

    

    def construct_dgl_graph(self,adj_mx):
        """
        :param adj_mx:邻接矩阵，[num_nodes,num_nodes],np.array
        :return: dgl_graph,将邻接矩阵中大于0的全部算成一条边，
        """
        num_nodes = adj_mx.shape[0]
        num_edges = np.count_nonzero(adj_mx)
        g = dgl.DGLGraph()
        g = g.to(self.device)
        g.add_nodes(self.num_nodes)
        src_index = []
        dst_index = []
        cost = torch.zeros(num_edges).to(self.device)
        edge_cnt = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_mx[i][j] > 0:
                    src_index.append(i)
                    dst_index.append(j)
                    cost[edge_cnt] = adj_mx[i][j]
                    edge_cnt += 1
        g.add_edges(src_index, dst_index)
        g = dgl.add_self_loop(g)
        g.edges[src_index,dst_index].data['w'] = cost

        return g

    def forward(self):
        s_out = self.s_gat(graph = self.s_graph,feat = self.inputs)#[num_nodes,num_heads,dim]
        s_out = s_out.reshape([s_out.shape[0],s_out.shape[1]*s_out.shape[2]])

        t_out = self.t_gat(graph = self.t_graph,feat = self.inputs)
        t_out = t_out.reshape([t_out.shape[0], t_out.shape[1] * t_out.shape[2]])

        poi_out = self.poi_gat(graph=self.poi_graph, feat=self.inputs)
        poi_out = poi_out.reshape([poi_out.shape[0], poi_out.shape[1] * poi_out.shape[2]])

        single_view_out = torch.stack([s_out,t_out,poi_out],dim = 0)
        fused_out = self.fused_layer(single_view_out)
        s_out = self.alpha * fused_out[0] + (1 - self.alpha) * s_out
        t_out = self.alpha * fused_out[1] + (1 - self.alpha) * t_out
        poi_out = self.alpha * fused_out[2] + (1 - self.alpha) * poi_out

        fused_out = torch.stack([s_out, t_out, poi_out], dim=0)
        mv_out = self.mv_layer(fused_out)
        s_out = self.beta * s_out + (1 - self.beta) * mv_out
        t_out = self.beta * t_out + (1 - self.beta) * mv_out
        poi_out = self.beta * poi_out + (1 - self.beta) * mv_out

        result = torch.stack([s_out, t_out, poi_out], dim=0)
        return mv_out, result

    def calculate_loss(self,embedding):
        """
        :param embedding:Tensor[num_views,num_nodes,embedding_dim]
        :return:
        """
        s_embeddings = embedding[0]
        t_embeddings = embedding[1]
        poi_embeddings = embedding[2]
        return self.calculate_mob_loss(s_embeddings,t_embeddings) +\
            self.calculate_poi_loss(poi_embeddings)

    def calculate_mob_loss(self,s_embeddings,t_embeddings):
        """
        :param s_embeddings:tensor[num_nodes,embedding_dim]
        :param t_embeddings: tensor[num_nodes,embedding_dim]
        :param mob_adj: np.array[num_nodes,num_nodes]
        :return:
        """

        inner_prod = self.pairwise_inner_product(s_embeddings, t_embeddings)
        phat = torch.softmax(inner_prod,dim=-1)
        loss = torch.sum(-torch.mul(self.mob_adj, torch.log(phat + 0.0001)))
        inner_prod = self.pairwise_inner_product(t_embeddings, s_embeddings)
        phat = torch.softmax(inner_prod,dim=-1)
        loss += torch.sum(-torch.mul(torch.transpose(self.mob_adj, 0, 1), torch.log(phat + 0.0001)))
        return loss


    def calculate_poi_loss(self,embedding):
        """
        :param embedding: tensor[num_nodes,embedding_dim]
        :param poi_adj: np.array[num_nodes,num_nodes]
        :return:
        """
        inner_prod = self.pairwise_inner_product(embedding, embedding)
        loss_function = nn.MSELoss(reduction="sum")
        loss = loss_function(inner_prod,self.poi_graph_org)
        return loss

    def pairwise_inner_product(self,mat_1, mat_2):
        result = torch.mm(mat_1, mat_2.t())
        return result



