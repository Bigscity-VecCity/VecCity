from logging import getLogger
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from torch import optim
from veccity.utils import need_train


from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel


class RelationGCN(nn.Module):
    def __init__(self, embedding_size, dropout, gcn_layers):
        super(RelationGCN, self).__init__()
        self.gcn_layers = gcn_layers
        self.dropout = dropout

        self.gcns = nn.ModuleList([GraphConv(in_feats=embedding_size, out_feats=embedding_size)
                                   for _ in range(self.gcn_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(embedding_size)
                                 for _ in range(self.gcn_layers - 1)])
        self.relation_transformation = nn.ModuleList([nn.Linear(embedding_size, embedding_size)
                                                      for _ in range(self.gcn_layers)])

    def forward(self, features, rel_emb, edge_index, is_training=True):
        n_emb = features
        poi_emb = features
        s_emb = features
        d_emb = features
        poi_r, s_r, d_r, n_r = rel_emb
        poi_edge_index, s_edge_index, d_edge_index, n_edge_index = edge_index
        for i in range(self.gcn_layers - 1):
            tmp = n_emb
            n_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](n_edge_index, torch.multiply(n_emb, n_r))))
            n_r = self.relation_transformation[i](n_r)
            if is_training:
                n_emb = F.dropout(n_emb, p=self.dropout)

            tmp = poi_emb
            poi_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i]( poi_edge_index, torch.multiply(poi_emb, poi_r))))
            poi_r = self.relation_transformation[i](poi_r)
            if is_training:
                poi_emb = F.dropout(poi_emb, p=self.dropout)

            tmp = s_emb
            s_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](s_edge_index, torch.multiply(s_emb, s_r))))
            s_r = self.relation_transformation[i](s_r)
            if is_training:
                s_emb = F.dropout(s_emb, p=self.dropout)

            tmp = d_emb
            d_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](d_edge_index, torch.multiply(d_emb, d_r))))
            d_r = self.relation_transformation[i](d_r)
            if is_training:
                d_emb = F.dropout(d_emb, p=self.dropout)

        n_emb = self.gcns[-1](n_edge_index, torch.multiply(n_emb, n_r))
        poi_emb = self.gcns[-1](poi_edge_index, torch.multiply(poi_emb, poi_r))
        s_emb = self.gcns[-1](s_edge_index, torch.multiply(s_emb, s_r))
        d_emb = self.gcns[-1](d_edge_index, torch.multiply(d_emb, d_r))

        # rel update
        n_r = self.relation_transformation[-1](n_r)
        poi_r = self.relation_transformation[-1](poi_r)
        s_r = self.relation_transformation[-1](s_r)
        d_r = self.relation_transformation[-1](d_r)

        return n_emb, poi_emb, s_emb, d_emb, n_r, poi_r, s_r, d_r


class CrossLayer(nn.Module):
    def __init__(self, embedding_size):
        super(CrossLayer, self).__init__()
        self.alpha_n = nn.Parameter(torch.tensor(0.95))
        self.alpha_poi = nn.Parameter(torch.tensor(0.95))
        self.alpha_d = nn.Parameter(torch.tensor(0.95))
        self.alpha_s = nn.Parameter(torch.tensor(0.95))

        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_size, num_heads=4)

    def forward(self, n_emb, poi_emb, s_emb, d_emb):
        stk_emb = torch.stack((n_emb, poi_emb, d_emb, s_emb))
        fusion, _ = self.attn(stk_emb, stk_emb, stk_emb)

        n_f = fusion[0] * self.alpha_n + (1 - self.alpha_n) * n_emb
        poi_f = fusion[1] * self.alpha_poi + (1 - self.alpha_poi) * poi_emb
        d_f = fusion[2] * self.alpha_d + (1 - self.alpha_d) * d_emb
        s_f = fusion[3] * self.alpha_s + (1 - self.alpha_s) * s_emb

        return n_f, poi_f, s_f, d_f


class AttentionFusionLayer(nn.Module):
    def __init__(self, embedding_size):
        super(AttentionFusionLayer, self).__init__()
        self.q = nn.Parameter(torch.randn(embedding_size))
        self.fusion_lin = nn.Linear(embedding_size, embedding_size)

    def forward(self, n_f, poi_f, s_f, d_f):
        n_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(n_f)) * self.q, dim=1))
        poi_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(poi_f)) * self.q, dim=1))
        s_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(s_f)) * self.q, dim=1))
        d_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(d_f)) * self.q, dim=1))

        w_stk = torch.stack((n_w, poi_w, s_w, d_w))
        w = torch.softmax(w_stk, dim=0)

        region_feature = w[0] * n_f + w[1] * poi_f + w[2] * s_f + w[3] * d_f
        return region_feature


class HRE(nn.Module):
    def __init__(self, embedding_size, dropout, gcn_layers):
        super(HRE, self).__init__()

        self.relation_gcns = RelationGCN(embedding_size, dropout, gcn_layers)

        self.cross_layer = CrossLayer(embedding_size)

        self.fusion_layer = AttentionFusionLayer(embedding_size)

    def forward(self, features, rel_emb, edge_index, is_training=True):
        poi_emb, s_emb, d_emb, n_emb, poi_r, s_r, d_r, n_r = self.relation_gcns(features, rel_emb,
                                                                                edge_index, is_training)
        n_f, poi_f, s_f, d_f = self.cross_layer(n_emb, poi_emb, s_emb, d_emb)
        region_feature = self.fusion_layer(n_f, poi_f, s_f, d_f)

        n_f = torch.multiply(region_feature, n_r)
        poi_f = torch.multiply(region_feature, poi_r)
        s_f = torch.multiply(region_feature, s_r)
        d_f = torch.multiply(region_feature, d_r)

        return region_feature, n_f, poi_f, s_f, d_f


class HREP(AbstractReprLearningModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device')
        self.model = config.get('model')
        self.dataset = config.get('dataset')
        self.exp_id = config.get('exp_id')
        self.epochs = config.get("max_epoch", 10)
        self.embedding_size = config.get("output_dim", 144)
        self.output_dim = self.embedding_size
        self.weight_dacay = config.get('weight_dacay', 1e-3)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.dropout = config.get("dropout", 0.1)
        self.gcn_layers = config.get("gcn_layers", 3)
        self.importance_k = config.get("importance_k", 10)
        self.num_regions = data_feature.get('num_regions')
        self.data_feature = data_feature
        self.txt_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './veccity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.net = HRE(self.embedding_size, self.dropout, self.gcn_layers).to(self.device)
        
    def mob_loss(self, s_emb, d_emb, mob):
        epsilon = 0.0001
        inner_prod = torch.mm(s_emb, d_emb.T)
        ps_hat = F.softmax(inner_prod, dim=-1)
        inner_prod = torch.mm(d_emb, s_emb.T)
        pd_hat = F.softmax(inner_prod, dim=-1)
        loss = torch.sum(-torch.mul(mob, torch.log(ps_hat + epsilon)) -
                        torch.mul(mob, torch.log(pd_hat + epsilon)))
        return loss
    
    def pair_sample(self, neighbor):
        positive = torch.zeros(self.num_regions, dtype=torch.long)
        negative = torch.zeros(self.num_regions, dtype=torch.long)
        for i in range(self.num_regions):
            region_idx = np.random.randint(len(neighbor[i]))
            pos_region = neighbor[i][region_idx]
            positive[i] = pos_region
        for i in range(self.num_regions):
            neg_region = np.random.randint(self.num_regions)
            while neg_region in neighbor[i] or neg_region == i:
                neg_region = np.random.randint(self.num_regions)
            negative[i] = neg_region
        return positive, negative
        
    def run(self, data=None,eval_data=None):
        if not need_train(self.config):
            return
        start_time = time.time()
        features = self.data_feature.get('features')
        rel_emb = self.data_feature.get('rel_emb')
        edge_index = self.data_feature.get('edge_index')
        poi_similarity = self.data_feature.get('poi_similarity')
        mobility = self.data_feature.get('mobility')
        neighbor = self.data_feature.get('neighbor')
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_dacay)
        loss_fn1 = torch.nn.TripletMarginLoss()
        loss_fn2 = torch.nn.MSELoss()
        final_loss = 0
        self._logger.info("start training,lr={},weight_dacay={}".format(self.learning_rate,self.weight_dacay))
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            region_emb, n_emb, poi_emb, s_emb, d_emb = self.net(features, rel_emb, edge_index)

            pos_idx, neg_idx = self.pair_sample(neighbor)

            geo_loss = loss_fn1(n_emb, n_emb[pos_idx], n_emb[neg_idx])

            m_loss = self.mob_loss(s_emb, d_emb, mobility)

            poi_loss = loss_fn2(torch.mm(poi_emb, poi_emb.T), poi_similarity)
            loss = poi_loss + m_loss + geo_loss
            loss.backward()
            optimizer.step()

            self._logger.info("Epoch {}, Loss {}".format(epoch, loss.item()))
            final_loss = loss.item()

        t1 = time.time()-start_time
        self._logger.info('cost time is '+str(t1/self.epochs))
        total_num = sum([param.nelement() for param in self.net.parameters()])
        embs = [region_emb, n_emb, poi_emb, s_emb, d_emb]
        for emb in embs:
            total_num += emb.view(-1).shape[0]
        self._logger.info('Total parameter numbers: {}'.format(total_num))
        region_emb = region_emb.cpu().detach().numpy()
        np.save(self.npy_cache_file, region_emb)
        
        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：(' + str(len(region_emb)) + ',' + str(len(region_emb[0])) + ')')
        return final_loss
