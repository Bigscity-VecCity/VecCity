import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
import time
import torch.optim as optim
from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel
from veccity.utils import need_train
import pdb
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import reset, uniform
import random
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
import pytorch_warmup as warmup
import math
import os


EPS = 1e-15


class HGI(AbstractReprLearningModel):
    def __init__(self,config,data_feature):
        super().__init__(config,data_feature)
        self._logger = getLogger()
        self.device = config.get('device')
        self.dim = config.get('output_dim', 64)
        self.dataset = config.get('dataset', '')
        self.epochs = config.get('max_epoch', 1)
        self.model = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.learning_rate = config.get('learning_rate', 0.006)
        self.weight_decay = config.get('weight_decay', 5e-4)

        self.alpha = config.get('alpha', 0.5)
        self.attention_head = config.get('attention_head', 4)
        self.max_norm = config.get('max_norm', 0.9)
        self.gamma = config.get('gamma', 1)
        self.warmup_period = config.get('warmup_period', 40)
        self.hgi_graph = data_feature.get('poi_graph')
        
        self.npy_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.dim)
        
        
    def run(self, train_dataloader=None, eval_dataloader=None):
        if not need_train(self.config):
            return
        start_time = time.time()

        """load the graph data of a study area"""
        data = self.hgi_graph.to(self.device)
        """load the Module"""
        model = HierarchicalGraphInfomax(
            hidden_channels=self.dim,
            poi_encoder=POIEncoder(data.num_features, self.dim),
            poi2region=POI2Region(self.dim, self.attention_head),
            region2city=lambda z, area: torch.sigmoid((z.transpose(0, 1) * area).sum(dim=1)),
            corruption=corruption,
            alpha=self.alpha,
        ).to(self.device)
        """load the optimizer, scheduler (including a warmup scheduler)"""
        
        self._logger.info(model)
        self._logger.info("start training,lr={},weight_dacay={}".format(self.learning_rate,self.weight_decay))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma, verbose=False)
        warmup_scheduler = warmup.LinearWarmup(optimizer, self.warmup_period)

        def train():
            model.train()
            optimizer.zero_grad()
            pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb = model(data)
            loss = model.loss(pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=self.max_norm)
            optimizer.step()
            with warmup_scheduler.dampening():
                scheduler.step()
            return loss.item()

        t = trange(1, self.epochs + 1)
        lowest_loss = math.inf
        region_emb_to_save = torch.FloatTensor(0)
        for epoch in t:
            loss = train()
            if loss < lowest_loss:
                """save the embeddings with the lowest loss"""
                region_emb_to_save = model.get_region_emb()
                lowest_loss = loss
            t.set_postfix(loss='{:.4f}'.format(loss), refresh=True)
            self._logger.info("Epoch {}, Loss {}".format(epoch, loss))

        t1 = time.time()-start_time
        outs = region_emb_to_save
        self._logger.info('cost time is '+str(t1/self.epochs))
        total_num = sum([param.nelement() for param in model.parameters()])
        total_num += outs.view(-1).shape[0]
        self._logger.info('Total parameter numbers: {}'.format(total_num))
        node_embedding = outs
        node_embedding = node_embedding.detach().cpu().numpy()
        np.save(self.npy_cache_file, node_embedding)
        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：(' + str(len(node_embedding)) + ',' + str(len(node_embedding[0])) + ')')



class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    """the aggregation from POIs to regions function based on multi-head attention mechanisms"""
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class POIEncoder(nn.Module):
    """POI GCN encoder"""
    def __init__(self, in_channels, hidden_channels):
        super(POIEncoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True, bias=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)
        return x


class POI2Region(nn.Module):
    """POI - region aggregation and GCN at regional level"""
    def __init__(self, hidden_channels, num_heads):
        super(POI2Region, self).__init__()
        self.PMA = PMA(dim=hidden_channels, num_heads=num_heads, num_seeds=1, ln=False)
        self.conv = GCNConv(hidden_channels, hidden_channels, cached=True, bias=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, zone, region_adjacency):
        region_emb = x.new_zeros((region_adjacency.max()+1, x.size()[1]))
        for index in range(zone.max() + 1):
            poi_index_in_region = (zone == index).nonzero(as_tuple=True)[0]
            region_emb[index] = self.PMA(x[poi_index_in_region].unsqueeze(0)).squeeze()
        region_emb = self.conv(region_emb, region_adjacency)
        region_emb = self.prelu(region_emb)
        return region_emb


def corruption(x):
    """corruption function to generate negative POIs through random permuting POI initial features"""
    return x[torch.randperm(x.size(0))]


class HierarchicalGraphInfomax(torch.nn.Module):
    r"""The Hierarchical Graph Infomax Module for learning region representations"""
    def __init__(self, hidden_channels, poi_encoder, poi2region, region2city, corruption, alpha):
        super(HierarchicalGraphInfomax, self).__init__()
        self.hidden_channels = hidden_channels
        self.poi_encoder = poi_encoder
        self.poi2region = poi2region
        self.region2city = region2city
        self.corruption = corruption
        self.alpha = alpha
        self.weight_poi2region = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.weight_region2city = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.region_embedding = torch.tensor(0)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.poi_encoder)
        reset(self.poi2region)
        reset(self.region2city)
        uniform(self.hidden_channels, self.weight_poi2region)
        uniform(self.hidden_channels, self.weight_region2city)

    def forward(self, data):
        """forward function to generate POI, region, and city representations"""
        pos_poi_emb = self.poi_encoder(data.x, data.edge_index, data.edge_weight)
        cor_x = self.corruption(data.x)
        neg_poi_emb = self.poi_encoder(cor_x, data.edge_index, data.edge_weight)
        region_emb = self.poi2region(pos_poi_emb, data.region_id, data.region_adjacency)
        self.region_embedding = region_emb
        neg_region_emb = self.poi2region(neg_poi_emb, data.region_id, data.region_adjacency)
        city_emb = self.region2city(region_emb, data.region_area)
        pos_poi_emb_list = []
        neg_poi_emb_list = []
        """hard negative sampling procedure"""
        for region in range(torch.max(data.region_id)+1):
            id_of_poi_in_a_region = (data.region_id == region).nonzero(as_tuple=True)[0]
            poi_emb_of_a_region = pos_poi_emb[id_of_poi_in_a_region]
            hard_negative_choice = random.random()
            if hard_negative_choice < 0.25:
                hard_example_range = ((data.coarse_region_similarity[region] > 0.6) & (data.coarse_region_similarity[region] < 0.8)).nonzero(as_tuple=True)[0]
                if hard_example_range.size()[0] > 0:
                    another_region_id = random.sample(hard_example_range.tolist(), 1)[0]
                else:
                    another_region_id = random.sample((set(data.region_id.tolist()) - set([region])), 1)[0]
            else:
                another_region_id = random.sample((set(data.region_id.tolist())-set([region])), 1)[0]
            id_of_poi_in_another_region = (data.region_id == another_region_id).nonzero(as_tuple=True)[0]
            poi_emb_of_another_region = pos_poi_emb[id_of_poi_in_another_region]
            pos_poi_emb_list.append(poi_emb_of_a_region)
            neg_poi_emb_list.append(poi_emb_of_another_region)
        return pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb

    def discriminate_poi2region(self, poi_emb_list, region_emb, sigmoid=True):
        values = []
        for region_id, region in enumerate(poi_emb_list):
            if region.size()[0] > 0:
                region_summary = region_emb[region_id]
                value = torch.matmul(region, torch.matmul(self.weight_poi2region, region_summary))
                values.append(value)
        values = torch.cat(values, dim=0)
        return torch.sigmoid(values) if sigmoid else values

    def discriminate_region2city(self, region_emb, city_emb, sigmoid=True):
        value = torch.matmul(region_emb, torch.matmul(self.weight_region2city, city_emb))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb):
        r"""Computes the mutual information maximization objective among the POI-region-city hierarchy."""
        pos_loss_region = -torch.log(
            self.discriminate_poi2region(pos_poi_emb_list, region_emb, sigmoid=True) + EPS).mean()
        neg_loss_region = -torch.log(
            1 - self.discriminate_poi2region(neg_poi_emb_list, region_emb, sigmoid=True) + EPS).mean()
        pos_loss_city = -torch.log(
            self.discriminate_region2city(region_emb, city_emb, sigmoid=True) + EPS).mean()
        neg_loss_city = -torch.log(
            1 - self.discriminate_region2city(neg_region_emb, city_emb, sigmoid=True) + EPS).mean()
        loss_poi2region = pos_loss_region + neg_loss_region
        loss_region2city = pos_loss_city + neg_loss_city
        return loss_poi2region * self.alpha + loss_region2city * (1 - self.alpha)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)

    def get_region_emb(self):
        return self.region_embedding.clone().cpu().detach()