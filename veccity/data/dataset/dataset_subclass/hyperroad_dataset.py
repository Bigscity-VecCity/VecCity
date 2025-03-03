from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import os
import torch
import pickle
import time
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
from logging import getLogger

from veccity.data.dataset import AbstractDataset
from veccity.data.preprocess import preprocess_all, cache_dir

class Road(Dataset):
    def __init__(self, edges, road_network, hyper_network, road2hyper, road_feature, hyper_feature, 
                    num_roads, num_hypers, neg_edge_size, neg_hyper_size, sampling='uniform'):
        self.edges=edges
        self.road2hyper=road2hyper
        self.road_feature=road_feature
        self.hyper_feature=hyper_feature
        self.road_network=road_network
        self.hyper_network=hyper_network
        self.num_roads=num_roads
        self.num_hypers=num_hypers
        self.neg_edge_size=neg_edge_size
        self.neg_hyper_size=neg_hyper_size

        self.sampling=sampling
    
    def __getitem__(self, idx):
        edge = self.edges[idx]
        anchor_node = edge[0]

        pos_edge = edge[1]
        neg_edges = self.edge_negative_sampling(anchor_node, pos_edge)

        # simple strategy for batches
        hyperedges = self.road2hyper[anchor_node]
        pos_hyper = np.random.choice(hyperedges, 1)[0]
        neg_hypers = self.hyperedge_negative_sampling(anchor_node, pos_hyper)

        hyper_class = self.hyper_feature[pos_hyper]['cluster']
        if hyper_class=='N':
            hyper_class = 0
        else:
            hyper_class = 1

        l_lanes = self.road_feature[anchor_node]['lanes']
        l_maxspeed = self.road_feature[anchor_node]['maxspeed']
        l_oneway = self.road_feature[anchor_node]['oneway']

        sample = {
            'anchor': torch.LongTensor([anchor_node]),
            'pos_edge': torch.LongTensor([pos_edge]),
            'neg_edges': torch.LongTensor(neg_edges),
            'pos_hyper': torch.LongTensor([pos_hyper]),
            'neg_hypers': torch.LongTensor(neg_hypers),
            'hyper_class': torch.LongTensor([hyper_class]),
            'l_lanes': torch.LongTensor([l_lanes]),
            'l_maxspeed': torch.LongTensor([l_maxspeed]),
            'l_oneway': torch.LongTensor([l_oneway]),
        }
        return sample

    def __len__(self):
        return len(self.edges)
    
    def edge_negative_sampling(self, node1, node2):
        sampling = self.sampling
        if sampling == "uniform":
            hold_nodes = [node1]
            neighs = list(np.nonzero(self.road_network[node1])[0])
            hold_nodes += neighs

            neg_edges = set()
            while len(neg_edges) < self.neg_edge_size:
                neg_edge = np.random.choice(self.num_roads, 1)[0]
                if neg_edge not in hold_nodes:
                    neg_edges.add(neg_edge)
            neg_edges = list(neg_edges)
            return neg_edges

        if sampling == "geo_sampling" or sampling == "graph_sampling":
            hold_nodes = [node1]
            neighs = list(np.nonzero(self.road_network[node1])[0])
            hold_nodes += neighs

            p = self.road_sampling[node1]
            neg_edges = set()
            while len(neg_edges) < self.neg_edge_size:
                neg_edge = np.random.choice(self.num_roads, 1, p=p)[0]
                if neg_edge not in hold_nodes:
                    neg_edges.add(neg_edge)
            neg_edges = list(neg_edges)
            return neg_edges

    # can sample node or sample hyper edge
    # double parameters, we do not need the hyper-emb in fact
    def hyperedge_negative_sampling(self, node, hyper_edge):
        sampling = self.sampling
        if sampling == "uniform":
            hold_hypers = self.road2hyper[node]

            neg_hypers = set()
            while len(neg_hypers) < self.neg_hyper_size:
                neg_hyper = np.random.choice(self.num_hypers, 1)[0]
                if neg_hyper not in hold_hypers:
                    neg_hypers.add(neg_hyper)
            neg_hypers = list(neg_hypers)
            return neg_hypers

        if sampling == "geo_sampling" or sampling == "graph_sampling":
            hold_hypers = self.road2hyper[node]

            p = self.hyper_sampling[node]
            neg_hypers = set()
            while len(neg_hypers) < self.neg_hyper_size:
                neg_hyper = np.random.choice(self.num_hypers, 1, p=p)[0]
                if neg_hyper not in hold_hypers:
                    neg_hypers.add(neg_hyper)
            neg_hypers = list(neg_hypers)
            return neg_hypers


class HyperRoadDataset(AbstractDataset):
    def __init__(self,config):
        self.config=config
        preprocess_all(config)
        self.device = config.get('device')
        self._logger = getLogger()
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        # 加载所有road的tag标签
        data_cache_dir = os.path.join(cache_dir, self.dataset)
        self.road_geo_path = os.path.join(data_cache_dir, 'road.csv')
        self.road_geo_df = pd.read_csv(self.road_geo_path, delimiter=',')
        self.road_tag = np.array(self.road_geo_df['highway'])
        self.road_length = np.array(self.road_geo_df['length'])
        self.road_num = len(self.road_length)
        self.traj_path = os.path.join(data_cache_dir, 'traj_road_train.csv')
        self.adj_json_path = os.path.join(data_cache_dir, 'road_neighbor.json')
        self.road_feature_path = os.path.join(data_cache_dir, 'road_features.csv')
        self.region_geo_path = os.path.join(data_cache_dir, 'region.csv')
        self.region_geo_df = pd.read_csv(self.region_geo_path, delimiter=',')
        self.region_num = len(self.region_geo_df)
        self.road_geometry = gpd.GeoSeries.from_wkt(self.road_geo_df['geometry'])
        self.sampling = config.get('sampling','uniform')
        self.centroid = self.road_geometry.centroid
        #统计lanes,oneway和speed的类别总数用于cls层
        self.lanes_cls = self.road_geo_df['lanes'].max()+1
        self.oneway_cls = self.road_geo_df['oneway'].max()+1
        self.speed_cls = self.road_geo_df['maxspeed'].max()+1
        
        self.road_adj = self.construct_road_adj()
        self.hyper_adj = self.construct_hyper_adj()
        self.road_feature = self.construct_road_feature()
        self.hyper_feature = self.construct_hyper_feature()

        self._logger.info(f'road_netword: {int(self.road_adj.sum())}')
        self._logger.info(f'hyper_netword: {int(self.hyper_adj.sum())}')

        self.neg_edge_size = config.get('neg_edge_size',5)
        self.neg_hyper_size = config.get('neg_hyper_size',2)
        self.num_roads = self.road_adj.shape[0]
        self.num_hypers = self.hyper_adj.shape[1]

        clusterset = set()
        for hyperedge in self.hyper_feature:
            cluster = self.hyper_feature[hyperedge]['cluster']
            clusterset.add(cluster)
        self.num_classes = len(clusterset)

        self._logger.info(f"num_roads: {self.num_roads} num_hypers: {self.num_hypers} num_classes: {self.num_classes}")
        
        coords, input_x = [], []
        input_feats = []
        for road in range(self.road_adj.shape[0]):
            input_x.append(road)
            coords.append(list(self.road_feature[road]['midpoint']))
            input_feats.append([self.road_feature[road]['lanes'], self.road_feature[road]['maxspeed'], self.road_feature[road]['oneway']])
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(input_feats)
        onehot_input_feats = enc.transform(input_feats).toarray()
        self.one_hot_dim = onehot_input_feats.shape[-1]
        coords = torch.FloatTensor(coords)
        input_x = torch.LongTensor(input_x)
        onehot_input_feats = torch.FloatTensor(onehot_input_feats)
        self.coords = coords
        self.input_x = input_x
        self.onehot_input_feats = onehot_input_feats

        self.road2hyper = {}
        self.hyper2road = {}
        num_roads = 0
        for hyperedge in range(self.hyper_adj.shape[1]):
            roadset = np.nonzero(self.hyper_adj[:,hyperedge])[0]
            self.hyper2road[hyperedge] = roadset
            for road in roadset:
                if road not in self.road2hyper:
                    self.road2hyper[road] = []
                self.road2hyper[road].append(hyperedge)
            num_roads += len(roadset)
        self._logger.info(f"road2hyper: {len(self.road2hyper)} hyper2road: {len(self.hyper2road)} Num: {num_roads}")

        edges = np.nonzero(self.road_adj)
        self.edges = set()
        for i, roadi in enumerate(edges[0]):
            roadj = edges[1][i]
            key = (roadi, roadj)
            self.edges.add(key)
        self.edges = list(self.edges)
        self._logger.info(f"edges in roadnetwork {len(self.edges)}")

        self.G = sp.coo_matrix(self.road_adj)
        self.H = sp.coo_matrix(self.hyper_adj)
        self._logger.info(f"graph for GNN {self.G.shape}")
        self._logger.info(f"hypergraph for HGNN {self.H.shape}")

        DvG = np.sum(self.road_adj, axis=1) ** (-1/2)
        DvG[np.isinf(DvG)] = 0.0
        DvG = sp.diags(DvG)
        self._logger.info(f"DvG {DvG.shape}")

        DH = np.sum(self.hyper_adj, axis=0) ** (-1.0)
        DH = sp.diags(DH)
        self._logger.info(f"DH {DH.shape}")

        DvH = np.sum(self.hyper_adj, axis=1) ** (-1/2)
        DvH = sp.diags(DvH)
        self._logger.info(f"DvH {DvH.shape}")

        # self.norm_GG = DvG  * DvG * self.G
        norm_GG = DvG * self.G * DvG
        norm_HH = DH * self.H.T
        norm_HG = DvH * DvH * self.H
        norm_GraphSAGE = DvG * DvG * self.G

        self.norm_GG = self.sparse_to_tensor(norm_GG)
        self.norm_HH = self.sparse_to_tensor(norm_HH)
        self.norm_HG = self.sparse_to_tensor(norm_HG)
        self.norm_GraphSAGE = self.sparse_to_tensor(norm_GraphSAGE)

        if self.sampling == "geo_sampling":
            road_dis_matrix = self.get_geo_dis(self.dataset)
            road_dis_matrix += 1.0 # shortest can be 1.0 (to handle these 0.0 and sampling prob)
            self.road_dis_matrix = road_dis_matrix
            data_file = "../resource/data/" + self.dataset + "/hyper_geo_dis_matrix.pkl"
            hyper_dis_matrix = self.init_hyper_distance(data_file)
            self.hyper_dis_matrix = hyper_dis_matrix
            road_sampling, hyper_sampling = self.get_sampling_matrix()
            self.road_sampling = road_sampling
            self.hyper_sampling = hyper_sampling

        if self.sampling == "graph_sampling":
            road_dis_matrix = self.get_graph_dis(self.dataset)
            road_dis_matrix += 1.0 # shortest can be 1.0 (to handle these 0.0 and sampling prob)
            self.road_dis_matrix = road_dis_matrix
            data_file = "../resource/data/" + self.dataset + "/hyper_graph_dis_matrix.pkl"
            hyper_dis_matrix = self.init_hyper_distance(data_file)
            self.hyper_dis_matrix = hyper_dis_matrix
            road_sampling, hyper_sampling = self.get_sampling_matrix()
            self.road_sampling = road_sampling
            self.hyper_sampling = hyper_sampling
        
        self.train_data = Road(self.edges,self.road_adj,self.hyper_adj,self.road2hyper,self.road_feature,self.hyper_feature,self.num_roads,self.num_hypers,self.neg_edge_size,self.neg_hyper_size,self.sampling)
        self.batch_size=config.get('batch_size',32)
        self.train_dataloader = DataLoader(self.train_data,batch_size=self.batch_size,shuffle=True)
        
    def construct_road_adj(self):
        road_adj = np.zeros(shape=[self.road_num, self.road_num])
        # 构建路网的邻接关系
        with open(self.adj_json_path, 'r', encoding='utf-8') as fp:
            road_adj_data = json.load(fp)
        for road in range(self.road_num):
            road_adj[road][road] = 1
            for neighbor in road_adj_data[str(road)]:
                road_adj[road][neighbor] = 1
        return road_adj
    
    def construct_hyper_adj(self):
        hyper_adj = np.zeros(shape=[self.road_num,self.region_num])
        df = pd.read_csv(os.path.join(self.data_path, self.dataset + '.grel'))
        road2region_df = df[df['rel_type'] == 'road2region']
        for _, row in road2region_df.iterrows():
            x = int(row['orig_geo_id']) - self.region_num
            y = int(row['dest_geo_id'])
            hyper_adj[x][y] = 1
        return hyper_adj
    
    def construct_road_feature(self):
        road_feature = dict()
        for i in range(len(self.road_geo_df)):
            road_feature_i = dict()
            midpoint = self.centroid[i]
            longitude = midpoint.x
            latitude = midpoint.y

            # 构建包含经纬度的元组
            coordinates = (longitude, latitude)
            road_feature_i['midpoint'] = coordinates
            road_feature_i['highway'] = self.road_geo_df.loc[i,'highway']
            road_feature_i['lanes'] = self.road_geo_df.loc[i, 'lanes']
            road_feature_i['tunnel'] = self.road_geo_df.loc[i, 'tunnel']
            road_feature_i['bridge'] = self.road_geo_df.loc[i, 'bridge']
            road_feature_i['roundabout'] = self.road_geo_df.loc[i, 'roundabout']
            road_feature_i['oneway'] = self.road_geo_df.loc[i, 'oneway']
            road_feature_i['length'] = self.road_geo_df.loc[i, 'length']
            road_feature_i['maxspeed'] = self.road_geo_df.loc[i, 'maxspeed']
            road_feature[i] = road_feature_i
        return road_feature
    
    def construct_hyper_feature(self):
        hyper_feature = dict()
        feature_col = self.config.get('region_clf_label', None)
        for i in range(len(self.region_geo_df)):
            hyper_feature_i = dict()
            if feature_col is not None:
                hyper_feature_i['cluster'] = self.region_geo_df.loc[i, feature_col]
            else:
                hyper_feature_i['cluster'] = i
            hyper_feature[i] = hyper_feature_i
        return hyper_feature
    
    def init_hyper_distance(self, data_file):
        if os.path.exists(data_file):
            hyper_dis_matrix = pickle.load(open(data_file, "rb"))
        else:
            hyper_dis_matrix = np.zeros((len(self.road2hyper), len(self.hyper2road)))
            start = time.time()
            for i, road in enumerate(self.road2hyper):
                if (i+1) % 2000 == 0:
                    end = time.time()
                    self._logger.info(f"{i + 1} nodes have been sampled, cost time {end - start}")
                for j, hyperedge in enumerate(self.hyper2road):
                    dist = self.get_dist(road, hyperedge)
                    hyper_dis_matrix[road][hyperedge] = dist

            with open(data_file, "wb") as f:
                pickle.dump(hyper_dis_matrix, f)
        return hyper_dis_matrix

    def get_dist(self, road, hyperedge):
        roadset = self.hyper2road[hyperedge]
        avg_dist = 0.0
        for road_h in roadset:
            min_node = min(road, road_h)
            max_node = max(road, road_h)
            dist = self.road_dis_matrix[min_node][max_node]
            avg_dist += dist
        return float(avg_dist) / len(roadset)

    def get_sampling_matrix(self):
        road_dis_matrix = self.road_dis_matrix
        hyper_dis_matrix = self.hyper_dis_matrix

        road_sampling = road_dis_matrix / road_dis_matrix.sum(1)[:, None]
        hyper_sampling = hyper_dis_matrix / hyper_dis_matrix.sum(1)[:, None]
        return road_sampling, hyper_sampling
    
    def get_geo_dis(self,city="Singapore"):
        data_file = self.data_path + city + "/geo_dis_matrix.pkl"
        if os.path.exists(data_file):
            geo_dis_matrix = pickle.load(open(data_file, "rb"))
        else:
            G, road_feature = preproces_network_dgl(city)
            nodes = list(G.nodes())
            geo_dis_matrix = np.zeros((len(nodes), len(nodes)))

            start = time.time()
            for i in range(len(nodes)):
                if (i+1) % 100 == 0:
                    end = time.time()
                    self._logger.info(f"{i + 1} nodes have been sampled, cost time {end - start}")

                for j in range(i+1, len(nodes)):
                    nodei, nodej = nodes[i], nodes[j]
                    feat_i, feat_j = road_feature[nodei], road_feature[nodej]
                    midpoint_i, midpoint_j = feat_i["midpoint"], feat_j["midpoint"]
                    midpoint_i = [midpoint_i[1], midpoint_i[0]]
                    midpoint_j = [midpoint_j[1], midpoint_j[0]]
                    distance = float(int(get_dis(midpoint_i, midpoint_j)))

                    min_node = min(nodei, nodej)
                    max_node = max(nodei, nodej)
                    geo_dis_matrix[min_node][max_node] = distance
                    
            with open(data_file, "wb") as f:
                pickle.dump(geo_dis_matrix, f)
        return geo_dis_matrix
    
    def sparse_to_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        return self.train_dataloader, None, None

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {'dataloader':self.train_dataloader,'num_nodes':self.road_num,'num_classes':self.num_classes,'input_x':self.input_x, 'c':self.coords, 'onehot_input_feats':self.onehot_input_feats,
                'norms':[self.norm_GG, self.norm_HH, self.norm_HG],"one_hot_dim":self.one_hot_dim,"lane_cls_num":self.lanes_cls,"speed_cls":self.speed_cls,"oneway_cls":self.oneway_cls}


def preproces_network_dgl(city="Singapore"):
    data_file = "../resource/data/" + city + "/preprocess_dgl.pkl"
    road_graph, road_feature = pickle.load(open(data_file, "rb"))
    return road_graph, road_feature

from geopy.distance import geodesic
def get_dis(point1, point2):
    distance = geodesic(point1, point2).m
    return distance
