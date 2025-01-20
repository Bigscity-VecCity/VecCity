import os
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
import random
import dgl

from veccity.data.dataset.dataset_subclass import MVUREDataset
from veccity.utils import need_train
from veccity.data.preprocess import preprocess_all, cache_dir
from veccity.data.utils import split_list



# Heterogeneous Region Embedding with Prompt Learning
class HREPDataset(MVUREDataset):

    def __init__(self, config):
        super().__init__(config)
        rel_df = pd.read_csv(os.path.join('./raw_data', self.dataset, self.dataset + '.grel'))
        region2region = rel_df[rel_df['rel_type'] == 'region2region']
        self.neighbor = [[] for _ in range(self.num_regions)]
        for i, row in region2region.iterrows():
            r1 = int(row['orig_geo_id'])
            r2 = int(row['dest_geo_id'])
            self.neighbor[r1].append(r2)
        for i in range(self.num_regions):
            if len(self.neighbor[i]) == 0:
                self.neighbor[i].append(np.random.randint(0, self.num_regions))  # TODO：解决无邻居采样出错的问题
        self.poi_similarity = self.poi_simi
        self.s_adj = self.inflow_adj
        self.d_adj = self.outflow_adj
        self.mobility = self.mob_adj
        self.device = config.get('device')
        self.embedding_size = config.get("output_dim", 144)
        self.importance_k = config.get("importance_k", 10)
        self.od_label_path = os.path.join(cache_dir, self.dataset, 'od_region_train_od.npy')
        self.od_label = np.load(self.od_label_path)
        self.construct_trip_data()
    
    def construct_trip_data(self):
        # 根据od 6：2：2将数据分成训练集，测试集和验证集（org,dst,vol)
        total_trip = [[i, j, self.od_label[i][j]] for i in range(self.num_regions) for j in range(self.num_regions)]
        random.shuffle(total_trip)
        split_ratio = [0.6, 0.2, 0.2]
        result = split_list(total_trip, split_ratio, 3)
        self.train_trip = np.array(result[0])
        self.test_trip = np.array(result[1])
        self.valid_trip = np.array(result[2])
        self.train_inflow = np.zeros(self.num_regions)
        self.train_outflow = np.zeros(self.num_regions)
        for i in range(self.train_trip.shape[0]):
            org = self.train_trip[i][0]
            dst = self.train_trip[i][1]
            vol = self.train_trip[i][2]
            self.train_inflow[int(dst)] += vol
            self.train_outflow[int(org)] += vol

    def graph_to_COO(self, similarity, importance_k):
        graph = torch.eye(self.num_regions)

        for i in range(self.num_regions):
            graph[np.argsort(similarity[:, i])[-importance_k:], i] = 1
            graph[i, np.argsort(similarity[:, i])[-importance_k:]] = 1

        graph = sp.coo_matrix(graph)
        edge_index = dgl.graph((graph.row, graph.col)).to(self.device)
        return edge_index

    def create_graph(self, similarity, importance_k):
        edge_index = self.graph_to_COO(similarity, importance_k)
        return edge_index

    def create_neighbor_graph(self, neighbor):
        graph = np.eye(self.num_regions)

        for i in range(len(neighbor)):
            for region in neighbor[i]:
                graph[i, region] = 1
                graph[region, i] = 1
        # graph = dgl.graph(graph)
        graph = sp.coo_matrix(graph)
        edge_index = dgl.graph((graph.row, graph.col)).to(self.device)
        return edge_index

    def get_data_feature(self):
        poi_edge_index = self.create_graph(self.poi_similarity, self.importance_k)
        s_edge_index = self.create_graph(self.s_adj, self.importance_k)
        d_edge_index = self.create_graph(self.d_adj, self.importance_k)
        n_edge_index = self.create_neighbor_graph(self.neighbor)

        # poi_edge_index = torch.tensor(poi_edge_index, dtype=torch.long).to(self.device)
        # s_edge_index = torch.tensor(s_edge_index, dtype=torch.long).to(self.device)
        # d_edge_index = torch.tensor(d_edge_index, dtype=torch.long).to(self.device)
        # n_edge_index = torch.tensor(n_edge_index, dtype=torch.long).to(self.device)

        self.mobility = torch.tensor(self.mobility, dtype=torch.float32).to(self.device)
        self.poi_similarity = torch.tensor(self.poi_similarity, dtype=torch.float32).to(self.device)
        features = torch.randn(self.num_regions, self.embedding_size).to(self.device)
        poi_r = torch.randn(self.embedding_size).to(self.device)
        s_r = torch.randn(self.embedding_size).to(self.device)
        d_r = torch.randn(self.embedding_size).to(self.device)
        n_r = torch.randn(self.embedding_size).to(self.device)
        rel_emb = [poi_r, s_r, d_r, n_r]
        edge_index = [poi_edge_index, s_edge_index, d_edge_index, n_edge_index]
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {
            'features': features,
            'rel_emb': rel_emb,
            'edge_index': edge_index,
            'poi_similarity': self.poi_similarity,
            'mobility': self.mobility,
            'neighbor': self.neighbor,
            'num_regions': self.num_regions,
            'inflow_labels':self.train_inflow,
            'outflow_labels':self.train_outflow
        }