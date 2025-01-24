import os
from logging import getLogger
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import torch
from veccity.data.dataset.abstract_dataset import AbstractDataset
from veccity.data.preprocess import preprocess_all, cache_dir
from veccity.utils import ensure_dir, need_train
import pdb


class HAFusionDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        preprocess_all(config)
        self._logger = getLogger()
        self.dataset = self.config.get('dataset', '')
        self.cache = self.config.get('cache', True)
        self.data_path = f'./veccity/cache/dataset_cache/{self.dataset}/HAFusion'
        ensure_dir(self.data_path)
        self.mob_dist_path = os.path.join(self.data_path, 'mob_dist.npy')
        self.mob_adj_path = os.path.join(self.data_path, 'mob_adj.npy')
        self.poi_dist_path = os.path.join(self.data_path, 'poi_dist.npy')
        self.poi_simi_path = os.path.join(self.data_path, 'poi_simi.npy')
        od_path = os.path.join(cache_dir, self.dataset, 'od_region_train_od.npy')
        od_matrix = np.load(od_path)
        num_regions = od_matrix.shape[0]
        self.num_regions = num_regions
        if not self.cache or not os.path.exists(self.mob_dist_path) or not os.path.exists(self.mob_adj_path):
            od_df = pd.read_csv(os.path.join(cache_dir, self.dataset, 'od_region_train.csv'))
            od_df = od_df.sort_values(by='start_time')
            SECONDS_PER_DAY = 86400
            days = od_df['start_time'].iloc[len(od_df) // 2] // SECONDS_PER_DAY
            self.mob_dist = np.zeros((num_regions, num_regions), dtype=int)
            self.mob_adj = np.zeros((num_regions, num_regions), dtype=int)
            for _, row in od_df.iterrows():
                o = int(row['origin_id'])
                d = int(row['destination_id'])
                if row['start_time'] // SECONDS_PER_DAY <= days:
                    self.mob_dist[o][d] += 1
                else:
                    self.mob_adj[o][d] += 1
            np.save(self.mob_dist_path, self.mob_dist)
            np.save(self.mob_adj_path, self.mob_adj)
        else:
            self.mob_dist = np.load(self.mob_dist_path)
            self.mob_adj = np.load(self.mob_adj_path)
        
        if not self.cache or not os.path.exists(self.poi_dist_path) or not os.path.exists(self.poi_simi_path):
            geo_df = pd.read_csv(os.path.join('raw_data', self.dataset, self.dataset + '.geo'))
            rel_df = pd.read_csv(os.path.join('raw_data', self.dataset, self.dataset + '.grel'))
            poi2region = rel_df[rel_df['rel_type'] == 'poi2region']
            poi_type_dict = {}
            total = 0
            for _, row in poi2region.iterrows():
                poi_id = int(row['origin_id'])
                poi_type = geo_df['poi_type'][poi_id]
                if poi_type not in poi_type_dict.keys():
                    poi_type_dict[poi_type] = total
                    total += 1
            self.poi_dist = np.zeros((num_regions, len(poi_type_dict.keys())), dtype=int)
            self.poi_dim = len(poi_type_dict.keys())
            for _, row in poi2region.iterrows():
                poi_id = int(row['origin_id'])
                region_id = int(row['destination_id'])
                poi_type = geo_df['poi_type'][poi_id]
                self.poi_dist[region_id][poi_type_dict[poi_type]] += 1
            self.poi_simi = cosine_similarity(self.poi_dist)
            np.save(self.poi_dist_path, self.poi_dist)
            np.save(self.poi_simi_path, self.poi_simi)
        else:
            self.poi_dist = np.load(self.poi_dist_path)
            self.poi_simi = np.load(self.poi_simi_path)
            self.poi_dim = self.poi_dist.shape[1]

    def get_data(self):
        return None, None, None

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        if not need_train(self.config):
            return {}
        device = self.config.get('device')
        self.mob_dist = self.mob_dist[np.newaxis]
        self.mob_dist = torch.Tensor(self.mob_dist).to(device)
        self.poi_dist = self.poi_dist[np.newaxis]
        self.poi_dist = torch.Tensor(self.poi_dist).to(device)
        self.mob_adj = self.mob_adj / np.mean(self.mob_adj)
        self.mob_adj = torch.Tensor(self.mob_adj).to(device)
        self.poi_simi = torch.Tensor(self.poi_simi).to(device)
        return {
            'mob_dist': self.mob_dist,
            'mob_adj': self.mob_adj,
            'poi_dist': self.poi_dist,
            'poi_simi': self.poi_simi,
            'poi_dim': self.poi_dim,
            'num_regions': self.num_regions
        }


