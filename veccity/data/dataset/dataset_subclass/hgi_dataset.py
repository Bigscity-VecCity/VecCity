import os
from logging import getLogger
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import torch
from scipy.spatial import Delaunay
from veccity.data.dataset.abstract_dataset import AbstractDataset
from veccity.data.preprocess import preprocess_all, cache_dir
from veccity.utils import ensure_dir, need_train
from tqdm import tqdm
from torch_geometric.data import Data
import geopandas as gpd
import pdb
from shapely.geometry import Point
import math

def convert_to_cartesian_mercator(lon, lat):
    # 假设地球半径为6371km
    EARTH_RADIUS = 6371
    lon_rad = math.radians(lon)
    lat_rad = math.radians(lat)
    x = lon_rad * EARTH_RADIUS
    y = math.log(math.tan(math.pi / 4 + lat_rad / 2)) * EARTH_RADIUS
    return x, y

def parse_coordinates(coordinates_string):
    # 移除字符串中的方括号和空格
    cleaned_string = coordinates_string.replace("[", "").replace("]", "").replace(" ", "")
    # 将字符串分割为坐标对列表
    coordinates = cleaned_string.split(",")
    return float(coordinates[0]), float(coordinates[1])


class HGIDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        preprocess_all(config)
        self._logger = getLogger()
        self.dataset = self.config.get('dataset', '')
        self.cache = self.config.get('cache', False)
        data_path = f'raw_data/{self.dataset}'
        cache_path = f'./veccity/cache/dataset_cache/{self.dataset}/HGI'
        ensure_dir(cache_path)
        poi_emb_size = config.get('poi_emb_size', 64)
        node_features_path = os.path.join(cache_path, 'node_features.npy')
        region_area_path = os.path.join(cache_path, 'region_area.npy')
        coarse_region_similarity_path = os.path.join(cache_path, 'coarse_region_similarity')
        region_id_path = os.path.join(cache_path, 'region_id.npy')
        region_adjacency_path = os.path.join(cache_path, 'region_adjacency.npy')
        edge_index_path = os.path.join(cache_path, 'edge_index.npy')
        edge_weight_path = os.path.join(cache_path, 'edge_weight.npy')
        if not os.path.exists(region_area_path):
            region_df = pd.read_csv(os.path.join(cache_dir, self.dataset, 'region.csv'))
            geometry = gpd.GeoSeries.from_wkt(region_df['geometry'])
            self.region_area = geometry.area.values
            self.region_area /= self.region_area.sum()
            np.save(region_area_path, self.region_area)
        else:
            self.region_area = np.load(region_area_path)  
        if not self.cache or not os.path.exists(region_adjacency_path) or not os.path.exists(region_id_path) or not os.path.exists(edge_index_path) \
            or not os.path.exists(node_features_path) or not os.path.exists(coarse_region_similarity_path):
            g_df = pd.read_csv(os.path.join(data_path, self.dataset + '.grel'))
            rel_df = g_df[g_df['rel_type'] == 'poi2region']
            geo_df = pd.read_csv(os.path.join(data_path, self.dataset + '.geo'))
            poi_df = geo_df[geo_df['traffic_type'] == 'poi']
            points = []
            for _, row in poi_df.iterrows():
                points.append(parse_coordinates(row['coordinates']))
            region_edges_df = g_df[g_df['rel_type'] == 'region2region']
            self.region_id = rel_df['destination_id'].values
            self.region_adjacency = region_edges_df[['origin_id', 'destination_id']].values.reshape(2, -1)
            points_2d = [(convert_to_cartesian_mercator(p[0], p[1])) for p in points]
            tri = Delaunay(np.array(points_2d))
            # 
            self.edge_index = np.array([[], []], dtype=int)
            for t in tri.simplices:
                self.edge_index = np.hstack((self.edge_index, np.array([[t[0], t[1], t[2]], [t[1], t[2], t[0]]], dtype=int)))
            self.edge_weight = np.zeros((self.edge_index.shape[1]))
            p1 = [-9999, -9999]
            p2 = [9999, 9999]
            for p in points:
                p1[0] = max(p1[0], p[0])
                p1[1] = max(p1[1], p[1])
                p2[0] = min(p2[0], p[0])
                p2[1] = min(p2[1], p[1])
            p1 = Point([(p1[0], p1[1])])
            p2 = Point([(p2[0], p2[1])])
            L = p1.distance(p2) # TODO 粗糙处理，选最大最小经纬度点距离
            
            for i in range(self.edge_index.shape[1]):
                x, y = self.edge_index[0][i], self.edge_index[1][i]
                wr = 1 if self.region_id[x] == self.region_id[y] else 0.4
                px = Point(points[x][0], points[x][1])
                py = Point(points[y][0], points[y][1])
                l = px.distance(py)
                self.edge_weight[i] = np.log((1 + L ** 1.5) / (1 + l ** 1.5)) * wr
            min_val = np.min(self.edge_weight)
            max_val = np.max(self.edge_weight)
            self.edge_weight = (self.edge_weight - min_val) / (max_val - min_val) # TODO linear re-scaling to [0, 1]，为什么最小值不是 0
            self.node_features = np.random.random((self.region_id.shape[0], poi_emb_size)) # TODO：需要 POI embedding
            region_embedding = np.zeros((self.region_area.shape[0], poi_emb_size))
            cnt = np.zeros((self.region_area.shape[0]), dtype=int)
            for poi, region in enumerate(self.region_id):
                region_embedding[region] += self.node_features[poi]
                cnt[region] += 1
            for i in range(len(cnt)):
                if cnt[i] > 0:
                    region_embedding[i] /= cnt[i]
            self.coarse_region_similarity = cosine_similarity(region_embedding)
            
            np.save(region_adjacency_path, self.region_adjacency)
            np.save(region_id_path, self.region_id)
            np.save(edge_index_path, self.edge_index)
            np.save(edge_weight_path, self.edge_weight)
            np.save(node_features_path, self.node_features)
            np.save(coarse_region_similarity_path, self.coarse_region_similarity)
        else:
            self.region_adjacency = np.load(region_adjacency_path)
            self.region_id = np.load(region_id_path)
            self.edge_index = np.load(edge_index_path)
            self.edge_weight = np.load(edge_weight_path)
            self.node_features = np.load(node_features_path)
            self.coarse_region_similarity = np.load(coarse_region_similarity_path)


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
        poi_graph = Data(x=torch.tensor(self.node_features, dtype=torch.float32),
                     edge_index=torch.tensor(self.edge_index, dtype=torch.int64),
                     edge_weight=torch.tensor(self.edge_weight, dtype=torch.float32),
                     region_id=torch.tensor(self.region_id, dtype=torch.int64),
                     region_area=torch.tensor(self.region_area, dtype=torch.float32),
                     coarse_region_similarity=torch.tensor(self.coarse_region_similarity, dtype=torch.float32),
                     region_adjacency=torch.tensor(self.region_adjacency, dtype=torch.int64))
        return {
            'poi_graph': poi_graph
        }


