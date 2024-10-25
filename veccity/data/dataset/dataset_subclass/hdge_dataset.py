from datetime import datetime
import os
from logging import getLogger
import geopandas as gpd
import numpy as np
import pandas as pd

from veccity.utils import ensure_dir
from veccity.data.dataset import AbstractDataset
from veccity.data.preprocess import preprocess_all, cache_dir


class HDGEDataset(AbstractDataset):
 def __init__(self,config):
     self.config = config
     preprocess_all(config)
     self._logger = getLogger()
     self.dataset = self.config.get('dataset', '')
     self.data_path = './raw_data/' + self.dataset + '/'
     self.time_slice = self.config.get('time_slice',8)
     self.C = self.config.get('C',100)
     assert (24%self.time_slice == 0)
     self.od_label_path = os.path.join(cache_dir, self.dataset, 'od_region_train_od.npy')
     self.mob_adj = np.load(self.od_label_path)
     self.num_regions = self.mob_adj.shape[0]
     self.num_nodes = self.num_regions
     self.flow_graph_path = './veccity/cache/dataset_cache/{}/HDGE/{}_slice_flow_graph.npy'.format(self.dataset,self.time_slice)
     self.spatial_graph_path = './veccity/cache/dataset_cache/{}/HDGE/C={}_spatial_graph.npy'.format(self.dataset,self.C)
     self.combine_graph_path = './veccity/cache/dataset_cache/{}/HDGE/C={} and {}_slice.npy'.format(self.dataset ,self.C ,self.time_slice)
     ensure_dir(f'veccity/cache/dataset_cache/{self.dataset}/HDGE')
     self.flow_graph = self.construct_flow_graph()
     self.construct_spatial_graph()
     self.combine_graph = self.combine_matrix(self.flow_graph,self.spatial_graph)

 def get_data(self):
    return None, None, None

 def construct_flow_graph(self):
    """
    这里需要对时间分片，一共是time_slice * node_num个节点
    :return:
    """
    if os.path.exists(self.flow_graph_path):
        flow_graph = np.load(self.flow_graph_path)
        self._logger.info("finish constructing flow graph")
        return flow_graph
    time_each_slice = 24 // self.time_slice
    od_file = pd.read_csv(os.path.join(cache_dir, self.dataset, 'od_region_train.csv'))
    flow_graph = np.zeros([self.time_slice, self.num_nodes, self.num_nodes])
    for _, row in od_file.iterrows():
        origin_region = int(row['origin_id'])
        destination_region = int(row['destination_id'])
        time = datetime.fromtimestamp(int(row['end_time']))
        flow_graph[time.hour // time_each_slice][origin_region][destination_region] += 1
    np.save(self.flow_graph_path, flow_graph)
    return flow_graph

 def construct_spatial_graph(self):
    if os.path.exists(self.spatial_graph_path):
        self.spatial_graph = np.load(self.spatial_graph_path)
        self._logger.info("finish constructing spatial graph")
        return
    region_geo_file = pd.read_csv(os.path.join('raw_data', self.dataset, self.dataset + '.geo'))
    self.region_geometry = gpd.GeoSeries.from_wkt(region_geo_file['region_geometry'].dropna())
    self.centroid = self.region_geometry.centroid
    self.spatial_graph = np.zeros([self.num_nodes,self.num_nodes])
    for i in range(self.num_nodes):
        for j in range(i,self.num_nodes):
            distance = self.centroid[i].distance(self.centroid[j])
            self.spatial_graph[i][j] = np.exp(-self.C * distance)
            self.spatial_graph[j][i] = np.exp(-self.C * distance)
    np.save(self.spatial_graph_path,self.spatial_graph)
    self._logger.info("finish consturcting spatial graph")


 def combine_matrix(self, flow_mx, spatial_mx):
     """
     将两个矩阵归一化，合成一个图
     :param flow_mx: 流图的结构[time_slice,num_node,num_node]
     :param spatial_mx: [num_node,num_node]
     :return:combined_mx:[time_slice,num_node,num_node]
     """
     if os.path.exists(self.combine_graph_path):
         combine_graph = np.load(self.combine_graph_path)
         self._logger.info("finish combine graph")
         return combine_graph

     spatial_mx_scaler = np.zeros(shape=(self.num_nodes, self.num_nodes), dtype=np.float32)
     combined_mx = np.zeros(shape=(self.time_slice, self.num_nodes, self.num_nodes), dtype=np.float32)
     for i in range(self.num_nodes):
         unnormalized_probs = [spatial_mx[i][j] for j in range(self.num_nodes)]
         norm_cost = 2 * sum(unnormalized_probs)
         spatial_mx_scaler[i] = spatial_mx[i] / norm_cost

     for t in range(self.time_slice):
         for i in range(self.num_nodes):
             unnormalized_probs_probs = [flow_mx[t][i][j] for j in range(self.num_nodes)]
             norm_cost = 2 * sum(unnormalized_probs_probs)
             if norm_cost == 0:
                 combined_mx[t][i] = spatial_mx_scaler[i] * 2
             else:
                 combined_mx[t][i] = flow_mx[t][i] / norm_cost + spatial_mx_scaler[i]
     np.save(self.combine_graph_path,combined_mx)
     self._logger.info("finish combine graph")
     return combined_mx

 def get_data_feature(self):
     """
     返回一个 dict，包含数据集的相关特征

     Returns:
         dict: 包含数据集的相关特征的字典
     """
     return {"combine_graph":self.combine_graph,"flow_graph":self.flow_graph,"spatial_graph":self.spatial_graph,"time_slice":self.time_slice,
             "C":self.C,"num_nodes": self.num_nodes}



