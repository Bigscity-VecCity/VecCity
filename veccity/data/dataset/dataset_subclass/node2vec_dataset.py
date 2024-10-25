import os

import numpy as np

from veccity.data.dataset.traffic_representation_dataset import TrafficRepresentationDataset
from veccity.utils import ensure_dir
from veccity.data.preprocess import cache_dir


class Node2VecDataset(TrafficRepresentationDataset):
    def __init__(self,config):
        self.config = config
        super().__init__(config)
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.grel')
        dir = 'veccity/cache/dataset_cache/{}/Node2Vec'
        ensure_dir(dir)
        self.od_label_path = os.path.join(cache_dir, self.dataset, 'od_region_train_od.npy')
        self.od_label = np.load(self.od_label_path)
        self.construct_graph()

    def get_data(self):
        return None,None,None


    def construct_graph(self):
        """
        先采用region_od矩阵当作邻接矩阵
        :return:adj_mx
        """
        self.adj_mx = self.od_label
    
    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"adj_mx": self.adj_mx, "num_nodes": self.num_nodes,
                "label":{"od_matrix_predict":self.od_label}}