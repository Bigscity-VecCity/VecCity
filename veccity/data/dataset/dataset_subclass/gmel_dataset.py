import random
from veccity.data.dataset.od_region_representation_dataset import ODRegionRepresentationDataset
import numpy as np
from veccity.data.utils import split_list


class GMELDataset(ODRegionRepresentationDataset):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.construct_trip_data()
        self.construct_geo_adj()
        self.construct_distance_mx()
        self.construct_node_feats()

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

    def construct_node_feats(self):
        self.feature = np.random.uniform(-1, 1, size=(self.num_nodes, 250))

    def construct_geo_adj(self):
        self.adj_mx = self.adj_mx / self.adj_mx.max().max()

    def construct_distance_mx(self):
        self.distance_mx = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.distance_mx[i][j] = self.centroid[i].distance(self.centroid[j])

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"train": self.train_trip, "valid": self.valid_trip, "test": self.test_trip,
                "train_inflow": self.train_inflow, "train_outflow": self.train_outflow,
                "num_nodes": self.num_nodes, "node_feats": self.feature, "ct_adjacency_withweight": self.adj_mx,
                "distm": self.distance_mx,
                "label": {"od_matrix_predict": self.od_label.flatten()}}

    def get_data(self):
        return None, None, None
