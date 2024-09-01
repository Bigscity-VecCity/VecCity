import os
from veccity.data.dataset.traffic_representation_dataset import TrafficRepresentationDataset
from veccity.downstream.utils import generate_road_representaion_downstream_data
import numpy as np
import pandas as pd


class RoadNetWorkDataset(TrafficRepresentationDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.label_data_path = os.path.join('veccity', 'cache', 'dataset_cache', self.dataset, 'label_data')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.edge_index = []
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.rel')
        assert os.path.exists(self.data_path + self.dyna_file + '.gpstraj')
        super().__init__(config)
        self.construct_graph()
        self.construct_od_matrix()
        self.read_processed_data()

    def get_data(self):
        return None, None, None

    def construct_graph(self):
        """
        先采用region_od矩阵当作邻接矩阵
        :return:adj_mx
        """
        assert self.representation_object == "road"
        self.adj_mx = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for traj in self.traj_region:
            origin_region = traj[0]
            destination_region = traj[-1]
            self.adj_mx[origin_region][destination_region] += 1
            self.edge_index.append((origin_region, destination_region))
        return self.adj_mx

    def construct_od_matrix(self):
        assert self.representation_object == "road"
        self.od_label = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for traj in self.traj_region:
            origin_region = traj[0]
            destination_region = traj[-1]
            self.od_label[origin_region][destination_region] += 1
        return self.od_label

    def read_processed_data(self):
        assert self.representation_object == "road"
        data_path1 = os.path.join("veccity/cache/dataset_cache", self.dataset, "label_data", "speed.csv")
        data_path2 = os.path.join("veccity/cache/dataset_cache", self.dataset, "label_data", "time.csv")
        if not os.path.exists(data_path1) or not os.path.exists(data_path2):
            generate_road_representaion_downstream_data(self.dataset)
        self.label = {"speed_inference": {}, "travel_time_estimation": {}}
        self.length_label = pd.read_csv(os.path.join(self.label_data_path, "length.csv"))

        self.speed_label = pd.read_csv(os.path.join(self.label_data_path, "speed.csv"))
        self.speed_label.sort_values(by="index", inplace=True, ascending=True)

        min_len, max_len = self.config.get("min_len", 1), self.config.get("max_len", 100)
        self.time_label = pd.read_csv(os.path.join(self.label_data_path, "time.csv"))

        self.time_label['path'] = self.time_label['trajs'].map(eval)

        self.time_label['path_len'] = self.time_label['path'].map(len)
        self.time_label = self.time_label.loc[
            (self.time_label['path_len'] > min_len) & (self.time_label['path_len'] < max_len)]

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"adj_mx": self.adj_mx, "num_nodes": self.num_nodes,
                "geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
                "label": {"od_matrix_predict": self.od_label,
                          'speed_inference': {'speed': self.speed_label},
                          'travel_time_estimation': {'time': self.time_label, 'padding_id': self.num_nodes}}}
