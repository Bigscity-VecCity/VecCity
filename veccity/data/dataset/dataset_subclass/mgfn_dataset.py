from datetime import datetime
import os
from logging import getLogger

import numpy as np
import pandas as pd

from veccity.data.dataset import AbstractDataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from tqdm import  *
from veccity.data.preprocess import preprocess_all, cache_dir
from veccity.utils import ensure_dir, need_train

class MGFNDataset(AbstractDataset):
    def __init__(self,config):
        self.config = config
        preprocess_all(config)
        self._logger = getLogger()
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.time_slice = self.config.get('time_slice',6)
        self.n_cluster = self.config.get('n_cluster',2)
        assert (24 % self.time_slice == 0)
        self.multi_graph = None
        ensure_dir(f'./veccity/cache/dataset_cache/{self.dataset}/HDGE')
        ensure_dir(f'./veccity/cache/dataset_cache/{self.dataset}/MGFN')

        self.flow_graph_path = './veccity/cache/dataset_cache/{}/HDGE/{}_slice_flow_graph.npy'.format(self.dataset,self.time_slice)
        self.mob_patterns_path = './veccity/cache/dataset_cache/{}/MGFN/{}_slice_{}_clusters_mob_patterns.npy'.format(self.dataset,self.time_slice,self.n_cluster)
        self.od_label_path = os.path.join(cache_dir, self.dataset, 'od_region_train_od.npy')
        self.mob_adj = np.load(self.od_label_path)
        self.num_regions = self.mob_adj.shape[0]
        self.num_nodes = self.num_regions

        if os.path.exists(self.mob_patterns_path):
            self.mob_patterns = np.load(self.mob_patterns_path)
            self._logger.info("finish get Mobility Pattern")
        else:
            self.multi_graph = self.construct_multi_graph()
            self.mob_patterns, self.cluster_label = self.getPatternWithMGD(self.multi_graph)

    def get_data(self):
        return None, None, None

    def construct_multi_graph(self):

        if os.path.exists(self.flow_graph_path):
            flow_graph = np.load(self.flow_graph_path)
            self._logger.info("finish constructing flow graph")
            return flow_graph
        time_each_slice = 24 // self.time_slice
        od_file = pd.read_csv(os.path.join(cache_dir, self.dataset, 'od_region_train.csv'))
        flow_graph = np.zeros([self.time_slice, self.num_nodes, self.num_nodes])
        for _, row in od_file.iterrows():
            origin_region = int(row['orig_geo_id'])
            destination_region = int(row['dest_geo_id'])
            time = datetime.fromtimestamp(int(row['end_time']))
            flow_graph[time.hour // time_each_slice][origin_region][destination_region] += 1
        np.save(self.flow_graph_path, flow_graph)
        return flow_graph

    def propertyFunc_var(self,adj_matrix):
        return adj_matrix.var()

    def propertyFunc_mean(self,adj_matrix):
        return adj_matrix.mean()

    def propertyFunc_std(self,adj_matrix):
        return adj_matrix.std()

    def propertyFunc_UnidirectionalIndex(self,adj_matrix):
        unidirectionalIndex = 0
        for i in trange(len(adj_matrix)):
            for j in range(len(adj_matrix[0])):
                unidirectionalIndex = unidirectionalIndex + \
                                      abs(adj_matrix[i][j] - adj_matrix[j][i])
        return unidirectionalIndex

    def getPropertyArrayWithPropertyFunc(self,data_input, property_func):
        result = []
        for i in range(len(data_input)):
            result.append(property_func(data_input[i]))
        # -- standardlize
        return np.array(result)

    def getDistanceMatrixWithPropertyArray(self,data_x, property_array, isSigmoid=False):
        sampleNum = data_x.shape[0]
        disMatrix = np.zeros([sampleNum, sampleNum])
        for i in range(0, sampleNum):
            for j in range(0, sampleNum):
                if isSigmoid:
                    hour_i = i % 24
                    hour_j = j % 24
                    hour_dis = abs(hour_i - hour_j)
                    if hour_dis == 23:
                        hour_dis = 1
                    c = self.sigmoid(hour_dis / 24)
                else:
                    c = 1
                disMatrix[i][j] = c * abs(property_array[i] - property_array[j])
        disMatrix = (disMatrix - disMatrix.min()) / (disMatrix.max() - disMatrix.min())
        return disMatrix

    def getDistanceMatrixWithPropertyFunc(self,data_x, property_func, isSigmoid=False):
        property_array = self.getPropertyArrayWithPropertyFunc(data_x, property_func)
        disMatrix = self.getDistanceMatrixWithPropertyArray(data_x, property_array, isSigmoid=isSigmoid)
        return disMatrix

    def get_SSEncode2D(self,one_data, mean_data):
        result = []
        for i in range(len(one_data)):
            for j in range(len(one_data[0])):
                if one_data[i][j] > mean_data[i][j]:
                    result.append(1)
                else:
                    result.append(0)
        return np.array(result)

    def getDistanceMatrixWith_SSIndex(self,input_data, isSigmoid=True):
        sampleNum = len(input_data)
        input_data_mean = input_data.mean(axis=0)
        property_array = []
        for i in range(len(input_data)):
            property_array.append(self.get_SSEncode2D(input_data[i], input_data_mean))
        property_array = np.array(property_array)
        disMatrix = np.zeros([sampleNum, sampleNum])
        for i in range(0, sampleNum):
            for j in range(0, sampleNum):
                if isSigmoid:
                    hour_i = i % 24
                    hour_j = j % 24
                    sub_hour = abs(hour_i - hour_j)
                    if sub_hour == 23:
                        sub_hour = 1
                    c = self.sigmoid(sub_hour / 24)
                else:
                    c = 1
                sub_encode = abs(property_array[i] - property_array[j])
                disMatrix[i][j] = c * sub_encode.sum()
        disMatrix = (disMatrix - disMatrix.min()) / (disMatrix.max() - disMatrix.min())
        label_pred = self.getClusterLabelWithDisMatrix(disMatrix, display_dis_matrix=False)
        return disMatrix

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def Mobility_Graph_Distance(self,m_graphs):
        """
        :param m_graphs: (N, M, M).  N graphs, each graph has M nodes
        :return: (N, N). Distance matrix between every two graphs
        """
        # Mean
        isSigmoid = True
        mean_dis_matrix = self.getDistanceMatrixWithPropertyFunc(
            m_graphs, self.propertyFunc_mean, isSigmoid=isSigmoid)
        # Uniflow
        unidirIndex_dis_matrix = self.getDistanceMatrixWithPropertyFunc(
            m_graphs, self.propertyFunc_UnidirectionalIndex, isSigmoid=isSigmoid
        )
        # Var
        var_dis_matrix = self.getDistanceMatrixWithPropertyFunc(
            m_graphs, self.propertyFunc_var, isSigmoid=isSigmoid
        )
        # SS distance
        ss_dis_matrix = self.getDistanceMatrixWith_SSIndex(m_graphs, isSigmoid=isSigmoid)
        c_mean_dis = 1
        c_unidirIndex_dis = 1
        c_std_dis = 1
        c_ss_dis = 1
        disMatrix = (c_mean_dis * mean_dis_matrix) \
                    + (c_unidirIndex_dis * unidirIndex_dis_matrix) \
                    + (c_std_dis * var_dis_matrix) \
                    + (c_ss_dis * ss_dis_matrix)
        return disMatrix

    def getClusterLabelWithDisMatrix(self,dis_matrix, display_dis_matrix=False):
        n_clusters = self.n_cluster
        # # linkage: single, average, complete
        linkage = "complete"
        # ---
        # t1 = time.time()
        if display_dis_matrix:
            sns.heatmap(dis_matrix)
            plt.show()
        # ---
        estimator = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage, affinity="precomputed", )
        estimator.fit(dis_matrix)
        label_pred = estimator.labels_
        return label_pred

    def getPatternWithMGD(self,m_graphs):
        """
        :param m_graphs: (N, M, M).  N graphs, each graph has M nodes
        :return mob_patterns:
        :return cluster_label:
        """
        n_clusters = self.n_cluster
        linkage = "complete"
        disMatrix = self.Mobility_Graph_Distance(m_graphs)
        # -- Agglomerative Cluster
        estimator = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity="precomputed", )
        estimator.fit(disMatrix)
        label_pred = estimator.labels_
        cluster_label = label_pred
        # -- Generate Mobility Pattern
        patterns = []
        pbar = tqdm(range(n_clusters))
        pbar.set_description('get Pattern')
        for i in pbar:
            this_cluster_index_s = np.argwhere(label_pred == i).flatten()
            this_cluster_graph_s = m_graphs[this_cluster_index_s]
            patterns.append(this_cluster_graph_s.sum(axis=0))
        mob_patterns = np.array(patterns)
        np.save(self.mob_patterns_path,mob_patterns)
        self._logger.info("finish get Mobility Pattern")
        return mob_patterns, cluster_label

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"n_cluster":self.n_cluster,"mob_adj":self.mob_adj,"mob_patterns":self.mob_patterns,"num_nodes": self.num_nodes}

