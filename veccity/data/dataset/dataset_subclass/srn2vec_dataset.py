from itertools import chain, combinations
from logging import getLogger

import numpy as np
import pandas as pd
import geopandas as gpd
import json
import os
import pickle

import torch
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from veccity.data.dataset import AbstractDataset
from veccity.data.preprocess import preprocess_all, cache_dir
from veccity.utils import ensure_dir


class SRN2VecDataset(AbstractDataset):
    def __init__(self,config):
        self.config = config
        preprocess_all(config)
        self._logger = getLogger()
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        data_cache_dir = os.path.join(cache_dir, self.dataset)
        #加载所有road的tag标签
        self.road_geo_path = os.path.join(data_cache_dir, 'road.csv')
        self.road_geo_df = pd.read_csv(self.road_geo_path,delimiter=',')
        self.road_tag = np.array(self.road_geo_df['lanes'])
        self.road_length = np.array(self.road_geo_df['length'])
        self.road_geometry = gpd.GeoSeries.from_wkt(self.road_geo_df['geometry'])
        self.centroid = self.road_geometry.centroid
        self.road_num = len(self.road_tag)
        self.adj_json_path = os.path.join(data_cache_dir, 'road_neighbor.json')
        self.construct_road_adj()
        self.n_short_paths = config.get('n_short_paths',80)
        self.data_cache_file = f'./veccity/cache/dataset_cache/{self.dataset}/SRN2Vec'
        ensure_dir(self.data_cache_file)
        node_paths = self.generate_shortest_paths(self.road_adj,self.n_short_paths)
        self.number_negative = config.get('number_negative',3)
        self.batch_size = config.get('batch_size',128)
        self.window_size = config.get('window_size',900)
        self.cache_file = os.path.join(self.data_cache_file, f'cache_{self.window_size}_{self.number_negative}_{self.n_short_paths}.npy')
        if os.path.exists(self.cache_file):
            self.train_pairs = np.load(self.cache_file)
        else:
            self.train_pairs = self.extract_pairs(self.road_length,self.road_tag,node_paths,self.window_size,self.number_negative)
            np.save(self.cache_file,self.train_pairs)


    def construct_road_adj(self):
        self._logger.info('Start construct road adj...')
        self.road_adj = np.zeros(shape=[self.road_num,self.road_num])

        #构建路网的邻接关系
        with open(self.adj_json_path,'r',encoding='utf-8') as fp:
            road_adj_data = json.load(fp)
        for road in range(self.road_num):
            for neighbor in road_adj_data[str(road)]:
                distance = self.centroid[road].distance(self.centroid[neighbor])
                self.road_adj[road][neighbor] = distance
        self._logger.info('Finish construct road adj.')


    def generate_shortest_paths(self,adj,n_shortest_paths):
        self._logger.info('Start generate shortest paths...')
        #最短路算法实现，返回所有的路径
        config_file = os.path.join(self.data_cache_file, f'srn2vec_config_{n_shortest_paths}.json')
        nodes_paths_cache_file = os.path.join(self.data_cache_file, f'srn2vec_nodes_paths_{n_shortest_paths}')
        if not os.path.exists(config_file):
            def get_path(Pr, i, j):
                path = [j]
                k = j
                while Pr[i, k] != -9999:
                    path.append(Pr[i, k])
                    k = Pr[i, k]
                if Pr[i, k] == -9999 and k != i:
                    return []
                return path[::-1]
            ensure_dir(nodes_paths_cache_file)
            total = 500000
            counter = 0
            _, P = shortest_path(
                adj, directed=True, method="D", return_predecessors=True, unweighted=True
            )
            nodes_paths = []
            for source in tqdm(range(self.road_num)):
                targets = np.random.choice(np.setdiff1d(np.arange(self.road_num),[source]),size=n_shortest_paths,replace=False)
                for target in targets:
                    path = get_path(P,source,target)
                    if path != []:
                        nodes_paths.append(path)
                        if len(nodes_paths) == total:
                            with open(os.path.join(nodes_paths_cache_file, 'path_{}'.format(counter)), 'wb') as f:
                                pickle.dump(nodes_paths, f)
                            nodes_paths = []
                            counter += 1
            if nodes_paths != []:
                with open(os.path.join(nodes_paths_cache_file, 'path_{}'.format(counter)), 'wb') as f:
                    pickle.dump(nodes_paths, f)
                nodes_paths = []
                counter += 1
            data_config = {'paths_count': counter}
            with open(config_file, 'w') as f:
                json.dump(data_config, f)

        with open(config_file, 'r') as f:
            data_config = json.load(f)
        nodes_paths = []
        for i in range(data_config['paths_count']):
            with open(os.path.join(nodes_paths_cache_file, 'path_{}'.format(i)), 'rb') as f:
                nodes_paths.extend(pickle.load(f))
        self._logger.info('Finish generate shortest paths.')
        return nodes_paths

    def extract_pairs(self,info_length, info_highway,node_paths,window_size,number_negative):
        """
         info_length (np.array): length for each node in graph (ordered by node ordering in graph)
         info_highway (np.array): type for each node in graph (ordered by node ordering in graph)
        :param paths:  shortest paths
        :param window_size:window_size in meter
        :param number_negative: snumber negative to draw for each node
        :return:
        Returns:
        list: training pairs
        """
        res = []
        # lengths of orginal sequences in flatted with cumsum to get real position in flatted
        orig_lengths = np.array([0] + [len(x) for x in node_paths]).cumsum()
        flatted = list(chain.from_iterable(node_paths))
        # get all lengths of sequence roads
        # TODO：BJ疑似因为flatted长度太大报错
        # 2344563208
        # flat_lengths = info_length[flatted]

        # generate window tuples
        node_combs = []
        for i in tqdm(range(len(orig_lengths) - 1)):
            lengths = info_length[flatted[orig_lengths[i]: orig_lengths[i + 1]]]
            # cumsum = lengths.cumsum()
            for j in range(len(lengths)):
                mask = 0
                sum = 0
                while j + mask < len(lengths) and sum + lengths[j + mask] < window_size:
                    sum += lengths[j + mask]
                    mask += 1
                window = node_paths[i][j: j + mask]
                if len(window) > 1:
                    combs = tuple(combinations(window, 2))
                    node_combs.extend(combs)
        # save distinct tuples
        node_combs = list(dict.fromkeys(node_combs))
        node_combs = list(chain.from_iterable(node_combs))

        highways = info_highway[node_combs].reshape(int(len(node_combs) / 2), 2)

        # Generate pairs: node1, node2, true (on same walk), true if same degree
        pairs = np.c_[
            np.array(node_combs).reshape(int(len(node_combs) / 2), 2),
            np.ones(highways.shape[0]),
            highways[:, 0] == highways[:, 1],
        ].astype(
            int
        )  # same type
        res.extend(tuple(pairs.tolist()))
        # generate negative sample with same procedure as for positive
        neg_nodes = np.random.choice(
            np.setdiff1d(np.arange(0, self.road_num), node_combs),
            size=pairs.shape[0] * number_negative,
        )
        neg_pairs = pairs.copy()
        neg_pairs = neg_pairs.repeat(repeats=number_negative, axis=0)
        replace_mask = np.random.randint(0, 2, size=neg_pairs.shape[0]).astype(bool)
        neg_pairs[replace_mask, 0] = neg_nodes[replace_mask]
        neg_pairs[~replace_mask, 1] = neg_nodes[~replace_mask]
        neg_pairs[:, 2] -= 1
        neg_highways = info_highway[neg_pairs[:, :2].flatten()].reshape(
            neg_pairs.shape[0], 2
        )
        neg_pairs[:, 3] = neg_highways[:, 0] == neg_highways[:, 1]
        res.extend(tuple(neg_pairs.tolist()))
        return res

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        return None, None, None

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        train_data = np.array(self.train_pairs)
        data = torch.tensor(train_data[:,[0,1]])
        labels = torch.tensor(train_data[:,[2,3]])
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        return {'dataloader':dataloader,'num_nodes':self.road_num}

