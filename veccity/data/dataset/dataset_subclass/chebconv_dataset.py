import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from logging import getLogger

from veccity.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir
from veccity.data.dataset import AbstractDataset
from veccity.data.preprocess import preprocess_all, cache_dir


class ChebConvDataset(AbstractDataset):
    def __init__(self, config):
        self.representation_object = config.get('representation_object', 'road')
        self.config = config
        preprocess_all(config)
        self.dataset = self.config.get('dataset', '')
        self.cache_dataset = self.config.get('cache_dataset', True)
        self.train_rate = self.config.get('train_rate', 0.7)
        self.eval_rate = self.config.get('eval_rate', 0.1)
        self.scaler_type = self.config.get('scaler', 'none')
        # 路径等参数
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.train_rate) + '_' \
            + str(self.eval_rate) + '_' + str(self.scaler_type)
        self.cache_file_name = os.path.join(f'./veccity/cache/dataset_cache/{self.parameters_str}',
                                            'road_rep.npz')
        self.cache_file_folder = './veccity/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))
        self.label_data_path = os.path.join('veccity', 'cache', 'dataset_cache', self.dataset, 'label_data')

        # 加载数据集的config.json文件
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        # 初始化
        self.adj_mx = None
        self.scaler = None
        self.feature_dim = 0
        self.num_nodes = 0
        self._logger = getLogger()
        self._load_geo()
        self._load_rel()

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_uid, type, coordinates, properties(若干列)]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_uids = list(geofile['geo_uid'])
        # self.num_nodes = len(self.geo_uids)
        self.num_nodes = geofile[geofile['traffic_type'] == 'road'].shape[0]
        self.num_regions = geofile[geofile['traffic_type'] == 'region'].shape[0]
        self.geo_to_ind = {}
        for index, idx in enumerate(self.geo_uids):
            self.geo_to_ind[idx] = index
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_uids)))
        self.road_info = geofile

    def _load_rel(self):
        """
        加载.rel文件，格式[rel_uid, type, orig_geo_id, dest_geo_id, properties(若干列)],
        生成N*N的矩阵，默认.rel存在的边表示为1，不存在的边表示为0

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        map_info = pd.read_csv(self.data_path + self.rel_file + '.grel')
        map_info = map_info[map_info['rel_type'] == 'road2road'].reset_index(drop=True)
        # 使用稀疏矩阵构建邻接矩阵
        adj_row = []
        adj_col = []
        adj_data = []
        adj_set = set()
        cnt = 0
        for i in range(map_info.shape[0]):
            if map_info['orig_geo_id'][i] in self.geo_to_ind and map_info['dest_geo_id'][i] in self.geo_to_ind:
                f_id = self.geo_to_ind[map_info['orig_geo_id'][i]] - self.num_regions
                t_id = self.geo_to_ind[map_info['dest_geo_id'][i]] - self.num_regions
                if (f_id, t_id) not in adj_set:
                    adj_set.add((f_id, t_id))
                    adj_row.append(f_id)
                    adj_col.append(t_id)
                    adj_data.append(1.0)
                    cnt = cnt + 1
        self.adj_mx = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(self.num_nodes, self.num_nodes))
        save_path = os.path.join(self.cache_file_folder, self.dataset, "adj_mx.npz")
        sp.save_npz(save_path, self.adj_mx)
        self._logger.info('Total link between geo = {}'.format(cnt))
        self._logger.info('Adj_mx is saved at {}'.format(save_path))

    def _split_train_val_test(self):
        node_features = pd.read_csv(os.path.join(cache_dir, self.dataset, 'road.csv'))
        drop_features = ['id', 'geometry', 'u', 'v', 's_lon', 's_lat', 
             'e_lon', 'e_lat', 'm_lon', 'm_lat', 'geo_location',
             'type']
        for drop_feature in drop_features:
            if drop_feature in node_features.keys():
                node_features = node_features.drop(drop_feature, axis=1)

        node_features = node_features.values
        np.save(os.path.join(self.cache_file_folder, self.dataset, 'node_features.npy'), node_features)

        # mask 索引
        sindex = list(range(self.num_nodes))
        np.random.shuffle(sindex)

        test_rate = 1 - self.train_rate - self.eval_rate
        num_test = round(self.num_nodes * test_rate)
        num_train = round(self.num_nodes * self.train_rate)
        num_val = self.num_nodes - num_test - num_train

        train_mask = np.array(sorted(sindex[0: num_train]))
        valid_mask = np.array(sorted(sindex[num_train: num_train + num_val]))
        test_mask = np.array(sorted(sindex[-num_test:]))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                node_features=node_features,
                train_mask=train_mask,
                valid_mask=valid_mask,
                test_mask=test_mask
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        self._logger.info("len train feature\t" + str(len(train_mask)))
        self._logger.info("len eval feature\t" + str(len(valid_mask)))
        self._logger.info("len test feature\t" + str(len(test_mask)))
        return node_features, train_mask, valid_mask, test_mask

    def _load_cache_train_val_test(self):
        """
        加载之前缓存好的训练集、测试集、验证集
        """
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name, allow_pickle=True)
        node_features = cat_data['node_features']
        train_mask = cat_data['train_mask']
        valid_mask = cat_data['valid_mask']
        test_mask = cat_data['test_mask']
        self._logger.info("len train feature\t" + str(len(train_mask)))
        self._logger.info("len eval feature\t" + str(len(valid_mask)))
        self._logger.info("len test feature\t" + str(len(test_mask)))
        return node_features, train_mask, valid_mask, test_mask

    def _get_scalar(self, scaler_type, data):
        """
        根据全局参数`scaler_type`选择数据归一化方法

        Args:
            data: 训练数据X

        Returns:
            Scaler: 归一化对象
        """
        if scaler_type == "normal":
            scaler = NormalScaler(maxx=data.max())
            self._logger.info('NormalScaler max: ' + str(scaler.max))
        elif scaler_type == "standard":
            scaler = StandardScaler(mean=data.mean(), std=data.std())
            self._logger.info('StandardScaler mean: ' + str(scaler.mean) + ', std: ' + str(scaler.std))
        elif scaler_type == "minmax01":
            scaler = MinMax01Scaler(
                maxx=data.max(), minn=data.min())
            self._logger.info('MinMax01Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "minmax11":
            scaler = MinMax11Scaler(
                maxx=data.max(), minn=data.min())
            self._logger.info('MinMax11Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "log":
            scaler = LogScaler()
            self._logger.info('LogScaler')
        elif scaler_type == "none":
            scaler = NoneScaler()
            self._logger.info('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            batch_data: dict
        """
        # 加载数据集
        # if self.cache_dataset and os.path.exists(self.cache_file_name):
        #     node_features, train_mask, valid_mask, test_mask = self._load_cache_train_val_test()
        # else:
        node_features, train_mask, valid_mask, test_mask = self._split_train_val_test()
        # 数据归一化
        self.feature_dim = node_features.shape[-1]
        self.scaler = self._get_scalar(self.scaler_type, node_features)
        node_features = self.scaler.transform(node_features)
        self.train_dataloader = {'node_features': node_features, 'mask': train_mask}
        self.eval_dataloader = {'node_features': node_features, 'mask': valid_mask}
        self.test_dataloader = {'node_features': node_features, 'mask': test_mask}
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {
                    "scaler": self.scaler, "adj_mx": self.adj_mx,
                    "num_nodes": self.num_nodes, "feature_dim": self.feature_dim
                }
