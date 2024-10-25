from logging import getLogger

import numpy as np
import pandas as pd
import geopandas as gpd
import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from veccity.data.dataset import AbstractDataset
from veccity.data.preprocess import preprocess_all, cache_dir
import pickle


class TrajRoadDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)

class JCLRNTDataset(AbstractDataset):

    def __init__(self,config):
        self.config = config
        self.model='jclrnt'
        preprocess_all(config)
        self.device = config.get('device')
        self._logger = getLogger()
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        data_cache_dir = os.path.join(cache_dir, self.dataset)
        #加载所有road的tag标签
        self.road_geo_path = os.path.join(data_cache_dir, 'road.csv')
        self.road_geo_df = pd.read_csv(self.road_geo_path,delimiter=',')
        self.road_tag = np.array(self.road_geo_df['highway'])
        self.road_length = np.array(self.road_geo_df['length'])
        self.road_geometry = gpd.GeoSeries.from_wkt(self.road_geo_df['geometry'])
        self.edge_threshold = self.config.get("edge_threshold", 0.6)
        self.centroid = self.road_geometry.centroid
        self.road_num = len(self.road_tag)
        self.traj_path = os.path.join(data_cache_dir, 'traj_road_train.csv')
        self.adj_json_path = os.path.join(data_cache_dir, 'road_neighbor.json')
        self.road_feature_path = os.path.join(data_cache_dir, 'road_features.csv')
        self.batch_size = config.get('batch_size',64)
        self.min_len=config.get('min_len',10)
        self.max_len = config.get('max_len', 128)

        # cache_dir
        self.cache_file_folder = './veccity/cache/dataset_cache/'
        self.cache_file_name = os.path.join(self.cache_file_folder,
                                            f'{self.model}_{self.dataset}_{int(self.edge_threshold*10)}_{self.max_len}_{self.min_len}.pickle')
        self.construct_road_adj()
        # if os.path.exists(self.cache_file_name):
        #     self._logger.info('cache_file_name='+self.cache_file_name)
        #     with open(self.cache_file_name, 'rb') as file:
        #         self.edge_index=pickle.load(file)
        #         self.od_matrix=pickle.load(file)
        #         self.edge_index_aug=pickle.load(file)
        #         self.traj_arr_train=pickle.load(file)
        #         self.traj_arr_test=pickle.load(file)
        #         self._logger.info('load cache file success')
        # else: 
        self.train_path = os.path.join(data_cache_dir, 'traj_road_train.csv')
        self.traj_arr = self.prepare_traj_data()
        train_path = os.path.join(data_cache_dir, 'traj_road_train.csv')
        test_path = os.path.join(data_cache_dir, 'traj_road_test.csv')
        self.traj_arr_train = self.prepare_traj_test_data(train_path)
        self.traj_arr_test = self.prepare_traj_test_data(test_path)
        #     with open(self.cache_file_name, 'wb') as file:
        #         pickle.dump(self.edge_index,file)
        #         pickle.dump(self.od_matrix,file)
        #         pickle.dump(self.edge_index_aug,file)
        #         pickle.dump(self.traj_arr_train,file)
        #         pickle.dump(self.traj_arr_test,file)
        #         self._logger.info('save cache file success')
        
        self.generate_train_data()

    


    def construct_road_adj(self):
        self.road_adj = np.zeros(shape=[self.road_num,self.road_num])
        #构建路网的邻接关系
        with open(self.adj_json_path,'r',encoding='utf-8') as fp:
            road_adj_data = json.load(fp)
        for road in range(self.road_num):
            for neighbor in road_adj_data[str(road)]:
                self.road_adj[road][neighbor] = 1


    def prepare_traj_data(self):
        self.edge_index=[]
        self.od_matrix=np.zeros((self.road_num,self.road_num),dtype=np.float32)
        traj_df = pd.read_csv(self.train_path)
        traj_list = []
        # construct od flow and traj # cache traj
        for i in tqdm(range(len(traj_df))):
            path = traj_df.loc[i, 'path']
            path = path[1:len(path) - 1].split(',')
            path = [int(s) for s in path]
            if len(path)>self.min_len and len(path)<self.max_len:
                traj_list.append(path)
                origin_road=path[0]
                destination_road=path[-1]
                self.od_matrix[origin_road][destination_road]+=1
                self.edge_index.append((origin_road,destination_road))
        # padding traj
        arr = np.full([len(traj_list), self.max_len], self.road_num, dtype=np.int32)
        for i in range(len(traj_list)):
            path_arr = np.array(traj_list[i],dtype=np.int32)
            arr[i,:len(traj_list[i])]=path_arr
        
        self.edge_index = np.array(self.edge_index, dtype=np.int32).transpose()
        self.edge_index = torch.Tensor(self.edge_index).int().to(self.device)
        self.tran_matrix = self.od_matrix / (self.od_matrix.max(axis=1, keepdims=True, initial=0.) + 1e-9)
        row, col = np.diag_indices_from(self.tran_matrix)
        self.tran_matrix[row, col] = 0
        self.tran_matrix_b = (self.tran_matrix > self.edge_threshold)
        self.edge_index_aug = [(i // self.road_num, i % self.road_num) for i, n in
                               enumerate(self.tran_matrix_b.flatten()) if n]
        self.edge_index_aug = np.array(self.edge_index_aug, dtype=np.int32).transpose()
        self.edge_index_aug = torch.Tensor(self.edge_index_aug).int().to(self.device)
        self_loop = torch.tensor([[self.road_num - 1], [self.road_num - 1]]).to(self.device)
        self.edge_index = torch.cat((self.edge_index, self_loop), axis=1)
        self.edge_index_aug = torch.cat((self.edge_index_aug, self_loop), axis=1)
        return arr
    
    def prepare_traj_test_data(self,traj_path):
        traj_df = pd.read_csv(traj_path)
        traj_list = []
        for i in tqdm(range(len(traj_df))):
            path = traj_df.loc[i, 'path']
            path = path[1:len(path) - 1].split(',')
            path = [int(s) for s in path]
            traj_list.append(path)
        arr = np.full([len(traj_list), self.max_len], self.road_num, dtype=np.int32)
        for i in range(len(traj_list)):
            path_arr = np.array(traj_list[i],dtype=np.int32)
            traj_len = min(len(traj_list[i]),self.max_len)
            arr[i,:traj_len]=path_arr[:traj_len]
        self._logger.info('test_set_shape='+str(arr.shape))
        return arr
    
    def generate_train_data(self):
        train_dataset=TrajRoadDataset(self.traj_arr_train)
        self.train_dataloader=DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True, num_workers=4)

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

        return {'dataloader':self.train_dataloader,'num_nodes':self.road_num,"edge_index":self.edge_index,
                "edge_index_aug":self.edge_index_aug,"traj_arr_test":self.traj_arr_test,"traj_arr_train":self.traj_arr_train}

