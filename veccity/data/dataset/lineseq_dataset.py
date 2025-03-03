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
from veccity.data.dataset.dataset_subclass.bert_vocab import WordVocab
from veccity.data.dataset.dataset_subclass.bert_base_dataset import TrajectoryProcessingDataset
import datetime
import random



class TrajRoadDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)

class LineSeqDataset(AbstractDataset):

    def __init__(self,config):
        self.config = config
        self.model = config.get("model")
        preprocess_all(config)
        self.device = config.get('device')
        self._logger = getLogger()
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        data_cache_dir = os.path.join(cache_dir, self.dataset)
        self.data_cache_dir=data_cache_dir
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
        self.seq_len = config.get('max_len', 128)
        self.min_freq=config.get('min_freq',1)
        self.vocab_path = data_cache_dir+'/vocab_{}_True_{}.pkl'.format(self.dataset, self.min_freq)
        self.add_cls = self.config.get('add_cls', True)
        self.merge = self.config.get('merge', True)
        self.max_train_size = self.config.get('max_train_size', None)
        self.num_workers = self.config.get('num_workers', 0)

        # cache_dir
        self.cache_file_folder = './veccity/cache/dataset_cache/'
        self.cache_file_name = os.path.join(self.cache_file_folder,
                                            f'{self.model}_{self.dataset}_{int(self.edge_threshold*10)}_{self.seq_len}_{self.min_len}.pickle')
        self.build_vocab()
        self.construct_road_adj()
        self.geo_to_ind=self.vocab.loc2index
        self.ind_to_geo=self.vocab.index2loc
        select_geos=self.ind_to_geo[self.vocab.specials_num:]
        # change the same size and order to the vocab
        
        self.road_adj=self.road_adj[:,select_geos]
        self.road_adj=self.road_adj[select_geos,:]
        self.road_geo_df=self.road_geo_df.iloc[select_geos]
        self.num_nodes=self.vocab_size

    def construct_road_adj(self):
        self.road_adj = np.zeros(shape=[self.road_num,self.road_num])
        #构建路网的邻接关系
        with open(self.adj_json_path,'r',encoding='utf-8') as fp:
            road_adj_data = json.load(fp)
        for road in range(self.road_num):
            for neighbor in road_adj_data[str(road)]:
                self.road_adj[road][neighbor] = 1

    def load_cdtraj(self):
        trajs=pd.read_csv(self.traj_path)
        self.traj_list=[]
        self.temporal_mat_list=[]
        for i in tqdm(range(trajs.shape[0]), desc='traj_process'):
            traj = trajs.iloc[i]
            loc_list = eval(traj['path'])
            tim_list = eval(traj['tlist'])
            usr_id = traj['usr_id']
            new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in loc_list]
            new_tim_list = [datetime.datetime.fromtimestamp(tim) for tim in tim_list]
            minutes = [new_tim.hour * 60 + new_tim.minute + 1 for new_tim in new_tim_list]
            weeks = [new_tim.weekday() + 1 for new_tim in new_tim_list]
            usr_list = [self.vocab.usr2index.get(usr_id, self.vocab.unk_index)] * len(new_loc_list)
            if self.add_cls:
                new_loc_list = [self.vocab.sos_index] + new_loc_list
                minutes = [self.vocab.pad_index] + minutes
                weeks = [self.vocab.pad_index] + weeks
                usr_list = [usr_list[0]] + usr_list
                tim_list = [tim_list[0]] + tim_list
            temporal_mat = self._cal_mat(tim_list)
            self.temporal_mat_list.append(temporal_mat)
            traj_feat = np.array([new_loc_list, tim_list, minutes, weeks, usr_list]).transpose((1, 0))
            self.traj_list.append(traj_feat)
        # traj_feat: loc_list, tlist, minutes_list, weeks_list,user_list
        # temporal_mat_list for START

    def _gen_dataset(self):
        train_dataset = TrajectoryProcessingDataset(data_name=self.dataset,
                                                    data_type='train', vocab=self.vocab,
                                                    seq_len=self.seq_len, add_cls=self.add_cls,
                                                    merge=self.merge, min_freq=self.min_freq,
                                                    max_train_size=self.max_train_size)
        eval_dataset = TrajectoryProcessingDataset(data_name=self.dataset,
                                                   data_type='val', vocab=self.vocab,
                                                   seq_len=self.seq_len, add_cls=self.add_cls,
                                                   merge=self.merge, min_freq=self.min_freq,
                                                   max_train_size=None)
        test_dataset = TrajectoryProcessingDataset(data_name=self.dataset,
                                                   data_type='test', vocab=self.vocab,
                                                   seq_len=self.seq_len, add_cls=self.add_cls,
                                                   merge=self.merge, min_freq=self.min_freq,
                                                   max_train_size=None)
        # just for test
        # if self.choice:
        #     train_dataset=random.choices(train_dataset,k=1000)
        #     eval_dataset=random.choices(eval_dataset,k=1000)
        #     test_dataset=random.choices(test_dataset,k=1000)
        #     print(f'do random choice {self.choice}')
        return train_dataset, eval_dataset, test_dataset


    def _gen_dataloader(self, train_dataset, eval_dataset, test_dataset):
        assert self.collate_fn is not None
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers, shuffle=True,
                                      collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                           vocab=self.vocab, add_cls=self.add_cls),drop_last=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=True,
                                     collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                          vocab=self.vocab, add_cls=self.add_cls),drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=False,
                                     collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                          vocab=self.vocab, add_cls=self.add_cls))
        return train_dataloader, eval_dataloader, test_dataloader
    
    def build_vocab(self):
        if not os.path.exists(self.vocab_path):
            self.vocab = WordVocab(traj_path=self.traj_path,
                            min_freq=self.min_freq, use_mask=True, seq_len=self.seq_len)
            self.vocab.save_vocab(self.vocab_path)
            print("VOCAB SIZE ", len(self.vocab))
        else:
            self.vocab = WordVocab.load_vocab(self.vocab_path)
        self.usr_num=self.vocab.user_num
        self.vocab_size=self.vocab.vocab_size
        self._logger.info(f"user num:{self.vocab.user_num}")
        self._logger.info(f"vocab size:{self.vocab.vocab_size}")
        self._logger.info(f"del edge:{self.vocab.del_edge}")
        self._logger.info(f"len(vocab.all_edge):{len(self.vocab.all_edge)}")

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
        self._logger.info("Loading Dataset!")
        train_dataset, eval_dataset, test_dataset = self._gen_dataset()
        self._logger.info('Size of dataset: ' + str(len(train_dataset)) +
                         '/' + str(len(eval_dataset)) + '/' + str(len(test_dataset)))

        self._logger.info("Creating Dataloader!")
        return self._gen_dataloader(train_dataset, eval_dataset, test_dataset)

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {'dataloader':self.train_dataloader,'num_nodes':self.road_num,"edge_index":self.edge_index,
                "edge_index_aug":self.edge_index_aug,"traj_arr_test":self.traj_arr_test,"traj_arr_train":self.traj_arr_train}




# collate for MLM
def collate_unsuperv_mask(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, masks, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)  # (batch_size, padded_length, padded_length)

    # masks related to objective
    target_masks = torch.zeros_like(X, dtype=torch.bool)  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    target_masks = ~target_masks  # (batch_size, padded_length, feat_dim)
    target_masks = target_masks * padding_masks.unsqueeze(-1)

    targets = X.clone()
    targets = targets.masked_fill_(target_masks == 0, vocab.pad_index)

    X[..., 0:1].masked_fill_(target_masks[..., 0:1] == 1, vocab.mask_index)  # loc -> mask_index
    X[..., 1:].masked_fill_(target_masks[..., 1:] == 1, vocab.pad_index)  # others -> pad_index
    return X.long(), targets.long(), target_masks, padding_masks, batch_temporal_mat.long()

def padding_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))