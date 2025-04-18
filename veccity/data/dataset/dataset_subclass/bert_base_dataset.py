import os
import json
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import datetime
import pickle
import pandas as pd
import scipy.sparse as sp
from logging import getLogger
from torch.utils.data import DataLoader
from veccity.data.dataset import AbstractDataset
from veccity.data.dataset.dataset_subclass.bert_vocab import WordVocab
from veccity.utils import ensure_dir
from veccity.data.preprocess import preprocess_all, cache_dir

class BaseDataset(AbstractDataset):
    def __init__(self, config):
        self.logger = getLogger()
        self.config = config

        self.model_name=config.get('model','START')
        self.dataset = self.config.get('dataset', '')
        self.max_train_size = self.config.get('max_train_size', None)
        self.batch_size = self.config.get('batch_size', 64)
        self.num_workers = self.config.get('num_workers', 0)
        self.vocab_path = self.config.get('vocab_path', None)
        self.baseline_bert = self.config.get('baseline_bert', False)
        self.baseline_tf = self.config.get('baseline_tf', False)
        self.min_freq = self.config.get('min_freq', 1)
        self.merge = self.config.get('merge', True)
        preprocess_all(config)

        if self.vocab_path is None:
            self.vocab_path = 'raw_data/vocab_{}_True_{}.pkl'.format(self.dataset, self.min_freq)
            if self.merge:
                self.vocab_path = 'raw_data/vocab_{}_True_{}_merge.pkl'.format(self.dataset, self.min_freq)
        if self.baseline_bert:
            self.vocab_path = self.vocab_path[:-4]
            self.vocab_path += '_eos.pkl'
        self.seq_len = self.config.get('seq_len', 512)
        self.add_cls = self.config.get('add_cls', True)
        self.usr_num = 0
        self.vocab_size = 0
        self.vocab = None
        self.data_path = './raw_data/' + self.dataset + '/'
        self.roadnetwork = self.dataset
        self.gat_K = self.config.get('gat_K', 1)

        self.generate_constrastive_data()

        self._load_vocab()
        self.collate_fn = None
        self._logger = getLogger()

        
        
        # self.geo_file = self.config.get('geo_file', self.roadnetwork)
        # self.rel_file = self.config.get('rel_file', self.roadnetwork)
        self.bidir_adj_mx = self.config.get('bidir_adj_mx', False)
        print(self.selected_path + self.geo_file + '.geo')
        assert os.path.exists(self.selected_path + self.geo_file + '.geo')# 数据
        assert os.path.exists(self.selected_path + self.rel_file + '.grel')# 数据

        self.append_degree2gcn = self.config.get('append_degree2gcn', True)
        self.add_gat = self.config.get('add_gat', True)
        
        self.load_trans_prob = self.config.get('load_trans_prob', True) # 数据
        self.normal_feature = self.config.get('normal_feature', False)
        self.device = self.config.get('device', torch.device('cpu'))

        self.geo_file = self._load_geo()
        self.rel_file = self._load_rel()

        if self.add_gat:
            self.node_features, self.node_fea_dim = self._load_geo_feature(self.geo_file)
            self.edge_index, self.loc_trans_prob = self._load_k_neighbors_and_trans_prob()
            self.adj_mx_encoded = None
        else:
            self.node_features, self.node_fea_dim = None, 0
            self.adj_mx_encoded = None
            self.edge_index = None
            self.loc_trans_prob = None
    
    def generate_constrastive_data(self):
        # 首先生成vocab
        self.use_mask=True
        self.min_freq=1
        roadmap_path=""
        base_path = "veccity/cache/dataset_cache/{}/".format(self.model_name)
        ensure_dir(base_path)
        self.vocab_path=base_path+"/vocab_{}_{}_{}_merge.pkl".format(self.dataset,self.use_mask, self.min_freq)
        self.traj_path="veccity/cache/dataset_cache/{}/traj_road.csv".format(self.dataset)
        self.vocab=set_vocab(roadmap_path,self.traj_path,self.vocab_path)
        # 其次生成新的geo文件
        new_data_name = '{}_{}_{}'.format(self.dataset, self.use_mask, self.min_freq)
        road_path="veccity/cache/dataset_cache/{}/road.csv".format(self.dataset)
        self.selected_path = base_path
        self.geo_file_path = self.data_path+self.config.get('geo_file', self.roadnetwork) + '.geo'
        self.rel_file_path = self.data_path+self.config.get('rel_file', self.roadnetwork) + '.grel'
       
        select_geo_rel(self.vocab.all_edge,self.selected_path,new_data_name,road_path,self.geo_file_path,self.rel_file_path,self.use_mask,self.min_freq)
        # 然后进行with degree
        # 这里需要新生成的替换掉之前的geo file name 
        append_degree(self.selected_path,new_data_name, self.use_mask, self.min_freq)
        self.geo_file=new_data_name+'_withdegree'
        self.rel_file=new_data_name+'_withdegree'
        # 生成trans prob
        self.geo_file_path = os.path.join(self.selected_path,  self.geo_file) + '.geo'
        self.rel_file_path = os.path.join(self.selected_path, self.geo_file) + '.grel'
        traj_train="veccity/cache/dataset_cache/{}/traj_road_train.csv".format(self.dataset)
        trans_probs(self.geo_file_path,self.rel_file_path,base_path,self.dataset,self.gat_K,self.seq_len,traj_train)
        # 生成shift

        # 生成trim
        

    def _load_vocab(self):
        self.logger.info("Loading Vocab from {}".format(self.vocab_path)) # 数据
        # if self.vocab==None:
        self.vocab = WordVocab.load_vocab(self.vocab_path)
        self.selected_geo_uids=self.vocab.all_edge
        self.usr_num = self.vocab.user_num
        self.vocab_size = self.vocab.vocab_size
        self.logger.info('vocab_path={}, usr_num={}, vocab_size={}'.format(
            self.vocab_path, self.usr_num, self.vocab_size))

    def _load_geo(self):
        geofile = pd.read_csv(self.selected_path + self.geo_file + '.geo')
        self.geo_uids = list(geofile['id'])
        self.num_nodes = len(self.geo_uids)
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        for index, geo_uid in enumerate(self.geo_uids):
            self.geo_to_ind[geo_uid] = index
            self.ind_to_geo[index] = geo_uid
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_uids)))
        return geofile

    def _load_rel(self):
        relfile = pd.read_csv(self.selected_path + self.rel_file + '.grel')[['orig_geo_id', 'dest_geo_id']]# 数据od
        self.adj_mx = np.zeros((len(self.geo_uids), len(self.geo_uids)), dtype=np.float32)
        for row in relfile.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
            if self.bidir_adj_mx:
                self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1

        self._logger.info("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx.shape) +
                          ', edges=' + str(self.adj_mx.sum()))
        return relfile


    def _load_geo_feature(self, road_info):
        node_fea_path = self.selected_path + '{}_node_features.npy'.format(self.roadnetwork)
        if self.append_degree2gcn:
            node_fea_path = node_fea_path[:-4] + '_degree.npy'
        if os.path.exists(node_fea_path):
            node_features = np.load(node_fea_path)
        else:
            useful = ['highway', 'lanes', 'length', 'maxspeed']
            if self.append_degree2gcn:
                useful += ['outdegree', 'indegree']
            node_features = road_info[useful]
            norm_dict = {
                'length': 2,
            }
            for k, v in norm_dict.items():
                d = node_features[k]
                min_ = d.min()
                max_ = d.max()
                dnew = (d - min_) / (max_ - min_)
                node_features = node_features.drop(k, 1)
                node_features.insert(v, k, dnew)
            onehot_list = ['lanes', 'maxspeed', 'highway']
            if self.append_degree2gcn:
                onehot_list += ['outdegree', 'indegree']
            for col in onehot_list:
                dum_col = pd.get_dummies(node_features[col], col)
                node_features = node_features.drop(col, axis=1)
                node_features = pd.concat([node_features, dum_col], axis=1)
            node_features = node_features.values
            np.save(node_fea_path, node_features)

        self._logger.info('node_features: ' + str(node_features.shape))  # (N, fea_dim)
        node_fea_vec = np.zeros((self.vocab.vocab_size, node_features.shape[1]))
        for ind in range(len(node_features)):
            geo_uid = self.ind_to_geo[ind]
            encoded_geo_uid = self.vocab.loc2index[geo_uid]
            node_fea_vec[encoded_geo_uid] = node_features[ind]
        if self.normal_feature:
            self._logger.info('node_features by a/row_sum(a)')  # (vocab_size, fea_dim)
            row_sum = np.clip(node_fea_vec.sum(1), a_min=1, a_max=None)
            for i in range(len(node_fea_vec)):
                node_fea_vec[i, :] = node_fea_vec[i, :] / row_sum[i]
        node_fea_pe = torch.from_numpy(node_fea_vec).float().to(self.device)  # (vocab_size, fea_dim)
        self._logger.info('node_features_encoded: ' + str(node_fea_pe.shape))  # (vocab_size, fea_dim)
        return node_fea_pe, node_fea_pe.shape[1]

    def _load_k_neighbors_and_trans_prob(self):
        """
        Args:

        Returns:
            (vocab_size, pretrain_dim)
        """
        source_nodes_ids, target_nodes_ids = [], []
        seen_edges = set()
        g2n_path= os.path.join(self.selected_path, '{0}_neighbors_{1}.json'.format(self.dataset, self.gat_K))
        l2p_path= os.path.join(self.selected_path, '{0}_trans_prob_{1}.json'.format(self.dataset, self.gat_K))
        geoid2neighbors = json.load(open(g2n_path))
        if self.load_trans_prob:
            loc_trans_prob = []
            link2prob = json.load(open(l2p_path)) # 数据
        for k, v in geoid2neighbors.items():
            src_node = self.vocab.loc2index[int(k)]
            for tgt in v:
                trg_node = self.vocab.loc2index[int(tgt)]
                if (src_node, trg_node) not in seen_edges:
                    source_nodes_ids.append(src_node)
                    target_nodes_ids.append(trg_node)
                    seen_edges.add((src_node, trg_node))
                    if self.load_trans_prob:
                        loc_trans_prob.append(link2prob[str(k) + '_' + str(tgt)])
        # add_self_edge
        for i in range(self.vocab.vocab_size):
            if (i, i) not in seen_edges:
                source_nodes_ids.append(i)
                target_nodes_ids.append(i)
                seen_edges.add((i, i))
                if self.load_trans_prob:
                    loc_trans_prob.append(link2prob.get(str(i) + '_' + str(i), 0.0))
        # shape = (2, E), where E is the number of edges in the graph
        edge_index = torch.from_numpy(np.row_stack((source_nodes_ids, target_nodes_ids))).long().to(self.device)
        self.logger.info('edge_index: ' + str(edge_index.shape))  # (vocab_size, pretrain_dim)
        if self.load_trans_prob:
            loc_trans_prob = torch.from_numpy(np.array(loc_trans_prob)).unsqueeze(1).float().to(self.device)  # (E, 1)
            self._logger.info('Trajectory loc-transfer prob shape={}'.format(loc_trans_prob.shape))
        else:
            loc_trans_prob = None
        return edge_index, loc_trans_prob

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
        return train_dataset, eval_dataset, test_dataset

    def _gen_dataloader(self, train_dataset, eval_dataset, test_dataset):
        assert self.collate_fn is not None
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers, shuffle=True,
                                      collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                           vocab=self.vocab, add_cls=self.add_cls))
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=True,
                                     collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                          vocab=self.vocab, add_cls=self.add_cls))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=False,
                                     collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                          vocab=self.vocab, add_cls=self.add_cls))
        return train_dataloader, eval_dataloader, test_dataloader

    def get_data(self):
        self.logger.info("Loading Dataset!")
        train_dataset, eval_dataset, test_dataset = self._gen_dataset()
        self.logger.info('Size of dataset: ' + str(len(train_dataset)) +
                         '/' + str(len(eval_dataset)) + '/' + str(len(test_dataset)))

        self.logger.info("Creating Dataloader!")
        return self._gen_dataloader(train_dataset, eval_dataset, test_dataset)

    def get_data_feature(self):
        data_feature = {'usr_num': self.usr_num, 'vocab_size': self.vocab_size, 'vocab': self.vocab,
                        "adj_mx": self.adj_mx, "num_nodes": self.num_nodes,
                        "geo_file": self.geo_file, "rel_file": self.rel_file,
                        "geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
                        "node_features": self.node_features, "node_fea_dim": self.node_fea_dim,
                        "adj_mx_encoded": self.adj_mx_encoded, "edge_index": self.edge_index,
                        "loc_trans_prob": self.loc_trans_prob}
        return data_feature


class TrajectoryProcessingDataset(Dataset):

    def __init__(self, data_name, data_type, vocab, seq_len=512,
                 add_cls=True, merge=True, min_freq=1, min_seq_len=10, max_train_size=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.add_cls = add_cls
        self.max_train_size = max_train_size
        self._logger = getLogger()
        self.data_type=data_type

        self.data_path = cache_dir+'/{}/traj_road_{}.csv'.format(data_name, data_type)
        self.cache_path = cache_dir+'/{}/cache_{}_{}_{}_{}_{}.pkl'.format(
            data_name,data_name, data_type, add_cls, merge, min_freq)
        self.temporal_mat_path = self.cache_path[:-4] + '_temporal_mat.pkl'
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.cache_path) and os.path.exists(self.temporal_mat_path):
            self.traj_list = pickle.load(open(self.cache_path, 'rb'))
            self.temporal_mat_list = pickle.load(open(self.temporal_mat_path, 'rb'))
            self._logger.info('Load dataset from {}'.format(self.cache_path))
        else:
            origin_data_df = pd.read_csv(self.data_path) #数据
            self.traj_list, self.temporal_mat_list = self.data_processing(origin_data_df)
        if self.max_train_size is not None:
            self.traj_list = self.traj_list[:self.max_train_size]
            self.temporal_mat_list = self.temporal_mat_list[:self.max_train_size]

    def _cal_mat(self, tim_list):
        # calculate the temporal relation matrix
        seq_len = len(tim_list)
        mat = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                off = abs(tim_list[i] - tim_list[j])
                mat[i][j] = off
        return mat  # (seq_len, seq_len)

    def data_processing(self, origin_data, desc=None, cache_path=None, tmat_path=None):
        self._logger.info(f'Processing {self.data_type} dataset in TrajectoryProcessingDataset!')
        sub_data = origin_data[['path', 'tlist', 'usr_id', 'traj_uid']]
        traj_list = []
        temporal_mat_list = []
        for i in tqdm(range(sub_data.shape[0]), desc=desc):
            traj = sub_data.iloc[i]
            loc_list = eval(traj['path'])
            if len(loc_list) < 10:
                continue
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

            temporal_mat_list.append(temporal_mat)
            traj_feat = np.array([new_loc_list, tim_list, minutes, weeks, usr_list]).transpose((1, 0))
            traj_list.append(traj_feat)
        if cache_path is None:
            cache_path = self.cache_path
        if tmat_path is None:
            tmat_path = self.temporal_mat_path
        pickle.dump(traj_list, open(cache_path, 'wb'))
        pickle.dump(temporal_mat_list, open(tmat_path, 'wb'))
        return traj_list, temporal_mat_list

    def __len__(self):
        return len(self.traj_list)

    def __getitem__(self, ind):
        traj_ind = self.traj_list[ind]  # (seq_length, feat_dim)
        temporal_mat = self.temporal_mat_list[ind]  # (seq_length, seq_length)
        return torch.LongTensor(traj_ind), torch.LongTensor(temporal_mat)


def padding_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))



def set_vocab(roadmap_path,traj_path,vocab_path):
    min_freq=0
    use_mask=True
    seq_len=128

    if not os.path.exists(vocab_path):
        vocab = WordVocab(traj_path=traj_path, roadmap_path=roadmap_path,
                        min_freq=min_freq, use_mask=use_mask, seq_len=seq_len)
        vocab.save_vocab(vocab_path)
        print("VOCAB SIZE ", len(vocab))
    else:
        vocab = WordVocab.load_vocab(vocab_path)

    print('user num ', vocab.user_num)
    print("vocab size ", vocab.vocab_size)
    print("del edge ", vocab.del_edge)
    print("len(vocab.all_edge) ", len(vocab.all_edge))
    return vocab

def trans_probs(geo_path,rel_path,base_path,road_name,K,max_length,traj_train):

    geo = pd.read_csv(geo_path)
    print('Geo', geo.shape)

    geo_uids = list(geo['id'])
    num_nodes = len(geo_uids)
    geo_to_ind = {}
    ind_to_geo = {}
    for index, geo_uid in enumerate(geo_uids):
        geo_to_ind[geo_uid] = index
        ind_to_geo[index] = geo_uid

    path = os.path.join(base_path, '{0}_neighbors_{1}.json'.format(road_name, K))
    if os.path.exists(path):
        geoid2neighbors = json.load(open(path, 'r'))
    else:
        relfile = pd.read_csv(rel_path)[['orig_geo_id', 'dest_geo_id']]
        print('Rel', relfile.shape)
        adj_mx = np.zeros((len(geo_uids), len(geo_uids)), dtype=np.float32)
        for row in relfile.values:
            if row[0] not in geo_to_ind or row[1] not in geo_to_ind:
                continue
            adj_mx[geo_to_ind[row[0]], geo_to_ind[row[1]]] = 1

        adj_mx_bool = adj_mx.astype('bool')
        k_adj_mx_list = [adj_mx_bool]
        for i in tqdm(range(2, K + 1)):
            k_adj_mx_list.append(cal_matmul(k_adj_mx_list[-1], adj_mx_bool))
            np.save(os.path.join(base_path, '{0}_adj_{1}.npy'.format(road_name, i)), k_adj_mx_list[-1])
        print('Finish K order adj_mx')
        for i in tqdm(range(1, len(k_adj_mx_list))):
            adj_mx_bool += k_adj_mx_list[i]
        print('Finish sum of K order adj_mx')
        geoid2neighbors = {}
        for i in tqdm(range(len(adj_mx_bool)), desc='count neighbors'):
            geo_uid = int(ind_to_geo[i])
            geoid2neighbors[geo_uid] = []
            for j in range(adj_mx_bool.shape[1]):
                if adj_mx_bool[i][j] == 0:
                    continue
                ner_id = int(ind_to_geo[j])
                geoid2neighbors[geo_uid].append(ner_id)
        json.dump(geoid2neighbors, open(path, 'w'))
        print('Total edge@{} = {}'.format(1, adj_mx.sum()))
        print('Total edge@{} = {}'.format(K, adj_mx_bool.sum()))

    path = os.path.join(base_path, '{0}_trans_prob_{1}.json'.format(road_name, K))
    if os.path.exists(path):
        link2prob = json.load(open(path, 'r'))
    else:
        node_array = np.zeros([num_nodes, num_nodes], dtype=float)
        print(node_array.shape)
        count_array_row = np.zeros([num_nodes], dtype=int)
        count_array_col = np.zeros([num_nodes], dtype=int)

        train_file = traj_train
        train = pd.read_csv(train_file, dtype={'id': int, 'hop': int, 'traj_uid': int})

        for _, row in tqdm(train.iterrows(), total=train.shape[0], desc='count traj prob'):
            plist = eval(row.path)[:max_length]
            for i in range(len(plist) - 1):
                prev_geo = plist[i]
                for j in range(1, K+1):
                    if i + j >= len(plist):
                        continue
                    next_geo = plist[i + j]
                    prev_ind = geo_to_ind[prev_geo]
                    next_ind = geo_to_ind[next_geo]
                    count_array_row[prev_ind] += 1
                    count_array_col[next_ind] += 1
                    node_array[prev_ind][next_ind] += 1

        assert (count_array_row == (node_array.sum(1))).sum() == len(count_array_row)  # 按行求和
        assert (count_array_col == (node_array.sum(0))).sum() == len(count_array_col)  # 按列求和

        node_array_out = node_array.copy()
        for i in tqdm(range(node_array_out.shape[0])):
            count = count_array_row[i]
            if count == 0:
                print(i, 'no out-degree')
                continue
            node_array_out[i, :] /= count

        node_array_in = node_array.copy()
        for i in tqdm(range(node_array_in.shape[0])):
            count = count_array_col[i]
            if count == 0:
                print(i, 'no in-degree')
                continue
            node_array_in[:, i] /= count

        link2prob = {}
        for k, v in geoid2neighbors.items():
            for tgt in v:
                id_ = str(k) + '_' + str(tgt)
                p = node_array_in[geo_to_ind[int(k)]][geo_to_ind[int(tgt)]]
                link2prob[id_] = float(p)
        path = os.path.join(base_path, '{0}_trans_prob_{1}.json'.format(road_name, K))
        json.dump(link2prob, open(path, 'w'))
    

def cal_matmul(mat1, mat2):
    n = mat1.shape[0]
    assert mat1.shape[0] == mat1.shape[1] == mat2.shape[0] == mat2.shape[1]
    res = np.zeros((n, n), dtype='bool')
    for i in tqdm(range(n), desc='outer'):
        for j in tqdm(range(n), desc='inner'):
            res[i, j] = np.dot(mat1[i, :], mat2[:, j])
    return res


def select_geo_rel(selected_geo_uids, selected_path,new_data_name,road_path, geo_path,rel_path, use_mask, min_freq):
    # 去掉没有被vocab选上的road所对应的geo和

    ensure_dir(selected_path)
    selected_geo_uids = set(selected_geo_uids)


    if os.path.exists(selected_path + '/{}.geo'.format(new_data_name)) and \
            os.path.exists(selected_path + '/{}.grel'.format(new_data_name)):
        return

    geofile = pd.read_csv(road_path)
    geo_df = pd.read_csv(geo_path)
    offset = geo_df[geo_df['traffic_type'] == 'region'].shape[0]
    geo = []
    if 'id' not in geofile.keys():
        geofile['id'] = list(range(len(geofile)))
    for i in tqdm(range(geofile.shape[0]), desc='geo'):
        if int(geofile.iloc[i]['id']) in selected_geo_uids:
           geo.append(geofile.iloc[i].values.tolist())
    geo = pd.DataFrame(geo, columns=geofile.columns)
    geo.to_csv(selected_path + '/{}.geo'.format(new_data_name), index=False)

    relfile = pd.read_csv(rel_path)
    relfile = relfile[relfile.rel_type=='road2road']
    relfile.orig_geo_id=relfile.orig_geo_id-offset
    relfile.dest_geo_id=relfile.dest_geo_id-offset
    rel = []
    for i in tqdm(range(relfile.shape[0]), desc='rel'):
        oid = relfile.iloc[i]['orig_geo_id']
        did = relfile.iloc[i]['dest_geo_id']
        if oid not in selected_geo_uids or did not in selected_geo_uids:
            continue
        rel.append(relfile.iloc[i].values.tolist())
    rel = pd.DataFrame(rel, columns=relfile.columns)
    rel.to_csv(selected_path + '/{}.grel'.format(new_data_name), index=False)

    config = {"info": {
        "geo_file": new_data_name,
        "rel_file": new_data_name
    }}
    json.dump(config, open(selected_path + '/config.json', 'w'), indent=4)


def append_degree(selected_path, new_data_name, use_mask, min_freq):
  
    ensure_dir(selected_path)

    if os.path.exists(selected_path + '/{}_withdegree.geo'.format(new_data_name)) and \
            os.path.exists(selected_path + '/{}_withdegree.grel'.format(new_data_name)):
        return

    geo_file = selected_path + '/{}.geo'.format(new_data_name)
    rel_file = selected_path + '/{}.grel'.format(new_data_name)

    geo = pd.read_csv(geo_file)
    rel = pd.read_csv(rel_file)[['orig_geo_id', 'dest_geo_id']]

    geo_uids = list(geo['id'])
    geo_to_ind = {}
    ind_to_geo = {}
    for index, geo_uid in enumerate(geo_uids):
        geo_to_ind[geo_uid] = index
        ind_to_geo[index] = geo_uid

    adj_mx = np.zeros((len(geo_uids), len(geo_uids)), dtype=np.float32)
    for row in rel.values:
        if row[0] not in geo_to_ind or row[1] not in geo_to_ind:
            print(row[0], row[1])
            continue
        adj_mx[geo_to_ind[row[0]], geo_to_ind[row[1]]] = 1

    outdegree = np.sum(adj_mx, axis=1)  # (N, )
    indegree = np.sum(adj_mx.T, axis=1)  # (N, )
    outdegree_list = []
    indegree_list = []

    for i, row in tqdm(geo.iterrows(), total=geo.shape[0]):
        geo_uid = row.id
        outdegree_i = outdegree[geo_to_ind[geo_uid]]
        indegree_i = indegree[geo_to_ind[geo_uid]]
        outdegree_list.append(int(outdegree_i))
        indegree_list.append(int(indegree_i))

    geo.insert(loc=geo.shape[1], column='outdegree', value=outdegree_list)
    geo.insert(loc=geo.shape[1], column='indegree', value=indegree_list)

    rel.to_csv(rel_file[:-4] + '_withdegree.grel', index=False)
    geo.to_csv(geo_file[:-4] + '_withdegree.geo', index=False)





    
    
