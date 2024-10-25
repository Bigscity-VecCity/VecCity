from multiprocessing import Process,Manager
from veccity.data.dataset.dataset_subclass.bertlm_dataset import noise_mask
from veccity.data.dataset.lineseq_dataset import LineSeqDataset
from veccity.data.preprocess import preprocess_all, cache_dir
import torch
from veccity.data.dataset.dataset_subclass.bert_base_dataset import TrajectoryProcessingDataset, padding_mask
import numpy as np
import random
from logging import getLogger
import pickle
import os
from tqdm import trange


class ToastDataset(LineSeqDataset):
    def __init__(self,config):
        super().__init__(config)
        self.id2node,self.node2id,self.node2type,self.type_num=self.setup_vocab()
        self.walker=RandomWalker(self.road_adj,self.node2id,self.node2type)
        walk_nums=1000
        self.fake_traj=self.walker.generate_sentences_bert(self.dataset,walk_nums)
        # 要保存下来
        self.collate_fn=collate_tost

    def _gen_dataset(self):
        train_dataset = ToastTrajDataset(data_name=self.dataset,
                                                    data_type='train', vocab=self.vocab,
                                                    seq_len=self.seq_len, add_cls=self.add_cls,
                                                    merge=self.merge, min_freq=self.min_freq,
                                                    max_train_size=self.max_train_size,fake_traj=self.fake_traj)
        eval_dataset = ToastTrajDataset(data_name=self.dataset,
                                                   data_type='val', vocab=self.vocab,
                                                   seq_len=self.seq_len, add_cls=self.add_cls,
                                                   merge=self.merge, min_freq=self.min_freq,
                                                   max_train_size=None,fake_traj=self.fake_traj)
        test_dataset = ToastTrajDataset(data_name=self.dataset,
                                                   data_type='test', vocab=self.vocab,
                                                   seq_len=self.seq_len, add_cls=self.add_cls,
                                                   merge=self.merge, min_freq=self.min_freq,
                                                   max_train_size=None,fake_traj=self.fake_traj)
        return train_dataset, eval_dataset, test_dataset
    
    def setup_vocab(self):
        id2node={}
        node2id={}
        node2type={}

        for i in range(self.num_nodes-self.vocab.specials_num):
            id2node[i] = i
            node2id[i] = i
            node2type[i] = int(self.road_geo_df.iloc[i]['highway'])

        type_num=np.max(list(node2type.values()))+1
        
        return id2node,node2id,node2type,type_num

    

    
    def get_data_feature(self):
        """
        返回一个 dict,包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {'node2id':self.node2id,'id2node':self.id2node,'num_nodes':self.num_nodes, \
                'adj_mx':self.road_adj,'node2type':self.node2type,'road_lengths':self.road_length, 'type_num':self.type_num,"vocab":self.vocab}




# collate for general
def collate_tost(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, masks, is_traj = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    target_masks = torch.zeros_like(X, dtype=torch.bool)  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    target_masks = ~target_masks  # (batch_size, padded_length, feat_dim)
    target_masks = target_masks * padding_masks.unsqueeze(-1)

    targets = X.clone()
    targets = targets.masked_fill_(target_masks == 0, vocab.pad_index)

    X[..., 0:1].masked_fill_(target_masks[..., 0:1] == 1, vocab.mask_index)  # loc -> mask_index
    X[..., 1:].masked_fill_(target_masks[..., 1:] == 1, vocab.pad_index)  # others -> pad_index
    batch={}
    batch['seq']=X.long().squeeze()
    batch['targets']=targets.long()
    batch['target_masks']=target_masks
    batch['padding_masks']=padding_masks
    batch['is_traj']=torch.tensor(is_traj)
    batch['length']=torch.tensor(lengths, dtype=torch.int16)
    return batch


class RandomWalker():
    def __init__(self, adj_matrix, node2id, node2type):
        self.G = adj_matrix
        self.neighbors = [[]] * self.G.shape[0]
        for v in range(self.G.shape[0]):
            self.neighbors[v] = list(np.where(self.G[v]==1)[0].tolist())
        self.node2id = node2id
        self.node2type = node2type
        self.sentences = []
        self._logger = getLogger()

    def generate_sentences_bert(self,dataset, num_walks=24):
        sts = []
        self._logger.info('num_walks ' + str(num_walks))
        sts_path = f'veccity/cache/dataset_cache/{dataset}/sts_{num_walks}.pkl'
        if not os.path.exists(sts_path):
            for _ in trange(num_walks):
                sts.extend(self.random_walks())
            with open(sts_path, 'wb') as f:
                pickle.dump(sts, f, protocol=4)
            self._logger.info('Saved sts.')
        with open(sts_path, 'rb') as f:
            sts = pickle.load(f)
        self._logger.info('Loaded sts.')
        return sts

    def generate_sentences_bert_type(self, num_walks=24):
        sts = []
        for _ in range(num_walks):
            sts.extend(self.random_walks_type())
        return sts

    def generate_sentences_dw(self, num_walks=24):
        sts = []
        for _ in range(num_walks):
            sts.extend(self.random_walks_dw())
        return sts

    def random_walks_type(self):
        # random walk with every node as start point once

        walks = []
        nodes = list(self.G.nodes())
        random.shuffle(nodes)
        for node in nodes:
            walk = [self.node2id[node]+2]
            tp = []
            if node in self.node2type:
                tp.append(self.node2type[node])
            else:
                tp.append(random.randint(0, 4))
            v = node
            length_walk = random.randint(5, 100)
            for _ in range(length_walk):
                nbs = list(np.where(self.G[v]==1)[0].tolist())
                if len(nbs) == 0:
                    break
                v = random.choice(nbs)
                walk.append(self.node2id[v] + 2)
                if v in self.node2type:
                    tp.append(self.node2type[v])
                else:
                    tp.append(random.randint(0, 4))
            walks.append([walk,tp])
        return walks
    
    def gen_one_random_walk(self, node):
        walk = [self.node2id[node] + 2]
        v = node
        length_walk = random.randint(6, 40)
        for _ in range(length_walk):
            nbs = self.neighbors[v]
            # nbs = list(np.where(self.G[v]==1)[0].tolist())
            if len(nbs) == 0:
                break
            v = random.choice(nbs)
            walk.append(self.node2id[v] + 2)
        return walk

    def random_walks(self):
        # random walk with every node as start point once
        walks = []
        nodes = list(range(self.G.shape[0]))
        random.shuffle(nodes)
        if len(nodes) < 1000:
            for node in nodes:
                walks.append(self.gen_one_random_walk(node))
        else:
            manager = Manager()
            shared_list = manager.list()
            processes = []
            num_processes = 8
            num = (len(nodes) + num_processes - 1) // num_processes
            
            for i in range(num_processes):
                start = num * i
                end = min(num * (i + 1), len(nodes))
                p = Process(target=self.multiprocess_random_walks, args=(shared_list, nodes[start:end]))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            walks = list(shared_list)

        return walks
    
    def multiprocess_random_walks(self, shared_list, nodes):
        for node in nodes:
            shared_list.append(self.gen_one_random_walk(node))    

    def random_walks_dw(self):
        # random walk with every node as start point once
        walks = []
        nodes = list(range(self.G.shape[0]))
        random.shuffle(nodes)
        for node in nodes:
            walk = [str(node)]
            v = node
            length_walk = random.randint(5, 100)
            for _ in range(length_walk):
                nbs = list(np.where(self.G[v]==1)[0].tolist())
                if len(nbs) == 0:
                    break
                v = random.choice(nbs)
                walk.append(str(v))
            walks.append(walk)
        return walks


class ToastTrajDataset(TrajectoryProcessingDataset):

    def __init__(self, **kwargs):
        self.fake_traj=kwargs['fake_traj']
        kwargs.pop('fake_traj')
        super().__init__(**kwargs)
        self.vocab=kwargs['vocab']

    def __len__(self):
        return len(self.traj_list)

    def __getitem__(self, ind):
        if random.random() > 0.5:
            traj_ind = self.traj_list[ind]  # (seq_length, feat_dim)
            is_traj = True
            
        else:
            traj_ind=random.choice(self.fake_traj)
            traj_ind = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in traj_ind]
            is_traj = False

        
        traj_ind=np.array(traj_ind)
        if len(traj_ind.shape)==1:
            traj_ind=np.expand_dims(traj_ind,-1)
        else:
            traj_ind=traj_ind[:,0:1]
        
        
        mask = noise_mask(traj_ind, 0.2, 3, "together", "random",
                          None, True)  # (seq_length, feat_dim) boolean array
        
        return torch.LongTensor(traj_ind), torch.LongTensor(mask), is_traj

            