import numpy as np
import torch
import random
from torch.utils.data import Dataset
from veccity.data.dataset import AbstractDataset
from veccity.data.preprocess import preprocess_all, cache_dir
from logging import getLogger
import os
import pandas as pd
import json
from tqdm import trange
import pickle
from multiprocessing import Process, Manager


PAD=0
MASK=1
global_config = None

class ToastDataset(AbstractDataset):
    def __init__(self, config):
        global global_config
        global_config = config
        self.config = config    
        preprocess_all(config)
        self._logger = getLogger()
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.device = config.get('device', torch.device('cpu'))
        data_cache_dir = os.path.join(cache_dir, self.dataset)

        self.max_len=config.get('max_len',32)

        # file_path, node2id, seq_len, max_pred, mask_ratio
        self.traj_path = os.path.join(data_cache_dir, 'traj_road_train.csv')
        self.adj_json_path = os.path.join(data_cache_dir, 'road_neighbor.json')
        self.road_feature_path = os.path.join(data_cache_dir, 'road_features.csv')
        self.road_feature = pd.read_csv(self.road_feature_path,delimiter=',')
        self.road_geo_path = os.path.join(data_cache_dir, 'road.csv')
        self.road_geo_df = pd.read_csv(self.road_geo_path, delimiter=',')
        
        self.road_length = np.array(self.road_geo_df['length'])
        self.road_num = len(self.road_length)
        self.road_adj = self.construct_road_adj()
        self.id2node,self.node2id,self.node2type,self.type_num=self.setup_vocab()
        self.max_pred=int(self.max_len * 0.25)
        self.dataloader=ToastDataLoader(self.dataset,self.road_adj,self.node2id,self.node2type,self.max_len,self.max_pred)

    def setup_vocab(self):
        id2node={}
        node2id={}
        node2type={}

        for i in range(self.road_num):
            id2node[i] = i
            node2id[i] = i
            node2type[i] = int(self.road_feature['highway'][i])


        type_num=np.max(list(node2type.values()))+1
        
        return id2node,node2id,node2type,type_num

    def construct_road_adj(self):
        road_adj = np.zeros(shape=[self.road_num, self.road_num])
        # 构建路网的邻接关系
        with open(self.adj_json_path, 'r', encoding='utf-8') as fp:
            road_adj_data = json.load(fp)
        for road in range(self.road_num):
            road_adj[road][road] = 1
            for neighbor in road_adj_data[str(road)]:
                road_adj[road][neighbor] = 1
        return road_adj

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        return self.dataloader, None, None

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {'dataloader':self.dataloader, 'node2id':self.node2id,'id2node':self.id2node,'num_node':self.road_num, \
                'adj_mx':self.road_adj,'node2type':self.node2type,'road_lengths':self.road_length, 'type_num':self.type_num}




class ToastDataLoader(Dataset):
    def __init__(self, dataset,adj_matrix, node2id, node2type, seq_len, max_pred, mode=0, mask_ratio=0.25):
        self.node2id = node2id
        self.mask_ratio = mask_ratio
        self.seq_len = seq_len
        self.max_pred = max_pred
        if mode == 1:
            self.walker = RandomWalker_Traj(adj_matrix,node2id)
        else:
            self.walker = RandomWalker(adj_matrix, node2id, node2type)
            # print("use unbiased random walk")
        self.traj_path="./veccity/cache/dataset_cache/{}/traj_road_train.csv".format(dataset)

        self.load(self.traj_path)

    def load(self, file_path):
        data=pd.read_csv(file_path)
        self.trajs=data.path.apply(eval)
        print(self.trajs[0])
        print("load trajectory completed")

    def gen_new_walks(self, num_walks):
        self.walks = self.walker.generate_sentences_bert(num_walks)
        random.shuffle(self.walks)

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, item):

        if random.random() > 0.5:
            traj = self.trajs[item]
            if len(traj) > self.seq_len:
                traj = self.cut_traj(traj)
            is_traj = True
            traj_input, traj_masked_tokens, traj_masked_pos, traj_masked_weights, traj_label = self.random_word(traj)
        else:
            lens=len(self.walks)
            ind=item%lens
            walk = self.walks[ind]
            if len(walk) > self.seq_len:
                traj = self.cut_traj(walk)
            is_traj = False
            traj_input, traj_masked_tokens, traj_masked_pos, traj_masked_weights, traj_label = self.process_walk(walk)

        traj_input = traj_input
        traj_label = traj_label
        input_mask = [1] * len(traj_input)
        lenth = [len(traj_input)]

        masked_lenth = len(traj_masked_tokens)
        padding = [0 for _ in range(self.seq_len - len(traj_input))]
        traj_input.extend(padding)
        input_mask.extend(padding)
        traj_label.extend(padding)

        if self.max_pred > masked_lenth:
            padding = [0] * (self.max_pred - masked_lenth)
            traj_masked_tokens.extend(padding)
            traj_masked_pos.extend(padding)
            traj_masked_weights.extend(padding)
        else:
            traj_masked_tokens = traj_masked_tokens[:self.max_pred]
            traj_masked_pos = traj_masked_pos[:self.max_pred]
            traj_masked_weights = traj_masked_weights[:self.max_pred]



        output = {"traj_input": traj_input, "traj_label": traj_label, "input_mask": input_mask, 'length': lenth,
                  "masked_pos": traj_masked_pos, "masked_tokens": traj_masked_tokens, "masked_weights": traj_masked_weights, "is_traj": is_traj}

        # print(output)

        return {key: torch.tensor(value) for key, value in output.items()}

    def cut_traj(self, traj):
        start_idx = int((len(traj)-self.seq_len) * random.random())
        return traj[start_idx: start_idx+self.seq_len]

    def process_walk(self, walk):
        tokens = walk
        output_label = []

        mask_len = int(len(tokens) * self.mask_ratio)
        start_loc = round( len(tokens) * random.random() * (1 - self.mask_ratio))

        masked_pos = list(range(start_loc, start_loc + mask_len))
        masked_tokens = tokens[start_loc: start_loc + mask_len]
        masked_weights = [0] * len(masked_tokens)

        for i, token in enumerate(tokens):
            if i >= start_loc and i < start_loc + mask_len:
                tokens[i] = MASK
                output_label.append(token)
            else:
                output_label.append(0)

        assert len(tokens) == len(output_label)

        return tokens, masked_tokens, masked_pos, masked_weights, output_label


    def random_word(self, sentence):
        tokens = sentence
        output_label = []

        mask_len = int(len(tokens) * self.mask_ratio)
        start_loc = round(len(tokens) * random.random() * (1- self.mask_ratio))

        masked_pos = list(range(start_loc, start_loc+mask_len))
        masked_tokens = tokens[start_loc: start_loc+mask_len]
        masked_weights = [1] * len(masked_tokens)

        for i, token in enumerate(tokens):
            if i >= start_loc and i < start_loc + mask_len:
                tokens[i] = MASK
                output_label.append(token)
            else:
                output_label.append(0)

        assert len(tokens) == len(output_label)

        return tokens, masked_tokens, masked_pos, masked_weights, output_label


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

    def generate_sentences_bert(self, num_walks=24):
        sts = []
        self._logger.info('num_walks ' + str(num_walks))
        global global_config
        dataset = global_config.get('dataset')
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
        length_walk = random.randint(5, 100)
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
        if len(nodes) < 10000:
            for node in nodes:
                walks.append(self.gen_one_random_walk(node))
        else:
            manager = Manager()
            shared_list = manager.list()
            processes = []
            num_processes = 24
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

class RandomWalker_Traj():
    def __init__(self, adj_matrix,node2id):
        self.G = adj_matrix
        self.node2id = node2id
        self.sentences = []

    def generate_sentences_bert(self, num_walks=24):
        sts = []
        for _ in trange(num_walks):
            sts.extend(self.random_walks())
        return sts

    def generate_sentences_dw(self, num_walks=24):
        sts = []
        for _ in range(num_walks):
            sts.extend(self.random_walks_dw())
        return sts

    def random_walks(self):
        # random walk with every node as start point once
        walks = []
        nodes = list(range(self.G.shape[0]))
        random.shuffle(nodes)
        for node in nodes:
            walk = [self.node2id[node] + 2]
            v = node
            length_walk = random.randint(5, 100)
            for _ in range(length_walk):
                dst = list(np.where(self.G[v]==1)[0].tolist())
                weights = self.G[v][dst]
                # print(dst, weights)
                probs = np.array(weights) / np.sum(weights)
                v = np.random.choice(dst, 1, p=probs)[0]
                walk.append(self.node2id[v] + 2)
            walks.append(walk)
        return walks

    def random_walks_dw(self):
        walks = []
        nodes = list(range(self.G.shape[0]))
        random.shuffle(nodes)
        for node in nodes:
            walk = [str(node)]
            v = node
            length_walk = random.randint(5, 100)
            for _ in range(length_walk):
                dst = list(np.where(self.G[v]==1)[0].tolist())
                weights = self.G[v][dst]
                weights = [w['weight'] for w in weights]
                probs = np.array(weights) / np.sum(weights)
                v = np.random.choice(dst, 1, p=probs)[0]
                walk.append(str(v))
            walks.append(walk)
        return walks

