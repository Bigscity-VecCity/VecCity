import os
import math
import pandas as pd
import numpy as np
from logging import getLogger
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import networkx as nx
from itertools import cycle, islice
from tqdm import tqdm
import copy
from veccity.data.dataset.dataset_subclass.sts_dataset import STSDataset
from veccity.downstream.downstream_models.abstract_model import AbstractModel
from veccity.data.preprocess import preprocess_detour

def k_shortest_paths_nx(G, source, target, k, weight='weight'):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def build_graph(rel_file, geo_file):

    rel = pd.read_csv(rel_file)
    geo = pd.read_csv(geo_file)
    
    edge2len = {}
    geoid2coord = {}
    for i, row in tqdm(geo.iterrows(), total=geo.shape[0]):
        geo_uid = row.geo_uid
        length = float(row.length)
        edge2len[geo_uid] = length
        # geoid2coord[geo_uid] = row.coordinates

    graph = nx.DiGraph()

    for i, row in tqdm(rel.iterrows(), total=rel.shape[0]):
        prev_id = row.orig_geo_id
        curr_id = row.dest_geo_id

        # Use length as weight
        # weight = geo.iloc[prev_id].length
        
        # Use avg_speed as weight
        weight = row.avg_time
        if weight == float('inf'):
            # weight = 9999999
            pass
            # print(row)
        graph.add_edge(prev_id, curr_id, weight=weight)

    return graph

class TrajEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, embedding, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.n_layers = n_layers
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=0.1 if n_layers > 1 else 0.0, batch_first=True)

    def forward(self, batch):
        path=batch['seq'][:,:,0]
        # valid_len=batch['lengths']
        padding_masks=batch['padding_masks']
        
        # original_shape = path.shape  # [batch_size, traj_len]
        # full_embed = [torch.from_numpy(self.embedding[int(i)]).to(torch.float32) for i in path.reshape(-1)]
        # full_embed = torch.stack(full_embed)
        # full_embed = full_embed.view(*original_shape, self.input_dim).to(self.device)  # [batch_size, traj_len, embed_size]
        full_embed = self.embedding.encode(path)
        # pack_x = pack_padded_sequence(full_embed, lengths=valid_len, batch_first=True, enforce_sorted=False).to(self.device)
        h0 = torch.zeros(self.n_layers, full_embed.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, full_embed.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(full_embed, (h0, c0))
        padding_masks=padding_masks.unsqueeze(-1).to(self.device)
        # out, _ = pad_packed_sequence(out, batch_first=True)
        out = torch.sum(out * padding_masks, 1) / torch.sum(padding_masks, 1)
        return out

    def encode_sequence(self,batch):
        return self.forward(batch)


class STSModel(nn.Module):
    def __init__(self, embedding,device,input_size=128,dropout_prob=0.2):
        super().__init__()
        
        self.traj_encoder = TrajEncoder(input_size,input_size,1,embedding,device)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        # self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.projection=nn.Sequential(nn.Linear(input_size,input_size),nn.ReLU(),nn.Linear(input_size,input_size))
        self.device=device
        self.temperature=0.05
        
    def forward(self, batch):
        # out=self.projection(self.traj_encoder.encode_sequence(batch)) # bd
        out=self.traj_encoder.encode_sequence(batch)
        return out

    def calculate_loss(self,batch1,batch2):
        out_view1=self.forward(batch1)
        out_view2=self.forward(batch2)
        # out_view1 = F.normalize(out_view1, dim=-1)
        # out_view2 = F.normalize(out_view2, dim=-1)

        # similarity_matrix = F.cosine_similarity(out_view1.unsqueeze(1), out_view2.unsqueeze(0), dim=-1)
        similarity_matrix = torch.cdist(out_view1,out_view2)
        # 分类loss，轨迹模型精度下降
        # labels = torch.arange(similarity_matrix.shape[0]).long().to(self.device)
        # loss_res = self.criterion(similarity_matrix, labels)
        # 二分类loss，只计算对角线的对错
        index=torch.eye(similarity_matrix.shape[0]).bool()
        preds=similarity_matrix[index]
        labels=torch.ones(similarity_matrix.shape[0]).to(self.device)
        loss_res=self.criterion(preds,labels)
        # # infoNCE
        # # 只计算正样本对（对角线）
        # logits = torch.matmul(out_view1, out_view2.T) / self.temperature
        # # 构造标签：每个样本的正样本为同一位置，即对角线
        # labels = torch.arange(logits.shape[0]).to(self.device)
        # # 使用交叉熵损失，最大化正样本对的相似度，同时降低负样本对的相似度
        # loss_res = F.cross_entropy(logits, labels)
        return loss_res

class SimilaritySearchModel(AbstractModel):
    def __init__(self, config):
        preprocess_detour(config)
        self._logger = getLogger()
        self._logger.warning('Evaluating Trajectory Similarity Search')

        self.dataset=STSDataset(config,filte=False)
        self.train_ori_dataloader,self.train_qry_dataloader,self.test_ori_dataloader,self.test_qry_dataloader = self.dataset.get_data()
        self.device=config.get('device')        
        self.epochs=config.get('task_epoch',50)
        self.learning_rate=1e-4#config.get('learning_rate',1e-4)
        self.weight_decay=config.get('weight_decay',1e-3)
    
    def run(self,model,**kwargs):
        self._logger.info("-------- STS START --------")
        # self.evaluation()
        self.train(model,**kwargs)
        return self.result

    def train(self,model,**kwargs):
        """
        返回评估结果
        """
        self.model = STSModel(embedding=model, device=self.device)
        self.model.to(self.device)
        optimizer = Adam(lr=self.learning_rate, params=self.model.parameters(), weight_decay=self.weight_decay)
        # # 先测试
        self.evaluation()
        best_loss=-1
        best_model=None
        best_epoch=0
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            for step,(batch1,batch2) in enumerate(zip(self.train_ori_dataloader,self.train_qry_dataloader)):
                batch1.update(kwargs)
                batch2.update(kwargs)
                loss=self.model.calculate_loss(batch1,batch2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            total_loss=total_loss/len(self.train_ori_dataloader)
            
            valid_loss=0.0
            with torch.no_grad():
                for step,(batch1,batch2) in enumerate(zip(self.test_ori_dataloader,self.test_qry_dataloader)):
                    batch1.update(kwargs)
                    batch2.update(kwargs)
                    loss=self.model.calculate_loss(batch1,batch2)
                    valid_loss += loss.item()
            valid_loss=valid_loss/len(self.test_ori_dataloader)

            if best_loss==-1 or valid_loss<best_loss:
                best_model=copy.copy(self.model)
                best_loss=valid_loss
                best_epoch=epoch
            self._logger.info("epoch {} complete! training loss {:.2f}, valid loss {:2f}, best_epoch {}, best_loss {:2f}".format(epoch, total_loss, valid_loss,best_epoch,best_loss))
        if best_model:
            self.model=best_model

        self.evaluation(**kwargs)
        return self.result

    def evaluation(self,**kwargs):
        ori_dataloader=self.dataset.test_ori_dataloader
        qry_dataloader=self.dataset.test_qry_dataloader
        num_queries=len(self.dataset.test_ori_dataloader)*self.dataset.test_ori_dataloader.batch_size

        self.model.eval()
        x = []

        for batch in ori_dataloader:
            batch.update(kwargs)
            # seq_rep = self.model.traj_encoder.encode_sequence(batch)
            seq_rep = self.model(batch)
            if isinstance(seq_rep, tuple):
                seq_rep = seq_rep[0]
            x.append(seq_rep.detach().cpu())
        x = torch.cat(x, dim=0).numpy()

        q = []
        for batch in qry_dataloader:
            batch.update(kwargs)
            # seq_rep = self.model.traj_encoder.encode_sequence(batch)
            seq_rep = self.model(batch)
            if isinstance(seq_rep, tuple):
                seq_rep = seq_rep[0]
            q.append(seq_rep.detach().cpu())
        q = torch.cat(q, dim=0).numpy()

        y=np.arange(x.shape[0])                         
        # index 类型
        # metric_type = faiss.METRIC
        index = faiss.IndexFlatL2(x.shape[1])
        index.add(x)
        D, I = index.search(q, 3000)
        self.result = {}
        top = [10]
        rank_sum=0
        hit=0
        no_hit=0
        for i, r in enumerate(I):
            if y[i] in r:
                rank_sum += np.where(r == y[i])[0][0]
                if y[i] in r[:3]:
                    hit += 1
            else:
                no_hit += 1
        self.result['Mean Rank'] = rank_sum / num_queries + 1.0
        self.result['No Hit'] = no_hit 
        self.result['HR@3'] =  hit / (num_queries - no_hit)
        self._logger.info('HR@3: {}'.format(self.result['HR@3']))
        self._logger.info('Mean Rank: {}, No Hit: {}'.format(self.result['Mean Rank'], self.result['No Hit']))

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        pass