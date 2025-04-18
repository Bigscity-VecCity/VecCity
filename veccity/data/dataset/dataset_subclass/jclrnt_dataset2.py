from veccity.data.dataset.lineseq_dataset import LineSeqDataset
import torch
from veccity.data.dataset.dataset_subclass.bert_base_dataset import padding_mask
from veccity.data.dataset.dataset_subclass.bertlm_dataset import BERTSubDataset
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm


class JCLRNTDataset(LineSeqDataset):
    def __init__(self,config):
        super().__init__(config)
        
        self.geo_to_ind=self.vocab.loc2index
        self.ind_to_geo=self.vocab.index2loc
        self.construct_road_edge()
        self.construct_od_graph()
        self.collate_fn=collate_nomask_seq

    def construct_road_edge(self):
        self.road_adj = np.zeros(shape=[self.num_nodes,self.num_nodes])
        #构建路网的邻接关系
        self.struct_edge_index=[]
        with open(self.adj_json_path,'r',encoding='utf-8') as fp:
            road_adj_data = json.load(fp)
        for road in range(self.vocab.specials_num,self.num_nodes):
            geo_uid=self.ind_to_geo[road]
            for neighbor in road_adj_data[str(geo_uid)]:
                if neighbor not in self.geo_to_ind:
                    continue
                n_id=self.geo_to_ind[neighbor]
                self.struct_edge_index.append((road,n_id))
                self.struct_edge_index.append((n_id,road))
        self.struct_edge_index=torch.Tensor(list(set(self.struct_edge_index))).long().transpose(1,0).to(self.device)
        
    def construct_od_graph(self):
        # jclrnt build contrastive graph
        self.edge_index=[]
        od_path=os.path.join(self.data_path,f"{self.dataset}.srel")
        self.od_matrix=np.zeros((self.num_nodes,self.num_nodes),dtype=np.float32)
        
        if not os.path.exists(od_path):
            traj_df = pd.read_csv(self.traj_path)
            traj_list = []
            for i in tqdm(range(len(traj_df))):
                path = traj_df.loc[i, 'path']
                path = path[1:len(path) - 1].split(',')
                path = [int(s) for s in path]
                origin_road=path[0]
                destination_road=path[-1]
                if origin_road in self.geo_to_ind and destination_road in self.geo_to_ind:
                    o_id=self.geo_to_ind[origin_road]
                    d_id=self.geo_to_ind[destination_road]
                    self.od_matrix[o_id][d_id]+=1
                    self.edge_index.append((o_id,d_id))
        else:
            od_data=pd.read_csv(od_path)
        
            for i in range(od_data.shape[0]):
                origin_road=od_data['orig_geo_id'][i]
                destination_road=od_data['dest_geo_id'][i]
                if origin_road in self.geo_to_ind and destination_road in self.geo_to_ind:
                    o_id=self.geo_to_ind[origin_road]
                    d_id=self.geo_to_ind[destination_road]
                    self.od_matrix[o_id][d_id]=od_data['flow'][i]
                    self.edge_index.append((o_id,d_id))

        self.edge_index = torch.Tensor(list(self.edge_index)).transpose(1,0).to(self.device)

        self.tran_matrix = self.od_matrix / (self.od_matrix.max(axis=1, keepdims=True, initial=0.) + 1e-9)
        row, col = np.diag_indices_from(self.tran_matrix)
        self.tran_matrix[row, col] = 0
        self.tran_matrix_b = (self.tran_matrix > self.edge_threshold)
        self.edge_index_aug = [(i // self.num_nodes, i % self.num_nodes) for i, n in
                               enumerate(self.tran_matrix_b.flatten()) if n]
        self.edge_index_aug = np.array(self.edge_index_aug, dtype=np.int32).transpose()
        self.edge_index_aug = torch.Tensor(self.edge_index_aug).int().to(self.device)
        self_loop = torch.tensor([[self.num_nodes - 1], [self.num_nodes - 1]]).to(self.device)
        self.edge_index = torch.cat((self.edge_index, self_loop), axis=1)
        self.edge_index_aug = torch.cat((self.edge_index_aug, self_loop), axis=1)

    
    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {'num_nodes':self.num_nodes,"struct_edge_index":self.struct_edge_index,"trans_edge_index":self.edge_index_aug,"vocab":self.vocab}
    
# collate for general
def collate_nomask_seq(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    return X.long(),  padding_masks

class JCLRNTABLDataset(LineSeqDataset):
    def __init__(self,config):
        super().__init__(config)
        
        self.geo_to_ind=self.vocab.loc2index
        self.ind_to_geo=self.vocab.index2loc
        self.construct_road_edge()
        self.construct_od_graph()
        self.collate_fn=collate_unsuperv_mask
        self.lanes_cls = self.road_geo_df['lanes'].max()+1
        self.clf_label = self.road_geo_df['lanes'].tolist()
        self.masking_ratio = self.config.get('masking_ratio', 0.2)
        self.masking_mode = self.config.get('masking_mode', 'together')
        self.distribution = self.config.get('distribution', 'random')
        self.avg_mask_len = self.config.get('avg_mask_len', 3)
    
    def _gen_dataset(self):
        train_dataset = BERTSubDataset(data_name=self.dataset, data_type='train',
                                       vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                       merge=self.merge, min_freq=self.min_freq,
                                       max_train_size=self.max_train_size,
                                       masking_ratio=self.masking_ratio,
                                       masking_mode=self.masking_mode, distribution=self.distribution,
                                       avg_mask_len=self.avg_mask_len)
        eval_dataset = BERTSubDataset(data_name=self.dataset, data_type='val',
                                      vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                      merge=self.merge, min_freq=self.min_freq,
                                      max_train_size=None,
                                      masking_ratio=self.masking_ratio,
                                      masking_mode=self.masking_mode, distribution=self.distribution,
                                      avg_mask_len=self.avg_mask_len)
        test_dataset = BERTSubDataset(data_name=self.dataset, data_type='test',
                                      vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                      merge=self.merge, min_freq=self.min_freq,
                                      max_train_size=None,
                                      masking_ratio=self.masking_ratio,
                                      masking_mode=self.masking_mode, distribution=self.distribution,
                                      avg_mask_len=self.avg_mask_len)
        return train_dataset, eval_dataset, test_dataset

    def construct_road_edge(self):
        self.road_adj = np.zeros(shape=[self.num_nodes,self.num_nodes])
        #构建路网的邻接关系
        self.struct_edge_index=[]
        with open(self.adj_json_path,'r',encoding='utf-8') as fp:
            road_adj_data = json.load(fp)
        for road in range(self.vocab.specials_num,self.num_nodes):
            geo_uid=self.ind_to_geo[road]
            for neighbor in road_adj_data[str(geo_uid)]:
                if neighbor not in self.geo_to_ind:
                    continue
                n_id=self.geo_to_ind[neighbor]
                self.struct_edge_index.append((road,n_id))
                self.struct_edge_index.append((n_id,road))
        self.struct_edge_index=torch.Tensor(list(set(self.struct_edge_index))).long().transpose(1,0).to(self.device)
        
    def construct_od_graph(self):
        # jclrnt build contrastive graph
        self.edge_index=[]
        od_path=os.path.join(self.data_path,f"{self.dataset}.srel")
        self.od_matrix=np.zeros((self.num_nodes,self.num_nodes),dtype=np.float32)
        
        if not os.path.exists(od_path):
            traj_df = pd.read_csv(self.traj_path)
            traj_list = []
            for i in tqdm(range(len(traj_df))):
                path = traj_df.loc[i, 'path']
                path = path[1:len(path) - 1].split(',')
                path = [int(s) for s in path]
                origin_road=path[0]
                destination_road=path[-1]
                if origin_road in self.geo_to_ind and destination_road in self.geo_to_ind:
                    o_id=self.geo_to_ind[origin_road]
                    d_id=self.geo_to_ind[destination_road]
                    self.od_matrix[o_id][d_id]+=1
                    self.edge_index.append((o_id,d_id))
        else:
            od_data=pd.read_csv(od_path)
        
            for i in range(od_data.shape[0]):
                origin_road=od_data['orig_geo_id'][i]
                destination_road=od_data['dest_geo_id'][i]
                if origin_road in self.geo_to_ind and destination_road in self.geo_to_ind:
                    o_id=self.geo_to_ind[origin_road]
                    d_id=self.geo_to_ind[destination_road]
                    self.od_matrix[o_id][d_id]=od_data['flow'][i]
                    self.edge_index.append((o_id,d_id))

        self.edge_index = torch.Tensor(list(self.edge_index)).transpose(1,0).to(self.device)

        self.tran_matrix = self.od_matrix / (self.od_matrix.max(axis=1, keepdims=True, initial=0.) + 1e-9)
        row, col = np.diag_indices_from(self.tran_matrix)
        self.tran_matrix[row, col] = 0
        self.tran_matrix_b = (self.tran_matrix > self.edge_threshold)
        self.edge_index_aug = [(i // self.num_nodes, i % self.num_nodes) for i, n in
                               enumerate(self.tran_matrix_b.flatten()) if n]
        self.edge_index_aug = np.array(self.edge_index_aug, dtype=np.int32).transpose()
        self.edge_index_aug = torch.Tensor(self.edge_index_aug).int().to(self.device)
        self_loop = torch.tensor([[self.num_nodes - 1], [self.num_nodes - 1]]).to(self.device)
        self.edge_index = torch.cat((self.edge_index, self_loop), axis=1)
        self.edge_index_aug = torch.cat((self.edge_index_aug, self_loop), axis=1)

    
    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {'num_nodes':self.num_nodes,"struct_edge_index":self.struct_edge_index,"trans_edge_index":self.edge_index_aug,"vocab":self.vocab,'ablation_num_class':self.lanes_cls,'clf_label':self.clf_label}
    

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
    seq=X.clone()
    X[..., 0:1].masked_fill_(target_masks[..., 0:1] == 1, vocab.mask_index)  # loc -> mask_index
    X[..., 1:].masked_fill_(target_masks[..., 1:] == 1, vocab.pad_index)  # others -> pad_index
    batch={}
    batch['seq']=seq.long()
    batch['masked_input']=X.long()
    batch['targets']=targets[...,0].long()
    batch['targets_mask']=target_masks[...,0]
    batch['padding_masks']=padding_masks
    batch['batch_temporal_mat']=batch_temporal_mat.long()
    return batch