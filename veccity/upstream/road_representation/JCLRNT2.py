from logging import getLogger
from tqdm import tqdm
from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import time
import dgl
from dgl.nn import GATConv,GraphConv
import os

class JCLRNT(AbstractReprLearningModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get('device')
        self.dataloader = data_feature.get('dataloader')
        self.num_nodes = data_feature.get("num_nodes")
        self._logger = getLogger()
        self.output_dim = config.get('output_dim', 128)
        self.iter = config.get('max_epoch', 5)
        self.model = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.dataset = config.get('dataset', '')
        e1 = data_feature.get('struct_edge_index')
        e2 = data_feature.get('trans_edge_index')
        self.ablation=config.get('abl','gnn')

        self.edge_index1 = dgl.graph((e1[0], e1[1]))
        self.edge_index2 = dgl.graph((e2[0], e2[1]))
        self.edge_index1 = dgl.add_self_loop(self.edge_index1)
        self.edge_index2 = dgl.add_self_loop(self.edge_index2)

        self.model_cache_file = './veccity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.traj_train_embedding_file = './veccity/cache/{}/evaluate_cache/traj_train_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.traj_test_embedding_file = './veccity/cache/{}/evaluate_cache/traj_test_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.road_embedding_path = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.hidden_size = config.get('hidden_size',128)
        self.drop_rate = config.get('drop_rate', 0.2)
        self.drop_edge_rate = config.get('drop_edge_rate', 0.2)
        self.drop_road_rate = config.get('drop_road_rate', 0.2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-6)
        self.measure = config.get('loss_measure', "jsd")
        self.is_weighted = config.get('weighted_loss', False)
        self.mode = config.get('mode', 's')
        self.l_st = config.get('lambda_st', 0.8)
        # self.traj_train = torch.from_numpy(data_feature.get('traj_arr_train'))
        # self.traj_test = torch.from_numpy(data_feature.get('traj_arr_test'))
        self.l_ss = self.l_tt = 0.5 * (1 - self.l_st)
        self.activation = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}[config.get("activation", "relu")]
        self.num_epochs = config.get('num_epochs', 5)
        self.vocab=data_feature.get("vocab")

        if 'gnn' == self.ablation:
            self.graph_encoder1 = GraphEncoder(self.output_dim, self.hidden_size, GraphConv, 2, self.activation)
            self.graph_encoder2 = GraphEncoder(self.output_dim, self.hidden_size, GraphConv, 2, self.activation)
        else:
            self.graph_encoder1 = GraphEncoder(self.output_dim, self.hidden_size, GATConv, 2, self.activation)
            self.graph_encoder2 = GraphEncoder(self.output_dim, self.hidden_size, GATConv, 2, self.activation)
        
        if self.ablation!='lstm':
            self.seq_encoder = TransformerModel(self.hidden_size, 4, self.hidden_size, 2, self.drop_rate)
        else:
            self.seq_encoder = nn.LSTM(self.hidden_size,self.hidden_size,2,dropout=self.drop_rate,batch_first=True)
        
        self.num_class=data_feature.get('ablation_num_class',2)
        self.clf_labels=data_feature.get('clf_label',None)
        self.clf_head=nn.Sequential(nn.Linear(2*self.hidden_size,self.hidden_size),nn.ReLU(),nn.Linear(self.hidden_size,self.num_class))
        self.clf_loss_fn=nn.CrossEntropyLoss()

        self.mask_l=MaskedLanguageModel(self.hidden_size, self.num_nodes)
        self.mask_loss_fn = torch.nn.NLLLoss(ignore_index=0, reduction='none')

        self.model = MultiViewModel(self.num_nodes, self.output_dim, self.hidden_size, self.edge_index1, self.edge_index2,
                               self.graph_encoder1, self.graph_encoder2, self.seq_encoder,self.ablation,self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    
    def recon_loss(self,batch):
        # 首先要随机选择mask
        # 其次区分出input和target
        mask_input=batch['masked_input'].to(self.device) # b s
        padding_masks=batch['padding_masks'].to(self.device)# b s
        targets=batch['targets'].to(self.device) # b s, 非target部分为0 
        targets_mask=batch['targets_mask'].to(self.device) # b,s 非target部分为0

        seq_rep,_,_=self.model.encode_sequence2(mask_input,padding_masks)
        preds=self.mask_l(seq_rep) # b s v

        batch_loss_list = self.mask_loss_fn(preds.transpose(1, 2), targets)
        batch_loss = torch.sum(batch_loss_list)
        num_active = targets_mask.sum()
        mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization
        return mean_loss
    

    def infer_loss(self,batch):
        # 需要有一种label，可以选择和hyper road一样的label
        # 节点分类
        
        labels=torch.tensor(self.clf_labels).long().to(self.device)
        node_embedding=self.model.encode_graph()
        node_embedding=torch.hstack(node_embedding[1:])
        preds=self.clf_head(node_embedding[4:])#4: 去掉special token
        clf_loss=self.clf_loss_fn(preds,labels)
        return clf_loss

    def calculate_loss(self,batch,w_batch):
        seq,padding_mask=batch['seq'],batch['padding_masks']
        seq = seq.to(self.device)
        padding_mask=padding_mask.to(self.device)
        node_rep1, node_rep2, seq_rep1, seq_rep2 = self.model(seq,padding_mask)
        # loss_ss = node_node_loss(node_rep1, node_rep2, self.measure,self.device)
        loss_ss = 0
        loss_tt = seq_seq_loss(seq_rep1, seq_rep2, self.measure,self.device)# 负数
        if self.is_weighted:
            loss_st1 = weighted_ns_loss(node_rep1, seq_rep2, w_batch, self.measure)
            loss_st2 = weighted_ns_loss(node_rep2, seq_rep1, w_batch, self.measure)
        else:
            loss_st1 = node_seq_loss(node_rep1, seq_rep2, batch['seq'], self.measure,self.device)# 负数
            loss_st2 = node_seq_loss(node_rep2, seq_rep1, batch['seq'], self.measure,self.device)
        loss_st = (loss_st1 + loss_st2) / 2
        loss = self.l_ss * loss_ss + self.l_tt * loss_tt + self.l_st * loss_st
        if 'recon' == self.ablation:
            
            loss+=self.recon_loss(batch)

        if 'infer' == self.ablation:
            loss+=self.infer_loss(batch)

        if 'recon&infer' == self.ablation:
            loss+=self.recon_loss(batch) + self.infer_loss(batch)

        return loss


    def get_static_embedding(self):
        node_embedding = self.model.encode_graph()[0].cpu().detach().numpy().squeeze()
        return node_embedding
    
    def encode_sequence(self,batch):
        seq=batch['seq'].to(self.device)
        padding_masks=batch['padding_masks'].to(self.device)
        out,_,_=self.model.encode_sequence(seq,padding_masks)
        return out
        
    def save_traj_embedding(self,traj_test,traj_embedding_file):
        result_list = []
        traj_num = traj_test.shape[0]
        self._logger.info('traj_num=' + str(traj_num))
        start_index = 0
        while start_index < traj_num:
            end_index = min(traj_num, start_index + 1280)
            batch_embedding = self.model.encode_sequence(traj_test[start_index:end_index].to(self.device))[
                0].cpu().detach().numpy()
            result_list.append(batch_embedding)
            start_index = end_index
        traj_embedding = np.concatenate(result_list, axis=0)
        self._logger.info('词向量维度：(' + str(len(traj_embedding)) + ',' + str(len(traj_embedding[0])) + ')')
        np.save(traj_embedding_file, traj_embedding)
        self._logger.info('保存至  '+traj_embedding_file)


def jsd(z1, z2, pos_mask):

    neg_mask = 1 - pos_mask
    sim_mat = torch.mm(z1, z2.t())
    E_pos = math.log(2.) - F.softplus(-sim_mat)
    E_neg = F.softplus(-sim_mat) + sim_mat - math.log(2.)
    return (E_neg * neg_mask).sum() / neg_mask.sum() - (E_pos * pos_mask).sum() / pos_mask.sum()

def nce(z1, z2, pos_mask):
    sim_mat = torch.mm(z1, z2.t())
    return nn.BCEWithLogitsLoss(reduction='none')(sim_mat, pos_mask).sum(1).mean()

def ntx(z1, z2, pos_mask, tau=0.5, normalize=False):
    if normalize:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    sim_mat = torch.mm(z1, z2.t())
    sim_mat = torch.exp(sim_mat / tau)
    return -torch.log((sim_mat * pos_mask).sum(1) / sim_mat.sum(1) / pos_mask.sum(1)).mean()


def node_node_loss(node_rep1, node_rep2, measure,device):
    num_nodes = node_rep1.shape[0]

    pos_mask = torch.eye(num_nodes).to(device)

    if measure == 'jsd':
        return jsd(node_rep1, node_rep2, pos_mask)
    elif measure == 'nce':
        return nce(node_rep1, node_rep2, pos_mask)
    elif measure == 'ntx':
        return ntx(node_rep1, node_rep2, pos_mask)


def seq_seq_loss(seq_rep1, seq_rep2, measure,device):
    batch_size = seq_rep1.shape[0]

    pos_mask = torch.eye(batch_size).to(device)
    if measure == 'jsd':
        return jsd(seq_rep1, seq_rep2, pos_mask)
    elif measure == 'nce':
        return nce(seq_rep1, seq_rep2, pos_mask)
    elif measure == 'ntx':
        return ntx(seq_rep1, seq_rep2, pos_mask)


def node_seq_loss(node_rep, seq_rep, sequences, measure,device):
    batch_size = seq_rep.shape[0]
    num_nodes = node_rep.shape[0]
    sequences=sequences[:,:,0]

    pos_mask = torch.zeros((batch_size, num_nodes + 1)).to(device)

    for row_idx, row in enumerate(sequences):
        row = row.type(torch.long)
        pos_mask[row_idx][row] = 1.
    pos_mask = pos_mask[:, :-1]

    if measure == 'jsd':
        return jsd(seq_rep, node_rep, pos_mask)
    elif measure == 'nce':
        return nce(seq_rep, node_rep, pos_mask)
    elif measure == 'ntx':
        return ntx(seq_rep, node_rep, pos_mask)


def weighted_ns_loss(node_rep, seq_rep, weights, measure):
    if measure == 'jsd':
        return jsd(seq_rep, node_rep, weights)
    elif measure == 'nce':
        return nce(seq_rep, node_rep, weights)
    elif measure == 'ntx':
        return ntx(seq_rep, node_rep, weights)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output


class GraphEncoder(nn.Module):
    def __init__(self, input_size, output_size, encoder_layer, num_layers, activation,heads=1):
        super(GraphEncoder, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        if encoder_layer == GraphConv:
            self.layers = [encoder_layer(input_size, output_size)]
            for _ in range(1, num_layers):
                self.layers.append(encoder_layer(output_size, output_size))
            self.layers = nn.ModuleList(self.layers)
        else:
            self.layers = [encoder_layer(input_size, output_size,heads)]
            for _ in range(1, num_layers):
                self.layers.append(encoder_layer(output_size, output_size,heads))
            self.layers = nn.ModuleList(self.layers)

    def forward(self, edge_index, x):
        for i in range(self.num_layers):
            x = self.activation(self.layers[i](edge_index, x))
        return x


class MultiViewModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, edge_index1, edge_index2,
                 graph_encoder1, graph_encoder2, seq_encoder,ablation,device):
        super(MultiViewModel, self).__init__()

        self.vocab_size = vocab_size
        self.node_embedding = nn.Embedding(vocab_size, embed_size)
        self.padding = torch.zeros(1, hidden_size, requires_grad=False).to(device)
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        self.graph_encoder1 = graph_encoder1
        self.graph_encoder2 = graph_encoder2
        self.seq_encoder = seq_encoder
        self.ablation=ablation


    def encode_graph(self):
        node_emb = self.node_embedding.weight.detach()

        node_enc1 = self.graph_encoder1(self.edge_index1,node_emb)
        node_enc2 = self.graph_encoder2(self.edge_index2,node_emb)
        return node_enc1 + node_enc2, node_enc1.view(self.vocab_size, -1), node_enc2.view(self.vocab_size, -1)

    def encode_sequence(self, sequences,padding_mask=None):

        _, node_enc1, node_enc2 = self.encode_graph()

        # feat:loc_list, tim_list, minutes, weeks, usr
        if len(sequences.shape)==3:
            sequences=sequences[:,:,0]
        batch_size, max_seq_len = sequences.size()
        
        if padding_mask != None :
            padding_mask = ~padding_mask
            pool_mask=(1 - padding_mask.int()).transpose(0, 1).unsqueeze(-1)
        else:
            padding_mask = (sequences == self.vocab_size)
            pool_mask = (1 - padding_mask.int()).transpose(0, 1).unsqueeze(-1)

        lookup_table1 = torch.cat([node_enc1, self.padding], 0)
        
        seq_emb1 = torch.index_select(
            lookup_table1, 0, sequences.view(-1)).view(batch_size, max_seq_len, -1).transpose(0, 1)
        
        if self.ablation == 'lstm':
            seq_enc1=self.seq_encoder(seq_emb1)[0]
        else:
            seq_enc1 = self.seq_encoder(seq_emb1, None, padding_mask)

        seq_pooled1 = (seq_enc1 * pool_mask).sum(0) / pool_mask.sum(0)

        lookup_table2 = torch.cat([node_enc2, self.padding], 0)
        seq_emb2 = torch.index_select(
            lookup_table2, 0, sequences.view(-1)).view(batch_size, max_seq_len, -1).transpose(0, 1)
        if self.ablation=='lstm':
            seq_enc2=self.seq_encoder(seq_emb2)[0]
        else:
            seq_enc2 = self.seq_encoder(seq_emb2, None, padding_mask)
        seq_pooled2 = (seq_enc2 * pool_mask).sum(0) / pool_mask.sum(0)
        return seq_pooled1 + seq_pooled2, seq_pooled1, seq_pooled2
    
    def encode_sequence2(self, sequences,padding_mask=None):

        _, node_enc1, node_enc2 = self.encode_graph()

        # feat:loc_list, tim_list, minutes, weeks, usr
        if len(sequences.shape)==3:
            sequences=sequences[:,:,0]
        batch_size, max_seq_len = sequences.size()
        
        if padding_mask != None :
            padding_mask = ~padding_mask
            pool_mask=(1 - padding_mask.int()).transpose(0, 1).unsqueeze(-1)
        else:
            padding_mask = (sequences == self.vocab_size)
            pool_mask = (1 - padding_mask.int()).transpose(0, 1).unsqueeze(-1)

        lookup_table1 = torch.cat([node_enc1, self.padding], 0)
        
        seq_emb1 = torch.index_select(
            lookup_table1, 0, sequences.view(-1)).view(batch_size, max_seq_len, -1).transpose(0, 1)
        
        if self.ablation == 'lstm':
            seq_enc1=self.seq_encoder(seq_emb1)[0]
        else:
            seq_enc1 = self.seq_encoder(seq_emb1, None, padding_mask)

        # seq_pooled1 = (seq_enc1 * pool_mask).sum(0) / pool_mask.sum(0)

        lookup_table2 = torch.cat([node_enc2, self.padding], 0)
        seq_emb2 = torch.index_select(
            lookup_table2, 0, sequences.view(-1)).view(batch_size, max_seq_len, -1).transpose(0, 1)
        if self.ablation=='lstm':
            seq_enc2=self.seq_encoder(seq_emb2)[0]
        else:
            seq_enc2 = self.seq_encoder(seq_emb2, None, padding_mask)
        # seq_pooled2 = (seq_enc2 * pool_mask).sum(0) / pool_mask.sum(0)
        return seq_enc2 + seq_enc1, seq_enc1, seq_enc2

    def forward(self, sequences,padding_mask=None):
        _, node_enc1, node_enc2 = self.encode_graph()
        _, seq_pooled1, seq_pooled2 = self.encode_sequence(sequences,padding_mask)
        return node_enc1, node_enc2, seq_pooled1, seq_pooled2


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """

        Args:
            hidden: output size of BERT model
            vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))