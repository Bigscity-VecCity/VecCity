import math
import os
import ot
import pdb
import pickle as pkl
import random
import time
from logging import getLogger

import numpy as np
import pytorch_warmup as warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.nn.functional import dropout, linear, softmax
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import reset, uniform
from tqdm import tqdm, trange
from veccity.data.dataset.dataset_subclass.redcl_dataset import (
    FreezePatternForwardDataset, FreezePatternPretrainDataset,
    UnsupervisedPatternDataset)
from veccity.upstream.abstract_replearning_model import \
    AbstractReprLearningModel
from veccity.utils import need_train


class ReDCL(AbstractReprLearningModel):
    def __init__(self,config,data_feature):
        super().__init__(config,data_feature)
        self._logger = getLogger()
        self.device = config.get('device')
        self.dim = config.get('output_dim', 64)
        self.dataset = config.get('dataset', '')
        self.epochs = config.get('max_epoch', 1)
        self.model = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.weight_decay = config.get('weight_decay', 0.0001)

        self.city = self.dataset
        self.no_random = config.get('no_random', False)
        self.fixed = config.get('fixed', False)
        self.d_feedforward = config.get('d_feedforward', 1024)
        self.building_head = config.get('building_head', 8)
        self.building_layers = config.get('building_layers', 2)
        self.building_dropout = config.get('building_dropout', 0.2)
        self.building_activation = config.get('building_activation', 'relu')
        self.bottleneck_head = config.get('bottleneck_head', 8)
        self.bottleneck_layers = config.get('bottleneck_layers', 2)
        self.bottleneck_dropout = config.get('bottleneck_dropout', 0.2)
        self.bottleneck_activation = config.get('bottleneck_activation', 'relu')
        self.gamma = config.get('gamma', 0.999)
        self.save_name = config.get('save_name', 'pattern_embedding')

        self.city_data = data_feature.get('city_data')
        
        self.npy_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.dim)
        
    def run(self, train_dataloader=None, eval_dataloader=None):
        if not need_train(self.config):
            return
        start_time = time.time()
        device = self.device
        pattern_encoder = PatternEncoder(d_building=self.city_data.building_feature_dim,
                                        d_poi=self.city_data.poi_feature_dim,
                                        d_hidden=self.dim,
                                        d_feedforward=self.d_feedforward,
                                        building_head=self.building_head,
                                        building_layers=self.building_layers,
                                        building_dropout=self.building_dropout,
                                        building_distance_penalty=1,
                                        building_activation=self.building_activation,
                                        bottleneck_head=self.bottleneck_head,
                                        bottleneck_layers=self.bottleneck_layers,
                                        bottleneck_dropout=self.bottleneck_dropout,
                                        bottleneck_activation=self.bottleneck_activation).to(device)
        # Encode building pattern
        pattern_optimizer = torch.optim.Adam(pattern_encoder.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        pattern_scheduler = torch.optim.lr_scheduler.StepLR(pattern_optimizer, step_size=1, gamma=self.gamma)
        pattern_trainer = PatternTrainer(self.city_data, pattern_encoder, pattern_optimizer, pattern_scheduler, self.config)
        pattern_trainer.train_pattern_contrastive(epochs=self.epochs, save_name=self.save_name)
        region_aggregator = RegionEncoder(d_hidden=self.dim, d_head=8).to(device)
        region_optimizer = torch.optim.Adam(region_aggregator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        region_scheduler = torch.optim.lr_scheduler.StepLR(region_optimizer, step_size=1, gamma=self.gamma)
        region_trainer = RegionTrainer(self.city_data, pattern_encoder, pattern_optimizer, pattern_scheduler, region_aggregator,
                                    region_optimizer, region_scheduler, self.config)
        # embeddings = pattern_trainer.get_embeddings()
        # Alternatively, you can load the trained pattern embedding
        exp_id = self.config.get('exp_id')
        embeddings = np.load(f'VecCity/veccity/cache/{exp_id}/evaluate_cache/{self.save_name}_{self.epochs}.npy')
        region_trainer.train_region_triplet_freeze(epochs=self.epochs, embeddings=embeddings, adaptive=not self.fixed, save_name=self.npy_cache_file,
                                                window_sizes=[1000, 2000, 3000])
        t1 = time.time()-start_time
        outs = np.load(self.npy_cache_file)
        self._logger.info('cost time is '+str(t1/self.epochs))
        total_num = sum([param.nelement() for param in pattern_encoder.parameters()])
        total_num += sum([param.nelement() for param in region_aggregator.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))
        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：(' + str(len(outs)) + ',' + str(len(outs[0])) + ')')        

class BiasedEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(BiasedEncoderLayer, self).__init__()
        self.self_attn = BiasedMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(BiasedEncoderLayer, self).__setstate__(state)

    def forward(self, src, bias, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src = self.norm1(src)
        # become nan
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, bias=bias)[0]
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class DistanceBiasedTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, activation,
                 distance_penalty):
        super(DistanceBiasedTransformer, self).__init__()
        self.num_layers = num_layers
        self.distance_layers = nn.ModuleList([
            BiasedEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
            for _ in range(num_layers)])
        self.distance_penalty = nn.Parameter(torch.normal(0, distance_penalty, [nhead], requires_grad=True))

    def forward(self, x, src_key_padding_mask, distance):
        # Building_feature: [seq_len, batch_size, d_model]
        # Calculate distance bias. Each head will have a distance bias
        distance_bias = self.distance_penalty.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * -distance.unsqueeze(1)
        hidden = x
        for layer in self.distance_layers:
            hidden = layer(hidden, distance_bias, src_key_padding_mask=src_key_padding_mask)
        return hidden


class PatternEncoder(nn.Module):
    def __init__(self, d_building, d_poi, d_hidden, d_feedforward,
                 building_head, building_layers,
                 building_dropout, building_activation, building_distance_penalty,
                 bottleneck_head, bottleneck_layers, bottleneck_dropout, bottleneck_activation):
        super(PatternEncoder, self).__init__()
        self.building_projector = nn.Linear(d_building, d_hidden)
        self.poi_projector = nn.Linear(d_poi, d_hidden)
        self.building_encoder = DistanceBiasedTransformer(d_model=d_hidden,
                                                          nhead=building_head,
                                                          num_layers=building_layers,
                                                          dim_feedforward=d_feedforward, dropout=building_dropout,
                                                          activation=building_activation,
                                                          distance_penalty=building_distance_penalty)
        self.bottleneck = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_hidden, nhead=bottleneck_head, dim_feedforward=d_feedforward,
                                       dropout=bottleneck_dropout, activation=bottleneck_activation), bottleneck_layers)

    def forward(self, building_feature, building_mask, xy, poi_feature, poi_mask):
        return self.get_all(building_feature, building_mask, xy, poi_feature, poi_mask).mean(dim=0)

    def get_embedding(self, building_feature, building_mask, xy, poi_feature, poi_mask):
        # add up the 0-100 dimension of building density to test the performance
        return self.get_all(building_feature, building_mask, xy, poi_feature, poi_mask).mean(dim=0)

    def get_all(self, building_feature, building_mask, xy, poi_feature, poi_mask):
        building_encoding = self.building_projector(building_feature)
        batch_size = building_encoding.shape[1]
        building_loc = xy.transpose(0, 1)
        building_distance = torch.norm(building_loc.unsqueeze(2) - building_loc.unsqueeze(1), dim=3)
        # # Test the new formula
        building_distance[building_mask.unsqueeze(1) | building_mask.unsqueeze(2)] = 0
        # # get maximum_distance per pattern
        max_distance = torch.max(building_distance.view(batch_size, -1), dim=1)[0]
        normalized_distance = torch.log(
            (torch.pow(max_distance.unsqueeze(1).unsqueeze(1), 1.5) + 1) / (torch.pow(building_distance, 1.5) + 1))
        # here become nan TODO
        building_encoding = self.building_encoder(building_encoding, building_mask, normalized_distance)
        encoding_list = [building_encoding]
        mask_list = [building_mask]
        if poi_feature is not None:
            poi_encoding = self.poi_projector(poi_feature)
            encoding_list.append(poi_encoding)
            mask_list.append(poi_mask)
        encoding = torch.cat(encoding_list, dim=0)
        encoding_mask = torch.cat(mask_list, dim=1)
        # bottleneck
        bottleneck_encoding = self.bottleneck(encoding, src_key_padding_mask=encoding_mask)
        # concatenate bottleneck and bottleneck embedding
        return bottleneck_encoding


class TransformerPatternEncoder(nn.Module):
    def __init__(self, d_building, d_hidden, d_feedforward,
                 building_head, building_layers,
                 building_dropout, building_activation):
        super(TransformerPatternEncoder, self).__init__()
        self.building_projector = nn.Linear(d_building, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self.building_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_hidden, nhead=building_head, dim_feedforward=d_feedforward,
                                       dropout=building_dropout, activation=building_activation), building_layers,
            norm=self.norm)

    def forward(self, building_feature, building_mask, xy):
        building_encoding = self.building_projector(building_feature)
        return self.building_encoder(building_encoding, src_key_padding_mask=building_mask).mean(dim=0)


class RegionEncoder(nn.Module):
    """
    In RegionEncoder, we use a simple Transformer encoder to aggregate all patterns in a region
    """

    def __init__(self, d_hidden, d_head):
        super(RegionEncoder, self).__init__()
        self.attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_hidden, nhead=d_head), 1)

    def forward(self, x, sigmoid=True):
        # x: [seq_len, batch_size, d_hidden]
        x = x.unsqueeze(1)
        # x = torch.cat([self.region_embedding.repeat(1, x.shape[1], 1), x], dim=0)
        x = self.attention(x).squeeze(1)
        return x

    def get_embedding(self, x):
        return self.forward(x, sigmoid=False).mean(dim=0)
    
'''
    Biased Self-Attention Layer
    Modified from pytorch official multihead attention to apply bias
'''


class BiasedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super(BiasedMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self._qkv_same_embed_dim = True

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(BiasedMultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, bias, key_padding_mask=None, attn_mask=None):
        return self.biased_attention(
            query, key, value, bias, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, attn_mask=attn_mask)

    @staticmethod
    def biased_attention(query,  # type: Tensor
                         key,  # type: Tensor
                         value,  # type: Tensor
                         bias,  # type: Tensor
                         embed_dim_to_check,  # type: int
                         num_heads,  # type: int
                         in_proj_weight,  # type: Tensor
                         in_proj_bias,  # type: Tensor
                         dropout_p,  # type: float
                         out_proj_weight,  # type: Tensor
                         out_proj_bias,  # type: Tensor
                         training=True,  # type: bool
                         key_padding_mask=None,  # type: Optional[Tensor]
                         attn_mask=None,  # type: Optional[Tensor]
                         q_proj_weight=None,  # type: Optional[Tensor]
                         k_proj_weight=None,  # type: Optional[Tensor]
                         v_proj_weight=None,  # type: Optional[Tensor]
                         static_k=None,  # type: Optional[Tensor]
                         static_v=None  # type: Optional[Tensor]
                         ):
        
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        assert key.size() == value.size()

        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
            

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
        q = q * scaling

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if static_k is not None:
            assert static_k.size(0) == bsz * num_heads
            assert static_k.size(2) == head_dim
            k = static_k

        if static_v is not None:
            assert static_v.size(0) == bsz * num_heads
            assert static_v.size(2) == head_dim
            v = static_v

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        # apply bias term before masking
        if bias is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights + bias
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                -100, # 如果用 -inf 会出现 nan TODO
                # float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights = softmax(
            attn_output_weights, dim=-1)

        # if key_padding_mask is not None:
        #     attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        #     attn_output_weights = attn_output_weights.masked_fill(
        #         key_padding_mask.unsqueeze(1).unsqueeze(2),
        #         float(0),
        #     )
        #     attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    
'''
    Trainer for RegionDCL
    Though the dual contrastive learning can be trained together,
    We found it unnecessary to do so as it is time-wasting and the performance is not significantly improved.
    Therefore, we remove the co-training and let them train together.
'''


class PatternTrainer(object):
    def __init__(self, city_data, model, optimizer, scheduler, config):
        self.city_data = city_data
        self.config = config
        self.device = config.get('device', 'cpu')
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = 10
        self._logger = getLogger()

    def forward(self, building_feature, building_mask, xy, poi_feature, poi_mask):
        torch.cuda.empty_cache()
        building_feature = torch.from_numpy(building_feature).to(self.device)
        building_mask = torch.from_numpy(building_mask).to(self.device)
        xy = torch.from_numpy(xy).to(self.device)
        if poi_feature is not None:
            poi_feature = torch.from_numpy(poi_feature).to(self.device)
            poi_mask = torch.from_numpy(poi_mask).to(self.device)
        return self.model(building_feature, building_mask, xy, poi_feature, poi_mask)

    def infonce_loss(self, y_pred, lamda=0.05):
        N = y_pred.shape[0]
        idxs = torch.arange(0, N, device=self.device)
        y_true = idxs + 1 - idxs % 2 * 2
        # similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
        y_pred = F.softmax(y_pred, dim=1)
        off_diag = np.ones((N, N))
        indices = np.where(off_diag)
        rec_idx = torch.LongTensor(indices[0]).to(self.device)
        send_idx = torch.LongTensor(indices[1]).to(self.device)
        senders = y_pred[send_idx]
        receivers = y_pred[rec_idx]
        
        similarities = 1 - F.kl_div(senders.log(), receivers, reduction='none').sum(dim=1).view(N, N)
        similarities = similarities - torch.eye(y_pred.shape[0], device=self.device) * 1e12
        similarities = similarities / lamda
        loss = F.cross_entropy(similarities, y_true)
        return torch.mean(loss)

    def train_pattern_contrastive(self, epochs, save_name):
        dataset = UnsupervisedPatternDataset(self.city_data)
        exp_id = self.config.get('exp_id')
        save_path = f'veccity/cache/{exp_id}/evaluate_cache'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
                                                   collate_fn=UnsupervisedPatternDataset.collate_fn_dropout)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False,
                                                  collate_fn=UnsupervisedPatternDataset.collate_fn)
        criterion = self.infonce_loss
        for epoch in range(1, epochs+1):
            tqdm_batch = tqdm(train_loader, desc='Epoch {}'.format(epoch))
            losses = []
            for data in tqdm_batch:
                self.optimizer.zero_grad()
                building_feature, building_mask, xy, poi_feature, poi_mask = data
                pred = self.forward(building_feature, building_mask, xy, poi_feature, poi_mask)
                loss = criterion(pred)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                tqdm_batch.set_postfix(loss=loss.item())
                losses.append(loss.item())
            self._logger.info('Epoch {}: InfoNCE Loss {}'.format(epoch, np.mean(losses)))
            self.scheduler.step()
            self.save_embedding(os.path.join(save_path, save_name + '_' + str(epoch) + '.npy'), test_loader)

    def save_embedding(self, output, data_loader):
        all_embeddings = self.get_embedding(data_loader)
        np.save(output, all_embeddings)

    def get_embedding(self, data_loader):
        embedding_list = []
        self.model.eval()
        with torch.no_grad():
            for data in data_loader:
                building_feature, building_mask, xy, poi_feature, poi_mask = data
                embedding = self.forward(building_feature, building_mask, xy, poi_feature, poi_mask)
                embedding_list.append(embedding.detach().cpu().numpy())
        all_embeddings = np.concatenate(embedding_list, axis=0)
        return all_embeddings

    def get_embeddings(self):
        dataset = UnsupervisedPatternDataset(self.city_data)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,
                                                  collate_fn=UnsupervisedPatternDataset.collate_fn)
        return self.get_embedding(test_loader)


class RegionTrainer(object):
    def __init__(self, city_data, pattern_model, pattern_optimizer, pattern_scheduler, region_model, region_optimizer,
                 region_scheduler, config, early_stopping=10):
        self.city_data = city_data

        self.pattern_model = pattern_model
        self.pattern_optimizer = pattern_optimizer
        self.pattern_scheduler = pattern_scheduler

        self.region_model = region_model
        self.region_optimizer = region_optimizer
        self.region_scheduler = region_scheduler

        self.early_stopping = early_stopping
        self.device = config.get('device')
        self.config = config
        self._logger = getLogger()

    def save_embedding_freeze(self, path, loader):
        self.pattern_model.eval()
        self.region_model.eval()
        embeddings = {}
        with torch.no_grad():
            for data in loader:
                for key, pattern in data:
                    tensor = torch.from_numpy(np.vstack(pattern)).to(self.device)
                    embeddings[key] = self.region_model.get_embedding(tensor).cpu().numpy()
        emb = None
        first = True
        for k, v in embeddings.items():
            if first:
                emb = np.array([v])
                first = False
            else:
                emb = np.vstack((emb, v))
        np.save(path, emb)
        # with open(path, 'wb') as f:
        #     pkl.dump(embeddings, f)

    def train_region_triplet_freeze(self, epochs, embeddings, save_name, adaptive=True, window_sizes=None):
        if adaptive:
            criterion = adaptive_triplet_loss
        else:
            criterion = triplet_loss

        test_dataset = FreezePatternForwardDataset(embeddings, self.city_data)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False,
                                                  collate_fn=FreezePatternForwardDataset.collate_fn)
        save_path = 'embeddings/' + self.city_data.city + '/'

        self._logger.info('Building pretraining dataset...')
        if window_sizes is None:
            train_datasets = [FreezePatternPretrainDataset(embeddings, self.city_data, self.config, window_size=3000)]
        else:
            train_datasets = [FreezePatternPretrainDataset(embeddings, self.city_data, self.config, window_size=window_size) for
                              window_size in window_sizes]
        train_loaders = [torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                                     collate_fn=FreezePatternPretrainDataset.collate_fn) for
                         train_dataset in train_datasets]
        for epoch in range(1, epochs+1):
            train_losses = []
            self.region_model.train()
            train_loader = train_loaders[epoch % len(train_loaders)]
            with tqdm(train_loader, desc='Epoch {}'.format(epoch)) as tqdm_batch:
                for data in tqdm_batch:
                    self.region_optimizer.zero_grad()
                    regions = []
                    patterns = []
                    try:
                        # In rare cases, several huge regions are generated, which will cause OOM error
                        # Ignore these cases
                        for pattern in data:
                            packed = np.vstack(pattern)
                            tensor = torch.from_numpy(packed).to(self.device)
                            patterns.append(tensor)
                            regions.append(self.region_model(tensor))
                        anchor = regions[0].mean(dim=0)
                        positive = regions[1].mean(dim=0)
                        negative = regions[2].mean(dim=0)
                        if adaptive:
                            loss = criterion(anchor, positive, negative, patterns[1], patterns[2], self.device)
                        else:
                            loss = criterion(anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0), self.device)
                        loss.backward()
                    except RuntimeError as e:
                        print(e)
                        continue
                    clip_grad_norm_(self.region_model.parameters(), 1.0)
                    self.region_optimizer.step()
                    tqdm_batch.set_postfix(loss=loss.item())
                    train_losses.append(loss.item())
            self.save_embedding_freeze(save_name, test_loader)
            self._logger.info('Epoch {}, Tiplet Loss: {}'.format(epoch, np.mean(train_losses)))
            self.region_scheduler.step()

def wasserstein_distance(x, y, device):
    """ Calculate Wasserstein distance between two regions.
    x.shape=[m,d], y.shape=[n,d]
    """
    off_diag = np.ones((x.shape[0], y.shape[0]))
    indices = np.where(off_diag)
    send_idx = torch.LongTensor(indices[0]).to(device)
    rec_idx = torch.LongTensor(indices[1]).to(device)
    senders = x[send_idx]
    receivers = y[rec_idx]
    # pair-wise matching cost
    similarity = js_div(senders, receivers, get_softmax=True).sum(dim=-1)
    d_cpu = similarity.view(x.shape[0], y.shape[0]).detach().cpu().numpy()
    # calculate optimal transport cost through python optimal transport library
    # This should be faster than using sklearn linear programming api
    p = np.ones(x.shape[0]) / x.shape[0]
    q = np.ones(y.shape[0]) / y.shape[0]
    return ot.emd2(p, q, d_cpu)


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logit:
    Empirically this is better than KL, since it's between 0 and 1.
    """
    if get_softmax:
        p_output = F.softmax(p_output, dim=-1)
        q_output = F.softmax(q_output, dim=-1)
    log_mean_output = ((p_output + q_output) / 2).log()
    return (F.kl_div(log_mean_output, p_output, reduction='none') + F.kl_div(log_mean_output, q_output,
                                                                             reduction='none')) / 2


def adaptive_triplet_loss(anchor, positive, negative, positive_patterns, negative_patterns, device):
    """
        The proposed adaptive triplet loss function.
        batch_size = 1 always hold.
        Since the wasserstein distance will be very small for similar regions
        We can use a large margin without compromise the embedding quality (e.g. 50 - 100)
        The L1 in land use can be sometimes very low (e.g., 0.46 in Singapore), but we don't do such cherry-picking.
    """
    margin = 100 * wasserstein_distance(positive_patterns, negative_patterns, device)
    return triplet_loss(anchor, positive, negative, device, margin=margin)


def triplet_loss(anchor, positive, negative, device, margin=20):
    """
        For fair comparison, we implement the original triplet loss function, in case that some tricks in pytorch
        implementation of triplet loss affect the performance.
    """
    positive_distance = (anchor - positive).abs().sum(dim=-1)
    negative_distance = (anchor - negative).abs().sum(dim=-1)
    loss = torch.max(positive_distance - negative_distance + margin, torch.zeros((1)).to(device)).to(device)
    return loss