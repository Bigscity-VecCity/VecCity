import torch
import torch.nn as nn
from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
import time
from logging import getLogger
import os


class HyperRoad(AbstractReprLearningModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device')
        self.exp_id = config.get('exp_id', None)
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.dataloader = data_feature.get('dataloader')
        self.num_nodes = data_feature.get("num_nodes")
        self.num_class = data_feature.get("num_classes")
        self.emb_size = config.get('output_dim',128)
        self.agg = config.get('agg')
        self.layer_num = config.get('layer_num')
        self.lamb = config.get('lamb')
        self.gama = config.get('gama')
        self.with_p = config.get('with_p')
        self.one_hot_dim = data_feature.get('one_hot_dim')
        self.lane_cls_num = data_feature.get('lane_cls_num')
        
        self.speed_cls_num = data_feature.get('speed_cls')
        self.oneway_cls_num = data_feature.get('oneway_cls')
        self.epoches = config.get('max_epoch')
        self.lr = config.get('learning_rate')
        self.weight_decay = config.get('weight_decay')
        self.road_embedding_path = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.emb_size)
        self.model_cache_file = './veccity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.emb_size)

        self.model = PEHyper(num_nodes=self.num_nodes, num_classes=self.num_class,out_emb_dim=self.emb_size, device=self.device, agg_method=self.agg,
                layer_num=self.layer_num, lamb=self.lamb, gama=self.gama, with_p=self.with_p,one_hot_dim=self.one_hot_dim,lane_cls_num=self.lane_cls_num,speed_cls_num=self.speed_cls_num,oneway_cls_num=self.oneway_cls_num)
        self.model=self.model.to(self.device)
        self.model.set_paras(data_feature.get("input_x"),data_feature.get("c"),data_feature.get("onehot_input_feats"),data_feature.get("norms")[0],data_feature.get("norms")[1],data_feature.get("norms")[2])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    

    def forward(self, batch):
        return self.model(batch)
    
    def run(self, data=None):
        if not self.config.get('train') and os.path.exists(self.road_embedding_path):
            return
        start_time = time.time()
        for epoch in range(self.epoches):
            loss_data = 0.0
            gnn_loss_data = 0.0
            hgnn_loss_data = 0.0
            tag_loss_data = 0.0
            for i_batch, sample_batched in enumerate(self.dataloader):
                anchor = sample_batched['anchor'].to(self.device)
                pos_edge = sample_batched['pos_edge'].to(self.device)
                neg_edges = sample_batched['neg_edges'].to(self.device)
                pos_hyper = sample_batched['pos_hyper'].to(self.device)
                neg_hypers = sample_batched['neg_hypers'].to(self.device)
                hyper_class = sample_batched['hyper_class'].to(self.device)

                l_lanes = sample_batched['l_lanes'].to(self.device)
                l_maxspeed = sample_batched['l_maxspeed'].to(self.device)
                l_oneway = sample_batched['l_oneway'].to(self.device)
                batch_data = anchor, pos_edge, neg_edges, pos_hyper, neg_hypers, hyper_class, l_lanes, l_maxspeed, l_oneway

                self.optimizer.zero_grad()
                loss, gnn_loss, hgnn_loss, tag_loss = self.model(batch_data)
                loss_data += loss.data.cpu().numpy()
                gnn_loss_data += gnn_loss.data.cpu().numpy()
                hgnn_loss_data += hgnn_loss.data.cpu().numpy()
                tag_loss_data += tag_loss.data.cpu().numpy() 

                loss.backward()
                self.optimizer.step()
            
            self._logger.info(f"epoch num: {epoch}  loss: {loss_data}")
            
        t1 = time.time()-start_time
        self._logger.info('cost time is '+str(t1/self.epoches))
        node_embedding,_ = self.model.update()
        node_embedding=node_embedding.cpu().detach().numpy()
        np.save(self.road_embedding_path,node_embedding)
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), self.model_cache_file)
    
    def static_embedding(self):
        road_emb, hyper_emb=self.model.update()
        return road_emb
    

class LayerNorm(nn.Module):
    """
    layer normalization
    Simple layer norm object optionally used with the convolutional encoder.
    """

    def __init__(self, feature_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.register_parameter("gamma", self.gamma)
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.register_parameter("beta", self.beta)
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, embed_dim]
        # normalize for each embedding
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # output shape is the same as x
        # Type not match for self.gamma and self.beta
        # output: [batch_size, embed_dim]
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

def get_activation_function(activation, context_str):
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise Exception("{} activation not recognized.".format(context_str))

class SingleFeedForwardNN(nn.Module):
    """
        Creates a single layer fully connected feed forward neural network.
        this will use non-linearity, layer normalization, dropout
        this is for the hidden layer, not the last layer of the feed forard NN
    """

    def __init__(self, input_dim,
                 output_dim,
                 dropout_rate=None,
                 activation="sigmoid",
                 use_layernormalize=False,
                 skip_connection=False,
                 context_str=''):
        '''
        Args:
            input_dim (int32): the input embedding dim
            output_dim (int32): dimension of the output of the network.
            dropout_rate (scalar tensor or float): Dropout keep prob.
            activation (string): tanh or relu or leakyrelu or sigmoid
            use_layernormalize (bool): do layer normalization or not
            skip_connection (bool): do skip connection or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(SingleFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.act = get_activation_function(activation, context_str)

        if use_layernormalize:
            # the layer normalization is only used in the hidden layer, not the last layer
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None

        # the skip connection is only possible, if the input and out dimention is the same
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = False

        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size,..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        # Linear layer
        output = self.linear(input_tensor)
        # non-linearity
        output = self.act(output)
        # dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # skip connection
        if self.skip_connection:
            output = output + input_tensor

        # layer normalization
        if self.layernorm is not None:
            output = self.layernorm(output)

        return output

class MultiLayerFeedForwardNN(nn.Module):
    """
        Creates a fully connected feed forward neural network.
        N fully connected feed forward NN, each hidden layer will use non-linearity, layer normalization, dropout
        The last layer do not have any of these
    """

    def __init__(self, input_dim,
                 output_dim,
                 num_hidden_layers=0,
                 dropout_rate=0.5,
                 hidden_dim=-1,
                 activation="relu",
                 use_layernormalize=True,
                 skip_connection=False,
                 context_str=None):
        '''
        Args:
            input_dim (int32): the input embedding dim
            num_hidden_layers (int32): number of hidden layers in the network, set to 0 for a linear network.
            output_dim (int32): dimension of the output of the network.
            dropout (scalar tensor or float): Dropout keep prob.
            hidden_dim (int32): size of the hidden layers
            activation (string): tanh or relu
            use_layernormalize (bool): do layer normalization or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(MultiLayerFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str

        self.layers = nn.ModuleList()
        if self.num_hidden_layers <= 0:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))
        else:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.hidden_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=self.use_layernormalize,
                                                   skip_connection=self.skip_connection,
                                                   context_str=self.context_str))

            for i in range(self.num_hidden_layers - 1):
                self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                       output_dim=self.hidden_dim,
                                                       dropout_rate=self.dropout_rate,
                                                       activation=self.activation,
                                                       use_layernormalize=self.use_layernormalize,
                                                       skip_connection=self.skip_connection,
                                                       context_str=self.context_str))

            self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size, ..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        output = input_tensor
        for i in range(len(self.layers)):
            output = self.layers[i](output)
        return output

def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        # log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) / (frequency_num * 1.0 - 1))
        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) / (frequency_num * 1.0))
        timescales = min_radius * np.exp(np.arange(frequency_num).astype(float) * log_timescale_increment)
        freq_list = 1.0 / timescales
    return freq_list

class GridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(self, spa_embed_dim, coord_dim=2, frequency_num=16,
                 max_radius=10000, min_radius=10,
                 freq_init="geometric", ffn=None, device='cpu'):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(GridCellSpatialRelationEncoder, self).__init__()

        self.device = device
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.ffn = ffn
        # the frequency for each block, alpha
        self.cal_freq_list()
        self.cal_freq_mat()
        self.input_embed_dim = self.cal_input_dim()

        if self.ffn is not None:
            self.ffn = MultiLayerFeedForwardNN(2 * frequency_num * 2, spa_embed_dim,
                                               activation="tanh", dropout_rate=None).to(self.device)

    def cal_elementwise_angle(self, coord, cur_freq):
        '''
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        '''
        return coord / (np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1)))

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (input_embed_dim)
        return embed

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = np.repeat(freq_mat, 2, axis=1)

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords_mat = np.repeat(coords_mat, 2, axis=4)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords_mat * self.freq_mat
        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=input_embed_dim)
        spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1
        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds

class GrahSAGE_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GrahSAGE_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft*2, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, norm_GraphSAGE):
        emb = torch.sparse.mm(norm_GraphSAGE, x)
        emb = torch.cat([x, emb], dim=1)
        x = emb.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = F.relu(x)
        return x

class GNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, activation=True):
        super(GNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.activation = activation

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, norm_GG):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        # x = norm_GG.matmul(x) # sparse dot
        x = torch.sparse.mm(norm_GG, x)
        if self.activation:
            x = F.relu(x)
        return x

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, activation=True):
        super(HGNN_conv, self).__init__()

        self.weight1 = Parameter(torch.Tensor(in_ft, out_ft))
        self.weight2 = Parameter(torch.Tensor(out_ft, in_ft))
        if bias:
            self.bias1 = Parameter(torch.Tensor(out_ft))
            self.bias2 = Parameter(torch.Tensor(in_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.activation = activation

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        self.weight2.data.uniform_(-stdv2, stdv2)
        if self.bias1 is not None:
            self.bias1.data.uniform_(-stdv1, stdv1)
            self.bias2.data.uniform_(-stdv2, stdv2)

    def forward(self, x, norm_HH, norm_HG):
        x = x.matmul(self.weight1)
        if self.bias1 is not None:
            x = x + self.bias1
        # hyper_emb = norm_HH.matmul(x) # sparse dot (H*N)
        hyper_emb = torch.sparse.mm(norm_HH, x)
        if self.activation:
            hyper_emb = F.relu(hyper_emb)

        z = hyper_emb.matmul(self.weight2)
        if self.bias2 is not None:
            z = z + self.bias2
        # x = norm_HG.matmul(z) # sparse dot (N*H)
        x = torch.sparse.mm(norm_HG, z)
        if self.activation:
            x = F.relu(x)
        return x, hyper_emb

class RoadLayer(nn.Module):
    def __init__(self, in_ft, agg_method):
        super(RoadLayer, self).__init__()

        self.gnn_layer = GNN_conv(in_ft, in_ft, bias=True, activation=True)
        self.hgnn_layer = HGNN_conv(in_ft, in_ft, bias=True, activation=True)

        self.agg_method = agg_method
        if self.agg_method == "mean":
            self.layer_attention = None
            self.layer_mlp = None
        if self.agg_method == "attention":
            self.layer_attention = nn.Linear(in_ft, 1)
            self.layer_mlp = None
        if self.agg_method == "mlp":
            self.layer_mlp = nn.Linear(in_ft*3, in_ft)
            self.layer_attention = None

    def layer_agg(self, x, gnn_emb, hgnn_emb, att_layer, mlp_layer):
        method = self.agg_method
        if method == "mean":
            return (x + gnn_emb + hgnn_emb) / 3
        if method == "attention":
            emb = torch.stack([x, gnn_emb, hgnn_emb]) # 3 * N * E
            emb = emb.transpose(0, 1) # N * 3 * E
            att_vals = att_layer(emb) # N * 3 * 1
            att_vals = att_vals.transpose(1, 2) # N * 1 * 3
            att_vals = F.softmax(att_vals, dim=2)  # N * 1 * 3
            fused_emb = torch.bmm(att_vals, emb).squeeze(1) # N * E
            return fused_emb
        if method == "mlp":
            emb = torch.cat([x, gnn_emb, hgnn_emb], dim=1) # N * 3E
            fused_emb = mlp_layer(emb) # N * E
            return fused_emb

    def forward(self, x, norm_GG, norm_HH, norm_HG):
        gnn_emb = self.gnn_layer(x, norm_GG)
        hgnn_emb, hyper_emb = self.hgnn_layer(x, norm_HH, norm_HG)
        x = self.layer_agg(x, gnn_emb, hgnn_emb, self.layer_attention, self.layer_mlp)
        return x, hyper_emb

    
# first conv on all data and then sampling a batch to give batch loss
class PEHyper(nn.Module):
    def __init__(self, num_nodes, num_classes, out_emb_dim=64, emb_dim=128, device='cpu', agg_method="mean",
                 layer_num=4, lamb=1.0, with_p="True", one_hot_dim = 17,gama=1.0,lane_cls_num=9,speed_cls_num=6,oneway_cls_num=2):
        super(PEHyper, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.out_emb_dim = out_emb_dim
        self.layer_num = layer_num
        self.lamb = lamb
        self.gama = gama
        self.with_p = with_p

        self.spenc = GridCellSpatialRelationEncoder(spa_embed_dim=emb_dim, ffn=True,
                                                    min_radius=10, max_radius=10000, device=device)
        self.node_embed = nn.Embedding(num_nodes, emb_dim)

        if self.with_p == "True":
            self.fusion_fc = nn.Linear(emb_dim*2, out_emb_dim-one_hot_dim)
        else:
            self.fusion_fc = nn.Linear(emb_dim, out_emb_dim-one_hot_dim)

        layers = []
        for i in range(self.layer_num):
            layers.append(RoadLayer(out_emb_dim, agg_method))
        self.layers = nn.ModuleList(layers)
        self.clf_fc = nn.Linear(out_emb_dim, num_classes)

        self.lane_cls_fc = nn.Linear(out_emb_dim, lane_cls_num)
        self.speed_cls_fc = nn.Linear(out_emb_dim, speed_cls_num)
        self.oneway_cls_fc = nn.Linear(out_emb_dim, oneway_cls_num)

    def set_paras(self, input_x, c, onehot_input_feats, norm_GG, norm_HH, norm_HG):
        self.input_x = input_x.to(self.device)
        self.c = c.to(self.device)
        self.onehot_input_feats = onehot_input_feats.to(self.device)
        self.norm_GG = norm_GG.to(self.device)
        self.norm_HH = norm_HH.to(self.device)
        self.norm_HG = norm_HG.to(self.device)

    def conv(self, x):
        for layer in self.layers:
            x, hyper_emb = layer(x, self.norm_GG, self.norm_HH, self.norm_HG)
        return x, hyper_emb

    def update(self):
        input_x = self.input_x
        c = self.c

        c = c.reshape(1, c.shape[0], c.shape[1])  # c: batch*2
        pe_emb = self.spenc(c.detach().cpu().numpy())
        pe_emb = pe_emb.reshape(pe_emb.shape[1], pe_emb.shape[2])

        x_emb = self.node_embed(input_x)
        if self.with_p == "True":
            x_emb = torch.cat((x_emb, pe_emb), dim=1)
        road_emb = self.fusion_fc(x_emb)
        road_emb = torch.cat([road_emb, self.onehot_input_feats], dim=1)

        road_emb, hyper_emb = self.conv(road_emb)
        return road_emb, hyper_emb

    def forward(self, batch_data):
        road_emb, hyper_emb = self.update()
        anchor, pos_edge, neg_edges, pos_hyper, neg_hypers, hyper_class, l_lanes, l_maxspeed, l_oneway = batch_data
        batch_size = anchor.size()[0]
        emb_size = self.out_emb_dim

        # batch * 1
        anchor_emb = road_emb[anchor.view(-1)].view(batch_size,-1,emb_size)
        pos_edge_emb = road_emb[pos_edge.view(-1)].view(batch_size,-1,emb_size)
        neg_edges_emb = road_emb[neg_edges.view(-1)].view(batch_size,-1,emb_size)

        pos_hyper_emb = hyper_emb[pos_hyper.view(-1)].view(batch_size,-1,emb_size)
        neg_hypers_emb = hyper_emb[neg_hypers.view(-1)].view(batch_size,-1,emb_size)
        hyper_class = hyper_class.view(-1)

        # GNN task using simple dot
        pos_edge_score = anchor_emb.bmm(pos_edge_emb.transpose(1,2)).squeeze(1) # b*1*k
        neg_edge_score = anchor_emb.bmm(neg_edges_emb.transpose(1,2)).squeeze(1)
        gnn_loss = self.hinge_loss(pos_edge_score, neg_edge_score)

        # HGNN task using simple dot
        pos_hyper_score = anchor_emb.bmm(pos_hyper_emb.transpose(1,2)).squeeze(1)
        neg_hyper_score = anchor_emb.bmm(neg_hypers_emb.transpose(1,2)).squeeze(1)
        hgnn_loss = self.hinge_loss(pos_hyper_score, neg_hyper_score)

        # cluster task using cross entropy
        hyper_clf_res = self.clf_fc(pos_hyper_emb).squeeze(1)
        clf_loss = self.crossentropy_loss(hyper_clf_res, hyper_class)

        # tag reconstruct using cross entropy
        l_lanes = l_lanes.view(-1)
        lanes_clf_res = self.lane_cls_fc(anchor_emb).squeeze(1)
        lanes_loss = self.crossentropy_loss(lanes_clf_res, l_lanes)

        l_maxspeed = l_maxspeed.view(-1)
        speed_clf_res = self.speed_cls_fc(anchor_emb).squeeze(1)
        speed_loss = self.crossentropy_loss(speed_clf_res, l_maxspeed)

        l_oneway = l_oneway.view(-1)
        oneway_clf_res = self.oneway_cls_fc(anchor_emb).squeeze(1)
        oneway_loss = self.crossentropy_loss(oneway_clf_res, l_oneway)
        tag_loss = lanes_loss + speed_loss + oneway_loss

        loss = self.gama * gnn_loss + self.lamb * (hgnn_loss + clf_loss) + tag_loss

        return loss, self.gama * gnn_loss, self.lamb * (hgnn_loss + clf_loss), tag_loss

    def encode(self):
        road_emb, hyper_emb = self.update()
        return road_emb

    def hinge_loss(self, pos_score, neg_score):
        # pos_score = F.logsigmoid(pos_score)
        # neg_score = F.logsigmoid(neg_score) # add function
        return (1 - pos_score + neg_score).clamp(min=0).mean()

    def crossentropy_loss(self, clf_res, classes):
        loss_fuc = nn.CrossEntropyLoss().to(self.device)
        clf_loss = loss_fuc(clf_res, classes)
        return clf_loss

class POS(nn.Module):
    def __init__(self, emb_dim=128, device='cpu'):
        super(POS, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.spenc = GridCellSpatialRelationEncoder(spa_embed_dim=emb_dim, ffn=None,
                                                    min_radius=100, max_radius=10000, device=device)

    def encode(self, c):
        c = c.reshape(1, c.shape[0], c.shape[1])  # c: batch*2
        pe_emb = self.spenc(c.detach().cpu().numpy())
        pe_emb = pe_emb.reshape(pe_emb.shape[1], pe_emb.shape[2])
        return pe_emb
    
        


        

        