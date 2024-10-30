from logging import getLogger
import numpy as np
import os

from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel
from veccity.upstream.region_representation import utils
from veccity.utils import need_train

import torch
import torch.nn as nn
import torch.nn.functional as F

class GMEL(AbstractReprLearningModel):
    def __init__(self,config,data_feature):
        super().__init__(config,data_feature)
        self.device = config.get('device', torch.device('cpu'))
        self.exp_id = config.get('exp_id', None)
        self.dataset = config.get('dataset', '')
        self.model = config.get('model', '')
        self.num_nodes = data_feature.get("num_nodes")
        self._logger = getLogger()
        self.output_dim = config.get('output_dim', 96)
        assert self.output_dim % 2 == 0
        self.txt_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './veccity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.org_npy_cache_file = './veccity/cache/{}/evaluate_cache/org_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.dst_npy_cache_file = './veccity/cache/{}/evaluate_cache/dst_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.output_dim //= 2
        self.iter = config.get('max_epoch', 10)
        self._logger = getLogger()
        self.mini_batch_size = config.get('mini_batch_size',1000)
        self.num_hidden_layers = config.get('num_hidden_layers',3)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.multitask_ratio = config.get('multitask_ratio',[1,0,0])
        self.grad_norm = config.get('grad_norm',1.0)

    def run(self,train_data=None,eval_data=None):
        if not need_train(self.config):
            return
        g,num_nodes,node_feats,train_data,train_inflow,train_outflow,trip_od_valid,trip_volume_valid,trip_od_train,trip_volume_train = self.data_post_process()
        model = GMELModel(g,num_nodes,in_dim=node_feats.shape[1],h_dim = self.output_dim,num_hidden_layers=self.num_hidden_layers,dropout=0, device=self.device, reg_param=0).to(self.device)
        total_num = sum([param.nelement() for param in model.parameters()])
        self._logger.info('Total number of parameters: {}'.format(total_num))
        best_rmse = 1e20
        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
        src_embedding = None
        dst_embedding = None
        self._logger.info('Start training...')
        
        for epoch in range(self.iter):
            model.train()
            mini_batch_gen = utils.mini_batch_gen(train_data,self.mini_batch_size,num_nodes=num_nodes,negative_sampling_rate=0)
        
            for step,mini_batch in enumerate(mini_batch_gen):
                optimizer.zero_grad()
                trip_od = mini_batch[:, :2].long().to(self.device)
                trip_volume = mini_batch[:, -1].float().to(self.device)
                loss = model.get_loss(trip_od, trip_volume, train_inflow, train_outflow, g, multitask_weights=self.multitask_ratio)
                self._logger.info("Epoch {:04d} Step {:04d} | mini batch Loss = {:.4f}".format(epoch, step, loss))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm)
                optimizer.step()
            scheduler.step()
            
            #eval
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    loss = model.get_loss(trip_od_valid, trip_volume_valid, train_inflow, train_outflow, g,multitask_weights=self.multitask_ratio)
                rmse, mae, mape, cpc, cpl = utils.evaluate(model, g, trip_od_valid, trip_volume_valid)
                self._logger.info("-----------------------------------------")
                #self._logger.info("Evaluation on valid dataset:")
                self._logger.info("Epoch {:04d} | Loss = {:.4f}".format(epoch, loss))
                self._logger.info(
                    "RMSE {:.4f} | MAE {:.4f} | MAPE {:.4f} | CPC {:.4f} | CPL {:.4f} |".format(rmse, mae, mape, cpc,
                                                                                                cpl))
                if rmse < best_rmse:
                    best_rmse = rmse
                    src_embedding = model(g).detach().cpu().numpy()  # get embeddings
                    dst_embedding = model.forward2(g).detach().cpu().numpy()  # get embeddings
                    region_embedding = np.concatenate([src_embedding, dst_embedding], axis=1)
                    self._logger.info('Best RMSE found on epoch {}'.format(epoch))
        np.save(self.org_npy_cache_file,src_embedding)
        np.save(self.dst_npy_cache_file, dst_embedding)
        np.save(self.npy_cache_file, region_embedding)
        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：(' + str(len(region_embedding)) + ',' + str(len(region_embedding[0])) + ')')


    def data_post_process(self):
        train_data = self.data_feature['train']
        valid_data = self.data_feature['valid']
        test_data = self.data_feature['test']
        train_inflow = self.data_feature['train_inflow']
        train_outflow = self.data_feature['train_outflow']
        node_feats = self.data_feature['node_feats']
        ct_adj = self.data_feature['ct_adjacency_withweight']
        num_nodes = self.data_feature['num_nodes']
        train_data = torch.from_numpy(train_data)
        trip_od_train = train_data[:, :2].long().to(self.device)
        trip_volume_train = train_data[:, -1].float().to(self.device)
        trip_od_valid = torch.from_numpy(valid_data[:, :2]).long().to(self.device)
        trip_volume_valid = torch.from_numpy(valid_data[:, -1]).float().to(self.device)
        # trip_od_test = torch.from_numpy(test_data[:, :2]).long().to(self.device)
        # trip_volume_test = torch.from_numpy(test_data[:, -1]).float().to(self.device)
        # in/out flow data for multitask target in/out flow
        train_inflow = torch.from_numpy(train_inflow).view(-1, 1).float().to(self.device)
        train_outflow = torch.from_numpy(train_outflow).view(-1, 1).float().to(self.device)
        # construct graph using adjacency matrix
        g = utils.build_graph_from_matrix(ct_adj, node_feats.astype(np.float32), self.device)

        return g,num_nodes,node_feats,train_data,train_inflow,train_outflow,trip_od_valid,trip_volume_valid,trip_od_train,trip_volume_train

class GAT(nn.Module):

    def __init__(self, g, num_nodes, in_dim, h_dim, out_dim, num_hidden_layers=1, dropout=0, device='cpu'):
        # initialize super class
        super().__init__()
        # handle the parameters
        self.g = g
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.device = device
        # create gcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # layer: input to hidden
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # layer: hidden to hidden
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # layer: hidden to output
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        act = F.relu
        return GATInputLayer(self.g, self.in_dim, self.h_dim).to(self.device)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return GATLayer(self.g, self.h_dim, self.h_dim).to(self.device)

    def build_output_layer(self):
        return None

    def forward(self, g):
        h = g.ndata['attr']
        for layer in self.layers:
            h = layer(h)
        return h


class Bilinear(nn.Module):

    def __init__(self, num_feats, dropout=0, device='cpu'):
        return super().__init__()
        # bilinear
        self.bilinear = nn.Bilinear(num_feats, num_feats, 1)
        # dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x1, x2):
        return self.bilinear(x1, x2)

class FNN(nn.Module):

    def __init__(self, num_feats, dropout=0, device=False):
        # init super class
        super().__init__()
        # handle parameters
        self.in_feat = num_feats
        self.h1_feat = num_feats // 2
        self.h2_feat = self.h1_feat // 2
        self.out_feat = 1
        self.device = device
        # dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        # define functions
        self.linear1 = nn.Linear(self.in_feat, self.h1_feat)
        self.linear2 = nn.Linear(self.h1_feat, self.h2_feat)
        self.linear3 = nn.Linear(self.h2_feat, self.out_feat)
        self.activation = F.relu

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        # x = F.relu(x) # enforce the prediction to be non-zero
        return x


def MSE(y_hat, y):
    limit = 20000
    if y_hat.shape[0] < limit:
        return torch.mean((y_hat - y) ** 2)
    else:
        acc_sqe_sum = 0  # accumulative squred error sum
        for i in range(0, y_hat.shape[0], limit):
            acc_sqe_sum += torch.sum((y_hat[i: i + limit] - y[i: i + limit]) ** 2)
        return acc_sqe_sum / y_hat.shape[0]

class GATLayer(nn.Module):
    def __init__(self, g, in_ndim, out_ndim, in_edim=1, out_edim=1):
        '''
        g: the graph
        in_dim: input node feature dimension
        out_dim: output node feature dimension
        edf_dim: input edge feature dimension
        '''
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc0 = nn.Linear(in_edim, out_edim, bias=False)
        self.fc1 = nn.Linear(in_ndim, out_ndim, bias=False)
        self.fc2 = nn.Linear(in_ndim, out_ndim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_ndim + out_edim, 1, bias=False)
        # equation (4)
        self.activation = F.relu
        # parameters
        self.weights = nn.Parameter(torch.Tensor(2, 1))  # used to calculate convex combination weights
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

    def edge_feat_func(self, edges):
        '''
        deal with edge features
        '''
        return {'t': self.fc0(edges.data['d'])}

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['t']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4). this is the core update part.
        z_neighbor = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        z_i = nodes.data['z_i']
        # calculate the convex combination weights
        lambda_ = F.softmax(self.weights, dim=0)
        # update
        h = self.activation(z_i + z_neighbor)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        self.g.apply_edges(self.edge_feat_func)
        z = self.fc1(h)
        self.g.ndata['z'] = z  # message passed to the others
        z_i = self.fc2(h)
        self.g.ndata['z_i'] = z_i  # message passed to self
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class GATInputLayer(nn.Module):

    def __init__(self, g, in_ndim, out_ndim, in_edim=1, out_edim=1):
        '''
        g: the graph
        in_ndim: input node feature dimension
        out_ndim: output node feature dimension
        in_edim: input edge feature dimension
        out_edim: output edge feature dimension
        dropout: dropout rate
        '''
        # initialize super class
        super().__init__()
        # handle parameters
        self.g = g
        # equation (1)
        self.fc0 = nn.Linear(in_edim, out_edim, bias=False)
        self.fc1 = nn.Linear(in_ndim, out_ndim, bias=False)
        self.fc2 = nn.Linear(in_ndim, out_ndim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_ndim + out_edim, 1, bias=False)
        # equation (4)
        self.activation = F.relu
        # parameters
        self.weights = nn.Parameter(torch.Tensor(2, 1))  # used to calculate convex combination weights
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

    def edge_feat_func(self, edges):
        '''
        transform edge features
        '''
        return {'t': self.fc0(edges.data['d'])}

    def edge_attention(self, edges):
        
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['t']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        
        alpha = F.softmax(nodes.mailbox['e'], dim=1) # fix
        # equation (4). this is the core update part.
        z_neighbor = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        z_i = nodes.data['z_i']
        # calculate the convex combination weights
        lambda_ = F.softmax(self.weights, dim=0)
        # update
        h = self.activation(z_i + z_neighbor)
        return {'h': h}

    def forward(self, attr):
        # equation (1)
        self.g.apply_edges(self.edge_feat_func)
        z = self.fc1(attr)  # message passed to the others
        self.g.ndata['z'] = z
        z_i = self.fc2(attr)  # message passed to self
        self.g.ndata['z_i'] = z_i
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)# fix
        return self.g.ndata.pop('h')

class GMELModel(nn.Module):

    def __init__(self, g, num_nodes, in_dim, h_dim, num_hidden_layers=1, dropout=0, device='cpu', reg_param=0):
        '''
        Inputs:
        ---------------------
        g: the graph
        num_nodes: number of nodes in g
        in_dim: original node attributes' dimension
        h_dim: node embedding dimension
        num_hidden_layers: number of hidden layers in graph neural network
        dropout: dropout rate
        device: device
        reg_param: regularization loss coefficient

        Output:
        ---------------------
        embedding of nodes.

        To train the model, use get_loss() to get the overall loss.
        '''
        # init super class
        super().__init__()
        # handle the parameter
        self.reg_param = reg_param
        # create modules
        self.gat = GAT(g, num_nodes, in_dim, h_dim, h_dim, num_hidden_layers, dropout,
                       device)  # GAT for origin node
        self.gat2 = GAT(g, num_nodes, in_dim, h_dim, h_dim, num_hidden_layers, dropout,
                        device)  # GAT for destination nodes.
        # linear plan
        self.edge_regressor = nn.Bilinear(h_dim, h_dim, 1).to(device)
        self.in_regressor = nn.Linear(h_dim, 1).to(device)
        self.out_regressor = nn.Linear(h_dim, 1).to(device)
        # FNN plan
        # self.edge_regressor = FNN(h_dim * 2, dropout, device)
        # self.in_regressor = FNN(h_dim, dropout, device)
        # self.out_regressor = FNN(h_dim, dropout, device)

    def forward(self, g):
        '''
        forward propagate of the graph to get embeddings for the origin node
        '''
        return self.gat.forward(g)

    def forward2(self, g):
        '''
        forward propagate of the graph to get embeddings for destination node
        '''
        return self.gat2.forward(g)

    def get_loss(self, trip_od, trip_volume, in_flows, out_flows, g, multitask_weights=[0.5, 0.25, 0.25]):
        '''
        defines the procedure of evaluating loss function

        Inputs:
        ----------------------------------
        trip_od: list of origin destination pairs
        trip_volume: ground-truth of volume of trip which serves as our target.
        g: DGL graph object

        Outputs:
        ----------------------------------
        loss: value of loss function
        '''
        # calculate the in/out flow of nodes
        # scaled back trip volume
        # get in/out nodes of this batch

        out_nodes, out_flows_idx = torch.unique(trip_od[:, 0], return_inverse=True)
        in_nodes, in_flows_idx = torch.unique(trip_od[:, 1], return_inverse=True)
        # scale the in/out flows of the nodes in this batch
        out_flows = out_flows[out_nodes]
        in_flows = in_flows[in_nodes]
        # get embeddings of each node from GNN
        src_embedding = self.forward(g)
        dst_embedding = self.forward2(g)
        # get edge prediction
        edge_prediction = self.predict_edge(src_embedding, dst_embedding, trip_od)
        # get in/out flow prediction
        in_flow_prediction = self.predict_inflow(dst_embedding, in_nodes)
        out_flow_prediction = self.predict_outflow(src_embedding, out_nodes)
        # get edge prediction loss
        edge_predict_loss = MSE(edge_prediction, trip_volume)
        # get in/out flow prediction loss
        in_predict_loss = MSE(in_flow_prediction, in_flows)
        out_predict_loss = MSE(out_flow_prediction, out_flows)
        # get regularization loss
        reg_loss = 0.5 * (self.regularization_loss(src_embedding) + self.regularization_loss(dst_embedding))
        # return the overall loss
        return multitask_weights[0] * edge_predict_loss + multitask_weights[1] * in_predict_loss + \
               multitask_weights[2] * out_predict_loss + self.reg_param * reg_loss

    def predict_edge(self, src_embedding, dst_embedding, trip_od):
        '''
        using node embeddings to make prediction on given trip OD.
        '''
        # construct edge feature
        src_emb = src_embedding[trip_od[:, 0]]
        dst_emb = dst_embedding[trip_od[:, 1]]
        # get predictions
        # edge_feat = torch.cat((src_emb, dst_emb), dim=1)
        # self.edge_regressor(edge_feat)
        return self.edge_regressor(src_emb, dst_emb)

    def predict_inflow(self, embedding, in_nodes_idx):
        # make prediction
        return self.in_regressor(embedding[in_nodes_idx])

    def predict_outflow(self, embedding, out_nodes_idx):
        # make prediction
        return self.out_regressor(embedding[out_nodes_idx])

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))

