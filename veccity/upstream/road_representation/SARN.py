import sys
sys.path.append('..')
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
import logging
import time
import copy
import random
import typing
import numpy as np
import pickle
import dgl
from veccity.utils import tool_funcs, OSMLoader, EdgeIndex
from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class FeatEmbedding(nn.Module):
    def __init__(self, nsegid_code, nhighway_code,
            nlength_code, nradian_code, nlon_code, nlat_code, hd=16,ld=16,rd=16,lld=32):
        super(FeatEmbedding, self).__init__()
        self.sarn_seg_feat_highwaycode_dim=hd
        self.sarn_seg_feat_lengthcode_dim=ld
        self.sarn_seg_feat_radiancode_dim=rd
        self.sarn_seg_feat_lonlatcode_dim=lld

        logging.debug('FeatEmbedding args. {}, {}, {}, {}, {}, {}'.format(nsegid_code, nhighway_code, nlength_code, \
                        nradian_code, nlon_code, nlat_code))
        
        self.emb_highway = nn.Embedding(nhighway_code, self.sarn_seg_feat_highwaycode_dim)
        self.emb_length = nn.Embedding(nlength_code, self.sarn_seg_feat_lengthcode_dim)
        self.emb_radian = nn.Embedding(nradian_code, self.sarn_seg_feat_radiancode_dim)
        self.emb_lon = nn.Embedding(nlon_code, self.sarn_seg_feat_lonlatcode_dim)
        self.emb_lat = nn.Embedding(nlat_code, self.sarn_seg_feat_lonlatcode_dim)

    # inputs = [N, nfeat]
    def forward(self, inputs):
        return torch.cat( (
                self.emb_highway(inputs[: , 1]),
                self.emb_length(inputs[: , 2]),
                self.emb_radian(inputs[: , 3]),
                self.emb_lon(inputs[: , 4]),
                self.emb_lat(inputs[: , 5]),
                self.emb_lon(inputs[: , 6]),
                self.emb_lat(inputs[: , 7])), dim = 1)

class SARN(AbstractReprLearningModel):
    def __init__(self, config, data_feature):
        super(SARN, self).__init__(config, data_feature)
        self.config = config
        self.dataset_path=config.get("data_path","")
        self.exp_id = config.get('exp_id', None)
        self.dataset = config.get('dataset', '')
        self.device = config.get('device')
        self.osm_data = OSMLoader(self.dataset_path, schema = 'SARN', device=self.device)
        self.osm_data.load_cikm_data(self.dataset)
        self.model = 'SARN'
       
        self.sarn_seg_feat_dim = config.get("sarn_seg_feat_dim", 176)
        assert self.sarn_seg_feat_dim % 4 == 0
        self.sarn_embedding_dim = config.get("sarn_embedding_dim", 128)
        self.sarn_out_dim = config.get("output_dim", 32)
        self.sarn_moco_each_queue_size = self.osm_data.sarn_moco_each_queue_size
        self.sarn_moco_temperature = config.get("sarn_moco_temperature", 0.05)
        self.sarn_moco_total_queue_size = config.get("sarn_moco_total_queue_size", 1000)
        self.sarn_moco_multi_queue_cellsidelen = config.get("sarn_moco_multi_queue_cellsidelen", 0)
        self.sarn_moco_loss_local_weight = config.get("sarn_moco_loss_local_weight", 0.4)
        self.sarn_learning_rate = config.get("sarn_learning_rate", 0.005)
        self.sarn_learning_weight_decay = config.get("sarn_learning_weight_decay", 0.0001)
        self.sarn_training_bad_patience = config.get("sarn_training_bad_patience", 50)
        self.sarn_epochs = config.get("max_epoch", 5)
        self.sarn_moco_loss_global_weight=config.get("sarn_moco_loss_global_weight", 1-self.sarn_moco_loss_local_weight)
        self.sarn_batch_size = config.get("sarn_batch_size", 128)
        self.sarn_learning_rated_adjusted=config.get("sarn_learning_rated_adjusted", True)

        self.seg_feats = self.osm_data.seg_feats
        self.checkpoint_path = './veccity/cache/{}/model_cache/{}_SARN_best.pkl'.format(self.exp_id, self.dataset)
        self.embs_path = './veccity/cache/{}/model_cache/{}_SARN_best_embs.pickle'.format(self.exp_id, self.dataset)
        self.road_embedding_path = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.sarn_embedding_dim)
        
        self.feat_emb = FeatEmbedding(self.osm_data.count_segid_code,
                                    self.osm_data.count_highway_cls,
                                    self.osm_data.count_length_code,
                                    self.osm_data.count_radian_code,
                                    self.osm_data.count_s_lon_code,
                                    self.osm_data.count_s_lat_code).to(self.device)

        self.model = MoCo(nfeat = self.sarn_seg_feat_dim,
                                nemb = self.sarn_embedding_dim, 
                                nout = self.sarn_out_dim,
                                queue_size = self.sarn_moco_each_queue_size,
                                nqueue = self.osm_data.cellspace.lon_size * self.osm_data.cellspace.lat_size,
                                temperature = self.sarn_moco_temperature).to(self.device)

        logging.info('[Moco] total_queue_size={:.0f}, multi_side_length={:.0f}, '
                        'multi_nqueues={}*{}, each_queue_size={}, real_total={}, local_weight={:.2f}' \
                    .format(self.sarn_moco_total_queue_size, \
                            self.sarn_moco_multi_queue_cellsidelen, \
                            self.osm_data.cellspace.lon_size, \
                            self.osm_data.cellspace.lat_size, \
                            self.sarn_moco_each_queue_size, \
                            self.sarn_moco_each_queue_size * self.osm_data.cellspace.lon_size * self.osm_data.cellspace.lat_size, \
                            self.sarn_moco_loss_local_weight))

        self.seg_id_to_idx = self.osm_data.seg_id_to_idx_in_adj_seg_graph
        self.seg_idx_to_id = self.osm_data.seg_idx_to_id_in_adj_seg_graph
        self.seg_id_to_cellid = dict(self.osm_data.segments.reset_index()[['inc_id','c_cellid']].values.tolist()) # contains those not legal segments
        self.seg_idx_to_cellid = [-1] * len(self.seg_idx_to_id)
        for _id, _cellid in self.seg_id_to_cellid.items():
            _idx = self.seg_id_to_idx.get(_id, -1)
            if _idx >= 0:
                self.seg_idx_to_cellid[_idx] = _cellid
        assert sum(filter(lambda x: x < 0, self.seg_idx_to_cellid)) == 0


    def run(self,data=None):
        if not self.config.get('train') and os.path.exists(self.road_embedding_path):
            return
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("[Training] START! timestamp={:.0f}".format(training_starttime))
        torch.autograd.set_detect_anomaly(True)
        
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.feat_emb.parameters()), 
                                        lr = self.sarn_learning_rate,
                                        weight_decay = self.sarn_learning_weight_decay)
        
        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = self.sarn_training_bad_patience
        start_time = time.time()
        for i_ep in range(self.sarn_epochs):
            _time_ep = time.time()
            loss_ep = []
            train_gpu = []
            train_ram = []

            self.feat_emb.train()
            self.model.train()

            if self.sarn_learning_rated_adjusted:
                tool_funcs.adjust_learning_rate(optimizer, self.sarn_learning_rate, i_ep, self.sarn_epochs)
            
            # drop edges from edge_index
            edge_index_1 = copy.deepcopy(self.osm_data.edge_index)
            edge_index_1 = graph_aug_edgeindex(edge_index_1)

            edge_index_2 = copy.deepcopy(self.osm_data.edge_index)
            edge_index_2 = graph_aug_edgeindex(edge_index_2)

            for i_batch, batch in enumerate(self.__train_data_generator_batchi(edge_index_1, edge_index_2, shuffle=True)):
                _time_batch = time.time()
                
                optimizer.zero_grad()
                (sub_seg_feats_1, sub_edge_index_1, mapping_to_origin_idx_1), \
                        (sub_seg_feats_2, sub_edge_index_2, mapping_to_origin_idx_2), \
                        sub_seg_ids, sub_cellids = batch

                sub_seg_feats_1 = self.feat_emb(sub_seg_feats_1)
                sub_seg_feats_2 = self.feat_emb(sub_seg_feats_2)

                model_rtn = self.model(sub_seg_feats_1, sub_edge_index_1, mapping_to_origin_idx_1, \
                                        sub_seg_feats_2, sub_edge_index_2, mapping_to_origin_idx_2, \
                                        sub_cellids, sub_seg_ids)
                loss = self.model.loss_mtl(*model_rtn, self.sarn_moco_loss_local_weight, self.sarn_moco_loss_global_weight)

                loss.backward()
                optimizer.step()
                loss_ep.append(loss.item())
                train_gpu.append(tool_funcs.GPUInfo.mem()[0])
                train_ram.append(tool_funcs.RAMInfo.mem())

                if i_batch % 50 == 0:
                    logging.debug("[Training] ep-batch={}-{}, loss={:.3f}, @={:.3f}, gpu={}, ram={}" \
                            .format(i_ep, i_batch, loss.item(), time.time() - _time_batch, \
                                    tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))


            loss_ep_avg = tool_funcs.mean(loss_ep)
            logging.info("[Training] ep={}: avg_loss={:.3f}, @={:.3f}, gpu={}, ram={}" \
                    .format(i_ep, loss_ep_avg, time.time() - _time_ep, tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))
            
            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)

            # early stopping
            if loss_ep_avg < best_loss_train:
                best_epoch = i_ep
                best_loss_train = loss_ep_avg
                bad_counter = 0
                torch.save({'model_state_dict': self.model.state_dict(),
                            'feat_emb_state_dict': self.feat_emb.state_dict()},
                            self.checkpoint_path)
                node_embedding=self.get_embeddings(False)
                np.save(self.road_embedding_path,node_embedding.cpu().detach().numpy())
            else:
                bad_counter += 1

            if bad_counter == bad_patience or (i_ep + 1) == self.sarn_epochs:
                logging.info("[Training] END! best_epoch={}, best_loss_train={:.6f}" \
                            .format(best_epoch, best_loss_train))
                break
        t1 = time.time()-start_time
        logging.info('cost time is '+str(t1/self.sarn_epochs))
        logging.info("enc_train_time:{} \n enc_train_gpu:{} \n enc_train_ram:{} \n".format(time.time()-training_starttime, training_gpu_usage, training_ram_usage))


    def finetune_forward(self, sub_seg_idxs, is_training: bool):
        if is_training:
            self.feat_emb.train()
            self.model.train()
            embs = self.model.encoder_q(self.feat_emb(self.seg_feats), self.t_edge_index)[sub_seg_idxs]

        else:
            with torch.no_grad():
                self.feat_emb.eval()
                self.model.eval()
                embs = self.model.encoder_q(self.feat_emb(self.seg_feats), self.t_edge_index)[sub_seg_idxs]
        return embs


    def __train_data_generator_batchi(self, edge_index_1: EdgeIndex, \
                                            edge_index_2: typing.Union[EdgeIndex, None], \
                                            shuffle = True):
        cur_index = 0
        n_segs = len(self.seg_idx_to_id)
        seg_idxs = list(range(n_segs))

        if shuffle: # for training
            random.shuffle(seg_idxs)

        while cur_index < n_segs:
            end_index = cur_index + self.sarn_batch_size \
                            if cur_index + self.sarn_batch_size < n_segs \
                            else n_segs
            sub_seg_idx = seg_idxs[cur_index: end_index]

            sub_edge_index_1, new_x_idx_1, mapping_to_origin_idx_1 = \
                                edge_index_1.sub_edge_index(sub_seg_idx)
            sub_seg_feats_1 = self.seg_feats[new_x_idx_1]
            sub_seg_feats_1 = sub_seg_feats_1.to(self.device)
            sub_edge_index_1 = torch.tensor(sub_edge_index_1, dtype = torch.long, device = self.device)
            
            if edge_index_2 != None:
                sub_edge_index_2, new_x_idx_2, mapping_to_origin_idx_2 = \
                                    edge_index_2.sub_edge_index(sub_seg_idx)
                sub_seg_feats_2 = self.seg_feats[new_x_idx_2]
                sub_seg_feats_2 = sub_seg_feats_2.to(self.device)
                sub_edge_index_2 = torch.tensor(sub_edge_index_2, dtype = torch.long, device = self.device)

                sub_seg_ids = [self.seg_idx_to_id[idx] for idx in sub_seg_idx]
                sub_cellids = [self.seg_idx_to_cellid[idx] for idx in sub_seg_idx]
                sub_seg_ids = torch.tensor(sub_seg_ids, dtype = torch.long, device = self.device)
                sub_cellids = torch.tensor(sub_cellids, dtype = torch.long, device = self.device)
                
                yield (sub_seg_feats_1, sub_edge_index_1, mapping_to_origin_idx_1), \
                        (sub_seg_feats_2, sub_edge_index_2, mapping_to_origin_idx_2), \
                        sub_seg_ids, sub_cellids
            else:
                yield sub_seg_feats_1, sub_edge_index_1, mapping_to_origin_idx_1

            cur_index = end_index
    

    @torch.no_grad()
    def load_model_state(self, f_path):
        checkpoint = torch.load(f_path)
        self.feat_emb.load_state_dict(checkpoint['feat_emb_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feat_emb.to(self.device)
        self.model.to(self.device)


    @torch.no_grad()
    def get_embeddings(self, from_checkpoint): # return embs on cpu!
        if from_checkpoint:
            self.load_model_state(self.checkpoint_path)

        edge_index_1 = copy.deepcopy(self.osm_data.edge_index)

        self.feat_emb.eval()
        self.model.eval()
        embs = torch.empty((0), device = self.device)

        with torch.no_grad():
            for i_batch, batch in enumerate(self.__train_data_generator_batchi(edge_index_1, None, shuffle = False)):
                    
                sub_seg_feats_1, sub_edge_index_1, mapping_to_origin_idx_1 = batch
                sub_seg_feats_1 = self.feat_emb(sub_seg_feats_1)

                emb = self.model.encoder_q(sub_seg_feats_1, sub_edge_index_1)
                emb = emb[mapping_to_origin_idx_1]
                embs = torch.cat((embs, emb), 0)

            embs = F.normalize(embs, dim = 1) # dim=0 feature norm, dim=1 obj norm
            return embs


    @torch.no_grad()
    def dump_embeddings(self, embs = None):
        if embs == None:
            embs = self.get_embeddings(True)
        with open(self.embs_path, 'wb') as fh:
            pickle.dump(embs, fh, protocol = pickle.HIGHEST_PROTOCOL)
        logging.info('[dump embedding] done.')
        return

class Projector(nn.Module):
    def __init__(self, nin, nhid, nout):
        super(Projector, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(nin, nhid), 
                                nn.ReLU(), 
                                nn.Linear(nhid, nout))
        self.reset_parameter()

    def forward(self, x):
        return self.mlp(x)

    def reset_parameter(self):
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1.414)
                torch.nn.init.zeros_(m.bias)
        
        self.mlp.apply(_weights_init)
    
    
# multi queue
class MoCo(nn.Module):
    def __init__(self, nfeat, nemb, nout,
                queue_size, nqueue, mmt = 0.999, temperature = 0.07):
        super(MoCo, self).__init__()

        self.queue_size = queue_size
        self.nqueue = nqueue
        self.mmt = mmt
        self.temperature = temperature
        self.nfeat = nfeat
        # for simplicity, initialize gat objects here
        self.encoder_q = GAT(nfeat = nfeat, nhid = nfeat // 2, nout = nemb, nhead = 4, nlayer = 2)
        self.encoder_k = GAT(nfeat = nfeat, nhid = nfeat // 2, nout = nemb, nhead = 4, nlayer = 2)

        self.mlp_q = Projector(nemb, nemb, nout) 
        self.mlp_k = Projector(nemb, nemb, nout)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queues
        self.queues = MomentumQueue(nout, queue_size, nqueue)


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1. - self.mmt)
        
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1. - self.mmt)


    def forward(self, inputs_q, edge_index_q, idx_in_adjsub_q, 
                        inputs_k, edge_index_k, idx_in_adjsub_k,
                        q_ids, elem_ids):
        # length of different parameteres may be different
        # compute query features        
        q = self.mlp_q(self.encoder_q(inputs_q, edge_index_q)) # q: [?, nfeat]
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.mlp_k(self.encoder_k(inputs_k, edge_index_k))  
            k = nn.functional.normalize(k, dim=1)

        q = q[idx_in_adjsub_q] # q: [batch, nfeat]
        k = k[idx_in_adjsub_k] # k: [batch, nfeat]

        # positive logits
        l_pos_local = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # [batch, 1]

        neg_local = self.queues.queue[q_ids].clone().detach() # [batch, nfeat, queue_size]
        l_neg_local = torch.einsum('nc,nck->nk', q, neg_local) # [batch, queue_size]

        neg_local_ids = self.queues.ids[q_ids].clone().detach() # [batch, queue_size]
        l_neg_local[neg_local_ids == elem_ids.unsqueeze(1).repeat(1, neg_local_ids.shape[1])] = -9e15  # [batch, queue_size]

        neg_global = torch.mean(self.queues.queue.clone().detach(), dim = 2) # [nqueue, nfeat], readout
        l_neg_global = torch.einsum('nc,ck->nk', q, neg_global.T) # [batch, nqueue]

        # logits: 
        logits_local = torch.cat([l_pos_local, l_neg_local], dim=1) # [batch, (1+queue_size)]
        logits_global = l_neg_global

        # apply temperature
        logits_local /= self.temperature
        logits_global /= self.temperature

        # labels: positive key indicators
        labels_local = torch.zeros_like(l_pos_local, dtype = torch.long).squeeze(1)
        labels_global = q_ids.clone().detach()

        # dequeue and enqueue
        for i, q_id in enumerate(q_ids):
            self.queues.dequeue_and_enqueue(k[i,:], elem_ids[i].item(), q_id)

        return logits_local, labels_local, logits_global, labels_global


    # local and global losses are served as a multi-task-learning loss
    def loss_mtl(self, logits_local, labels_local, logits_global, labels_global,
                w_local, w_global):
        
        # temperature has applied in forward()
        sfmax_local = F.softmax(logits_local, dim = 1) # [batch, 1+queue_size]
        sfmax_global = F.softmax(logits_global, dim = 1) # [batch, n_queue]
        p_local = torch.log(
                    sfmax_local.gather(1, labels_local.view(-1,1))) # [batch, 1]
        p_global = torch.log(
                    sfmax_global.gather(1, labels_global.view(-1,1))) # [batch, 1]

        loss_local = F.nll_loss(p_local, torch.zeros_like(labels_local))
        loss_global = F.nll_loss(p_global, torch.zeros_like(labels_local)) 

        return loss_local * w_local + loss_global * w_global

class MomentumQueue(nn.Module):

    def __init__(self, nhid, queue_size, nqueue):
        super(MomentumQueue, self).__init__()
        self.queue_size = queue_size
        self.nqueue =  nqueue

        self.register_buffer("queue", torch.randn(nqueue, nhid, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim = 1)
        self.register_buffer("ids", torch.full([nqueue, queue_size], -1, dtype = torch.long))
        self.register_buffer("queue_ptr", torch.zeros((nqueue), dtype = torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, k, elem_id, q_id):
        # k: feature
        # elem_id: used in ids
        # q_id: queue id

        ptr = int(self.queue_ptr[q_id].item())
        self.queue[q_id, :, ptr] = k
        self.ids[q_id, ptr] = elem_id

        ptr = (ptr + 1) % self.queue_size
        self.queue_ptr[q_id] = ptr

def graph_aug_edgeindex(edge_index: EdgeIndex):
    # 1. sample to-be-removed edges by weights respectively
    # 2. union all these to-be-removed edges in one set
    sarn_break_edge_topo_prob = 0.4

    _time = time.time()
    n_ori_edges = edge_index.length()
    n_topo_remove = n_spatial_remove = 0

    edges_topo_weight = edge_index.tweight # shallow copy
    edges_topo_weight_0 = (edges_topo_weight == 0) # to mask
    n_topo = n_ori_edges - sum(edges_topo_weight_0)

    max_tweight = max(edges_topo_weight) + 1.5
    edges_topo_weight = np.log(max_tweight - edges_topo_weight) / np.log(1.5)
    edges_topo_weight[edges_topo_weight_0] = 0
    sum_tweight = sum(edges_topo_weight)
    edges_topo_weight = edges_topo_weight / sum_tweight
    edges_topo_weight = edges_topo_weight.tolist()

    edges_idxs_to_remove = set(np.random.choice(n_ori_edges, p = edges_topo_weight, \
                                                size = int(sarn_break_edge_topo_prob * n_topo), \
                                                replace = False))
    n_topo_remove = len(edges_idxs_to_remove)

    edges_idxs_to_remove = list(edges_idxs_to_remove)
    edge_index.remove_edges(edges_idxs_to_remove)

    logging.debug("[Graph Augment] @={:.0f}, #original_edges={}, #edges_broken={} ({}+{}), #edges_left={}" \
                    .format(time.time() - _time, n_ori_edges, len(edges_idxs_to_remove), \
                            n_topo_remove, n_spatial_remove, edge_index.length() ))
    
    return edge_index

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, nhead, nlayer = 1):
        super(GAT, self).__init__()
        assert nlayer >= 1
        self.nfeat = nfeat
        self.nout = nout
        self.nlayer = nlayer
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(nfeat, nhid, nhead, 
                                    feat_drop = 0.2, negative_slope = 0.2))
        for _ in range(nlayer - 1):
            self.layers.append(GATConv(nhid * nhead, nhid, nhead, 
                                    feat_drop = 0.2, negative_slope = 0.2))
        self.layer_out = GATConv(nhead * nhid, nout, 1,
                            feat_drop = 0.2, negative_slope = 0.2)

    def forward(self, x, edge_index):
        edge_index = dgl.graph((edge_index[0], edge_index[1]))
        edge_index = dgl.add_self_loop(edge_index)
        for l in range(self.nlayer):
            x = F.dropout(x, p = 0.2, training = self.training)
            x = self.layers[l](edge_index,x).view(-1, self.nfeat * 2)
            x = F.elu(x)
        # output projection
        x = F.dropout(x, p = 0.2, training = self.training)
        x = self.layer_out(edge_index,x).view(-1, self.nout)
        return x
