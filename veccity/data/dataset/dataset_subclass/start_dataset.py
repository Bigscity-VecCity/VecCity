from veccity.data.dataset.lineseq_dataset import LineSeqDataset
from veccity.data.preprocess import ensure_dir
import torch
from veccity.data.dataset.dataset_subclass.bert_base_dataset import TrajectoryProcessingDataset, padding_mask
import numpy as np
import os
import json
from tqdm import tqdm
import pandas as pd
from veccity.data.dataset.dataset_subclass.bertlm_dataset import BERTSubDataset
from veccity.data.dataset.dataset_subclass.bertlm_constrastive_dataset import collate_unsuperv_contrastive_lm


class STARTDataset(LineSeqDataset):
    def __init__(self,config):
        super().__init__(config)
        self.gat_K = self.config.get('gat_K', 1)
        
        self.append_degree2gcn = self.config.get('append_degree2gcn', True)
        self.add_gat = self.config.get('add_gat', True)
        self.load_trans_prob = self.config.get('load_trans_prob', True) # 数据
        self.normal_feature = self.config.get('normal_feature', False) # 数据
        self.device = self.config.get('device', torch.device('cpu'))
        self.selected_path="veccity/cache/dataset_cache/{}/".format(self.model)
        self.masking_ratio = self.config.get('masking_ratio', 0.2)
        self.masking_mode = self.config.get('masking_mode', 'together')
        self.distribution = self.config.get('distribution', 'random')
        self.avg_mask_len = self.config.get('avg_mask_len', 3)

        self.generate_constrastive_data()

        if self.add_gat:
            self.node_features, self.node_fea_dim = self._load_geo_feature(self.road_geo_df)
            self.edge_index, self.loc_trans_prob = self._load_k_neighbors_and_trans_prob()
            self.adj_mx_encoded = None
        else:
            self.node_features, self.node_fea_dim = None, 0
            self.adj_mx_encoded = None
            self.edge_index = None
            self.loc_trans_prob = None
        
        self.collate_fn=collate_unsuperv_contrastive_lm

    def generate_constrastive_data(self):
        base_path = "veccity/cache/dataset_cache/{}/".format(self.model)
        ensure_dir(base_path)
        # 生成trans prob
        traj_train="veccity/cache/dataset_cache/{}/traj_road_train.csv".format(self.dataset)
        geo_to_ind={key:self.geo_to_ind[key]-self.vocab.specials_num for key in self.geo_to_ind}
        ind_to_geo=self.ind_to_geo[self.vocab.specials_num:]
        trans_probs(self.road_geo_df,geo_to_ind, ind_to_geo,self.road_adj,base_path,self.dataset,self.gat_K,self.seq_len,traj_train)

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
        self._logger.info('edge_index: ' + str(edge_index.shape))  # (vocab_size, pretrain_dim)
        if self.load_trans_prob:
            loc_trans_prob = torch.from_numpy(np.array(loc_trans_prob)).unsqueeze(1).float().to(self.device)  # (E, 1)
            self._logger.info('Trajectory loc-transfer prob shape={}'.format(loc_trans_prob.shape))
        else:
            loc_trans_prob = None
        return edge_index, loc_trans_prob    

    def _load_geo_feature(self, road_info):
        # change road_info same with geo_to_ind
        node_fea_path = self.selected_path + '{}_node_features.npy'.format(self.dataset)
        outdegree = np.sum(self.road_adj, axis=1)  # (N, )
        indegree = np.sum(self.road_adj.T, axis=1)  # (N, )
        road_info['outdegree']=outdegree
        road_info['indegree']=indegree
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
            # change the node_features same with the vocab
            # node_features = node_features[self.ind_to_geo]
            np.save(node_fea_path, node_features)

        self._logger.info('node_features: ' + str(node_features.shape))  # (N, fea_dim)
        node_fea_vec = np.zeros((self.vocab.vocab_size, node_features.shape[1]))
        for ind in range(len(node_features)):
            geo_id = self.ind_to_geo[ind]
            encoded_geo_id = self.vocab.loc2index[geo_id]
            node_fea_vec[encoded_geo_id] = node_features[ind]
        if self.normal_feature:
            self._logger.info('node_features by a/row_sum(a)')  # (vocab_size, fea_dim)
            row_sum = np.clip(node_fea_vec.sum(1), a_min=1, a_max=None)
            for i in range(len(node_fea_vec)):
                node_fea_vec[i, :] = node_fea_vec[i, :] / row_sum[i]
        node_fea_pe = torch.from_numpy(node_fea_vec).float().to(self.device)  # (vocab_size, fea_dim)
        self._logger.info('node_features_encoded: ' + str(node_fea_pe.shape))  # (vocab_size, fea_dim)
        return node_fea_pe, node_fea_pe.shape[1]
    
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

    
    def get_data_feature(self):
        data_feature = {'usr_num': self.usr_num, 'vocab_size': self.vocab_size, 'vocab': self.vocab,
                        "adj_mx": self.road_adj, "num_nodes": self.num_nodes,
                        "geo_file": self.road_geo_df,
                        "geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
                        "node_features": self.node_features, "node_fea_dim": self.node_fea_dim,
                        "adj_mx_encoded": self.adj_mx_encoded, "edge_index": self.edge_index,
                        "loc_trans_prob": self.loc_trans_prob}
        return data_feature

def trans_probs(road_df,geo_to_ind, ind_to_geo,adj_mx,base_path,road_name,K,max_length,traj_train):

    
    geo_ids = list(road_df['id'])
    num_nodes = len(ind_to_geo)

    path = os.path.join(base_path, '{0}_neighbors_{1}.json'.format(road_name, K))
    if os.path.exists(path):
        geoid2neighbors = json.load(open(path, 'r'))
    else:
        adj_mx_bool = adj_mx.astype('bool')
        k_adj_mx_list = [adj_mx_bool]
        for i in tqdm(range(2, K + 1)):
            k_adj_mx_list.append(cal_matmul(k_adj_mx_list[-1], adj_mx_bool)) # 
            np.save(os.path.join(base_path, '{0}_adj_{1}.npy'.format(road_name, i)), k_adj_mx_list[-1])
        print('Finish K order adj_mx')
        for i in tqdm(range(1, len(k_adj_mx_list))):
            adj_mx_bool += k_adj_mx_list[i]
        print('Finish sum of K order adj_mx')
        geoid2neighbors = {}
        for i in tqdm(range(len(adj_mx_bool)), desc='count neighbors'):
            geo_id = int(ind_to_geo[i])
            geoid2neighbors[geo_id] = []
            for j in range(adj_mx_bool.shape[1]):
                if adj_mx_bool[i][j] == 0:
                    continue
                ner_id = int(ind_to_geo[j])
                geoid2neighbors[geo_id].append(ner_id)
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
        train = pd.read_csv(train_file, dtype={'id': int, 'hop': int, 'traj_id': int})

        for _, row in tqdm(train.iterrows(), total=train.shape[0], desc='count traj prob'):
            plist = eval(row.path)[:max_length]
            for i in range(len(plist) - 1):
                prev_geo = plist[i]
                for j in range(1, K+1):
                    if i + j >= len(plist):
                        continue
                    next_geo = plist[i + j]
                    # 减去开始的4个字符，还要看是不是有cls
                    try:
                        prev_ind = geo_to_ind[prev_geo]
                        next_ind = geo_to_ind[next_geo]
                    except Exception:
                        continue
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
                try:
                    p = node_array_in[geo_to_ind[int(k)]][geo_to_ind[int(tgt)]]
                except Exception:
                    continue
                link2prob[id_] = float(p)
        path = os.path.join(base_path, '{0}_trans_prob_{1}.json'.format(road_name, K))
        json.dump(link2prob, open(path, 'w'))


def cal_matmul(mat1, mat2):
    n = mat1.shape[0]
    assert mat1.shape[0] == mat1.shape[1] == mat2.shape[0] == mat2.shape[1]
    res = np.zeros((n, n), dtype='bool')
    for i in tqdm(range(n), desc='outer'):
        for j in range(n):
            res[i, j] = np.dot(mat1[i, :], mat2[:, j])
    return res