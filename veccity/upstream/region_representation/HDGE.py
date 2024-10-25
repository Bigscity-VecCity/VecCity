import time
from logging import getLogger
import random

import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel
from veccity.utils import need_train

#[2017-CIKM region Representation Learning via Mobility Flow]
class HDGE(AbstractReprLearningModel):

    def __init__(self,config,data_feature):
        super().__init__(config,data_feature)
        self.combine_graph = data_feature.get("combine_graph")
        self.flow_graph = data_feature.get("flow_graph")
        self.spatial_graph = data_feature.get("spatial_graph")
        self.time_slice = data_feature.get("time_slice")
        self.C = data_feature.get("C")
        self.num_nodes = data_feature.get("num_nodes")
        self.geo_to_ind = data_feature.get('geo_to_ind', None)
        self.ind_to_geo = data_feature.get('ind_to_geo', None)
        self._logger = getLogger()
        self.output_dim = config.get('output_dim', 96)
        self.is_directed = config.get('is_directed', True)
        self.p = config.get('p', 2)
        self.q = config.get('q', 1)
        self.dataset = config.get('dataset', '')
        self.num_walks = config.get('num_walks', 100)
        self.walk_length = self.time_slice+1
        self.window_size = config.get('window_size', 10)
        self.num_workers = config.get('num_workers', 10)
        self.iter = config.get('max_epoch', 1000)
        self.accept_tables = {}
        self.alias_tables = {}
        self.model = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.txt_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './veccity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)

    #一共是time_slice * num_nodes
    def run(self, train_dataloader=None, eval_dataloader=None):
        if not need_train(self.config):
            return
        start_time = time.time()
        combine_matrix = self.combine_graph
        walks = self.simulate_walks(combine_matrix,self.num_walks,self.walk_length)
        model = self.learn_embeddings(walks = walks, dimensions=self.output_dim,
                                 window_size=self.window_size, workers=self.num_workers, iters=self.iter)
        t1 = time.time()-start_time
        self._logger.info('cost time is '+str(t1/self.iter))
        vocab_size = len(model.wv.index_to_key)
        vector_size = model.vector_size
        # 计算参数量
        params_total = vocab_size * vector_size
        self._logger.info('Number of model parameters: {}'.format(params_total))
        model.wv.save_word2vec_format(self.txt_cache_file)
        model.save(self.model_cache_file)
        node_embedding = np.zeros(shape=(self.num_nodes, self.output_dim), dtype=np.float32)
        for node in range(self.num_nodes):
            for t in range(self.time_slice+1):
                if model.wv.__contains__(str(node)+"_"+str(t)):
                    node_embedding[node] = node_embedding[node]+model.wv[str(node)+"_"+str(t)]
            node_embedding[node] = node_embedding[node]/self.time_slice
        np.save(self.npy_cache_file, node_embedding)
        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：(' + str(len(node_embedding)) + ',' + str(len(node_embedding[0])) + ')')

    def learn_embeddings(self,walks, dimensions, window_size, workers, iters, min_count=1, sg=1, hs=0):
        model = Word2Vec(walks, vector_size=dimensions, window=window_size, min_count=min_count, sg=sg, hs=hs,
            workers=workers, epochs=iters)
        return model

    def simulate_walks(self,adj_mx,num_walks,walk_length):
        """
        Repeatedly simulate random walks from each node.
        """
        walks = []
        self._logger.info('Walk iteration:')
        nodes = [str(i)+"_0" for i in range(self.num_nodes)]
        for walk_iter in range(num_walks):
            self._logger.info(str(walk_iter + 1) + '/' + str(num_walks))
            random.shuffle(nodes)
            for node in tqdm(nodes):
                walks.append(self.random_walk(adj_mx=adj_mx, start_node=node))
        return walks


    def random_walk(self,adj_mx,start_node):
        """
        Simulate a random walk starting from start node.
        节点形式：节点标号_时间片
        """
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_node_id = int(cur.split("_")[0])
            cur_t = len(walk)-1
        #alias_sample
            if cur in self.accept_tables and cur in self.alias_tables:
                next_node_id = int(self.alias_sample(self.accept_tables[cur],self.alias_tables[cur]))
                next_t = cur_t + 1
                walk.append(str(next_node_id)+"_"+str(next_t))
            else:
                norm_prob = adj_mx[cur_t][cur_node_id]
                alias,accept = self.create_alias_table(cur,norm_prob)
                next_node_id = int(self.alias_sample(accept,alias))
                next_t = cur_t + 1
                walk.append(str(next_node_id)+"_"+str(next_t))
        return walk

    def create_alias_table(self,cur_node,norm_prob):
        """
        """
        length = len(norm_prob)
        accept,alias = np.zeros(length),np.zeros(length)
        small,big = [],[]
        transform_N = np.array(norm_prob) * length
        for i ,prob in enumerate(transform_N):
            if prob<1.0:
                small.append(i)
            else:
                big.append(i)
        while small and big:
            small_idx,large_idx= small.pop(),big.pop()
            accept[small_idx] = transform_N[small_idx]
            alias[small_idx] = large_idx
            transform_N[large_idx] = transform_N[large_idx] - (1 - transform_N[small_idx])
            if np.float32(transform_N[large_idx]) < 1.:
                small.append(large_idx)
            else:
                big.append(large_idx)
        while big:
            large_idx = big.pop()
            accept[large_idx] = 1
        while small:
            small_idx = small.pop()
            accept[small_idx] = 1
        self.alias_tables[cur_node] = alias
        self.accept_tables[cur_node] = accept
        return alias,accept


    def alias_sample(self,accept, alias):
        N = len(accept)
        i = int(np.random.random()*N)
        r = np.random.random()
        if r < accept[i]:
            return i
        else:
            return alias[i]






