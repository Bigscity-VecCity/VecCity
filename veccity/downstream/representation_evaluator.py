import json
import numpy as np
from logging import getLogger
import importlib

from veccity.downstream.abstract_evaluator import AbstractEvaluator


class RepresentationEvaluator(AbstractEvaluator):

    def __init__(self, config, data_feature):
        self._logger = getLogger()
        self.config = config
        self.evaluate_tasks = self.config.get('evaluate_tasks', ["function_cluster"])
        self.evaluate_model = self.config.get('evaluate_model', ["KmeansModel"])
        self.all_result = []
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.output_dim = config.get('output_dim', 32)
        self.data_path = './raw_data/' + self.dataset + '/'
        self.data_feature = data_feature
        self.embedding_path = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy' \
            .format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.result_path = './veccity/cache/{}/evaluate_cache/result_{}_{}_{}.json' \
            .format(self.exp_id, self.model, self.dataset, self.output_dim)

    def get_downstream_model(self, model):
        try:
            return getattr(importlib.import_module('veccity.downstream.downstream_models'), model)(self.config)
        except AttributeError:
            raise AttributeError('evaluate model is not found')

    def collect(self, batch):
        pass

    def evaluate(self):
        node_emb = np.load(self.embedding_path)  # (N, F)
        for task, model in zip(self.evaluate_tasks, self.evaluate_model):
            downstream_model = self.get_downstream_model(model)
            label = self.data_feature["label"][task]
            x = node_emb
            if task == "od_matrix_predict":
                # 将起止点的向量拼接作为x
                l = []
                num_node = node_emb.shape[0]
                for i in range(num_node):
                    for j in range(num_node):
                        o_embs = node_emb[i]
                        d_embs = node_emb[j]
                        l.append(np.concatenate((o_embs, d_embs), axis=0))
                x = np.stack(l, axis=0)
            result = downstream_model.run(x, label)
            result = {task: result}
            self.all_result.append(result)

        self._logger.info('result {}'.format(self.all_result))
        self.save_result()

    def save_result(self, filename=None):
        json.dump(self.all_result, open(self.result_path, 'w'))
        self._logger.info('result save in {}'.format(self.result_path))

    def clear(self):
        pass
