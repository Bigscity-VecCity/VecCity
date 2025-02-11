import csv
import importlib
import json
import numpy as np
import pandas as pd
from logging import getLogger
from veccity.downstream.abstract_evaluator import AbstractEvaluator


class RoadRepresentationEvaluator(AbstractEvaluator):

    def __init__(self, config, data_feature):
        self._logger = getLogger()
        self.config = config
        self.evaluate_tasks = self.config.get('evaluate_task', ["speed_inference", "travel_time_estimation"])
        self.evaluate_model = self.config.get('evaluate_model', ["SpeedInferenceModel", "TravelTimeEstimationModel"])
        self.result = {}
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = config.get('geo_file', self.dataset)
        self.output_dim = config.get('output_dim', 32)
        self.data_feature = data_feature
        self.embedding_path = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy' \
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

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_uid, type, coordinates, properties(若干列)]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_uids = list(geofile['geo_uid'])
        self.num_nodes = len(self.geo_uids)
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        for index, idx in enumerate(self.geo_uids):
            self.geo_to_ind[idx] = index
            self.ind_to_geo[index] = idx
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_uids)))
        return geofile

    def evaluate(self):
        def add_prefix_to_keys(dictionary, prefix):
            new_dictionary = {}
            for key, value in dictionary.items():
                new_key = prefix + str(key)
                new_dictionary[new_key] = value
            return new_dictionary
        
        def dict_to_csv(dictionary, filename):
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
                writer.writeheader()
                writer.writerow(dictionary)

        road_emb = np.load(self.embedding_path)  # (N, F)
        for task, model in zip(self.evaluate_tasks, self.evaluate_model):
            downstream_model = self.get_downstream_model(model)
            x = road_emb
            label = self.data_feature["label"][task]
            result = downstream_model.run(x, label)
            self.result.update(add_prefix_to_keys(result, task + '_'))
        del self.result['travel_time_estimation_best epoch']
        print(f'Evaluate result: {self.result}')
        self._logger.info(f'Evaluate result: {self.result}')
        result_path = './raw_data/new/evaluate_cache/{}_evaluate_{}_{}_{}.csv'. \
            format(self.exp_id, self.exp_id, self.model, self.dataset, self.output_dim)
        # result_path = './veccity/cache/{}/evaluate_cache/{}_evaluate_{}_{}_{}.csv'. \
        #     format(self.exp_id, self.exp_id, self.model, self.dataset, self.output_dim)
        dict_to_csv(self.result, result_path)
        self._logger.info('Evaluate result is saved at {}'.format(result_path))
        return

        # !这个load_geo必须跟dataset部分相同，也就是得到同样的geo_uid和index的映射，否则就会乱码
        # TODO: 把dataset部分得到的geo_to_ind和ind_to_geo传过来
        rid_file = self._load_geo()
        # 记录每个类别都有哪些geo实体
        result_token = dict()
        for i in range(len(y_predict)):
            kind = int(y_predict[i])
            if kind not in result_token:
                result_token[kind] = []
            result_token[kind].append(self.ind_to_geo[i])
        result_path = './veccity/cache/{}/evaluate_cache/kmeans_category_{}_{}_{}_{}.json'. \
            format(self.exp_id, self.model, self.dataset, str(self.output_dim), str(kinds))
        json.dump(result_token, open(result_path, 'w'))
        self._logger.info('Kmeans category is saved at {}'.format(result_path))

        # QGIS可视化
        rid_type = rid_file['type'][0]
        rid_pos = rid_file['geo_location']
        rid2wkt = dict()
        if rid_type == 'LineString':
            for i in range(rid_pos.shape[0]):
                rid_list = eval(rid_pos[i])  # [(lat1, lon1), (lat2, lon2)...]
                wkt_str = 'LINESTRING('
                for j in range(len(rid_list)):
                    rid = rid_list[j]
                    wkt_str += (str(rid[0]) + ' ' + str(rid[1]))
                    if j != len(rid_list) - 1:
                        wkt_str += ','
                wkt_str += ')'
                rid2wkt[i] = wkt_str
        elif rid_type == 'Point':
            for i in range(rid_pos.shape[0]):
                rid_list = eval(rid_pos[i])  # [lat1, lon1]
                wkt_str = 'Point({} {})'.format(rid_list[0], rid_list[1])
                rid2wkt[i] = wkt_str
        else:
            raise ValueError('Error geo type!')

        df = []
        for i in range(len(y_predict)):
            df.append([i, self.ind_to_geo[i], y_predict[i], rid2wkt[i]])
        df = pd.DataFrame(df)
        df.columns = ['id', 'rid', 'class', 'wkt']
        df = df.sort_values(by='class')
        result_path = './veccity/cache/{}/evaluate_cache/kmeans_qgis_{}_{}_{}_{}.csv'. \
            format(self.exp_id, self.model, self.dataset, str(self.output_dim), str(kinds))
        df.to_csv(result_path, index=False)
        self._logger.info('Kmeans result for QGIS is saved at {}'.format(result_path))

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        pass
