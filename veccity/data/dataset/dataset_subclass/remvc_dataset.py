import os
import pickle
import multiprocessing
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from veccity.data.dataset.traffic_representation_dataset import TrafficRepresentationDataset
from veccity.data.preprocess import cache_dir, preprocess_all
from veccity.utils import ensure_dir, need_train


def gen_index_map(df, column, offset=0):
    index_map = {origin: index
                 for index, origin in enumerate(df[column].drop_duplicates())}
    return index_map


class ReMVCDataset(TrafficRepresentationDataset):
    def __init__(self, config):
        self.config = config
        preprocess_all(config)
        if not need_train(config):
            return
        super().__init__(config)
        self.data_cache_file = f'./veccity/cache/dataset_cache/{self.dataset}/ReMVC'
        ensure_dir(self.data_cache_file)
        self.get_region_dict()
        self.get_poi_features()
        self.get_matrix_dict()
        self.num_processes = min(200, self.num_regions // 10)
        self.get_model_flow()
        self.get_model_poi()

    def get_region_dict(self):
        self._logger.info('Start get region dict...')
        poi_dict = {}
        poi_df = self.geofile[self.geofile['traffic_type'] == 'poi']
        poi_map = gen_index_map(poi_df, self.config.get('poi_type_name'))
        self.num_poi_types = len(poi_map)
        region_dict_path = self.data_cache_file + 'remvc_region_dict.pkl'
        if not os.path.exists(region_dict_path):
            region_dict = {}
            time_slices_num = self.config.get('time_slices_num', 48)
            assert 86400 % time_slices_num == 0
            for i in range(self.num_regions):
                region_dict[i] = {
                    'poi': [],
                    'pickup_matrix': np.array([[0] * self.num_regions for _ in range(time_slices_num)]),
                    'dropoff_matrix': np.array([[0] * self.num_regions for _ in range(time_slices_num)])
                }

            # poi
            for _, row in poi_df.iterrows():
                poi_dict[row['geo_id']] = poi_map[row[self.config.get('poi_type_name')]]
            for _, row in self.region2poi.iterrows():
                region_id = row['origin_id']
                poi_id = row['destination_id']
                region_dict[region_id]['poi'].append(poi_dict[poi_id])

            # matrix
            od_df = pd.read_csv(os.path.join(cache_dir, self.dataset, 'od_region_train.csv'))
            for _, row in od_df.iterrows():
                origin = int(row['origin_id'])
                destination = int(row['destination_id'])
                o_time_slice = (int(row['start_time']) % 86400) // (86400 // time_slices_num)
                d_time_slice = (int(row['end_time']) % 86400) // (86400 // time_slices_num)
                region_dict[origin]['pickup_matrix'][o_time_slice][destination] += 1
                region_dict[destination]['dropoff_matrix'][d_time_slice][origin] += 1
            for i in range(self.num_regions):
                region_dict[i]['pickup_matrix'] = csr_matrix(region_dict[i]['pickup_matrix'])
                region_dict[i]['dropoff_matrix'] = csr_matrix(region_dict[i]['dropoff_matrix'])

            with open(region_dict_path, 'wb') as f:
                pickle.dump(region_dict, f)

        with open(region_dict_path, 'rb') as f:
            self.region_dict = pickle.load(f)
        self._logger.info('Finish get region dict.')

    def get_poi_features(self):
        self._logger.info('Start get poi features...')
        poi_features = {}
        for i in range(self.num_regions):
            poi_features[i] = np.zeros(self.num_poi_types)
            for j in self.region_dict[i]['poi']:
                poi_features[i][j] += 1
        self.poi_features = poi_features
        self._logger.info('Finish get poi features.')

    def process_model_flow(self, args):
        idx, all = args
        model_flow_path = os.path.join(self.data_cache_file, 'model_flow', 'remvc_model_flow_{}.pkl'.format(idx))
        num = (self.num_regions + all - 1) // all
        start = num * idx
        end = min(num * (idx + 1), self.num_regions)
        model_flow = {}
        # self._logger.info('Process {} {} ~ {}'.format(idx, start, end))
        for i in range(start, end):
            ll = 0
            model_flow[i] = np.zeros(self.num_regions - 1)
            for j in range(self.num_regions):
                if i != j:
                    model_flow[i][ll] = \
                        np.sqrt(np.sum((self.region_dict[i]['pickup_matrix'].astype(float).toarray().flatten() -
                                        self.region_dict[j]['pickup_matrix'].astype(float).toarray().flatten()) ** 2)) + \
                        np.sqrt(np.sum((self.region_dict[i]['dropoff_matrix'].astype(float).toarray().flatten() -
                                        self.region_dict[j]['dropoff_matrix'].astype(float).toarray().flatten()) ** 2))
                    ll += 1
            model_flow[i] = model_flow[i] / np.sum(model_flow[i])
            self._logger.info('Process {} Finish {}.'.format(idx, i - start + 1))
        with open(model_flow_path, 'wb') as f:
                pickle.dump(model_flow, f)
        self._logger.info('Process {} End.'.format(idx))

    def get_model_flow(self):
        self._logger.info('Start get model flow...')
        model_flow_path = os.path.join(self.data_cache_file, 'model_flow')
        all = self.num_processes
        ensure_dir(model_flow_path)
        pool = multiprocessing.Pool(all + 5)
        tmp = []
        for i in range(all):
            model_flow_path = os.path.join(self.data_cache_file, 'model_flow', 'remvc_model_flow_{}.pkl'.format(i))
            if not os.path.exists(model_flow_path):
                tmp.append(i)
        pool.map(self.process_model_flow, [(item, all) for item in tmp])
        pool.close()
        pool.join()
        self.model_flow = {}
        for i in range(all):
            model_flow_path = os.path.join(self.data_cache_file, 'model_flow', 'remvc_model_flow_{}.pkl'.format(i))
            with open(model_flow_path, 'rb') as f:
                tmp_dict = pickle.load(f)
                self.model_flow.update(tmp_dict)
        self._logger.info('Finish get model flow.')
    
    def get_model_poi(self):
        self._logger.info('Start get model poi...')
        poi_features = {}
        for i in range(self.num_regions):
            poi_features[i] = np.zeros(self.num_poi_types)
            for j in self.region_dict[i]['poi']:
                poi_features[i][j] += 1
            poi_features[i] /= np.sum(poi_features[i])
            where_are_NaNs = np.isnan(poi_features[i])
            poi_features[i][where_are_NaNs] = 0
        model_poi = {}
        for i in range(self.num_regions):
            model_poi[i] = [1.0 / (self.num_regions - 1)] * (self.num_regions - 1)
            ll = 0
            model_poi[i] = np.zeros(self.num_regions - 1)
            for j in range(self.num_regions):
                if i != j:
                    model_poi[i][ll] = np.sqrt(np.sum((poi_features[i] - poi_features[j]) ** 2))
                    ll += 1
            model_poi[i] = model_poi[i] / np.sum(model_poi[i])
        self.model_poi = model_poi
        self._logger.info('Finish get model poi.')

    def get_matrix_dict(self):
        self._logger.info('Start get matrix dict...')
        matrix_dict_path = self.data_cache_file + 'remvc_matrix_dict.pkl'
        if not os.path.exists(matrix_dict_path):
            matrix_dict = {}
            for idx in range(self.num_regions):
                pickup_matrix = self.region_dict[idx]["pickup_matrix"].astype(float).toarray()
                dropoff_matrix = self.region_dict[idx]["dropoff_matrix"].astype(float).toarray()

                pickup_matrix = pickup_matrix / pickup_matrix.sum()
                where_are_NaNs = np.isnan(pickup_matrix)
                pickup_matrix[where_are_NaNs] = 0

                dropoff_matrix = dropoff_matrix / dropoff_matrix.sum()
                where_are_NaNs = np.isnan(dropoff_matrix)
                dropoff_matrix[where_are_NaNs] = 0

                matrix_dict[idx] = csr_matrix(pickup_matrix), csr_matrix(dropoff_matrix)
                self.region_dict[idx]["pickup_matrix"] = csr_matrix(pickup_matrix)
                self.region_dict[idx]["dropoff_matrix"] = csr_matrix(dropoff_matrix)
            with open(matrix_dict_path, 'wb') as f:
                pickle.dump(matrix_dict, f)

        with open(matrix_dict_path, 'rb') as f:
            self.matrix_dict = pickle.load(f)
        self._logger.info('Finish get matrix dict.')

    def get_data(self):
        return None, None, None

    def get_data_feature(self):
        if not need_train(self.config):
            return {}
        function = np.zeros(self.num_regions)
        region_df = self.geofile[self.geofile['traffic_type'] == 'region']
        for i, row in region_df.iterrows():
            function[i] = row[self.config.get('poi_type_name')]
        return {
            'region_dict': self.region_dict,
            'matrix_dict': self.matrix_dict,
            'model_flow': self.model_flow,
            'model_poi': self.model_poi,
            'sampling_pool': [i for i in range(self.num_regions)],
            'num_pois': self.num_pois,
            'num_poi_types': self.num_poi_types,
            'num_regions': self.num_regions,
            'label': {
                'function_cluster': function
            }
        }
