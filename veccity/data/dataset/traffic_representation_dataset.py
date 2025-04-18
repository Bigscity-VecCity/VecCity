import os
from logging import getLogger
import geopandas as gpd
import numpy as np
import pandas as pd
from veccity.data.dataset import AbstractDataset
from veccity.utils import ensure_dir
from veccity.utils import geojson2geometry
from tqdm import tqdm
import copy
from veccity.data.preprocess import preprocess_all, cache_dir

class TrafficRepresentationDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        preprocess_all(config)
        self.dataset = self.config.get('dataset', '')
        self.batch_size = self.config.get('batch_size', 64)
        self.cache_dataset = self.config.get('cache_dataset', True)
        self.num_workers = self.config.get('num_workers', 0)
        self.pad_with_last_sample = self.config.get('pad_with_last_sample', True)
        self.train_rate = self.config.get('train_rate', 0.7)
        self.eval_rate = self.config.get('eval_rate', 0.1)
        self.scaler_type = self.config.get('scaler', 'none')
        self.ext_scaler_type = self.config.get('ext_scaler', 'none')
        self.load_external = self.config.get('load_external', False)
        self.normal_external = self.config.get('normal_external', False)
        self.add_time_in_day = self.config.get('add_time_in_day', False)
        self.add_day_in_week = self.config.get('add_day_in_week', False)
        self.representation_object = self.config.get('representation_object', 'region')  #####
        self.input_window = self.config.get('input_window', 12)
        self.output_window = self.config.get('output_window', 12)
        self.region_geometry = None
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.input_window) + '_' + str(self.output_window) + '_' \
            + str(self.train_rate) + '_' + str(self.eval_rate) + '_' + str(self.scaler_type) + '_' \
            + str(self.batch_size) + '_' + str(self.load_external) + '_' + str(self.add_time_in_day) + '_' \
            + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample)
        self.cache_file_name = os.path.join('./veccity/cache/dataset_cache/',
                                            'traffic_representation_{}.npz'.format(self.parameters_str))
        self.cache_file_folder = './veccity/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))
        ensure_dir('./veccity/cache/dataset_cache/{}'.format(self.dataset))
        # 加载数据集的config.json文件
        self.weight_col = self.config.get('weight_col', '')
        self.data_col = self.config.get('data_col', '')
        self.ext_col = self.config.get('ext_col', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.data_files = self.config.get('data_files', self.dataset)
        self.ext_file = self.config.get('ext_file', self.dataset)
        self.output_dim = self.config.get('output_dim', 1)
        self.time_intervals = self.config.get('time_intervals', 300)  # s
        self.init_weight_inf_or_zero = self.config.get('init_weight_inf_or_zero', 'inf')
        self.set_weight_link_or_dist = self.config.get('set_weight_link_or_dist', 'dist')
        self.bidir_adj_mx = self.config.get('bidir_adj_mx', False)
        self.calculate_weight_adj = self.config.get('calculate_weight_adj', False)
        self.weight_adj_epsilon = self.config.get('weight_adj_epsilon', 0.1)
        self.distance_inverse = self.config.get('distance_inverse', False)
        self.remove_node_type = self.config.get("remove_node_type", None)
        self.ind_to_geo_path = './veccity/cache/dataset_cache/{}/ind_to_geo_{}.npy'.format(self.dataset,
                                                                                           self.remove_node_type)
        # self.func_label_path = './veccity/cache/dataset_cache/{}/func_label_{}.npy'.format(self.dataset,
        #                                                                                    self.remove_node_type)
        # 初始化
        self.data = None
        self.feature_name = {'X': 'float', 'y': 'float'}  # 此类的输入只有X和y
        self.adj_mx = None
        self.scaler = None
        self.ext_scaler = None
        self.feature_dim = 0
        self.ext_dim = 0
        self.num_nodes = 0
        self.num_batches = 0
        self._logger = getLogger()
        self.num_regions = 0
        self.num_roads = 0
        self.num_pois = 0
        self.traj_region = []
        self.traj_road = []
        self.traj_time = []
        self.road2region = None
        self.region2road = None
        self.poi2region = None
        self.region2poi = None
        self.poi2road = None
        self.road2poi = None
        if os.path.exists(self.data_path + self.geo_file + '.geo'):
            self._load_geo()
        else:
            raise ValueError('Not found .geo file!')
        if os.path.exists(self.data_path + self.rel_file + '.grel'):  # .rel file is not necessary
            self._load_rel()
        else:
            self.adj_mx = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        if self.representation_object == "region":
            if self.remove_node_type == "traj":
                self.remove_0_degree_nodes()
            elif self.remove_node_type == "od":
                self.keep_od_nodes()
            elif self.remove_node_type == "non_zero":
                self.keep_non_zero_od_nodes()
            elif self.remove_node_type == "MVURE":
                self.keep_mvure_nodes()
            else:
                geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
                # self.function = list(geofile[geofile['traffic_type'] == 'region']['region_FUNCTION'])
        elif self.representation_object == "road":
            self.geo_to_ind = {}
            self.ind_to_geo = {}
            for index, idx in enumerate(self.geo_uids):
                self.geo_to_ind[idx] = index
                self.ind_to_geo[index] = idx

            geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
            self.geofile = geofile
            self.num_nodes = self.num_roads
            # self.function = list(geofile[geofile['traffic_type'] == 'road']['function'])

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_uid, type, coordinates, function,traffic_type]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geofile = geofile
        coordinates_list = geofile[geofile['traffic_type'] == 'region']['region_geometry']
        # l = coordinates_list if coordinates_list[0][0].isalpha() else [geojson2geometry(coordinate) for coordinate in coordinates_list]
        self.region_geometry = gpd.GeoSeries.from_wkt(coordinates_list)
        self.geo_uids = list(geofile['geo_uid'])
        self.region_ids = list(geofile[geofile['traffic_type'] == 'region']['geo_uid'])
        self.num_regions = len(self.region_ids)
        self.road_ids = list(geofile[geofile['traffic_type'] == 'road']['geo_uid'])
        self.num_roads = len(self.road_ids)
        self.poi_ids = list(geofile[geofile['traffic_type'] == 'poi']['geo_uid'])
        self.num_pois = len(self.poi_ids)
        if self.representation_object == "region":
            self.num_nodes = self.num_regions
        elif self.representation_object == "road":
            self.num_nodes = self.num_roads
        else:
            self.num_nodes = self.num_pois
        self._logger.info(
            "Loaded file " + self.geo_file + '.geo' + ',num_regions=' + str(self.num_regions) + ',num_roads=' + str(
                self.num_roads) + ',num_pois=' + str(self.num_pois) + ', num_nodes=' + str(self.num_nodes))

    def _load_rel(self):
        """
        加载各个实体的联系，格式['rel_uid','type','orig_geo_id','dest_geo_id','rel_type']
        后续可能会将两种实体之间的对应做成1-->n的映射
        """
        relfile = pd.read_csv(self.data_path + self.rel_file + '.grel')
        self.road2region = relfile[relfile['rel_type'] == 'road2region']
        self.region2road = relfile[relfile['rel_type'] == 'region2road']
        self.poi2region = relfile[relfile['rel_type'] == 'poi2region']
        self.region2poi = relfile[relfile['rel_type'] == 'region2poi']
        self.poi2road = relfile[relfile['rel_type'] == 'poi2road']
        self.road2poi = relfile[relfile['rel_type'] == 'road2poi']
        self._logger.info("Loaded file " + self.rel_file + '.grel')

    def remove_0_degree_nodes(self):
        # dict保存为
        if os.path.exists(self.ind_to_geo_path): # and os.path.exists(self.func_label_path)
            region_list = np.load(self.ind_to_geo_path)
            self.geo_to_ind = {}
            self.ind_to_geo = {}
            index = 0
            for region in region_list:
                self.geo_to_ind[region] = index
                self.ind_to_geo[index] = region
                index += 1
            self.num_nodes = index
            self.num_regions = index
            # self.function = np.load(self.func_label_path)
            self._logger.info("remove 0 degree nodes,num_nodes = {},num_regions = {}".format(index, index))
            return
        # 去除在轨迹中不出现的节点，使用geo_to_ind,ind_to_geo来对节点重新编号
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        region_set = set()  # 记录所有在轨迹中出现过的区域
        for road_list in self.traj_road:
            for road in road_list:
                region = list(self.road2region[self.road2region['orig_geo_id'] == road]['dest_geo_id'])[0]
                region_set.add(region)
        index = 0
        for region in region_set:
            self.geo_to_ind[region] = index
            self.ind_to_geo[index] = region
            index += 1
        self.num_nodes = index
        self.num_regions = index
        # 对 function_label进行映射
        # self.function = np.zeros(self.num_regions)
        # for i in range(index):
        #     self.function[i] = self.geofile.loc[self.ind_to_geo[i], "function"]
        region_list = list(region_set)
        region_array = np.array(region_list)
        np.save(self.ind_to_geo_path, region_array)
        # np.save(self.func_label_path, self.function)
        self._logger.info("remove 0 degree nodes,num_nodes = {},num_regions = {}".format(index, index))

    def keep_od_nodes(self):
        # dict保存为
        if os.path.exists(self.ind_to_geo_path): # and os.path.exists(self.func_label_path):
            region_list = np.load(self.ind_to_geo_path)
            self.geo_to_ind = {}
            self.ind_to_geo = {}
            index = 0
            for region in region_list:
                self.geo_to_ind[region] = index
                self.ind_to_geo[index] = region
                index += 1
            self.num_nodes = index
            self.num_regions = index
            # self.function = np.load(self.func_label_path)
            self._logger.info("remove 0 degree nodes,num_nodes = {},num_regions = {}".format(index, index))
            return
        # 去除在od矩阵中为0的点，使用geo_to_ind,ind_to_geo来对节点重新编号
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        region_set = set()  # 记录所有在od中出现过的区域
        for road_list in self.traj_road:
            road = road_list[0]
            region = list(self.road2region[self.road2region['orig_geo_id'] == road]['dest_geo_id'])[0]
            region_set.add(region)
            road = road_list[-1]
            region = list(self.road2region[self.road2region['orig_geo_id'] == road]['dest_geo_id'])[0]
            region_set.add(region)
        index = 0
        for region in region_set:
            self.geo_to_ind[region] = index
            self.ind_to_geo[index] = region
            index += 1
        self.num_nodes = index
        self.num_regions = index
        # 对 function_label进行映射
        # self.function = np.zeros(self.num_regions)
        # for i in range(index):
        #     self.function[i] = self.geofile.loc[self.ind_to_geo[i], "function"]
        region_list = list(region_set)
        region_array = np.array(region_list)
        np.save(self.ind_to_geo_path, region_array)
        # np.save(self.func_label_path, self.function)
        self._logger.info("remove 0 degree nodes,num_nodes = {},num_regions = {}".format(index, index))

    def keep_non_zero_od_nodes(self):
        # dict保存为
        if os.path.exists(self.ind_to_geo_path): # and os.path.exists(self.func_label_path):
            region_list = np.load(self.ind_to_geo_path)
            self.geo_to_ind = {}
            self.ind_to_geo = {}
            index = 0
            for region in region_list:
                self.geo_to_ind[region] = index
                self.ind_to_geo[index] = region
                index += 1
            self.num_nodes = index
            self.num_regions = index
            # self.function = np.load(self.func_label_path)
            self._logger.info("remove 0 degree nodes,num_nodes = {},num_regions = {}".format(index, index))
            return
        # 去除在od矩阵中为0的点，使用geo_to_ind,ind_to_geo来对节点重新编号
        # 计算od矩阵
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        od_label = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        if self.representation_object == 'road':
            for road_list in self.traj_road:
                origin = road_list[0]
                destination = road_list[-1]
                od_label[origin][destination] += 1
        elif self.representation_object == 'region':
            for region_list in self.traj_region:
                origin = region_list[0]
                destination = region_list[-1]
                od_label[origin][destination] += 1

        # 首先筛除没有自己到自己的轨迹的区域，这部分区域一定不在最后的结果中
        final_vertex = []
        for i in range(self.num_nodes):
            if od_label[i][i] != 0:
                final_vertex.append(i)
        # 得到第一步筛除后的od矩阵
        screened_od = np.zeros((len(final_vertex), len(final_vertex)), dtype=int)
        for indi, vi in enumerate(final_vertex):
            for indj, vj in enumerate(final_vertex):
                screened_od[indi][indj] = int(od_label[vi][vj])

        #
        # 这里可以看看有没有别的算法，保留的区域会更多
        del_zero = True
        while del_zero:
            del_zero = False
            row_col_zero_num = np.zeros(len(final_vertex), dtype=int)
            for i in range(len(final_vertex)):
                row_col_zero_num[i] += sum(screened_od[i] == 0)
                row_col_zero_num[i] += sum(screened_od[:, i] == 0)

            max_ind = np.argmax(row_col_zero_num)
            if row_col_zero_num[max_ind] > 0:
                screened_od = np.delete(screened_od, max_ind, axis=0)
                screened_od = np.delete(screened_od, max_ind, axis=1)
                final_vertex.pop(max_ind)
                del_zero = True

        region_set = set(final_vertex)

        index = 0
        for region in region_set:
            self.geo_to_ind[region] = index
            self.ind_to_geo[index] = region
            index += 1
        self.num_nodes = index
        self.num_regions = index
        # 对 function_label进行映射
        # self.function = np.zeros(self.num_regions)
        # for i in range(index):
        #     self.function[i] = self.geofile.loc[self.ind_to_geo[i], "function"]
        region_list = list(region_set)
        region_array = np.array(region_list)
        np.save(self.ind_to_geo_path, region_array)
        # np.save(self.func_label_path, self.function)
        self._logger.info("remove 0 degree nodes,num_nodes = {},num_regions = {}".format(index, index))

    def keep_mvure_nodes(self):
        # dict保存为
        if os.path.exists(self.ind_to_geo_path): # and os.path.exists(self.func_label_path):
            region_list = np.load(self.ind_to_geo_path)
            self.geo_to_ind = {}
            self.ind_to_geo = {}
            index = 0
            for region in region_list:
                self.geo_to_ind[region] = index
                self.ind_to_geo[index] = region
                index += 1
            self.num_nodes = index
            self.num_regions = index
            # self.function = np.load(self.func_label_path)
            self._logger.info("remove 0 degree nodes,num_nodes = {},num_regions = {}".format(index, index))
            return
        # 去除在od矩阵中为0的点，使用geo_to_ind,ind_to_geo来对节点重新编号
        # 计算全量的od矩阵
        od_label = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for traj in self.traj_road:
            origin_region = list(self.road2region[self.road2region['orig_geo_id'] == traj[0]]['dest_geo_id'])[0]
            destination_region = list(self.road2region[self.road2region['orig_geo_id'] == traj[-1]]['dest_geo_id'])[0]
            od_label[origin_region][destination_region] += 1
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        poi_set = set()
        for region in range(self.num_regions):
            if len(list(self.region2poi[self.region2poi["orig_geo_id"] == region]["dest_geo_id"])) > 0:
                poi_set.add(region)
        region_set = copy.copy(poi_set)
        stop = False
        while not stop:
            stop = True
            for region in region_set.copy():
                if (sum(od_label[region, :]) == 0) or (sum(od_label[:, region]) == 0):
                    region_set.remove(region)
                    # 修改od_label
                    od_label[region, :] = 0
                    od_label[:, region] = 0
                    stop = False
        index = 0
        for region in region_set:
            self.geo_to_ind[region] = index
            self.ind_to_geo[index] = region
            index += 1
        self.num_nodes = index
        self.num_regions = index
        # 对 function_label进行映射
        # self.function = np.zeros(self.num_regions)
        # for i in range(index):
        #     self.function[i] = self.geofile.loc[self.ind_to_geo[i], "function"]
        region_array = np.array([region for region in region_set])
        np.save(self.ind_to_geo_path, region_array)
        # np.save(self.func_label_path, self.function)
        self._logger.info("remove 0 degree nodes,num_nodes = {},num_regions = {}".format(index, index))
