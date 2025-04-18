from veccity.data.dataset import AbstractDataset
import os
from logging import getLogger
import geopandas as gpd
import numpy as np
import pandas as pd
from veccity.utils import ensure_dir, geojson2geometry
from veccity.data.preprocess import preprocess_all, cache_dir


class ODRegionRepresentationDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        preprocess_all(config)
        self.dataset = self.config.get('dataset', '')
        self.batch_size = self.config.get('batch_size', 64)
        self.cache_dataset = self.config.get('cache_dataset', True)
        self.pad_with_last_sample = self.config.get('pad_with_last_sample', True)
        self.train_rate = self.config.get('train_rate', 0.7)
        self.eval_rate = self.config.get('eval_rate', 0.1)
        self.scaler_type = self.config.get('scaler', 'none')
        self.ext_scaler_type = self.config.get('ext_scaler', 'none')
        self.load_external = self.config.get('load_external', False)
        self.normal_external = self.config.get('normal_external', False)
        self.add_time_in_day = self.config.get('add_time_in_day', False)
        self.add_day_in_week = self.config.get('add_day_in_week', False)
        self.representation_object = self.config.get('representation_object', 'region')
        self.input_window = self.config.get('input_window', 12)
        self.output_window = self.config.get('output_window', 12)
        self.region_geometry = None
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.input_window) + '_' + str(self.output_window) + '_' \
            + str(self.train_rate) + '_' + str(self.eval_rate) + '_' + str(self.scaler_type) + '_' \
            + str(self.batch_size) + '_' + str(self.load_external) + '_' + str(self.add_time_in_day) + '_' \
            + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample)
        self.cache_file_name = os.path.join('./veccity/cache/dataset_cache/',
                                            'od_region_representation_{}.npz'.format(self.parameters_str))
        self.cache_file_folder = './veccity/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))
        ensure_dir('./veccity/cache/dataset_cache/{}'.format(self.dataset))
        self.od_label_path = './veccity/cache/dataset_cache/{}/od_mx.npy'.format(self.dataset)
        # 加载数据集的config.json文件
        self.weight_col = self.config.get('weight_col', '')
        self.data_col = self.config.get('data_col', '')
        self.ext_col = self.config.get('ext_col', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.od_file = self.config.get('od_file', self.dataset)
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
        self.remove_node_type = self.config.get("remove_node_type", "od")
        self.ind_to_geo_path = './veccity/cache/dataset_cache/{}/ind_to_geo_{}.npy'.format(self.dataset,
                                                                                           self.remove_node_type)

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
        self.traj_road = []
        self.traj_region = []
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
        self.od_label_path = os.path.join(cache_dir, self.dataset, 'od_region_train_od.npy')
        self.od_label = np.load(self.od_label_path)

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_uid,type,geometry,crime_count,traffic_type,venue_category_name]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geofile = geofile
        coordinates_list = geofile[geofile['traffic_type'] == 'region']['geo_location']
        l = coordinates_list if coordinates_list[0][0].isalpha() else [geojson2geometry(coordinate) for coordinate in coordinates_list]
        self.region_geometry = gpd.GeoSeries.from_wkt(l)
        self.centroid = self.region_geometry.centroid
        self.geo_uids = list(geofile['geo_uid'])
        self.region_ids = list(geofile[geofile['traffic_type'] == 'region']['geo_uid'])
        self.num_regions = len(self.region_ids)
        self.poi_ids = list(geofile[geofile['traffic_type'] == 'poi']['geo_uid'])
        self.num_pois = len(self.poi_ids)
        self.num_nodes = self.num_regions
        self._logger.info(
            "Loaded file " + self.geo_file + '.geo' + ',num_regions=' + str(self.num_regions) + ',num_roads=' + str(
                self.num_roads) + ',num_pois=' + str(self.num_pois) + ', num_nodes=' + str(self.num_nodes))
        return

    def _load_rel(self):
        """
        加载各个实体的联系，[rel_uid,type,orig_geo_id,dest_geo_id,rel_type]
        将region2poi和poi2region保存
        把region2region直接做成self.adj_mx，邻接的保存为质心距离，不邻接的保存为0
        """
        relfile = pd.read_csv(self.data_path + self.rel_file + '.grel')
        self.poi2region = relfile[relfile['rel_type'] == 'poi2region']
        self.region2poi = relfile[relfile['rel_type'] == 'region2poi']
        self.region2region = relfile[relfile['rel_type'] == 'region2region']
        if self.region2region.shape[0]==0:
            n=len(self.centroid)
            tempa=[]
            tempb=[]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        tempa.append(i)
                        tempb.append(j)

            data={"orig_geo_id":tempa,"dest_geo_id":tempb}
            self.region2region=pd.DataFrame(data)
        self.region2region = self.region2region.reset_index(drop=True)
        self.adj_mx = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i in range(len(self.region2region)):
            origin_region_id = self.region2region.loc[i, "orig_geo_id"]
            destination_region_id = self.region2region.loc[i, "dest_geo_id"]
            distance = self.centroid[origin_region_id].distance(self.centroid[destination_region_id])
            self.adj_mx[origin_region_id][destination_region_id] = distance
        """
        加载各个实体的联系，格式['rel_uid','type','orig_geo_id','dest_geo_id','rel_type']
        后续可能会将两种实体之间的对应做成1-->n的映射
        """
        self.road2region = relfile[relfile['rel_type'] == 'road2region']
        self.region2road = relfile[relfile['rel_type'] == 'region2road']
        self.poi2road = relfile[relfile['rel_type'] == 'poi2road']
        self.road2poi = relfile[relfile['rel_type'] == 'road2poi']
        self._logger.info("Loaded file " + self.rel_file + '.grel')
