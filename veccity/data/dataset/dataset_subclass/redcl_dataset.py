import math
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pdb
import pickle as pkl
import random
from logging import getLogger

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import timm
import torch
import torch.nn.functional as F
import torch.utils.data
from matplotlib import pyplot as plt
from pyproj import Transformer
from scipy.spatial import KDTree
from shapely import affinity
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from veccity.data.dataset.abstract_dataset import AbstractDataset
from veccity.data.preprocess import cache_dir, preprocess_all
from veccity.utils import ensure_dir, need_train


class ReDCLDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        preprocess_all(config)
        self._logger = getLogger()
        self.dataset = self.config.get('dataset', '')
        self.cache = self.config.get('cache', True)
        data_path = f'raw_data/{self.dataset}'
        cache_path = f'./veccity/cache/dataset_cache/{self.dataset}/ReDCL'
        ensure_dir(cache_path)
        radius = config.get('radius', 200)
        preprocessor = Preprocess(config)
        building, poi = preprocessor.get_building_and_poi()
        random_point = preprocessor.poisson_disk_sampling(building, poi, radius)
        preprocessor.rasterize_buildings(building)
        preprocessor.partition(building, poi, random_point, radius)
        self._logger.info(f'Random Points: {len(random_point)}')
        train_unsupervised(config)
        self._logger.info('start get CityData.')
        self.city_data = CityData(self.config)
        self._logger.info('end get CityData.')

    def get_data(self):
        return None, None, None

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        if not need_train(self.config):
            return {}
        return {
            'city_data': self.city_data
        }


class CityData(object):
    def __init__(self, config, random_radius=100, with_type=True, with_random=True, cached_region_path=None,
                 cached_grid_path=None):
        with_random = not config.get('no_random', False)
        random_radius = config.get('radius', 200)
        city = config.get('dataset')
        self.city = city
        use_cache = config.get('cache', True)
        self._logger = getLogger()
        # Try to load cached data
        in_path = os.path.join(f'./veccity/cache/dataset_cache/{city}/ReDCL', 'processed')
        # Create cache path
        cache_path = os.path.join(f'./veccity/cache/dataset_cache/{city}/ReDCL', 'cache')
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        cached_pattern_path = os.path.join(cache_path, f'pattern_{random_radius}_' + (
            'with_type' if with_type else '') + ('_no_random' if not with_random else '') + '.pkl')
        if use_cache and os.path.exists(cached_pattern_path):
            with open(cached_pattern_path, 'rb') as f:
                self.patterns = pkl.load(f)
                self.building_feature_dim = self.patterns[0]['building_feature'].shape[1]
                for pattern in self.patterns:
                    if pattern['poi_feature'] is not None:
                        self.poi_feature_dim = pattern['poi_feature'].shape[1]
                        break
        else:
            # process pattern data
            with open(os.path.join(in_path, 'building.pkl'), 'rb') as f:
                buildings = pkl.load(f)
            self.building_shape_feature = np.load(os.path.join(in_path, 'building_features.npy'))
            self.building_rotation = np.load(os.path.join(in_path, 'building_rotation.npz'))['arr_0']
            with open(os.path.join(in_path, 'random_point_' + str(random_radius) + 'm.pkl'), 'rb') as f:
                self.random_points = pkl.load(f)
            # Poi outside buildings
            with open(os.path.join(in_path, 'poi.pkl'), 'rb') as f:
                pois = pkl.load(f)
            self.building_feature_dim = 1 + 2 + self.building_shape_feature.shape[1] + len(buildings[0]['poi'])
            if with_type:
                self.building_feature_dim += len(buildings[0]['onehot'])
                random_path = os.path.join(in_path, 'random_point_with_type.npy')
            else:
                random_path = os.path.join(in_path, 'random_point.npy')
            if os.path.exists(random_path):
                self.random_feature = np.load(random_path)
            if not os.path.exists(random_path) or self.random_feature.shape[1] != self.building_feature_dim:
                self.random_feature = np.random.randn(1, self.building_feature_dim)
                np.save(random_path, self.random_feature)

            self.poi_feature_dim = len(pois[0]['onehot'])

            self.patterns = []
            # Road network segmentation
            self._logger.info('Pre-calculating pattern features...')
            with open(os.path.join(in_path, f'segmentation/{random_radius}.pkl'), 'rb') as f:
                raw_patterns = pkl.load(f)
            for pattern in tqdm(raw_patterns):
                if with_random:
                    building_num = len(pattern['building']) + len(pattern['random_point'])
                else:
                    building_num = len(pattern['building'])
                building_feature = np.zeros((building_num, self.building_feature_dim))
                for row, idx in enumerate(pattern['building']):
                    building_feature[row, 0] = buildings[idx]['shape'].area
                    building_feature[row, 1:3] = self.building_rotation[idx]
                    building_feature[row, 3:3 + self.building_shape_feature.shape[1]] = \
                        self.building_shape_feature[idx]
                    building_feature[row,
                    3 + self.building_shape_feature.shape[1]: 3 + self.building_shape_feature.shape[1] + len(
                        buildings[0]['poi'])] = \
                        buildings[idx]['poi']
                    if with_type:
                        building_feature[row, 3 + self.building_shape_feature.shape[1] + len(buildings[0]['poi']):] = \
                            buildings[idx]['onehot']

                if len(pattern['poi']) > 0:
                    poi_feature = np.zeros((len(pattern['poi']), self.poi_feature_dim))
                    for row, idx in enumerate(pattern['poi']):
                        poi_feature[row, :] = pois[idx]['onehot']
                else:
                    poi_feature = None
                xy = np.zeros((building_num, 2))
                for row, idx in enumerate(pattern['building']):
                    xy[row, 0] = buildings[idx]['shape'].centroid.x
                    xy[row, 1] = buildings[idx]['shape'].centroid.y

                if with_random:
                    for row, idx in enumerate(pattern['random_point']):
                        building_feature[len(pattern['building']) + row, :] = self.random_feature
                    for row, idx in enumerate(pattern['random_point']):
                        xy[len(pattern['building']) + row, :] = self.random_points[idx]

                self.patterns.append({
                    'building_feature': building_feature,
                    'poi_feature': poi_feature,
                    'xy': xy,
                })
            with open(cached_pattern_path, 'wb') as f:
                pkl.dump(self.patterns, f)
        if cached_region_path is None:
            cached_region_path = os.path.join(cache_path, 'region.pkl')
        if use_cache and os.path.exists(cached_region_path):
            with open(cached_region_path, 'rb') as f:
                self.regions = pkl.load(f)
        else:
            with open(os.path.join(in_path, f'segmentation/{random_radius}.pkl'), 'rb') as f:
                raw_patterns = pkl.load(f)
            # with open(os.path.join(in_path, 'downstream_region.pkl'), 'rb') as f:
            #     downstream_regions = pkl.load(f)
            self.regions = {}
            pattern_tree = KDTree(
                np.array([[pattern['shape'].centroid.x, pattern['shape'].centroid.y] for pattern in raw_patterns]))
            region_df = pd.read_csv(os.path.join(cache_dir, self.city, 'region.csv'))
            geometry = gpd.GeoSeries.from_wkt(region_df['geometry'])
            # miss = 0
            for idx in range(len(geometry)):
                # calculate the diameter of the region
                geo = transform_geometry(geometry[idx])
                bounds = geo.bounds
                diameter = np.sqrt((bounds[2] - bounds[0]) ** 2 + (bounds[3] - bounds[1]) ** 2)
                # find all pattern centers within the diameter
                pattern_idx = pattern_tree.query_ball_point([geo.centroid.x, geo.centroid.y],
                                                            diameter)
                # if len(pattern_idx) == 0:
                #     self._logger.info('No pattern found for region {}'.format(idx))
                #     miss += 1
                #     continue
                # find all patterns that are intersected with the region
                pattern_idx = [i for i in pattern_idx if raw_patterns[i]['shape'].intersects(geo)]
                if len(pattern_idx) == 0:
                    k = 5
                    _, pattern_idx = pattern_tree.query([geo.centroid.x, geo.centroid.y], k)
                    pattern_idx = list(pattern_idx)
                    # 为了保证每个 region 都有 embedding，如果没找到就找最近的 K 个
                    # self._logger.info('No pattern found for region {}'.format(idx))
                    # continue
                self.regions[idx] = pattern_idx
            with open(cached_region_path, 'wb') as f:
                pkl.dump(self.regions, f)
        # if cached_grid_path is None:
        #     cached_grid_path = os.path.join(cache_path, 'grid.pkl')

        # if os.path.exists(cached_grid_path):
        #     with open(cached_grid_path, 'rb') as f:
        #         self.grids = pkl.load(f)
        # else:
        #     assert(False)
            # grid_path = 'data/raw/{}/grid/Singapore_tessellation_2km_square_projected.shp'.format(city)
            # if os.path.exists(grid_path):
            #     grid_shapefile = gpd.read_file(grid_path)
            #     self.grids = {}
            #     with open(in_path + 'segmentation.pkl', 'rb') as f:
            #         raw_patterns = pkl.load(f)
            #     pattern_tree = KDTree(
            #         np.array([[pattern['shape'].centroid.x, pattern['shape'].centroid.y] for pattern in raw_patterns]))
            #     for idx, grid in tqdm(enumerate(grid_shapefile['geometry'])):
            #         bounds = grid.bounds
            #         diameter = np.sqrt((bounds[2] - bounds[0]) ** 2 + (bounds[3] - bounds[1]) ** 2)
            #         pattern_idx = pattern_tree.query_ball_point([grid.centroid.x, grid.centroid.y], diameter)
            #         if len(pattern_idx) == 0:
            #             print('No pattern found for grid {}'.format(idx))
            #             continue
            #         pattern_idx = [i for i in pattern_idx if raw_patterns[i]['shape'].intersects(grid)]
            #         if len(pattern_idx) == 0:
            #             print('No pattern found for grid {}'.format(idx))
            #             continue
            #         self.grids[idx] = pattern_idx
            #     with open(cached_grid_path, 'wb') as f:
            #         pkl.dump(self.grids, f)



class UnsupervisedPatternDataset(Dataset):
    def __init__(self, city_data: CityData):
        self.patterns = city_data.patterns

    def __getitem__(self, index):
        return self.patterns[index]

    def __len__(self):
        return len(self.patterns)

    @staticmethod
    def collate_fn_dropout(batch):
        """
                building_feature: [max_seq_len, batch_size, feature_dim]
                building_density: [max_seq_len, batch_size, density_dim]
                building_location: [max_seq_len, batch_size, location_dim]
                poi_feature: [max_seq_len, batch_size, feature_dim]
                poi_mask: [batch_size, max_seq_len]
                xy: [max_seq_len, batch_size, 2]
        """
        # duplicate the batch
        new_batch = []
        for pattern in batch:
            new_batch.append(pattern)
            new_batch.append(pattern)
        batch = new_batch
        return UnsupervisedPatternDataset.collate_fn(batch, 0.2)

    @staticmethod
    def collate_fn(batch, dropout=0.0, max_seq_len_limit=256):
        batch_size = len(batch)
        max_building_seq_len = 0
        building_feature_list = []
        xy_list = []
        positive = 0
        for pattern in batch:
            building_seq_len = pattern['building_feature'].shape[0]
            if building_seq_len > max_seq_len_limit:
                idx = np.random.choice(building_seq_len, max_seq_len_limit, replace=False)
                building_feature_list.append(pattern['building_feature'][idx, :])
                xy_list.append(pattern['xy'][idx, :])
                building_seq_len = max_seq_len_limit
            elif positive % 2 == 0 and dropout > 0 and int(building_seq_len * (1 - dropout)) > 2:
                idx = np.random.choice(building_seq_len, int(building_seq_len * (1 - dropout)), replace=False)
                building_feature_list.append(pattern['building_feature'][idx, :])
                accurate = pattern['xy'][idx, :]
                xy_list.append(accurate)
                building_seq_len = int(building_seq_len * (1 - dropout))
            else:
                building_feature_list.append(pattern['building_feature'])
                xy_list.append(pattern['xy'])
            positive += 1
            max_building_seq_len = max(max_building_seq_len, building_seq_len)
        building_feature = np.zeros((max_building_seq_len, batch_size, building_feature_list[0].shape[1]),
                                    dtype=np.float32)
        building_mask = np.ones((batch_size, max_building_seq_len), dtype=np.bool)
        xy = np.zeros((max_building_seq_len, batch_size, 2), dtype=np.float32)
        for i in range(batch_size):
            building_seq_len = building_feature_list[i].shape[0]
            building_feature[:building_seq_len, i, :] = building_feature_list[i]
            building_mask[i, :building_seq_len] = False
            xy[:building_seq_len, i, :] = xy_list[i]

        max_poi_seq_len = 0
        poi_feature_dim = 0
        for pattern in batch:
            if pattern['poi_feature'] is not None:
                max_poi_seq_len = max(max_poi_seq_len, pattern['poi_feature'].shape[0])
                poi_feature_dim = pattern['poi_feature'].shape[1]
        if max_poi_seq_len == 0:
            poi_feature = None
            poi_mask = None
        else:
            if max_poi_seq_len > max_seq_len_limit:
                max_poi_seq_len = max_seq_len_limit
            poi_feature = np.zeros((max_poi_seq_len, batch_size, poi_feature_dim), dtype=np.float32)
            poi_mask = np.ones((batch_size, max_poi_seq_len), dtype=np.bool)
            for i in range(batch_size):
                if batch[i]['poi_feature'] is not None:
                    poi_seq_len = batch[i]['poi_feature'].shape[0]
                    if poi_seq_len > max_seq_len_limit:
                        idx = np.random.choice(poi_seq_len, max_seq_len_limit, replace=False)
                        poi_feature[:max_seq_len_limit, i, :] = batch[i]['poi_feature'][idx, :]
                        poi_mask[i, :max_seq_len_limit] = 0
                    else:
                        poi_feature[:poi_seq_len, i, :] = batch[i]['poi_feature']
                        poi_mask[i, :poi_seq_len] = 0
        return building_feature, building_mask, xy, poi_feature, poi_mask


class FreezePatternPretrainDataset(Dataset):
    def __init__(self, patterns, city_data, config, window_size=2000):
        self.patterns = patterns
        self.city = city_data.city
        self.window_size = window_size
        radius = config.get('radius', 200)
        out_path = os.path.join(cache_dir, self.city, 'ReDCL', 'processed')
        raw_pattern_path = os.path.join(out_path, f'segmentation/{radius}.pkl')
        with open(raw_pattern_path, 'rb') as f:
            self.raw_patterns = pkl.load(f)
        self.centroids = [[pattern['shape'].centroid.x, pattern['shape'].centroid.y] for pattern in self.raw_patterns]
        # get the max_x, max_y, min_x, min_y of centroids
        self.max_x = max([centroid[0] for centroid in self.centroids])
        self.max_y = max([centroid[1] for centroid in self.centroids])
        self.min_x = min([centroid[0] for centroid in self.centroids])
        self.min_y = min([centroid[1] for centroid in self.centroids])
        # divide the city into grids
        self.grid_size = self.window_size / 2
        self.num_grid_x = int((self.max_x - self.min_x) / self.grid_size) + 1
        self.num_grid_y = int((self.max_y - self.min_y) / self.grid_size) + 1
        self.grid = [[[] for _ in range(self.num_grid_y)] for _ in range(self.num_grid_x)]
        for idx, centroid in enumerate(self.centroids):
            grid_x = int((centroid[0] - self.min_x) / self.grid_size)
            grid_y = int((centroid[1] - self.min_y) / self.grid_size)
            self.grid[grid_x][grid_y].append(idx)
        self.windows = {}
        for i in range(self.num_grid_x - 1):
            for j in range(self.num_grid_y - 1):
                current = []
                current.extend(self.grid[i][j])
                current.extend(self.grid[i + 1][j])
                current.extend(self.grid[i][j + 1])
                current.extend(self.grid[i + 1][j + 1])
                if len(current) > 0:
                    self.windows[(i, j)] = current
        self.anchors = {}
        self.overlaps = {}
        self.neighbors = {}

        for key, value in self.windows.items():
            idx = len(self.anchors)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    if (key[0] + i, key[1] + j) in self.windows:
                        if idx not in self.anchors:
                            self.anchors[idx] = key
                            self.overlaps[idx] = []
                            self.neighbors[idx] = []
                        self.overlaps[idx].append((key[0] + i, key[1] + j))
            if idx in self.anchors:
                neighbor_candidate = [
                    (key[0] - 2, key[1] - 2),
                    (key[0] - 2, key[1]),
                    (key[0] - 2, key[1] + 2),
                    (key[0], key[1] - 2),
                    (key[0], key[1] + 2),
                    (key[0] + 2, key[1] - 2),
                    (key[0] + 2, key[1]),
                    (key[0] + 2, key[1] + 2),
                ]
                for neighbor in neighbor_candidate:
                    if neighbor in self.windows:
                        self.neighbors[idx].append(neighbor)

        self.negative_keys = list(self.windows.keys())

    def __getitem__(self, index):
        anchor = self.windows[self.anchors[index]]
        positive = self.windows[random.choice(self.overlaps[index])]
        # easy_negative = random.random() < 0.5
        easy_negative = True
        if easy_negative or index not in self.neighbors or len(self.neighbors[index]) == 0:
            negative = random.choice(self.negative_keys)
            while negative in self.overlaps[index]:
                negative = random.choice(self.negative_keys)
        else:
            negative = random.choice(self.neighbors[index])
            while negative in self.overlaps[index]:
                negative = random.choice(self.negative_keys)
        negative = self.windows[negative]
        anchor = [self.patterns[i] for i in anchor]
        positive = [self.patterns[i] for i in positive]
        negative = [self.patterns[i] for i in negative]
        return anchor, positive, negative

    def __len__(self):
        return len(self.anchors)

    @staticmethod
    def collate_fn(batch):
        return batch[0]


class FreezePatternForwardDataset(Dataset):
    def __init__(self, patterns, city_data):
        self.patterns = patterns
        self.regions = city_data.regions
        self.idx2key = {idx: key for idx, key in enumerate(self.regions.keys())}

    def __getitem__(self, index):
        key = self.idx2key[index]
        pattern = [self.patterns[i] for i in self.regions[key]]
        return key, pattern

    def __len__(self):
        return len(self.regions)

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)
        regions = []
        for i in range(batch_size):
            key, patterns = batch[i]
            regions.append([key, patterns])
        return regions
    

DEBUG = True
src_epsg = "EPSG:4326"
dst_epsg = "EPSG:3414"
transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)

def transform_geometry(geometry):
    if isinstance(geometry, Polygon):
        exterior_coords = list(geometry.exterior.coords)
        new_exterior_coords = []
        for x, y in exterior_coords:
            lon, lat = transformer.transform(x, y)
            new_exterior_coords.append((lon, lat))
        new_geometry = Polygon(new_exterior_coords)
    elif isinstance(geometry, MultiPolygon):
        new_polygons = []
        for polygon in geometry.geoms:
            exterior_coords = list(polygon.exterior.coords)
            new_exterior_coords = []
            for x, y in exterior_coords:
                lon, lat = transformer.transform(x, y)
                new_exterior_coords.append((lon, lat))
            new_polygons.append(Polygon(new_exterior_coords))
        new_geometry = MultiPolygon(new_polygons)
    else:
        raise ValueError("Input geometry must be a Polygon or MultiPolygon.")
    return new_geometry


class Preprocess(object):
    def __init__(self, config):
        self._logger = getLogger()
        city = config.get('dataset')
        self.cache = config.get('cache', True)
        in_path = f'/home/tangyb/private/tyb/remote/RegionDCL/data/projected/{city}/' # TODO
        out_path = os.path.join(cache_dir, city, 'ReDCL', 'processed')
        self.in_path = in_path
        self.out_path = out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.building_in_path = os.path.join(in_path, 'building/building.shp')
        self.poi_in_path = f'raw_data/{city}/{city}.geo'
        self.building_out_path = os.path.join(out_path, 'building.pkl')
        self.poi_out_path = os.path.join(out_path, 'poi.pkl')
        self.segmentation_in_path = os.path.join(in_path, 'segmentation/segmentation.shp')
        self.segmentation_out_path = os.path.join(out_path, 'segmentation')
        ensure_dir(self.segmentation_out_path)
        self.boundary_in_path = os.path.join(in_path, 'boundary/' + city + '.shp')
        self._logger.info('Loading boundary from {}'.format(self.boundary_in_path))
        boundary_shapefile = gpd.read_file(self.boundary_in_path)
        boundary = [transform_geometry(boundary_row['geometry']) for index, boundary_row in boundary_shapefile.iterrows()]
        if len(boundary) > 1:
            boundary = unary_union(boundary)
        else:
            boundary = boundary[0]
        self.boundary = boundary

    def get_building_and_poi(self):
        """
        This function will process the building and the pois.
        1. Load the building and poi from shapefile
        2. Turn the two shapefile into list of dict
            building: shape, type
            poi: x, y, code, fclass
        3. Turn the building type, poi code/fclass into one hot
        4. Attach the pois to buildings
        5. Save them to pickle, and return
        """
        if self.cache and os.path.exists(self.building_out_path):
            self._logger.info('Loading building from {}'.format(self.building_out_path))
            with open(self.building_out_path, 'rb') as f:
                building = pkl.load(f)
            self._logger.info('Loading poi from {}'.format(self.poi_out_path))
            with open(self.poi_out_path, 'rb') as f:
                poi = pkl.load(f)
            return building, poi
        self._logger.info('Preprocessing building and poi data...')
        buildings_shapefile = gpd.read_file(self.building_in_path)
        pois_shapefile = pd.read_csv(self.poi_in_path)
        pois_shapefile = pois_shapefile[pois_shapefile['traffic_type'] == 'poi']
        building = []
        poi = []
        for index, building_row in tqdm(buildings_shapefile.iterrows(), total=buildings_shapefile.shape[0]):
            output = {}
            # process polygon
            shape = transform_geometry(building_row['geometry'])
            output['shape'] = shape
            output['type'] = building_row['type']
            building.append(output)
        for index, poi_row in tqdm(pois_shapefile.iterrows(), total=pois_shapefile.shape[0]):
            output = {}
            # process point
            x, y = poi_row['coordinates'][1:-1].split(',')
            x, y = transformer.transform(x, y)
            output['x'] = float(x)
            output['y'] = float(y)
            # output['code'] = poi_row['poi_id']
            output['fclass'] = poi_row['poi_type']
            poi.append(output)
        self._logger.info('Turning building type and poi fclass into one-hot...')
        building_type = set([b['type'] for b in building])
        # poi_code = set([p['code'] for p in poi])
        poi_fclass = set([p['fclass'] for p in poi])
        building_type_dict = {t: i for i, t in enumerate(building_type)}
        # poi_code_dict = {c: i for i, c in enumerate(poi_code)}
        poi_fclass_dict = {f: i for i, f in enumerate(poi_fclass)}
        for b in building:
            b['onehot'] = [0] * len(building_type)
            b['onehot'][building_type_dict[b['type']]] = 1
        # poi_dim = len(poi_code) + len(poi_fclass)
        poi_dim = len(poi_fclass)
        for p in poi:
            p['onehot'] = [0] * poi_dim
            # p['onehot'][poi_code_dict[p['code']]] = 1
            # p['onehot'][len(poi_code) + poi_fclass_dict[p['fclass']]] = 1
            p['onehot'][poi_fclass_dict[p['fclass']]] = 1
        self._logger.info('Attaching pois to buildings...')
        # build a kd-tree for poi
        poi_x = [p['x'] for p in poi]
        poi_y = [p['y'] for p in poi]
        poi_tree = KDTree(np.array([poi_x, poi_y]).T)
        attached_poi = []
        for b in tqdm(building):
            # sum up all the pois in the building
            b['poi'] = [0] * poi_dim
            bounds = b['shape'].bounds
            cx = (bounds[0] + bounds[2]) / 2
            cy = (bounds[1] + bounds[3]) / 2
            height = bounds[3] - bounds[1]
            width = bounds[2] - bounds[0]
            radius = np.sqrt(height ** 2 + width ** 2) / 2
            # find all the pois in the radius
            poi_index = poi_tree.query_ball_point([cx, cy], radius)
            for i in poi_index:
                if not b['shape'].contains(Point(poi[i]['x'], poi[i]['y'])):
                    continue
                b['poi'] = [b['poi'][j] + poi[i]['onehot'][j] for j in range(poi_dim)]
                attached_poi.append(poi[i])
        poi_not_attached = [p for p in poi if p not in attached_poi]
        self._logger.info('Saving building and poi data...')
        with open(self.building_out_path, 'wb') as f:
            pkl.dump(building, f, protocol=4)
        with open(self.poi_out_path, 'wb') as f:
            pkl.dump(poi_not_attached, f, protocol=4)
        return building, poi_not_attached

    def poisson_disk_sampling(self, building_list, poi_list, radius):
        random_point_out_path = os.path.join(self.out_path, 'random_point_' + str(radius) + 'm.pkl')
        if self.cache and os.path.exists(random_point_out_path):
            with open(random_point_out_path, 'rb') as f:
                result = pkl.load(f)
            return result
        grid = Grid(self.boundary, radius, building_list, poi_list)
        result = grid.poisson_disk_sampling()
        with open(random_point_out_path, 'wb') as f:
            pkl.dump(result, f, protocol=4)
        return result

    def partition(self, building_list, poi_list, random_point_list, radius):
        if self.cache and os.path.exists(os.path.join(self.segmentation_out_path, f'{radius}.pkl')):
            with open(os.path.join(self.segmentation_out_path, f'{radius}.pkl'), 'rb') as f:
                result = pkl.load(f)
            return result
        self._logger.info('Partition city data by road network...')
        segmentation_shapefile = gpd.read_file(self.segmentation_in_path)
        segmentation_polygon_list = []
        for row in segmentation_shapefile.iterrows():
            it = row[1]
            segmentation_polygon_list.append(transform_geometry(it.geometry))
        result = []
        building_loc = [[b['shape'].centroid.x, b['shape'].centroid.y] for b in building_list]
        poi_loc = [[p['x'], p['y']] for p in poi_list]
        random_point_loc = random_point_list
        building_tree = KDTree(building_loc)
        poi_tree = KDTree(poi_loc)
        random_point_tree = KDTree(random_point_loc)
        for i in trange(len(segmentation_polygon_list)):
            shape = segmentation_polygon_list[i]
            pattern = {
                'shape': shape,
                'building': [],
                'poi': [],
                'random_point': []
            }
            # calculate the diameter of the shape
            bounds = shape.bounds
            dx = bounds[2] - bounds[0]
            dy = bounds[3] - bounds[1]
            diameter = math.sqrt(dx * dx + dy * dy) / 2
            # find the buildings in the shape
            building_index = building_tree.query_ball_point([shape.centroid.x, shape.centroid.y], diameter)
            for j in building_index:
                if shape.intersects(building_list[j]['shape']):
                    pattern['building'].append(j)
            # find the poi in the shape
            poi_index = poi_tree.query_ball_point([shape.centroid.x, shape.centroid.y], diameter)
            for j in poi_index:
                if shape.contains(Point(poi_loc[j][0], poi_loc[j][1])):
                    pattern['poi'].append(j)
            # find the random points in the shape
            random_point_index = random_point_tree.query_ball_point([shape.centroid.x, shape.centroid.y], diameter)
            for j in random_point_index:
                if shape.contains(Point(random_point_loc[j][0], random_point_loc[j][1])):
                    pattern['random_point'].append(j)
            # ignore the pattern without any building & random point
            if len(pattern['building']) + len(pattern['poi']) + len(pattern['random_point']) == 0:
                continue
            result.append(pattern)
        with open(os.path.join(self.segmentation_out_path, f'{radius}.pkl'), 'wb') as f:
            pkl.dump(result, f, protocol=4)
        return result

    def rasterize_buildings(self, building_list, rotation=True):
        image_out_path = os.path.join(self.out_path, 'building_raster.npz')
        rotation_out_path = os.path.join(self.out_path, 'building_rotation.npz')
        if self.cache and os.path.exists(image_out_path):
            return np.load(image_out_path)['arr_0']
        self._logger.info('Rasterize buildings...')
        images = np.zeros((len(building_list), 224, 224), dtype=np.uint8)
        rotations = np.zeros((len(building_list), 2), dtype=float)
        for i in trange(len(building_list)):
            polygon = building_list[i]['shape']
            if rotation:
                # rotate the polygon to align with the x-axis
                rectangle = polygon.minimum_rotated_rectangle
                xc = polygon.centroid.x
                yc = polygon.centroid.y
                rec_x = []
                rec_y = []
                for point in rectangle.exterior.coords:
                    rec_x.append(point[0])
                    rec_y.append(point[1])
                top = np.argmax(rec_y)
                top_left = top - 1 if top > 0 else 3
                top_right = top + 1 if top < 3 else 0
                x0, y0 = rec_x[top], rec_y[top]
                x1, y1 = rec_x[top_left], rec_y[top_left]
                x2, y2 = rec_x[top_right], rec_y[top_right]
                d1 = np.linalg.norm([x0 - x1, y0 - y1])
                d2 = np.linalg.norm([x0 - x2, y0 - y2])
                if d1 > d2:
                    cosp = (x1 - x0) / d1
                    sinp = (y0 - y1) / d1
                else:
                    cosp = (x2 - x0) / d2
                    sinp = (y0 - y2) / d2
                rotations[i] = [cosp, sinp]
                matrix = (cosp, -sinp, 0.0,
                          sinp, cosp, 0.0,
                          0.0, 0.0, 1.0,
                          xc - xc * cosp + yc * sinp, yc - xc * sinp - yc * cosp, 0.0)
                polygon = affinity.affine_transform(polygon, matrix)
            # get the polygon bounding box
            min_x, min_y, max_x, max_y = polygon.bounds
            length_x = max_x - min_x
            length_y = max_y - min_y
            # ensure the bounding box is square
            if length_x > length_y:
                min_y -= (length_x - length_y) / 2
                max_y += (length_x - length_y) / 2
            else:
                min_x -= (length_y - length_x) / 2
                max_x += (length_y - length_x) / 2
            length = max(length_x, length_y)
            # enlarge the bounding box by 20%
            min_x -= length * 0.1
            min_y -= length * 0.1
            max_x += length * 0.1
            max_y += length * 0.1
            # get transform from the new bounding box to the image
            transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, 224, 224)
            image = rasterio.features.rasterize([polygon], out_shape=(224, 224), transform=transform)
            images[i] = image
        np.savez_compressed(image_out_path, images)
        np.savez_compressed(rotation_out_path, rotations)


class Grid(object):
    def __init__(self, boundary, radius, buildings, pois):
        self._logger = getLogger()
        self.boundary = boundary
        self.bounds = boundary.bounds
        self.radius = radius
        self.min_x = self.bounds[0]
        self.max_x = self.bounds[2]
        self.min_y = self.bounds[1]
        self.max_y = self.bounds[3]
        # 代码使用 SVY21 投影坐标，和一般的经纬度表示不同
        self.grid_size = radius * np.sqrt(2) / 2
        self.num_grid_x = int((self.max_x - self.min_x) / self.grid_size) + 1
        self.num_grid_y = int((self.max_y - self.min_y) / self.grid_size) + 1
        self.buildings = [[building['shape'].centroid.x, building['shape'].centroid.y] for building in buildings]
        self.pois = [[poi['x'], poi['y']] for poi in pois]
        self.point_tree = KDTree(self.buildings + self.pois)
        self._logger.info(f'Grid:{self.num_grid_x} x {self.num_grid_y} grid size:{self.grid_size}')
        self.grid = [[[] for _ in range(self.num_grid_y)] for _ in range(self.num_grid_x)]
        transform = rasterio.transform.from_bounds(self.min_x, self.max_y, self.max_x, self.min_y, self.num_grid_x, self.num_grid_y)
        self.valid_grid = rasterio.features.rasterize(self.boundary.geoms, out_shape=(self.num_grid_y, self.num_grid_x), transform=transform)
        # plt.imshow(self.valid_grid)
        # plt.show()
        self.random_points = []

    def poisson_disk_sampling(self):
        # pick a starting point
        start_x = np.random.rand() * (self.max_x - self.min_x) + self.min_x
        start_y = np.random.rand() * (self.max_y - self.min_y) + self.min_y
        # put in grid
        grid_x = int((start_x - self.min_x) / self.grid_size)
        grid_y = int((start_y - self.min_y) / self.grid_size)
        self.grid[grid_x][grid_y].append([start_x, start_y])
        active_list = [[start_x, start_y, grid_x, grid_y]]
        # perform poisson disk sampling
        self._logger.info('Performing poisson disk sampling...')
        count = 1
        radius = self.radius
        radius2 = 2 * radius
        radiusSq = radius * radius
        while len(active_list) > 0:
            # randomly pick an active point
            index = random.randint(0, len(active_list)-1)
            point = active_list[index]
            found = False
            for i in range(30):
                new_x, new_y = self._random_point_in_ring(radius, radius2)
                new_x += point[0]
                new_y += point[1]
                # check if the new point is in the boundary
                if new_x < self.min_x or new_x > self.max_x or new_y < self.min_y or new_y > self.max_y:
                    continue
                # get the new point's grid
                new_grid_x = int((new_x - self.min_x) / self.grid_size)
                new_grid_y = int((new_y - self.min_y) / self.grid_size)
                new_grid = self.grid[new_grid_x][new_grid_y]
                if len(new_grid) > 0:
                    continue
                # for all grids nearby, check if any inside points are within the radius
                valid = True
                min_grid_x = int ((new_x - self.min_x - self.radius) / self.grid_size)
                max_grid_x = int ((new_x - self.min_x + self.radius) / self.grid_size)
                min_grid_y = int ((new_y - self.min_y - self.radius) / self.grid_size)
                max_grid_y = int ((new_y - self.min_y + self.radius) / self.grid_size)
                for i in range(min_grid_x, max_grid_x + 1):
                    if i < 0:
                        continue
                    if i >= self.num_grid_x:
                        break
                    for j in range(min_grid_y, max_grid_y + 1):
                        if j < 0:
                            continue
                        if j >= self.num_grid_y:
                            break
                        if i == new_grid_x and j == new_grid_y:
                            continue
                        if len(self.grid[i][j]) > 0:
                            point = self.grid[i][j][0]
                            delta_x = new_x - point[0]
                            delta_y = new_y - point[1]
                            if delta_x * delta_x + delta_y * delta_y < radiusSq:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    active_list.append([new_x, new_y, new_grid_x, new_grid_y])
                    new_grid.append([new_x, new_y])
                    count += 1
                    found = True
                    break
            if not found:
                active_list.pop(index)
        # output the random points
        self._logger.info(f'Total Random points: {count}')
        self.pick_valid_points()
        return self.random_points

    def pick_valid_points(self):
        count = 0
        for i in range(self.num_grid_x):
            for j in range(self.num_grid_y):
                if len(self.grid[i][j]) > 0 and self.valid_grid[j][i] > 0 and len(self.point_tree.query_ball_point([self.grid[i][j][0][0], self.grid[i][j][0][1]], self.radius)) == 0:
                    self.random_points.append(self.grid[i][j][0])
                    count += 1
        self._logger.info(f'Valid random points: {count}')
        # self.plot_random_points()

    def plot_random_points(self):
        plt.figure(figsize=(150, 150))
        for building in self.buildings:
            plt.plot(building[0], building[1], 'bo')
        for poi in self.pois:
            plt.plot(poi[0], poi[1], 'bo')
        for point in self.random_points:
            plt.plot(point[0], point[1], 'ro')
        plt.savefig('random_points.png')


    @staticmethod
    def _random_point_in_ring(r1, r2):
        """
        在圆环内随机取点, r1<=r2
        :param r1: 内径
        :param r2: 外径
        :return:
        """
        a = 1 / (r2 * r2 - r1 * r1)
        random_r = math.sqrt(random.uniform(0, 1) / a + r1 * r1)
        random_theta = random.uniform(0, 2 * math.pi)
        return random_r * math.cos(random_theta), random_r * math.sin(random_theta)

class ImageDataset(Dataset):
    """
    This dataset is used to finetune the pretrained ResNet model    
    to extract the building contour information
    """

    def __init__(self, config):
        super(ImageDataset, self).__init__()
        self._logger = getLogger()
        city = config.get('dataset')
        self._logger.info('Loading image data for {}...'.format(city))
        out_path = os.path.join(cache_dir, city, 'ReDCL', 'processed')
        self.images = np.load(os.path.join(out_path, 'building_raster.npz'))['arr_0']
        self._logger.info('Image data loaded.')

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return self.images.shape[0]

    @staticmethod
    def collate_fn_augmentation(batch):
        result = []
        augmentations = [lambda x: x,
                         lambda x: np.flip(x, axis=0),
                         lambda x: np.flip(x, axis=1),
                         lambda x: np.rot90(x, k=1, axes=(0, 1)),
                         lambda x: np.rot90(x, k=2, axes=(0, 1)),
                         lambda x: np.rot90(x, k=3, axes=(0, 1))]
        for pic in batch:
            choice1 = random.choice(augmentations)
            choice2 = random.choice(augmentations)
            result.append(choice1(pic)[np.newaxis, :, :])
            result.append(choice2(pic)[np.newaxis, :, :])
        return np.concatenate(result, axis=0)

    @staticmethod
    def collate_fn_embed(batch):
        return np.vstack([pic[np.newaxis, :, :] for pic in batch])


class ResNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super(ResNet, self).__init__()
        net = timm.create_model('resnet18', pretrained=True, **kwargs)
        self.net = nn.Sequential(*(list(net.children())[:-1]))
        self.projector = nn.Sequential(
            nn.Linear(net.num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64))

    def forward(self, x):
        return self.projector(self.get_feature(x))

    def get_feature(self, x):
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        return self.net(x)

class SimCLRTrainer(object):
    def __init__(self, config):
        city = config.get('dataset')
        self._logger = getLogger()
        self.data = ImageDataset(config)
        self.device = config.get('device', 'cpu')
        self.model = ResNet().to(self.device)
        self.criterion = self.infonce_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.train_loader = torch.utils.data.DataLoader(self.data, batch_size=128, shuffle=True, collate_fn=ImageDataset.collate_fn_augmentation)
        self.test_loader = torch.utils.data.DataLoader(self.data, batch_size=128, shuffle=False, collate_fn=ImageDataset.collate_fn_embed)

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            losses = []
            with tqdm(self.train_loader, total=len(self.train_loader)) as t:
                for x in t:
                    x = torch.from_numpy(x).float().to(self.device)
                    self.optimizer.zero_grad()
                    y_pred = self.model(x)
                    loss = self.criterion(y_pred)
                    loss.backward()
                    self.optimizer.step()
                    t.set_description(f'Epoch {epoch} loss: {loss.item()}')
                    losses.append(loss.item())
            self._logger.info(f'Epoch {epoch} loss: {np.mean(losses)}')

    def embed(self):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for x in self.test_loader:
                x = torch.from_numpy(x).float().to(self.device)
                embeddings.append(self.model.get_feature(x).cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def infonce_loss(self, y_pred, lamda=0.05):
        idxs = torch.arange(0, y_pred.shape[0], device=self.device)
        y_true = idxs + 1 - idxs % 2 * 2
        similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
        similarities = similarities - torch.eye(y_pred.shape[0], device=self.device) * 1e12
        similarities = similarities / lamda
        loss = F.cross_entropy(similarities, y_true)
        return torch.mean(loss)

def train_unsupervised(config):
    city = config.get('dataset')
    cache = config.get('cache', False)
    out_path = os.path.join(cache_dir, city, 'ReDCL', 'processed', 'building_features.npy')
    if not cache or not os.path.exists(out_path):
        trainer = SimCLRTrainer(config)
        trainer.train(3)
        embeddings = trainer.embed()
        np.save(out_path, embeddings)
