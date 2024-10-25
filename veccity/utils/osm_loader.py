import json
import sys
sys.path.append('.')
import time
import logging
import pickle
from ast import literal_eval
import pandas as pd
import numpy as np
import networkx as nx
import math
import torch
import re
import os

from veccity.utils.Config import Config
from veccity.utils import cell
sys.modules['cell'] = cell
from cell import CellSpace
from veccity.utils.tool_funcs import haversine_np 
from veccity.utils.edge_index import EdgeIndex
from veccity.data.preprocess import cache_dir


class OSMLoader:

    def __init__(self, osm_raw_file, schema = '', device=None):
        self.schema = schema
        self.osm_raw_file = osm_raw_file
        self.device = device
        self.node_file = osm_raw_file + '_node'
        self.segment_file = osm_raw_file + '_segment'
        self.way_file = osm_raw_file + '_way'
        self.adj_segment_file = osm_raw_file + '_adjsegment'
        self.cellspace_file = osm_raw_file + '_cellspace_info.pickle'
        self.segment_middistance_file = osm_raw_file + '_segment_middistance_ndarray.pickle'
        self.segment_middistance_idxmap_file = osm_raw_file + '_segment_middistance_ndarray_idxmap.pickle'

        self.seg_feats=None
        self.nodes = None # DataFrame; all intersection nodes on road network;
        self.segments = None # DataFrame; all segments on road network; several segments form a way (real road)
        self.ways = None # DataFrame; all ways (real road) on road network;
        
        self.adj_segments = None # DataFrame; relations of adjacent segments;
        self.adj_segments_graph = None # DiGraph; adjacent segment graph

        self.cellspace = None
        self.cell_distance_dic = {} # (cell_id_1, cell_id_2) -> mid_point_dis

        self.seg_startendid_to_incid = None
        self.seg_incid_to_startendid = None

        self.seg_dis_matrix = None # distance matrix
        self.seg_dis_matrix_segid_indices = None


    def __str__(self):
        return ''

    def load_cikm_data(self,dataset):
        _time = time.time()

        # ======== basis for all =========
        # self.cikm_data_path = 'raw_data/{}/'.format(dataset)
        data_cache_dir = os.path.join(cache_dir, dataset)
        self.segments_file_path = os.path.join(data_cache_dir, 'road.csv')
        self.segments = pd.read_csv(self.segments_file_path)
        self.road_num = len(self.segments)
        self.adj_json_path = os.path.join(data_cache_dir, 'road_neighbor.json')

        def construct_road_adj():
            road_adj = np.zeros(shape=[self.road_num, self.road_num])
            # 构建路网的邻接关系
            with open(self.adj_json_path, 'r', encoding='utf-8') as fp:
                road_adj_data = json.load(fp)
            for road in range(self.road_num):
                road_adj[road][road] = 1
                for neighbor in road_adj_data[str(road)]:
                    road_adj[road][neighbor] = 1
            return road_adj

        self.road_adj = construct_road_adj()

        # bj dataset 没有 {s, e, m}_{lat, lon} 这几列，要手动添加
        if 'm_lat' not in self.segments:
            s_lat, s_lon, e_lat, e_lon, m_lat, m_lon = [], [], [], [], [], []
            for index, row in self.segments.iterrows():
                # 使用正则表达式匹配小数
                matches = re.findall(r"[-+]?\d*\.\d+", row['geometry'])
                # 提取匹配到的小数
                decimal_values = [float(match) for match in matches]
                s_lon.append(decimal_values[0])
                s_lat.append(decimal_values[1])
                e_lon.append(decimal_values[-2])
                e_lat.append(decimal_values[-1])
                m_lon.append((decimal_values[0] + decimal_values[-2]) / 2)
                m_lat.append((decimal_values[1] + decimal_values[-1]) / 2)
            self.segments['s_lon'] = s_lon
            self.segments['s_lat'] = s_lat
            self.segments['e_lon'] = e_lon
            self.segments['e_lat'] = e_lat
            self.segments['m_lon'] = m_lon
            self.segments['m_lat'] = m_lat

        if 'id' not in self.segments:
            self.segments['id'] = list(range(len(self.segments)))

        self.segments['inc_id'] = self.segments.id
        self.segments['c_lat'] = self.segments.m_lat
        self.segments['c_lon'] = self.segments.m_lon
        self.segments['radian'] = 0
        for i in range(self.road_num):
            lat1 = self.segments.loc[i,'s_lat']
            lon1 = self.segments.loc[i, 's_lon']
            lat2 = self.segments.loc[i, 'e_lat']
            lon2 = self.segments.loc[i, 'e_lon']
            self.segments.loc[i,'radian'] = self.calculate_radian(lat1,lon1, lat2, lon2)
        lon_min = min(self.segments.c_lon.min(), self.segments.s_lon.min(), self.segments.e_lon.min())
        lon_max = max(self.segments.c_lon.max(), self.segments.s_lon.max(), self.segments.e_lon.max())
        lat_min = min(self.segments.c_lat.min(), self.segments.s_lat.min(), self.segments.e_lat.min())
        lat_max = max(self.segments.c_lat.max(), self.segments.s_lat.max(), self.segments.e_lat.max())
        lon_unit = 0.00568828214
        lat_unit = 0.004496402877
        self.cellspace = CellSpace(lon_unit,lat_unit,lon_min,lat_min,lon_max,lat_max)
        _seg_codes, _ = pd.factorize(self.segments.inc_id)
        self.segments['segid_code'] = _seg_codes
        self.segments['length_code'] = (self.segments.length / Config.sarn_seg_length_unit).astype('int64')
        self.segments['radian_code'] = (self.segments.radian / Config.sarn_seg_radian_unit).astype('int64')
        self.segments['radian_code'] = self.segments['radian_code'] - self.segments['radian_code'].min()
        _lon_unit = 0.000568828214
        _lat_unit = 0.0004496402877
        cellspace_finegrained = CellSpace(_lon_unit, _lat_unit, self.cellspace.lon_min, self.cellspace.lat_min, \
                                          self.cellspace.lon_max, self.cellspace.lat_max)
        self.segments['s_lon_code'] = (
                    (self.segments.s_lon - cellspace_finegrained.lon_min) / cellspace_finegrained.lon_unit).astype(
            'int64')
        self.segments['s_lat_code'] = (
                    (self.segments.s_lat - cellspace_finegrained.lat_min) / cellspace_finegrained.lat_unit).astype(
            'int64')
        self.segments['e_lon_code'] = (
                    (self.segments.e_lon - cellspace_finegrained.lon_min) / cellspace_finegrained.lon_unit).astype(
            'int64')
        self.segments['e_lat_code'] = ((self.segments.e_lat - cellspace_finegrained.lat_min) / cellspace_finegrained.lat_unit).astype('int64')
        self.segments.loc[(self.segments.lanes == 'NG'), 'lanes'] = 0
        self.segments.lanes = self.segments.lanes.astype('int32')
        if self.schema == 'SARN':
            _lon_unit = 0.0068259385680000005
            _lat_unit = 0.0053956834530000004
            self.cellspace = CellSpace(_lon_unit, _lat_unit, self.cellspace.lon_min, self.cellspace.lat_min, \
                                            self.cellspace.lon_max, self.cellspace.lat_max)
            self.segments['c_cellid'] = self.segments[['c_lon','c_lat']].apply(lambda x: self.cellspace.get_cell_id_by_point(*x), axis=1)
            self.sarn_moco_each_queue_size = math.ceil(Config.sarn_moco_total_queue_size / \
                                                    (self.cellspace.lon_size * self.cellspace.lat_size))
        self.count_segid_code = len(np.unique(self.segments['segid_code'].values))
        self.count_highway_cls = self.segments['highway'].max() + 1
        self.count_length_code = self.segments['length_code'].max() + 1
        self.count_radian_code = self.segments['radian_code'].max() + 1
        self.count_s_lon_code = max(self.segments['s_lon_code'].max(), self.segments['e_lon_code'].max()) + 1
        self.count_s_lat_code = max(self.segments['s_lat_code'].max(), self.segments['e_lat_code'].max()) + 1
        self.count_lanes = self.segments['lanes'].max() + 1  # HRNR uses only
        self.adj_segments_graph = nx.from_numpy_array(self.road_adj,create_using=nx.DiGraph())
        segment_attr_dicts = self.segments.reset_index().set_index('inc_id')[:].to_dict(orient='index')
        nx.set_node_attributes(self.adj_segments_graph, segment_attr_dicts)
        #adjsegs_subgraphs = sorted(nx.weakly_connected_components(self.adj_segments_graph), key=len, reverse=True)
       # adjsegs_remove_nodes = set.union(*adjsegs_subgraphs[1:])
        #logging.debug('adj_segments_graph subgraphs={}, #remove_nodes={}'.format([len(x) for x in adjsegs_subgraphs],
         #                                                                        len(adjsegs_remove_nodes)))
       # self.adj_segments_graph.remove_nodes_from(adjsegs_remove_nodes)
        logging.debug('adj_segments_graph. #nodes={}, #edge={}'.format( \
            len(self.adj_segments_graph), self.adj_segments_graph.number_of_edges()))
        if self.schema == 'SARN':
            spatial_segments_graph = self.__create_spatial_weighted_seg_graph()
            self.adj_segments_graph.add_edges_from(spatial_segments_graph.edges(data=True))
        # after all adj_segments_graph related operations. sorted
        self.segid_in_adj_segments_graph = sorted(list(self.adj_segments_graph.nodes()))
        self.seg_id_to_idx_in_adj_seg_graph = dict()
        self.seg_idx_to_id_in_adj_seg_graph = [-1] * len(self.segid_in_adj_segments_graph)
        for _idx, _id in enumerate(self.segid_in_adj_segments_graph):
            self.seg_id_to_idx_in_adj_seg_graph[_id] = _idx
            self.seg_idx_to_id_in_adj_seg_graph[_idx] = _id

        self.edge_index = EdgeIndex(self.adj_segments_graph, self.seg_id_to_idx_in_adj_seg_graph)
        # segment features
        # self.seg_feats has dependency on self.segid_in_adj_segments_graph
        # , so dont move this part up
        _feat_columns = ['segid_code', 'highway', 'length_code', \
                         'radian_code', 's_lon_code', 's_lat_code', 'e_lon_code', 'e_lat_code', \
                         'lanes']
        self.seg_feats = self.segments.reset_index().set_index('inc_id')[_feat_columns]
        self.seg_feats = torch.tensor(self.seg_feats.loc[self.segid_in_adj_segments_graph].values, dtype=torch.long,
                                      device=self.device)  # [N, n_feat_columns]

        logging.debug("seg_feats statistics: min={}, max={}".format( \
            torch.min(self.seg_feats, dim=0)[0].tolist(), \
            torch.max(self.seg_feats, dim=0)[0].tolist()))

    def calculate_radian(self,lat1,lon1,lat2,lon2):
        # 将经纬度转换为弧度
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # 计算经纬度差值
        delta_lon = lon2_rad - lon1_rad

        # 使用反三角函数计算弧度
        radian = math.atan2(math.sin(delta_lon) * math.cos(lat2_rad),
                            math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
                                lat2_rad) * math.cos(delta_lon))

        return radian

    def load_data(self):
        if self.osm_raw_file == '':
            return -1
        _time = time.time()

        #======== basis for all =========
        self.nodes = pd.read_csv(self.node_file, delimiter = ',', index_col = 'node_id')
        
        self.segments = pd.read_csv(self.segment_file, delimiter = ',', index_col = ['s_id','e_id'])
        self.segments.way_ids = self.segments.way_ids.apply(literal_eval)
        
        self.ways = pd.read_csv(self.way_file, delimiter = ',', index_col = 'way_id')
        self.ways.node_ids = self.ways.node_ids.apply(literal_eval)
        
        self.adj_segments = pd.read_csv(self.adj_segment_file, delimiter = ',')

        self.cellspace = pickle.load(open(self.cellspace_file, 'rb'))

        # (2653532349, 2653532346) -> {'inc_id': 15643}
        self.seg_startendid_to_incid = self.segments[['inc_id']].to_dict(orient = 'index')
        # 100 -> {'s_id': 276558558, 'e_id': 314623102}
        self.seg_incid_to_startendid = self.segments.reset_index().set_index('inc_id')[['s_id','e_id']].to_dict(orient = 'index')


        #======== more data processing =========

        # segment features
        _way_codes, _ = pd.factorize(self.segments.way_ids.map(lambda x: x[0]))
        self.segments['wayid_code'] = _way_codes

        _seg_codes, _ = pd.factorize(self.segments.inc_id)
        self.segments['segid_code'] = _seg_codes

        self.segments['length_code'] = (self.segments.length / Config.sarn_seg_length_unit).astype('int64')

        self.segments['radian_code'] = (self.segments.radian / Config.sarn_seg_radian_unit).astype('int64')
        # add lat and lon code 
        _lon_unit = 50 * Config.dataset_lon2Euc_unit # 50 meters
        _lat_unit = 50 * Config.dataset_lat2Euc_unit
        cellspace_finegrained = CellSpace(_lon_unit, _lat_unit, self.cellspace.lon_min, self.cellspace.lat_min, \
                                            self.cellspace.lon_max, self.cellspace.lat_max)
        self.segments['s_lon_code'] = ((self.segments.s_lon - cellspace_finegrained.lon_min) / cellspace_finegrained.lon_unit).astype('int64')
        self.segments['s_lat_code'] = ((self.segments.s_lat - cellspace_finegrained.lat_min) / cellspace_finegrained.lat_unit).astype('int64')
        self.segments['e_lon_code'] = ((self.segments.e_lon - cellspace_finegrained.lon_min) / cellspace_finegrained.lon_unit).astype('int64')
        self.segments['e_lat_code'] = ((self.segments.e_lat - cellspace_finegrained.lat_min) / cellspace_finegrained.lat_unit).astype('int64')

        # self.segments.lanes.loc[self.segments.lanes == 'NG'] = 0
        self.segments.loc[(self.segments.lanes =='NG'), 'lanes'] = 0
        self.segments.lanes = self.segments.lanes.astype('int32')

        # cellid of segment base on self.cellspace # caution, we have multiple cellspaces, ~= 1000meters
        if self.schema == 'SARN':
            _lon_unit = Config.dataset_lon2Euc_unit * Config.sarn_moco_multi_queue_cellsidelen
            _lat_unit = Config.dataset_lat2Euc_unit * Config.sarn_moco_multi_queue_cellsidelen
            self.cellspace = CellSpace(_lon_unit, _lat_unit, self.cellspace.lon_min, self.cellspace.lat_min, \
                                            self.cellspace.lon_max, self.cellspace.lat_max)
            self.segments['c_cellid'] = self.segments[['c_lon','c_lat']].apply(lambda x: self.cellspace.get_cell_id_by_point(*x), axis=1)
            Config.sarn_moco_each_queue_size = math.ceil(Config.sarn_moco_total_queue_size / \
                                                    (self.cellspace.lon_size * self.cellspace.lat_size))

        # counting for Embedding constructor
        self.count_wayid_code = len(np.unique(self.segments['wayid_code'].values))
        self.count_segid_code = len(np.unique(self.segments['segid_code'].values))
        self.count_highway_cls = self.segments['highway_cls'].max() + 1
        self.count_length_code = self.segments['length_code'].max() + 1
        self.count_radian_code = self.segments['radian_code'].max() + 1
        self.count_s_lon_code = max(self.segments['s_lon_code'].max(), self.segments['e_lon_code'].max()) + 1
        self.count_s_lat_code = max(self.segments['s_lat_code'].max(), self.segments['e_lat_code'].max()) + 1
        self.count_lanes = self.segments['lanes'].max() + 1 # HRNR uses only

        # connected graph
        # 1. read adjacent segment graph from files
        # 2. add segment features to the adjacent graph
        # 3. verify connectness of graph, remove small subgraphs, create strongly/weakly? connected graph 
        self.adj_segments_graph = nx.from_pandas_edgelist(self.adj_segments, \
                                        's_id', 'e_id', edge_attr = True, create_using = nx.DiGraph())
        segment_attr_dicts = self.segments.reset_index().set_index('inc_id')[:].to_dict(orient = 'index')
        # https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.set_node_attributes.html
        nx.set_node_attributes(self.adj_segments_graph, segment_attr_dicts)
        # connected graph
        # https://www.cnblogs.com/wushaogui/p/9204797.html
        adjsegs_subgraphs = sorted(nx.weakly_connected_components(self.adj_segments_graph), key = len, reverse = True)
        adjsegs_remove_nodes = set.union(*adjsegs_subgraphs[1: ])
        logging.debug('adj_segments_graph subgraphs={}, #remove_nodes={}'.format([len(x) for x in adjsegs_subgraphs], len(adjsegs_remove_nodes)))
        self.adj_segments_graph.remove_nodes_from(adjsegs_remove_nodes)
        logging.debug('adj_segments_graph. #nodes={}, #edge={}'.format( \
                        len(self.adj_segments_graph), self.adj_segments_graph.number_of_edges()))
        
        if self.schema == 'SARN':
            # 1. create distance-and-radian weighted segment graph. (dont add to many edge attributes)
            # 2. combine edge atrributes of spatial_segments_graph into self.adj_segments_graph
            spatial_segments_graph = self.__create_spatial_weighted_seg_graph()
            self.adj_segments_graph.add_edges_from(spatial_segments_graph.edges(data=True))

        # after all adj_segments_graph related operations. sorted
        self.segid_in_adj_segments_graph = sorted(list(self.adj_segments_graph.nodes()))
        self.seg_id_to_idx_in_adj_seg_graph = dict()
        self.seg_idx_to_id_in_adj_seg_graph = [-1] * len(self.segid_in_adj_segments_graph)
        for _idx, _id in enumerate(self.segid_in_adj_segments_graph):
            self.seg_id_to_idx_in_adj_seg_graph[_id] = _idx
            self.seg_idx_to_id_in_adj_seg_graph[_idx] = _id

        self.edge_index = EdgeIndex(self.adj_segments_graph, self.seg_id_to_idx_in_adj_seg_graph)

        # segment features
        # self.seg_feats has dependency on self.segid_in_adj_segments_graph
        # , so dont move this part up
        _feat_columns = ['wayid_code', 'segid_code', 'highway_cls', 'length_code', \
                        'radian_code', 's_lon_code', 's_lat_code', 'e_lon_code', 'e_lat_code', \
                        'lanes']
        self.seg_feats = self.segments.reset_index().set_index('inc_id')[_feat_columns]
        self.seg_feats = torch.tensor(self.seg_feats.loc[self.segid_in_adj_segments_graph].values, dtype = torch.long, device = self.device) # [N, n_feat_columns]

        logging.debug("seg_feats statistics: min={}, max={}".format( \
                        torch.min(self.seg_feats, dim = 0)[0].tolist(), \
                        torch.max(self.seg_feats, dim = 0)[0].tolist()))
   
        logging.info('osm_data load ended. @={:.3f}, #nodes={}, #segments={}, #ways={}, '
                        '#nodes_in_adj_seg={}, #edges_in_adj_seg={}'.format( \
                        time.time() - _time, self.nodes.shape[0], self.segments.shape[0], 
                        self.ways.shape[0], len(self.segid_in_adj_segments_graph), 
                        self.adj_segments_graph.number_of_edges()))


    def __create_spatial_weighted_seg_graph(self):
        # dont use self.cellspace here, because we may use a fine-grained cell space to represent lonlat.
        # For simplicity, we can create a new cellspace that the unit size 
        # is equal to Config.sarn_seg_weight_distance_thres, which will not slow down the program.

        _time = time.time()
        # if os.path.exists('sarn_data_tmp.csv'):
        #     all_seg_pairs = pd.read_csv('sarn_data_tmp.csv')
        # else:
        # ['inc_id_x', 'inc_id_y', 'distance', 'radian_delta', 'distance_weight', 'radian_weight', 'spatial_weight']
        all_seg_pairs = pd.DataFrame() 

        # create cellspace for 
        _lon_unit = 0.0022753128560000003
        _lat_unit = 0.001798561151
        cs = CellSpace(_lon_unit, _lat_unit, self.cellspace.lon_min, self.cellspace.lat_min, \
                        self.cellspace.lon_max, self.cellspace.lat_max)

        all_cell_pairs = cs.all_neighbour_cell_pairs_permutated() # all legal neighbouring cell pairs
        segs = self.segments[['inc_id','c_lon','c_lat','radian']].copy()
        segs.loc[:, 'c_cellid_tmp'] = segs.loc[:,['c_lon','c_lat']].apply(lambda x: cs.get_cell_id_by_point(*x), axis=1)
        for ((i_lon_1, i_lat_1), (i_lon_2, i_lat_2)) in all_cell_pairs:
            cell_id_1 = cs.get_cell_id(i_lon_1, i_lat_1)
            cell_id_2 = cs.get_cell_id(i_lon_2, i_lat_2)
            segs_in_cell_1 = segs[segs['c_cellid_tmp'] == cell_id_1]
            segs_in_cell_2 = segs[segs['c_cellid_tmp'] == cell_id_2]
            segs_pairs = segs_in_cell_1[['inc_id','c_lon','c_lat','radian']].merge(segs_in_cell_2[['inc_id','c_lon','c_lat','radian']], how='cross')
            segs_pairs['distance'] = haversine_np(segs_pairs['c_lon_x'].values, segs_pairs['c_lat_x'].values, \
                                                    segs_pairs['c_lon_y'].values, segs_pairs['c_lat_y'].values)
            # remove all <seg_i, seg_i> self connections, distance thres, radian thres
            segs_pairs = segs_pairs[ (segs_pairs['distance'] <= Config.sarn_seg_weight_distance_thres) \
                                    & (segs_pairs['distance'] > 0) ]
            segs_pairs['radian_delta'] = np.abs(segs_pairs['radian_x'] - segs_pairs['radian_y'])
            segs_pairs = segs_pairs[ segs_pairs['radian_delta'] <= Config.sarn_seg_weight_radian_delta_thres ]
            segs_pairs = segs_pairs[['inc_id_x', 'inc_id_y', 'distance', 'radian_delta']]

            all_seg_pairs = pd.concat([all_seg_pairs, segs_pairs], axis = 0, ignore_index = True)
        
        # calculate distance weight and radian weight
        all_seg_pairs['distance_weight'] = np.cos(0.5 * np.pi * all_seg_pairs['distance'] / Config.sarn_seg_weight_distance_thres)
        all_seg_pairs['radian_weight'] = np.cos(0.5 * np.pi * all_seg_pairs['radian_delta'] / Config.sarn_seg_weight_radian_delta_thres)
        all_seg_pairs['spatial_weight'] = (all_seg_pairs['distance_weight'] + all_seg_pairs['radian_weight']) / 2
        
        # duplicate alone diagonal
        all_seg_pairs = pd.concat([all_seg_pairs, all_seg_pairs.rename(columns={'inc_id_x':'inc_id_y', 'inc_id_y':'inc_id_x'})], \
                                    axis = 0, ignore_index = True)
        # all_seg_pairs.to_csv('sarn_data_tmp.csv', index=False)
        # TODO：可能可以缓存

        # randomly sampling remove to allow the spatial_segments_graph
        # , having the same edge to self.adj_semgnts_graph 
        # all_seg_pairs = all_seg_pairs.sample(n = self.adj_segments_graph.number_of_edges(),
        #                                     replace = False, weights = 'spatial_weight',
        #                                     axis = 0)

        # convert to nx.graph
        spatial_segments_graph = nx.from_pandas_edgelist(all_seg_pairs, \
                                            'inc_id_x', 'inc_id_y', \
                                            edge_attr = True, create_using = nx.DiGraph())
        
        segs_remove_nodes = set(spatial_segments_graph.nodes).difference(set(self.adj_segments_graph.nodes))
        spatial_segments_graph.remove_nodes_from(segs_remove_nodes)
        segs_add_nodes = set(self.adj_segments_graph.nodes).difference(set(spatial_segments_graph.nodes))
        spatial_segments_graph.add_nodes_from(segs_add_nodes)
        logging.debug('spatial_adj_segments_graph #remove_nodes={}, #add_nodes={}'.format(len(segs_remove_nodes), len(segs_add_nodes)))

        # both graph should be have same number of nodes, even same nodes.
        # we have a loose assertion here. # TODO
        assert len(self.adj_segments_graph) == len(spatial_segments_graph) 
        
        logging.info('spatial_adj_segments_graph. #nodes={}, #edge={}, @={:.3f}'.format( \
                        len(spatial_segments_graph), spatial_segments_graph.number_of_edges(), \
                        time.time() - _time))

        return spatial_segments_graph
