import os
import json
import numpy as np
import pandas as pd
import sys 
sys.path.append("/home/panda/remote/zwt/Map3.0")
from logging import getLogger
from datetime import datetime
from tqdm import tqdm,trange
import networkx as nx
from itertools import cycle, islice
from random import randint
from veccity.utils import ensure_dir
from random import shuffle


cache_dir = os.path.join('veccity', 'cache', 'dataset_cache')


def gen_index_map(df, column, offset=0):
        index_map = {origin: index + offset
                     for index, origin in enumerate(df[column].drop_duplicates())}
        return index_map


def str2timestamp(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())


def timestamp2str(timestamp):
    dt = datetime.fromtimestamp(timestamp)
    formatted_datetime = dt.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime


def str2timestampTZ(s):
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp())


def str2date(s):
    # TODO check 是不是冲突了
    return datetime.fromtimestamp(s).strftime("%Y-%m-%d")


class PreProcess():
    def __init__(self, config):
        self._logger = getLogger()
        self.dataset = config.get('dataset')
        self.data_dir = os.path.join(cache_dir, self.dataset)
        ensure_dir(self.data_dir)
        self.geo_file = os.path.join('raw_data', self.dataset, config.get('geo_file', self.dataset) + '.geo')
        self.rel_file = os.path.join('raw_data', self.dataset, config.get('rel_file', self.dataset) + '.grel')
        self.dyna_file = os.path.join('raw_data', self.dataset, config.get('dyna_file', self.dataset) + '.cdtraj')
        self.od_file = os.path.join('raw_data', self.dataset, config.get('od_file', self.dataset) + '.srel')


class preprocess_traj(PreProcess):
    def __init__(self, config):
        super().__init__(config)
        if not os.path.exists(self.dyna_file):
            return
        file_name = 'traj_road.csv'
        self.data_file = os.path.join(self.data_dir, file_name)
        self.min_seq_len=config.get('min_seq_len',10)
        
        if not os.path.exists(self.data_file):
            dyna_df = pd.read_csv(self.dyna_file)
            if 'location' in dyna_df.keys():
                return
            self._logger.info('Start preprocess traj.')
            id = []
            path = []
            tlist = []
            length = []
            speed = []
            duration = []
            hop = []
            usr_id = []
            traj_uid = []
            start_time = []
            lst_traj_uid, lst_usr_id = None, None
            geo_df = pd.read_csv(self.geo_file, low_memory=False)
            num_regions = int(geo_df[geo_df['traffic_type'] == 'region'].shape[0])
            gen_region_traj = True
            for _, row in tqdm(dyna_df.iterrows(), total=dyna_df.shape[0]):
                if lst_traj_uid != row['traj_uid'] or lst_usr_id != row['user_id']:  
                    idx = len(id)
                    id.append(idx)
                    path.append([])
                    tlist.append([])
                    length.append(0.)
                    speed.append(0.)
                    duration.append(0)
                    hop.append(0)
                    usr_id.append(row['user_id'])
                    traj_uid.append(row['traj_uid'])
                if isinstance(row['time'], float):
                    tlist[idx].append(round(row['time']))
                    gen_region_traj = False
                else:
                    try:
                        tlist[idx].append(str2timestamp(row['time']))
                    except:
                        tlist[idx].append(str2timestampTZ(row['time']))
                path[idx].append(int(row['geo_id']) - num_regions)
                lst_traj_uid = row['traj_uid']
                lst_usr_id = row['user_id']
            start_time, end_time, orig_geo_id, dest_geo_id = [], [], [], []
            for i in range(id[-1] + 1):
                duration[i] = tlist[i][-1] - tlist[i][0]
                hop[i] = len(path[i])
                if isinstance(tlist[i][0], int):
                    start_time.append(tlist[i][0])
                    end_time.append(tlist[i][-1])
                else:
                    start_time.append(timestamp2str(tlist[i][0]))
                    end_time.append(timestamp2str(tlist[i][-1]))
                orig_geo_id.append(path[i][0])
                dest_geo_id.append(path[i][-1])
            df = pd.concat(
                [
                    pd.Series(id, name='id'), 
                    pd.Series(path, name='path'), 
                    pd.Series(tlist, name='tlist'),
                    pd.Series(length, name='length'),
                    pd.Series(speed, name='speed'),
                    pd.Series(duration, name='duration'),
                    pd.Series(hop, name='hop'),
                    pd.Series(usr_id, name='usr_id'),
                    pd.Series(traj_uid, name='traj_uid'),
                    pd.Series(start_time, name='start_time'),
                    pd.Series(end_time, name='end_time')
                ], axis=1)
            df.to_csv(self.data_file, index=False)
            # 3:1:1
            train_file = os.path.join(self.data_dir, 'traj_road_train.csv')
            val_file = os.path.join(self.data_dir, 'traj_road_val.csv')
            test_file = os.path.join(self.data_dir, 'traj_road_test.csv')
            train_df = df.sample(frac=3/5, random_state=1)
            df = df.drop(train_df.index)
            val_df = df.sample(frac=1/2, random_state=1)
            df = df.drop(val_df.index)
            test_df = df
            train_df.to_csv(train_file, index=False)
            val_df.to_csv(val_file, index=False)
            test_df.to_csv(test_file, index=False)

            if gen_region_traj:
                # region traj
                file_name = 'traj_region.csv'
                data_file = os.path.join(self.data_dir, file_name)
                rel_df = pd.read_csv(self.rel_file)
                road2region_df = rel_df[rel_df['rel_type'] == 'road2region']
                road2region = {}
                for _, row in road2region_df.iterrows():
                    x = int(row['orig_geo_id']) - num_regions
                    y = int(row['dest_geo_id'])
                    road2region[x] = y
                df = pd.concat(
                    [
                        pd.Series(range(len(start_time)), name='dyna_id'), 
                        pd.Series(start_time, name='start_time'), 
                        pd.Series(end_time, name='end_time'),
                        pd.Series(orig_geo_id, name='orig_geo_id'),
                        pd.Series(dest_geo_id, name='dest_geo_id')
                    ], axis=1)
                df['orig_geo_id'] = df['orig_geo_id'].map(road2region)
                df['dest_geo_id'] = df['dest_geo_id'].map(road2region)
                df['flow'] = [1] * len(df)
                df.to_csv(self.od_file, index=False)
                region_paths = []
                region_tlists = []
                for i, road_path in enumerate(path):
                    tmp1, tmp2 = [], []
                    lst_region = None
                    for j, road in enumerate(road_path):
                        region = int(road2region[int(road)])
                        if region != lst_region:
                            tmp1.append(region)
                            tmp2.append(tlist[i][j])
                            lst_region = region
                    region_paths.append(tmp1)
                    region_tlists.append(tmp2)
                df = pd.concat(
                    [
                        pd.Series(id, name='id'), 
                        pd.Series(region_paths, name='path'), 
                        pd.Series(tlist, name='tlist'),
                        pd.Series(usr_id, name='usr_id'),
                        pd.Series(traj_uid, name='traj_uid'),
                        pd.Series(start_time, name='start_time')
                    ], axis=1)
                df.to_csv(data_file, index=False)
                # 3:1:1
                train_file = os.path.join(self.data_dir, 'traj_region_train.csv')
                val_file = os.path.join(self.data_dir, 'traj_region_val.csv')
                test_file = os.path.join(self.data_dir, 'traj_region_test.csv')
                train_df = df.sample(frac=3/5, random_state=1)
                df = df.drop(train_df.index)
                val_df = df.sample(frac=1/2, random_state=1)
                df = df.drop(val_df.index)
                test_df = df
                train_df.to_csv(train_file, index=False)
                val_df.to_csv(val_file, index=False)
                test_df.to_csv(test_file, index=False)
            self._logger.info('Finish preprocess traj.')
        

class preprocess_csv(PreProcess):
    def __init__(self, config):
        super().__init__(config)
        if not os.path.exists(os.path.join(self.data_dir, 'POI.csv')) or\
           not os.path.exists(os.path.join(self.data_dir, 'region.csv')) or\
           not os.path.exists(os.path.join(self.data_dir, 'road.csv')):
            self._logger.info('Start preprocess csv.')
            geo_df = pd.read_csv(self.geo_file, low_memory=False)
            df_dict = {
                'poi': pd.DataFrame(),
                'region': pd.DataFrame(),
                'road': pd.DataFrame()
            }
            for key in geo_df.keys():
                for traffic_type in df_dict.keys():
                    if key.startswith(traffic_type):
                        df_dict[traffic_type][key[len(traffic_type) + 1:]] = geo_df[key].dropna()
            for key in ['id']:
                try:
                    df_dict['poi'][key] = df_dict['poi'][key].astype(int)
                except:
                    continue
            for key in ['id', 'PARCEL_ID', 'FUNCTION', 'BLD_Count', 'InCBD', 'FORM_TYPE']:
                try:
                    df_dict['region'][key] = df_dict['region'][key].astype(int)
                except:
                    continue
            for key in ['id', 'highway', 'lanes', 'tunnel', 'bridge', 'roundabout', 'oneway', 'maxspeed', 'u', 'v']:
                try:
                    df_dict['road'][key] = df_dict['road'][key].astype(int)
                except:
                    continue
            if 'geometry' not in df_dict['region'].keys():
                df_dict['region']['geometry'] = geo_df[geo_df['traffic_type'] == 'region']['geo_location']
            if 'geometry' not in df_dict['road'].keys():
                df_dict['road']['geometry'] = geo_df[geo_df['traffic_type'] == 'road']['geo_location']
            df_dict['poi'].to_csv(os.path.join(self.data_dir, 'POI.csv'), index=False)
            df_dict['region'].to_csv(os.path.join(self.data_dir, 'region.csv'), index=False)
            df_dict['road'].to_csv(os.path.join(self.data_dir, 'road.csv'), index=False)
            self._logger.info('Finish preprocess csv.')


def save_traj_od_matrix(data_dir, file_name, n):
    file_path = os.path.join(data_dir, file_name + '_od.npy')
    if not os.path.exists(file_path):
        try:
            traj_df = pd.read_csv(os.path.join(data_dir, file_name + '.csv'))
        except:
            return
        od_matrix = np.zeros((n, n))
        for _, row in traj_df.iterrows():
            tmp = row['path'].split(',')
            if len(tmp) == 1:
                origin = destination = int(tmp[0][1:-1])
            else:
                origin = int(tmp[0][1:])
                destination = int(tmp[-1][:-1])
            od_matrix[origin][destination] += 1
        np.save(file_path, od_matrix)


def save_od_od_matrix(data_dir, file_name, n):
    file_path = os.path.join(data_dir, file_name + '_od.npy')
    if not os.path.exists(file_path):
        try:
            od_df = pd.read_csv(os.path.join(data_dir, file_name + '.csv'))
        except:
            return
        od_matrix = np.zeros((n, n))
        for _, row in od_df.iterrows():
            origin = int(row['orig_geo_id'])
            destination = int(row['dest_geo_id'])
            if "prt" in data_dir:
                # ps:porto数据集的数值比较大，为了和其他数据集保持一直，我们对其OD数据进行了除10操作
                od_matrix[origin][destination] += 0.1
            else:
                od_matrix[origin][destination] += 1
        np.save(file_path, od_matrix)


def save_in_avg(data_dir, file_name, num_days):
    file_path = os.path.join(data_dir, file_name + '_in_avg.npy')
    if not os.path.exists(file_path):
        try:
            od_file = np.load(os.path.join(data_dir, file_name + '_od.npy'))
            np.save(file_path, np.sum(od_file, axis=0))
        except:
            pass


def save_out_avg(data_dir, file_name, num_days):
    file_path = os.path.join(data_dir, file_name + '_out_avg.npy')
    if not os.path.exists(file_path):
        try:
            od_file = np.load(os.path.join(data_dir, file_name + '_od.npy'))
            np.save(file_path, np.sum(od_file, axis=1))
        except:
            pass


class preprocess_od(PreProcess):
    def __init__(self, config):
        super().__init__(config)
        if os.path.exists(os.path.join(self.data_dir, 'od_region_train_od.npy')):
            return 
        preprocess_traj(config)
        if os.path.exists(self.dyna_file):
            traj_road_path = os.path.join(self.data_dir, 'traj_road.csv')
            if os.path.exists(traj_road_path):
                # traj_df = pd.read_csv(traj_road_path)
                # num_days = traj_df['start_time'].map(str2date).drop_duplicates().shape[0]
                num_days = 0
                geo_df = pd.read_csv(self.geo_file, low_memory=False)
                num_regions = geo_df[geo_df['traffic_type'] == 'region'].shape[0]
                num_roads = geo_df[geo_df['traffic_type'] == 'road'].shape[0]
                save_traj_od_matrix(self.data_dir, 'traj_region'      , num_regions)
                save_traj_od_matrix(self.data_dir, 'traj_region_train', num_regions)
                save_traj_od_matrix(self.data_dir, 'traj_region_test' , num_regions)
                save_traj_od_matrix(self.data_dir, 'traj_road'        , num_roads  )
                save_traj_od_matrix(self.data_dir, 'traj_road_train'  , num_roads  )
                save_traj_od_matrix(self.data_dir, 'traj_road_test'   , num_roads  )
                save_in_avg(self.data_dir, 'traj_region_test', num_days)
                save_in_avg(self.data_dir, 'traj_road_test', num_days)
                save_out_avg(self.data_dir, 'traj_region_test', num_days)
                save_out_avg(self.data_dir, 'traj_road_test', num_days)
        
        train_file = os.path.join(self.data_dir, 'od_region_train.csv')
        test_file = os.path.join(self.data_dir, 'od_region_test.csv')
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            if os.path.exists(self.od_file):
                df = pd.read_csv(self.od_file)
                df['start_time'] = df['start_time']#.map(str2timestamp)
                df['end_time'] = df['end_time']#.map(str2timestamp)
                train_df = df.sample(frac=4/5, random_state=1)
                test_df = df.drop(train_df.index)
                train_df.to_csv(train_file, index=False)
                test_df.to_csv(test_file, index=False)
        file_path_train = os.path.join(self.data_dir, 'od_region_train_od.npy')
        file_path_test = os.path.join(self.data_dir, 'od_region_test_od.npy')
        if not os.path.exists(file_path_train) or not os.path.exists(file_path_test):
            geo_df = pd.read_csv(self.geo_file, low_memory=False)
            num_regions = geo_df[geo_df['traffic_type'] == 'region'].shape[0]
            save_od_od_matrix(self.data_dir, 'od_region_train', num_regions)
            save_od_od_matrix(self.data_dir, 'od_region_test' , num_regions)
        if not os.path.exists(os.path.join(self.data_dir, 'od_region_test_in_avg.npy')) or\
           not os.path.exists(os.path.join(self.data_dir, 'od_region_test_out_avg.npy')):
            if os.path.exists(self.od_file):
                od_df = pd.read_csv(self.od_file)
                num_days = od_df['start_time'].map(str2date).drop_duplicates().shape[0]
                save_in_avg(self.data_dir, 'od_region_test', num_days)
                save_out_avg(self.data_dir, 'od_region_test', num_days)


class preprocess_feature(PreProcess):
    def __init__(self, config):
        super().__init__(config)
        if os.path.exists(os.path.join(self.data_dir, 'region_features.csv')) and \
            os.path.exists(os.path.join(self.data_dir, 'road_features.csv')):
            return
        geo_df = pd.read_csv(self.geo_file, low_memory=False)
        if not os.path.exists(os.path.join(self.data_dir, 'region_features.csv')):
            flag = True
            for key in ['region_FUNCTION', 'region_InCBD', 'region_FORM_TYPE']:
                flag &= (key in geo_df.keys())
            if flag:
                FUNCTION = geo_df['region_FUNCTION'].dropna().astype(int)
                InCBD = geo_df['region_InCBD'].dropna().astype(int)
                FORM_TYPE = geo_df['region_FORM_TYPE'].dropna().astype(int)
                region_df = pd.concat(
                            [
                                pd.Series(FUNCTION, name='FUNCTION'), 
                                pd.Series(InCBD, name='InCBD'), 
                                pd.Series(FORM_TYPE, name='FORM_TYPE')
                            ], axis=1)
                region_df.to_csv(os.path.join(self.data_dir, 'region_features.csv'), index=False)

        if not os.path.exists(os.path.join(self.data_dir, 'road_features.csv')):
            flag = True
            for key in ['road_highway', 'road_lanes', 'road_maxspeed']:
                flag &= (key in geo_df.keys())
            if flag:
                try:
                    highway = geo_df['road_highway'].dropna().astype(int)
                except:
                    mp = gen_index_map(geo_df, 'road_highway')
                    highway = geo_df['road_highway'].map(mp).dropna().astype(int)
                lanes = geo_df['road_lanes'].dropna().astype(int)
                try:
                    maxspeed = geo_df['road_maxspeed'].dropna().astype(int)
                except:
                    maxspeed = [None] * len(geo_df)
                road_df = pd.concat(
                            [
                                pd.Series(highway, name='highway'), 
                                pd.Series(lanes, name='lanes'),
                                pd.Series(maxspeed, name='maxspeed')
                            ], axis=1)
                road_df.to_csv(os.path.join(self.data_dir, 'road_features.csv'), index=False)

def k_shortest_paths_nx(G, source, target, k, weight='weight'):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def build_graph(rel_file, geo_file):

    rel = pd.read_csv(rel_file)
    geo = pd.read_csv(geo_file, low_memory=False)

    node_size=geo[geo['traffic_type'] == 'road'].shape[0]
    offset=geo[geo['traffic_type'] == 'region'].shape[0]

    edge2len = {}
    geoid2coord = {}
    for i, row in tqdm(geo.iterrows(), total=geo.shape[0]):
        geo_uid = row.geo_uid-offset
        length = float(row.road_length)
        edge2len[geo_uid] = length
        # geoid2coord[geo_uid] = row.coordinates

    graph = nx.DiGraph()
    rel=rel[rel['rel_type']=="road2road"]
    for i, row in tqdm(rel.iterrows(), total=rel.shape[0]):
        prev_id = row.orig_geo_id-offset
        curr_id = row.dest_geo_id-offset

        # Use length as weight
        weight = geo.iloc[prev_id+offset].road_length
        
        # Use avg_speed as weight
        # weight = row.road_length
        if weight == float('inf') or weight < 0:
            # weight = 9999999
            pass
            # print(row)
        graph.add_edge(prev_id, curr_id, weight=weight)

    return graph, node_size


def avg_speed(config):
    road_name = config.get('dataset')
    train_file = cache_dir+f'/{road_name}/traj_road_train.csv'
    path_file = train_file
    print(cache_dir+f'/{road_name}/label_data/avg_speeds.csv')
    if os.path.exists(cache_dir+f'/{road_name}/label_data/avg_speeds.csv'):
        return 
    ensure_dir(cache_dir+f'/{road_name}/label_data')
    rel_file = os.path.join('raw_data', '{0}/{0}.grel'.format(road_name))
    geo_file = os.path.join('raw_data', '{0}/{0}.geo'.format(road_name))
    
    print(geo_file, rel_file, 'raw_data')

    rel = pd.read_csv(rel_file)
    geo = pd.read_csv(geo_file, low_memory=False)
    #调整成只有road的版本
    geo=geo[geo.traffic_type=='road']
    rel=rel[rel.rel_type=='road2road']

    path = pd.read_csv(path_file)

    max_id = max(geo.geo_uid)
    link_array = np.zeros([max_id + 1, max_id + 1, 2])
    node_array = np.zeros([max_id + 1, 2])
    
    max_length = 128
    for i, row in tqdm(path.iterrows(), total=path.shape[0]):
        plist = eval(row.path)[:max_length]
        tlist = eval(row.tlist)[:max_length]
        for i in range(len(plist) - 1):
            prev_id = plist[i]
            next_id = plist[i + 1]
            prev_time = tlist[i]
            next_time = tlist[i + 1]
            diff_time = next_time - prev_time
            link_array[prev_id][next_id][0] += diff_time
            link_array[prev_id][next_id][1] += 1
            node_array[prev_id][0] += diff_time
            node_array[prev_id][1] += 1

    err_geo_count = 0
    road_id=[]
    avg_speeds=[]
    for i, row in tqdm(geo.iterrows(), total=geo.shape[0]):

        curr_id = int(row.road_id)
        all_duration, all_time = node_array[curr_id]
        if all_time == 0:
            err_geo_count += 1
        else:
            # pdb.set_trace()
            lens=row.road_length
            road_id.append(curr_id)
            avg_speeds.append(lens/(all_duration / all_time))

    print("err_geo_count", err_geo_count, '/', geo.shape[0])
    data={'index':road_id,'speed':avg_speeds}
    df=pd.DataFrame(data)
    df=df.replace([np.inf, -np.inf], np.nan)
    df=df.dropna()
    df=df.drop(df.query('speed > 300').index)
    df.to_csv(cache_dir+f'/{road_name}/label_data/avg_speeds.csv',index=None)

def detour(graph, path, times,  avg_time, node_size, max_len=128):
    ind=np.random.randint(len(path)-2)
    new_len=2
    o_id=path[ind]
    o_time=times[ind]
    d_id=path[ind+new_len]
    d_time=times[ind]
    valid_length=0

    try:
        shortest_paths = k_shortest_paths_nx(graph,o_id,d_id,2)
    except:
        shortest_paths = [[o_id,d_id],[o_id,d_id]]
        valid_length=1
        
    original_path = path[ind:ind+new_len+1]

    new_sub_path=None
    
    if original_path == shortest_paths[0]:
        if len(shortest_paths)==2:
            new_sub_path = shortest_paths[1]
        else:
            new_sub_path = [o_id,d_id]
    else:
        new_sub_path = shortest_paths[0]

    if np.max(new_sub_path) > node_size:
        new_sub_path = [o_id,d_id]

    new_sub_time=[o_time]
    last_time=o_time


    for i in range(1, len(new_sub_path)):
        prev_id=new_sub_path[i-1]
        aft_id=new_sub_path[i]
        if avg_time[prev_id,aft_id] != 0.0:
            new_sub_time.append(o_time+avg_time[prev_id,aft_id])
        else:
            new_sub_path = [o_id,d_id]
            new_sub_time = [o_time,d_time]
            break
    difftime=new_sub_time[-1]-d_time

    new_path=path[:ind]+new_sub_path+path[ind+new_len+1:]
    
    new_time=times[:ind]+new_sub_time
    for i in times[ind+new_len+1:]:
        new_time.append(i+difftime)

    if len(new_path) > max_len:
        new_path = new_path[:max_len]
        new_time = new_time[:max_len]

    return new_path, new_time


def preprocess_detour(config):

    dataset=config['dataset']
    
    if os.path.exists(cache_dir+'/{}/ori_trajs.npz'.format(dataset)):
        return
    
    if not os.path.exists(cache_dir+'/{}/traj_road_test.csv'.format(dataset)):
        return 
    

    geo_path="./raw_data/{}/{}.geo".format(dataset,dataset)
    rel_path="./raw_data/{}/{}.grel".format(dataset,dataset)

    
    graph,node_size = build_graph(rel_path, geo_path)
    def cal_time():

        geo = pd.read_csv(geo_path,low_memory=False)
        geo=geo[geo['traffic_type']=="road"]
        tall = pd.read_csv(cache_dir+'/{}/traj_road.csv'.format(dataset))

        max_id = int(max(geo.road_id))
        link_array = np.zeros([max_id + 1, max_id + 1, 2])
        node_array = np.zeros([max_id + 1, 2])
        max_length = 128
        for i, row in tqdm(tall.iterrows(), total=tall.shape[0]):
            plist = eval(row.path)[:max_length]
            tlist = eval(row.tlist)[:max_length]
            for i in range(len(plist) - 1):
                prev_id = plist[i]
                next_id = plist[i + 1]
                prev_time = tlist[i]
                next_time = tlist[i + 1]
                diff_time = next_time - prev_time
                link_array[prev_id][next_id][0] += diff_time
                link_array[prev_id][next_id][1] += 1
                node_array[prev_id][0] += diff_time
                node_array[prev_id][1] += 1
        avg_time=np.zeros([max_id + 1, max_id + 1])
        for i in range(max_id):
            for j in range(max_id):
                all_duration, all_time = link_array[i][j]
                if all_time == 0:
                    avg_time[i, j] = 0
                else:
                    avg_time[i, j] = all_duration / all_time
        
        return avg_time
    
    if os.path.exists(cache_dir+f'/{dataset}/avg_time.npy'):
        avg_time=np.load(cache_dir+f'/{dataset}/avg_time.npy')
    else:
        avg_time = cal_time()
        np.save(cache_dir+f'/{dataset}/avg_time.npy',avg_time)

    traj=pd.read_csv(cache_dir+'/{}/traj_road_test.csv'.format(dataset))
    traj.path=traj.path.apply(eval)
    # traj_path=traj.path.to_numpy()
    random_choice=np.random.randint(0,traj.shape[0],12000)
    traj=traj.iloc[random_choice]
    new_paths=[]
    new_ori_paths=[]
    new_times=[]
    new_ori_times=[]
    a=0 
    for ind in trange(12000):
        path=traj.iloc[ind].path
        times=eval(traj.iloc[ind].tlist)
        
        assert len(path) == len(times)


        if len(path) <= 4:
            continue
        new_ori_paths.append(path)
        new_ori_times.append(times)
        new_path,new_time=detour(graph,path,times,avg_time,node_size)
        new_paths.append(new_path)
        new_times.append(new_time)

    new_path_lengths=[]
    for i in new_paths:
        new_path_lengths.append(len(i))

    new_ori_path_lengths=[]
    for i in new_ori_paths:
        new_ori_path_lengths.append(len(i))
    
    max_len=0
    max_len=np.max(new_path_lengths)
    max_len=max(max_len,np.max(new_ori_path_lengths))

    temp=[]
    for i in new_ori_paths:
        sequence=i[:max_len]
        padded_sequence = np.pad(sequence, (0, max_len - len(sequence)), 'constant').tolist()
        temp.append(padded_sequence)
    
    new_ori_paths=temp

    temp=[]
    for i in new_ori_times:
        sequence=i[:max_len]
        padded_sequence = np.pad(sequence, (0, max_len - len(sequence)), 'constant').tolist()
        temp.append(padded_sequence)
    
    new_ori_times=temp

    temp=[]
    for i in new_paths:
        sequence=i[:max_len]
        padded_sequence = np.pad(sequence, (0, max_len - len(sequence)), 'constant').tolist()
        temp.append(padded_sequence)
    
    new_paths=temp

    temp=[]
    for i in new_times:
        sequence=i
        padded_sequence = np.pad(sequence, (0, max_len - len(sequence)), 'constant').tolist()
        temp.append(padded_sequence)
    
    new_times=temp

    np.savez(cache_dir+'/{}/ori_trajs'.format(dataset),trajs=new_ori_paths,tlist=new_ori_times,lengths=new_ori_path_lengths)
    np.savez(cache_dir+'/{}/query_trajs'.format(dataset),trajs=new_paths,tlist=new_times,lengths=new_path_lengths)



def save_neighbor(traffic_type, df, offset, data_dir, n):
    file_path = os.path.join(data_dir, traffic_type + '_neighbor.json')
    if not os.path.exists(file_path):
        df = df[df['rel_type'] == traffic_type + '2' + traffic_type]
        neighbor_dict = {}
        for i in range(n):
            neighbor_dict[i] = []
        for _, row in df.iterrows():
            x = row['orig_geo_id'] - offset
            y = row['dest_geo_id'] - offset
            neighbor_dict[x].append(y)
        with open(file_path, 'w') as f:
            json.dump(neighbor_dict, f)


class preprocess_neighbor(PreProcess):
    def __init__(self, config):
        super().__init__(config)
        file_path_road = os.path.join(self.data_dir, 'road_neighbor.json')
        file_path_region = os.path.join(self.data_dir, 'region_neighbor.json')
        if os.path.exists(file_path_road) and os.path.exists(file_path_region):
            return
        rel_df = pd.read_csv(self.rel_file)
        geo_df = pd.read_csv(self.geo_file, low_memory=False)
        num_regions = geo_df[geo_df['traffic_type'] == 'region'].shape[0]
        num_roads = geo_df[geo_df['traffic_type'] == 'road'].shape[0]
        save_neighbor('road', rel_df, num_regions, self.data_dir, num_roads)
        save_neighbor('region', rel_df, 0, self.data_dir, num_regions)


def preprocess_all(config):
    preprocess_csv(config)
    preprocess_feature(config)
    preprocess_neighbor(config)
    preprocess_traj(config)
    preprocess_od(config)
    avg_speed(config)
    preprocess_detour(config)
    
