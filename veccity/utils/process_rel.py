import pdb
import json
import os
import math
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from shapely import wkt
from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon, LineString

dataset = None
dict_list = {}
total_cnt = {}

def parse_linestring_coordinates(coordinates_string):
    # 移除字符串中的方括号和空格
    cleaned_string = coordinates_string.replace("[", "").replace("]", "").replace(" ", "")
    # 将字符串分割为坐标对列表
    coordinates = cleaned_string.split(",")
    # 将字符串坐标对转换为浮点数列表
    coordinates = [(float(coordinates[i]), float(coordinates[i + 1])) for i in range(0, len(coordinates), 2)]
    # 创建 LineString 对象
    linestring = LineString(coordinates)
    return linestring

def work(s):
    df = pd.read_csv(os.path.join(dataset, s + '_' + dataset + '.csv'))
    s = s.lower()
    t = {}
    total = 0
    for i, row in df.iterrows():
        tmp_dict = row.to_dict()
        for key, value in tmp_dict.items():
            if key not in t:
                t[key] = []
            t[key].append(value)
        total += 1
    global dict_list
    dict_list[s] = t
    global total_cnt
    total_cnt[s] = total

def generate_geo():
    work('POI')
    work('region')
    work('road')
    df = pd.DataFrame()
    df['geo_uid'] = list(range(total_cnt['region'] + total_cnt['road'] + total_cnt['poi']))
    df['type'] = ['Polygon'] * total_cnt['region'] + ['LineString'] * total_cnt['road'] + ['Point'] * total_cnt['poi']
    df['geo_location'] = [''] * (total_cnt['region'] + total_cnt['road'] + total_cnt['poi'])
    df['traffic_type'] = ['region'] * total_cnt['region'] + ['road'] * total_cnt['road'] + ['poi'] * total_cnt['poi']
    for key, value in dict_list['region'].items():
        df['region_' + key] = value + [None] * (total_cnt['road'] + total_cnt['poi'])
    for key, value in dict_list['road'].items():
        df['road_' + key] = [None] * total_cnt['region'] + value + [None] * total_cnt['poi']
    for key, value in dict_list['poi'].items():
        df['poi_' + key] = [None] * (total_cnt['region'] + total_cnt['road']) + value
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row['traffic_type'] == 'region':
            df.loc[i, 'geo_location'] = row['region_geometry']
        elif row['traffic_type'] == 'road':
            df.loc[i, 'geo_location'] = row['road_coordinates']
        else:
            df.loc[i, 'geo_location'] = '[' + str(row['poi_1']) + ', ' + str(row['poi_2']) + ']'
    drop_cols = ['region_Unnamed: 0', 'poi_Unnamed: 0', 'road_coordinates', 'poi_1', 'poi_2']
    for col in drop_cols:
        if col in df.keys():
            df = df.drop(col, axis=1)
    rename_cols = {
        'poi_3': 'poi_type',
        'poi_4': 'poi_country',
        'poi_0': 'poi_id'
    }
    df = df.rename(columns=rename_cols)
    pdb.set_trace()
    df.to_csv(os.path.join(dataset, dataset + '.geo'), index=False)

def generate_rel():
    rel_path = os.path.join(dataset, f'{dataset}.grel')
    if os.path.exists(os.path.join(dataset, 'roadsegment.grel')):
        rel_path = os.path.join(dataset, 'roadsegment.grel')
    elif os.path.exists(os.path.join(dataset, f'{dataset}_roadmap_edge.grel')):
        rel_path = os.path.join(dataset, f'{dataset}_roadmap_edge.grel')
    road_rel_df = pd.read_csv(rel_path)
    rel_type_col = ['road2road'] * len(road_rel_df)
    road_rel_df['rel_type'] = pd.Series(rel_type_col)
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    region_cnt = len(geo_df[geo_df['traffic_type'] == 'region'])
    road_rel_df['orig_geo_id'] += region_cnt
    road_rel_df['dest_geo_id'] += region_cnt
    road_rel_df.to_csv(os.path.join(dataset, f'{dataset}.grel'), index=False)

def generate_crimes_count():
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    g = []
    for i, row in geo_df.iterrows():
        if row['type'] not in ['Polygon', 'MultiPolygon']:
            break
        g.append(wkt.loads(row['geo_location']))
    count = [0] * len(geo_df)
    crime_df = pd.read_csv(os.path.join(dataset, 'crimes.csv'))
    crime_df = crime_df.dropna()
    for i, row in tqdm(crime_df.iterrows(), total=len(crime_df)):
        year = row['Date'].split('/')[2][:4]
        if year != '2020':
            continue
        # p = Point(float(row['lng']), float(row['lat']))
        p = Point(float(row['Longitude']), float(row['Latitude']))
        for j in range(len(g)):
            if g[j].contains(p):
                count[j] += 1
                break
    print(sum(count))
    geo_df['region_crimes_count'] = count
    geo_df.to_csv(f'{dataset}.geo', index=False)

def generate_poi2region():
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    g = []
    for i, row in geo_df.iterrows():
        if row['type'] not in ['Polygon', 'MultiPolygon']:
            break
        g.append(wkt.loads(row['geo_location']))
        geo_df.loc[i, 'type'] = g[-1].geom_type

    rel_df = pd.read_csv(os.path.join(dataset, f'{dataset}.grel'))
    new_rows = []
    rel_uid = len(rel_df)
    SWAP = True
    for i, row in tqdm(geo_df.iterrows(), total=len(geo_df)):
        if row['traffic_type'] != 'poi':
            continue
        lng, lat = row['geo_location'][1:-1].split(',')
        lat = lat[1:]
        if SWAP:
            geo_df.loc[i, 'geo_location'] = f'[{lat}, {lng}]'
            lat, lng = lng, lat
        p = Point(float(lng), float(lat))
        for j, r in enumerate(g):
            if r.contains(p):
                new_rows.append({
                    'rel_uid': rel_uid,
                    'type': 'geo',
                    'orig_geo_id': int(row['geo_uid']),
                    'dest_geo_id': j,
                    'rel_type': 'poi2region'
                })
                rel_uid += 1
                new_rows.append({
                    'rel_uid': rel_uid,
                    'type': 'geo',
                    'orig_geo_id': j,
                    'dest_geo_id': int(row['geo_uid']),
                    'rel_type': 'region2poi'
                })
                rel_uid += 1

    rel_df = rel_df.append(new_rows, ignore_index=True)
    pdb.set_trace()
    if SWAP:
        geo_df.to_csv(os.path.join(dataset, f'{dataset}.geo'), index=False)
    rel_df.to_csv(os.path.join(dataset, f'{dataset}.grel'), index=False)

def generate_od():
    df = pd.read_csv(os.path.join(dataset, f'{dataset}.srel'))
    df['flow'] = [1] * len(df)
    df.to_csv(os.path.join(dataset, f'{dataset}.srel'), index=False)

def f0(time_str):
    # 定义时间字符串的格式
    time_format = "%a %b %d %H:%M:%S %z %Y"
    # 使用strptime函数将时间字符串转换为datetime对象
    dt = datetime.strptime(time_str, time_format)
    return dt.timestamp()

def f1(timestamp):
    # 使用datetime.fromtimestamp函数将时间戳转换为datetime对象
    dt = datetime.fromtimestamp(timestamp)
    # 定义要转换为的时间格式
    time_format = "%a %b %d %H:%M:%S %z %Y"
    # 使用strftime函数将datetime对象格式化为字符串
    formatted_time = dt.strftime(time_format)
    return formatted_time

def generate_checkin():
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    checkins_df = pd.read_csv(os.path.join(dataset, 'checkins.csv'))
    checkins_df['time'] = checkins_df['time'].map(f0) + checkins_df['timeoffset'] * 60
    checkins_df['time'] = checkins_df['time'].map(f1)
    rename_cols = {
        'user': 'user_id',
        'poi': 'location'
    }
    geo_df = geo_df.rename(columns=rename_cols)
    region_cnt = geo_df[geo_df['traffic_type'] == 'region'].shape[0]
    road_cnt = geo_df[geo_df['traffic_type'] == 'road'].shape[0]
    mp = {}
    for i, row in geo_df.iterrows():
        if row['traffic_type'] == 'poi':
            mp[row['poi_id']] = int(row['geo_uid']) - region_cnt - road_cnt
    def f2(x):
        return mp[x]
    checkins_df = checkins_df.rename(columns=rename_cols)
    checkins_df['type'] = ['trajectory'] * len(checkins_df)
    checkins_df['dyna_id'] = list(range(len(checkins_df)))
    users = list(checkins_df['user_id'].drop_duplicates())
    mp_usr = {}
    for i, user in enumerate(users):
        mp_usr[user] = i
    def f3(x):
        return mp_usr[x]
    checkins_df['user_id'] = checkins_df['user_id'].map(f3)
    checkins_df['location'] = checkins_df['location'].map(f2)
    pdb.set_trace()
    drop_cols = ['index', 'timeoffset']
    for drop_col in drop_cols:
        if drop_col in checkins_df.keys():
            checkins_df = checkins_df.drop(columns=drop_col, axis=1)
    checkins_df.to_csv(os.path.join(dataset, f'{dataset}_poi.dyna'), index=False)

def generate_road2region():
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    for i, row in geo_df.iterrows():
        if row['traffic_type'] == 'poi':
            break
        if row['type'] == 'LineString':
            geo_df.loc[i, 'traffic_type'] = 'road'
    geo_df.to_csv(os.path.join(dataset, f'{dataset}.geo'), index=False)
    g = []
    for i, row in geo_df.iterrows():
        if row['type'] not in ['Polygon', 'MultiPolygon']:
            break
        g.append(wkt.loads(row['geo_location']))
    
    # for i in range(len(g)):
    #     for j in range(i + 1, len(g)):
    #         assert not g[i].contains(g[j])
    # print('ok')
    rel_df = pd.read_csv(os.path.join(dataset, f'{dataset}.grel'))
    new_rows, new_rows2 = [], []
    rel_uid = len(rel_df)
    rel_uid2 = rel_uid

    geo_df = pd.read_csv(f'{dataset}/{dataset}.geo')
    num_regions = geo_df[geo_df['traffic_type'] == 'region'].shape[0]
    geo_df = geo_df[geo_df['traffic_type'] == 'road']
    cnt = 0
    for i, row in tqdm(geo_df.iterrows(), total=len(geo_df)):
        road = parse_linestring_coordinates(row['geo_location'])
        # pdb.set_trace()
        t = 0
        min_id = 0
        point = Point(road.coords[0])
        for j, region in enumerate(g):
            if region.contains(point):
                t += 1
                new_rows.append({
                    'rel_uid': rel_uid,
                    'type': 'geo',
                    'orig_geo_id': int(row['geo_uid']),
                    'dest_geo_id': j,
                    'rel_type': 'road2region'
                })
                rel_uid += 1
                new_rows.append({
                    'rel_uid': rel_uid,
                    'type': 'geo',
                    'orig_geo_id': j,
                    'dest_geo_id': int(row['geo_uid']),
                    'rel_type': 'region2road'
                })
                rel_uid += 1
            elif road.distance(region) < road.distance(g[min_id]):
                min_id = j
        if t != 1:
            assert t == 0
            new_rows2.append({
                'rel_uid': rel_uid2,
                'type': 'geo',
                'orig_geo_id': int(row['geo_uid']),
                'dest_geo_id': min_id,
                'rel_type': 'road2region'
            })
            rel_uid2 += 1
            new_rows2.append({
                'rel_uid': rel_uid2,
                'type': 'geo',
                'orig_geo_id': min_id,
                'dest_geo_id': int(row['geo_uid']),
                'rel_type': 'region2road'
            })
            rel_uid2 += 1
            cnt += 1
    print(f'invalid {cnt}, total {len(geo_df)}')
    pdb.set_trace()
    rel_df = rel_df.append(new_rows, ignore_index=True)
    rel_df = rel_df.append(new_rows2, ignore_index=True)
    rel_df.to_csv(os.path.join(dataset, f'{dataset}.grel'), index=False)

def get_middle_point(line:LineString):
    line_length = line.length
    middle_point_distance = line_length / 2
    middle_point = line.interpolate(middle_point_distance)
    return Point(middle_point.x, middle_point.y)

def merge_traj():
    t1 = pd.read_csv(os.path.join(dataset, 'traj_test.csv'))
    t2 = pd.read_csv(os.path.join(dataset, 'traj_train.csv'))
    t = pd.concat([t1, t2], ignore_index=True)
    t.to_csv(os.path.join(dataset, f'{dataset}_traj.csv'), index=False)

def gen_index_map(df, column, offset=0):
    index_map = {origin: index + offset
                    for index, origin in enumerate(df[column].drop_duplicates())}
    return index_map

def str2timestamp(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())

def str2timestampTZ(s):
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp())

def timestamp2str(timestamp, flag=False):
    dt = datetime.fromtimestamp(timestamp)
    if flag:
        formatted_datetime = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        formatted_datetime = dt.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime

def generate_map_match():
    if not os.path.exists(f'{dataset}_mm'):
        os.makedirs(f'{dataset}_mm')

    if not os.path.exists(os.path.join(f'{dataset}_mm', f'{dataset}.geo')):
        geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
        keys = geo_df.keys()
        for col in keys:
            if col not in ['geo_uid', 'type', 'geo_location']:
                geo_df = geo_df.drop(col, axis=1)
        geo_df = geo_df[geo_df['type'] == 'LineString']
        geo_df['geo_uid'] = list(range(len(geo_df)))
        geo_df.to_csv(os.path.join(f'{dataset}_mm', f'{dataset}.geo'), index=False)

    if not os.path.exists(os.path.join(f'{dataset}_mm', f'{dataset}.grel')):
        org_path = os.path.join(f'{dataset}', f'{dataset}.grel')
        new_path = os.path.join(f'{dataset}_mm', f'{dataset}.grel')
        os.system(f'cp {org_path} {new_path}')

    if not os.path.exists(os.path.join(f'{dataset}_mm', f'{dataset}.dyna')):
        offset = 30
        df = pd.read_csv(os.path.join(dataset, f'{dataset}_traj.csv'))
        mp = gen_index_map(df, 'TAXI_ID')
        dyna = {'time': [], 'user_id': [], 'geo_location': [], 'traj_uid': []}
        for _, row in tqdm(df.iterrows(), total=len(df)):
            coords = row['POLYLINE'][1:-1].replace('],[', ']|[').split('|')
            t = int(row['TIMESTAMP'])
            if len(coords) < 1:
                continue
            for i, coord in enumerate(coords):
                dyna['time'].append(t + i * offset)
                dyna['user_id'].append(mp[row['TAXI_ID']])
                dyna['geo_location'].append(coord)
                dyna['traj_uid'].append(_)
        dyna_df = pd.DataFrame(dyna)
        dyna_df['time'] = dyna_df['time'].map(timestamp2str)
        dyna_df['type'] = ['trajectory'] * len(dyna_df)
        dyna_df['dyna_id'] = list(range(len(dyna_df)))
        dyna_df = dyna_df.dropna()
        dyna_df.to_csv(os.path.join(f'{dataset}_mm', f'{dataset}.dyna'), index=False)
        # print(dyna_df)

    if not os.path.exists(os.path.join(f'{dataset}_mm', f'{dataset}.usr')):
        dyna_df = pd.read_csv(os.path.join(f'{dataset}_mm', f'{dataset}.dyna'))
        df = pd.DataFrame()
        df['usr_id'] = list(range(int(dyna_df['user_id'].max())))
        df.to_csv(os.path.join(f'{dataset}_mm', f'{dataset}.usr'), index=False)
    
def replace_road():
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    new_keys = ['highway', 'lanes', 'length', 'maxspeed']
    for key in new_keys:
        geo_df['road_' + key] = [None] * len(geo_df)
    road_df = pd.read_csv(os.path.join(dataset, f'{dataset}_roadmap_edge.geo'))
    keys = geo_df.keys()
    for key in keys:
        if key.startswith('road_'):
            geo_df = geo_df.drop(key, axis=1)
        elif key.startswith('region_') or key.startswith('poi_'):
            road_df[key] = [None] * len(road_df)
    region_df = geo_df[geo_df['traffic_type'] == 'region']
    poi_df = geo_df[geo_df['traffic_type'] == 'poi']
    road_df['geo_uid'] += len(region_df)
    geo_df = pd.concat([region_df, road_df, poi_df], axis=0)
    rename_cols = {
        'highway': 'road_highway',
        'lanes': 'road_lanes',
        'length': 'road_length',
        'maxspeed': 'road_maxspeed'
    }
    geo_df = geo_df.rename(columns=rename_cols)
    geo_df.to_csv(os.path.join(dataset, f'{dataset}.geo'), index=False)

def generate_dyna():
    df1 = pd.read_csv(os.path.join(dataset, f'{dataset}_train.csv'), sep=';')
    df2 = pd.read_csv(os.path.join(dataset, f'{dataset}_test.csv'), sep=';')
    df3 = pd.read_csv(os.path.join(dataset, f'{dataset}_eval.csv'), sep=';')
    df = pd.concat([df1, df2, df3], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    dyna_dict = {'time': [], 'user_id': [], 'traj_uid': [], 'total_traj_uid': [], 'geo_uid': []}
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    region_cnt = len(geo_df[geo_df['traffic_type'] == 'region'])
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # pdb.set_trace()
        path = eval(row['path'])
        tlist = eval(row['tlist'])
        usr_id = int(row['usr_id'])
        for i in range(len(path)):
            dyna_dict['time'].append(timestamp2str(tlist[i]))
            dyna_dict['geo_uid'].append(path[i] + region_cnt)
            dyna_dict['user_id'].append(usr_id)
            dyna_dict['traj_uid'].append(_)
            dyna_dict['total_traj_uid'].append(_)
    dyna_df = pd.DataFrame(dyna_dict)
    dyna_df['dyna_id'] = list(range(len(dyna_df)))
    dyna_df['type'] = ['trajectory'] * len(dyna_df)
    mp = gen_index_map(dyna_df, 'user_id')
    dyna_df['user_id'] = dyna_df['user_id'].map(mp)
    # print(dyna_df)
    dyna_df.to_csv(os.path.join(dataset, f'{dataset}.dyna'), index=False, sep=',')

def add_road_geometry():
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    geo_df['road_geometry'] = [None] * len(geo_df)
    for i, row in tqdm(geo_df.iterrows(), total=len(geo_df)):
        if row['traffic_type'] == 'poi':
            break
        if row['traffic_type'] == 'region':
            continue
        tmp = parse_linestring_coordinates(row['geo_location'])
        geo_df.loc[i, 'road_geometry'] = tmp
        geo_df.loc[i, 'geo_location'] = tmp
    # pdb.set_trace()
    geo_df.to_csv(os.path.join(dataset, f'{dataset}.geo'), index=False)

def add_road_id():
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    geo_df['road_id'] = [None] * len(geo_df)
    cnt = 0
    for i, row in tqdm(geo_df.iterrows(), total=len(geo_df)):
        if row['traffic_type'] == 'poi':
            break
        if row['traffic_type'] == 'road':
            geo_df.loc[i, 'road_id'] = int(cnt)
            cnt += 1
    geo_df.to_csv(os.path.join(dataset, f'{dataset}.geo'), index=False)
    
def add_road_features():
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    # road_df = pd.read_csv(os.path.join(dataset, f'road_{dataset}.csv'))
    features = ['oneway', 'bridge', 'width', 'junction', 'tunnel', 'roundabout']
    region_cnt = geo_df[geo_df['traffic_type'] == 'region'].shape[0]
    poi_cnt = geo_df[geo_df['traffic_type'] == 'poi'].shape[0]
    road_cnt = geo_df[geo_df['traffic_type'] == 'road'].shape[0]
    pdb.set_trace()
    for feature in features:
        feature_col = f'road_{feature}'
        if feature_col not in geo_df.keys():
            print(f'add {feature_col}')
            geo_df[feature_col] = [None] * region_cnt + [0] * road_cnt + [None] * poi_cnt
        # geo_df[f'road_{feature}'] = [None] * region_cnt + list(road_df[feature]) + [None] * poi_cnt
    geo_df.to_csv(os.path.join(dataset, f'{dataset}.geo'), index=False)

def generate_od_from_points():
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    g = []
    for i, row in geo_df.iterrows():
        if row['type'] not in ['Polygon', 'MultiPolygon']:
            break
        g.append(wkt.loads(row['geo_location']))
    
    df = pd.read_csv(f'{dataset}/{dataset}_traj.csv')
    # df = df[:100]
    gap = 15
    od_dict = {'start_time': [], 'end_time': [], 'orig_geo_id': [], 'dest_geo_id': []}
    invalid = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            linestring = parse_linestring_coordinates(row['POLYLINE'])
        except:
            invalid += 1
            continue
        origin = Point(linestring.coords[0])
        destination = Point(linestring.coords[-1])
        orig_geo_id, dest_geo_id = None, None
        start_time = timestamp2str(row['TIMESTAMP'])
        end_time = timestamp2str(row['TIMESTAMP'] + (len(linestring.coords) - 1) * gap)
        for j, region in enumerate(g):
            if region.contains(origin):
                orig_geo_id = j
            if region.contains(destination):
                dest_geo_id = j
            if orig_geo_id is not None and dest_geo_id is not None:
                break
        if orig_geo_id is None or dest_geo_id is None or orig_geo_id == dest_geo_id:
            invalid += 1
            continue
        # assert orig_geo_id is not None and dest_geo_id is not None
        od_dict['start_time'].append(start_time)
        od_dict['end_time'].append(end_time)
        od_dict['orig_geo_id'].append(orig_geo_id)
        od_dict['dest_geo_id'].append(dest_geo_id)
    print(f'invalid: {invalid}, total: {len(df)}')
    df = pd.DataFrame(od_dict)
    df['dyna_id'] = list(range(len(df)))
    df['flow'] = [1] * len(df)
    df.to_csv(f'{dataset}/{dataset}.srel', index=False)

def reindex():
    dyna_df = pd.read_csv(os.path.join(dataset, f'{dataset}.dyna'))
    mp = gen_index_map(dyna_df, 'user_id')
    dyna_df['user_id'] = dyna_df['user_id'].map(mp)
    pdb.set_trace()
    dyna_df.to_csv(os.path.join(dataset, f'{dataset}.dyna'), index=False)

def generate_od_from_road_traj():
    dyna_df = pd.read_csv(f'{dataset}/{dataset}.cdtraj')
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
    geo_df = pd.read_csv(f'{dataset}/{dataset}.geo')
    num_regions = geo_df[geo_df['traffic_type'] == 'region'].shape[0]
    for _, row in tqdm(dyna_df.iterrows(), total=dyna_df.shape[0]):
        # idx = int(row['total_traj_uid'])
        if lst_traj_uid != row['traj_uid'] or lst_usr_id != row['user_id']:  # 轨迹划分依据还存疑，靠 traj_uid 和 total_traj_uid 都不行
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
        tlist[idx].append(str2timestampTZ(row['time']))
        path[idx].append(row['geo_uid'] - num_regions)
        lst_traj_uid = row['traj_uid']
        lst_usr_id = row['user_id']
    rel_df = pd.read_csv(f'{dataset}/{dataset}.grel')
    rel_df = rel_df[rel_df['rel_type'] == 'road2region']
    road2region = [0 for _ in range(int(rel_df['orig_geo_id']) + 100)]
    for i, row in rel_df.iterrows():
        road2region[int(row['orig_geo_id'])] = int(row['dest_geo_id'])
    start_time, end_time, orig_geo_id, dest_geo_id = [], [], [], []
    cnt = 0
    for i in range(id[-1] + 1):
        o = road2region[path[i][0]]
        d = road2region[path[i][-1]]
        if o == d:
            cnt += 1
            continue
        start_time.append(timestamp2str(tlist[i][0]))
        end_time.append(timestamp2str(tlist[i][-1]))
        orig_geo_id.append(o)
        dest_geo_id.append(d)
    print(f'total {id[-1] + 1}, invalid {cnt}')
    df = pd.concat(
        [
            pd.Series(orig_geo_id, name='orig_geo_id'),
            pd.Series(dest_geo_id, name='dest_geo_id'),
            pd.Series(start_time, name='start_time'),
            pd.Series(end_time, name='end_time')
        ], axis=1)
    df['flow'] = [1] * len(df)
    df['dyna_id'] = list(range(len(df)))
    pdb.set_trace()
    df.to_csv(f'{dataset}/{dataset}.srel', index=False)

def mp_property(x):
    if isinstance(x, str):
        x = eval(x)
    if isinstance(x, list):
        return int(x[0])
    elif x is None or math.isnan(x):
        return 0
    else:
        return int(x)

def modify_property(property):
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    for i, row in tqdm(geo_df.iterrows(), total=len(geo_df)):
        if row['traffic_type'] == 'poi':
            break
        if row['traffic_type'] == 'region':
            continue
        try:
            geo_df.loc[i, property] = mp_property(row[property])
        except:
            pdb.set_trace()
    # pdb.set_trace()
    geo_df.to_csv(os.path.join(dataset, f'{dataset}.geo'), index=False)

def str2line():
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    for i, row in geo_df.iterrows():
        if row['traffic_type'] == 'poi':
            break
        if row['traffic_type'] == 'region':
            continue
        geo_df.loc[i, 'geo_location'] = parse_linestring_coordinates(row['geo_location'])
    pdb.set_trace()
    geo_df.to_csv(os.path.join(dataset, f'{dataset}.geo'), index=False)

def mp_highway(x):
    highway_list = ['trunk', 'road', 'motorway', 'residential', 'trunk_link', 'tertiary',
                                   'primary_link', 'secondary', 'tertiary_link', 'secondary_link', 'motorway_link',
                                   'living_street', 'primary']
    mp = {}
    for i, v in enumerate(highway_list):
        mp[v] = i + 1
    if isinstance(x, str):
        try:
            x = eval(x)
        except:
            pass
    if isinstance(x, list):
        x = x[0]
    if x is None or x == 'unclassified':
        return 0
    else:
        return mp[x]

def modify_highway():
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    for i, row in geo_df.iterrows():
        if row['traffic_type'] == 'poi':
            break
        if row['traffic_type'] == 'region':
            continue
        geo_df.loc[i, 'road_highway'] = mp_highway(row['road_highway'])
    pdb.set_trace()
    geo_df.to_csv(os.path.join(dataset, f'{dataset}.geo'), index=False)

def mp_col(entity_type, col, mp):
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    for i, row in tqdm(geo_df.iterrows(), total=len(geo_df)):
        if row['traffic_type'] != entity_type:
            continue
        t = entity_type + '_' + col
        geo_df.loc[i, t] = mp(row[t])
    pdb.set_trace()
    geo_df.to_csv(os.path.join(dataset, f'{dataset}.geo'), index=False)

def add_offset():
    dyna_df = pd.read_csv(os.path.join(dataset, f'{dataset}_poi.dyna'))
    checkins_df = pd.read_csv(os.path.join(dataset, 'checkins.csv'))
    dyna_df['time'] = dyna_df['time'].map(f0) + checkins_df['offset'] * 60
    dyna_df['time'] = dyna_df['time'].map(f1)
    pdb.set_trace()
    dyna_df.to_csv(os.path.join(dataset, f'{dataset}_poi.dyna'), index=False)

def add_rel_road_id():
    geo_df = pd.read_csv(os.path.join(dataset, f'{dataset}.geo'))
    num_regions = geo_df[geo_df['traffic_type'] == 'region'].shape[0]
    rel_df = pd.read_csv(os.path.join(dataset, f'{dataset}.grel'))
    for i, row in tqdm(rel_df.iterrows(), total=len(rel_df)):
        if row['rel_type'] != 'road2road':
            break
        rel_df.loc[i, 'orig_geo_id'] += num_regions
        rel_df.loc[i, 'dest_geo_id'] += num_regions
    rel_df.to_csv(os.path.join(dataset, f'{dataset}.grel'), index=False)
