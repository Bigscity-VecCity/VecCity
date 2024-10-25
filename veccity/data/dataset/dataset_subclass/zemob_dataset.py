import os
from datetime import datetime
from logging import getLogger
import pandas as pd
import numpy as np
import geatpy as ea
import torch
import geopandas as gpd
from veccity.data.dataset import AbstractDataset
from veccity.data.preprocess import preprocess_all, cache_dir
from veccity.utils import ensure_dir, need_train


class ZEMobDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        preprocess_all(config)
        self._logger = getLogger()
        if not need_train(config):
            return
        self.dataset = self.config.get('dataset', '')
        self.device = config.get('device', torch.device('cpu'))
        self.od_label_path = os.path.join(cache_dir, self.dataset, 'od_region_train_od.npy')
        self.mob_adj = np.load(self.od_label_path)
        self.num_regions = self.mob_adj.shape[0]
        self.num_nodes = self.num_regions
        # 用于求解ppmi_matrix（即公式中的M）的中间变量，三个字典分别存储了event的出现次数，zone的出现次数，zone和event的共现次数
        self.mobility_events = dict()
        self.zones = dict()
        self.co_occurs = dict()
        self.co_occurs_num = 0

        # 用于求解G_matrix的中间变量，三组变量分别对应文章中的A、P、T
        self.max_gen = config.get('MaxGen')
        self.NIND = config.get('NIND')
        self.arrive_num_weekday = np.zeros(self.num_nodes)
        self.arrive_num_weekend = np.zeros(self.num_nodes)
        self.leave_num_weekday = np.zeros(self.num_nodes)
        self.leave_num_weekend = np.zeros(self.num_nodes)
        self.T_weekday = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.T_weekend = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.distance = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        # ppmi_matrix为二维矩阵，维度为z*e
        self.ppmi_matrix = []

        # G_matrix为二维矩阵，维度为z*e
        self.G_matrix = []

        # zone的数量
        self.zone_num = 0

        # mobility_event的数量
        self.mobility_event_num = 0

        # 设置中间变量的缓存路径
        ensure_dir('./veccity/cache/dataset_cache/{}/ZEMob'.format(self.dataset))
        self.distance_matrix_path = './veccity/cache/dataset_cache/{}/ZEMob/distance_matrix.npy'.format(self.dataset)
        self.ppmi_matrix_path = './veccity/cache/dataset_cache/{}/ZEMob/ppmi_matrix.npy'.format(self.dataset)
        self.G_matrix_path = './veccity/cache/dataset_cache/{}/ZEMob/G_matrix.npy'.format(self.dataset)

        self.mobility_events_path = './veccity/cache/dataset_cache/{}/ZEMob/mobility_events.npy'.format(self.dataset)

        self.A_wd_path = './veccity/cache/dataset_cache/{}/ZEMob/A_wd.npy'.format(self.dataset)
        self.A_we_path = './veccity/cache/dataset_cache/{}/ZEMob/A_we.npy'.format(self.dataset)
        self.T_wd_path = './veccity/cache/dataset_cache/{}/ZEMob/T_wd.npy'.format(self.dataset)
        self.T_we_path = './veccity/cache/dataset_cache/{}/ZEMob/T_we.npy'.format(self.dataset)
        self.P_wd_path = './veccity/cache/dataset_cache/{}/ZEMob/P_wd.npy'.format(self.dataset)
        self.P_we_path = './veccity/cache/dataset_cache/{}/ZEMob/P_we.npy'.format(self.dataset)

        self.label_path = './veccity/cache/dataset_cache/{}/ZEMob/label.npy'.format(self.dataset)

        self.process_mobility_data()
        self.construct_distance_matrix()
        self.construct_gravity_matrix()
        self.construct_ppmi_matrix()

    def process_mobility_data(self):
        """
        统计mobility event和co-occurs的信息，用于创建ppmi matrix和gravity matrix

        :return:
        """
        mobility_event_index = 0
        od_file = pd.read_csv(os.path.join(cache_dir, self.dataset, 'od_region_train.csv'))
        for _, row in od_file.iterrows():
            # 得到起始zone和起始zone对应的mobility_event
            origin_region = int(row['origin_id'])
            origin_date = datetime.fromtimestamp(row['start_time'])
            origin_hour = origin_date.hour
            origin_date_type = 1 if origin_date.weekday() in range(5) else 0
            origin_mobility_event = (origin_region, origin_hour, origin_date_type, 'o')

            # 得到目的zone和目的zone对应的mobility_event
            destination_region = int(row['destination_id'])
            destination_date = datetime.fromtimestamp(row['end_time'])
            destination_hour = destination_date.hour
            destination_date_type = 1 if destination_date.weekday() in range(5) else 0
            destination_mobility_event = (destination_region, destination_hour, destination_date_type, 'd')


            # 计算每个mobility_event出现的次数，同时给每个mobility一个index
            # 即一个一维字典，key是mobility_event，value是一个二元列表，第一个值是co-occur的次数，第二个值是index
            if self.mobility_events.get(origin_mobility_event) is None:
                self.mobility_events[origin_mobility_event] = [1, mobility_event_index]
                mobility_event_index += 1
            else:
                self.mobility_events[origin_mobility_event][0] += 1
            if self.mobility_events.get(destination_mobility_event) is None:
                self.mobility_events[destination_mobility_event] = [1, mobility_event_index]
                mobility_event_index += 1
            else:
                self.mobility_events[destination_mobility_event][0] += 1

            # 计算每个zone出现的次数，即一个一维字典，key是zone，value是co-occur的次数
            if self.zones.get(origin_region) is None:
                self.zones[origin_region] = 1
            else:
                self.zones[origin_region] += 1
            if self.zones.get(destination_region) is None:
                self.zones[destination_region] = 1
            else:
                self.zones[destination_region] += 1

            # 计算每种co-occur的次数，即一个二维字典，第一维的key是zone，第二维的key是mobility_event，value是co-occur的次数
            if self.co_occurs.get(origin_region) is None:
                event_dict = dict()
                event_dict[destination_mobility_event] = 1
                self.co_occurs[origin_region] = event_dict
            else:
                event_dict = self.co_occurs[origin_region]
                if event_dict.get(destination_mobility_event) is None:
                    event_dict[destination_mobility_event] = 1
                else:
                    event_dict[destination_mobility_event] += 1
            if self.co_occurs.get(destination_region) is None:
                event_dict = dict()
                event_dict[origin_mobility_event] = 1
                self.co_occurs[destination_region] = event_dict
            else:
                event_dict = self.co_occurs[destination_region]
                if event_dict.get(origin_mobility_event) is None:
                    event_dict[origin_mobility_event] = 1
                else:
                    event_dict[origin_mobility_event] += 1

            # 统计总的co-occur的数量
            self.co_occurs_num += 2

            # 计算A、P、T。
            # 出发的时间在工作日的话，这个pattern就属于工作日。
            # 出发的时间在周末的话，这个pattern就属于周末。
            if origin_date_type == 1:
                self.arrive_num_weekday[destination_region] += 1
                self.leave_num_weekday[origin_region] += 1
                self.T_weekday[origin_region][destination_region] += 1
            else:
                self.arrive_num_weekend[destination_region] += 1
                self.leave_num_weekend[origin_region] += 1
                self.T_weekend[origin_region][destination_region] += 1

        # 统计zone和mobility_event的数量
        self.zone_num = self.num_nodes
        self.mobility_event_num = len(self.mobility_events)

        # 保存记录了mobility_events信息的字典
        if not os.path.exists(self.mobility_events_path):
            np.save(self.mobility_events_path, self.mobility_events)

        # 保存A、P、T
        if not os.path.exists(self.A_wd_path):
            np.save(self.A_wd_path, self.arrive_num_weekday)
        if not os.path.exists(self.A_we_path):
            np.save(self.A_we_path, self.arrive_num_weekend)
        if not os.path.exists(self.P_wd_path):
            np.save(self.P_wd_path, self.leave_num_weekday)
        if not os.path.exists(self.P_we_path):
            np.save(self.P_we_path, self.leave_num_weekend)
        if not os.path.exists(self.T_wd_path):
            np.save(self.T_wd_path, self.T_weekday)
        if not os.path.exists(self.T_we_path):
            np.save(self.T_we_path, self.T_weekend)
        self._logger.info("finish constructing mobility basic data")

    def construct_distance_matrix(self):
        if os.path.exists(self.distance_matrix_path):
            self.distance = np.load(self.distance_matrix_path)
            self._logger.info("finish constructing distance matrix")
            return
        region_geo_file = pd.read_csv(os.path.join('raw_data', self.dataset, self.dataset + '.geo'))
        self.region_geometry = gpd.GeoSeries.from_wkt(region_geo_file['region_geometry'].dropna())
        centroid = self.region_geometry.centroid
        for i in range(self.zone_num):
            for j in range(i, self.zone_num):
                distance = centroid[i].distance(centroid[j])
                self.distance[i][j] = distance
                self.distance[j][i] = distance
        np.save(self.distance_matrix_path, self.distance)
        self._logger.info("finish constructing distance matrix")

    def construct_gravity_matrix(self):
        """
        创建gravity matrix

        :return:
        """

        # 遗传算法求解工作日的beta
        problem = GravityMatrix(self.zone_num, self.device, self.arrive_num_weekday, self.leave_num_weekday, self.T_weekday, self.distance)
        algorithm = ea.soea_SEGA_templet(
            problem,
            ea.Population(Encoding='RI', NIND=self.NIND),
            MAXGEN=self.max_gen,
            trappedValue=1e-6,
            maxTrappedCount=10,
            logTras=0,
        )
        res = ea.optimize(
            algorithm,
            verbose=True,
            drawing=0,
            outputMsg=False,
            drawLog=False,
            saveFlag=False
        )
        beta_weekday = res['Vars'][0][0]

        # 遗传算法求解周末的beta
        problem = GravityMatrix(self.zone_num, self.device, self.arrive_num_weekend, self.leave_num_weekend, self.T_weekend, self.distance)
        algorithm = ea.soea_SEGA_templet(
            problem,
            ea.Population(Encoding='RI', NIND=self.NIND),
            MAXGEN=self.max_gen,
            trappedValue=1e-6,
            maxTrappedCount=10,
            logTras=0,
        )
        res = ea.optimize(
            algorithm,
            verbose=True,
            drawing=0,
            outputMsg=False,
            drawLog=False,
            saveFlag=False
        )
        beta_weekend = res['Vars'][0][0]

        # 求解工作日的G
        F = np.exp((-beta_weekday) * self.distance)
        G_weekday = (F * self.arrive_num_weekday) / np.matmul(F, self.arrive_num_weekday.reshape(self.zone_num, 1))

        # 求解周末的G
        F = np.exp((-beta_weekend) * self.distance)
        G_weekend = (F * self.arrive_num_weekend) / np.matmul(F, self.arrive_num_weekend.reshape(self.zone_num, 1))


        # 上述求解的G维度为z*z，如果要转化为z*e，还需要根据每个event进行筛选，得到z*e维度的G*矩阵
        self.G_matrix = np.zeros((self.zone_num, self.mobility_event_num))
        for zone_id in range(self.zone_num):
            for mobility_event in self.mobility_events.keys():
                mb_id = self.mobility_events[mobility_event][1]
                mb_type = mobility_event[2]
                mb_sta = mobility_event[3]
                mb_zone_id = mobility_event[0]
                if mb_type == 1:
                    if mb_sta == 'o':
                        self.G_matrix[zone_id][mb_id] = G_weekday[mb_zone_id][zone_id]
                    else:
                        self.G_matrix[zone_id][mb_id] = G_weekday[zone_id][mb_zone_id]
                else:
                    if mb_sta == 'o':
                        self.G_matrix[zone_id][mb_id] = G_weekend[mb_zone_id][zone_id]
                    else:
                        self.G_matrix[zone_id][mb_id] = G_weekend[zone_id][mb_zone_id]

        # 保存G matrix
        if not os.path.exists(self.G_matrix_path):
            np.save(self.G_matrix_path, self.G_matrix)

        self._logger.info("finish constructing gravity matrix, beta_wd is {}, beta_we is {}".format(str(beta_weekday), str(beta_weekend)))

    def construct_ppmi_matrix(self):
        """
        创建ppmi矩阵

        :return:
        """
        if os.path.exists(self.ppmi_matrix_path):
            self.ppmi_matrix = np.load(self.ppmi_matrix_path)
            self._logger.info("finish constructing ppmi matrix")
            return
        self.ppmi_matrix = np.zeros((self.zone_num, len(self.mobility_events)), dtype=np.float32)
        for region_id in self.co_occurs.keys():
            tmp_mobility_events = self.co_occurs[region_id].keys()
            for mobility_event in tmp_mobility_events:
                mb_id = self.mobility_events[mobility_event][1]
                self.ppmi_matrix[region_id][mb_id] = max(0, np.log2(
                    (self.co_occurs[region_id][mobility_event] * self.co_occurs_num) /
                    (self.zones[region_id] * self.mobility_events[mobility_event][0])
                ))
        np.save(self.ppmi_matrix_path, self.ppmi_matrix)
        self._logger.info("finish constructing ppmi matrix")
    
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
        return {'ppmi_matrix': self.ppmi_matrix, 'G_matrix': self.G_matrix, 
                'region_num': self.zone_num, 'mobility_event_num': self.mobility_event_num}


# 遗传算法求解gravity matrix的模型
class GravityMatrix(ea.Problem):
    def __init__(self, zone_num, device, A, P, T, D):
        ea.Problem.__init__(
            self, name='GravityMatrix', M=1, maxormins=[1], Dim=1,
            varTypes=[0], lb=[-100], ub=[100], lbin=[1], ubin=[1]
        )
        self.zone_num = zone_num
        self.device = device
        self.A = torch.tensor(A, dtype=torch.float32).to(self.device)
        self.P = torch.tensor(P, dtype=torch.float32).to(self.device)
        self.T = torch.tensor(T, dtype=torch.float32).to(self.device)
        self.D = torch.tensor(D, dtype=torch.float32).to(self.device)

    # 给每个个体计算目标函数值
    # 只需要关注这个目标函数，即为文章第三页最后提到的需要使用遗传算法进行最小化的那个函数
    # 注意T_hat的计算，这里将文章中的P*G中的G进行了展开，然后分别求解了展开后的分子和分母
    def evalVars(self, betas):
        res = np.zeros_like(betas)
        for i in range(len(res)):
            beta = betas[i][0]
            F = torch.exp((-beta) * self.D)
            G_denominator = torch.matmul(F, self.A.reshape(self.zone_num, 1))
            G_numerator = torch.matmul(self.P.reshape(self.zone_num, 1), self.A.reshape(1, self.zone_num)) * F
            T_hat = G_numerator / G_denominator
            res[i][0] = torch.sum((self.T - T_hat) ** 2)
        return res
