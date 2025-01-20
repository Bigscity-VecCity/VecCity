import math
import os
import ast
import torch
import sklearn
import numpy as np
import pandas as pd
from logging import getLogger
from collections import Counter
from random import *
from itertools import zip_longest
from veccity.data.utils import pad_session_data_one
from veccity.data.dataset import AbstractDataset
from veccity.upstream.poi_representation.utils import next_batch, init_seed
from veccity.upstream.poi_representation.tale import TaleData
from veccity.upstream.poi_representation.poi2vec import P2VData
from veccity.upstream.poi_representation.teaser import TeaserData
from veccity.upstream.poi_representation.w2v import SkipGramData
from datetime import datetime
# from veccity.upstream.poi_representation.cacsr import CacsrData
from veccity.utils import parse_time, cal_timeoff, ensure_dir
import pickle

class POIRepresentationDataLoader:

    def __init__(self, config, data_feature, data):
        self.config = config
        self.data_feature = data_feature
        self.data = data
        self.batch_size = self.config.get('batch_size', 64)
        self.device = self.config.get('device', torch.device('cpu'))
        self.model_name = self.config.get('model')
        self.w2v_window_size = self.config.get('w2v_window_size', 1)
        self.skipgram_neg = self.config.get('skipgram_neg', 5)
        w2v_data = self.data_feature.get('w2v_data')
        self.embed_train_users, self.embed_train_sentences, self.embed_train_weekdays, \
        self.embed_train_timestamp, _length = zip(*w2v_data)
        seed = self.config.get('seed', 0)
        init_seed(seed)

    def next_batch(self):
        pass


class CTLEDataLoader(POIRepresentationDataLoader):

    def gen_random_mask(self, src_valid_lens, src_len, mask_prop):
        """
        @param src_valid_lens: valid length of sequence, shape (batch_size)
        """
        # all_index = np.arange((batch_size * src_len)).reshape(batch_size, src_len)
        # all_index = shuffle_along_axis(all_index, axis=1)
        # mask_count = math.ceil(mask_prop * src_len)
        # masked_index = all_index[:, :mask_count].reshape(-1)
        # return masked_index
        index_list = []
        for batch, l in enumerate(src_valid_lens):
            mask_count = torch.ceil(mask_prop * l).int()
            masked_index = torch.randperm(l)[:mask_count]
            masked_index += src_len * batch
            index_list.append(masked_index)
        return torch.cat(index_list).long().to(src_valid_lens.device)

    def gen_casual_mask(self, seq_len, include_self=True):
        """
        Generate a casual mask which prevents i-th output element from
        depending on any input elements from "the future".
        Note that for PyTorch Transformer model, sequence mask should be
        filled with -inf for the masked positions, and 0.0 else.

        :param seq_len: length of sequence.
        :return: a casual mask, shape (seq_len, seq_len)
        """
        if include_self:
            mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
        else:
            mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
        return mask.bool()

    def next_batch(self):
        mask_prop = self.config.get('mask_prop', 0.2)
        num_vocab = self.data_feature.get('num_loc')
        user_ids, src_tokens, src_weekdays, src_ts, src_lens, _, _, _, _ = zip(*self.data)
        for batch in next_batch(sklearn.utils.shuffle(list(zip(src_tokens, src_weekdays, src_ts, src_lens))),
                                batch_size=self.batch_size):
            # Value filled with num_loc stands for masked tokens that shouldn't be considered.
            src_batch, _, src_t_batch, src_len_batch = zip(*batch)
            src_batch = np.transpose(np.array(list(zip_longest(*src_batch, fillvalue=num_vocab))))
            src_t_batch = np.transpose(np.array(list(zip_longest(*src_t_batch, fillvalue=0))))

            src_batch = torch.tensor(src_batch).long().to(self.device)
            src_t_batch = torch.tensor(src_t_batch).float().to(self.device)
            hour_batch = (src_t_batch % (24 * 60 * 60) / 60 / 60).long()

            batch_len, src_len = src_batch.size(0), src_batch.size(1)
            src_valid_len = torch.tensor(src_len_batch).long().to(self.device)

            mask_index = self.gen_random_mask(src_valid_len, src_len, mask_prop=mask_prop)

            src_batch = src_batch.reshape(-1)
            hour_batch = hour_batch.reshape(-1)
            origin_tokens = src_batch[mask_index]  # (num_masked)
            origin_hour = hour_batch[mask_index]

            # Value filled with num_loc+1 stands for special token <mask>.
            masked_tokens = \
                src_batch.index_fill(0, mask_index, num_vocab + 1).reshape(batch_len, -1)
            # (batch_size, src_len)

            yield origin_tokens, origin_hour, masked_tokens, src_t_batch, mask_index


class HierDataLoader(POIRepresentationDataLoader):

    def next_batch(self):
        user_ids, src_tokens, src_weekdays, src_ts, src_lens,_,_,_,_ = zip(*self.data)
        for batch in next_batch(sklearn.utils.shuffle(list(zip(src_tokens, src_weekdays, src_ts, src_lens))),
                                batch_size=self.batch_size):
            src_token, src_weekday, src_t, src_len= zip(*batch)
            src_token, src_weekday = [
                torch.from_numpy(np.transpose(np.array(list(zip_longest(*item, fillvalue=0))))).long().to(self.device)
                for item in (src_token, src_weekday)]
            src_t = torch.from_numpy(np.transpose(np.array(list(zip_longest(*src_t, fillvalue=0))))).float().to(
                self.device)
            src_len = torch.tensor(src_len).long().to(self.device)

            src_hour = (src_t % (24 * 60 * 60) / 60 / 60).long()
            src_duration = ((src_t[:, 1:] - src_t[:, :-1]) % (24 * 60 * 60) / 60 / 60).long()
            src_duration = torch.clamp(src_duration, 0, 23)
            yield src_token, src_weekday, src_hour, src_duration, src_len


class TaleDataLoader(POIRepresentationDataLoader):

    def __init__(self, config, data_feature, data):
        super().__init__(config, data_feature, data)
        if self.model_name == 'Tale':
            tale_slice = config.get('slice', 60)
            tale_span = config.get('span', 0)
            tale_indi_context = config.get('indi_context', True)
            self.dataset = TaleData(self.embed_train_sentences, self.embed_train_timestamp, tale_slice, tale_span,
                                    indi_context=tale_indi_context)
        elif self.model_name == 'POI2Vec':
            poi2vec_theta = data_feature.get('theta', 0.1)
            poi2vec_indi_context = config.get('indi_context', False)
            id2coor_df = data_feature.get('id2coor_df')
            self.dataset = P2VData(self.embed_train_sentences, id2coor_df, theta=poi2vec_theta,
                                   indi_context=poi2vec_indi_context)
        self.train_set = self.dataset.get_path_pairs(self.w2v_window_size)

    def next_batch(self):
        embed_epoch = self.config.get('embed_epoch', 5)
        batch_count = math.ceil(embed_epoch * len(self.train_set) / self.batch_size)
        for pair_batch in next_batch(sklearn.utils.shuffle(self.train_set), self.batch_size):
            flatten_batch = []
            for row in pair_batch:
                flatten_batch += [[row[0], p, n, pr] for p, n, pr in zip(*row[1:])]

            context, pos_pairs, neg_pairs, prop = zip(*flatten_batch)
            context = torch.tensor(context).long().to(self.device)
            pos_pairs, neg_pairs = (
                torch.tensor(list(zip_longest(*item, fillvalue=0))).long().to(self.device).transpose(0, 1)
                for item in (pos_pairs, neg_pairs)
            )  # (batch_size, longest)
            prop = torch.tensor(prop).float().to(self.device)
            yield batch_count, context, pos_pairs, neg_pairs, prop


class TeaserDataLoader(POIRepresentationDataLoader):

    def __init__(self, config, data_feature, data):
        super().__init__(config, data_feature, data)
        teaser_num_ne = config.get('num_ne', 0)  # (number of unvisited locations)
        teaser_num_nn = config.get('num_nn', 0)  # (number of non-neighbor locations)
        teaser_indi_context = config.get('indi_context', False)
        coor_mat = data_feature.get('coor_mat')
        self.dataset = TeaserData(self.embed_train_users, self.embed_train_sentences, self.embed_train_weekdays,
                                  coor_mat,
                                  num_ne=teaser_num_ne, num_nn=teaser_num_nn,
                                  indi_context=teaser_indi_context)
        self.pos_pairs = self.dataset.gen_pos_pairs(self.w2v_window_size)

    def next_batch(self):
        num_neg = self.skipgram_neg
        embed_epoch = self.config.get('embed_epoch', 5)
        batch_count = math.ceil(embed_epoch * len(self.pos_pairs) / self.batch_size)
        for pair_batch in next_batch(sklearn.utils.shuffle(self.pos_pairs), self.batch_size):
            neg_v = self.dataset.get_neg_v_sampling(len(pair_batch), num_neg)
            neg_v = torch.tensor(neg_v).long().to(self.device)

            user, pos_u, week, pos_v, neg_ne, neg_nn = zip(*pair_batch)
            user, pos_u, week, pos_v, neg_ne, neg_nn = (torch.tensor(item).long().to(self.device)
                                                        for item in (user, pos_u, week, pos_v, neg_ne, neg_nn))
            yield batch_count, pos_u, pos_v, neg_v, user, week, neg_ne, neg_nn


class SkipGramDataLoader(POIRepresentationDataLoader):

    def __init__(self, config, data_feature, data):
        super().__init__(config, data_feature, data)
        self.dataset = SkipGramData(self.embed_train_sentences)
        window_size = self.w2v_window_size
        self.pos_pairs = self.dataset.gen_pos_pairs(window_size)

    def next_batch(self):
        num_neg = self.skipgram_neg
        embed_epoch = self.config.get('embed_epoch', 5)
        batch_count = math.ceil(embed_epoch * len(self.pos_pairs) / self.batch_size)
        for pair_batch in next_batch(sklearn.utils.shuffle(self.pos_pairs), self.batch_size):
            neg_v = self.dataset.get_neg_v_sampling(len(pair_batch), num_neg)
            neg_v = torch.tensor(neg_v).long().to(self.device)

            pos_u, pos_v = zip(*pair_batch)
            pos_u, pos_v = (torch.tensor(item).long().to(self.device)
                            for item in (pos_u, pos_v))
            yield batch_count, pos_u, pos_v, neg_v
            
class CacsrDataLoader(POIRepresentationDataLoader):

    def next_batch(self):
        num_vocab = self.data_feature.get('num_loc')
        user_ids, src_tokens, src_weekdays, src_ts, src_lens, _, _, _, _ = zip(*self.data)
        for batch in next_batch(sklearn.utils.shuffle(list(zip(user_ids,src_tokens, src_ts, src_lens))),
                                batch_size=self.batch_size):
            batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)  
            src_user,src_batch, src_t_batch, src_len_batch = zip(*batch)
            src_batch = torch.from_numpy(np.transpose(np.array(list(zip_longest(*src_batch, fillvalue=num_vocab))))).long().to(self.device)
            src_t_batch = np.transpose(np.array(list(zip_longest(*src_t_batch, fillvalue=0))))
            hour_batch = (src_t_batch % (24 * 60 * 60) / 60 / 60)
            hour_batch=torch.from_numpy(hour_batch).long().to(self.device)
            src_len_batch = torch.tensor(src_len_batch).long().to(self.device)
            src_user = torch.tensor(src_user).long().to(self.device)

            yield src_batch, hour_batch, src_len_batch,src_user       
        
    

def get_dataloader(config, data_feature, data):
    model_name = config.get('model')
    if model_name in ['CTLE']:
        return CTLEDataLoader(config, data_feature, data)
    if model_name in ['Hier']:
        return HierDataLoader(config, data_feature, data)
    if model_name in ['Tale', 'POI2Vec']:
        return TaleDataLoader(config, data_feature, data)
    if model_name in ['Teaser']:
        return TeaserDataLoader(config, data_feature, data)
    if model_name in ['SkipGram', 'CBOW']:
        return SkipGramDataLoader(config, data_feature, data)
    if model_name in ['CACSR']:
        return CacsrDataLoader(config,data_feature,data)
    return None


class POIRepresentationDataset(AbstractDataset):

    def __init__(self, config):
        """
        @param raw_df: raw DataFrame containing all mobile signaling records.
            Should have at least three columns: user_id, latlng and datetime.
        @param coor_df: DataFrame containing coordinate information.
            With an index corresponding to latlng, and two columns: lat and lng.
        """
        self.config = config
        self.cache_file_folder = './veccity/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self._logger = getLogger()
        self._logger.info('Starting load data ...')
        self.cache=self.config.get('cache',True)
        self.dataset = self.config.get('dataset')
        self.test_scale = self.config.get('test_scale', 0.4)
        self.min_len = self.config.get('poi_min_len', 5)  # 轨迹最短长度
        self.min_frequency = self.config.get('poi_min_frequency', 10)  # POI 最小出现次数
        self.min_poi_cnt = self.config.get('poi_min_poi_cnt', 50)  # 用户最少拥有 POI 数
        self.pre_len = self.config.get('pre_len', 3)  # 预测后 pre_len 个 POI
        self.min_sessions = self.config.get('min_sessions', 3)# 每个user最少的session数
        self.time_threshold = self.config.get('time_threshold', 24)# 超过24小时就切分,暂时
        self.cut_method = self.config.get('cut_method','time_interval') # time_interval, same_day, fix_len
        self.w2v_window_size = self.config.get('w2v_window_size', 1)
        self.max_seq_len=self.config.get('poi_max_seq_len',32)
        self.data_path = './raw_data/' + self.dataset + '/'
        self.offset = 0
        self.cache_file_name = os.path.join(self.cache_file_folder,
                                            f'cache_{self.dataset}_{self.cut_method}_{self.max_seq_len}_{self.min_len}_{self.min_frequency}_{self.min_poi_cnt}_{self.pre_len}_{self.min_sessions}_{self.time_threshold}.pickle')

        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))
        self.usr_file = self.config.get('usr_file', self.dataset)
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        if os.path.exists(os.path.join(self.data_path, self.geo_file + '.geo')):
            self._load_geo()
        else:
            raise ValueError('Not found .geo file!')
        if os.path.exists(os.path.join(self.data_path, self.dyna_file + '.citraj')):
            self._load_dyna()
        else:
            raise ValueError('Not found .citraj file!')

        if self.cache and os.path.exists(self.cache_file_name):
            self._logger.info(f'load data from cache file: {self.cache_file_name}')
            with open(self.cache_file_name,'rb') as f:
                self.train_set=pickle.load(f)
                self.test_set=pickle.load(f)
                self.w2v_data=pickle.load(f)
                self.loc_index_map=pickle.load(f)
                self.user_index_map=pickle.load(f)
                loc_index_map=self.loc_index_map
                user_index_map=self.user_index_map
                self.max_seq_len=pickle.load(f)
                self.df = self.df[self.df['user_index'].isin(self.user_index_map)]
                self.df = self.df[self.df['loc_index'].isin(self.loc_index_map)]
                self.coor_df = self.coor_df[self.coor_df['geo_uid'].isin(loc_index_map)]
                self.df['user_index'] = self.df['user_index'].map(user_index_map)
                self.coor_df['geo_uid'] = self.coor_df['geo_uid'].map(loc_index_map)
                self.df['loc_index'] = self.df['loc_index'].map(loc_index_map)
                self.num_user = len(user_index_map)
                self.num_loc = self.coor_df.shape[0]
                self.coor_mat = self.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index').to_numpy()
                self.id2coor_df = self.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index'). \
                    set_index('loc_index').sort_index()
                
        else:
            self.res=self.cutter_filter()
            self._init_data_feature()

        self._logger.info('User num: {}'.format(self.num_user))
        self._logger.info('Location num: {}'.format(self.num_loc))
        self._logger.info('Total checkins: {}'.format(self.df.shape[0]))
        self._logger.info('Train set: {}'.format(len(self.train_set)))
        self._logger.info('Test set: {}'.format(len(self.test_set)))
        self.con = self.config.get('con', 7e8)
        self.theta = self.num_user * self.num_loc / self.con
        
    def _load_geo(self):
        geo_df = pd.read_csv(os.path.join(self.data_path, self.geo_file + '.geo'),low_memory=False)
        geo_df = geo_df[geo_df['type'] == 'Point']
        self.offset = geo_df['geo_uid'].min()
        poi_list = geo_df['geo_location'].tolist()
        lng_list = []
        lat_list = []
        for s in poi_list:
            lng, lat = ast.literal_eval(s)
            lng_list.append(lng)
            lat_list.append(lat)
        lng_col = pd.Series(lng_list, name='lng')
        lat_col = pd.Series(lat_list, name='lat')
        idx_col = pd.Series(list(range(len(geo_df))), name='geo_uid')
        type_name = self.config.get('poi_type_name', None)
        if type_name is not None:
            category_list=list(geo_df[type_name].drop_duplicates())
            c2i={name:i for i,name in enumerate(category_list)}
            cid_list=[]
            for name in list(geo_df[type_name]):
                cid_list.append(c2i[name])
            cid_list=pd.Series(cid_list,name='category')
            self.coor_df = pd.concat([idx_col, lat_col, lng_col, cid_list], axis=1)
        else:
            self.coor_df = pd.concat([idx_col, lat_col, lng_col], axis=1)

    def _load_dyna(self):
        dyna_df = pd.read_csv(os.path.join(self.data_path, self.dyna_file + '.citraj'))
        # TODO 区分 trajectory 和 check-in
        dyna_df = dyna_df[dyna_df['type'] == 'trajectory']
        # dyna_df['location'] = dyna_df['geo_uid'] - self.offset
        dyna_df = dyna_df.merge(self.coor_df, left_on='geo_id', right_on='geo_uid', how='left')
        dyna_df.rename(columns={'time': 'datetime'}, inplace=True)
        dyna_df.rename(columns={'geo_id': 'loc_index'}, inplace=True)
        dyna_df.rename(columns={'user_id': 'user_index'}, inplace=True)
        self.df = dyna_df[['user_index', 'loc_index', 'datetime', 'lat', 'lng']]
        user_counts = self.df['user_index'].value_counts()
        self.df = self.df[self.df['user_index'].isin(user_counts.index[user_counts >= self.min_poi_cnt])]
        loc_counts = self.df['loc_index'].value_counts()
        self.coor_df = self.coor_df[self.coor_df['geo_uid'].isin(loc_counts.index[loc_counts >= self.min_frequency])]
        self.df = self.df[self.df['loc_index'].isin(loc_counts.index[loc_counts >= self.min_frequency])]
        # 等到最后处理完再进行映射
        # loc_index_map = self.gen_index_map(self.coor_df, 'geo_uid')
        # self.coor_df['geo_uid'] = self.coor_df['geo_uid'].map(loc_index_map)
        # self.df['loc_index'] = self.df['loc_index'].map(loc_index_map)
        # user_index_map = self.gen_index_map(self.df, 'user_index')
        # self.df['user_index'] = self.df['user_index'].map(user_index_map)
        # self.num_user = len(user_index_map)
        # self.num_loc = self.coor_df.shape[0]

    def _split_days(self):
        data = pd.DataFrame(self.df, copy=True)

        data['datetime'] = pd.to_datetime(data["datetime"])
        data['nyr'] = data['datetime'].apply(lambda x: datetime.fromtimestamp(x.timestamp()).strftime("%Y-%m-%d"))

        days = sorted(data['nyr'].drop_duplicates().to_list())
        num_days = None #self.config.get('num_days', 20)
        if num_days is not None:
            days = days[:num_days]
        if len(days) <= 1:
            raise ValueError('Dataset contains only one day!')
        test_count = max(1, min(math.ceil(len(days) * self.test_scale), len(days)))
        self.split_days = [days[:-test_count], days[-test_count:]]
        self._logger.info('Days for train: {}'.format(self.split_days[0]))
        self._logger.info('Days for test: {}'.format(self.split_days[1]))

    def _load_usr(self):
        pass

    def gen_index_map(self, df, column, offset=0):
        index_map = {origin: index + offset
                     for index, origin in enumerate(df[column].drop_duplicates())}
        return index_map

    def _init_data_feature(self):
        # 变换user的id
        # 划分train、test数据集
        # 生成seq
        # self.max_seq_len = 0
        #one_set = [user_index, 'loc_index', 'weekday'list(),'timestamp'.tolist(), loc_length]
        self.train_set = []
        self.test_set = []
        self.w2v_data = []
        # 这里之后将不会变换，所以可以进行映射了
        u_list=self.res.keys()
        self.df=self.df[self.df['user_index'].isin(u_list)]
        loc_keys = self.df['loc_index'].value_counts().keys()
        self.coor_df=self.coor_df[self.coor_df['geo_uid'].isin(loc_keys)]
        loc_index_map = self.gen_index_map(self.coor_df, 'geo_uid')
        self.coor_df['geo_uid'] = self.coor_df['geo_uid'].map(loc_index_map)
        self.df['loc_index'] = self.df['loc_index'].map(loc_index_map)
        user_index_map = self.gen_index_map(self.df, 'user_index')
        self.df['user_index']=self.df['user_index'].map(user_index_map)
        self.num_user=len(user_index_map)
        self.num_loc=len(loc_index_map)
        assert len(loc_index_map) == self.coor_df.shape[0]
        
        for user_index in self.res:            
            lens=len(self.res[user_index])
            train_lens=int((1-self.test_scale)*lens)
            
            for i in range(lens):
                uid=user_index_map[user_index]
                loc_list=[]
                week_list=[]
                timestamp_list=[]
                delta_list=[]
                dist_list=[]
                lats=[]
                longs=[]
                loc_len=len(self.res[user_index][i])
                prev_time=self.res[user_index][i][0][2].timestamp()
                prev_loc=(self.res[user_index][i][0][3],self.res[user_index][i][0][4])
                for row in self.res[user_index][i]:
                    loc_list.append(loc_index_map[row[1]])
                    week_list.append(row[2].weekday())
                    timestamp_list.append(row[5])
                    delta_list.append(row[5]-prev_time)
                    prev_time=row[5]
                    coordist=np.array([row[3]-prev_loc[0],row[4]-prev_loc[1]])
                    dist_list.append(np.sqrt((coordist**2).sum(-1)))
                    prev_loc=[row[3],row[4]]
                    lats.append(row[3])
                    longs.append(row[4])
                if i <= train_lens:
                    self.train_set.append([uid, loc_list, week_list, timestamp_list, loc_len,delta_list,dist_list,lats,longs])
                    self.w2v_data.append([uid, loc_list, week_list, timestamp_list, loc_len])
                else:
                    self.test_set.append([uid, loc_list, week_list, timestamp_list, loc_len,delta_list,dist_list,lats,longs])
        
        self.coor_mat = self.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index').to_numpy()
        self.id2coor_df = self.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index'). \
            set_index('loc_index').sort_index()
        self.user_index_map=user_index_map
        self.loc_index_map=loc_index_map

        # todo list 这里不应该每次都生成，而是应该有缓存，如果有缓存则直接加载
        # 需要保存那些东西呢，train_set, test_set,w2v_set, loc_index_map, user_index_map
        # cache dir 需要包括cut_type，test_scaler, dataset, user_filter, checkin_filter,pre_len
        if self.cache:
            self._logger.info(f'save data cache in {self.cache_file_name}')
            with open(self.cache_file_name,'wb') as f:
                pickle.dump(self.train_set,f)
                pickle.dump(self.test_set,f)
                pickle.dump(self.w2v_data,f)
                pickle.dump(loc_index_map,f)
                pickle.dump(user_index_map,f)
                pickle.dump(self.max_seq_len,f)
                

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        return get_dataloader(self.config, self.get_data_feature(), self.train_set), None, None

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {
            "max_seq_len": self.max_seq_len,
            "num_loc": self.num_loc,
            "num_user": self.num_user,
            "train_set": self.train_set,
            "test_set": self.test_set,
            "w2v_data": self.w2v_data,
            "coor_mat": self.coor_mat,
            "id2coor_df": self.id2coor_df,
            "theta" : self.theta,
            "coor_df" : self.coor_df,
            "df":self.df,
            "loc_index_map":self.loc_index_map,
            "offset":self.offset
        }

    def gen_sequence(self, min_len=None, select_days=None, include_delta=False):
        """
        Generate moving sequence from original trajectories.

        @param min_len: minimal length of sentences.
        @param select_day: list of day to select, set to None to use all days.
        """

        if min_len is None:
            min_len = self.min_len
        data = pd.DataFrame(self.df, copy=True)
        data['datetime'] = pd.to_datetime(data["datetime"]) # take long time, can we just store the right format?
        data['day'] = data['datetime'].dt.day
        data['nyr'] = data['datetime'].apply(lambda x: datetime.fromtimestamp(x.timestamp()).strftime("%Y-%m-%d"))
        if select_days is not None:
            data = data[data['nyr'].isin(self.split_days[select_days])]
        
        data['weekday'] = data['datetime'].dt.weekday
        data['timestamp'] = data['datetime'].apply(lambda x: x.timestamp())

        if include_delta:
            data['time_delta'] = data['timestamp'].shift(-1) - data['timestamp']
            coor_delta = (data[['lng', 'lat']].shift(-1) - data[['lng', 'lat']]).to_numpy()
            data['dist'] = np.sqrt((coor_delta ** 2).sum(-1))
        seq_set = []
        for (user_index, day), group in data.groupby(['user_index', 'day']):
            if group.shape[0] < min_len:
                continue
            one_set = [user_index, group['loc_index'].tolist(), group['weekday'].astype(int).tolist(),
                       group['timestamp'].astype(int).tolist(), group.shape[0]]

            if include_delta:
                one_set += [[0] + group['time_delta'].iloc[:-1].tolist(),
                            [0] + group['dist'].iloc[:-1].tolist(),
                            group['lat'].tolist(),
                            group['lng'].tolist()]

            seq_set.append(one_set)
        return seq_set

    def cutter_filter(self):
        """
        切割后的轨迹存储格式: (dict)
            {
                uid: [
                    [
                        checkin_record,
                        checkin_record,
                        ...
                    ],
                    [
                        checkin_record,
                        checkin_record,
                        ...
                    ],
                    ...
                ],
                ...
            }
        """
        # load data according to config
        traj = pd.DataFrame(self.df, copy=True)
        traj['datetime'] = pd.to_datetime(traj["datetime"]) # take long time, can we just store the right format?
        traj['timestamp'] = traj['datetime'].apply(lambda x: x.timestamp())
        # user_set = pd.unique(traj['user_id'])
        res = {}
        min_session_len = self.min_len # 每个session中至少有3个轨迹
        min_sessions = self.min_sessions # 最少的session数
        window_size = self.time_threshold # 超过24小时就切分,暂时
        cut_method = self.cut_method
        loc_set=set()
        if cut_method == 'time_interval':
            # 按照时间窗口进行切割
            for user_index, group in traj.groupby(['user_index']):
                if type(user_index)==tuple:
                    user_index=user_index[0]
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                lens=group.shape[0]
                for index in range(lens):
                    row=group.iloc[index]
                    now_time = row['timestamp']
                    if index == 0:
                        session.append(row.tolist())
                        prev_time = now_time
                    else:
                        time_off = (now_time-prev_time)/3600
                        if time_off < window_size and time_off >= 0 and len(session) < self.max_seq_len:
                            session.append(row.tolist())
                        else:
                            if len(session) >= min_session_len:
                                sessions.append(session)
                            session = []
                            session.append(row.tolist())
                    prev_time = now_time
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    res[user_index] = sessions
        # elif cut_method == 'same_date':
        #     # 将同一天的 check-in 划为一条轨迹
        #     for uid in tqdm(user_set, desc="cut and filter trajectory"):
        #         usr_traj = traj[traj['user_id'] == uid].to_numpy()
        #         sessions = []  # 存放该用户所有的 session
        #         session = []  # 单条轨迹
        #         prev_date = None
        #         for index, row in enumerate(usr_traj):
        #             now_time = parse_time(row[2])
        #             now_date = now_time.day
        #             if index == 0:
        #                 session.append(row.tolist())
        #             else:
        #                 if prev_date == now_date and len(session) < max_session_len:
        #                     # 还是同一天
        #                     session.append(row.tolist())
        #                 else:
        #                     if len(session) >= min_session_len:
        #                         sessions.append(session)
        #                     else:
        #                         print(session)
        #                     session = []
        #                     session.append(row.tolist())
        #             prev_date = now_date
        #         if len(session) >= min_session_len:
        #             sessions.append(session)
        #         if len(sessions) >= min_sessions:
        #             res[str(uid)] = sessions
        # else:
        #     # cut by fix window_len
        #     if max_session_len != window_size:
        #         raise ValueError('the fixed length window is not equal to max_session_len')
        #     for uid in tqdm(user_set, desc="cut and filter trajectory"):
        #         usr_traj = traj[traj['user_id'] == uid].to_numpy()
        #         sessions = []  # 存放该用户所有的 session
        #         session = []  # 单条轨迹
        #         for index, row in enumerate(usr_traj):
        #             if len(session) < window_size:
        #                 session.append(row.tolist())
        #             else:
        #                 sessions.append(session)
        #                 session = []
        #                 session.append(row.tolist())
        #         if len(session) >= min_session_len:
        #             sessions.append(session)
        #         if len(sessions) >= min_sessions:
        #             res[str(uid)] = sessions
        return res

