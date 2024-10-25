import math
from itertools import zip_longest
from math import radians, cos, sin, asin, sqrt
import numpy as np
import torch
import random
from torch import nn
from torch.nn import init
import time
import copy

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def gen_index_map(df, column, offset=0):
    index_map = {origin: index + offset
                 for index, origin in enumerate(df[column].drop_duplicates())}
    return index_map


def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true > 0)
    y_true = y_true[non_zero_index]
    y_pred = y_pred[non_zero_index]

    mape = np.abs((y_true - y_pred) / y_true)
    mape[np.isinf(mape)] = 0
    return np.mean(mape) * 100


def create_src_trg(full_seq, pre_len, fill_value):
    src_seq, trg_seq = zip(*[[s[:-pre_len], s[pre_len:]] for s in full_seq])
    src_seq = np.transpose(np.array(list(zip_longest(*src_seq, fillvalue=fill_value))))
    trg_seq = np.transpose(np.array(list(zip_longest(*trg_seq, fillvalue=fill_value))))
    # index_matrix = [1 if i < cl for i in range(len(src_seq) for cl in range())] 
    return src_seq, trg_seq


def create_src(full_seq, fill_value):
    return np.transpose(np.array(list(zip_longest(*full_seq, fillvalue=fill_value))))


def top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    successes = 0
    for i, truth in enumerate(truths):
        if truth in best_n[i, :]:
            successes += 1
    return float(successes) / truths.shape[0]


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.Embedding):
        embed_size = m.weight.size(-1)
        if embed_size > 0:
            init_range = 0.5/m.weight.size(-1)
            init.uniform_(m.weight.data, -init_range, init_range)


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]
    
def distance(lon1, lat1, lon2, lat2):  
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
   
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    if lon1 == 0 or lat1 ==0 or lon2==0 or lat2==0:
        return 0
    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  
    return c * r  

def construct_spatial_matrix_accordingDistance(distance_theta, venue_cnt, venue_lng, venue_lat, gaussian_beta=None):
    SS_distance = np.zeros((venue_cnt, venue_cnt))  
    SS_gaussian_distance = np.zeros((venue_cnt, venue_cnt))  
    SS_proximity = np.zeros((venue_cnt, venue_cnt))  
    for i in range(venue_cnt):
        for j in range(venue_cnt):
            distance_score = distance(venue_lng[i], venue_lat[i], venue_lng[j], venue_lat[j])
            SS_distance[i, j] = distance_score  
            if gaussian_beta is not None:
                distance_gaussian_score = np.exp(-gaussian_beta * distance_score) 
                SS_gaussian_distance[i, j] = distance_gaussian_score  
            if SS_distance[i, j] < distance_theta:  
                SS_proximity[i, j] = 1
        if i % 500 == 0:
            print("constructing spatial matrix: ", i, "/", venue_cnt)
    return SS_distance, SS_proximity, SS_gaussian_distance

def get_relativeTime(arrival_times): 
    first_time_list = [arrival_times[0] for _ in range(len(arrival_times))]
    return list(map(delta_minutes, arrival_times, first_time_list))


def get_delta(arrival_times):

    copy_times = copy.deepcopy(arrival_times)
    copy_times.insert(0, copy_times[0]) 
    copy_times.pop(-1)
    return list(map(delta_minutes, arrival_times, copy_times))


def split_sampleSeq2sessions(sampleSeq_delta_times, min_session_mins):

    sessions = []  
    split_index = []  #
    sessions_lengths = []

    for i in range(1, len(sampleSeq_delta_times)):
        if sampleSeq_delta_times[i] >= min_session_mins:
            split_index.append(i)
    # print('split_index:', split_index)
    if len(split_index) == 0:  
        sessions.append(sampleSeq_delta_times)
        sessions_lengths.append(len(sampleSeq_delta_times))
        # print('sessions:', sessions)
        # print('sessions_lengths:', sessions_lengths)
        return sessions, split_index, sessions_lengths
    else:
        start_index = 0
        for i in range(0, len(split_index)):
            split = split_index[i]
            if split-start_index > 1:  
                sampleSeq_delta_times[start_index] = 0  
                sessions.append(sampleSeq_delta_times[start_index:split])
                sessions_lengths.append(len(sampleSeq_delta_times[start_index:split]))
            start_index = split
        if len(sampleSeq_delta_times[split_index[-1]:]) > 1: 
            sampleSeq_delta_times[split_index[-1]] = 0
            sessions.append(sampleSeq_delta_times[split_index[-1]:])
            sessions_lengths.append(len(sampleSeq_delta_times[split_index[-1]:]))
        # print('sessions:', sessions)
        # print('sessions_lengths:', sessions_lengths)
        return sessions, split_index, sessions_lengths  



def splitSeq_basedonSessions(seq, split_index):

    sessions = []
    if len(split_index) == 0:
        sessions.append(seq)
    else:
        start_index = 0
        for i in range(0, len(split_index)):
            split = split_index[i]
            if split-start_index > 1:
                sessions.append(seq[start_index:split])
            start_index = split
        if len(seq[split_index[-1]:]) > 1: 
            sessions.append(seq[split_index[-1]:])
    return sessions

def delta_minutes(ori, cp):
    delta = (ori.timestamp() - cp.timestamp())/60
    if delta < 0:
        delta = 1
    return delta

def tid_list_48(tm):
    if tm.weekday() in [0, 1, 2, 3, 4]:
        tid = int(tm.hour)
    else:
        tid = int(tm.hour) + 24
    return tid