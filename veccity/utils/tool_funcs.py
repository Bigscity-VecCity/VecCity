import time
import math
import numpy as np
from itertools import tee
from pynvml import *
import os
import psutil
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from datetime import datetime, timezone, timedelta

# initializing here!!
nvmlInit() 

def mean(x):
    if x == []:
        return 0.0
    return sum(x) / len(x)


def std(x):
    return np.std(x)


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def l2_distance(lon1, lat1, lon2, lat2):
    return math.sqrt( (lon2 - lon1) ** 2 + (lat2 - lat1) ** 2 )


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2.0)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6367 * 1000


# distance between two points on Earth using their latitude and longitude
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6367 * 1000


# radian of the segment
def radian(lon1, lat1, lon2, lat2):
    dy = lat2 - lat1
    dx = lon2 - lon1
    r = 0.0

    if dx == 0:
        if dy >= 0:
            r = 1.5707963267948966 # math.pi / 2
        else: 
            r = 4.71238898038469 # math.pi * 1.5
        return round(r, 3)
    
    r = math.atan(dy / dx)
    # angle_in_degrees = math.degrees(angle_in_radians)
    if dx < 0:
        r = r + 3.141592653589793
    else:
        if dy < 0:
            r = r + 6.283185307179586
        else:
            pass
    return round(r, 3)


def accuracy(lst_true, lst_pred):
    return accuracy_score(lst_true, lst_pred)


def f1(lst_true, lst_pred):
    return f1_score(lst_true, lst_pred, average = 'micro')


def auc(lst_true, lst_pred, classes):
    if len(classes) == 2:
        return roc_auc_score(lst_true, lst_pred)
    else:
    # https://stackoverflow.com/questions/63303682/sklearn-multiclass-roc-auc-score
        return roc_auc_score(label_binarize(lst_true, classes = classes),
                        label_binarize(lst_pred, classes = classes),
                        average = 'micro', multi_class = 'ovo')


# "accuracy={:.6f}, precision={:.6f}, recall={:.6f}, f1={:.6f}"
def classification_metrics(lst_true, lst_pred):
    return accuracy_score(lst_true, lst_pred), \
            precision_score(lst_true, lst_pred, average = 'micro'), \
            recall_score(lst_true, lst_pred, average = 'micro'), \
            f1_score(lst_true, lst_pred, average = 'micro')


def dump_config_to_strs(file_path):
    # return list
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('from') or line.startswith('import') or line.startswith('#') \
                    or line.strip() == '':
                continue
            lines.append(line.strip())
    return lines


def log_file_name():
    dt = datetime.now(timezone(timedelta(hours=8)))
    return dt.strftime("%Y%m%d_%H%M%S") + '.log'


def adjust_learning_rate(optimizer, lr, epoch, nepoch):
    # Decay the learning rate based on cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / nepoch)) # in range (0, 1] * lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def slicing(list_len, num_slices):
    if list_len <= num_slices:
        return [(0, list_len)]
    
    slice_size = int(list_len / num_slices)
    slice_range_idxs = []
    for i in range(num_slices):
        if i < num_slices - 1:
            slice_range_idxs.append((i * slice_size, i * slice_size + slice_size))
        else:
            slice_range_idxs.append((i * slice_size, list_len))
    return slice_range_idxs


def list_sum_n_continous_items(lst: list, n):
    if n < 2 or len(lst) == 0:
        return lst
    rtn = [0]
    for i in range(n, len(lst), n):
        rtn.append(sum(lst[i - n + 1 : i + 1]))
    return rtn


class Metrics:
    def __init__(self):
        self.dic = {} # {a:[], b:[], ...}
   
    def __str__(self) -> str:
        s = ''
        kvs = list(self.mean().items()) # + list(self.std().items())
        for i, (k, v) in enumerate(kvs):
            if i != 0:
                s += ','
            s = s + '{}={:.6f}'.format(k, v)
        return s

    def add(self, d_: dict):
        for k, v in d_.items():
            self.dic[k] = self.dic.get(k, []) + [v]
    
    def get(self, k):
        return self.dic.get(k, [])
    
    def mean(self, k = None):
        if type(k) == str:
            return mean(self.get(k)) # division by zero
        dic_mean = {}
        for k, v in self.dic.items():
            dic_mean[k] = mean(v)
        return dic_mean

    def std(self, k = None):
        if type(k) == str:
            return std(self.get(k)) # division by zero
        dic_mean = {}
        for k, v in self.dic.items():
            dic_mean[k] = std(v)
        return dic_mean
    

class Timer(object):
    _instance = None
    _last_t = 0.0

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self):
        _last_t = time.time()

    def tick(self):
        _t = time.time()
        _delta = _t - self._last_t
        self._last_t = _t
        return _delta

timer = Timer()


class GPUInfo:

    _h = nvmlDeviceGetHandleByIndex(0)

    @classmethod
    def mem(cls):
        info = nvmlDeviceGetMemoryInfo(cls._h)
        return info.used // 1048576, info.total // 1048576 # in MB

class RAMInfo:
    @classmethod
    def mem(cls):
        return int(psutil.Process(os.getpid()).memory_info().rss / 1048576) # in MB
