import ast
from datetime import datetime
import json
from logging import getLogger
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from veccity.data.dataset.abstract_dataset import AbstractDataset
import torch.nn.utils.rnn as rnn_utils

cache_dir = os.path.join("veccity", "cache", "dataset_cache")


def extract_weekday_and_minute_from_list(timestamp_list):
    weekday_list = []
    minute_list = []
    for timestamp in timestamp_list:
        dt = datetime.fromtimestamp(timestamp)
        weekday = dt.weekday() + 1
        minute = dt.hour * 60 + dt.minute + 1
        weekday_list.append(weekday)
        minute_list.append(minute)
    return torch.tensor(weekday_list).long(), torch.tensor(minute_list).long()


def prepare_gps_data(df, mat_padding_value, data_padding_value, max_len):
    """

    Args:
        df: opath_list,
        mat_padding_value: default num_nodes
        data_padding_value: default 0

    Returns:
        gps_data: (batch, gps_max_length, num_features)
        gps_assign_mat: (batch, gps_max_length)

    """

    # padding opath_list
    opath_list = [
        torch.tensor(opath_list, dtype=torch.float32) for opath_list in df["opath_list"]
    ]
    gps_assign_mat = rnn_utils.pad_sequence(
        opath_list, padding_value=mat_padding_value, batch_first=True
    )
    # padding gps point data
    data_package = []
    for col in df.drop(columns="opath_list").columns:
        features = df[col].tolist()
        features = [torch.tensor(f, dtype=torch.float32) for f in features]
        features = rnn_utils.pad_sequence(
            features, padding_value=torch.nan, batch_first=True
        )
        features = features.unsqueeze(dim=2)
        data_package.append(features)

    gps_data = torch.cat(data_package, dim=2)

    # 对除第一维特征进行标准化
    for i in range(1, gps_data.shape[2]):
        fea = gps_data[:, :, i]
        nozero_fea = torch.masked_select(
            fea, torch.isnan(fea).logical_not()
        )  # 计算不为nan的值的fea的mean与std
        gps_data[:, :, i] = (gps_data[:, :, i] - torch.mean(nozero_fea)) / torch.std(
            nozero_fea
        )

    # 把因为数据没有前置节点因此无法计算，加速度等特征的nan置0
    gps_data = torch.where(
        torch.isnan(gps_data), torch.full_like(gps_data, data_padding_value), gps_data
    )

    return gps_data, gps_assign_mat


def prepare_route_data(df, mat_padding_value, data_padding_value, max_len):
    """

    Args:
        df: cpath_list, road_timestamp, road_interval
        mat_padding_value: default num_nodes
        data_padding_value: default 0

    Returns:
        route_data: (batch, route_max_length, num_features)
        route_assign_mat: (batch, route_max_length)

    """

    # padding capath_list
    cpath_list = [
        torch.tensor(cpath_list, dtype=torch.float32) for cpath_list in df["cpath_list"]
    ]
    route_assign_mat = rnn_utils.pad_sequence(
        cpath_list, padding_value=mat_padding_value, batch_first=True
    )

    # padding route data
    weekday_route_list, minute_route_list = zip(
        *df["road_timestamp"].apply(extract_weekday_and_minute_from_list)
    )

    weekday_route_list = [
        weekday.clone().detach().long() for weekday in weekday_route_list
    ]
    minute_route_list = [minute.clone().detach().long() for minute in minute_route_list]
    weekday_data = rnn_utils.pad_sequence(
        weekday_route_list, padding_value=0, batch_first=True
    )
    minute_data = rnn_utils.pad_sequence(
        minute_route_list, padding_value=0, batch_first=True
    )

    new_road_interval = []
    for interval_list in df["road_interval"]:
        new_road_interval.append(torch.Tensor(interval_list).long())

    delta_data = rnn_utils.pad_sequence(
        new_road_interval, padding_value=-1, batch_first=True
    )

    route_data = torch.cat(
        [
            weekday_data.unsqueeze(dim=2),
            minute_data.unsqueeze(dim=2),
            delta_data.unsqueeze(dim=2),
        ],
        dim=-1,
    )  # (batch_size,max_len,2)

    # 填充nan
    route_data = torch.where(
        torch.isnan(route_data),
        torch.full_like(route_data, data_padding_value),
        route_data,
    )

    return route_data, route_assign_mat


def _split_duplicate_subseq(opath_list, max_len):
    length_list = []
    subsequence = [opath_list[0]]
    for i in range(0, len(opath_list) - 1):
        if opath_list[i] == opath_list[i + 1]:
            subsequence.append(opath_list[i])
        else:
            length_list.append(len(subsequence))
            subsequence = [opath_list[i]]
    length_list.append(len(subsequence))
    if len(length_list) > max_len:
        raise ValueError("max_len is too small")
    return length_list + [0] * (max_len - len(length_list))


def preprocess_data(
    data, mat_padding_value, data_padding_value, gps_max_len, route_max_len
):
    gps_length = (
        data["opath_list"]
        .apply(
            lambda opath_list: _split_duplicate_subseq(
                opath_list, data["gps_length"].max()
            )
        )
        .tolist()
    )
    gps_length = torch.tensor(gps_length, dtype=torch.int)

    gps_columns = [
        "opath_list",
        # "tlist",
        "lng_list",
        "lat_list",
        # "length",
        # "speed",
        # "duration",
        # "hop",
    ]
    for col in gps_columns:
        data[col] = data[col].apply(ast.literal_eval)
    gps_data, gps_assign_mat = prepare_gps_data(
        data[gps_columns],
        mat_padding_value,
        data_padding_value,
        gps_max_len,
    )

    route_columns = ["cpath_list", "road_timestamp", "road_interval"]

    for col in route_columns:
        data[col] = data[col].apply(ast.literal_eval)
    # todo 路段本身的属性特征怎么放进去
    route_data, route_assign_mat = prepare_route_data(
        data[route_columns],
        mat_padding_value,
        data_padding_value,
        route_max_len,
    )

    return gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length


def get_dataset(
    raw_dataset,
    route_min_len,
    route_max_len,
    gps_min_len,
    gps_max_len,
    mat_padding_value,
):
    raw_dataset["route_length"] = raw_dataset["cpath_list"].map(len)
    raw_dataset = raw_dataset[
        (raw_dataset["route_length"] > route_min_len)
        & (raw_dataset["route_length"] < route_max_len)
    ].reset_index(drop=True)

    raw_dataset["gps_length"] = raw_dataset["opath_list"].map(len)
    raw_dataset = raw_dataset[
        (raw_dataset["gps_length"] > gps_min_len)
        & (raw_dataset["gps_length"] < gps_max_len)
    ].reset_index(drop=True)

    # 获取最大路段id
    uniuqe_path_list = []
    raw_dataset["cpath_list"].apply(
        lambda cpath_list: uniuqe_path_list.extend(list(set(cpath_list)))
    )
    uniuqe_path_list = list(set(uniuqe_path_list))

    # mat_padding_value = max(uniuqe_path_list) + 1
    data_padding_value = 0.0

    gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length = (
        preprocess_data(
            raw_dataset,
            mat_padding_value,
            data_padding_value,
            gps_max_len,
            route_max_len,
        )
    )

    return TensorDataset(
        gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length
    )


class JGRMDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.model = config.get("model")
        # preprocess_all(config)
        self.device = config.get("device")
        self._logger = getLogger()
        self.dataset = self.config.get("dataset", "")
        self.data_path = "./raw_data/" + self.dataset + "/"
        data_cache_dir = os.path.join(cache_dir, self.dataset)
        self.data_cache_dir = data_cache_dir
        # TODO 这里先用cache_data
        self.road_geo_path = os.path.join(data_cache_dir, "road.csv")
        self.gps_data_path = os.path.join(data_cache_dir, "train.csv")
        self.traj_train_path = os.path.join(data_cache_dir, "traj_gps_road_train.csv")
        self.traj_val_path = os.path.join(data_cache_dir, "traj_gps_road_val.csv")
        self.traj_test_path = os.path.join(data_cache_dir, "traj_gps_road_test.csv")
        self.adj_json_path = os.path.join(data_cache_dir, "road_neighbor.json")

        self.init_road_emb_path = os.path.join(data_cache_dir, "init_w2v_road_emb.pt")
        if os.path.exists(self.init_road_emb_path):
            self.init_road_emb = torch.load(self.init_road_emb_path).to(self.device)
        else:
            raise ValueError(
                "init_road_emb_path not exists"
            )  # TODO: add generate road emb code

        self.batch_size = config.get("batch_size", 64)
        self.num_workers = config.get("num_workers", 0)
        self.route_min_len = config.get("route_min_len", 10)
        self.route_max_len = config.get("route_max_len", 100)
        self.gps_min_len = config.get("gps_min_len", 10)
        self.gps_max_len = config.get("gps_max_len", 100)

        self.vocab_size = config.get("vocab_size", 100)  # 路段数量

        self.edge_index = self._gen_edge_index(self.adj_json_path).to(self.device)

    def _gen_edge_index(self, adj_json_path):
        with open(adj_json_path, "r") as f:
            data = json.load(f)
        edges = []
        for source, targets in data.items():
            source_int = int(source)
            for target in targets:
                edges.append([source_int, target])

        edge_index = np.array(edges).T
        return torch.tensor(edge_index)

    def _gen_dataset(self):
        traj_train_data = pd.read_csv(self.traj_train_path)
        traj_val_data = pd.read_csv(self.traj_val_path)
        traj_test_data = pd.read_csv(self.traj_test_path)

        train_dataset = get_dataset(
            traj_train_data,
            self.route_min_len,
            self.route_max_len,
            self.gps_min_len,
            self.gps_max_len,
            mat_padding_value=self.vocab_size,
        )

        val_dataset = get_dataset(
            traj_val_data,
            self.route_min_len,
            self.route_max_len,
            self.gps_min_len,
            self.gps_max_len,
            mat_padding_value=self.vocab_size,
        )

        test_dataset = get_dataset(
            traj_test_data,
            self.route_min_len,
            self.route_max_len,
            self.gps_min_len,
            self.gps_max_len,
            mat_padding_value=self.vocab_size,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
        )
        return train_loader, val_loader, test_loader

    def get_data(self):
        return self._gen_dataset()

    def get_data_feature(self):
        return {"edge_index": self.edge_index, "init_road_emb": self.init_road_emb}
