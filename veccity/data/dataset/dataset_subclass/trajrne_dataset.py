import json
import os
import pdb
import random
from collections import Counter
from itertools import chain, combinations
from logging import getLogger

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import geopandas as gpd
from tqdm import tqdm, trange
from tqdm.asyncio import trange
from veccity.data.dataset.abstract_dataset import AbstractDataset
from veccity.data.preprocess import cache_dir, preprocess_all
from veccity.utils import ensure_dir, need_train


class TrajRNEDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        preprocess_all(config)
        self._logger = getLogger()
        self.dataset = self.config.get('dataset', '')
        self.cache = self.config.get('cache', True)
        data_path = f'raw_data/{self.dataset}'
        dataset_cache_path = f'./veccity/cache/dataset_cache/{self.dataset}'
        cache_path = os.path.join(dataset_cache_path, 'TrajRNE')
        ensure_dir(cache_path)

        file_path = os.path.join(dataset_cache_path, "road_neighbor.json")
        with open(file_path, "r") as fp:
            adj_data = json.load(fp)
        adj_data = {int(k): set(v) for k, v in adj_data.items()}

        self._logger.info("Calculate adj")
        adj = generate_node_traj_adj(adj_dict=adj_data, add_self_loops=True)
        np.savetxt(os.path.join(cache_path, "traj_adj_k_1.gz"), X=adj)
            
        if not os.path.exists(os.path.join(cache_path, "traj_adj_k_2.gz")):
            
            # 计算邻接矩阵的平方
            self._logger.info("Calculate adj squared")
            adj = csr_matrix(adj)
            adj_matrix_squared = np.dot(adj, adj).toarray()
            adj = adj.toarray()
            np.savetxt(os.path.join(cache_path, "traj_adj_k_2.gz"), X=adj_matrix_squared)


        if not os.path.exists(os.path.join(cache_path, 'sre_traindata.json')):
            traj = Trajectory(os.path.join(dataset_cache_path, 'traj_road.csv'))
            data = pd.read_csv(os.path.join(dataset_cache_path, 'road.csv'))
            # Generate traj features
            traj_features = traj.generate_speed_features(data)
            # Generate Training Data
            self._logger.info("Generate training data")
            generate_data(
                data,
                window_size=900,
                number_negative=3,
                save_batch_size=64,
                wheighted_adj=adj,  # needed for trajectory walks
                traj_feature_df=traj_features,
                traj_feats="util",  # which traj feat to use: util or avg_speed
                file_path=cache_path,
                file_name="sre_traindata.json",
            )

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
        dataset_cache_path = f'./veccity/cache/dataset_cache/{self.dataset}'
        sfc_data = pd.read_csv(os.path.join(dataset_cache_path, 'road.csv'))
        sfc_data = gpd.GeoDataFrame(sfc_data, geometry=gpd.GeoSeries.from_wkt(sfc_data["geometry"]), crs="EPSG:4326")
        sfc_data["x"] = sfc_data["geometry"].centroid.x / 100  # normalize to -2/2
        sfc_data["y"] = sfc_data["geometry"].centroid.y / 100  # normalize to -1/1
        adj = np.loadtxt(os.path.join(dataset_cache_path, "TrajRNE/traj_adj_k_2.gz"))
        file_path = os.path.join(dataset_cache_path, "TrajRNE/sre_traindata.json")
        with open(file_path, "r") as fp:
            sre_data = np.array(json.load(fp))
        return {
            'sfc_data': sfc_data,
            'sre_data': sre_data,
            'adj': adj,
        }

def generate_data(
        data,
        window_size: int = 900,
        number_negative: int = 3,
        save_batch_size=128,
        wheighted_adj=None,  # needed for trajectory walks
        traj_feature_df=None,
        traj_feats="util",  # which traj feat to use: util or avg_speed
        file_path=".",
        file_name="sre_traindata.json"
):
    """
    Generates the dataset like described in the corresponding paper. Since this needs alot of ram we use a batching approach.
    Args:
        n_shortest_paths (int, optional): how many shortest paths per node in graph. Defaults to 1280.
        window_size (int, optional): window size for distance neighborhood in meters. Defaults to 900.
        number_negative (int, optional): Negative samples to draw per positive sample.
        save_batch_size (int, optional): how many shortest paths to process between each save. Defaults to 128.
        file_path (str, optional): path where the dataset should be saved. Defaults to ".".
    """

    # Generate Walks
    walk = RandomWalker_Traj(wheighted_adj)
    paths = walk.random_walks()
    nodes = np.arange(wheighted_adj.shape[0])

    # Feature extraction: length, degree, traj_feats
    info = data
    node_list = np.array(info["id"])

    # Length
    node_idx_to_length_map = info.loc[node_list.astype(list), "length"].to_numpy()
    # Road Type
    node_idx_to_highway_map = info.loc[node_list.astype(list), "highway"].to_numpy()

    # Add degree column to edge feature df
    degree = np.sum(wheighted_adj, axis=1)
    info["degree"] = degree

    node_idx_to_degree_map = info.loc[node_list.astype(list), "degree"].to_numpy()

    # Add trajectory feature
    if not (traj_feature_df is None and traj_feats is None):
        info = pd.merge(info, traj_feature_df, on='id', how='inner')
        # As those are continuous values we use binning to create categorical classes
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        info['traffic_bin'] = pd.qcut(info[traj_feats], q=bins, labels=range(len(bins) - 1))
        node_idx_to_traffic_map = info.loc[node_list.astype(list), 'traffic_bin'].to_numpy()

    # Iterate over batches and save
    save_path = os.path.join(file_path, file_name)
    # and generate train pairs
    for i in tqdm(range(0, len(paths), save_batch_size * len(nodes))):
        trainset = extract_pairs(
            node_idx_to_length_map,
            node_idx_to_highway_map,
            node_idx_to_degree_map,
            node_idx_to_traffic_map,
            node_list,
            paths[i: i + (save_batch_size * len(nodes))],
            window_size,
            number_negative,
        )
        # save batch
        if not os.path.isfile(save_path):
            with open(save_path, "w") as fp:
                json.dump(trainset, fp)
        else:
            with open(save_path, "r") as fp:
                a = np.array(json.load(fp))
                a = np.unique(np.vstack([a, np.array(trainset)]), axis=0)
            with open(save_path, "w") as fp:
                json.dump(a.tolist(), fp)


def extract_pairs(
        info_length: np.array,
        info_highway: np.array,
        info_degree: np.array,
        info_traffic: np.array,
        node_list: np.array,
        node_paths: list,
        window_size: int,
        number_negative: int,
):
    """_summary_
    
     Generates the traning pairs consisting of (v_x, v_y, in window_size?, same degree?, same traffic?). This is highly optimized.

    Args:
        info_length (np.array): length for each node in graph (ordered by node ordering in graph)
        info_highway (np.array): type for each node in graph (ordered by node ordering in graph)
        node_list (np.array): nodes in graph (ordered by node ordering in graph)
        node_paths (list): shortest paths
        window_size (int): window_size in meters
        number_negative (int): number negative to draw for each node

    Returns:
        list: training pairs
    """
    res = []
    # lengths of orginal sequences in flatted with cumsum to get real position in flatted
    orig_lengths = np.array([0] + [len(x) for x in node_paths]).cumsum()
    flatted = list(chain.from_iterable(node_paths))
    # get all lengths of sequence roads
    flat_lengths = info_length[flatted]

    # generate window tuples
    node_combs = []
    for i in range(len(orig_lengths) - 1):
        lengths = flat_lengths[orig_lengths[i]: orig_lengths[i + 1]]
        # cumsum = lengths.cumsum()
        for j in range(len(lengths)):
            mask = (lengths[j:].cumsum() < window_size).sum()
            # idx = (np.abs(lengths[j:].cumsum() - window_size)).argmin()
            window = node_paths[i][j: j + mask]
            if len(window) > 1:
                combs = tuple(combinations(window, 2))
                node_combs.extend(combs)

    # save distinct tuples
    node_combs = list(dict.fromkeys(node_combs))
    node_combs = list(chain.from_iterable(node_combs))

    # generate same degree labels
    degrees = info_degree[node_combs].reshape(int(len(node_combs) / 2), 2)

    # generate same rad type labels
    highways = info_highway[node_combs].reshape(int(len(node_combs) / 2), 2)

    # generate same traffic labels
    traffic_labels = info_traffic[node_combs].reshape(int(len(node_combs) / 2), 2)

    # Generate pairs: node1, node2, true (on same walk), true if same degree
    pairs = np.c_[
        np.array(node_combs).reshape(int(len(node_combs) / 2), 2),
        np.ones(degrees.shape[0]),
        degrees[:, 0] == degrees[:, 1],
        traffic_labels[:, 0] == traffic_labels[:, 1],
        highways[:, 0] == highways[:, 1],
    ].astype(
        int
    )  # same type

    res.extend(tuple(pairs.tolist()))

    # generate negative sample with same procedure as for positive
    neg_nodes = np.random.choice(
        np.setdiff1d(np.arange(0, len(node_list)), node_combs),
        size=pairs.shape[0] * number_negative,
    )

    neg_pairs = pairs.copy()
    neg_pairs = neg_pairs.repeat(repeats=number_negative, axis=0)
    replace_mask = np.random.randint(0, 2, size=neg_pairs.shape[0]).astype(bool)
    neg_pairs[replace_mask, 0] = neg_nodes[replace_mask]
    neg_pairs[~replace_mask, 1] = neg_nodes[~replace_mask]
    neg_pairs[:, 2] -= 1

    neg_degree = info_degree[neg_pairs[:, :2].flatten()].reshape(
        neg_pairs.shape[0], 2
    )

    neg_traffic = info_traffic[neg_pairs[:, :2].flatten()].reshape(
        neg_pairs.shape[0], 2
    )

    neg_highways = info_highway[neg_pairs[:, :2].flatten()].reshape(
        neg_pairs.shape[0], 2
    )

    neg_pairs[:, 3] = neg_degree[:, 0] == neg_degree[:, 1]
    neg_pairs[:, 4] = neg_traffic[:, 0] == neg_traffic[:, 1]
    neg_pairs[:, 5] = neg_highways[:, 0] == neg_highways[:, 1]

    res.extend(tuple(neg_pairs.tolist()))

    return res


class RandomWalker_Traj():
    def __init__(self, adj_matrix):
        self.G = adj_matrix
        self.sentences = []

    def generate_sentences_bert(self, num_walks=24):
        sts = []
        for _ in trange(num_walks):
            sts.extend(self.random_walks())
        return sts

    def generate_sentences_dw(self, num_walks=24):
        sts = []
        for _ in range(num_walks):
            sts.extend(self.random_walks_dw())
        return sts

    def random_walks(self):
        # random walk with every node as start point once
        walks = []
        nodes = list(range(self.G.shape[0]))
        random.shuffle(nodes)
        for node in nodes:
            if node < len(self.G) - 2:
                walk = [node + 2]
                v = node
                length_walk = random.randint(5, 100)
                for _ in range(length_walk):
                    dst = list(np.where(self.G[v] == 1)[0].tolist())
                    if len(dst) == 0:
                        continue
                    weights = self.G[v][dst]
                    probs = np.array(weights) / np.sum(weights)
                    v = np.random.choice(dst, 1, p=probs)[0]
                    if (v + 2) < len(self.G):
                        walk.append(v + 2)
                walks.append(walk)
        return walks

    def random_walks_dw(self):
        walks = []
        nodes = list(range(self.G.shape[0]))
        random.shuffle(nodes)
        for node in nodes:
            walk = [str(node)]
            v = node
            length_walk = random.randint(5, 100)
            for _ in range(length_walk):
                dst = list(np.where(self.G[v] == 1)[0].tolist())
                weights = self.G[v][dst]
                weights = [w['weight'] for w in weights]
                probs = np.array(weights) / np.sum(weights)
                v = np.random.choice(dst, 1, p=probs)[0]
                walk.append(str(v))
            walks.append(walk)
        return walks


class Trajectory:
    """
    Trajectory class which has methods to get data from a trajectory dataframe.
    """

    def __init__(self, df_path: str):
        self.df = pd.read_csv(df_path, sep=",")

    def generate_TTE_datatset(self):
        """
        Generates dataset for TimeTravel Estimation.
        Returns dataframe with traversed road segments and needed time.
        """
        tte = self.df[["id", "path", "duration"]].copy()
        tte["travel_time"] = tte["duration"].apply(np.sum)
        tte.drop("duration", axis=1, inplace=True)
        return tte

    @staticmethod
    def load_processed_dataset(path):
        df = pd.read_csv(path, index_col=0)
        df["seg_seq"] = df["seg_seq"].swifter.apply(
            lambda x: np.fromstring(
                x.replace("\n", "").replace("(", "").replace(")", "").replace(" ", ""),
                sep=",",
                dtype=np.int,
            )
        )

        return df

    def generate_speed_features(self, df) -> pd.DataFrame:
        """
        Generates features containing average speed, utilization and accelaration
        for each edge i.e road segment.

        Returns:
            pd.DataFrame: features in shape num_edges x features
        """
        rdf = pd.DataFrame({"id": df.id}, index=df.index)
        seg_seqs = self.df["path"].values
        counter = Counter()
        for seqs in tqdm(seg_seqs):
            seqs = seqs[1: -1].split(",")
            seqs = list(map(int, seqs))
            counter.update(Counter(seqs))
        rdf["util"] = rdf.id.map(counter)
        rdf["util"] = (rdf["util"] - rdf["util"].min()) / (
           rdf["util"].max() - rdf["util"].min()
        )  # min max normalization

        # generate average speed feature
        # little bit complicater

        # key: edge_id, value: tuple[speed, count]
        # speed_counter, count_counter = Trajectory.calc_avg_speed(self.df)
        # rdf["avg_speed"] = rdf.id.map(
        #     {
        #         k: (float(speed_counter[k]) / count_counter[k]) * 111000 * 3.6
        #         for k in speed_counter
        #     }
        # )  # calculate average speed in km/h

        # rdf["avg_speed"] = (rdf["avg_speed"] - rdf["avg_speed"].min()) / (
        #    rdf["avg_speed"].max() - rdf["avg_speed"].min()
        # )

        return rdf

    @staticmethod
    def calc_avg_speed(data: pd.DataFrame):
        cpaths = data["path"].values
        opaths = data["opath"].values
        speeds = data["speed"].values
        speed_counter = Counter()
        count_counter = Counter()

        for opath, cpath, speed in tqdm(zip(opaths, cpaths, speeds)):
            last_lidx, last_ridx = 0, 0
            for l, r, s in zip(opath[0::1], opath[1::1], speed):
                if s * 111000 * 3.6 >= 200:  # check unrealistic speed values
                    continue

                lidxs, ridxs = np.where(cpath == l)[0], np.where(cpath == r)[0]
                lidx = lidxs[lidxs >= last_lidx][0]
                ridx = ridxs[(ridxs >= last_ridx) & (ridxs >= lidx)][0]

                assert lidx <= ridx
                traversed_edges = cpath[lidx : ridx + 1]
                speed_counter.update(
                    dict(zip(traversed_edges, [s] * len(traversed_edges)))
                )
                count_counter.update(
                    dict(zip(traversed_edges, [1] * len(traversed_edges)))
                )
                last_lidx, last_ridx = lidx, ridx

        return speed_counter, count_counter

    def generate_time_trajectory_data(self, time_interval_minutes: int):
        ...


def generate_node_traj_adj(adj_dict, add_self_loops=True):
    nodes = sorted(set(adj_dict.keys()).union(*adj_dict.values()))
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    if add_self_loops:
        adj_matrix += np.eye(len(nodes), len(nodes), dtype=int)
    for node, neighbors in adj_dict.items():
        for neighbor in neighbors:
            adj_matrix[node_to_index[node], node_to_index[neighbor]] = 1
    return adj_matrix