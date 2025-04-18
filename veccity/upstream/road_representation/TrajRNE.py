import json
import os
import pdb
from logging import getLogger

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv, InnerProductDecoder
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import (add_self_loops, from_networkx,
                                   negative_sampling, remove_self_loops)

from veccity.upstream.abstract_replearning_model import \
    AbstractReprLearningModel
from veccity.utils import ensure_dir


class TrajRNE(AbstractReprLearningModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()

        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.output_dim = config.get('output_dim', 64)
        self.device = config.get('device')
        
        self.sfc_epochs = config.get('sfc_epochs', 1000)
        self.sre_epochs = config.get('sre_epochs', 10)
        self.sfc_data = data_feature.get('sfc_data')
        self.sre_data = data_feature.get('sre_data')
        self.adj = data_feature.get('adj')
        
        self.txt_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.txt'.\
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './veccity/cache/{}/model_cache/embedding_{}_{}_{}.m'.\
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy'.\
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        
        

    def run(self, train_dataloader=None, eval_dataloader=None):
        if not self.config.get('train') and os.path.exists(self.npy_cache_file):
            return

        self._logger.info("Start SFC training")
        model_sfc = SpatialFlowConvolution(self.sfc_data, self.device, adj=self.adj)
        model_sfc.train(epochs=self.sfc_epochs)
        ensure_dir(f"veccity/cache/{self.exp_id}/model_states/sfc/")
        model_sfc.save_model(path=f"veccity/cache/{self.exp_id}/model_states/sfc/")
        self._logger.info("Finished SFC training")
        
        self._logger.info("Start SRE training")
        model_sre = StructuralRoadEncoder(self.sre_data, self.device, self.adj, out_dim=4)
        model_sre.train(epochs=self.sre_epochs, batch_size=256)
        ensure_dir(f"veccity/cache/{self.exp_id}/model_states/sre/")
        model_sre.save_model(path=f"veccity/cache/{self.exp_id}/model_states/sre/")
        self._logger.info("Finished SRE training")

        # TrajRNE
        self._logger.info("Start TrajRNE training")
        trajrne = TrajRNE_model(self.device, aggregator="concate", models=[model_sfc, model_sre])
        outs = trajrne.load_emb()
        np.save(self.npy_cache_file, outs)
        total_num = sum([param.nelement() for param in model_sfc.model.parameters()])
        total_num += sum([param.nelement() for param in model_sre.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))
        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：(' + str(len(outs)) + ',' + str(len(outs[0])) + ')')


from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class for all evaluated models
    """

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def load_model(self):
        ...

    @abstractmethod
    def load_emb(self):
        ...

class SpatialFlowConvolution(Model):
    def __init__(
            self,
            data,
            device,
            emb_dim: int = 128,
            adj=None,
            norm=False,
    ):
        self.train_data = transform_data(data, adj)
        self.train_data.x_coord = torch.tensor(self.train_data[["x", "y"]].to_numpy(), dtype=torch.float32)
        self.model = SFCSubConv(self.train_data.x_coord.shape[1], emb_dim, norm=norm)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model = self.model.to(device)
        self.device = device

        # 将张量移动到设备（如 GPU 或 CPU）
        self.train_data.x_coord = self.train_data.x_coord.to(device)
        self.train_data.edge_weight = self.train_data.edge_weight.to(device)
        self.train_data.edge_traj_index = self.train_data.edge_traj_index.to(device)

    def train(self, epochs: int = 1000):
        logger = getLogger()
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            z = self.model(
                self.train_data.x_coord,
                self.train_data.edge_traj_index,
                self.train_data.edge_weight,
            )
            loss = self.recon_loss(z, self.train_data.edge_traj_index)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

            if e > 0 and e % 10 == 0:
                logger.info("Epoch: {}, avg_loss: {}".format(e, avg_loss / e))

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        decoder = InnerProductDecoder()
        EPS = 1e-15

        pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def save_model(self, path="save/"):
        torch.save(self.model.state_dict(), os.path.join(path + "/model.pt"))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_emb(self, path):
        np.savetxt(
            os.path.join(path + "/embedding.out"),
            X=self.model.encode(self.train_data.x_coord, self.train_data.edge_index)
            .detach()
            .cpu()
            .numpy(),
        )

    def load_emb(self, path=None):
        if path:
            return np.loadtxt(path)
        return (
            self.model(
                self.train_data.x_coord,
                self.train_data.edge_traj_index,
                self.train_data.edge_weight,
            )
            .detach()
            .cpu()
            .numpy()
        )

    def to(self, device):
        self.model.to(device)
        return self


class SFCSubConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm=False):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.norm_layer = LayerNorm(out_dim)
        self.norm = norm

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        if self.norm:
            x = self.norm_layer(x)
        return x


class GTNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        ...


class StructuralRoadEncoder(Model):
    def __init__(
        self,
        data,
        device,
        adj_data,
        emb_dim: int = 128,
        out_dim=3,
        
    ):
        """
        Initialize SRN2Vec
        Args:
            data (_type_): placeholder
            device (_type_): torch device
            network (nx.Graph): graph of city where nodes are intersections and edges are roads
            emb_dim (int, optional): embedding dimension. Defaults to 128.
        """
        self.data = data
        self.device = device
        self.emb_dim = emb_dim
        self.nodes = adj_data.shape[0]
        self.model = SRN2Vec(self.nodes, device=device, emb_dim=emb_dim, out_dim=out_dim)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_func = nn.BCELoss()

    def train(self, epochs: int = 1000, batch_size: int = 128):
        """
        Train the SRN2Vec Model (load dataset before with .load_data())
        Args:
            epochs (int, optional): epochs to train. Defaults to 1000.
            batch_size (int, optional): batch_size. Defaults to 128.
        """
        self.model.to(self.device)
        loader = DataLoader(
            SRN2VecDataset(self.data, self.nodes),
            batch_size=batch_size,
            shuffle=True,
        )
        logger = getLogger()
        for e in range(epochs):
            self.model.train()
            total_loss = 0
            for i, (X, y) in enumerate(loader):
                X = X.to(self.device)
                y = y.to(self.device)

                self.optim.zero_grad()
                yh = self.model(X)
                loss = self.loss_func(yh.squeeze(), y.squeeze())

                loss.backward()
                self.optim.step()
                total_loss += loss.item()
                if i % 1000 == 0:
                    logger.info(
                        f"Epoch: {e}, Iteration: {i}, sample_loss: {loss.item()}, Avg. Loss: {total_loss/(i+1)}"
                    )

            logger.info(f"Average training loss in episode {e}: {total_loss/len(loader)}")

    def set_dataset(self, data):
        self.data = data
    
    def load_dataset(self, path: str):
        with open(path, "r") as fp:
            self.data = np.array(json.load(fp))

    def save_model(self, path="save/"):
        torch.save(self.model.state_dict(), os.path.join(path + "/model.pt"))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_emb(self, path):
        ...

    def load_emb(self):
        return self.model.embedding.weight.data.cpu()

    def to(self, device):
        self.model.to(device)
        return self


class SRN2Vec(nn.Module):
    def __init__(self, nodes, device, emb_dim: int = 128, out_dim: int = 2):
        super(SRN2Vec, self).__init__()
        self.embedding = nn.Embedding(nodes, emb_dim)
        self.lin_vx = nn.Linear(emb_dim, emb_dim)
        self.lin_vy = nn.Linear(emb_dim, emb_dim)

        self.lin_out = nn.Linear(emb_dim, out_dim)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)
        # y_emb = self.embedding(vy)

        # x = self.lin_vx(emb[:, 0])
        # y = self.lin_vy(emb[:, 1])
        x = emb[:, 0, :] * emb[:, 1, :]  # aggregate embeddings
        x = self.lin_out(x)
        yh = self.act_out(x)

        return yh


class SRN2VecDataset(Dataset):
    def __init__(self, data, num_classes: int):
        self.X = data[:, :2]
        self.y = data[:, 2:]
        self.num_cls = num_classes

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=int), torch.Tensor(
            self.y[idx]
        )  # F.one_ont(self.X[idx], self.num_cls)>



class TrajRNE_model(Model):
    def __init__(self, device, aggregator: str, models: list):
        self.models = models
        self.device = device
        self.aggregator = aggregator

    def train(self):
        ...

    def load_emb(self):
        embs = [m.load_emb() for m in self.models]
        if self.aggregator == "add":
            emb = embs[0]
            for e in embs[1:]:
                emb = emb + e
            return emb

        elif self.aggregator == "concate":
            return np.concatenate(embs, axis=1)

    def load_model(self, path: str):
        ...


def recon_loss(z, pos_edge_index, neg_edge_index=None):
    decoder = InnerProductDecoder()
    EPS = 1e-15

    pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

    # Do not include self-loops in negative samples
    pos_edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index, _ = add_self_loops(pos_edge_index)
    if neg_edge_index is None:
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
    neg_loss = -torch.log(1 - decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

    return pos_loss + neg_loss


def generate_trajid_to_nodeid(network):
    map = {}
    nodes = list(network.line_graph.nodes)
    for index, id in zip(network.gdf_edges.index, network.gdf_edges.fid):
        map[id] = nodes.index(index)
    return map


def transform_data(data, adj):
    G = nx.from_numpy_array(adj.T, create_using=nx.DiGraph)
    data_traj = from_networkx(G)
    data.edge_traj_index = data_traj.edge_index
    data.edge_weight = data_traj.weight
    return data


def generate_dataset(
    data,
    seq_len,
    pre_len,
    time_len=None,
    reconstruct=False,
    split_ratio=0.8,
    normalize=True,
):
    """
    https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-PyTorch/utils/data/functions.py
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        for i in range(data.shape[-1]):
            max_val = np.max(data[:, :, i])
            data[:, :, i] = data[:, :, i] / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    t = seq_len if reconstruct else seq_len - pre_len
    for i in range(len(train_data) - t):
        train_X.append(np.array(train_data[i : i + seq_len]))
        if reconstruct:
            train_Y.append(np.array(train_data[i : i + seq_len, :, 0]))
        else:
            train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(test_data) - t):
        test_X.append(np.array(test_data[i : i + seq_len]))
        if reconstruct:
            test_Y.append(np.array(test_data[i : i + seq_len, :, 0]))
        else:
            test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(
    data,
    seq_len,
    pre_len,
    time_len=None,
    reconstruct=False,
    split_ratio=0.8,
    normalize=True,
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        reconstruct=reconstruct,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset