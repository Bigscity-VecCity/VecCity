from logging import getLogger
import os
from tqdm import tqdm

from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel
import numpy as np
import torch
import torch.nn as nn
import time


class SRN2Vec(AbstractReprLearningModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.config = config
        self.device = config.get('device')
        self.dataloader = data_feature.get('dataloader')
        self.num_nodes = data_feature.get("num_nodes")
        self._logger = getLogger()
        self.output_dim = config.get('output_dim', 128)
        self.iter = config.get('max_epoch', 10)
        self.model = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.dataset = config.get('dataset', '')
        self.txt_cache_file = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './veccity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.road_embedding_path = './veccity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model = SRN2VecModule(node_num=self.num_nodes, device=self.device, emb_dim=self.output_dim, out_dim=2).to(self.device)

    def calculate_loss(self, batch):

        X = batch[0].to(self.device)
        y = batch[1].float().to(self.device)
        yh = self.model(X)
        loss = self.model.loss_func(yh.squeeze(), y.squeeze())
        return loss
    
    def run(self,train_dataloader,eval_dataloader=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.iter):
            self.model.train()
            for batch in tqdm(self.dataloader):
                optimizer.zero_grad()
                loss = self.calculate_loss(batch)
                loss.backward()
                optimizer.step()
            if eval_dataloader is not None:
                self.evaluate(eval_dataloader)
        self.save_model()
        self.save_embedding()
    def save_model(self):
        torch.save(self.model, self.model_cache_file)
        self._logger.info('Model saved to {}'.format(self.model_cache_file))
    
    def save_embedding(self):
        self.model.eval()
        embedding = self.model.embedding.weight.cpu().detach().numpy()
        np.save(self.npy_cache_file, embedding)
        self._logger.info('Embedding saved to {}'.format(self.npy_cache_file))


    def encode(self, batch):
        X = batch[0].to(self.device)
        return self.model.embedding(X)


class SRN2VecModule(nn.Module):
    def __init__(self, node_num, device, emb_dim: int = 128, out_dim: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(node_num, emb_dim)
        self.lin_vx = nn.Linear(emb_dim, emb_dim)
        self.lin_vy = nn.Linear(emb_dim, emb_dim)
        self.lin_out = nn.Linear(emb_dim, out_dim)
        self.act_out = nn.Sigmoid()
        self.loss_func = nn.BCELoss()

    def forward(self, x):
        emb = self.embedding(x)
        x = emb[:, 0, :] * emb[:, 1, :]  # aggregate embeddings
        x = self.lin_out(x)
        yh = self.act_out(x)
        return yh
