import numpy as np
import time
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from logging import getLogger
from veccity.upstream.abstract_replearning_model import AbstractReprLearningModel
from veccity.utils import need_train


class ZEMob(AbstractReprLearningModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.config = config
        self.data_feature = data_feature
        self._logger = getLogger()
        # 超参数
        self.output_dim = config.get('output_dim', 64)
        self.iter = config.get('max_epoch', 1000)
        self.batch_size = config.get('batch_size', 64)
        self.lr = config.get('learning_rate', 0.001)

        # 其他参数
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.txt_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './veccity/cache/{}/model_cache/embedding_{}_{}_{}.pkl'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './veccity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.device = self.config.get('device', torch.device('cpu'))

        # 从数据预处理中拿到PPMI矩阵（即M），G*矩阵，以及region的总数和mobility_event的总数，用于生成两个embedding
        # 这里我们生成0-region_num的regionid，即embedding矩阵的编号，分别对应的是数据预处理得到的self.region_idx中的顺序
        # 因此这里并不需要拿到具体的region和event的列表，我们最终只关注region的embedding
        self.ppmi_matrix = torch.from_numpy(self.data_feature.get('ppmi_matrix'))
        self.G_matrix = torch.from_numpy(self.data_feature.get('G_matrix'))
        self.region_num = self.data_feature.get('region_num')
        self.mobility_event_num = self.data_feature.get('mobility_event_num')
        self.region_list = torch.arange(self.region_num)

        # 分批加载数据，这里为了防止大型数据集的G和ppmi过大，需要在cpu分批进行读取
        self.train_data = ZEMobDataSet(self.region_list)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # 设置模型，模型定义见下
        self.ZEMob_model = ZEMobModel(self.region_num, self.mobility_event_num, self.output_dim, self.ppmi_matrix, self.G_matrix, self.device)
        self.ZEMob_model.to(self.device)

    def run(self, train_dataloader=None, eval_dataloader=None):
        if not need_train(self.config):
            return
        start_time = time.time()
        # 优化器
        self.optimizer = optim.SGD(self.ZEMob_model.parameters(), lr=self.lr)

        # 共训练iter轮
        for epoch in range(1, self.iter + 1):
            self.train(epoch)
        t1 = time.time()-start_time
        self._logger.info('cost time is '+str(t1/self.iter))
        # 保存结果
        with open(self.txt_cache_file, 'w', encoding='UTF-8') as f:
            f.write('{} {}\n'.format(self.region_num, self.output_dim))
            embeddings = self.ZEMob_model.zone_embedding.weight.data.to('cpu').numpy()
            for i in range(self.region_num):
                embedding = embeddings[i]
                embedding = str(i) + ' ' + (' '.join(map((lambda x: str(x)), embedding)))
                f.write('{}\n'.format(embedding))
        np.save(self.npy_cache_file, embeddings)
        torch.save(self.ZEMob_model.state_dict(), self.model_cache_file)
        
        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：(' + str(self.region_num) + ',' + str(self.output_dim) + ')')

    # 训练
    def train(self, epoch):
        train_loss = 0
        for data in self.train_loader:
            self.optimizer.zero_grad()
            loss = self.ZEMob_model.forward(data)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        self._logger.info('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

class ZEMobModel(nn.Module):
    def __init__(self, zone_num, mobility_event_num, embedding_dim, ppmi_matrix, G_matrix, device):
        super(ZEMobModel, self).__init__()

        self.device = device
        self._logger = getLogger()
        self.zone_num = zone_num
        self.mobility_event_num = mobility_event_num
        self.embedding_dim = embedding_dim
        self.ppmi_matrix = ppmi_matrix
        self.G_matrix = G_matrix

        # 模型主要结构为两层embedding
        self.zone_embedding = nn.Embedding(self.zone_num, self.embedding_dim)
        self.event_embedding = nn.Embedding(self.mobility_event_num, self.embedding_dim)

        # 初始化embedding
        initrange = 0.5 / self.embedding_dim
        self.zone_embedding.weight.data.uniform_(-initrange, initrange)
        self.event_embedding.weight.data.uniform_(-initrange, initrange)

        # 一些辅助变量，在分批进行loss的计算时，当我们选定了一部分的z，需要对每一个z遍历所有的e
        self.all_events = torch.arange(mobility_event_num).to(self.device)

        # 判断GPU内存是足够否全部加载到GPU
        self.memory_sufficient=False
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            memory_usage = self.G_matrix.element_size() * self.G_matrix.nelement()\
                           +self.ppmi_matrix.element_size() * self.ppmi_matrix.nelement()
            memory_allocated = torch.cuda.max_memory_allocated(self.device)
            self._logger.info("GPU : {} Memory is sufficient, {}/{}"
                                  .format(self.device,memory_usage,total_memory))
            if memory_usage + memory_allocated <= total_memory:
                self.G_matrix = self.G_matrix.to(self.device)
                self.ppmi_matrix = self.ppmi_matrix.to(self.device)
                self.memory_sufficient = True
                self._logger.info("G_matrix and ppmi are already on GPU")

        else:
            self._logger.info("CUDA is not available")

    # 前向传播，公式即为文章中需要最小化的函数，得到的结果即为loss
    # 注意ppmi和G在cpu上
    def forward(self, batch):
        batch_gpu = batch.to(self.device)
        batch_zone = self.zone_embedding(batch_gpu)
        batch_event = self.event_embedding(self.all_events)
        batch_ppmi = self.ppmi_matrix[batch]
        batch_G = self.G_matrix[batch]
        # 内存不够，分批次放到gpu
        if not self.memory_sufficient:
            batch_ppmi = batch_ppmi.to(self.device)
            batch_G = batch_G.to(self.device)
        return torch.sum(torch.pow(torch.sub(batch_ppmi, torch.mm(batch_zone, batch_event.t())), 2) * batch_G) / 2

# 数据加载
class ZEMobDataSet(Dataset):
    def __init__(self, zones):
        self.zones = zones

    def __len__(self):
        return len(self.zones)

    def __getitem__(self, idx):
        zone = self.zones[idx]
        return zone
