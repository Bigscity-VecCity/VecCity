import os
import math
import pandas as pd
import numpy as np
from logging import getLogger
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import copy
from veccity.data.dataset.dataset_subclass.sts_dataset2 import STSDataset

from veccity.downstream.downstream_models.abstract_model import AbstractModel
from veccity.data.preprocess import preprocess_detour
from tqdm import tqdm

class STSModel(nn.Module):
    def __init__(self, embedding,device,input_size=128,dropout_prob=0.2):
        super().__init__()
        self.traj_encoder = embedding
        self.criterion = torch.nn.MSELoss(reduction='none')
        # self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        # self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.projection=nn.Sequential(nn.Linear(input_size,input_size),nn.ReLU(),nn.Linear(input_size,input_size))
        self.device=device
        self.temperature=0.05
        
    def forward(self, batch):
        # out=self.projection(self.traj_encoder.encode_sequence(batch)) # bd
        out=self.traj_encoder.encode_sequence(batch)
        return out

    def calculate_loss(self,batch,labels):
        out_view=self.forward(batch)
        sim_labels=torch.from_numpy(labels).to(self.device)
        pred_l1_simi = torch.cdist(out_view,out_view)
        pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal=1) == 1].float()
        truth_l1_simi = sim_labels[torch.triu(torch.ones(sim_labels.shape), diagonal=1) == 1].float()
        
        batch_loss_list = self.criterion(pred_l1_simi, truth_l1_simi)
        batch_loss = torch.sum(batch_loss_list)
        num_active = len(batch_loss_list)  # batch_size
        mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization
        
        return mean_loss



class STSExecutor(AbstractModel):
    def __init__(self, config):
        preprocess_detour(config)
        self._logger = getLogger()
        self._logger.warning('Evaluating Trajectory Similarity Search')
        self.dataset=STSDataset(config,filte=True)
        self.train_dataloader,self.val_dataloader, self.test_dataloader = self.dataset.get_data()
        self.device=config.get('device')        
        self.epochs=config.get('task_epoch',50)
        self.learning_rate=5e-4#config.get('learning_rate',1e-3)
        self.weight_decay=config.get('weight_decay',1e-3)
        self.train_labels=self.dataset.train_labels
        self.val_labels=self.dataset.val_labels
        self.test_labels=self.dataset.test_labels
    
    def run(self,model,**kwargs):
        self._logger.info("-------- STS START --------")
        # self.evaluation()
        self.train(model,**kwargs)
        
        return self.result

    def train(self,model,**kwargs):
        """
        返回评估结果
        """
        self.model = STSModel(embedding=model, device=self.device)
        self.model.to(self.device)
        optimizer = Adam(lr=self.learning_rate, params=self.model.parameters(), weight_decay=self.weight_decay)
        # # 先测试
        self.evaluation()
        best_loss=-1
        best_model=None
        best_epoch=0
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for step,batch in enumerate(self.train_dataloader):
                batch.update(kwargs)
                indexs=batch['index']
                labels=self.train_labels[indexs,:][:,indexs]
                loss=self.model.calculate_loss(batch,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            total_loss=total_loss/len(self.train_dataloader)
            
            valid_loss=0.0
            with torch.no_grad():
                for step,batch in enumerate(self.val_dataloader):
                    batch.update(kwargs)
                    indexs=batch['index']
                    labels=self.val_labels[indexs,:][:,indexs]
                    loss=self.model.calculate_loss(batch,labels)
                    valid_loss += loss.item()
            valid_loss=valid_loss/len(self.val_dataloader)

            if best_loss==-1 or valid_loss<best_loss:
                best_model=copy.deepcopy(self.model)
                best_loss=valid_loss
                best_epoch=epoch
            self._logger.info("epoch {} complete! training loss {:.2f}, valid loss {:2f}, best_epoch {}, best_loss {:2f}".format(epoch, total_loss, valid_loss,best_epoch,best_loss))
        if best_model:
            self.model=best_model

        self.evaluation(**kwargs)
        return self.result

    def evaluation(self,**kwargs):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        #epoch_loss,sim_preds = self._test_epoch(test_dataloader, 0, mode='Test')
        sim_preds = None
        sim_target = None
        batch_loss = None
        traj_embs=[]
        for i, batch in tqdm(enumerate(self.test_dataloader)):
            batch.update(kwargs)
            traj_emb = self.model(batch)
            traj_emb=traj_emb.cpu().detach()
            traj_embs.append(traj_emb)
        traj_embs=torch.vstack(traj_embs)
        sim_preds = torch.cdist(traj_embs, traj_embs, 1).numpy()
        sim_target = self.test_labels
        self._logger.info('test_simi_preds is '+str(sim_preds.shape))
        self._logger.info('test_simi_traget is ' + str(sim_target.shape))
        self.evaluate_most_sim(sim_target,sim_preds)
    
    def evaluate_most_sim(self,sim_label,sim_preds):
        #给sim_label和sim_pred的对角元素都变成inf，不然label全成自己了
        np.fill_diagonal(sim_preds,np.inf)
        np.fill_diagonal(sim_label,np.inf)
        #给sim_preds每一行排序
        topk = [1, 5, 10,20]
        sorted_pred_index = sim_preds.argsort(axis=1)
        label_most_sim = np.argmin(sim_label,axis=1)
        total_num = sim_preds.shape[0]
        self.result = {}
        hit = {}
        for k in topk:
            hit[k] = 0
        rank = 0
        rank_p = 0.0
        for i in tqdm(range(total_num)):
            # 在sim_labe找到这一行distance最小的的作为真正的最相似轨迹
            label = label_most_sim[i]
            rank_list = list(sorted_pred_index[i])
            rank_index = rank_list.index(label)
            rank += (rank_index + 1)
            rank_p += 1.0 / (rank_index + 1)
            for k in topk:
                if label in sorted_pred_index[i][:k]:
                    hit[k] += 1
        self.result['MR'] = rank / total_num
        self.result['MRR'] = rank_p / total_num
        for k in topk:
            self.result['HR@{}'.format(k)] = hit[k] / total_num
        self._logger.info("Evaluate result is {}".format(self.result))

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        pass