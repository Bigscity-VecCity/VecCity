from abc import ABC
from itertools import zip_longest
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from logging import getLogger

import torch.utils
import torch.utils.data
from veccity.upstream.poi_representation.utils import weight_init
from torch.nn.utils.rnn import pack_padded_sequence
from veccity.downstream.utils import accuracy
from torch.utils.data import Dataset
import copy


class TrajectoryPredictor(nn.Module, ABC):
    def __init__(self, embed_layer, num_slots, aux_embed_size,input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.__dict__.update(locals())

        self.time_embed = nn.Embedding(num_slots+1, aux_embed_size)

        self.encoder = nn.LSTM(input_size + aux_embed_size, hidden_size, num_layers, dropout=0.3,batch_first=True)
        self.out_linear = nn.Sequential(nn.Linear(hidden_size, hidden_size*2),
                                        nn.ReLU(), nn.Linear(hidden_size*2, output_size))
        self.apply(weight_init)

        self.embed_layer = embed_layer
        try:
            self.add_module('embed_layer', self.embed_layer)
        except Exception:
            pass
        
    def forward(self, inputs):
        self.encoder.flatten_parameters()
        full_embed = self.embed_layer.encode(inputs)  # (batch_size, seq_len, input_size)
        aux_input = self.time_embed(inputs['hour'])

        lstm_input = torch.cat([full_embed, aux_input],dim=-1)  # (batch_size, seq_len, input_size + aux_embed_size * 2)

        src_padded_embed = pack_padded_sequence(lstm_input, inputs['length'], batch_first=True, enforce_sorted=False)
        _, hc = self.encoder(src_padded_embed)
        hc=hc[0][-1]

        out = self.out_linear(hc)
        return out
    

def pad_to_tensor(data,fill_value=0):
    src=np.transpose(np.array(list(zip_longest(*data, fillvalue=fill_value))))
    return torch.from_numpy(src)

def split_src_trg(full,pre_len=1):
    src_seq, trg_seq = zip(*[[s[:-pre_len], s[-pre_len:]] for s in full])
    return src_seq,trg_seq


class collection:
    def __init__(self,padding_value,device) -> None:
        self.padding_value=padding_value
        self.device=device

    def collection_TUL(self,batch):
        # build batch
        user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
        inputs_seq=pad_to_tensor(full_seq,self.padding_value).long().to(self.device)
        inputs_weekday=pad_to_tensor(weekday,0).long().to(self.device)
        inputs_timestamp=pad_to_tensor(timestamp).to(self.device)
        length=np.array(length)
        imputs_time_delta = pad_to_tensor(time_delta).to(self.device)
        dist =  pad_to_tensor(dist).to(self.device)
        lat = pad_to_tensor(lat).to(self.device)
        lng = pad_to_tensor(lng).to(self.device)
        inputs_hour = (inputs_timestamp % (24 * 60 * 60) / 60 / 60).long()
        src_duration = ((inputs_timestamp[:, 1:] - inputs_timestamp[:, :-1]) % (24 * 60 * 60) / 60 / 60).long()
        src_duration = torch.clamp(src_duration, 0, 23)
        res=torch.zeros([src_duration.size(0),1],dtype=torch.long).to(self.device)
        inputs_duration = torch.hstack([res,src_duration])
        # user_index=torch.tensor(user_index).long().to(self.device)
        targets=torch.tensor(user_index).long().to(self.device)
        
        inputs={
            'seq':inputs_seq,
            'timestamp':inputs_timestamp,
            'length':length,
            'time_delta':imputs_time_delta,
            'hour':inputs_hour,
            'duration':inputs_duration,
            'weekday':inputs_weekday,
            'dist':dist,
            'lat':lat,
            'lng':lng,
            'user':user_index
        }

        return inputs, targets
    
    def collection_LP(self,batch):
        # build batch
        user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
        src_seq, trg_seq = split_src_trg(full_seq)
        inputs_seq=pad_to_tensor(src_seq,self.padding_value).long().to(self.device)
        targets=torch.tensor(trg_seq).squeeze().to(self.device)
        
        src_weekday,_ = split_src_trg(weekday)
        inputs_weekday=pad_to_tensor(src_weekday,0).long().to(self.device)

        src_time,_ = split_src_trg(timestamp)
        inputs_timestamp=pad_to_tensor(src_time).to(self.device)
        
        length=np.array(length)-1

        src_td,_ = split_src_trg(time_delta)
        imputs_time_delta = pad_to_tensor(src_td).to(self.device)

        src_dist,_ = split_src_trg(dist)
        dist =  pad_to_tensor(src_dist).to(self.device)

        src_lat,_ = split_src_trg(lat)
        src_lng,_ = split_src_trg(lng)
        lat = pad_to_tensor(src_lat).to(self.device)
        lng = pad_to_tensor(src_lng).to(self.device)
        inputs_hour = (inputs_timestamp % (24 * 60 * 60) / 60 / 60).long()
        src_duration = ((inputs_timestamp[:, 1:] - inputs_timestamp[:, :-1]) % (24 * 60 * 60) / 60 / 60).long()
        src_duration = torch.clamp(src_duration, 0, 23)
        res=torch.zeros([src_duration.size(0),1],dtype=torch.long).to(self.device)
        inputs_duration = torch.hstack([res,src_duration])
        user_index=torch.tensor(user_index).long().to(self.device)
        
        inputs={
            'seq':inputs_seq,
            'timestamp':inputs_timestamp,
            'length':length,
            'time_delta':imputs_time_delta,
            'hour':inputs_hour,
            'duration':inputs_duration,
            'weekday':inputs_weekday,
            'dist':dist,
            'lat':lat,
            'lng':lng,
            'user':user_index
        }

        return inputs, targets

class List_dataset(Dataset):
    def __init__(self,data):
        self.data=data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def trajectory_based_classification(train_set, test_set, num_class, embed_layer,embed_size,hidden_size, num_epoch, num_loc,batch_size,task='LP', aux_embed_size=32, device="CPU"):
    # build dataset
    eval_ind=int(len(train_set)*0.8)
    eval_set=train_set[-eval_ind:]
    train_set=train_set[:eval_ind]
    logger=getLogger()
    collect=collection(num_loc,device)

    train_dataloader=torch.utils.data.DataLoader(List_dataset(train_set),batch_size=batch_size,shuffle=True,collate_fn=collect.collection_LP if task=="LP" else collect.collection_TUL)
    eval_dataloader=torch.utils.data.DataLoader(List_dataset(eval_set),batch_size=batch_size,shuffle=False,collate_fn=collect.collection_LP if task=="LP" else collect.collection_TUL)
    test_dataloader=torch.utils.data.DataLoader(List_dataset(test_set),batch_size=batch_size,shuffle=False,collate_fn=collect.collection_LP if task=="LP" else collect.collection_TUL)
    # build model
    model=TrajectoryPredictor(embed_layer,num_slots=24,aux_embed_size=aux_embed_size,input_size=embed_size,hidden_size=hidden_size,output_size=num_class,num_layers=2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()
    # train 
    best_model=model
    best_acc=0
    patience=10
    for epoch in range(num_epoch):
        losses=[]
        for (inputs,targets) in train_dataloader:
            preds=model(inputs)
            loss=loss_func(preds,targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # valid
        model.eval()
        y_preds=[]
        y_trues=[]
        for (inputs,targets) in eval_dataloader:
            preds=model(inputs)
            preds=preds.argmax(-1)
            y_preds.extend(preds.cpu().detach().tolist())
            y_trues.extend(targets.cpu().detach().tolist())

        model.train()
        
        c_acc=accuracy_score(y_trues,y_preds)
        if c_acc > best_acc:
            patience=10
            best_acc=c_acc
            best_model=copy.deepcopy(model)
        else:
            patience-=1
            if patience==0:
                break
        
        logger.info(f"epoch:{epoch} loss:{round(np.mean(losses),4)} valid_acc:{round(c_acc,4)} best_acc:{round(best_acc,4)}")
    model=best_model
    # test
    model.eval()
    y_preds=[]
    y_trues=[]
    for (inputs,targets) in test_dataloader:
        preds=model(inputs)
        y_preds.extend(preds.cpu().detach())
        if len(targets.shape)==0:
            targets=targets.unsqueeze(0)
        y_trues.extend(targets.cpu().detach())

    y_preds=torch.vstack(y_preds)
    y_trues=torch.vstack(y_trues)

    acc1,acc5=accuracy(y_preds,y_trues,topk=(1,5))
    pres = y_preds.argmax(-1)
    f1ma=f1_score(y_trues.numpy(), pres.numpy(), average='macro')
    result=[acc1.item(),acc5.item(),f1ma]
    logger.info(f"task:{task} acc1:{round(acc1.item(),4)} acc5:{round(acc5.item(),4)} f1ma:{round(f1ma,4)}")
    return result

