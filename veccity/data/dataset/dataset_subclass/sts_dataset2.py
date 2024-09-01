import datetime
import torch
from veccity.data.dataset.dataset_subclass.bert_base_dataset import padding_mask
from veccity.data.dataset.lineseq_dataset import LineSeqDataset
from veccity.data.preprocess import preprocess_all, cache_dir
from torch.utils.data import Dataset,DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd


class STSDataset(LineSeqDataset):
    def __init__(self, config,filte=True):
        super().__init__(config)
        self.collate_fn = collate_superv_sts
        self.train_trajs_path=cache_dir+"/{}/train_path_10000.csv".format(self.dataset)
        self.val_trajs_path=cache_dir+"/{}/val_path_5000.csv".format(self.dataset)
        self.test_trajs_path=cache_dir+"/{}/test_path_5000.csv".format(self.dataset)
        self.train_labels_path=cache_dir+"/{}/simi_label_train.npy".format(self.dataset)
        self.val_labels_path=cache_dir+"/{}/simi_label_val.npy".format(self.dataset)
        self.test_labels_path=cache_dir+"/{}/simi_label_test.npy".format(self.dataset)
        self.filte=filte


    def process_trajs(self):
        train_trajs=pd.read_csv(self.train_trajs_path,sep=';')
        val_trajs=pd.read_csv(self.val_trajs_path,sep=';')
        test_trajs=pd.read_csv(self.test_trajs_path,sep=';')
        self.train_labels=np.load(self.train_labels_path)
        self.val_labels=np.load(self.val_labels_path)
        self.test_labels=np.load(self.test_labels_path)


        train_traj=train_trajs.path.apply(eval).tolist()
        train_tlist=train_trajs.tlist.apply(eval).tolist()
        train_len=train_trajs.hop.tolist()

        val_traj=val_trajs.path.apply(eval).tolist()
        val_tlist=val_trajs.tlist.apply(eval).tolist()
        val_len=val_trajs.hop.tolist()

        test_traj=test_trajs.path.apply(eval).tolist()
        test_tlist=test_trajs.tlist.apply(eval).tolist()
        test_len=test_trajs.hop.tolist()



        train_dataset=List_Dataset(train_traj,train_tlist,train_len,self.train_labels,self.seq_len,self.vocab,self.add_cls,self.filte)
        val_dataset=List_Dataset(val_traj,val_tlist,val_len,self.val_labels,self.seq_len,self.vocab,self.add_cls,self.filte)
        test_dataset=List_Dataset(test_traj,test_tlist,test_len,self.test_labels,self.seq_len,self.vocab,self.add_cls,self.filte)

        self.train_dataloader=DataLoader(train_dataset,self.batch_size,collate_fn=collate_superv_sts,shuffle=False)
        self.val_dataloader=DataLoader(val_dataset,self.batch_size,collate_fn=collate_superv_sts,shuffle=False)
        self.test_dataloader=DataLoader(test_dataset,self.batch_size,collate_fn=collate_superv_sts,shuffle=False)

        return self.train_dataloader,self.val_dataloader,self.test_dataloader   

    
    
    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        self._logger.info("Loading Dataset!")
        self.process_trajs()
        self._logger.info('Size of dataset: ' + str(len(self.train_dataloader)) +
                         '/' + str(len(self.val_dataloader)) + '/' + str(len(self.test_dataloader)))

        self._logger.info("Creating Dataloader!")
        return self.train_dataloader,self.val_dataloader,self.test_dataloader
        


class List_Dataset(Dataset):
    def __init__(self,traj,tlist,lens,labels,seq_len,vocab,add_cls,filte=True):
        self.traj=traj
        self.tlist=tlist
        self.lens=list(lens)
        self.seq_len=seq_len
        self.add_cls=add_cls
        self.labels=labels
        self.vocab=vocab
        if filte:
            self.temporal_mat_list,self.traj_list,self.lens = self.datapropocess()
        else:
            self.temporal_mat_list,self.traj_list = self.datapropocess2()
    
    def datapropocess2(self):
        temporal_mat_list=[]
        traj_list=[]
        for i in tqdm(range(len(self.traj))):
            loc_list = self.traj[i]
            tim_list = self.tlist[i]
            if type(tim_list) == type("str"):
                tim_list=eval(tim_list)

            # new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in loc_list]
            new_tim_list = [datetime.datetime.fromtimestamp(tim) for tim in tim_list]
            new_loc_list=loc_list
            # tim_list=tim_list 
            minutes = [new_tim.hour * 60 + new_tim.minute + 1 for new_tim in new_tim_list]
            weeks = [new_tim.weekday() + 1 for new_tim in new_tim_list]
            usr_list = [self.vocab.unk_index] * len(new_loc_list)
            temporal_mat = self._cal_mat(tim_list)
            temporal_mat_list.append(temporal_mat)
            traj_feat = np.array([new_loc_list, tim_list, minutes, weeks, usr_list]).transpose((1, 0))
            traj_list.append(traj_feat)
        return temporal_mat_list,traj_list

    def datapropocess(self):
        temporal_mat_list=[]
        traj_list=[]
        lens_list=[]
        for i in tqdm(range(len(self.traj))):
            loc_list = self.traj[i][:self.seq_len]
            tim_list = self.tlist[i][:self.seq_len]
            lens=len(loc_list)
            if type(tim_list) == type("str"):
                tim_list=eval(tim_list)

            new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in loc_list]
            new_tim_list = [datetime.datetime.fromtimestamp(tim) for tim in tim_list]
            # new_loc_list=loc_list
            # tim_list=tim_list 
            minutes = [new_tim.hour * 60 + new_tim.minute + 1 for new_tim in new_tim_list]
            weeks = [new_tim.weekday() + 1 for new_tim in new_tim_list]
            usr_list = [self.vocab.unk_index] * len(new_loc_list)
            if self.add_cls:
                new_loc_list = [self.vocab.sos_index] + new_loc_list
                minutes = [self.vocab.pad_index] + minutes
                weeks = [self.vocab.pad_index] + weeks
                usr_list = [usr_list[0]] + usr_list
                tim_list = [tim_list[0]] + tim_list
            temporal_mat = self._cal_mat(tim_list)
            temporal_mat_list.append(temporal_mat)
            traj_feat = np.array([new_loc_list, tim_list, minutes, weeks, usr_list]).transpose((1, 0))
            traj_list.append(traj_feat)
            lens_list.append(lens)
        return temporal_mat_list,traj_list,lens_list
        

    def _cal_mat(self, tim_list):
        # calculate the temporal relation matrix
        seq_len = len(tim_list)
        mat = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                off = abs(tim_list[i] - tim_list[j])
                mat[i][j] = off
        return mat  # (seq_len, seq_len)


    def __getitem__(self, index):
        return torch.LongTensor(self.traj_list[index]),self.lens[index],torch.LongTensor(self.temporal_mat_list[index]),index

    def __len__(self):
        return len(self.traj)



def collate_superv_sts(data, max_len=None, vocab=None, add_cls=False):
    batch_size = len(data)
    features, lengths, temporal_mat, index = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    if max_len is None:
        max_len = max(lengths)
        
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)  # (batch_size, padded_length, padded_length)

    for i in range(batch_size):
        end = min(lengths[i], max_len)
        try:
            X[i, :end, :] = features[i][:end]
        except Exception:
            end=features[i].shape[0]            
            X[i, :end, :] = features[i][:end]
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]


    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    batch={}
    batch['seq']=X.long()
    batch['length'] = lengths
    batch['padding_masks']=padding_masks
    batch['batch_temporal_mat'] = batch_temporal_mat.long()
    batch['index']=index
    return batch
