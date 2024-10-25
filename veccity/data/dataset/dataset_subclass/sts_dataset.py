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
        self.ori_path= cache_dir+'/{}/ori_trajs.npz'.format(self.dataset)
        self.qry_path= cache_dir+'/{}/query_trajs.npz'.format(self.dataset)
        print(self.qry_path)
        # self.train_ori_path=cache_dir+'/{}/{}_decup_origin_test_topk0.2_0.2_1.0_3000.csv'.format(self.dataset,self.dataset)
        # self.train_qry_path=cache_dir+'/{}/{}_decup_detoured_test_topk0.2_0.2_1.0_3000.csv'.format(self.dataset,self.dataset)
        # # self.test_ori_path=cache_dir+'/{}/{}_decup_origin_test_topk0.2_0.2_1.0_3000.csv'.format(self.dataset,self.dataset)
        # # self.test_qry_path=cache_dir+'/{}/{}_decup_detoured_test_topk0.2_0.2_1.0_3000.csv'.format(self.dataset,self.dataset)
        # self.test_ori_path=cache_dir+'/{}/{}_decup_others_test_topk0.2_0.2_1.0_3000_30000.csv'.format(self.dataset,self.dataset)
        # self.test_qry_path=cache_dir+'/{}/{}_decup_othersdetour_test_topk0.2_0.2_1.0_3000_30000.csv'.format(self.dataset,self.dataset)
        self.filte=filte


    # def process_trajs(self):
    #     train_ori_data=pd.read_csv(self.train_ori_path,sep=';')
    #     train_qry_data=pd.read_csv(self.train_qry_path,sep=';')
    #     test_ori_data=pd.read_csv(self.test_ori_path,sep=';')
    #     test_qry_data=pd.read_csv(self.test_qry_path,sep=';')

    #     ori_traj=train_ori_data.path.apply(eval).tolist()
    #     ori_tlist=train_ori_data.tlist.apply(eval).tolist()
    #     ori_len=train_ori_data.hop.tolist()

    #     ori_traj1=test_ori_data.path.apply(eval).tolist()
    #     ori_tlist1=test_ori_data.tlist.apply(eval).tolist()
    #     ori_len1=test_ori_data.hop.tolist()

    #     qry_traj=train_qry_data.path.apply(eval).tolist()
    #     qry_tlist=train_qry_data.tlist.apply(eval).tolist()
    #     qry_len=train_qry_data.hop.tolist()

    #     qry_traj1=test_qry_data.path.apply(eval).tolist()
    #     qry_tlist1=test_qry_data.tlist.apply(eval).tolist()
    #     qry_len1=test_qry_data.hop.tolist()

    #     ori_dataset=List_Dataset(ori_traj,ori_tlist,ori_len,self.seq_len,self.vocab,self.add_cls,self.filte)
    #     qry_dataset=List_Dataset(qry_traj,qry_tlist,qry_len,self.seq_len,self.vocab,self.add_cls,self.filte)
    #     self.train_ori_dataloader=DataLoader(ori_dataset,self.batch_size,collate_fn=collate_superv_sts,shuffle=False)
    #     self.train_qry_dataloader=DataLoader(qry_dataset,self.batch_size,collate_fn=collate_superv_sts,shuffle=False)

    #     ori_dataset1=List_Dataset(ori_traj1,ori_tlist1,ori_len1,self.seq_len,self.vocab,self.add_cls,self.filte)
    #     qry_dataset1=List_Dataset(qry_traj1,qry_tlist1,qry_len1,self.seq_len,self.vocab,self.add_cls,self.filte)
    #     self.test_ori_dataloader=DataLoader(ori_dataset1,self.batch_size,collate_fn=collate_superv_sts,shuffle=False)
    #     self.test_qry_dataloader=DataLoader(qry_dataset1,self.batch_size,collate_fn=collate_superv_sts,shuffle=False)
    #     return self.train_ori_dataloader,self.train_qry_dataloader,self.test_ori_dataloader,self.test_qry_dataloader


    def process_trajs(self):
        ori_data=np.load(self.ori_path,allow_pickle=True)
        qry_data=np.load(self.qry_path,allow_pickle=True)

        ori_traj=ori_data['trajs'].tolist()[:8000]
        ori_tlist=ori_data['tlist'].tolist()[:8000]
        ori_len=ori_data['lengths'].tolist()[:8000]

        ori_traj1=ori_data['trajs'].tolist()[8000:]
        ori_tlist1=ori_data['tlist'].tolist()[8000:]
        ori_len1=ori_data['lengths'].tolist()[8000:]

        qry_traj=qry_data['trajs'].tolist()[:8000]
        qry_tlist=qry_data['tlist'].tolist()[:8000]
        qry_len=qry_data['lengths'].tolist()[:8000]

        qry_traj1=qry_data['trajs'].tolist()[8000:]
        qry_tlist1=qry_data['tlist'].tolist()[8000:]
        qry_len1=qry_data['lengths'].tolist()[8000:]

        ori_dataset=List_Dataset(ori_traj,ori_tlist,ori_len,self.seq_len,self.vocab,self.add_cls)
        qry_dataset=List_Dataset(qry_traj,qry_tlist,qry_len,self.seq_len,self.vocab,self.add_cls)
        self.train_ori_dataloader=DataLoader(ori_dataset,self.batch_size,collate_fn=collate_superv_sts,shuffle=False)
        self.train_qry_dataloader=DataLoader(qry_dataset,self.batch_size,collate_fn=collate_superv_sts,shuffle=False)

        ori_dataset1=List_Dataset(ori_traj1,ori_tlist1,ori_len1,self.seq_len,self.vocab,self.add_cls)
        qry_dataset1=List_Dataset(qry_traj1,qry_tlist1,qry_len1,self.seq_len,self.vocab,self.add_cls)
        self.test_ori_dataloader=DataLoader(ori_dataset1,self.batch_size,collate_fn=collate_superv_sts,shuffle=False)
        self.test_qry_dataloader=DataLoader(qry_dataset1,self.batch_size,collate_fn=collate_superv_sts,shuffle=False)
        return self.train_ori_dataloader,self.train_qry_dataloader,self.test_ori_dataloader,self.test_qry_dataloader

    

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
        self._logger.info('Size of dataset: ' + str(len(self.train_ori_dataloader)) +
                         '/' + str(len(self.test_ori_dataloader)) + '/' + str(len(self.test_ori_dataloader)))

        self._logger.info("Creating Dataloader!")
        return self.train_ori_dataloader,self.train_qry_dataloader,self.test_ori_dataloader,self.test_qry_dataloader
        


class List_Dataset(Dataset):
    def __init__(self,traj,tlist,lens,seq_len,vocab,add_cls,filte=True):
        self.traj=traj
        self.tlist=tlist
        self.lens=list(lens)
        self.seq_len=128#seq_len
        self.add_cls=add_cls
        self.vocab=vocab
        if filte:
            self.temporal_mat_list,self.traj_list = self.datapropocess()#,self.lens
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

            loc_list = self.traj[i]
            tim_list = self.tlist[i]
            
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
            lens_list.append(len(new_loc_list))
        return temporal_mat_list,traj_list#,lens_list
        

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
        return torch.LongTensor(self.traj_list[index]),self.lens[index],torch.LongTensor(self.temporal_mat_list[index])

    def __len__(self):
        return len(self.traj)



def collate_superv_sts(data, max_len=None, vocab=None, add_cls=False):
    batch_size = len(data)
    features, lengths, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

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
    return batch

def collate_unsuperv_down(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)

    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)
    batch={}
    batch['seq']=X.long()
    batch['length'] = lengths
    batch['padding_masks']=padding_masks
    batch['batch_temporal_mat'] = batch_temporal_mat.long()
    return batch