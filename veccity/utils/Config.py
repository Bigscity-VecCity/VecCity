import os
import random
import numpy
import torch


def set_seed(seed = -1):
    if seed == -1:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# all DEFAULT hyperparamters for models.
class Config:
    seed = 2001

    debug = False
    # dumpfile_uniqueid: To enable running same model at the same time,
    # the dumped model file will have diff file names
    # e.g. _timestamp_in_nanoseconds
    dumpfile_uniqueid = '' 

    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.abspath(__file__)[:-10] # dont use os.getcwd()

    # !! values here will be updated by post_value_updates() !!
    dataset = '' # 'SF'
    dataset_prefix = '' # 'OSM_SanFrancisco_downtown_raw'
    trajsimi_prefix = '' # 'sfcab_len60_10k_mm_from_40k' # trajsimi test
    trajdata_timestamp_offset = 0
    dataset_path =  '' # root_dir + '/data/' + dataset_prefix
    
    dataset_lon2Euc_unit = 0.0 # dataset_SanFrancisco_lon2Euc_unit
    dataset_lat2Euc_unit = 0.0 # dataset_SanFrancisco_lat2Euc_unit


    #=============== SARN ===============
    sarn_epochs = 5
    sarn_batch_size = 128
    sarn_optimizer = 'Adam'
    sarn_learning_rated_adjusted = True
    sarn_learning_rate = 0.005
    sarn_learning_momentum = 0.999
    sarn_learning_weight_decay = 0.0001 #1e-4
    sarn_training_bad_patience = 20 # epoch

    sarn_seg_length_unit = 5
    sarn_seg_radian_unit = 0.174533 # 10 degree
    sarn_seg_feat_wayidcode_dim = 32
    sarn_seg_feat_segidcode_dim = 64 
    sarn_seg_feat_highwaycode_dim = 16
    sarn_seg_feat_lengthcode_dim = 16
    sarn_seg_feat_radiancode_dim = 16
    sarn_seg_feat_lonlatcode_dim = 32

    sarn_seg_feat_dim = 176 
    sarn_embedding_dim = 128
    sarn_out_dim = 32

    sarn_moco_total_queue_size = 1000
    sarn_moco_multi_queue_cellsidelen = 0 # will be set later
    sarn_moco_each_queue_size = 0 # each single queue capacity, calculated in osm_loader.py
    sarn_moco_loss_local_weight = 0.4
    sarn_moco_loss_global_weight = 1 - sarn_moco_loss_local_weight
    sarn_moco_temperature = 0.05

    sarn_break_edge_prob = 0.2
    sarn_break_edge_topo_prob = 0.4
    sarn_break_edge_spatial_prob = 0.4
    sarn_seg_weight_distance_thres = 200
    sarn_seg_weight_radian_delta_thres = 0.7853981634 / 2 # 90 / 2 = 45 degree
    sarn_seg_weight_radian_epsilon = 1e-5


    #================================================
    #=============== Downstream tasks ===============
    task_pretrained_model = False # SARN
    task_encoder_model = 'SARN' # SARN
    task_name = 'classify' # classify, trajsimi, locpred
    task_repeats = 1
    task_encoder_mode = ''  # {'SARN':'dump'}
    task_finetune_lr_rescale = 0.2 

    #=============== classifier ===============
    fcnclassifier_classify_colname = 'maxspeed' # maxspeed
    fcnclassifier_epoch = 2000
    fcnclassifier_batch_size = 2048
    fcnclassifier_learning_rate = 0.01 # SARN:0.01
    fcnclassifier_learning_weight_decay = 0.0001 # SARN: 0.0001
    fcnclassifier_training_bad_patience = 100 # epoch
    fcnclassifier_nhidden = 32

    #=============== traj similarity measurement ===============
    trajsimi_epoch = 80
    trajsimi_batch_size = 256 # was 256
    trajsimi_learning_rated_adjusted = True
    trajsimi_learning_rate = 0.0002 # bactch256:0.0002
    trajsimi_learning_weight_decay = 0.0001 # 0.0001
    trajsimi_training_bad_patience = 10 # epoch
    trajsimi_rnn_hidden_dim = 1024

    #=============== spd, following RNE's exp setting ===============
    spd_epoch = 100
    spd_batch_size = 4096 # 4096
    spd_learning_rated_adjusted = True
    spd_learning_rate = 0.05 # batch4096:0.05
    spd_learning_weight_decay = 0.00001
    spd_training_bad_patience = 20 # epoch

    spd_rne_dataset_sampling_rate = 0.02
    spd_rne_dataset_train_partition = 0.001 # 0.002
    spd_rne_dataset_test_partition = 0.00001 # 0.00002


    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_value_updates()


    # remember to call this method explicitly after the class declaration!
    @classmethod
    def post_value_updates(cls):
        if 'CD' == cls.dataset:
            cls.dataset_prefix = 'OSM_Chengdu_2thRings_raw'
            cls.trajsimi_prefix = 'didi_len60_10k_mm_from_30k' # trajsimi test
            cls.dataset_lon2Euc_unit = 0.00001045478306 # 1 meter
            cls.dataset_lat2Euc_unit = 0.000008992805755  
            cls.trajdata_timestamp_offset = 0
            cls.spd_max_spd = 20574
            if cls.sarn_moco_multi_queue_cellsidelen == 0: # not given by program parameteres
                cls.sarn_moco_multi_queue_cellsidelen = 1200
        elif 'BJ' == cls.dataset:
            cls.dataset_prefix = 'OSM_Beijing_2thRings_raw'
            cls.trajsimi_prefix = 'tdrive_len60_10k_mm_from_48k' # trajsimi test
            cls.dataset_lon2Euc_unit = 0.00001172332943
            cls.dataset_lat2Euc_unit = 0.00000899280575
            cls.trajdata_timestamp_offset = 1201824000
            cls.spd_max_spd = 19044
            if cls.sarn_moco_multi_queue_cellsidelen == 0:
                cls.sarn_moco_multi_queue_cellsidelen = 1200
        elif 'SF' == cls.dataset:
            cls.dataset_prefix = 'OSM_SanFrancisco_downtown_raw'
            cls.trajsimi_prefix = 'sfcab_len60_10k_mm_from_40k' # trajsimi test
            cls.dataset_lon2Euc_unit = 0.00001137656428
            cls.dataset_lat2Euc_unit = 0.000008992805755
            cls.trajdata_timestamp_offset = 1211018400
            cls.spd_max_spd = 10133
            if cls.sarn_moco_multi_queue_cellsidelen == 0:
                cls.sarn_moco_multi_queue_cellsidelen = 600
        elif 'SFS' == cls.dataset:
            cls.dataset_prefix = 'OSM_SanFrancisco_downtownS_raw'
            cls.trajsimi_prefix = 'sfcab_S_len60_10k_mm_from_40k' # trajsimi test
            cls.dataset_lon2Euc_unit = 0.00001137656428
            cls.dataset_lat2Euc_unit = 0.000008992805755
            cls.trajdata_timestamp_offset = 1211018400
            cls.spd_max_spd = 10133 
            if cls.sarn_moco_multi_queue_cellsidelen == 0:
                cls.sarn_moco_multi_queue_cellsidelen = 600
        elif 'SFL' == cls.dataset:
            cls.dataset_prefix = 'OSM_SanFrancisco_downtownL_raw'
            cls.trajsimi_prefix = 'sfcab_L_len60_10k_mm_from_60k' # trajsimi test
            cls.dataset_lon2Euc_unit = 0.00001137656428
            cls.dataset_lat2Euc_unit = 0.000008992805755
            cls.trajdata_timestamp_offset = 1211018400
            cls.spd_max_spd = 10133
            if cls.sarn_moco_multi_queue_cellsidelen == 0:
                cls.sarn_moco_multi_queue_cellsidelen = 600

        cls.dataset_path = cls.root_dir + '/data/' + cls.dataset_prefix

        cls.task_encoder_mode = {'SARN':'dump', \
                                'SARN_ft':'finetune'}[cls.task_encoder_model]

        if cls.task_encoder_model == 'SARN_ft':
            if cls.task_name == 'classify':
                cls.task_finetune_lr_rescale = 0.5 #0.05
            elif cls.task_name == 'trajsimi':
                cls.task_finetune_lr_rescale = 0.5 # 0.2
            elif cls.task_name == 'spd':
                cls.task_finetune_lr_rescale = 0.1 # 0.1

        cls.sarn_moco_loss_global_weight = 1 - cls.sarn_moco_loss_local_weight

        set_seed(cls.seed)
    

    @classmethod
    def contain(cls, key): # __contains__(self, key)
        dic = cls.__dict__
        dic = dict(filter( \
                    lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                    dic.items() \
                        ))
        return key in dic


    @classmethod
    def to_str(cls): # __str__, self
        dic = cls.__dict__.copy()
        lst = list(filter( \
                    lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                    dic.items() \
                        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
