# author J-Y Zheng
# coding utf-8

import torch
import warnings

class StepLAConfig:
    # root and name
    model_name = 'BMASNModel'  # model name
    data_root = '.\\dataset.py'  # path of data 
    log_dir = '.\\logging'
    data_name = r'/home/zhengjiaying/py_code/LSTM-ccc/standcccdata2/css_dataset_stand_series.pkl'
    # output_dir = '.\\checkpoints'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device_id = [2]
    # check_point = '.\\checkpoints'
    check_point = '\\ndsorit805'
    output_dir = '\\ndsorit805'  
    log_dir = '\\ndsorit805'


    # basic param
    k_fold = 10   
    n_modality = 3
    num_class = 5
    num_workers = 1
    train_batch_size = 2
    test_batch_size = 2
    valid_batch_size = 2  
    best_f1 = 10
    pad_value = 0 
    pad_value_t = 9  # for label pad value must bigger than 7!!

    # hyperparam
    # fft
    n = 5
    b = 0  # flooding
    gama = 2
    # pylstm
    # !! 参数多！！谨慎确认
    epoch = 100  # 可以每隔50epoch再验证，降采样曲线也许能平稳一些
    retrain_epoch = 80
    input_c_ndi = [9,17,33]
    hidden_c_ndi = [4,8,24]  # 48
    input_c_dg = [6,14,30]
    hidden_c_dg = [4,8,20]  # 40
    input_c_ps = [2,10,26]
    hidden_c_ps = [4,8,18]  # 36
    
    # ablation:
    ablation = 0
    ab_n_layer = 3
    ab_dropout = 0
    ab_input_c_ndi = 9
    ab_hidden_c_ndi = 24  # 48
    ab_input_c_dg = 6
    ab_hidden_c_dg = 20  # 40
    ab_input_c_ps = 2
    ab_hidden_c_ps = 18  # 36

    n_layer = 1 
    num_direc = 2
    dropout = 0

    # bi-attention
    # out_dim1 = out_dim0
    cross_num_heads = 4
    ca_indim_ndidg = [48,40,88]  # qkv
    ca_outdim_ndidg = [24,24,32] 
    ca_indim_ndips = [48,36,84]  # qkv
    ca_outdim_ndips = [24,24,32] 
    ca_indim_dgps = [36,40,76]  # qkv
    ca_outdim_dgps = [20,20,32]    
    cross_layer = 1
    final_cross_layer = 1

    ln_dim = 32
    fc_outdim = 32
    outdim_fc =32

    # single lstm
    if_res = 1
    res_input_c=[48,40,36]
    res_hidden_c=[24,20,18]
    res_n_layer=1
    res_num_direc=2

    # final-attention
    final_cross = False
    fca_indim_ps=[96,124,220]
    fca_outdim_ps=[96,96,64]
    fcross_num_heads=4
    fln_dim=64
    ffc_outdim=64
    foutdim_fc=64

    final_indim_self=220
    final_outdim_self=128
    final_self_num_heads=4
    final_dim=128
    final_outdim_fc=128
    
    # decoder  {B,N,C}
    decode_dense_indim=[[64,32,16],[128,64,32]]
    decode_dense_outdim=[[32,16,5],[64,32,5]]
    decode_drop_p=0.2

    # barrierG  
    out_augutation=False
    bg_k=[3,3,3]
    bg_cc_inc=[[128,64,32],[64,32,16]]
    bg_cc_outc=[[64,32,1],[32,16,1]] 
    bg_cc_pad=[1,2,4]
    bg_cc_stride=[1,1,1]
    bg_cc_dilation=[1,2,4]

    # barrierP
    bp_kernel=8
    bp_alpha=0.2

    dropout = 0.4
    # loss and adam
    lr = 0.009  # learning rate
    lr_decay = 0.9  # when loss increase, lr=lr*lr_decay
    weight_decay = 1e-5  # loss function cross entropy  (equas to pl2)
    pl1 = 0.001


    #  use dict(kwargs) to update config
    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("warning:params has no attribute %s" % k)
            setattr(self, k, v)