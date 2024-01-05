# project:LSTM for ccc #
# using UTF-8 #
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pickle import FALSE

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# from tqdm.notebook import tqdm 
# as tqdm
import time
import random
import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from torch.nn.utils import clip_grad_norm_
from torch.autograd import gradcheck
from torch.utils.data import Subset
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
# from models import *
# from models.lstm import LstmFusionEncoder
# from models.att_model import LstmCrossAttention
# from models.lstm_attention import LstmLwfAttn
# from models.lstm_crosslwfatten import LstmLwfCrossAttn
from models.step_lstm import StepALstm
# from models.step_lstm_ori import StepALstm
# from utils.config import DefaultConfig
# from utils.config import AttConfig
# from utils.lstm_att_config import BiLstmAttentionConfig
# from utils.lstm_lwfcrossatt_config import BiLstmCrossLwfAttentionConfig
from utils.stepla_config import StepLAConfig
from utils.dataset import CSSDataset, collate_fn
from utils.loss import FocalLoss
from utils.loss import *
from utils.utils import *

matplotlib.use('agg')

# k fold validation for value different model
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def k_fold_train(configs,device_id):
    train_data = CSSDataset(configs, preprocess=False, split_type='train')
    test_data = CSSDataset(configs, preprocess=False, split_type='test')
    
    kf = KFold(n_splits=configs.k_fold, shuffle=True)
    # fold loop
    log = []
    models = []
    model_global_cm = np.zeros((5,5))
    writer = SummaryWriter(log_dir='./py_code/LSTM-ccc/runs')  # replace log_dir by yours
    # print()
    for fold_id, (train_index, val_index) in enumerate(kf.split(train_data)):  # k组索引 = for i in k
        # print('fold_id:', fold_id,'train_index:',train_index,'val_inddex:',val_index)
        # continue
        # only train use one fold
        # if fold_id == 6:
        #     continue
        # if fold_id > 0:  
        #      break
        #     continue
        # start = time.time()
        subset_train = Subset(train_data, train_index)
        subset_val = Subset(train_data, val_index)
        train_loader = DataLoader(subset_train, batch_size=configs.train_batch_size, 
                                  num_workers=configs.num_workers, collate_fn=collate_fn, worker_init_fn=seed_worker,
                                  shuffle=True, pin_memory=True, drop_last=False) 
        # print(train_loader.batch_sampler.sampler)
        # continue
        val_loader = DataLoader(subset_val, batch_size=configs.valid_batch_size,
                                num_workers=configs.num_workers, collate_fn=collate_fn, worker_init_fn=seed_worker,
                                shuffle=True, pin_memory=True, drop_last=False)
        # reset model in every fold
        model = StepALstm(configs).cuda(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id)
        # if use multi GPU, the input will be transported from packed-squence to tuple
        # train
        epoch_records, best_epoch_cm, best_model_name_i = train(configs, fold_id, model, train_loader, val_loader,device_id)
        # log
        log.append(epoch_records)  
        models.append(best_model_name_i)
        # plot loss
        tm = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        plot_loss(epoch_records['train_loss'], epoch_records['val_loss'], configs.output_dir, fold_id, configs.model_name,tm)
        model_global_cm += best_epoch_cm
    # all fold end
    tm = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    quota_path = os.path.join(configs.output_dir, configs.model_name + tm + 'quota.csv')
    save_log(log, os.path.join(configs.log_dir, configs.model_name + tm + 'fold_log.csv'), quota_path)
    # calculate this model's global quato
    save_cm(model_global_cm, name=os.path.join(configs.log_dir, configs.model_name + tm + 'cm.csv'))
    model_f1, model_acc, model_prec, model_rec = build_quota(model_global_cm)
    model_cross_val_log = model_prec + model_rec
    model_cross_val_log.append(model_acc)
    model_cross_val_log.append(model_f1)
    # log = np.reshape(np.array(model_cross_val_log), (1, 18))
    log = np.reshape(np.array(model_cross_val_log), (1, 12))
    # name_list = [
    #     'prec_0', 'prec_1', 'prec_2', 'prec_3', 'prec_4', 'prec_5',
    #     'prec_6', 'prec_7','rec_0', 'rec_1', 'rec_2', 'rec_3',
    #     'rec_4', 'rec_5', 'rec_6', 'rec_7', 'model_acc', 'model_f1']  #[global quato]
    name_list = [
        'prec_1', 'prec_2', 'prec_3', 'prec_4', 'prec_5',
        'rec_1', 'rec_2', 'rec_3','rec_4', 'rec_5', 'model_acc', 'model_f1']  #[global quato]
    log = pd.DataFrame(data=log, columns=name_list)
    log_path = os.path.join(configs.output_dir, configs.model_name + tm + '.csv') 
    
    log.to_csv(log_path, index=False)

# train in one fold
def train(configs, i, model, train_loader, val_loader,device_id):
    """
    return:epoch record: a dict includes losses and acc、 f1 of all epoch in one fold
    """
    model.train()
    epoch_records = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': [], 'best_model_name': [],
                     'train_f1': [], 'train_acc': [], 'train_prec': [], 'val_prec': [], 'train_rec': [],
                     'val_rec': [], 'lr': []}
    num_batchs = len(train_loader)
    patience = 0
    model_score = 0
    model_val_cm = np.zeros((5,5))
    new_config = {'best_f1': 0}
    configs.parse(new_config)
    # w = torch.from_numpy(np.array([1,1,1,1,1])).float()
    w = torch.from_numpy(np.array([10,3,1,1,10])).float()
    crossentropyweight = 1  # if use weight
    criterion = nn.CrossEntropyLoss(weight=w, reduction='none').cuda(device=device_id[0])
    # criterion = FocalLoss(weight=w, gamma=configs.gama, sampling='none').cuda(device=device_id[0])
    lr = configs.lr
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=configs.weight_decay)  # 优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.epoch, eta_min=0.001)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=25,T_mult=2, eta_min=0.003)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100],gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.9,patience=5,min_lr=0.00001)
    best_model_name_i = 0
    fold_loop = tqdm(total=configs.epoch, position=0)
    for epoch in range(configs.epoch):
        # train loop 
        e_loss = 0
        fold_score = 0
        tr_cm = np.zeros((configs.num_class, configs.num_class))
        fold_loop.set_description(f'fold:[{i + 1}/{configs.k_fold}]')
        # train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for batch_idx, (inputs1, inputs2, inputs3, targets, total_length) in enumerate(train_loader):
            # print(inputs.max())
            # data
            # inputs1.sort(key=lambda x: len(x), reverse=True)
            # inputs2.sort(key=lambda x: len(x), reverse=True)
            # inputs3.sort(key=lambda x: len(x), reverse=True)
            # targets.sort(key=lambda x: len(x), reverse=True)
            # ******
            seq_len = [s.size(0) for s in inputs1] # real length of data list
            # total_length = seq_len[0]
            seq_len = torch.tensor(seq_len).cuda(device_id[0], non_blocking=True)
            inputs1 = pad_sequence(inputs1, batch_first=True, padding_value=configs.pad_value)  # tensor:{batch-size,length,dim}
            inputs2 = pad_sequence(inputs2, batch_first=True, padding_value=configs.pad_value)
            inputs3 = pad_sequence(inputs3, batch_first=True, padding_value=configs.pad_value)
            targets = pad_sequence(targets, batch_first=True, padding_value=configs.pad_value_t)
            # print(targets.device) # : cpu
            # loss_mask =(targets!=configs.pad_value_t).to(configs.device)
            # ******
            # inputs1 = inputs1.float().to(configs.device)
            # inputs2 = inputs2.float().to(configs.device)
            # inputs3 = inputs3.float().to(configs.device)
            # targets = targets.long().to(configs.device)

            inputs1 = inputs1.float().cuda(device_id[0], non_blocking=True)  # float32
            inputs2 = inputs2.float().cuda(device_id[0], non_blocking=True)
            inputs3 = inputs3.float().cuda(device_id[0], non_blocking=True)
            targets = targets.long().cuda(device_id[0], non_blocking=True)
            loss_mask =(targets!=configs.pad_value_t).cuda(device_id[0], non_blocking=True)
            # train model
            # *****
            input0 = (inputs1, inputs2, inputs3, seq_len)
            # input0 = (inputs1, inputs2, inputs3)
            #            
            outputs = model(input0, total_length)
            # print(outputs)
            # quota for every batch
            losses, cm = cal_quota(outputs, targets, criterion,seq_len, loss_mask, w,
                                   crossentropyweight,'train',configs.check_point)
            # print(losses.item())
            losses = (losses-configs.b).abs()+configs.b   # average batch loss
            # l1 regularizier
            # l1loss = 0
            # for params in model.paramaters():
            #     l1loss += torch.sum(torch.abs(params))
            # losses = losses + configs.pl1 * l1loss
            # # 
            optimizer.zero_grad()
            losses.backward()
            # gradient clip
            # clip_grad_norm_(model.parameters(), max_norm=configs.maxnorm, norm_type=2)
            optimizer.step()
            
            tr_cm += cm
            e_loss += losses.item()  # sum of one epoch's loss
            # train_loop.update(1)
            # train_loop.set_postfix(train_loss=e_loss/(batch_idx+1))
            # train_loop.update(10)
        # after train loop (n iteration)
        
        tr_f1, tr_acc, tr_prec, tr_rec = build_quota(tr_cm)  # one epoch's quota:[float{1,} float list{1,9} list]
        # print(tr_acc)
        e_loss = e_loss / num_batchs 
  
        epoch_records['train_loss'].append(e_loss)
        epoch_records['train_f1'].append(tr_f1)
        epoch_records['train_acc'].append(tr_acc)  # shape:{epoch,1}
        epoch_records['train_prec'].append(tr_prec)  # shape:{epoch,9}
        epoch_records['train_rec'].append(tr_rec)
        

        # val model after one epoch, could change when to val
        # val_f1, val_acc, val_prec, val_rec is [float{1,} float list{1,9} list float]
        val_cm, val_loss = val(configs, model, val_loader, criterion,device_id, w, crossentropyweight)
        val_f1, val_acc, val_prec, val_rec = build_quota(val_cm)  # [float{1,} float list{1,9} list]

        epoch_model_name_i = save_model(configs, model, configs.model_name + '_' + str(i))
        epoch_cm = val_cm  # one epoch cm 
        # print(val_acc)
        # if val_f1 > configs.best_f1:
        #     best_model_name_i = save_model(configs, model, configs.model_name + '_' + str(i))
        #     new_config = {'best_f1': val_f1}
        #     configs.parse(new_config)
        #     best_epoch_cm = val_cm
        #     patience = 0
        # else:
        #     patience += 1   
        scheduler.step(val_loss) # update lr every epoch

        epoch_records['lr'].append(optimizer.param_groups[0]['lr'])
        epoch_records['val_loss'].append(val_loss)
        epoch_records['val_f1'].append(val_f1)
        epoch_records['val_acc'].append(val_acc)  # shape:{epoch,1}
        epoch_records['val_prec'].append(val_prec)  # shape:{epoch,9}
        epoch_records['val_rec'].append(val_rec)
        epoch_records['best_model_name'] = best_model_name_i  # str 
        
        fold_loop.update(1)
        fold_loop.set_postfix(train_loss=e_loss, val_acc=val_acc, lr=optimizer.param_groups[0]['lr'])
    # one fold end
    fold_loop.close()   
    # print(type(epoch_records))
    # print(patience)
    return epoch_records, epoch_cm, epoch_model_name_i


def val(configs, model, val_loader, criterion,device_id, w, crossentropyweight):
    model.eval()
    avg_loss = 0
    length = len(val_loader)
    # print(length)
    v_cm = np.zeros((configs.num_class, configs.num_class))
    for ii, (inputs1, inputs2, inputs3, targets, total_length) in enumerate(val_loader):  # ii =[0, val_batch]

        # ******
        seq_len = [s.size(0) for s in inputs1] # real length of data list
        seq_len = torch.tensor(seq_len, dtype=torch.int).cuda(device_id[0], non_blocking=True)
        # total_length = seq_len[0]
        inputs1 = pad_sequence(inputs1, batch_first=True)  # tensor:{batch-size,length,dim}
        inputs2 = pad_sequence(inputs2, batch_first=True)
        inputs3 = pad_sequence(inputs3, batch_first=True)
        targets = pad_sequence(targets, batch_first=True)
        # ******
        # print()
        # inputs1 = inputs1.float().to(configs.device)
        # inputs2 = inputs2.float().to(configs.device)
        # inputs3 = inputs3.float().to(configs.device)
        # targets = targets.long().to(configs.device)

        inputs1 = inputs1.float().cuda(device_id[0], non_blocking=True)
        inputs2 = inputs2.float().cuda(device_id[0], non_blocking=True)
        inputs3 = inputs3.float().cuda(device_id[0], non_blocking=True)
        targets = targets.long().cuda(device_id[0], non_blocking=True)
        # loss_mask =(targets!=configs.pad_value_t).to(configs.device)
        loss_mask =(targets!=configs.pad_value_t).cuda(device_id[0], non_blocking=True)
        # input0 = (inputs1, inputs2, inputs3)
        input0 = (inputs1, inputs2, inputs3, seq_len)
        outputs = model(input0, total_length)
        # print('val', outputs)
        losses, cm_ii = cal_quota(outputs, targets, criterion,seq_len, loss_mask, w, crossentropyweight,'train',configs.check_point)
        avg_loss += losses.item() # sum of batch loss = 
        v_cm += cm_ii

    model.train()
    return v_cm, avg_loss / length


def test(configs, model, test_loader):
    model.eval()
    num_batchs = len(test_loader)
    test_cm = np.zeros((configs.num_class, configs.num_class))
    # no use params
    criterion = 'none'
    w = 0
    crossentropyweight = 0

    for batch_idx, (inputs1, inputs2, inputs3, targets, total_length) in enumerate(test_loader):
        with torch.no_grad():

            # ******
            seq_len = [s.size(0) for s in inputs1] # real length of data list
            # total_length = seq_len[0]
            seq_len = torch.tensor(seq_len).cuda(device_id[0], non_blocking=True)
            inputs1 = pad_sequence(inputs1, batch_first=True, padding_value=configs.pad_value)  # tensor:{batch-size,length,dim}
            inputs2 = pad_sequence(inputs2, batch_first=True, padding_value=configs.pad_value)
            inputs3 = pad_sequence(inputs3, batch_first=True, padding_value=configs.pad_value)
            targets = pad_sequence(targets, batch_first=True, padding_value=configs.pad_value_t)
            # print(targets.device) # : cpu

            inputs1 = inputs1.float().cuda(device_id[0], non_blocking=True)  # float32
            inputs2 = inputs2.float().cuda(device_id[0], non_blocking=True)
            inputs3 = inputs3.float().cuda(device_id[0], non_blocking=True)
            targets = targets.long().cuda(device_id[0], non_blocking=True)
            loss_mask =(targets!=configs.pad_value_t).cuda(device_id[0], non_blocking=True)
            # train model
            # *****
            input0 = (inputs1, inputs2, inputs3, seq_len)
            # input0 = (inputs1, inputs2, inputs3)
            #            
            outputs = model(input0, total_length)
            losses, cm = cal_quota(outputs, targets, criterion, seq_len, loss_mask, w, crossentropyweight,'test',configs.check_point)
            # losses = 0
            test_cm += cm
    f1, acc, prec, rec = build_quota(test_cm)
    tm = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    save_cm(test_cm, name=os.path.join(configs.log_dir, configs.model_name + tm + 'cm.csv'))
    plot_confusion_matrix(configs, test_cm) 

    model_test_log = prec + rec
    model_test_log.append(acc)
    model_test_log.append(f1)
    # log = np.reshape(np.array(model_cross_val_log), (1, 18))
    log = np.reshape(np.array( model_test_log), (1, 12))
        # name_list = [
        #     'prec_0', 'prec_1', 'prec_2', 'prec_3', 'prec_4', 'prec_5',
        #     'prec_6', 'prec_7','rec_0', 'rec_1', 'rec_2', 'rec_3',
        #     'rec_4', 'rec_5', 'rec_6', 'rec_7', 'model_acc', 'model_f1']  #[global quato]
    name_list = [
        'prec_1', 'prec_2', 'prec_3', 'prec_4', 'prec_5',
        'rec_1', 'rec_2', 'rec_3','rec_4', 'rec_5', 'model_acc', 'model_f1']  #[global quato]
    log = pd.DataFrame(data=log, columns=name_list)
    log_path = os.path.join(configs.output_dir, configs.model_name + tm + 'test.csv')    
    log.to_csv(log_path, index=False)

    model.train()
    return f1, acc, prec, rec, test_cm

# train whole train_dataset
def retrain(configs,device_id):
    # model
    model = StepALstm(configs).cuda(device_id[0])
    model = nn.DataParallel(model, device_ids=device_id)
    # writer 
    writer = SummaryWriter(comment=configs.model_name)
    
    model.train()
    train_data = CSSDataset(configs, preprocess=False, split_type='train')
    train_loader = DataLoader(train_data, batch_size=configs.train_batch_size, 
                              num_workers=configs.num_workers, collate_fn=collate_fn, worker_init_fn=seed_worker,
                              shuffle=True, pin_memory=True, drop_last=False) 
    retrain_epoch_records = {'train_loss': [], 'train_f1': [], 'train_acc': [], 'train_prec': [], 'train_rec': [], 
                            'test_f1': [], 'test_acc': [], 'test_prec': [], 'test_rec': [], 'lr': []}
    num_batchs = len(train_loader)
    # best_test_cm = None

    w = torch.from_numpy(np.array([1,1,1,1,1])).float()
    # w = torch.from_numpy(np.array([4,2,1,1,4])).float()
    # w = torch.from_numpy(np.array([10,3,1,1,10])).float()
    crossentropyweight = 1  # if use weight
    criterion = nn.CrossEntropyLoss(weight=w, reduction='none').cuda(device=device_id[0])
    # criterion = FocalLoss(weight=w, gamma=configs.gama, sampling='none').cuda(device=device_id[0])
    lr = configs.lr
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=configs.weight_decay)  # 优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.epoch, eta_min=0.001)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=25,T_mult=2, eta_min=0.003)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100],gamma=0.5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.9,patience=5,min_lr=0.00001)

    retrain_loop = tqdm(total=configs.retrain_epoch, position=0)
    for epoch in range(configs.retrain_epoch):
        # train loop
        e_loss = 0
        tr_cm = np.zeros((configs.num_class, configs.num_class))
        retrain_loop.set_description(f'retrain')

        for batch_idx, (inputs1, inputs2, inputs3, targets, total_length) in enumerate(train_loader):
            # ******
            seq_len = [s.size(0) for s in inputs1] # real length of data list
            # total_length = seq_len[0]
            seq_len = torch.tensor(seq_len).cuda(device_id[0], non_blocking=True)
            inputs1 = pad_sequence(inputs1, batch_first=True, padding_value=configs.pad_value)  # tensor:{batch-size,length,dim}
            inputs2 = pad_sequence(inputs2, batch_first=True, padding_value=configs.pad_value)
            inputs3 = pad_sequence(inputs3, batch_first=True, padding_value=configs.pad_value)
            targets = pad_sequence(targets, batch_first=True, padding_value=configs.pad_value_t)
            # print(targets.device) # : cpu
            # loss_mask =(targets!=configs.pad_value_t).to(configs.device)
            # ******
            # inputs1 = inputs1.float().to(configs.device)
            # inputs2 = inputs2.float().to(configs.device)
            # inputs3 = inputs3.float().to(configs.device)
            # targets = targets.long().to(configs.device)

            inputs1 = inputs1.float().cuda(device_id[0], non_blocking=True)  # float32
            inputs2 = inputs2.float().cuda(device_id[0], non_blocking=True)
            inputs3 = inputs3.float().cuda(device_id[0], non_blocking=True)
            targets = targets.long().cuda(device_id[0], non_blocking=True)
            loss_mask =(targets!=configs.pad_value_t).cuda(device_id[0], non_blocking=True)
            # train model
            # *****
            input0 = (inputs1, inputs2, inputs3, seq_len)
            # input0 = (inputs1, inputs2, inputs3)
            #            
            outputs = model(input0, total_length)
            # print(outputs)
            # quota for every batch
            losses, cm = cal_quota(outputs, targets, criterion,seq_len, loss_mask, w, crossentropyweight,'train',configs.check_point)
            # flood = (losses-b)
            optimizer.zero_grad()
            losses.backward()
            # gradient clip
            # clip_grad_norm_(model.parameters(), max_norm=configs.maxnorm, norm_type=2)
            optimizer.step()
            tr_cm += cm
            # e_loss += losses
            e_loss += losses.item()  # sum of one epoch's loss
        # after train loop (n iteration)       
        # tr_f1, tr_acc, tr_prec, tr_rec = build_quota(tr_cm)  # one epoch's quota:[float{1,} float list{1,9} list]
        # print(tr_acc)
        e_loss = e_loss / num_batchs 
        model_name = save_model(configs, model, configs.model_name + 'retrain')
        
        tr_f1, tr_acc, tr_prec, tr_rec = build_quota(tr_cm)  # one epoch's quota:[float{1,} float list{1,9} list]
        
        retrain_epoch_records['train_loss'].append(e_loss)
        retrain_epoch_records['train_f1'].append(tr_f1)
        retrain_epoch_records['train_acc'].append(tr_acc)  # shape:{epoch,1}
        retrain_epoch_records['train_prec'].append(tr_prec)  # shape:{epoch,5}
        retrain_epoch_records['train_rec'].append(tr_rec)
        retrain_epoch_records['lr'].append(lr)

        tags = ["train_loss", "accuracy", "learning_rate"]
        writer.add_scalar(tags[0], e_loss, epoch)
        writer.add_scalar(tags[1], tr_acc, epoch)
        writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)


        for name, param in model.named_parameters():
            writer.add_histogram(name+'_grad',param.grad,epoch)
            writer.add_histogram(name,param.grad,epoch)

        
        # test
        # test_data = CSSDataset(configs, preprocess=False, split_type='test')
        # test_loader = DataLoader(test_data, batch_size=configs.test_batch_size,
        #                      num_workers=configs.num_workers, collate_fn=collate_fn,
        #                      shuffle=False, pin_memory=True, drop_last=True)
        # t_f1, t_acc, prec, rec, t_cm = test(configs, model, test_loader, retrain =1)
        # save log
        # retrain_epoch_records['test_loss'].append(t_loss)
        # retrain_epoch_records['test_f1'].append(t_f1)
        # retrain_epoch_records['test_acc'].append(t_acc)  # shape:{epoch,1}
        # retrain_epoch_records['test_prec'].append(prec)  # shape:{epoch,9}
        # retrain_epoch_records['test_rec'].append(rec)

        # if t_f1 > configs.best_f1:
        #     model_name = save_model(configs, model, configs.model_name + 'retrain')
        #     new_config = {'best_f1': t_f1}
        #     best_test_cm = t_cm
        #     configs.parse(new_config)
        #     patience = 0
        # else:
        #     patience += 1
        # # early stop 
        # if patience == 500:
        #     break

        # retrain_loop.update(1)
        # # retrain_loop.set_postfix(train_loss=e_loss, test_f1=t_f1, lr=lr)
        # retrain_loop.set_postfix(train_loss=e_loss, lr=lr)
        retrain_loop.update(1)
        # retrain_loop.set_postfix(train_loss=e_loss, test_f1=t_f1, lr=lr)
        retrain_loop.set_postfix(train_loss=e_loss, lr=lr)
    # all epoch end close tqdm  
    retrain_loop.close()
    writer.close()
    # test
    print("test")
    model = load_model(model_name)
    test_data = CSSDataset(configs, preprocess=False, split_type='test')
    test_loader = DataLoader(test_data, batch_size=configs.test_batch_size,
                             num_workers=configs.num_workers, collate_fn=collate_fn,
                             shuffle=False, pin_memory=True, drop_last=True)
    t_f1, t_acc, prec, rec, t_cm = test(configs, model, test_loader)   
    print('test_f1:{:.6f},test_acc:{:.6f}'.format(t_f1, t_acc))
    print(prec)
    print(rec)           
    # epoch records is a dict includes losses and acc、 f1 of all epoch in one fold
    log =  retrain_epoch_records # dict
    # save log
    tm = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    # log_path = os.path.join(configs.output_dir, configs.model_name + tm + 'retrain.csv')
    # save_log(log, log_path)
    quota_path = os.path.join(configs.output_dir, configs.model_name + tm + 'quota.csv')
    save_log(log, os.path.join(configs.log_dir, configs.model_name + tm + 'retrain_log.csv'), quota_path)
    # return retrain_epoch_records, model_name

def save_model(config, model, name):
    name = name + str(name) + '.pt'  # should delet '+ str(name)'
    model_name = os.path.join(config.output_dir, name)
    torch.save(model, os.path.join(config.output_dir, name))
    return model_name


def load_model(model_name):
    model = torch.load(model_name)
    return model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # torch.cuda.manual_seed_all(0)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.backends.cudnn.deterministic = True 
    # torch.backends.cudnn.enabled=False


if __name__ == '__main__':
    setup_seed(42)
    configs = StepLAConfig()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" 
    device_id = [2,3]
    k_fold_train(configs,device_id)  # it's used to cross validation model!!!remember to check the 'configs.model_name' before run!!!!
    # retrain(configs,device_id)

    # decode_drop_p=[0,0.1,0.2,0.3,0.4,0.5,0.6]
    # for i in range(len(decode_drop_p)):
    #     new_config={'decode_drop_p':decode_drop_p[i]}
    #     configs.parse(new_config)
    #     # print(learn_r[i])
    #     setup_seed(42)
    #     k_fold_train(configs,device_id)

