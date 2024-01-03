# ----coding:utf-8------ #

import datetime
import os
import pandas as pd
import numpy as np
from config import AttConfig

def read_cm(path):
    cm = pd.read_csv(path, header=0, index_col=0)
    cm_i = cm.values
    return cm_i

def build_quota(cm):
    """
    :param cm: confusion matrix as nd array
    :return: quotas
    """
    a, b = cm.shape
    prec = []
    rec = []
    acc = 0
    f1_tmp = []
    assert a == b
    for i in range(a):
        precision = 0 if np.sum(cm[:, i]) == 0 else cm[i, i] / np.sum(cm[:, i])
        recall = 0 if np.sum(cm[i, :]) == 0 else cm[i, i] / np.sum(cm[i, :])
        prec.append(precision)
        rec.append(recall)
        acc += cm[i, i]
    for j in range(len(prec)):
        o = np.array(prec[j]) + np.array(rec[j])
        p = np.array(prec[j]) * np.array(rec[j])
        sp = 0 if o == 0 else 2 * p / o
        f1_tmp.append(sp)
    f1 = np.mean(f1_tmp)
    acc = acc / np.sum(cm)
    return f1, acc, prec, rec


if __name__ == '__main__':
    # configs = DefaultConfig()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
    configs = AttConfig() 
    # configs =  # 记得改
    tm = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    path1 = '/home/zhengjiaying/py_code/LSTM-ccc/checkpoints/701lstmlwfatt/BiLstmAttention2022-07-01-23-29cm.csv'  # 9, 10
    path2 = '/home/zhengjiaying/py_code/LSTM-ccc/checkpoints/701lstmlwfatt/BiLstmAttention2022-07-02-00-17cm.csv'  # 7, 8
    path3 = '/home/zhengjiaying/py_code/LSTM-ccc/checkpoints/701lstmlwfatt/BiLstmAttention2022-07-02-08-49cm.csv'  # 1, 2, 3
    path4 = '/home/zhengjiaying/py_code/LSTM-ccc/checkpoints/701lstmlwfatt/BiLstmAttention2022-07-02-13-53cm.csv'  # 567
    cm1 = read_cm(path1)    #
    cm2 = read_cm(path2)     # 
    cm3 = read_cm(path3)
    cm4 = read_cm(path4)
    model_global_cm = cm1 + cm2 + cm3 + cm4
    model_f1, model_acc, model_prec, model_rec = build_quota(model_global_cm)
    model_cross_val_log = model_prec + model_rec
    model_cross_val_log.append(model_acc)
    model_cross_val_log.append(model_f1)
    log = np.reshape(np.array(model_cross_val_log), (1, 18))
    name_list = [
        'prec_0', 'prec_1', 'prec_2', 'prec_3', 'prec_4', 'prec_5',
        'prec_6', 'prec_7','rec_0', 'rec_1', 'rec_2', 'rec_3',
        'rec_4', 'rec_5', 'rec_6', 'rec_7', 'model_acc', 'model_f1']  #[global quato]
    log = pd.DataFrame(data=log, columns=name_list)
    log_path = os.path.join(configs.output_dir, configs.model_name + tm + 'model_cm.csv') 
    log.to_csv(log_path, index=False)