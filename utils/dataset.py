# --coding='utf-8'-- #
# data ver2
# 三模态数据集（动作识别：NDI\DG\PS 三类数据 9分类任务）
from distutils.command.config import config
import pickle
from scipy.fftpack import fft, ifft
import torch
from torch.utils.data import dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
import os
import numpy as np
import pandas as pd
import math


class CSSDataset(dataset.Dataset):
    """Multi-variate Time-Series Dataset
    Returns:
        [sample, label]
    """

    # 构造函数
    def __init__(self, configs, preprocess, split_type='train'):
        super(CSSDataset, self).__init__()
        """
        :param data_name: name of dataset.pkl 
        :param split_type: train, validation, test
        :param preprocess: weather do the preprocess
        :param n: dft sampling rate
        """
        self.split_type = split_type  # 'train'\'test'\'validation' 无预处理
        self.n = configs.n
        self.preprocess = preprocess
        self.data_name = configs.data_name
        f = open(self.data_name, 'rb+')
        data = pickle.load(f)  # a list of samples
        self.len_s = len(data)
        self.device = configs.device
        # split dataset (could change to k fold strategy)

        if split_type == 'train':
            self.data = data['train']
            # self.len_s = len(data)
        elif split_type == 'test':
            self.data = data['test']

    def frequency_trans(self, data):
        # data: tensor
        # to array
        data = data.numpy()
        t = data.shape[0]
        n_var = data.shape[1]
        w = data.shape[2]
        data_temp = np.reshape(data, ())
        # dim = data.shape[1]
        # new_data: n*dim array
        n = self.n
        new_data = fft(data, n)
        # 取大小、相位
        d = []
        dd = []
        for i in range(t):
            for j in range(n_var):
                for k in range(w):
                    r1 = new_data[j, k].real
                    i1 = new_data[j, k].imag
                    rr = abs(new_data[i, j])
                    ii = math.atan(i1 / r1)
                    d.append([rr])
                    d.append([ii])
            dd.append([d])
        return dd

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_ndi = None
        data_dg = None
        data_fs = None
        label = None

        data = self.data  # train or test data's name list or a
        if self.data_name == '/home/zhengjiaying/py_code/LSTM-ccc/standcccdata2/css_dataset_stand_series.pkl':
        # if self.data_name == '/home/zhengjiaying/py_code/LSTM-ccc/stand_data/css_dataset_stand_series.pkl':
        # if self.data_name == '/home/zhengjiaying/py_code/LSTM-ccc/stand_data/css_dataset_stand_series.pkl':
            # /home/zhengjiaying/py_code/LSTM-ccc/scaled_data/css_dataset_scale_series.pkl
            # /home/zhengjiaying/py_code/LSTM-ccc/stand_data/css_dataset_stand_series.pkl
            # data is a list of  series data's filename
            sample_name = data[item][0]
            # paths
            ndi_path = '/home/zhengjiaying/py_code/LSTM-ccc/standcccdata2/series_datas/datas'
            ndi_path = os.path.join(ndi_path, self.split_type, 'NDI')
            ps_path = '/home/zhengjiaying/py_code/LSTM-ccc/standcccdata2/series_datas/datas'
            ps_path = os.path.join(ps_path, self.split_type, 'PS')
            dg_path = '/home/zhengjiaying/py_code/LSTM-ccc/standcccdata2/series_datas/datas'
            dg_path = os.path.join(dg_path, self.split_type, 'DG')
            label_path = '/home/zhengjiaying/py_code/LSTM-ccc/standcccdata2/series_datas/labels'
            label_path = os.path.join(label_path, self.split_type)

            # get data
            data_ndi = pd.read_csv(os.path.join(ndi_path, sample_name), header=0, index_col=None)
            data_ndi = torch.tensor(data_ndi.values)  # shape:{length, dim}

            data_dg = pd.read_csv(os.path.join(dg_path, sample_name), header=0, index_col=None)
            data_dg = torch.tensor(data_dg.values)

            data_fs = pd.read_csv(os.path.join(ps_path, sample_name), header=0, index_col=None)
            data_fs = torch.tensor(data_fs.values)

            label = pd.read_csv(os.path.join(label_path, sample_name), header=0, index_col=None)
            label = torch.tensor(label.values)

        elif self.data_name == 'css_dataset_window.pkl':
            # window dataset :list[list[tensor,tensor...]]
            # a list of 3 data tensor, shape:{t,h,w} and one label tensor,shape:{t,h,1}
            data_m = data[item].to(self.device)
            data_ndi = data_m[0]
            data_dg = data_m[1]
            data_fs = data_m[2]
            # tensor:{time_steps,}
            label = data_m[4] if self.split_type == 'train' else torch.zeros(data_m[4].size)  # int or tensor

        # 3D tensor:{time_steps, num_variable, window_size}
        if self.preprocess:  # 不推荐
            data_ndi = self.frequency_trans(data_ndi)
            data_dg = self.frequency_trans(data_dg)
            data_fs = self.frequency_trans(data_fs)

        # sample = {"data_ndi": data_ndi, "data_dg": data_dg, "data_fs": data_fs}
        return data_ndi, data_dg, data_fs, label # tuple
        # return data_ndi.type(torch.float32), data_dg.type(torch.float32), data_fs.type(torch.float32), label.type(torch.float32)

def collate_fn(data):
    # data: list :[(data_ndi, data_dg, data_fs, label),(...)]
    # print(len(data))
    ndi_data = []  # (batch_size, length, dim)
    dg_data = []
    fs_data = []
    label_data = []
    
    for each in data:
        ndi_data.append(each[0])
        dg_data.append(each[1])
        fs_data.append(each[2])
        label_data.append(each[3])
        
    
    ndi_data.sort(key=lambda x: len(x), reverse=True)
    dg_data.sort(key=lambda x: len(x), reverse=True)
    fs_data.sort(key=lambda x: len(x), reverse=True)
    label_data.sort(key=lambda x: len(x), reverse=True)
    total_length = len(ndi_data[0])
    # ***********
    # seq_len = [s.size(0) for s in ndi_data] # real length of data list
    # ndi_data = pad_sequence(ndi_data, batch_first=True)  # tensor:{batch-size,length,dim}
    # dg_data = pad_sequence(dg_data, batch_first=True)
    # fs_data = pad_sequence(fs_data, batch_first=True)
    # label_data = pad_sequence(label_data, batch_first=True)

    # ndi_data = pack_padded_sequence(ndi_data, seq_len, batch_first=True)
    # dg_data = pack_padded_sequence(dg_data, seq_len, batch_first=True)
    # fs_data = pack_padded_sequence(fs_data, seq_len, batch_first=True)
    # ***********
    # label_data = pack_padded_sequence(label_data, seq_len, batch_first=True)
    # # print('dataset')
    # print(ndi_data[0].size(),ndi_data[1].size())
    return ndi_data, dg_data, fs_data, label_data, total_length,   #  tuple:(packedsepuence:{tensor,length,...,...},...,list)