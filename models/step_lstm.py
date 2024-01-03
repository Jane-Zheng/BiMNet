# --coding='utf-8'-- #
# author:J-Y Zheng

from cmath import inf
from turtle import forward
import torch
import math
import os
import numpy as np
import pandas as pd
import torch.nn as nn
# from modules.BarrierG import *
# from modules import BarrierPool
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, indim_self,outdim_self,self_num_heads,dim,outdim_fc):
        super().__init__()
        # Q, K, V ，
        # [220->128]
        self.q = nn.Linear(indim_self, outdim_self,bias=False)
        self.k = nn.Linear(indim_self, outdim_self,bias=False)
        self.v = nn.Linear(indim_self, outdim_self,bias=False)
        self.num_heads = self_num_heads
        self.ln = nn.LayerNorm(dim, 1e-5)
        self.fc = nn.Linear(outdim_self, outdim_fc,bias=False)
        self.out_c = outdim_self
    # ....forget time code 
    def forward(self, x1,x2):
        # attention = sofftmax(GRN(a)) 
        x = torch.cat((x1,x2),dim=2)
        B, N, C = x.shape
        # print(x.size())
        # pe = self.time_code(x)
        # multihead 
        # num_head * '-1' = outdim 
        # print(x.shape)  #[* ,*,158]
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # print(v.size())
        # mask_v = (v!=0)
        # dot product get attention score
        attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
        # print(attn.size())
        mask = (attn!=0)
        attn = attn.masked_fill_(mask==0, -1e20)
        attn = attn.softmax(dim=-1)
        attn = attn * mask
        
        # 乘上attention score并输出(点积transformer)
        v1 = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, self.out_c)
        v = v.permute(0, 2, 1, 3).reshape(B, N, self.out_c)
        out = self.ln(v+v1)  # add and norm
        # v2 = self.fc(v1)  # feed forward
        # out = self.ln(v2 + v1)  # add and norm
        return out

class Soft_attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.indim_softa, config.outdim_softa,bias=True)
        self.activate = nn.ReLU()

    def forward(self, input):
        # 并行化矩阵运算
        attention_score = self.dense(input)
        attn = self.activate(attention_score)
        attn = attn.softmax(dim=-1)
        # hadamard product
        out = torch.mul(attn,input)
        return out


class CrossModeAttention(nn.Module):
    def __init__(self, indim_self, outdim_self,cross_num_heads,ln_dim,fc_outdim_self,outdim_fc, cross_layer):
        super().__init__()
        self.q = nn.Linear(indim_self[0], outdim_self[0],bias=False)
        self.k = nn.Linear(indim_self[1], outdim_self[1],bias=False)
        self.v = nn.Linear(indim_self[2], outdim_self[2],bias=False)
        self.num_heads = cross_num_heads
        self.ln = nn.LayerNorm(ln_dim, 1e-5)
        self.fc = nn.Linear(fc_outdim_self, outdim_fc,bias=False)
        self.out_c = outdim_self[2]
        self.cross_layer = cross_layer

    def forward(self,x1,x2):
        '''
        input is two kinds of sensors' data feature. {B,L,D}
        '''
        x = torch.cat((x1,x2),dim=2)
        B,N,C = x1.shape  # x1 and x2 have same shape
        Q = self.q(x1).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        K = self.k(x2).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        V = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # print(V.size())
        # B,N,C = V.shape
        # mask_v = (V!=0)
        for i in range(self.cross_layer):
            attn = Q @ K.transpose(2, 3) * (x1.shape[-1] ** -0.5)
            mask = (attn!=0)
            attn = attn.masked_fill_(mask==0, -1e20)
            attn = attn.softmax(dim=-1)
            attn = attn*mask
            # print(attn.size())
            V1 = (attn @ V).permute(0, 2, 1, 3).reshape(B, N, self.out_c)
            V = V.permute(0, 2, 1, 3).reshape(B, N, self.out_c)
            output = self.ln(V1+V)  # output feature
            x = output
            V = x.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # v2 = self.fc(v1)  
        # output = self.ln(v2 + v1)  # without feed forward network
        return output


class PyLstmBlock(nn.Module):
    def __init__(self, input_c,hidden_c,n_layer,dropout,device):
        super().__init__()
        # self.gru1 = nn.GRU()
        self.lstm1 = nn.GRU(input_c[0], hidden_c[0], n_layer, bidirectional=True, 
                             dropout=dropout, batch_first=True)
        # for name, param in self.lstm1.named_parameters():
        #     nn.init.uniform_(param,-0.1,0.1)
        self.lstm2 = nn.GRU(input_c[1], hidden_c[1], n_layer, bidirectional=True, 
                             dropout=dropout, batch_first=True)
        # for name, param in self.lstm2.named_parameters():
        #     nn.init.uniform_(param,-0.1,0.1)                     

        self.lstm3 = nn.GRU(input_c[2], hidden_c[2], n_layer, bidirectional=True, 
                             dropout=dropout, batch_first=True)
        # for name, param in self.lstm3.named_parameters():
        #     nn.init.uniform_(param,-0.1,0.1)

        self.ln1 = nn.LayerNorm(input_c[1], 1e-5)
        self.ln2 = nn.LayerNorm(input_c[2], 1e-5)
        self.num_layers = n_layer
        self.num_direc = 2
        self.hidden_channel = hidden_c
        self.device = device
        # self.device_id = device_id
        
    
    def forward(self, input, length, total_length):
        '''
        input: packedsequence
        length:(tensor)
        '''
        #  B,L,D = input.size() batch size
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
        B = len(length)
        h_ndi_0 = torch.zeros(self.num_layers * self.num_direc, B, self.hidden_channel[0], device=self.device) #.cuda(self.device_id)
        # c_ndi_0 = torch.zeros(self.num_layers * self.num_direc, B, self.hidden_channel[0], device=self.device)
        # x, (h_n0, c_n0)= self.lstm1(input,(h_ndi_0, c_ndi_0))
        x, h_n0 = self.lstm1(input, h_ndi_0)   # gru only need h
        x = pad_packed_sequence(x, batch_first=True, total_length=total_length)
        input = pad_packed_sequence(input, batch_first=True, total_length=total_length)
        input = torch.cat((x[0],input[0]),2)
        # input = self.ln1(input)
        input = pack_padded_sequence(input, length, batch_first=True)
        
        delta_h = self.hidden_channel[1]-self.hidden_channel[0]
        h_ndi_1 = torch.zeros(self.num_layers * self.num_direc, B, delta_h, device=self.device) #.cuda(self.device_id)
        # c_ndi_1 = torch.zeros(self.num_layers * self.num_direc, B, delta_h, device=self.device)
        h_ndi_1 = torch.cat((h_n0, h_ndi_1),2)
        # c_ndi_1 = torch.cat((c_n0, c_ndi_1),2)
        # x, (h_n1, c_n1)= self.lstm2(input,(h_ndi_1, c_ndi_1))

        x, h_n1 = self.lstm2(input, h_ndi_1)
        x = pad_packed_sequence(x, batch_first=True, total_length=total_length)  # tuple
        input = pad_packed_sequence(input, batch_first=True, total_length=total_length)
        input = torch.cat((x[0],input[0]),2)
        # input = self.ln2(input)
        input = pack_padded_sequence(input, length, batch_first=True) 
        
        delta_h = self.hidden_channel[2]-self.hidden_channel[1]
        h_ndi_2 = torch.zeros(self.num_layers * self.num_direc, B, delta_h, device=self.device) #.cuda(self.device_id)
        # c_ndi_2 = torch.zeros(self.num_layers * self.num_direc, B, delta_h, device=self.device)
        h_ndi_2 = torch.cat((h_n1, h_ndi_2),2)
        # c_ndi_2 = torch.cat((c_n1, c_ndi_2),2)
        # out, (h_n2, c_n2)= self.lstm3(input,(h_ndi_2, c_ndi_2))  
        
        out, h_n2 = self.lstm3(input, h_ndi_2) 
        # return out, (h_n2, c_n2)  # packed sequence

        return out, h_n2

class ResLstm(nn.Module):
    def __init__(self,res_input_c, res_hidden_c, res_n_layer,res_num_direc, if_res, device):
        super().__init__()
        self.lstm1 = nn.GRU(res_input_c, res_hidden_c, res_n_layer, bidirectional=True, 
                             dropout=0, batch_first=True)
        self.ln = nn.LayerNorm(res_hidden_c*res_num_direc, 1e-5)
        self.num_layers = res_n_layer
        self.num_direc = res_num_direc
        self.hidden_channel = res_hidden_c
        self.if_res = if_res
        self.device = device
        # self.device_id = device_id
    
    def forward(self, input, length, total_length):
        '''
        input:padded sequence
        '''
        B = len(length)
        self.lstm1.flatten_parameters()
        h_ndi_0 = torch.zeros(self.num_layers * self.num_direc, B, self.hidden_channel, device=self.device) #.cuda(self.device_id)
        # c_ndi_0 = torch.zeros(self.num_layers * self.num_direc, B, self.hidden_channel, device=self.device)

        # ndi_u, (hn,cn) = self.lstm1(input,(h_ndi_0, c_ndi_0))
        ndi_u, hn = self.lstm1(input, h_ndi_0)
        ndi_u = pad_packed_sequence(ndi_u, batch_first=True, total_length=total_length) # tuple
        ndi_u = ndi_u[0]
        input = pad_packed_sequence(input, batch_first=True, total_length=total_length) # tuple
        if self.if_res:
            ndi_u = self.ln(ndi_u + input[0]) # tensor                
        # return ndi_u, (hn, cn)
        return ndi_u, hn

class Decoder(nn.Module):
    def __init__(self, decode_dense_indim, decode_dense_outdim, decode_drop_p):
        super(Decoder,self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(decode_dense_indim[0], decode_dense_outdim[0], bias=False),
            # nn.ReLU(),
            # nn.GELU(),
            nn.Hardswish(),
            nn.Dropout(decode_drop_p, inplace=False),
            nn.Linear(decode_dense_indim[1], decode_dense_outdim[1], bias=False),           
            # nn.ReLU(), 
            # nn.GELU(),
            nn.Hardswish(),
            nn.Dropout(decode_drop_p, inplace=False),   
            nn.Linear(decode_dense_indim[2], decode_dense_outdim[2], bias=False)
        )
        # self.dense1 = nn.linear(decode_dense_indim[0], decode_dense_outdim[0], bias=True)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(decode_drop_p, inplace=False)
        # self.dense2 = nn.linear(decode_dense_indim[1], decode_dense_outdim[1], bias=True)
        # self.relu = nn.ReLU()       
        # self.dense3 = nn.linear(decode_dense_indim[2], decode_dense_outdim[2], bias=True)
       
    def forward(self, feature):
        # f1 = self.relu(self.dense1(feature))
        # f1 = self.dropout(f1)
        # output = self.dense2(f1)
        output = self.layers(feature)
        return output 


class StepALstm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final_cross = config.final_cross
        self.out_aug = config.out_augutation
        self.pad_value = config.pad_value
        self.bg_cc_inc = config.bg_cc_inc[1] if self.final_cross else config.bg_cc_inc[0]
        self.bg_cc_outc = config.bg_cc_outc[1] if self.final_cross else config.bg_cc_outc[0]
        self.decode_dense_indim = config.decode_dense_indim[0] if self.final_cross else config.decode_dense_indim[1]
        self.decode_dense_outdim = config.decode_dense_outdim[0] if self.final_cross else config.decode_dense_outdim[1]
        self.ablation = config.ablation
        self.ab_nlayer = config.ab_n_layer  
        self.device = config.device      
        if config.ablation == 1:
            self.lstmblock1 = nn.GRU(config.ab_input_c_ndi,config.ab_hidden_c_ndi,config.ab_n_layer,bidirectional=True, 
                             dropout=config.ab_dropout, batch_first=True)
            self.lstmblock2 = nn.GRU(config.ab_input_c_dg,config.ab_hidden_c_dg,config.ab_n_layer,bidirectional=True, 
                             dropout=config.ab_dropout, batch_first=True)
            self.lstmblock3 = nn.GRU(config.ab_input_c_ps,config.ab_hidden_c_ps,config.ab_n_layer,bidirectional=True, 
                             dropout=config.ab_dropout, batch_first=True)
        else:
            self.lstmblock1 = PyLstmBlock(config.input_c_ndi,config.hidden_c_ndi,config.n_layer,config.dropout,config.device)
            self.lstmblock2 = PyLstmBlock(config.input_c_dg,config.hidden_c_dg,config.n_layer,config.dropout,config.device)
            self.lstmblock3 = PyLstmBlock(config.input_c_ps,config.hidden_c_ps,config.n_layer,config.dropout,config.device)
        
        self.ln_ndi = nn.LayerNorm(config.hidden_c_ndi[2]*config.num_direc, 1e-5)
        self.ln_dg = nn.LayerNorm(config.hidden_c_dg[2]*config.num_direc, 1e-5)
        self.ln_ps = nn.LayerNorm(config.hidden_c_ps[2]*config.num_direc, 1e-5)
        self.reslstm1 = ResLstm(config.res_input_c[0], config.res_hidden_c[0], config.res_n_layer,config.res_num_direc,config.if_res,config.device)
        self.reslstm2 = ResLstm(config.res_input_c[1], config.res_hidden_c[1], config.res_n_layer,config.res_num_direc,config.if_res,config.device)
        self.reslstm3 = ResLstm(config.res_input_c[2], config.res_hidden_c[2], config.res_n_layer,config.res_num_direc,config.if_res,config.device)

        self.crossattention1 = CrossModeAttention(config.ca_indim_ndidg, config.ca_outdim_ndidg,
                                                  config.cross_num_heads, config.ln_dim, config.fc_outdim,
                                                  config.outdim_fc, config.cross_layer)  # ndi dg
        self.crossattention2 = CrossModeAttention(config.ca_indim_ndips, config.ca_outdim_ndips,
                                                  config.cross_num_heads, config.ln_dim, config.fc_outdim,
                                                  config.outdim_fc, config.cross_layer)  # ndi ps
        self.crossattention3 = CrossModeAttention(config.ca_indim_dgps, config.ca_outdim_dgps,
                                                  config.cross_num_heads, config.ln_dim, config.fc_outdim,
                                                  config.outdim_fc, config.cross_layer)   # dg ps
        if self.final_cross:
            self.finalattention = CrossModeAttention(config.fca_indim_ps, config.fca_outdim_ps,
                                                  config.fcross_num_heads, config.fln_dim, config.ffc_outdim,
                                                  config.foutdim_fc, config.final_cross_layer)
        else:
             self. finalattention = MultiHeadSelfAttention(config.final_indim_self,config.final_outdim_self,config.final_self_num_heads,
                                                           config.final_dim,config.final_outdim_fc)

        self.decoder = Decoder(self.decode_dense_indim, self.decode_dense_outdim, config.decode_drop_p)

        if self.out_aug:
            self.bg = BarrierG(config.bg_k, self.bg_cc_inc, self.bg_cc_outc,
                                config.bg_cc_pad,config.bg_cc_stride,config.bg_cc_dilation)
            self.bp = BarrierPool.LocalBarrierPooling(config.kernel, config.alpha)

    def padding_mask(self, data, pad_idx):
        return (data!=pad_idx)

    def forward(self, input, total_length):
        # B,N,C
        ndi = input[0]  # padded tensor with config.pad_value
        dg= input[1]
        ps = input[2]
        length = input[3].cpu()  # length list should be on CPU()
        # print(ndi.device)
        # total_length = ndi.size(1)  # 模型并行化

        # pack  
        # consider a padding mask!! or else there would be some mistakes!!
        # *********
        ndi = pack_padded_sequence(ndi, length, batch_first=True) # packedsequence
        dg = pack_padded_sequence(dg, length, batch_first=True)
        ps = pack_padded_sequence(ps, length, batch_first=True)
        # *********
        #
        if self.ablation==1:
            B = len(length)
            self.lstmblock1.flatten_parameters()
            self.lstmblock2.flatten_parameters()
            self.lstmblock3.flatten_parameters()
            h_ndi_0 = torch.zeros(self.ab_nlayer * 2, B, 24, device=self.device) #.cuda(self.device_id)
            h_dg_0 = torch.zeros(self.ab_nlayer * 2, B, 20, device=self.device)
            h_ps_0 = torch.zeros(self.ab_nlayer * 2, B, 18, device=self.device)

            output_ndi, _ = self.lstmblock1(ndi, h_ndi_0)   # gru only need h
            output_dg, _ = self.lstmblock2(dg, h_dg_0)
            output_ps, _ = self.lstmblock3(ps, h_ps_0)

        else:
            output_ndi, _ = self.lstmblock1(ndi, length, total_length)  # packedsequence
            output_dg, _ = self.lstmblock2(dg, length, total_length)
            output_ps, _ = self.lstmblock3(ps, length, total_length)
        
        # ndi_u, _ = self.reslstm1(output_ndi, len, total_length)  # packedsequence 
        # tensor, this tensor is packed with 0, need a mask, and carefully check tensors dimention!!
        output_ndi = pad_packed_sequence(output_ndi, batch_first=True, total_length=total_length)
        output_ndi = self.ln_ndi(output_ndi[0])  # padded tensor(sorted)
        ndi_u = output_ndi
        # output_ndi = self.ln(output_ndi)
        # mask_ndi = self.padding_mask(output_ndi,self.pad_value)  # check shape :{B,N,C}

        
        # dg_u, _ = self.reslstm2(output_dg, len, total_length)
        output_dg = pad_packed_sequence(output_dg, batch_first=True, total_length=total_length)
        output_dg = self.ln_dg(output_dg[0])  # sortted
        dg_u = output_dg
        # output_dg = self.ln(output_dg)
        # mask_dg = self.padding_mask(output_dg, self.pad_value)

        
        # ps_u, _ = self.reslstm3(output_ps, len, total_length)
        output_ps = pad_packed_sequence(output_ps, batch_first=True, total_length=total_length)
        output_ps = self.ln_ps(output_ps[0]) # sortted
        ps_u = output_ps
        # print(output_ps.size())

        # mask_ps = self.padding_mask(output_ps, self.pad_value)
        # output_ps = self.ln(output_ps)
        # out:(batch, seq_len, hidden_size): tensor containing the output features (h_t)
        # out = (output_ndi, output_dg, output_ps)  # {batch, seq_len, hidden_dim}

        # pub_feature
        ndi_dg_f = self.crossattention1(output_ndi, output_dg)
        ndi_ps_f = self.crossattention2(output_ndi, output_ps)
        d_ps_f  = self.crossattention3(output_ps, output_dg)
    
        pub_data = torch.cat((ndi_dg_f, ndi_ps_f),2)
        pub_data = torch.cat((pub_data, d_ps_f),2)

        # sing_fearure
        tri_data = torch.cat((ndi_u, dg_u),2)
        tri_data = torch.cat((tri_data,ps_u),2)
        
        # final fuse
        
        out_feature = self.finalattention(pub_data, tri_data)
       
        # decoder
        output = self.decoder(out_feature)

        # barrier
        # original barrier is built by confidence score in BCN, 

        # if self.out_aug:
        #     barrier = self.bg(out_feature)
        #     output = self.bp(output, barrier)
        # print(type(output))
        return output #, out_feature # padded 