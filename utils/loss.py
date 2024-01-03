# --coding='utf-8'-- #
import os
from re import L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth, p, reduction):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        '''

        '''
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        # loss = 1 - loss.sum() / N
        return 1 - loss


class FocalLoss(nn.Module):
    def __init__(self, weight, gamma=2, logits=False, sampling='mean'):
        super(FocalLoss, self).__init__()
        '''
        focal loss:alpha*(1-p)^gama*cross_entropy
        prediction: output of the model
        target: real label
        weight: multi class weight is alpha for binary focal loss
        '''
        self.gamma = gamma
        self.logits = logits
        self.sampling = sampling
        self.crossentropy = nn.CrossEntropyLoss(weight, reduction='none')

    def forward(self, prediction, target):
        '''
        prediction: {batch,dim,length} padded to same length
        target:{batch,length} padding_value:0
        '''
        # alpha = self.alpha
        cross_loss = self.crossentropy(prediction, target)       
        mask = (prediction!=0)
        probs = F.softmax(prediction, dim=1) #{batch,dim,length}
        probs = probs*mask
        probs = probs.gather(dim=1,index=target.unsqueeze(dim=1)) # {batch, 1, length}
        # 按标签挑出来
        probs = probs.squeeze(dim=1)
        loss = (1-probs)*cross_loss  

        if self.sampling == 'mean':
            return loss.mean()
        elif self.sampling == 'sum':
            return loss.sum()
        elif self.sampling == 'none':
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, weight,ignore_index=-100,size_average=None,reduce=None,reduction='none'):
        '''
        defult: reduction = 'none'
        包含变长序列需要的mask和维度转换
        x: {batch,length,dim}
        target:{batch,length}
        smoothing:float
        '''
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.reduction = reduction
        self.nll = nn.NLLLoss(weight=weight,ignore_index=ignore_index,size_average=size_average,reduce=reduce,ruduction='none')
        
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        mask = (x!=0)
        logprobs = F.log_softmax(x, dim=-1)
        logprobs = logprobs*mask
        smooth_loss = -logprobs.mean(dim=-1)
        logprobs = logprobs.permute(0,2,1) # {batch,dim,length}
        nll_loss = self.nll(logprobs, target)
        # nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        # nll_loss = nll_loss.squeeze(1)
        
        loss = confidence * nll_loss + smoothing * smooth_loss  # {batch,length}
        if self.reduction == 'mean':
            loss = loss.mean(dim=-1)
        return loss

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss).__init__()

    def forward(self,output, targets):
        pass

class MultiClassDiceLoss(nn.Module):
    def __init__(self):
        super(MultiClassDiceLoss).__init__()

    def forward(self, output, targets):
        pass


class MultiClassFocalLoss(nn.Module):
    def __init__(self, config):
        super(MultiClassFocalLoss).__init__()
        self.alpha = config.alpha
        self.gama = config.acoe_of_focal
        self.weight = config.weight

    def forward(self, output, targets):
        '''
        from cross entropy loss, add an alpha and a attenuation coeffcient to control the loss of more difficult samples.
        output: pred labels without softmax,shape:{L,D} or {B,L,D}
        targets: true label (not onehot), used as index,shape:{L,} or {B,L}
        weight: inverse ratio of each class's data ratio.,shape:{1,D}

        function: -alpha*(1-pred)**gama(-pred-log(sum(pred_i))) = alpha*(1-pred)**gama * CrossEntropyLoss
        '''
        pt = F.softmax(output, dim=1) # {B,D,L} 
        class_mask = F.one_hot(targets, self.class_num)  # one hot {B,L,D}
        class_mask  = class_mask.permute(0, 2, 1).contiguous() 
        ids = targets.view(-1, 1)  # {B*L,1} 
        alpha = self.alpha[ids.data.view(-1)]

        probs = (pt * class_mask).sum(1).view(-1, 1)  # {B*L,1} 
        log_p = probs.log()

        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
        
                
        