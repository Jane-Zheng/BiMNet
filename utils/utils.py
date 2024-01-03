# ------------------------------------------------------------------------------
# --coding='utf-8'--python 3.8
#
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
import os
import logging
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score


# 保存检查点
def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'model_best.pth.tar'))


def cal_quota(outputs, targets, criterion, seq_length, mask, weight, crossentropyweight,stage, save_sequence_path):
    # loss\cm\f1\time
    """
    :param criterion: CrossEntropy()
    :param outputs: shape:{batch size,h,w}
    :param targets: shape:{batch size,h,1}
    :param seq-length: tensor shape:{batch_size, }
    :param mask:{batch size,h,1}
    :param stage:'train' or 'test'
    :return:
    """
    # print(outputs.size())
    # cal loss
    outputs = outputs.permute(0, 2, 1).contiguous()  # {b,d,l}   
    # print(targets.size())
    # print(mask.size())
    targets = targets.masked_fill(mask==0, 0) 
    if len(targets.size()) == 3:
        targets = torch.squeeze(targets, dim=2)  # {b,l}
    assert targets.size()[1] == outputs.size()[2], "size unmatched!!"
    batch, padlength = targets.size()
    # criterion is crossentropy loss with reduction='none'
    # print(targets.size())
    # print(mask.size())
    # mask.detach()
    if stage == 'train':

        losses = criterion(outputs, targets)  # since reduction ='none', losses is a tensor with grad(check shape!)
        # print(losses.size())
        # print(losses.type())
        losses = losses.unsqueeze(dim=-1)
        losses = losses * mask
        # mask is calculated by targets
        if crossentropyweight == 1:
            numofweight= torch.zeros(size=targets.size(), device=targets.device)
            for i in range(batch):
                for j in range(padlength):
                    numofweight[i,j]=weight[targets[i,j]]

            mask2 = torch.squeeze(mask, dim=2)
            numofweight=numofweight*mask2
            real_length = numofweight.sum()
        else:
            real_length = seq_length.sum()
    
        losses = losses.sum()/real_length 
    elif stage == 'test':
        losses = 0
     
    # need mask and the real length of label .
    # cm
    # outputs = outputs.cpu().detach().numpy()  # 
    outputs = outputs.detach()
    targets = targets.detach()   
    y_pred1 = torch.argmax(outputs, dim=1)  # {b,h}
    # print(y_pred1.size())
    b, h = y_pred1.shape
    # print(b)
    if stage == 'test':
        for batch_id in range(b):
           prediction = y_pred1[batch_id,0:seq_length[batch_id]]   
           # print(prediction.device)   
           true_lable = targets[batch_id,0:seq_length[batch_id]]
           plot_sequence(prediction,true_lable, batch_id, save_path=save_sequence_path)

    # print(seq_length[0])
    prediction = y_pred1[0,0:seq_length[0]]   
    # print(prediction.device)   
    true_lable = targets[0,0:seq_length[0]]
    
    for i in range(1,b):
        l = seq_length[i]
        prediction=torch.cat((prediction,y_pred1[i,0:l]),dim=0)
        true_lable = torch.cat((true_lable,targets[i,0:l]),dim=0)
    
    # now prediction is a single dim tensor 
    # print(prediction.shape)  
    # y_pred1 = y_pred1.reshape((b * h,))
    # y_true1 = targets.cpu().reshape((b * h,))
    prediction = prediction.cpu().detach().numpy()
    true_lable = true_lable.cpu().detach().numpy()
    # cm = confusion_matrix(true_lable, prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7])  # row true  column pred
    cm = confusion_matrix(true_lable, prediction, labels=[0, 1, 2, 3, 4]) 
    # cm_n = confusion_matrix(y_true1, y_pred1, normalize='true', labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    return losses, cm


def build_quota(cm):
    """
    :param cm: confusion matrix as ndarray
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
    # print(acc.type())
    # print(np.sum(cm).type())
    s = np.sum(cm)
    acc = acc / s
    # print(acc)
    return f1, acc, prec, rec


def plot_confusion_matrix(configs, cm):
    fig = plt.figure()
    # class_name = [1, 2, 3, 4, 5]
    # class_name = ['I','M','PE','MC','CP','T','P','O']
    class_name = ['I','MC','CP','T','O']
    # new_cm = cm
    # normalize
    h, c = cm.shape
    new_cm = np.zeros((h, c))
    # prec
    for j in range(h):
        co = np.sum(cm[:, j])
        for i in range(c):
            new_cm[i, j] = 0 if co == 0 else cm[i, j] / co
    # recall
    # for j in range(h):
    #     co = np.sum(cm[:, j])
    #     for i in range(c):
    #         new_cm[i, j] = 0 if co == 0 else cm[i, j] / co

    plt.imshow(new_cm, cmap='GnBu')
    plt.title('Test Confusion Matrix ', fontsize=12)
    xlocations = np.array(range(len(class_name)))
    plt.xticks(xlocations, class_name, rotation=0, fontsize=12)
    plt.yticks(xlocations, class_name, fontsize=12)
    plt.colorbar()
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicated Label', fontsize=12)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # plt.text(x=j, y=i, s=new_cm[i, j], va='center',
            #          ha='center', color='red', fontsize=6)
            plt.text(x=j, y=i, s=0 if new_cm[i, j] == 0 else '{:.2%}'.format(new_cm[i, j]), va='center',
                     ha='center', color='red', fontsize=12)
    path = os.path.join(configs.output_dir, 'confusion_matrix' + configs.model_name + '.png')
    plt.savefig(path)
    plt.close()

def save_cm(cm, name):
    # column = ['pred_I','pred_M','pred_PE','pred_MC','pred_CP','pred_T','pred_P','pred_O']
    # ind = ['true_I','true_M','true_PE','true_MC','true_CP','true_T','true_P','true_O']
    column = ['pred_I','pred_MC','pred_CP','pred_T','pred_O']
    ind = ['true_I','true_MC','true_CP','true_T','true_O']
    c = pd.DataFrame(data=cm, index=ind, columns=column)
    c.to_csv(name)

    

def plot_loss(list1, list2, save_path, fold, model_name,tm):
    # length1 = len(list1)
    # x = range(length1)
    l1, = plt.plot(list1)
    l2, = plt.plot(list2)
    plt.legend((l1, l2), ['train', 'validation'])
    plt.title("loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(save_path, model_name+tm+'fold_{}_loss.png'.format(str(fold))))
    plt.close()


def save_log(log, log_path, quota_path):
    """
    :param log: kfold:list{dict1,dict2,...}(dict:['**']:list:{epoch,n});retrain: dict
    :return: none
    """    
    if isinstance(log, list):
        # k fold train log
        n = len(log)
        # name_list = [
        # 'fold', 'train_prec_0', 'train_prec_1', 'train_prec_2', 'train_prec_3', 'train_prec_4', 'train_prec_5',
        # 'train_prec_6', 'train_prec_7', 'train_rec_0', 'train_rec_1', 'train_rec_2', 'train_rec_3',
        # 'train_rec_4', 'train_rec_5', 'train_rec_6', 'train_rec_7', 'train_acc', 'train_f1',
        # 'train_loss', 'val_prec_0', 'val_prec_1', 'val_prec_2', 'val_prec_3', 'val_prec_4', 'val_prec_5',
        # 'val_prec_6', 'val_prec_7', 'val_rec_0', 'val_rec_1', 'val_rec_2', 'val_rec_3', 'val_rec_4',
        # 'val_rec_5', 'val_rec_6', 'val_rec_7', 'val_acc', 'val_f1', 'val_loss', 'lr']
        name_list = [
        'fold', 'train_prec_1', 'train_prec_2', 'train_prec_3', 'train_prec_4', 'train_prec_5',
        'train_rec_1', 'train_rec_2', 'train_rec_3', 'train_rec_4', 'train_rec_5', 'train_acc', 
        'train_f1', 'train_loss', 'val_prec_1', 'val_prec_2', 'val_prec_3', 'val_prec_4', 
        'val_prec_5', 'val_rec_1', 'val_rec_2', 'val_rec_3', 'val_rec_4','val_rec_5', 'val_acc', 
        'val_f1', 'val_loss', 'lr']  # 26
        name_quota  = ['val_acc', 'val_f1'] * n
        logger = []
        quotalog = []
        for i in range(n):
            data = log[i]
            # epoch_records = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': [], 'best_model_name': [],
            #                      'train_f1': [], 'train_acc': [], 'train_prec': [], 'val_prc': [], 'train_rec': [],
            #                      'val_rec': []}
            ll = len(data['train_prec'])
            name = [i + 1] * ll
            name = np.array(name).reshape(ll, 1)
            train = np.hstack([np.array(data['train_prec']), np.array(data['train_rec'])])
            train = np.hstack([train, np.array(data['train_acc']).reshape(ll, 1)])
            train = np.hstack([train, np.array(data['train_f1']).reshape(ll, 1)])
            train = np.hstack([train, np.array(data['train_loss']).reshape(ll, 1)])  # {epoch,21}

            val = np.hstack([np.array(data['val_prec']), np.array(data['val_rec'])])
            val = np.hstack([val, np.array(data['val_acc']).reshape(ll, 1)])
            val = np.hstack([val, np.array(data['val_f1']).reshape(ll, 1)])
            val = np.hstack([val, np.array(data['val_loss']).reshape(ll, 1)])  # {epoch,21}
            quota = np.hstack([np.array(data['val_acc']).reshape(ll, 1),np.array(data['val_f1']).reshape(ll, 1)])
            d = np.hstack([train, val])
            d = np.hstack([d, np.array(data['lr']).reshape(ll, 1)])
            d = np.hstack([name, d])
            logger.append(d)
            quotalog.append(quota)
        logger = np.array(logger)  # {fold,epoch,43}
        quotalog = np.array(quotalog)  # {fold,epoch, 2} 
        a, b, c = logger.shape
        e, d, f = quotalog.shape
        # print(logger1.shape)
        logger = np.reshape(logger, (a * b, c))
        quotalog1 = np.stack([i for i in quotalog], axis=1) # {d, e, f}
        quotalog1 = np.reshape(quotalog1, (d, e*f))
        log_file = pd.DataFrame(data=logger, columns=name_list)
        quota_file =  pd.DataFrame(data=quotalog1, columns=name_quota)
        log_file.to_csv(log_path, index=False)
        quota_file.to_csv(quota_path, index=False)
        
    elif isinstance(log, dict):
        # retrain log
        # name_list = [
        # 'retrain_prec_0', 'retrain_prec_1', 'retrain_prec_2', 'retrain_prec_3', 'retrain_prec_4', 'retrain_prec_5',
        # 'retrain_prec_6', 'retrain_prec_7', 'retrain_rec_0', 'retrain_rec_1', 'retrain_rec_2', 'retrain_rec_3',
        # 'retrain_rec_4', 'retrain_rec_5', 'retrain_rec_6', 'retrain_rec_7', 'retrain_acc', 'retrain_f1',
        # 'retrain_loss', 'test_prec_0', 'test_prec_1', 'test_prec_2', 'test_prec_3', 'test_prec_4', 'test_prec_5',
        # 'test_prec_6', 'test_prec_7', 'test_rec_0', 'test_rec_1', 'test_rec_2', 'test_rec_3',
        # 'test_rec_4', 'test_rec_5', 'test_rec_6', 'test_rec_7', 'test_acc', 'test_f1','lr']  # 38
        name_list = [
        'retrain_prec_1', 'retrain_prec_2', 'retrain_prec_3', 'retrain_prec_4', 'retrain_prec_5',
        'retrain_rec_1', 'retrain_rec_2', 'retrain_rec_3','retrain_rec_4', 'retrain_rec_5', 
        'retrain_acc', 'retrain_f1','retrain_loss','lr']
        # , 'test_prec_1', 'test_prec_2', 'test_prec_3', 'test_prec_4', 'test_prec_5',
        # 'test_rec_1', 'test_rec_2', 'test_rec_3','test_rec_4', 'test_rec_5', 'test_acc', 'test_f1','lr']  # 26
        # retrain_epoch_records = {'train_loss': [], 'train_f1': [], 'train_acc': [], 'train_prec': [], 'train_rec': [], 'lr': []}
        data = log
        ll = len(data['train_prec'])
        train_log = np.hstack([np.array(data['train_prec']), np.array(data['train_rec'])])
        train_log = np.hstack([train_log, np.array(data['train_acc']).reshape(ll, 1)])
        train_log = np.hstack([train_log, np.array(data['train_f1']).reshape(ll, 1)])
        train_log = np.hstack([train_log, np.array(data['train_loss']).reshape(ll, 1)])  # {epoch,13}
        train_log = np.hstack([train_log, np.array(data['lr']).reshape(ll, 1)])

        # test = np.hstack([np.array(data['test_prec']), np.array(data['test_rec'])])
        # test = np.hstack([test, np.array(data['test_acc']).reshape(ll, 1)])
        # test = np.hstack([test, np.array(data['test_f1']).reshape(ll, 1)])
        # test = np.hstack([test, np.array(data['test_loss']).reshape(ll, 1)])  # {epoch,18}

        # d = np.hstack([train_log, test])
        # d = np.hstack([d, np.array(data['lr']).reshape(ll, 1)])

        log_file = pd.DataFrame(data=train_log, columns=name_list)
        log_file.to_csv(log_path, index=False)


def save_testlog(loglist, log_path):
    n = len(loglist)
    log = np.reshape(np.array(loglist), (1, 12))
    # name_list = [
    #     'prec_0', 'prec_1', 'prec_2', 'prec_3', 'prec_4', 'prec_5',
    #     'prec_6', 'prec_7','rec_0', 'rec_1', 'rec_2', 'rec_3',
    #     'rec_4', 'rec_5', 'rec_6', 'rec_7', 'test_acc', 'test_f1']  #,'model_average_score']
    name_list = [
        'prec_1', 'prec_2', 'prec_3', 'prec_4', 'prec_5','rec_1', 'rec_2', 'rec_3',
        'rec_4', 'rec_5', 'test_acc', 'test_f1']  #,'model_average_score']
    log = pd.DataFrame(data=log, columns=name_list)
    log.to_csv(log_path, index=False)

def plot_sequence(targets, ground_truth, batch, save_path):
    # color_dict = {
    #     '1': ['I', '#FAF1A1'],
    #     '2': ['M', '#E4DBFA'],
    #     '3': ['PE', '#90caf9'],
    #     '4': ['MC', '#B6B4F5'],
    #     '5': ['CP', '#8CE2F5'],
    #     '6': ['T', '#FFDDE8'],
    #     '7': ['P', '#e6e4e4'],
    #     '8': ['O', '#DCE775'],
    #     '0': ['E', '#C0F5D8']}
    color_dict = {
        '0': ['I', '#FFEAAC'],
        '1': ['MC', '#D0E5C5'],
        '2': ['CP', '#C9DEF0'],
        '3': ['T', '#F8D4BC'],
        '4': ['O', '#BCC6D4']}

    temp = targets.tolist()  
    temp_g = ground_truth.tolist()
    s = 0
    plt.figure(figsize=(20, 2))
    # subplot 1
    plt.subplot(2, 1, 1)
    ax = plt.gca()  # gca:get current axis
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')

    for j in range(len(temp)):
        a = temp[j]
        plt.bar(x=s, height=0.1, width=1, label=color_dict[str(a)][0], color=color_dict[str(a)][1], align='edge')
        s = s + 1

    plt.yticks([])
    plt.xticks(fontproperties='Times New Roman', size=10)
    # subplot 2
    plt.subplot(2, 1, 2)
    ax2 = plt.gca()  # gca:get current axis
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.spines['left'].set_color('none')
    s = 0
    for j in range(len(temp_g)):
        a = temp_g[j]
        plt.bar(x=s, height=0.1, width=1, label=color_dict[str(a)][0], color=color_dict[str(a)][1], align='edge')
        s = s + 1
    plt.yticks([])
    plt.xticks(fontproperties='Times New Roman', size=10)
    # location adjust
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.8, wspace=1, hspace=1)
    # save
    tm = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    plt.savefig(os.path.join(save_path, str(batch)+tm))


def plot_legend(path):
    "This part used to save sequence legend, remember to check the color is right"
    plt.figure(figsize=(5, 10))
    ax = plt.gca()  
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    plt.plot(np.arange(2), [6, 6], color='#FFDAB5', label='I', linewidth=5)  #
    # plt.plot(np.arange(2), [18, 18], color='#C7A4C4', label='M', linewidth=5)  #
    # plt.plot(np.arange(2), [6, 6], color='#D4DDE5', label='PE', linewidth=5)  #
    plt.plot(np.arange(2), [5, 5], color='#FFE0E5', label='MC', linewidth=5)  #
    plt.plot(np.arange(2), [4, 4], color='#70ACC2', label='CP', linewidth=5)  #
    plt.plot(np.arange(2), [3, 3], color='#F6B4A6', label='T', linewidth=5)  #
    # plt.plot(np.arange(2), [3, 3], color='#A1AEB1', label='P', linewidth=5)  #
    plt.plot(np.arange(2), [2, 2], color='#9DC1C6', label='O', linewidth=5)  #
    plt.legend(loc='right')
    plt.savefig(path)

