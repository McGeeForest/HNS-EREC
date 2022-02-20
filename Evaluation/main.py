# Target网络的网格搜索调参+交叉验证
# 前四个月做训练集，循环为验证集
import time, sys, os
from functools import partial

from numpy.lib.type_check import real
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

play_os = "3090"
if play_os == "win10":
    sys.path.append("D:/OneDriveEdu/file/project2/QAData/TargetNet/")
    sys.path.append("D:/OneDriveEdu/file/project2/QAData/SourceNet/")
    sys.path.append("D:/OneDriveEdu/file/project2/QAData/")
elif play_os =="3090":
    sys.path.append("/home/xxx/projects/project/QAData/TargetNet/")
    sys.path.append("/home/xxx/projects/project/QAData/SourceNet/")
    sys.path.append("/home/xxx/projects/project/QAData/")
else:
    sys.path.append("/home/student/xxx/project/QAData/TargetNet/")
    sys.path.append("/home/student/xxx/project/QAData/SourceNet/")
    sys.path.append("/home/student/xxx/project/QAData/")

# print(sys.path)
import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data
import numpy as np
from utils import getLogger, readDataEvaluation, getTrainLoader_Sou, getEvalLoader, getTestLoader_Sou, current_time, excutALBert, logout, passItem, saveParams
from eval_method import *
from TargetNet import TargetModel
from TargetNet import TargetModelLoss
from SourceNet import Sou1Model
from SourceNet import Sou1ModelLoss
import logging
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import sent_tokenize
from transformers import BertModel, BertTokenizer, AlbertModel, AlbertTokenizer
import nltk
import os

if __name__ == '__main__':

    src = '/home/xxx/projects/project/QAData/'
    datapath = src+'data/'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    tenoserBoardStr = ''
    # 创建参数列表：
    param = {}
    param['device'] = str(device)
    param['batch_size'] = 20
    param['learningrate'] = 0.01
    param['extracter_name'] = "CNNTextExtracterWithMHSelfAttion"
    param['extracter_kernel_num'] = 300
    param['extracter_kernel_size'] = 3
    param['extracter_vocab_size'] = 1000
    param['extracter_embed_dim'] = 768
    param['extracter_padding'] = 0
    param['extracter_n_hidden'] = 150
    # param['extracter_n_hidden_2'] = 80
    param['extracter_out_dim'] = 400

    param['MFC1_hidden'] = 200
    param['MFC1_out'] = 2
    param['dropout'] = 0.5
    schedulerDict = {}
    schedulerDict['name'] = "StepLR"
    schedulerDict['optimizer'] = "SGD"
    schedulerDict['lr'] = param['learningrate']  # 不要修改此值，对上面的学习率修改
    schedulerDict['step_size'] = 300
    schedulerDict['gamma'] = 0.85
    param['optimizer'] = schedulerDict
    param['MFC2_hidden'] = 50
    params = [param]

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    his_set, eval_set = readDataEvaluation(datapath,more_than=5)
    # train_loader = getTrainLoader_Sou(his_set, param)     # 用户历史只截取最后30条
    eval_loader = getEvalLoader(his_set, eval_set, param)

    logger = getLogger(src)
    # bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # bertModel = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(device, non_blocking=True)
    bertTokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    bertModel = AlbertModel.from_pretrained('albert-base-v2').to(device, non_blocking=True)
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')   #加载punkt句子分割器
    paramid = 0
    for i in range(len(params)):
        param = params[i]
        start = time.time()
        model = Sou1Model.Sou1M(param, device)
        modelWeights = torch.load('/home/xxx/projects/project/QAData/SourceNet/runs/2021-09-01 21-23-04/trainSouModel_step_5,acc_0.7.pth')
        model.load_state_dict(modelWeights)
        model = model.to(device, non_blocking=True)
        # sourceNetwork的更新与TargetNetwork的优化器一致，初始值也相同
        print("S模型及优化器初始化完成")
        logstr = current_time()
        param['board_str'] = logstr
        train_step = 0
        test_step = 0
        test_correct = 0
        min_train_loss = 1000000
        max_train_acc = 0.0
        step_loss = []
        train_correct = 0
        train_loss = 0.0
        test_loss = 0
        test_min_loss = 1000000

    
        model.eval()
        res_df = pd.DataFrame()
        q_id_list = []
        u_id_list = []
        score_list = []
        expert_prob_list = []
        real_u_set_list = []
        for [r_batch, q_text_batch, q_id_batch, u_id_batch, u_his_batch, score_batch, real_u_batch] in tqdm(eval_loader):
            r_batch = r_batch.long().to(device)
            step_time = time.time()
            q_vec_batch, q_real_len = excutALBert(q_text_batch, bertTokenizer, bertModel, sen_tokenizer, device)
            u_his_vec_batch, u_his_real_len = excutALBert(u_his_batch, bertTokenizer, bertModel, sen_tokenizer, device)

            q_vec_batch = q_vec_batch.to(device, non_blocking=True)
            q_real_len = torch.Tensor(q_real_len) # 代表每个句子的长度
            u_his_vec_batch = u_his_vec_batch.to(device, non_blocking=True)
            u_his_real_len = torch.Tensor(u_his_real_len) # 代表每个句子的长度

            # print("bert vec已生成")
            qu_represent, RS = model.forward(q_vec_batch, u_his_vec_batch, q_real_len, u_his_real_len)
            # print(RS)
            prob = torch.softmax(RS, dim=1)
            # print(prob, r_batch)
            # prob, cla = torch.max(prob, dim=1)
            indices = torch.tensor([0]).to(device)
            expert_prob = torch.index_select(prob, 1, indices).detach().cpu().numpy()
            # res_df = pd.DataFrame()
            q_id_list.extend(q_id_batch)
            u_id_list.extend(u_id_batch)
            score_list.extend(score_batch)
            expert_prob_list.extend(expert_prob)
            real_u_set_list.extend(real_u_batch)
            # break
            # print(res_df)
        q_id_list = [item.item() for item in q_id_list]
        u_id_list = [item.item() for item in u_id_list]
        expert_prob_list = [item.item() for item in expert_prob_list]
        res_df["q_id"] = q_id_list
        res_df["u_id"] = u_id_list
        res_df["score"] = score_list
        res_df["expert_prob"] = expert_prob_list
        res_df["real_u_set"] = real_u_set_list
        res_df.to_csv("res.csv")
        # eval_list = ['MRR@10', 'MRR@20', 'MRR@30', 'MRR@40', 'MRR@50', 'MRR@60', 'MRR@70', 'MRR@80', 'MRR@90', 'MRR@100', 
        #              'TOP@10', 'TOP@20', 'TOP@30', 'TOP@40', 'TOP@50', 'TOP@60', 'TOP@70', 'TOP@80', 'TOP@90', 'TOP@100'
        #              ]    # 评价指标列表
        eval_list = ['MRR@1', 'MRR@2', 'MRR@3', 'MRR@4', 'MRR@5', 'MRR@10', 'MRR@15', 'MRR@20', 'MRR@25', 'MRR@30', 'MRR@35', 'MRR@40', 'MRR@45', 'MRR@50',
                     'TOP@1', 'TOP@2', 'TOP@3', 'TOP@4', 'TOP@5', 'TOP@10', 'TOP@15', 'TOP@20', 'TOP@25', 'TOP@30', 'TOP@35', 'TOP@40', 'TOP@45', 'TOP@50'
                     ]    # 评价指标列表
        res_eval = pd.DataFrame(columns=eval_list)
        for index, item in res_df.groupby("q_id"):
            print(list(item["q_id"])[0])
            real_user_set = str(list(item["real_u_set"])[0]).split(",")
            print(real_user_set)
            expert_prob = np.array(item["expert_prob"])
            score = np.array(item["score"])
            predict_rank = [str(list(item["u_id"])[i]) for i in np.argsort(-expert_prob)]
            # print("预测列表", predict_rank)
            user_rank_with_random = [list(item["u_id"])[i] for i in np.argsort(-score)]    # 包含添加的随机用户
            # print("user_rank_with_random", user_rank_with_random)
            user_rank_real = []
            for u_id in user_rank_with_random:
                # print(str(u_id), str(u_id) in real_user_set)
                if str(u_id) in real_user_set:
                    user_rank_real.append(str(u_id))
            # real_user_rank = [str(u_id) for u_id in real_rank if str(u_id) in real_user_set]
            # print(user_rank_real)
            # print("******")
            # x = metrics(real_rank, predict_rank, eval_list)
            x = metrics(user_rank_real, predict_rank, eval_list)
            res_eval.loc[res_eval.shape[0]+1] = x
        res_eval.to_csv("res_eval.csv")


