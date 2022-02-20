# Target网络的网格搜索调参+交叉验证
# 前四个月做训练集，循环为验证集
import time, sys, os
from functools import partial
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

play_os = "3090"
if play_os == "win10":
    sys.path.append("D:/OneDriveEdu/file/project2/QAData/TargetNet/")
    sys.path.append("D:/OneDriveEdu/file/project2/QAData/SourceNet/")
    sys.path.append("D:/OneDriveEdu/file/project2/QAData/")
elif play_os =="3090":
    sys.path.append("/home/xxx/project/QAData/TargetNet/")
    sys.path.append("/home/xxx/project/QAData/SourceNet/")
    sys.path.append("/home/xxx/project/QAData/")
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
from utils import getLogger, readSet, getTrainLoader_Sou, getTestLoader_Sou, current_time, excutALBert, logout, passItem, saveParams
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
# from src3 import generateData
import os


# def trainColl(batch, train_set):
#     r_batch = []
#     a_text_batch = []
#     q_text_batch = []
#     q_id_batch = []
#     u_id_batch = []
#     u_his_batch = []
#     for r, a_text, q_text, q_id, u_id in batch:
#         # if len(a_vec)*len(q_vec) == 0:
#         #     continue
#         if len(a_text) == 0:
#             a_text = "there is null text."
#         if len(q_text) == 0:
#             q_text = "there is null text."
#         r_batch.append(torch.FloatTensor([eval(r)]))
#         a_text_batch.append(a_text)
#         q_text_batch.append(q_text)
#         q_id_batch.append(q_id)
#         u_id_batch.append(u_id)
#         u_history = ''
#         u_his_list = list(train_set[train_set['u_id'] == u_id]['q_text'])
#         for item in u_his_list:
#             u_history = u_history +"."+ str(item)
#         u_his_batch.append(u_history)
#     r_batch = torch.cat(r_batch, 0)
#     print(len(r_batch))
#     return [r_batch, q_text_batch, a_text_batch, q_id_batch, u_id_batch, u_his_batch]




# class loadDataSet(data.Dataset):
#     def __init__(self, train_set):
#         self.label = list(train_set['r'])
#         self.r=[]
        
#         # train_x['distance'] = [None for item in range(len(list(train_x['q_id'])))]
#         self.a_text = [str(item) for item in list(train_set['a_text'])]
#         self.q_text = [str(item) for item in list(train_set['q_text'])]
#         self.q_id = list(train_set['q_id'])
#         self.u_id = list(train_set['u_id'])
#         # self.distance = list(train_x['distance'])
#         self.data = []
#         for i in range(len(self.a_text)):
#             # print(eval(self.label[i]))
#             if eval(self.label[i]) == [1.0, 0.0,0.0,0.0]:
#                 # print("1")
#                 self.r.append([1.0,0.0])
#             else:
#                 # print("2")
#                 self.r.append([0.0, 1.0])
#             self.u_his_list = ".".join(list(train_set[train_set['u_id'] == self.u_id[i]]['q_text']))
#             self.data.append([self.r[i], self.q_text[i], self.a_text[i], self.q_id[i], self.u_id[i], self.u_his_list])
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]


# def getTrainLoader(train_set):
#     start = time.time()
#     dataset = loadDataSet(train_set)
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=param['batch_size'], shuffle=True, num_workers=20)
#     print("训练数据组装完成，耗时：", time.time()-start, "秒")
#     return train_loader


# def getTestLoader(test_set):
#     start = time.time()
#     dataset = loadDataSet(test_set)
#     test_loader = torch.utils.data.DataLoader(dataset, batch_size=param['batch_size'], shuffle=True, num_workers=20)
#     print("测试数据组装完成，耗时：", time.time()-start, "秒")
#     return test_loader



if __name__ == '__main__':

    src = '/home/xxx/project/QAData/'
    datapath = src+'data/'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    tenoserBoardStr = ''
    # 创建参数列表：
    param = {}
    param['device'] = str(device)
    param['batch_size'] = 30
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tarNet = TargetModel.TarM(param, device)
    tarNetWeights = torch.load('/home/xxx/project/QAData/TargetNet/runs/2021-06-15 22-44-12/trainTarModel_step_9685,loss_1.0332.pth')
    tarNet.load_state_dict(tarNetWeights)
    tarNet = tarNet.to(device, non_blocking=True)
    # tarNet = torch.load('D:/OneDriveEdu/file/project2/QAData/TargetNet/runs/2021-06-01 14-46-41ALbert词向量+CNN3层/trainTarModelEpoch_0,step_0,loss_1.3325.pth')
    print("T模型加载成功...")
    train_set, test_set = readSet(datapath)
    train_loader = getTrainLoader_Sou(train_set, param)     # 用户历史只截取最后30条
    test_loader = getTestLoader_Sou(test_set, param)        # 加载测试集，测试效果  用户历史只截取最后30条
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
        model = Sou1Model.Sou1M(param, device).to(device)
        sou1loss = Sou1ModelLoss.Sou1Loss().to(device)
        # sourceNetwork的更新与TargetNetwork的优化器一致，初始值也相同
        optimizer = torch.optim.SGD(model.parameters(), lr=param['learningrate'])
        scheduler = StepLR(optimizer, step_size=schedulerDict['step_size'], gamma=schedulerDict['gamma'])
        print("S模型及优化器初始化完成")
        logstr = current_time()
        param['board_str'] = logstr
        savestr = src + 'SourceNet/runs/' + logstr
        writer = SummaryWriter(savestr)  # GPU5
        saveParams(params, savestr+"/"+str(i)+".json")
        epoch_num = 200
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

        
        for epoch in range(epoch_num):
            model.train()
            for [r_batch, q_text_batch, a_text_batch, q_id_batch, u_id_batch, u_his_batch] in train_loader:
                r_batch = r_batch.long().to(device)
                step_time = time.time()
                optimizer.zero_grad()
                q_vec_batch = excutALBert(q_text_batch, bertTokenizer, bertModel, sen_tokenizer, device).to(device)
                a_vec_batch = excutALBert(a_text_batch, bertTokenizer, bertModel, sen_tokenizer, device).to(device)
                u_his_vec_batch = excutALBert(u_his_batch, bertTokenizer, bertModel, sen_tokenizer, device).to(device)
                # print("bert vec已生成")
                qu_represent, RS = model.forward(q_vec_batch, u_his_vec_batch)
                tar_represent, RT = tarNet.forward(q_vec_batch, a_vec_batch)
                # tar_represent = tar_represent.cpu()
                # qu_represent = qu_represent.cpu()
                # loss
                step_loss = sou1loss(tar_represent, qu_represent, RS, r_batch)
                step_loss.backward()
                optimizer.step()
                # 改到这里了
                prob = torch.softmax(RS, dim=1)
                prob, cla = torch.max(prob, dim=1)
                step_correct = torch.sum(cla==r_batch).item()
                train_correct += step_correct
                train_mean_loss = step_loss.item()/len(r_batch)
                train_step_acc = step_correct/len(r_batch)
                passStr = ""
                logStr = 'train:'+passStr+'\t'+str(train_step)+'\t'+str(step_loss.item())[:6]+'\t'+str(train_step_acc)
                logout(logger=logger, str=logStr)
                if train_step_acc > max_train_acc:
                    save_str = savestr+'/trainSouModel_step_'+str(train_step)+',acc_'+str(train_step_acc)[:6]
                    torch.save(model.state_dict(), save_str+'.pth')
                    torch.save(model, save_str +'.model')
                    max_train_acc = train_step_acc
                if train_step % 5 == 0:
                    writer.add_scalar("train_loss", step_loss.item(), train_step)
                    writer.add_scalar("train_acc", train_step_acc, train_step)
                train_step += 1
            if train_step == 12000:
                break


            #     step_correct = torch.sum(cla==torch.max(r_batch, dim=1)[1]).item()
            #     train_correct += step_correct

            #     logger.info('train'+'\t'+str(epoch)+'\t'+str(train_step)+'\t'+str(step_loss.item())[:6]+'\t'+str(train_correct)+'\t'+str(step_correct/param['batch_size']))
            #     print('train'+'\t'+str(epoch)+'\t'+str(train_step)+'\t'+str(step_loss.item())[:6]+'\t'+str(train_correct)+'\t'+str(step_correct/param['batch_size']))
            #     if step_correct > max_train_correct:
            #         save_str = '/home/student/xxx/project/transNetsQANew/src3/data3/model/trainsave/TarModel/epoch:' + str(epoch)+',step:'+str(train_step)+',loss:'+str(step_loss.item())[:6]
            #         torch.save(model.state_dict(), save_str+'.pth')
            #         torch.save(model, save_str +'.model')
            #         max_train_correct = step_correct
            #     
            #     scheduler.step()  # 每个epoch学习率动态变化
            # test_class_num={}
            # test_step = 0
            # test_correct = 0
            # for [r_batch, q_text_batch, a_text_batch, q_id_batch, u_id_batch] in test_loader:
            #     q_vec_batch = excutBert(q_text_batch, bertTokenizer, bertModel, sen_tokenizer)
            #     a_vec_batch = excutBert(a_text_batch, bertTokenizer, bertModel, sen_tokenizer)
            #     # print("bert vec已生成")
            #     tar_drop_vec, RT = model.forward(q_vec_batch, a_vec_batch)
            #     prob, cla = torch.max(RT, dim=1)
            #     test_step_correct = torch.sum(cla == torch.max(r_batch, dim=1)[1]).item()
            #     test_correct += test_step_correct
            #     test_class_num['cla1'] += torch.sum(r_batch == torch.FloatTensor([1.0,0.0,0.0,0.0])).item()
            #     test_class_num['cla2'] += torch.sum(r_batch == torch.FloatTensor([0.0,1.0,0.0,0.0])).item()
            #     test_class_num['cla3'] += torch.sum(r_batch == torch.FloatTensor([0.0,0.0,1.0,0.0])).item()
            #     test_class_num['cla4'] += torch.sum(r_batch == torch.FloatTensor([0.0,0.0,0.0,1.0])).item()
            #     logger.info('test'+'\t'+str(epoch)+'\t'+str(test_step)+'\t'+str(step_loss.item())[:6]+'\t'+str(test_correct))
            #     print('test'+'\t'+str(epoch)+'\t'+str(test_step)+'\t'+str(step_loss.item())[:6]+'\t'+str(test_correct))
            #     if test_step_correct > max_test_correct:
            #         save_str = '/home/student/xxx/project/transNetsQANew/src3/data3/model/testsave/TarModel/epoch:' + str(epoch)+',step:'+str(test_step)+',loss:'+str(step_loss.item())[:6]
            #         torch.save(model.state_dict(), save_str+'.pth')
            #         torch.save(model, save_str +'.model')
            #         max_train_correct = torch.sum(
            #             cla == torch.max(r_batch, dim=1)[1]).item()
            #     test_step += 1
            # print(epoch, "loss", train_loss, "训练准确率", train_correct/len(train_set), "测试集准确率：", test_correct/len(test_set))
            # logger.info('epoch'+'\t'+str(epoch) + "\t"+ str(train_loss) + '\t' + str(train_correct/len(train_set)) +'\t'+str(test_correct/len(test_set))+'\t'+str(test_class_num)+'\t'+str(train_class_num))

