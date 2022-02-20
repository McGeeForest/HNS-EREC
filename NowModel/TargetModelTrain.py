import sys
import os
import time
from typing import Mapping
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
from tensorboardX import SummaryWriter
from torchsummary import summary
import torch
from torch.optim.lr_scheduler import StepLR
from transformers import BertModel, BertTokenizer, AlbertModel, AlbertTokenizer
import nltk
# from bert_serving.client import BertClient
# bc = BertClient(ip='10.9.20.189')
from TargetNet import TargetModel
from TargetNet import TargetModelLoss
from utils import getLogger, readSet, getTrainLoader, getTestLoader, current_time, excutALBert, excutW2V, logout, passItem, saveParams

def main():
    src = '/home/xxx/project/QAData/'
    data_path = src + 'data/'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    tenoserBoardStr = ''
    # 创建参数列表：
    param = {}
    param['CUDA_VISIBLE_DEVICES'] = os.environ["CUDA_VISIBLE_DEVICES"]
    param['device'] = str(device)
    param['batch_size'] = 30
    param['learningrate'] = 0.01
    param['extracter_name'] = "CNNTextExtracterWithMHSelfAttion"
    # param['extracter_name'] = "CNNTextExtracterWithMHSelfAttion"
    param['extracter_kernel_num'] = 300
    param['extracter_kernel_size'] = 3
    param['extracter_vocab_size'] = 1000
    param['extracter_embed_dim'] = 300
    # param['extracter_embed_dim'] = 768
    param['extracter_padding'] = 0
    param['extracter_n_hidden'] = 150
    # param['extracter_n_hidden_2'] = 80
    param['extracter_out_dim'] = 400
    param['dropout'] = 0.5
    schedulerDict = {}
    schedulerDict['name'] = "StepLR"
    schedulerDict['optimizer'] = "SGD"
    schedulerDict['lr'] = param['learningrate']  # 不要修改此值，对上面的学习率修改
    schedulerDict['step_size'] = 300
    schedulerDict['gamma'] = 0.85
    param['optimizer']=schedulerDict
    params = [param]

    train_set, test_set = readSet(data_path)
    train_loader = getTrainLoader(train_set, param)
    test_loader = getTestLoader(test_set, param)  # 加载测试集，测试效果
    logger = getLogger(src)
    # bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # bertModel = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(device, non_blocking=True)
    bertTokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    bertModel = AlbertModel.from_pretrained('albert-base-v2').to(device, non_blocking=True)
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')  # 加载punkt句子分割器
    print("句子分割器模型加载完成")
    paramid = 0
    for i in range(len(params)):
        param = params[i]
        start = time.time()
        model = TargetModel.TarM(param, device).to(device, non_blocking=True)
        # summary(model, input_size=[(2000,768),(2000,768)],batch_size=-1)
        model._initialize_weights()
        tarloss = TargetModelLoss.TarLoss()
        # sourceNetwork的更新与TargetNetwork的优化器一致，初始值也相同
        # optimizer = torch.optim.Adam(model.parameters(), lr=param['learningrate'])
        # optimizer = torch.optim.SGD(model.parameters(), lr=param['learningrate'], weight_decay=1e-5)
        optimizer = torch.optim.SGD(model.parameters(), lr=param['learningrate'], weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=schedulerDict['step_size'], gamma=schedulerDict['gamma'])
        print("模型及优化器初始化完成")
        logstr = current_time()
        savestr = src + 'TargetNet/runs/' + logstr + tenoserBoardStr
        writer = SummaryWriter(savestr)  # GPU5
        param['board_str'] = logstr
        saveParams(params, savestr+"/"+str(i)+".json")
        epoch_num = 200
        train_step = 0
        test_step = 0
        test_correct = 0
        min_train_loss = 100000
        step_loss = []
        train_correct = 0
        train_loss = 0.0
        test_loss = 0
        test_min_loss = 100000
        # print("开始训练")
        model = model.train()
        for epoch in range(epoch_num):
            epoch_start = time.time()
            for [r_batch, q_text_batch, a_text_batch, q_id_batch, u_id_batch] in train_loader:
                step_time = time.time()
                # print("开始转换")
                # q_vec_batch, q_real_len = excutBert(q_text_batch, bertTokenizer, bertModel, sen_tokenizer)
                # a_vec_batch, a_real_len = excutBert(a_text_batch, bertTokenizer, bertModel, sen_tokenizer)
                # q_vec_batch, q_real_len = excutALBert(q_text_batch, bertTokenizer, bertModel, sen_tokenizer, device)
                # a_vec_batch, a_real_len = excutALBert(a_text_batch, bertTokenizer, bertModel, sen_tokenizer, device)
                q_vec_batch, q_real_len = excutW2V(q_text_batch)
                a_vec_batch, a_real_len = excutW2V(a_text_batch)
                # print("转向量用时：", time.time()- step_time)
                step_time0 = time.time()

                q_vec_batch = q_vec_batch.to(device, non_blocking=True)
                q_real_len = torch.Tensor(q_real_len) # 代表每个句子的长度
                a_vec_batch = a_vec_batch.to(device, non_blocking=True)
                a_real_len = torch.Tensor(a_real_len) # 代表每个句子的长度

                # print("转入gpu用时：", time.time()- step_time0)
                step_time1 = time.time()
                tar_drop_vec, RT = model.forward(q_vec_batch, a_vec_batch, q_real_len, a_real_len)
                tar_drop_vec = tar_drop_vec.cpu()
                RT = RT.cpu()
                # loss
                step_loss = tarloss(RT, r_batch)
                step_loss.backward()
                # print("正向及反向传播用时：", time.time()- step_time1)
                step_time2 = time.time()
                prob, cla = torch.max(RT, dim=1)
                # print(cla)
                step_correct = torch.sum(cla == r_batch).item()  # 单一标签适配
                train_correct += step_correct
                optimizer.step()  # 每反向传播一次，更新网络参数并清空梯度，每5个step判断一下模型是否有提升
                optimizer.zero_grad()
                train_step_acc = step_correct/(param['batch_size'])
                train_mean_loss = step_loss.detach().item()/len(r_batch)    # 防止内存泄漏，只使用值即可
                # print("计算loss和acc：", time.time()- step_time2)
                step_time3 = time.time()
                # passStr = passItem(len(train_set), (train_step+1)*(param['batch_size']))
                passStr = ""
                # logStr = 'train:'+str((train_step+1)*(param['batch_size']))+"/"+str(len(train_set))+'\t'+str(epoch)+'\t'+str(train_step)+'\t'+str(step_loss.item())[:6]+'\t'+str(train_step_acc)
                logStr = 'train:'+passStr+'\t'+str(epoch)+'\t'+str(train_step)+'\t'+str(step_loss.detach().item())[:6]+'\t'+str(train_step_acc)
                logout(logStr, logger)
                if train_mean_loss < min_train_loss:
                    save_str = savestr+'/trainTarModel_step_'+str(train_step)+',loss_'+str(step_loss.detach().item())[:6]
                    torch.save(model.state_dict(), save_str + '.pth')
                    torch.save(model, save_str + '.model')
                    min_train_loss = train_mean_loss
                if train_step % 5 == 0:
                    writer.add_scalar("train_loss", step_loss.detach().item(), train_step)
                    writer.add_scalar("train_acc", train_step_acc, train_step)
                train_step += 1
                # print("统计及保存模型用时：", time.time()- step_time2)
                scheduler.step()  # 每个epoch学习率动态变化
                del q_vec_batch
                del a_vec_batch
                torch.cuda.empty_cache()     # 清除缓存，用于显存占用随step递增的情况
                # print("其余耗时：", time.time()- step_time3 , "\n")



            # model = model.eval()
            # for [r_batch, q_text_batch, a_text_batch, q_id_batch, u_id_batch] in test_loader:
            #     # q_vec_batch = excutBert(q_text_batch, bertTokenizer, bertModel, sen_tokenizer).to(device, non_blocking=True)
            #     # a_vec_batch = excutBert(a_text_batch, bertTokenizer, bertModel, sen_tokenizer).to(device, non_blocking=True)
            #     q_vec_batch = excutALBert(q_text_batch, bertTokenizer, bertModel, sen_tokenizer, device).to(device, non_blocking=True)
            #     a_vec_batch = excutALBert(a_text_batch, bertTokenizer, bertModel, sen_tokenizer, device).to(device, non_blocking=True)
            #     # print("vec ok")
            #     # q_vec_batch = excutW2V(q_text_batch).to(device)
            #     # a_vec_batch = excutW2V(a_text_batch).to(device)
            #     tar_drop_vec, RT = model.forward(q_vec_batch, a_vec_batch)
            #     tar_drop_vec = tar_drop_vec.cpu()
            #     RT = RT.cpu()
            #     step_loss = tarloss(RT, r_batch)
            #     prob, cla = torch.max(RT, dim=1)
            #     test_step_correct = torch.sum(cla == r_batch).item()
            #     test_correct += test_step_correct
            #     test_step_acc = test_step_correct/param['batch_size']
            #     test_mean_loss = step_loss.item()/len(r_batch)
            #     logStr = 'test:'+str((test_step+1)*(param['batch_size']))+"/"+str(len(test_set)) + '\t' + str(epoch) + '\t' + str(test_step) + '\t' + str(step_loss.item())[:6] + '\t' + str(test_step_acc)
            #     logout(logStr, logger)
            #     if test_step_correct < test_min_loss:
            #         save_str = savestr +'/testTarModelEpoch_'+str(epoch)+',step_'+str(test_step)+',loss_'+str(step_loss.item())[:6]
            #         torch.save(model.state_dict(), save_str + '.pth')
            #         torch.save(model, save_str + '.model')
            #         test_min_loss = test_step_correct
            #     if test_step % 5 == 0:
            #         writer.add_scalar("test_loss", step_loss.item(), test_step)
            #         writer.add_scalar("test_acc", test_step_acc, test_step)
            #     test_step += 1
            epoch_end = time.time()
            print(epoch, "训练集loss：", train_loss, "测试集loss：", test_loss, "训练准确率：", train_correct / len(train_set), "测试集准确率：", test_correct / len(test_set), "用时：", str((epoch_end - epoch_start) / 60) + "分钟")
            logger.info('epoch' + '\t' + str(epoch) + '\t' + str(train_correct/len(train_set)) + '\t' + str(test_correct/len(test_set)))
        writer.close()

# Target网络的网格搜索调参+交叉验证
# 前四个月做训练集，循环为验证集
if __name__ == '__main__':
    main()
    

# 显存管理： https://zhuanlan.zhihu.com/p/138534708
