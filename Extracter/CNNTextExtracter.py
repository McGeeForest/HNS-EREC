from torch import nn
import torch
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(input_size, hidden_size, True)
        self.u = nn.Linear(hidden_size, 1)

    def forward(self, x):
        u = torch.tanh(self.W(x))
        a = F.softmax(self.u(u), dim=1)
        x = a.mul(x)        # 将权重融入到词表示中
        
        # x = x.sum(1)      # 如果生成带有权重的词向量序列，作为文章的表示，则不需要该步骤。如果需要直接压缩得到文章表示，则需要该步骤。
        return x


# 文本向量表示生成器
class Extracter(nn.Module):
    def __init__(self, param, device):
        super(Extracter, self).__init__()
        # 参数组装
        # （1）CNN相关参数
        ci = 1  # RGB的通道数，文本的话相当于灰度图只一个通道
        kernel_num = param['extracter_kernel_num']  # 卷积核数量，输出向量维度
        layer_kernel_num = int(kernel_num * 2)
        kernel_size = param['extracter_kernel_size']  # 卷积核尺寸
        vocab_size = param['extracter_vocab_size']  # 文本长度n，word-level
        embed_dim = param['extracter_embed_dim']  # 输入词嵌入的维度
        padding = param['extracter_padding']
        self.device = device

        # （2）MFC相关参数
        n_hidden = param['extracter_n_hidden']
        # n_hidden_2 = param['extracter_n_hidden_2']
        out_dim = param['extracter_out_dim']



        # 两层卷积
        CNN = nn.Sequential()
        CNN.add_module('CONV1', nn.Conv2d(in_channels=ci, out_channels=kernel_num, kernel_size=(kernel_size, embed_dim), padding=padding))  # 输出：（batch*layer_kernel_num*n*embed_dim）
        # CNN1.add_module('POOL1', nn.AdaptiveAvgPool2d(output_size=(1,1)))
        # CNN1.add_module('RELU1', nn.ReLU(True))
        self.CNN = CNN

        CNN2 = nn.Sequential()
        CNN2.add_module('CONV2', nn.Conv2d(in_channels=ci, out_channels=kernel_num, kernel_size=(kernel_size, kernel_num), padding=padding))
        # CNN2.add_module('POOL2', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        # CNN2.add_module('RELU2', nn.ReLU(True))  # 输出：（batch*kernel_num*1*1）
        self.CNN2 = CNN2

        CNN3 = nn.Sequential()
        CNN3.add_module('CONV3', nn.Conv2d(in_channels=ci, out_channels=kernel_num, kernel_size=(kernel_size, kernel_num), padding=padding))
        # CNN3.add_module('POOL3', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        # CNN3.add_module('RELU3', nn.ReLU(True))  # 输出：（batch*kernel_num*1*1）
        self.CNN3 = CNN3

        CNN4 = nn.Sequential()
        CNN4.add_module('CONV4', nn.Conv2d(in_channels=ci, out_channels=kernel_num, kernel_size=(kernel_size, kernel_num), padding=padding))
        # CNN4.add_module('POOL3', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        # CNN4.add_module('RELU3', nn.ReLU(True))  # 输出：（batch*kernel_num*1*1）
        self.CNN4 = CNN4

        
        CNN5 = nn.Sequential()
        CNN5.add_module('CONV5', nn.Conv2d(in_channels=ci, out_channels=kernel_num, kernel_size=(kernel_size, kernel_num), padding=padding))
        CNN5.add_module('POOL3', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        CNN5.add_module('RELU3', nn.ReLU(inplace=True))  # 输出：（batch*kernel_num*1*1）
        self.CNN5 = CNN5


        # 一个多层全连接
        MFC = nn.Sequential()
        MFC.add_module('linear1', nn.Linear(kernel_num, out_dim))
        # MFC.add_module('linear1', nn.Linear(512, n_hidden))
        # MFC.add_module('linear2', nn.Linear(n_hidden, n_hidden))
        # MFC.add_module('linear3', nn.Linear(n_hidden, out_dim))
        self.MFC = MFC

#  初始化权值的方法，线性层使用xavier
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, vec_batch):
        x = vec_batch.unsqueeze(1)
        del vec_batch
        x = self.CNN(x)
        x = x.permute(0, 3, 2, 1)

        # x = self.CNN2(x)
        # x = x.permute(0, 3, 2, 1)

        # x = self.CNN3(x)
        # x = x.permute(0, 3, 2, 1)

        # x = self.CNN4(x)        
        # x = x.permute(0, 3, 2, 1)

        x = self.CNN5(x)
        # print(x.size())
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = nn.BatchNorm1d(x.size()[1]).to(self.device)(x)
        x = self.MFC(x)
        return x
