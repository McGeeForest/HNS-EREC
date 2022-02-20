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
        
        x = x.sum(1)      # 如果生成带有权重的词向量序列，作为文章的表示，则不需要该步骤。如果需要直接压缩得到文章表示，则需要该步骤。
        return x


# 文本向量表示生成器
class Extracter(nn.Module):
    def __init__(self, param):
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

        # （2）MFC相关参数
        n_hidden = param['extracter_n_hidden']
        # n_hidden_2 = param['extracter_n_hidden_2']
        out_dim = param['extracter_out_dim']



        # GRULayer = nn.Sequential()
        # GRULayer.add_module('GRU1', nn.GRU(input_size=768, hidden_size=400, num_layers=5, bidirectional=True, batch_first=True))
        # self.GRULayer = GRULayer
        
        LSTMLayer = nn.Sequential()
        LSTMLayer.add_module('LSTM1', nn.LSTM(input_size=embed_dim, hidden_size=256, num_layers=3, bidirectional=True, batch_first=True))
        self.LSTMLayer = LSTMLayer

        SelfAttentionLayer = nn.Sequential()
        SelfAttentionLayer.add_module('SELF-ATT1', SelfAttention(512, 256))
        self.SelfAttentionLayer = SelfAttentionLayer

        # DropLayer = nn.Sequential()
        # DropLayer.add_module('DROPOUT', nn.Dropout(0.1))
        # self.DroupLayer = DropLayer

        # 一个多层全连接
        MFC = nn.Sequential()
        # MFC.add_module('linear1', nn.Linear(kernel_num, n_hidden))
        MFC.add_module('linear1', nn.Linear(512, n_hidden))
        MFC.add_module('linear2', nn.Linear(n_hidden, n_hidden))
        MFC.add_module('linear3', nn.Linear(n_hidden, out_dim))
        self.MFC = MFC

#  初始化权值的方法，线性层使用xavier
    def _initialize_weights(self):
        # print(self.modules())
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, vec_batch):
        x, (h_n, c_n) = self.LSTMLayer(vec_batch)
        # print(c_n.size())
        del vec_batch
        x = self.SelfAttentionLayer(x)
        x = self.MFC(x)
        return x
