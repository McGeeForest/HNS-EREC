from torch import nn


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
        n_hidden_1 = param['extracter_n_hidden_1']
        n_hidden_2 = param['extracter_n_hidden_2']
        out_dim = param['extracter_out_dim']

        # 两层卷积
        CNN = nn.Sequential()
        CNN.add_module('CONV1', nn.Conv2d(in_channels=ci, out_channels=layer_kernel_num, kernel_size=(kernel_size, embed_dim)))  # 输出：（batch*layer_kernel_num*n*embed_dim）
        # CNN1.add_module('POOL1', nn.AdaptiveAvgPool2d(output_size=(1,1)))
        # CNN1.add_module('RELU1', nn.ReLU(True))
        self.CNN = CNN

        CNN2 = nn.Sequential()
        CNN2.add_module('CONV2', nn.Conv2d(in_channels=ci, out_channels=kernel_num, kernel_size=(layer_kernel_num, kernel_size)))
        CNN2.add_module('POOL2', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        CNN2.add_module('RELU2', nn.ReLU(True))  # 输出：（batch*kernel_num*1*1）
        self.CNN2 = CNN2

        # 一个多层全连接
        MFC = nn.Sequential()
        MFC.add_module('linear1', nn.Linear(kernel_num, n_hidden_1))
        MFC.add_module('linear2', nn.Linear(n_hidden_1, n_hidden_2))
        MFC.add_module('linear3', nn.Linear(n_hidden_2, out_dim))
        self.MFC = MFC

    def forward(self, vec_batch):
        vec_batch = vec_batch.unsqueeze(1)
        x = self.CNN(vec_batch)

        x = x.squeeze(-1)
        x = x.unsqueeze(1)
        # print(x.size())
        x = self.CNN2(x)
        # print(x.size())
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = self.MFC(x)
        return x
