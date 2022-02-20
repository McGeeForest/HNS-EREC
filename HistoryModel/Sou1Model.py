import sys
sys.path.append("/home/student/xxx/project/transNetsQANew/src3/")
from torch import nn
import torch
from Extracter import CNNTextExtracterWithMHSelfAttion as TextExtracter
# Sou1Model
class Sou1M (nn.Module):
    def __init__(self, param, device):
        super(Sou1M, self).__init__()
        # MFC相关参数
        MFC1_hidden = param['MFC1_hidden']
        MFC1_out = param['MFC1_out']
        extracter_out_dim = param['extracter_out_dim']
        # extracter 相关参数
        self.eParam = {}
        self.eParam['extracter_out_dim'] = param['extracter_out_dim']
        self.eParam['extracter_kernel_num'] = param['extracter_kernel_num']
        self.eParam['extracter_kernel_size'] = param['extracter_kernel_size']
        self.eParam['extracter_vocab_size'] = param['extracter_vocab_size']
        self.eParam['extracter_embed_dim'] = param['extracter_embed_dim']
        self.eParam['extracter_padding'] = param['extracter_padding']
        self.eParam['extracter_n_hidden'] = param['extracter_n_hidden']
        # self.eParam['extracter_n_hidden_2'] = param['extracter_n_hidden_2']
        self.eParam['extracter_out_dim'] = param['extracter_out_dim']
        # QE 问题提取器
        self.QE = TextExtracter.Extracter(self.eParam, device).to(device)
        self.QE._initialize_weights()
        # AE 回答提取器
        self.AE = TextExtracter.Extracter(self.eParam, device).to(device)
        self.AE._initialize_weights()

        # 激活+DropLayer
        dropoutlayer = nn.Sequential()
        # dropoutlayer.add_module('ACTIVE1', nn.ReLU(inplace=True))
        # dropoutlayer.add_module('ACTIVE1', nn.Tanh())
        dropoutlayer.add_module('DROPOUT1', nn.Dropout(param['dropout']))
        self.dropoutlayer = dropoutlayer

        # MFC1
        MFC1 = nn.Sequential()
        MFC1.add_module('linear1', nn.Linear(extracter_out_dim*2, MFC1_hidden))
        MFC1.add_module('linear2', nn.Linear(MFC1_hidden, MFC1_out))
        self.MFC1 = MFC1

    def forward(self, q_vec_batch, u_his_vec_batch, q_real_len, u_his_real_len):
        q_represent = self.QE(q_vec_batch, q_real_len)
        u_represent = self.AE(u_his_vec_batch, u_his_real_len)
        qu_represent = torch.cat([q_represent, u_represent], 1)
        qu_represent = self.dropoutlayer(qu_represent)
        RS = self.MFC1(qu_represent)
        return qu_represent, RS
