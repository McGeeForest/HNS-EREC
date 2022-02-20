import sys
sys.path.append("/home/student/xxx/project/QAData/")
# sys.path.append("D:/OneDriveEdu/file/project/transNetsQANew/src3/")
from torch import nn
import torch
from Extracter import CNNTextExtracterWithMHSelfAttion as TextExtracter
# from Extracter import CNNTextExtracter as TextExtracter
# from Extracter import GRUTextExtracter as TextExtracter
# TargetModel
class TarM (nn.Module):
    def __init__(self, param, device):
        super(TarM, self).__init__()
        # 参数分解
        print(param)
        dropout = param['dropout']
        extracter_out_dim = param['extracter_out_dim']

        # MFC相关参数
        # MFC1_hidden = param['MFC1_hidden']
        # MFC1_out = param['MFC1_out']

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

        # # MFC1
        # MFC1 = nn.Sequential()
        # MFC1.add_module('linear1', nn.Linear(extracter_out_dim*2, MFC1_out))
        # # MFC1.add_module('linear2', nn.Linear(MFC1_hidden, MFC1_out))
        # self.MFC1 = MFC1

        # 激活+DropLayer
        dropoutlayer = nn.Sequential()
        # dropoutlayer.add_module('ACTIVE1', nn.ReLU(inplace=True))
        # dropoutlayer.add_module('ACTIVE1', nn.Tanh())
        dropoutlayer.add_module('DROPOUT1', nn.Dropout(dropout))
        self.dropoutlayer = dropoutlayer

        # MFC2+softmax
        MFC2 = nn.Sequential()
        MFC2.add_module('linear1', nn.Linear(extracter_out_dim*2, 4))
        MFC2.add_module('ACTIVE2', nn.Tanh())
        # MFC2.add_module('softmax', nn.Softmax(dim=1))
        self.MFC2 = MFC2

#  初始化权值的方法，线性层使用xavier
    def _initialize_weights(self):
        # print(self.modules())
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                # print(m.weight)

    def forward(self, q_vec_batch, a_vec_batch, q_real_len, a_real_len):
        q_represent = self.QE.forward(q_vec_batch, q_real_len)
        a_represent = self.AE.forward(a_vec_batch, a_real_len)
        # print(q_represent.size())
        # print(a_represent.size())
        qa_represent = torch.cat([q_represent, a_represent], 1)
        del q_represent, a_represent
        # print(qa_represent.size())
        # qa_represent = self.MFC1(qa_represent)
        tar_drop_vec = self.dropoutlayer(qa_represent)
        del qa_represent
        RT = self.MFC2(tar_drop_vec)
        return tar_drop_vec, RT

    def qExtracter(self, q_vec_batch):
        q_represent = self.QE.forward(q_vec_batch)
        return q_represent
    
    def aExtracter(self, a_vec_batch):
        a_represent = self.AE.forward(a_vec_batch)
        return a_represent

    def getClass(self, tar_drop_vec):
        R_pred = self.MFC(tar_drop_vec)
        return R_pred
