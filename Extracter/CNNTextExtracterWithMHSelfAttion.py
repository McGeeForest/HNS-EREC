import math

from torch import nn
import torch
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob, device):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.device = device
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)
        attention_probs_dropout_prob = hidden_dropout_prob
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        # print(x.size())
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # print("@@@", new_x_shape)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_data):
        input_tensor, lens = input_data
        # print(lens)
        # print("*********")
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        batch_size = input_tensor.size()
        # print(batch_size[0], type(batch_size[0]))
        max_len = int(lens.max().item()) #最大的句子长度，生成mask矩阵
        mask = torch.arange(lens.max().item())[None, :] < lens[:, None]
        mask = mask.unsqueeze(dim = 1) #[batch_size, 1, max_len]
        mask = mask.unsqueeze(dim = 1) #[batch_size, 1, max_len]
        mask = mask.expand(batch_size[0], self.num_attention_heads, max_len, max_len) #[batch_size, max_len, max_len]
        # print('\nmask is :', mask.size())
        #下面生成用来填充的矩阵
        padding_num = torch.ones_like(mask)
        padding_num = -2**31 * padding_num.float()
        padding_num = padding_num.to(self.device)
        mask = mask.to(self.device)


        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        #下面开始mask
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        # print(mask.size(), attention_scores.size(), padding_num.size())
        attention_scores = torch.where(mask, attention_scores, padding_num)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        # print(attention_probs.shape)
        # print(value_layer.shape)
        context_layer = torch.matmul(attention_probs, value_layer)

        # 多头注意力部分
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        # print(hidden_states.shape)
        # print(input_tensor.shape)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        del context_layer, attention_probs, attention_scores, mixed_value_layer, mixed_key_layer, mask, query_layer, key_layer, value_layer
        return hidden_states


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

        SelfAttentionLayer = nn.Sequential()
        SelfAttentionLayer.add_module('ATTENTION1', SelfAttention(4, 768, 768, 0.1, device))
        # SelfAttentionLayer.add_module('ATTENTION1', SelfAttention(4, 300, 300, 0.1, device))
        self.SelfAttentionLayer = SelfAttentionLayer


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
        CNN5.add_module('POOL5', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.CNN5 = CNN5

        ACTIVE = nn.Sequential()
        ACTIVE.add_module('ACTIVE2', nn.Tanh())
        # ACTIVE.add_module('RELU5', nn.ReLU(inplace=True))  # 输出：（batch*kernel_num*1*1）
        self.ACTIVE = ACTIVE


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
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
    def forward(self, vec_batch, seq_len):
        # print(vec_batch.size(), seq_len)
        vec_batch = self.SelfAttentionLayer([vec_batch, seq_len])
        x = vec_batch.unsqueeze(1)
        del vec_batch

        x = self.CNN(x)
        x = x.permute(0, 3, 2, 1)

        # x = self.CNN2(x)
        # x = x.permute(0, 3, 2, 1)

        x = self.CNN5(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = nn.BatchNorm1d(x.size()[1]).to(self.device)(x)      # 卷积层和激活层之间添加batch norm，进行数据归一化
        x = self.ACTIVE(x)
        # print(x.size())

        x = self.MFC(x)
        return x
