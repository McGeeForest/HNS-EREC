
import os
import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
print(torch.__version__)
if torch.cuda.is_available():
    print('available')
# torch.cuda.set_device(0)
# torch.cuda.set_device(1)

print("GPU数量：", torch.cuda.device_count())
print("GPU名字：", torch.cuda.get_device_name(0))

print("Active GPU：", torch.cuda.current_device())
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

testensor = torch.FloatTensor([1.0, 2.0, 3.0]).cuda()
print(testensor)
# 显存管理： https://zhuanlan.zhihu.com/p/138534708

# from transformers import AlbertModel,AlbertTokenizer
# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# model = AlbertModel.from_pretrained('albert-base-v2')
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute"))[:3].unsqueeze(0)
# print("input_ids:", input_ids)
# model(input_ids).last_hidden_state
# print(model(input_ids).last_hidden_state.squeeze(0).size())

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# import torch
# import torch.nn as nn
# import numpy as np



# class dot_attention(nn.Module):
#     """ 点积注意力机制"""

#     def __init__(self, attention_dropout=0.0):
#         super(dot_attention, self).__init__()
#         self.dropout = nn.Dropout(attention_dropout)
#         self.softmax = nn.Softmax(dim=2)

#     def forward(self, q, k, v, scale=None, attn_mask=None):
#         """
#         前向传播
#         :param q:
#         :param k:
#         :param v:
#         :param scale:
#         :param attn_mask:
#         :return: 上下文张量和attention张量。
#         """
#         attention = torch.bmm(q, k.transpose(1, 2))
#         if scale:
#             attention = attention * scale        # 是否设置缩放
#         if attn_mask:
#             attention = attention.masked_fill(attn_mask, -np.inf)     # 给需要mask的地方设置一个负无穷。
#         # 计算softmax
#         attention = self.softmax(attention)
#         # 添加dropout
#         attention = self.dropout(attention)
#         # 和v做点积。
#         context = torch.bmm(attention, v)
#         return context, attention


# class MultiHeadAttention(nn.Module):
#     """ 多头自注意力"""
#     def __init__(self, model_dim=768, num_heads=4, dropout=0.0):
#         super(MultiHeadAttention, self).__init__()

#         self.dim_per_head = model_dim//num_heads   # 每个头的维度
#         self.num_heads = num_heads
#         self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
#         self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
#         self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

#         self.dot_product_attention = dot_attention(dropout)

#         self.linear_final = nn.Linear(model_dim, model_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(model_dim)         # LayerNorm 归一化。

#     def forward(self, key, value, query, attn_mask=None):
#         # 残差连接
#         residual = query

#         dim_per_head = self.dim_per_head
#         num_heads = self.num_heads
#         batch_size = key.size(0)

#         # 线性映射。
#         key = self.linear_k(key)
#         value = self.linear_v(value)
#         query = self.linear_q(query)

#         # 按照头进行分割
#         key = key.view(batch_size * num_heads, -1, dim_per_head)
#         value = value.view(batch_size * num_heads, -1, dim_per_head)
#         query = query.view(batch_size * num_heads, -1, dim_per_head)

#         if attn_mask:
#             attn_mask = attn_mask.repeat(num_heads, 1, 1)

#         # 缩放点击注意力机制
#         scale = (key.size(-1) // num_heads) ** -0.5
#         context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

#         # 进行头合并 concat heads
#         context = context.view(batch_size, -1, dim_per_head * num_heads)
#         print(context.size())

#         # 进行线性映射
#         output = self.linear_final(context)

#         # dropout
#         output = self.dropout(output)

#         # 添加残差层和正则化层。
#         output = self.layer_norm(residual + output)

#         return output, attention


# if __name__ == '__main__':
#     q = torch.ones((1, 6, 768))
#     k = q
#     v = q
#     mutil_head_attention = MultiHeadAttention()
#     output, attention = mutil_head_attention(q, k, v)
#     # print("context:", output.size(), output)
#     # print("attention:", attention.size(), attention)
