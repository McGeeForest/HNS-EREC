from annoy import AnnoyIndex
import random
import pandas as pd
from tqdm import tqdm
q_vec_bert = pd.read_excel("q_vec_bert.xlsx")
q_list = list(pd.read_csv("qa_set_u21.csv")['q_id'])

f = 24      # 向量维度

# 构建索引
t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
i = 0
q_save_list = []
for (index, item) in tqdm(q_vec_bert.iterrows()):
    if item['q_id'] in q_list:
        t.add_item(i, eval(item['q_vec']))
        i += 1
        q_save_list.append(item['q_id'])
print("building")
t.build(1000)  # 10 trees
print("saving")
t.save('qa_set_u21.ann')
print("over", i, "项目")
# ...


u = AnnoyIndex(f, 'angular')
u.load('qa_set_u21.ann') # super fast, will just mmap the file
sim_q_list = []
i = 0
q_save_list = []
for (index, item) in tqdm(q_vec_bert.iterrows()):
    if item['q_id'] in q_list:
        q_save_list.append(item['q_id'])

for (index, item) in tqdm(q_vec_bert.iterrows()):
    if item['q_id'] in q_list:
        sim_index = u.get_nns_by_item(i, 100)
        sim_q = []
        for index in sim_index:
            q_id = q_save_list[index]
            sim_q.append(q_id)
        i += 1
        sim_q_list.append(sim_q)

sim_q = pd.DataFrame()
sim_q['q_id'] = q_save_list
sim_q['sim_q'] = sim_q_list
sim_q.to_csv("sim_q_u21.csv")

