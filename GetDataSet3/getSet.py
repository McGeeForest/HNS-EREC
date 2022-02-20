import pymongo, joblib, re
from tqdm import tqdm
import numpy as np
import pandas as pd

# 用户Ux<tab>问题Qy<tab>回答Axy<tab>得分Rxy的文件
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["stackoverflow_qa"]
mycol = mydb["qa_2017_copy1"]
# 对一串文本进行粗处理：
# 去掉换行 小写替换大写 去掉html标签 连续的空格替换为一个空格
# 返回这个文本
def textPreProcess(content):
    content = str(content).replace("\n", " ").lower()
    content = "".join(re.sub(u'\<.*?\>', '', content))
    content = ' '.join(content.split())
    return content

data_all = mycol.find({},
               {'q_question_id': 1, 'q_title': 1, 'q_body': 1, 'answers': 1, 'q_creation_date': 1, 'q_is_answered': 1,
                'q_accepted_answer_id': 1})
qa_set = pd.DataFrame()
for curr_question in tqdm(data_all):
    curr_question = dict(curr_question)
    q_id = str(curr_question['q_question_id'])  # 当前问题的id
    q_time = curr_question['q_creation_date']   # 问题创建时间
    answers = curr_question['answers']          # 回答的答案
    q_is_answered = curr_question['q_is_answered']  # 问题是否被回答
    q_accepted_answer_id = curr_question['q_accepted_answer_id']    # 问题的被接受的回答id
    q_content = str(curr_question['q_title'] + curr_question['q_body'])  # 当前问题的内容
    q_content = textPreProcess(q_content)
    if q_accepted_answer_id != None:
        if answers==None:
            continue
        if len(answers) > 0:
            answer_count = len(answers)
            for answer in answers:
                if answer['owner']['user_type'] != 'registered':
                    continue
                u_id = answer['owner']['user_id']
                a_time = answer['creation_date']
                a_id = answer['answer_id']
                a_content = str(answer['body'])  # 当前回答的内容
                a_content = textPreProcess(a_content)  # 去掉换行 去掉<>标签 转换成小写
                score = answer['score']  # 当前回答的得分情况
                if a_id == q_accepted_answer_id:
                    r = [1.0, 0.0, 0.0, 0.0]
                else:
                    r = [0.0, 1.0, 0.0, 0.0]
                item = pd.DataFrame()
                item['q_id'] = [q_id]
                item['q_time'] = [q_time]
                item['q_text'] = [q_content]
                item['u_id'] = [u_id]
                item['a_id'] = [a_id]
                item['a_text'] = [a_content]
                item['a_time'] = [a_time]
                item['r'] = [r]
                item['score'] = [score]
                # print(item)
                qa_set = pd.concat([qa_set, item])
qa_set = qa_set.dropna(axis=0, how='any')
qa_set.to_csv("qa_set_ori.csv")     # 存在回答数量较少的用户，后面根据用户回答数量进行筛选选出4000左右的用户






from tqdm import tqdm

qa_set = pd.read_csv("qa_set_ori.csv")
u_id_group = qa_set.groupby("u_id")
print("原始问答数据读取完成...")
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 0:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >0 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 1:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >1 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 2:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >2 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 3:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >3 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     # print(user)
#     if len(user[1]) > 4:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >4 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 5:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >5 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 6:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >6 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
# # new_qa.to_csv('qa_set_u7.csv')
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 7:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >7 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 8:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >8 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 9:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >9 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 10:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >10 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 11:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >11 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 12:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >12 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 13:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >13 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 14:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >14 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 15:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >15 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 16:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >16 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 17:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >17 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 18:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >18 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 19:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >19 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 20:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >20 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
# new_qa.to_csv('qa_set_u20.csv')

user_count = 0
qa_count = 0
new_qa = pd.DataFrame()
for user in tqdm(u_id_group):
    if len(user[1]) > 21:
        user_count += 1
        qa_count += len(user[1])
        new_qa = pd.concat([new_qa, user[1]])
print(" >21 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
new_qa.to_csv('qa_set_u21.csv')
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 22:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >22 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
# # new_qa.to_csv('qa_set_u22.csv')
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 23:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >23 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
# # new_qa.to_csv('qa_set_u23.csv')
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 24:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >24 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
# # new_qa.to_csv('qa_set_u24.csv')
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 25:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >25 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
# # new_qa.to_csv('qa_set_u25.csv')
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 26:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >26 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
# # new_qa.to_csv('qa_set_u26.csv')
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 27:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >27 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
# # new_qa.to_csv('qa_set_u27.csv')
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 28:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >28 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
# # new_qa.to_csv('qa_set_u28.csv')
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 29:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >29 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
# # new_qa.to_csv('qa_set_u29.csv')
#
# user_count = 0
# qa_count = 0
# new_qa = pd.DataFrame()
# for user in tqdm(u_id_group):
#     if len(user[1]) > 30:
#         user_count += 1
#         qa_count += len(user[1])
#         new_qa = pd.concat([new_qa, user[1]])
# print(" >30 用户数量：",user_count,"  回答数量：", qa_count, "  问题数量：", len(new_qa.groupby("q_id")))
# # new_qa.to_csv('qa_set_u30.csv')
