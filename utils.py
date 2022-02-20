import pickle
import sys, time, requests, json, logging

sys.path.append("D:/OneDriveEdu/file/project2/QAData/TargetNet/")
sys.path.append("D:/OneDriveEdu/file/project2/QAData/SourceNet/")
sys.path.append("D:/OneDriveEdu/file/project2/QAData/")
import pandas as pd
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from nltk import word_tokenize
from nltk.corpus import stopwords
import random


def current_time():
    return time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())


def coll(batch):
    r_batch = []
    a_text_batch = []
    q_text_batch = []
    q_id_batch = []
    u_id_batch = []
    for r, a_text, q_text, q_id, u_id in batch:
        if len(a_text) == 0:
            a_text = "there is null text."
        if len(q_text) == 0:
            q_text = "there is null text."
        # r_batch.append(torch.FloatTensor([eval(r)]))      # 适配onehot编码的标签
        r_batch.append(torch.FloatTensor([float(r)]))  # 适配未onehot的标签
        a_text_batch.append(a_text)
        q_text_batch.append(q_text)
        q_id_batch.append(q_id)
        u_id_batch.append(u_id)
    r_batch = torch.cat(r_batch, 0)
    return [r_batch, q_text_batch, a_text_batch, q_id_batch, u_id_batch]


def getVecBert(sentence, tokenizer, model, device):  # 取最后四层隐含层输出之和 作为词向量
    token_vecs_sum = []
    marked_sentence = "[CLS] " + sentence + " [SEP]"
    tokenized_sentence = tokenizer.tokenize(marked_sentence)[:512]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)
    segments_ids = [1] * len(tokenized_sentence)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device, non_blocking=True)
    segments_tensors = torch.tensor([segments_ids]).to(device, non_blocking=True)
    with torch.no_grad():  # 不生成计算图，不需要反馈，这样速度更快
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]  # 输出隐藏层
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec.cpu().numpy().tolist())
    return token_vecs_sum


def getVecALBert(sentence, tokenizer, model, device):
    token_vecs = []
    with torch.no_grad():  # 不生成计算图，不需要反馈，这样速度更快
        input_ids = torch.tensor(tokenizer.encode(sentence))[:512].unsqueeze(0).to(device, non_blocking=True)
        # last_hidden_state = model(input_ids).last_hidden_state.squeeze(0)
        out = model(input_ids)[0].squeeze(0)
    # return last_hidden_state.cpu().numpy().tolist()
    return out.cpu().numpy().tolist()


def getVecW2V(sentence, model):  # 取w2v 作为词向量
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(sentence)
    wordsVec = []
    count = 0
    for w in words:
        if w not in stopWords:
            try:
                count += 1
                wordsVec.append(requests.get('http://127.0.0.1:5000/getVecBy?word=' + w))
                # wordsVec.append(model[w].tolist())
            except KeyError:
                pass
    # print(count)
    if len(wordsVec) == 0:
        wordsVec = getVecW2V("there is null text.", model)
    return wordsVec


def do(args):
    text, sen_tokenizer, tokenizer, model = args
    item_vec = []
    text = text.lower()  # 将所有大写字母转换为小写字母
    sentences = sen_tokenizer.tokenize(text)  # 对句子进行分割
    for sentence in sentences:
        item_vec = item_vec + getVecBert(sentence, tokenizer, model)
    return item_vec


def excutBert(texts, tokenizer, model, sen_tokenizer, device):
    res_vec = []
    real_len = []
    for text in texts:
        item_vec = []
        text = text.lower()  # 将所有大写字母转换为小写字母
        sentences = sen_tokenizer.tokenize(text)  # 对句子进行分割
        for sentence in sentences:
            item_vec = item_vec + getVecBert(sentence, tokenizer, model, device)
        if (len(item_vec) == 0):
            res_vec.append([0.0] * 768)
        else:
            res_vec.append(item_vec)
        real_len.append(len(res_vec))
    res_vec = pad_sequence([torch.FloatTensor(item) for item in res_vec], batch_first=True)
    return res_vec, real_len


def excutALBert(texts, tokenizer, model, sen_tokenizer, device):
    res_vec = []
    seq_len = []
    for text in texts:
        text = text.lower()  # 将所有大写字母转换为小写字母
        item_vec = getVecALBert(text, tokenizer, model, device)
        # item_vec = bc(text)
        res_vec.append(item_vec)
        seq_len.append(len(item_vec))
    res_vec = pad_sequence([torch.FloatTensor(item) for item in res_vec], batch_first=True)
    return res_vec, seq_len




from gensim.models import KeyedVectors
w2vModel = KeyedVectors.load_word2vec_format('/home/xxx/gensim-data/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)
vocabs = w2vModel.key_to_index.keys()     # w2v的词汇表
stopWords = set(stopwords.words('english'))
def excutW2V(texts):
    # print(texts)
    # print("已经收到文本")
    t1 = time.time()
    res_vec = []
    real_len = []
    for text in texts:
        item_vec = []
        text = text.lower()  # ***将所有大写字母转换为小写字母***
        item_vec = [w2vModel[w].tolist() for w in word_tokenize(text) if (w not in stopWords) and (w in vocabs) ]
        # sentences = sen_tokenizer.tokenize(text)  # 对句子进行分割
        # for sentence in sentences:
        #     item_vec = item_vec + getVecW2V(sentence)
        #     res_vec.append(item_vec)
        if(len(item_vec)==0):
            item_vec = [w2vModel[w].tolist() for w in word_tokenize("there is null text.") if (w not in stopWords) and (w in vocabs) ]
        res_vec.append(item_vec)
        real_len.append(len(item_vec))
    res_vec = pad_sequence([torch.FloatTensor(item) for item in res_vec], batch_first=True).numpy().tolist()
    res_data = {'res_vec': res_vec, 'real_len': real_len}



    # res_data = eval(requests.post('http://localhost:5002/getVecSeqBy', data_json).text)
    vec_text = res_data['res_vec']
    real_len = list(res_data['real_len'])
    vec_text = torch.FloatTensor(vec_text)
    return vec_text, real_len


class loadDataSet(data.Dataset):
    def __init__(self, train_set):
        # self.r = list(train_set['r'])         # 两种标签方式
        self.r = list(train_set['label'])
        # train_x['distance'] = [None for item in range(len(list(train_x['q_id'])))]
        self.a_text = [str(item) for item in list(train_set['a_text'])]
        self.q_text = [str(item) for item in list(train_set['q_text'])]
        self.q_id = list(train_set['q_id'])
        self.u_id = list(train_set['u_id'])
        # self.distance = list(train_x['distance'])
        self.data = []
        for i in range(len(self.a_text)):
            self.data.append([self.r[i], self.a_text[i], self.q_text[i], self.q_id[i], self.u_id[i]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class loadDataSet_Sou(data.Dataset):
    def __init__(self, train_set):
        self.label = list(train_set['r'])
        self.r=[]
        
        # train_x['distance'] = [None for item in range(len(list(train_x['q_id'])))]
        self.a_text = [str(item) for item in list(train_set['a_text'])]
        self.q_text = [str(item) for item in list(train_set['q_text'])]
        self.q_id = list(train_set['q_id'])
        self.u_id = list(train_set['u_id'])
        # self.distance = list(train_x['distance'])
        self.data = []
        for i in range(len(self.a_text)):
            # print(eval(self.label[i]))
            if eval(self.label[i]) == [1.0, 0.0,0.0,0.0]:
                # print("1")
                self.r.append(0.0)
            else:
                # print("2")
                self.r.append(1.0)
            # 用户历史只截取最后30条，超长弃用
            u_his_list = list(train_set[train_set['u_id'] == self.u_id[i]]['q_text'])
            if(len(u_his_list)>100):
                u_his_list = u_his_list[-30:0]
            self.u_his_list = ".".join(u_his_list)
            self.data.append([self.r[i], self.q_text[i], self.a_text[i], self.q_id[i], self.u_id[i], self.u_his_list])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class loadDataSet_Eval(data.Dataset):
    def __init__(self, his_set,eval_set):
        # his_set.to_csv("his.csv")
        q_list = list(set(list(eval_set.q_id)))
        u_list = set(list(eval_set.u_id))
        self.label = list(his_set['r'])
        self.r=[]
        
        self.a_text = [str(item) for item in list(his_set['a_text'])]
        self.q_text = [str(item) for item in list(his_set['q_text'])]
        self.q_id = list(his_set['q_id'])
        self.u_id = list(his_set['u_id'])
        self.score = list(his_set['score'])
        self.u_his = []
        # self.distance = list(train_x['distance'])
        self.data = []
        for i in range(len(self.a_text)):
            # print(eval(self.label[i]))
            if eval(self.label[i]) == [1.0, 0.0,0.0,0.0]:
                # print("1")
                self.r.append(0.0)
            else:
                # print("2")
                self.r.append(1.0)
            # 用户历史只截取最后100条，超长弃用
            u_his_list = list(his_set[his_set['u_id'] == int(self.u_id[i])]['q_text'])[:100]
            u_his_str = ".".join(u_his_list)
            # 如果该问题是用作进行指标评价的问题，则保留
            if self.q_id[i] in q_list:
                # print(self.q_id[i],"\t", self.u_id[i])
                self.data.append([self.r[i], self.q_text[i], self.q_id[i], self.u_id[i], u_his_str, self.score[i]])

        ##### self.data 是没有扩充问答数据的items ####
        self.data_pd = pd.DataFrame()
        pd_r = []
        pd_q_text = []
        pd_q_id = []
        pd_u_id = []
        pd_u_his_str = []
        pd_score = []
        for item in self.data:
            self.data_pd["r"] = pd_r.append(item[0])
            self.data_pd["q_text"] = pd_q_text.append(item[1])
            self.data_pd["q_id"] = pd_q_id.append(item[2])
            self.data_pd["u_id"] = pd_u_id.append(item[3])
            self.data_pd["u_his_str"] = pd_u_his_str.append(item[4])
            self.data_pd["score"] = pd_score.append(item[5])
        self.data_pd["r"] = pd_r
        self.data_pd["q_text"] = pd_q_text
        self.data_pd["q_id"] = pd_q_id
        self.data_pd["u_id"] = pd_u_id
        self.data_pd["u_his_str"] = pd_u_his_str
        self.data_pd["score"] = pd_score
        real_u_set_list = ["" for i in range(len(pd_score))]
        self.data_pd["real_u_set"] = real_u_set_list


        # 给数据增加真实回答列表
        for index, item in self.data_pd.groupby("q_id"):
            self.data_pd.loc[(self.data_pd.q_id == list(item["q_id"])[0]), "real_u_set"] = str(set(list(item["u_id"]))).replace("{", "").replace("}", "").replace(" ", "")
        real_u_set_list = list(self.data_pd["real_u_set"])

        for index, item in self.data_pd.groupby("q_id"):
            uers = len(item)
            if uers < 55:  # 如果回答用户不够55，则随机填充到55人
                random_pool = list(u_list - set(list(item["u_id"])))
                # print("用户池", len(random_pool))
                for add_u in range(55-uers):
                    u_index = random.randint(0,len(random_pool)-1)
                    u_id = random_pool[u_index]
                    # print("选择", u_id)
                    # 用户历史只截取最后100条，超长弃用
                    u_his_list = list(his_set[his_set['u_id'] == int(u_id)]['q_text'])
                    if(len(u_his_list)>100):
                        u_his_list = u_his_list[-100:0]
                    u_his_str = ".".join(u_his_list)
                    pd_r.append(0.0)
                    pd_q_text.append(list(item["q_text"])[0])
                    pd_q_id.append(list(item["q_id"])[0])
                    pd_u_id.append(u_id)
                    pd_u_his_str.append(u_his_str)
                    pd_score.append(-5.0)
                    real_u_set_list.append(list(item["real_u_set"])[0])

        self.eval_data = pd.DataFrame()
        self.eval_data["r"] = pd_r
        self.eval_data["q_text"] = pd_q_text
        self.eval_data["q_id"] = pd_q_id
        self.eval_data["u_id"] = pd_u_id
        self.eval_data["u_his_str"] = pd_u_his_str
        self.eval_data["score"] = pd_score
        self.eval_data["real_u_set"] = real_u_set_list
        self.eval_data = self.eval_data.reset_index(drop=True)
        print("****请检查数据****")
        print(self.eval_data.loc[[0]])
        print("问题数量", len(list(set(pd_q_id))), "回答数量", len(pd_u_id))
        self.res = self.eval_data.values.tolist()    # __getitem__调用的数据
        print("数据集数量", len(self.res))
        print("*****************")
        self.eval_data[["q_id", "u_id", "score", "real_u_set"]].to_csv("eval_data.csv")


    def __len__(self):
        return len(self.res)

    def __getitem__(self, idx):
        # [r_batch, q_text_batch, q_id_batch, u_id_batch, u_his_batch, score_batch]
        return [self.res[idx][0], self.res[idx][1], self.res[idx][2], self.res[idx][3], self.res[idx][4], self.res[idx][5], self.res[idx][6]]

def getTrainLoader(train_set, param):
    start = time.time()
    dataset = loadDataSet(train_set)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=param['batch_size'], shuffle=True, collate_fn=coll,
                                               num_workers=10)
    print("训练数据组装完成，耗时：", str(float('%.2f' % float(time.time()-start))), "秒")
    return train_loader

def getTrainLoader_Sou(train_set, param):
    start = time.time()
    dataset = loadDataSet_Sou(train_set)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=param['batch_size'], shuffle=True, num_workers=20)
    print("训练数据组装完成，耗时：", str(float('%.2f' % float(time.time()-start))), "秒")
    return train_loader

def getEvalLoader(his_set, eval_set, param):
    start = time.time()
    dataset = loadDataSet_Eval(his_set,eval_set)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=param['batch_size'], shuffle=False, num_workers=0)
    print("验证数据组装完成，耗时：", str(float('%.2f' % float(time.time()-start))), "秒")
    return train_loader


def getTestLoader(test_set, param):
    start = time.time()
    dataset = loadDataSet(test_set)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=param['batch_size'], shuffle=True, collate_fn=coll,
                                              num_workers=0)
    print("测试数据组装完成，耗时：", str(float('%.2f' % float(time.time()-start))), "秒")
    return test_loader

def getTestLoader_Sou(test_set, param):
    start = time.time()
    dataset = loadDataSet_Sou(test_set)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=param['batch_size'], shuffle=True, num_workers=0)
    print("测试数据组装完成，耗时：", str(float('%.2f' % float(time.time()-start))), "秒")
    return test_loader


def readSet(data_path):
    # 1. 读取数据集
    start = time.time()
    month1 = pd.read_csv(data_path + 'month1.csv')
    month2 = pd.read_csv(data_path + 'month2.csv')
    month3 = pd.read_csv(data_path + 'month3.csv')
    month4 = pd.read_csv(data_path + 'month4.csv')
    month5 = pd.read_csv(data_path + 'month5.csv')
    month6 = pd.read_csv(data_path + 'month6.csv')
    month7 = pd.read_csv(data_path + 'month7.csv')
    month8 = pd.read_csv(data_path + 'month8.csv')
    month9 = pd.read_csv(data_path + 'month9.csv')
    month10 = pd.read_csv(data_path + 'month10.csv')
    month11 = pd.read_csv(data_path + 'month11.csv')
    month12 = pd.read_csv(data_path + 'month12.csv')

    month1_neg1 = pd.read_csv(data_path + 'month1_neg.csv')
    month2_neg1 = pd.read_csv(data_path + 'month2_neg.csv')
    month3_neg1 = pd.read_csv(data_path + 'month3_neg.csv')
    month4_neg1 = pd.read_csv(data_path + 'month4_neg.csv')
    month5_neg1 = pd.read_csv(data_path + 'month5_neg.csv')
    month6_neg1 = pd.read_csv(data_path + 'month6_neg.csv')
    month7_neg1 = pd.read_csv(data_path + 'month7_neg.csv')
    month8_neg1 = pd.read_csv(data_path + 'month8_neg.csv')
    month9_neg1 = pd.read_csv(data_path + 'month9_neg.csv')
    month10_neg1 = pd.read_csv(data_path + 'month10_neg.csv')
    month11_neg1 = pd.read_csv(data_path + 'month11_neg.csv')
    month12_neg1 = pd.read_csv(data_path + 'month12_neg.csv')

    month1_neg2 = pd.read_csv(data_path + 'month1_neg2.csv')
    month2_neg2 = pd.read_csv(data_path + 'month2_neg2.csv')
    month3_neg2 = pd.read_csv(data_path + 'month3_neg2.csv')
    month4_neg2 = pd.read_csv(data_path + 'month4_neg2.csv')
    month5_neg2 = pd.read_csv(data_path + 'month5_neg2.csv')
    month6_neg2 = pd.read_csv(data_path + 'month6_neg2.csv')
    month7_neg2 = pd.read_csv(data_path + 'month7_neg2.csv')
    month8_neg2 = pd.read_csv(data_path + 'month8_neg2.csv')
    month9_neg2 = pd.read_csv(data_path + 'month9_neg2.csv')
    month10_neg2 = pd.read_csv(data_path + 'month10_neg2.csv')
    month11_neg2 = pd.read_csv(data_path + 'month11_neg2.csv')
    month12_neg2 = pd.read_csv(data_path + 'month12_neg2.csv')

    train_set = pd.concat([month1, month2, month3, month4, month1_neg1, month2_neg1, month3_neg1, month4_neg1, month1_neg2, month2_neg2, month3_neg2, month4_neg2, month5, month5_neg1, month5_neg2]).drop(columns='Unnamed: 0')  # 真实问答对与负采样问答对合并 1:1的比例
    # train_set = pd.concat([month1, month2, month3, month4]).drop(columns='Unnamed: 0')      # 真实问答对与负采样问答对合并 1:1的比例
    # train_set = pd.concat([month1, month1_neg1, month1_neg2]).drop(columns='Unnamed: 0')      # 真实问答对与负采样问答对合并 1:1的比例
    test_set = pd.concat([month6, month6_neg1, month6_neg2, month7, month7_neg1, month7_neg2, ]).drop(columns='Unnamed: 0')  # 真实问答对与负采样问答对合并
    # test_set = pd.concat([month5, month6]).drop(columns='Unnamed: 0')      # 真实问答对与负采样问答对合并
    # test_set = pd.concat([month2, month2_neg1, month2_neg2]).drop(columns='Unnamed: 0')      # 真实问答对与负采样问答对合并
    val_set = pd.concat([month7, month7_neg1, month7_neg2]).drop(columns='Unnamed: 0')  # 一个月做验证
    print("数据读取完成，耗时：", str(float('%.2f' % float(time.time()-start))), "秒")
    return train_set, test_set


def readDataEvaluation(data_path, more_than):
    # 1. 读取数据集
    start = time.time()
    month1 = pd.read_csv(data_path + 'month1.csv')
    month2 = pd.read_csv(data_path + 'month2.csv')
    month3 = pd.read_csv(data_path + 'month3.csv')
    month4 = pd.read_csv(data_path + 'month4.csv')
    month5 = pd.read_csv(data_path + 'month5.csv')
    month6 = pd.read_csv(data_path + 'month6.csv')
    month7 = pd.read_csv(data_path + 'month7.csv')
    month8 = pd.read_csv(data_path + 'month8.csv')
    month9 = pd.read_csv(data_path + 'month9.csv')
    month10 = pd.read_csv(data_path + 'month10.csv')
    month11 = pd.read_csv(data_path + 'month11.csv')
    month12 = pd.read_csv(data_path + 'month12.csv')
    cols = ["q_id", "q_time", "q_text", "u_id", "a_id","a_text","a_time", "r", "score"]
    train_set = pd.concat([
        month1, month2, month3, month4, month5, month6, month7, month8, month9, month10,month11,month12
        ]).drop(columns='Unnamed: 0')[cols].drop_duplicates(subset=None, keep='first', inplace=False)  # 真实问答对与负采样问答对合并 1:1的比例
    # 筛选回答数量大于more_than的问答对，用作计算评价指标
    df2 = train_set.q_id.value_counts() >= more_than
    df2 = df2[df2]
    eval_set = train_set.loc[train_set.q_id.isin(df2.index)] 
    # print(new_set)
    # train_set只保留eval_set中存在的用户
    # his_set = train_set.loc[train_set.u_id.isin(eval_set.u_id)]
    his_set = train_set
    print("数据读取完成，耗时：", str(float('%.2f' % float(time.time()-start))), "秒")
    return his_set, eval_set




def getLogger(src):
    logger = logging.getLogger()  # 第一步，创建一个logger
    logger.setLevel(logging.INFO)  # Log等级总开关
    logfile = src + 'log/TargetTrain_log_at_' + str(time.asctime(time.localtime(time.time()))).replace(":", "_") + '.txt'
    fh = logging.FileHandler(logfile, mode='a')  # open的打开模式这里可以进行参考
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    formatter = logging.Formatter("")
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # 第五步，将logger添加到handler里面
    logger.info(
        'type' + '\t' + 'epoch' + '\t' + 'step' + '\t' + 'epoch_mean_loss' + '\t' + 'correct' + '\t' + 'stpe_acc')
    return logger


def getTestLogger(src):
    logger = logging.getLogger()  # 第一步，创建一个logger
    logger.setLevel(logging.INFO)  # Log等级总开关
    logfile = src + 'log/SourceTrain_log_at:' + str(time.asctime(time.localtime(time.time()))).replace(":", "_") + '.txt'
    fh = logging.FileHandler(logfile, mode='a')  # open的打开模式这里可以进行参考
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    formatter = logging.Formatter("")
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # 第五步，将logger添加到handler里面
    logger.info(
        'type' + '\t' + 'epoch' + '\t' + 'step' + '\t' + 'epoch_mean_loss' + '\t' + 'correct' + '\t' + 'stpe_acc')
    return logger



def logout(str, logger):
    logger.info(str)
    print("\r"+str, end="")

def passItem(all,now):
    passed = "■"*int(now/all*50)
    unpassed = "_"*(50-int(now/all*50))
    return "["+passed+unpassed+"]"

# 保存参数到runs/某时间文件夹下


def saveParams(dictParams, src):
    str = json.dumps(dictParams)
    with open(src, "w") as f:
        f.write(str)
