from flask import Flask
from flask import request
from gensim.models import word2vec
from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import datapath
import gensim.downloader
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from torch.nn.utils.rnn import pad_sequence
import torch
import json
import time
app = Flask(__name__)
w2vModel = None


@app.route("/getVecBy", methods=["GET", "POST"])
def getVecBy():
    print("request:")
    print(request)
    word = request.args.get("word")
    print(word)
    if w2vModel is None:
        return "w2v模型未加载"
    try:
        return {"res": w2vModel[word].tolist()}
    except BaseException:
        return "exception:"+word

@app.route("/getVecByV2", methods=["GET", "POST"])
def getVecByV2():
    print("request:")
    print(request)
    word = request.args.get("word")
    print(word)
    if w2vModel is None:
        return "w2v模型未加载"
    return {"res": w2vModel([word]).tolist()}
    # try:
    #     return {"res": w2vModel([word]).tolist()}
    # except BaseException:
    #     return "exception:"+word

def getVecW2V(sentence):  # 取w2v 作为词向量
    words = word_tokenize(sentence)
    wordsVec = []
    count = 0
    for w in words:
        if w not in stopWords:
            try:
                count += 1
                # wordsVec.append(requests.get('http://127.0.0.1:5000/getVecBy?word='+w))
                wordsVec.append(w2vModel[w].tolist())
                # wordsVec.append([1,2,3,4,5])
            except KeyError:
                pass
    # print(count)
    if len(wordsVec) == 0:
        wordsVec = getVecW2V("there is null text.")
    return wordsVec


@app.route("/getVecSeqBy", methods=["GET", "POST"])
def excutW2V():
    texts = json.loads(request.get_data(as_text=True))
    texts = texts.get("texts")
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
            item_vec = getVecW2V("there is null text.")
        res_vec.append(item_vec)
        real_len.append(len(item_vec))
    res_vec = pad_sequence([torch.FloatTensor(item) for item in res_vec], batch_first=True).numpy().tolist()
    res_data = {'res_vec': res_vec, 'real_len': real_len}
    print("词向量已经返回, 耗时：" + str(time.time()-t1))
    return res_data

if __name__ == "__main__":
    # w2vModel = gensim.downloader.load('word2vec-google-news-300')
    # w2vModel = word2vec.Word2Vec.load("/home/shulin/project/transNetsQANew/src3/ProcessData/w2vmodel/sgns.sogou.word")  # 词向量模型
    w2vModel = KeyedVectors.load_word2vec_format('/home/xxx/gensim-data/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)
    vocabs = w2vModel.key_to_index.keys()     # w2v的词汇表
    stopWords = set(stopwords.words('english'))
    # w2vModel = KeyedVectors.load_word2vec_format("./model/sgns.sogou.word", binary=False)
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')   #加载punkt句子分割器
    print("***********w2v加载完成***********")
    app.run(host='0.0.0.0', port=5002)
    # print("测试提交")
