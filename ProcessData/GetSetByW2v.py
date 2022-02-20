from gensim import models
import pandas as pd
import nltk
from tqdm import tqdm

def getStopwords(filepath):
    stopwords = [line.strip() for line in open(
        filepath, 'r', encoding='utf-8').readlines()]
    # print(stopwords)
    return stopwords


def excutW2V(texts, model, stopwds):
    res_vec = []
    for text in tqdm(texts):
        item_vec = []
        #将所有大写字母转换为小写字母
        text = text.lower()
        #对句子进行分割
        document_cut = nltk.word_tokenize(text)
        for word in document_cut:
            if word == document_cut[-1]:
                word.replace('\n', '')
            if word not in stopwords:
                try:
                    word = str(model[word].tolist())
                    words_vec.append(word)
                except KeyError:
                    print("KeyError")
    res_vec.append(item_vec)
    return res_vec



stopwds = getStopwords('/home/student/xxx/project/transNetsQANew/data/stopwords.txt')
model = models.KeyedVectors.load_word2vec_format('/home/student/xxx/project/transNetsQANew/src3/ProcessData/w2vmodel/knowledge-vectors-skipgram1000.bin', binary=True)  # 词向量模型
print(len(model.vocab))
# print(model['java'])

qa_set = pd.read_table("/home/student/xxx/project/transNetsQANew/data/qa_set.txt")
qa_set.columns = ['q_id', 'q_text', 'u_id', 'a_text', 'r', 'q_time', 'a_time']
qa_set = qa_set.head(1000)


# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # Load pre-trained model (weights)

# model.eval()
q_text = qa_set['q_text']
q_vec = excutW2V(q_text, model, stopwds)

# # # 答案文本转向量
# a_text = qa_set['a_text']
# a_vec = excut(a_text, tokenizer, model)

qa_set['q_vec'] = q_vec
# qa_set['a_vec'] = a_vec
# # # print(q_vec[0])
# # print(qa_set.head(5))
qa_set.to_csv("/home/student/xxx/project/transNetsQANew/src3/data3/qa_vec.1000.csv")
