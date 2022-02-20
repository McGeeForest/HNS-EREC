# __package__ = 'ProcessData'
import logging
from transformers import BertTokenizer, BertModel
import torch
import nltk
from tqdm import tqdm


def getStopwords(filepath):
    stopwords = [line.strip() for line in open(
        filepath, 'r', encoding='utf-8').readlines()]
    # print(stopwords)
    return stopwords


def excutBert(texts, tokenizer, model):
    res_vec = []
    for text in tqdm(texts):
        item_vec = []
        #将所有大写字母转换为小写字母
        text = text.lower()
        #加载punkt句子分割器
        sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        #对句子进行分割
        sentences = sen_tokenizer.tokenize(text)
        for sentence in sentences:
            item_vec = item_vec + getVecBert(sentence, tokenizer, model)
        res_vec.append(item_vec)
    return res_vec


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


def getVecBert(sentence, tokenizer, model):
    # print(sentence)
    # print(len(sentence))
    token_vecs_sum = []
    # Add the special tokens.
    marked_sentence = "[CLS] " + sentence + " [SEP]"
    # Split the sentence into tokens.
    tokenized_sentence = tokenizer.tokenize(marked_sentence)[:512]
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)
    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_sentence)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():       # 不生成计算图，不需要反馈，这样速度更快
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]  # 输出隐藏层
        token_embeddings = torch.stack(hidden_states, dim=0)
        # print(token_embeddings.size())
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec.numpy().tolist())
    return token_vecs_sum
if __name__ == "__main__":
    excut(texts, tokenizer, model)

