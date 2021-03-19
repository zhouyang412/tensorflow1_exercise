import pandas as pd
import numpy as np
import tensorflow.keras as kr
import collections
import jieba
import math


def read_file(file_dir, train=True):
    """
    读取csv文件
    """
    if train:
        comments, labels = [], []
        df_train = pd.read_csv(file_dir, sep='\t')
        for comment, label in zip(df_train.comment, df_train.label):
            comments.append(comment)
            labels.append(label)
        return comments, labels
    else:
        comments, comment_ids = [], []
        df_test = pd.read_csv(file_dir)
        for comment, comment_id in zip(df_test.comment, df_test.id):
            comments.append(comment)
            comment_ids.append(comment_id)
        return comments, comment_ids
    
def build_vocab(train_comments, vocab_size=5000):
    """
    对训练集分词,统计词频并取vocab_size个词作为词表
    """
    
    all_words = []
    
    for comment in train_comments:
        seg_comment = jieba.lcut(comment)
        all_words.extend(seg_comment)
    word_counter = collections.Counter(all_words)
    select_words = word_counter.most_common(vocab_size-2)
    select_words, _ = list(zip(*select_words))
    select_words = ["<PAD>"] + ["<UNK>"] + list(select_words)
    word2id = {word: idx for idx, word in enumerate(select_words)}
    
    return word2id
    
def convert_to_inputids(comments, word2id, max_len=40):
    
    input_ids = []
    
    for comment in comments:
        comment_ids = [word2id[w] if w in word2id  else word2id["<UNK>"] for w in comment]
        input_ids.append(comment_ids)
        
    input_ids = kr.preprocessing.sequence.pad_sequences(input_ids, max_len, padding='post')
    
    return input_ids

def batch_iter(x, y, batch_size=32, shuffle=False):
    """
    batch数据生成器
    """

    sample_len = len(x)
    batch_count = math.ceil(sample_len / batch_size)
    # 随机打散
    if shuffle:
        indices = np.random.permutation(np.arange(sample_len))
    else:
        indices = list(np.arange(sample_len))
    x_shuffle = np.array(x)[indices]
    y_shuffle = np.array(y)[indices]

    while True:
        for i in range(batch_count):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, sample_len)
            yield x_shuffle[start_id: end_id], y_shuffle[start_id: end_id]  
            
def test_batch_iter(x, batch_size=32):
    
    sample_len = len(x)
    batch_count = math.ceil(sample_len / batch_size)
    
    x = np.array(x)
    while True:    
        for i in range(batch_count):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, sample_len)
            yield x[start_id: end_id]
    