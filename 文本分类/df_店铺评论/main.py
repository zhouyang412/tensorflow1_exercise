import tensorflow as tf

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from models import BiRNN
from train import train_model
from predict import test_model
from utils import read_file, build_vocab, convert_to_inputids, batch_iter, test_batch_iter

import os
import random
import logging
import time
import config
import pandas as pd
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seed = 42
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
seed_all(seed)

rnn_config = config.RNNConfig
tf.reset_default_graph()
# 训练模型，并保存
model = BiRNN(rnn_config)    
train_comments, train_labels = read_file(rnn_config.train_dir)
word2id = build_vocab(train_comments)
train_ids = convert_to_inputids(train_comments, word2id, rnn_config.seq_length)
train_model(rnn_config, model, train_ids, train_labels)    

# 对测试集进行预测，并按5折进行集成logits
tf.reset_default_graph()    

test_comments, test_comment_ids = read_file(rnn_config.test_dir, train=False)
test_ids = convert_to_inputids(test_comments, word2id, rnn_config.seq_length)
test_result = test_model(rnn_config, model, test_ids)  
    
    

    
    
 


