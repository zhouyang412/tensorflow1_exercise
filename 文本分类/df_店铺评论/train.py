import math
import tensorflow as tf
from config import *

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from utils import read_file, build_vocab, convert_to_inputids, batch_iter, test_batch_iter

import logging
import time

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler('bert.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def evaluate(sess, model, eval_loader, eval_allsteps):
    total_loss = 0.0
    total_f1 = 0.0
    for step, (x_batch, y_batch) in tqdm(enumerate(eval_loader)):
        if step > eval_allsteps:
            break
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.keep_prob: 1.0
         }
        cur_loss, cur_f1 = sess.run([model.loss, model.cur_f1], 
                                    feed_dict=feed_dict)
        total_loss += total_loss
        
    return cur_f1, cur_loss
            
    

def train_model(config, model, all_ids, all_labels):
    tensorboard_dir = config.tensorboard_dir
    
    # 配置tensorboard
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("cur_precision", model.cur_precision)
    tf.summary.scalar("cur_recall", model.cur_recall)
    tf.summary.scalar("cur_f1", model.cur_f1)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    
    # 配置saver
    saver = tf.train.Saver()
    orig_decay = config.lr_decay
    learning_rate = config.learning_rate
    max_lr_epoch = config.max_lr_epoch
    
    # 构建训练集，验证集
    all_labels = np.array(all_labels)
    kf = StratifiedKFold(
        n_splits=config.fold_count, random_state=config.seed, shuffle=True).split(X=all_ids, y=all_labels)
    
    # 划分训练验证集
    for fold_idx, (train_idx, eval_idx) in enumerate(kf):
        # 创建session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
#         sess.run(tf.local_variables_initializer())
        writer.add_graph(sess.graph)
        
        # 当前折的训练集
        train_ids = all_ids[train_idx]
        train_labels = all_labels[train_idx]
        train_loader = batch_iter(train_ids, train_labels)
        # 当前折的验证集
        eval_ids = all_ids[eval_idx]
        eval_labels = all_labels[eval_idx]
        eval_loader = batch_iter(eval_ids, eval_labels, batch_size=64)
        
        best_f1 = -999
        earlystop_count = 0
        train_steps_fold = math.ceil(len(train_ids) / config.train_batch_size)
        eval_steps_fold = math.ceil(len(eval_ids) / config.eval_batch_size)
        for epoch in range(config.num_epochs):
            logger.info("epoch:{}, fold_idx:{}".format(epoch, fold_idx))
            # 初始化matrics算子
            sess.run(tf.local_variables_initializer())
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0)
            model.assign_lr(sess, learning_rate * new_lr_decay)
            logger.info(sess.run(model.learning_rate))
            if earlystop_count >= config.early_stop:
                break
                
            for step, (x_batch, y_batch) in tqdm(enumerate(train_loader)):
                
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.keep_prob: config.dropout_keep_prob
                 }
                if step % config.save_per_batch == 0  and epoch == 0 and fold_idx == 0:
                    # 每多少轮次将训练结果写入tensorboard scalar
                    s = sess.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, step)
                    
                # 反向传播迭代优化
                feed_dict[model.keep_prob] = config.dropout_keep_prob
                _, cur_f1 = sess.run([model.train_op, model.cur_f1], feed_dict=feed_dict)
                
                # 迭代step次时，实际已经完整的预测了traindata
                # 如sample/bs = 62.5需要迭代63次，step=62时已经迭代63次                
                if step + 1 >= train_steps_fold:
                    break
            logger.info("train_f1:{}".format(cur_f1))
            # 初始化matrics算子，进行eval
            sess.run(tf.local_variables_initializer())
            # 对验证集进行预测评估，并保存模型
            eval_f1, eval_loss = evaluate(sess, model, eval_loader, eval_steps_fold)
            logger.info("epoch:{}, eval_f1:{}".format(epoch, eval_f1))
            
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                earlystop_count = 0
                saver.save(sess=sess, save_path=config.save_dir + config.sava_model_name + '{}'.format(fold_idx))
            else:
                earlystop_count += 1
                            
                
if __name__ == '__main__':
    pass
    