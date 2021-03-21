import tensorflow as tf
import math

class BiRNN(object):
    
    def __init__(self, config, init_emd=None, is_training=True, is_testing=False):
        
        self.config = config
        self.init_emd = init_emd
        self.is_training = is_training
        self.is_testing = is_testing
#         self.learning_rate = self.config.learning_rate
        self.learning_rate = tf.Variable(self.config.learning_rate, trainable=False)
        
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, name='input_y')
        self.keep_prob = tf.placeholder(tf.int32, name='keep_prob')
        
        self.forward()
        
    def forward(self):
        
        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(self.config.hidden_dim)
        
        def gru_cell():
            return tf.nn.rnn_cell.GRUCell(self.config.hidden_dim)
        
        
        with tf.device('/cpu:0'):
            if self.init_emd is not None:
                embedding = tf.Variable(init_emd, trainable=True, name="emb", dtype=tf.float32)
            else:
                init_mat = tf.random_uniform([self.config.vocab_size, self.config.hidden_dim], -1, 1)
                embedding = tf.Variable(init_mat, trainable=True, name="emb", dtype=tf.float32)
            
            # [batch_size, seq_len, emb_dim]
            inputs_emb = tf.nn.embedding_lookup(embedding, self.input_x)
            # [seq_len, batch_size, emb_dim]
            inputs_emb = tf.transpose(inputs_emb, [1, 0, 2])
            # [seq_len * batch_size, emb_dim]
            inputs_emb = tf.reshape(inputs_emb, [-1, self.config.hidden_dim])
            # input_emb包含seq_len个tensor，每个tensor --> [batch_size, emb_dim]
            inputs_emb = tf.split(inputs_emb, self.config.seq_length, 0)
            
        with tf.name_scope("birnn_cell"):
            if self.config.rnn.lower() == 'lstm':
                cell_fw = lstm_cell()
                cell_bw = lstm_cell()
            else:
                cell_fw = gru_cell()
                cell_bw = gru_cell()

            if self.is_training:
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, 
                                            output_keep_prob=self.config.dropout_keep_prob)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, 
                                            output_keep_prob=self.config.dropout_keep_prob)   

            cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * self.config.num_layers)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * self.config.num_layers)
            
            # outputs: [batch_size, seq_len, hid_size * 2]
            # output_state_fw: [batch_size, hid_size] * num_layers
            # output_state_bw: [batch_size, hid_size] * num_layers
            outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(
                cell_fw,
                cell_bw,
                inputs_emb,
                dtype=tf.float32,
                )
            outputs = tf.concat(outputs, 1)
            outputs = tf.reshape(outputs, 
                                 [-1, self.config.seq_length, self.config.hidden_dim * 2])
            
            
        with tf.name_scope("pooling"):
            # [batch_size, 1, hid_size * 2]
            outputs_max = tf.nn.max_pool1d(outputs,
                                  ksize=self.config.seq_length, 
                                  strides=1, 
                                  padding="VALID")
            # [batch_size, 1, hid_size * 2]
            outputs_avg = tf.nn.avg_pool1d(outputs,
                                  ksize=self.config.seq_length, 
                                  strides=1, 
                                  padding="VALID")

#             avg_pool_res = tf.reduce_max(outputs,  reduction_indices=[1])
#             max_pool_res = tf.reduce_mean(outputs. reduction_indices=[1])
            
            # [batch_size, hid_size * 2]
            outputs_max = tf.squeeze(outputs_max, 1)
            outputs_avg = tf.squeeze(outputs_avg, 1)
            # [batch_size, hid_size * 4]
            pooling_outputs = tf.concat([outputs_max, outputs_avg], 1)
            
        with tf.name_scope("logits"):
            # [batch_size, num_classes]
            self.logits = tf.layers.dense(pooling_outputs, self.config.num_classes)
            self.logits_outputs = tf.nn.softmax(self.logits, axis=1)
            self.y_pred = tf.argmax(self.logits, axis=1)
        
        if self.is_testing: return 
        
        with tf.name_scope("loss"):
            self.labels_onehot = tf.one_hot(self.input_y, depth=self.config.num_classes)
            
#             cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.labels_onehot, 
                                              logits=self.logits) # label_smoothing=0.001
            self.loss = tf.reduce_mean(cross_entropy)
               
        with tf.name_scope("train_op"):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
            
        with tf.name_scope("acc"):
            _, self.cur_precision = tf.metrics.precision(self.input_y, self.y_pred)
            
        with tf.name_scope("recall"):
            _, self.cur_recall = tf.metrics.recall(self.input_y, self.y_pred)
            
        with tf.name_scope("f1"):
            self.cur_f1 = 2 * self.cur_precision * self.cur_recall / (self.cur_precision + self.cur_recall)
            
        with tf.name_scope("new_lr"):
            # 用于更新 学习率
            self.new_lr = tf.placeholder(tf.float32, shape=[])
            # 将new_lr赋值给lr_update
            self.lr_update = tf.assign(self.learning_rate, self.new_lr)            
            
    # 更新 学习率
    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})
        
        
        
class TextCNN(object):
    
    def __init__(self, config, init_emd=None, is_training=True, is_testing=False):
        
        self.config = config
        self.init_emd = init_emd
        self.is_training = is_training
        self.is_testing = is_testing
        self.learning_rate = tf.Variable(self.config.learning_rate, trainable=False)
        
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, name='input_y')
        self.keep_prob = tf.placeholder(tf.int32, name='keep_prob')
        
        self.forward()
        
    def forward(self):
        
        with tf.device('/cpu:0'):
            if self.init_emd is not None:
                embedding = tf.Variable(init_emd, trainable=True, name="emb", dtype=tf.float32)
            else:
                init_mat = tf.random_uniform([self.config.vocab_size, self.config.hidden_dim], -1, 1)
                embedding = tf.Variable(init_mat, trainable=True, name="emb", dtype=tf.float32)
            inputs_emb = tf.nn.embedding_lookup(embedding, self.input_x)
            # [batch_size, seq_len, embed_dim] 
            inputs_emb = tf.contrib.layers.dropout(inputs_emb, self.config.dropout_keep_prob)
            
        with tf.name_scope("convs"):
            convs = []
            # [batch_size, conv_dim, num_filters]
            for kernel_size in self.config.kernel_size:
                conv = tf.layers.conv1d(inputs_emb, 
                                        self.config.num_filters, 
                                        kernel_size, 
                                        padding='valid',
                                        name='conv_{}'.format(kernel_size))
                convs.append(conv)
                
        with tf.name_scope("pooling"):
            pool_reses = []
            for conv_res  in convs:
                avg_pool_res = tf.reduce_max(conv_res,  reduction_indices=[1])
                max_pool_res = tf.reduce_mean(conv_res, reduction_indices=[1])
                pool_reses.append(avg_pool_res)
                pool_reses.append(max_pool_res)
            # [batch_size, num_filter * len(kernel_size)]
            pooling_outputs = tf.concat(pool_reses, 1)
                
        with tf.name_scope("linear"):
            linear_res = tf.layers.dense(pooling_outputs, self.config.hidden_dim)
            linear_res = tf.contrib.layers.dropout(linear_res, self.config.dropout_keep_prob)
            linear_res = tf.nn.relu(linear_res)
            
        with tf.name_scope("logits"):
            self.logits = tf.layers.dense(linear_res, self.config.num_classes)
            self.logits_outputs = tf.nn.softmax(self.logits)
            self.y_pred = tf.argmax(self.logits_outputs, axis=1)
            
        with tf.name_scope("loss"):
            # loss计算
            self.labels_onehot = tf.one_hot(self.input_y, depth=self.config.num_classes)
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.labels_onehot, 
                                                            logits=self.logits) # label_smoothing=0.001
            self.loss = tf.reduce_mean(cross_entropy)
        
        with tf.name_scope("train_op"):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
            
        with tf.name_scope("acc"):
            _, self.cur_precision = tf.metrics.precision(self.input_y, self.y_pred)
            
        with tf.name_scope("recall"):
            _, self.cur_recall = tf.metrics.recall(self.input_y, self.y_pred)
            
        with tf.name_scope("f1"):
            self.cur_f1 = 2 * self.cur_precision * self.cur_recall / (self.cur_precision + self.cur_recall)
        
        with tf.name_scope("new_lr"):
            # 更新learning_rate
            self.new_lr = tf.placeholder(tf.float32, shape=[])
            self.lr_update = tf.assign(self.learning_rate, self.new_lr)
        self.test = (self.train_op, self.loss)    
    
    def assign_lr(self, sess, lr_value):
        sess.run(self.lr_update, feed_dict={self.new_lr: lr_value})
                
        