class CNNConfig(object):
    
    
    embedding_dim = 128  # 词向量维度
    seq_length = 50  # 序列长度
    num_classes = 2  # 类别数
    num_filters = 128  # 卷积核数目
    kernel_size = [2, 3, 4, 5]  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 256  # 全连接层神经元

    dropout_keep_prob = 0.8 # 实际的prob=1-dropout_keep_prob dropout保留比例
    learning_rate = 6e-3   # 学习率

    train_batch_size = 32        # 每批训练大小
    eval_batch_size = 32
    test_batch_size = 32
    num_epochs = 6        # 总迭代轮次
    max_lr_epoch = 4
    lr_decay = 0.8
    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 5      # 每多少轮存入tensorboard
    fold_count = 5


    tensorboard_dir = './log/'
    save_dir = './best_models/'
    train_dir = '/home/dc2-user/p_data/dataset/NLP/df-店铺评论/train.csv'
    test_dir = '/home/dc2-user/p_data/dataset/NLP/df-店铺评论/test_new.csv'
    sample_dir = '/home/dc2-user/p_data/dataset/NLP/df-店铺评论/sample.csv'
    
    sava_model_name = 'cnnfold-'
    seed = 42
    early_stop = 2

class RNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 128      # 词向量维度
    seq_length = 50        # 序列长度
    num_classes = 2        # 类别数
    vocab_size = 6000       # 词汇表达小

    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'lstm'             # lstm 或 gru
    biderectional = True     # 是否双向

    dropout_keep_prob = 0.7 # 实际的prob=1-dropout_keep_prob dropout保留比例
    learning_rate = 5e-3   # 学习率

    train_batch_size = 32        # 每批训练大小
    eval_batch_size = 32
    test_batch_size = 32
    num_epochs = 6        # 总迭代轮次
    max_lr_epoch = 4
    lr_decay = 0.8
    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 5      # 每多少轮存入tensorboard
    fold_count = 5
    
    
    tensorboard_dir = './log/'
    save_dir = './best_models/'
    train_dir = '/home/dc2-user/p_data/dataset/NLP/df-店铺评论/train.csv'
    test_dir = '/home/dc2-user/p_data/dataset/NLP/df-店铺评论/test_new.csv'
    sample_dir = '/home/dc2-user/p_data/dataset/NLP/df-店铺评论/sample.csv'
    
    sava_model_name = 'rnnfold-'
    seed = 42
    early_stop = 2
    
 
  