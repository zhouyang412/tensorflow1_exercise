from config import *
import pandas as pd
import numpy as np
import math


                
def test_model(config, model, test_ids):
    
    # 读取数据 
    test_loader = test_batch_iter(test_ids, rnn_config.test_batch_size)
    test_steps_fold = math.ceil(len(test_ids) / config.test_batch_size)
    df_result = pd.read_csv(config.sample_dir)
    
    assert len(test_ids) == len(df_result), "check your pred_data!"
    test_logits = np.zeros((len(df_result), 2))
    
    for fold_idx in range(config.fold_count):
        model_path = config.save_dir + config.sava_model_name + '{}'.format(fold_idx)
        # 初始化
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=model_path)

        for step, x_batch in tqdm(enumerate(test_loader)): 
                
            start_id = step * config.test_batch_size
            end_id = min((step + 1) * config.test_batch_size, len(df_result))
            
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: 0.0,
                model.keep_prob: 1.0
             }
            logits_batch, pred_batch = sess.run([model.logits_outputs, model.y_pred],
                                                feed_dict=feed_dict)
#             test_preds.extend(pred_batch)
            test_logits[start_id: end_id] += logits_batch
            
            # 迭代step次时，实际已经完整的预测了testdata
            # 如sample/bs = 62.5需要迭代63次，step=62时已经迭代63次
            if step + 1 >= test_steps_fold:
                break  
        
        sess.close()
    
    test_preds = np.argmax(test_logits, axis=1)
    df_result["label"] = test_preds
    
    return df_result    