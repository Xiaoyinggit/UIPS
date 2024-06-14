# coding:utf8
import os
import time
import random
import numpy as np
import tensorflow.compat.v1 as tf
import pandas as pd
import sys
import argparse
from input import *
from evaluate import *
from utils import *
from beta_star.beta_star_model import *
from beta_star.beta_star_model_v2 import *


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="Beta_Star")
parser.add_argument("--epochs", default='5')
parser.add_argument("--topN", default='[10,20]')
parser.add_argument("--lr", default='0.0001')
parser.add_argument("--train_batch_size", default='512')
parser.add_argument("--usedata", default='Wiki_beta_star')
parser.add_argument(
    '--v2_para',
    type=lambda x: {k:float(v) for k,v in (i.split(':') for i in x.split(','))},
    default='0:0',
    help='comma-separated field:position pairs, e.g. Date:0,Amount:2,Payee:5,Memo:9'
)



args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

train_batch_size = eval(args.train_batch_size)
topN = eval(args.topN)
lr = eval(args.lr)

hyperparameter = {}

print('[model name]', args.model_name)
print('[data]', args.usedata)
print('[epochs]', args.epochs)
print('[topN]', args.topN)
print('[learning rate]', args.lr)
print('[train batch size]', args.train_batch_size)

data_dict = loadData(args.usedata)


gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    if args.model_name == 'beta_star_v2':
        model = BetaStar_V2(data_dict['user_count'],data_dict['item_count'],data_dict['feature_count'], v2_para=args.v2_para)
    else:
        model = BetaStart(data_dict['user_count'],data_dict['item_count'],data_dict['feature_count'])
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # model.initFeature(sess,data_dict['feature_list'])
    output_value_list = [getInfo() for i in range(len(topN))]
    best_epoch = [0 for _ in range(len(topN))]
    train_set = data_dict['train_set']
    for epoch in range(eval(args.epochs)):
        random.shuffle(train_set)
        loss_sum = 0.0
        count = 0
        start_time = time.time()
        for _, uij in DataInput(train_set, train_batch_size):
            count += 1
            loss = model.train(sess, uij, lr)
            loss_sum += loss
        print('Epoch {} Global_step {}\t''[joint]Train_loss: {}\t'.format(
            model.global_epoch_step.eval(), model.global_step.eval(),
            loss_sum / count))
        sys.stdout.flush()

        valid_pos_result, valid_user_pred = candidate_ranking(sess, model, data_dict['valid_set'], topN, data_dict['log_item'])
        test_pos_result, test_user_pred = candidate_ranking(sess, model,data_dict['test_set'], topN, data_dict['log_item'])



        print('[Valid set] Precision: {}\tRecall: {}\tNDCG: {}'.format(
            valid_pos_result[0], valid_pos_result[1], valid_pos_result[2]))
        print('[Test set] Precision: {}\tRecall: {}\tNDCG: {}'.format(
            test_pos_result[0], test_pos_result[1], test_pos_result[2]))

        for each_top in range(len(topN)):
            if output_value_list[each_top]['best_valid_recall'] < valid_pos_result[1][each_top]:
                output_value_list[each_top]['best_valid_recall'] = valid_pos_result[1][each_top]
                output_value_list[each_top]['this_time_valid_precision'] = valid_pos_result[0][each_top]
                output_value_list[each_top]['this_time_valid_recall'] = valid_pos_result[1][each_top]
                output_value_list[each_top]['this_time_valid_ndcg'] = valid_pos_result[2][each_top]
                output_value_list[each_top]['test_pos_recall'] = test_pos_result[1][each_top]
                output_value_list[each_top]['test_pos_precision'] = test_pos_result[0][each_top]
                output_value_list[each_top]['test_pos_ndcg'] = test_pos_result[2][each_top]
                best_epoch[each_top] = model.global_epoch_step.eval()
                model.save(sess, 'save_path/ckptTop' + str(topN[each_top]))  # save best
                best_recom_valid = valid_user_pred
                best_recom_test = test_user_pred
            # model.save(sess, 'save_path/ckptTop'+str(topN[each_top])+'_'+str(model.global_epoch_step.eval()))

        save_to = '_pos_epoch_' + str(model.global_epoch_step.eval()) + '__global_step_' + str(model.global_step.eval())

        print('Epoch %d DONE\tCost time: %.2f' % (model.global_epoch_step.eval(), time.time() - start_time))
        print('[now best epoch]', best_epoch)
        sys.stdout.flush()
        model.global_epoch_step_op.eval()

    save_to_json(best_recom_valid, 'save_path/valid_item', topN)
    save_to_json(best_recom_test, 'save_path/test_item', topN)

    print('[model name]', args.model_name)
    print('[data]', args.usedata)
    print('[epochs]', args.epochs)
    print('[topN]', args.topN)
    print('[learning rate]', args.lr)
    print('[train batch size]', args.train_batch_size)

    for each_top in range(len(topN)):
        print('----' * 5)
        print('[topN]', topN[each_top])
        print('[best epoch]', best_epoch[each_top])

        print('[best_ndcg in valid set]', output_value_list[each_top]['best_valid_ndcg'])
        print('[precision in valid set]', output_value_list[each_top]['this_time_valid_precision'])
        print('[now_best_recall in valid set]', output_value_list[each_top]['this_time_valid_recall'])

        print('[test.pos][precision]', output_value_list[each_top]['test_pos_precision'])
        print('[test.pos][recall]', output_value_list[each_top]['test_pos_recall'])
        print('[test.pos][ndcg]', output_value_list[each_top]['test_pos_ndcg'])
        print('----' * 5)

        sys.stdout.flush()
