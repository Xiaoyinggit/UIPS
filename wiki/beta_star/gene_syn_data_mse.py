# coding:utf8
import os
import time
import pickle
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


def sample_valid_bandit_data(data_type, train_batch_size, lr):
    new_data_set = []
    pos_ratio_sum, count = 0, 0
    max_pos_ratio, min_pos_ratio = 0, 1000
    cur_batch = 0
    for _, uij in DataInput(data_dict[data_type], train_batch_size):
       print('cur_batch: ', cur_batch)
       cur_batch +=1
       X_emb, beta_prob = model.run_eval(sess, uij, lr)
       
       for ind in range(len(X_emb)):
           cur_x, pos_l = X_emb[ind], uij[1][ind]
           if abs(np.sum(beta_prob[ind])-1)> 0.0001:
               raise AssertionError
           action_set = list(np.arange(0, data_dict['item_count']))
           cur_x_data = [cur_x, pos_l, beta_prob[ind]]
           label_l = [0 for i in range(data_dict['item_count'])]
           for p in pos_l:
               label_l[p] =1 
           T=0
           while T <20: 
              selected_a = random.choices(action_set, weights = beta_prob[ind], k=args.sample_size)
              samples = []
              for a in selected_a:
                 samples.append((a,label_l[a], beta_prob[ind][a]))
              cur_x_data.append(samples)
              T+=1

           if len(cur_x_data)!=23:
               raise AssertionError
           new_data_set.append(cur_x_data) #[x, pos_l, [(a,r)*n]]
    return new_data_set


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="Beta_Star")
parser.add_argument("--epochs", default='5')
parser.add_argument("--sample_size", default=1000, type=int)
parser.add_argument("--topN", default='[20,50]')
parser.add_argument("--lr", default='0.0001')
parser.add_argument("--train_batch_size", default='512')
parser.add_argument("--usedata", default='Wiki_beta_star')
parser.add_argument(
    '--para',
    type=lambda x: {k:float(v) for k,v in (i.split(':') for i in x.split(','))},
    default='temp:0.5',
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


###########LOAD CKPT PATH#############
optimal_beta_model_path = 'optimal_beta_start_v1/ckptTop20'

gpu_options = tf.GPUOptions(allow_growth=True)
new_train_set = []
new_valid_set = []
new_test_set = []
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    if args.model_name == 'beta_star_v2':
        model = BetaStar_V2(data_dict['user_count'],data_dict['item_count'],data_dict['feature_count'], v2_para=args.para)
    else:
        model = BetaStart(data_dict['user_count'],data_dict['item_count'],data_dict['feature_count'], para=args.para)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    model.restore(sess, optimal_beta_model_path)


    ### first do evaluation 
    valid_pos_result, valid_user_pred = candidate_ranking(sess, model, data_dict['valid_set'], topN, data_dict['log_item'])
    test_pos_result, test_user_pred = candidate_ranking(sess, model,data_dict['test_set'], topN, data_dict['log_item'])
    
    print('[Valid set] Precision: {}\tRecall: {}\tNDCG: {}'.format(
            valid_pos_result[0], valid_pos_result[1], valid_pos_result[2]))
    print('[Test set] Precision: {}\tRecall: {}\tNDCG: {}'.format(
            test_pos_result[0], test_pos_result[1], test_pos_result[2]))


    ##### generate simulate dataset, only involve trainset 
    count_batch = 0
    uid_pos = []
    uid = 0
   
    new_valid_set = sample_valid_bandit_data('valid_set', train_batch_size, lr)
    new_test_set = sample_valid_bandit_data('test_set', train_batch_size, lr)
    
 

print(' new_train_set: ', len(new_train_set), 'train_set: ', len(data_dict['train_set']))
with open('data/syn_Wiki_%d_mse_new_temp_%f.pkl'%(args.sample_size, args.para['temp']), 'wb') as f:
  pickle.dump(new_train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(new_valid_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(new_test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((data_dict['user_count'], data_dict['item_count'], data_dict['feature_count']), f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(data_dict['interaction'], f, pickle.HIGHEST_PROTOCOL)
   
    
# with open('data/uid_pos.pkl'%(args.sample_size), 'wb') as f:
#     pickle.dump(uid_pos, f, pickle.HIGHEST_PROTOCOL)
