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


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="Beta_Star")
parser.add_argument("--epochs", default='5')
parser.add_argument("--sample_size", default=100, type=int)
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
oracle_train_set = []
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
    distinct_sn = 0
    for _, uij in DataInput(data_dict['train_set'], train_batch_size):
       X_emb, beta_prob = model.run_eval(sess, uij, lr)

       for i in range(len(X_emb)):
           cur_x = X_emb[i]
           label = uij[1][i]
           action_set = list(np.arange(0, len(label)))
           
           pos_l = [i for i in range(len(label)) if label[i] > 0 ]
           uid_pos.append([uid,cur_x,len(pos_l)])
           uid+=1
           #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
           #print('pos: ',len(pos_l),'sum: ', np.sum( beta_prob[i]), '\n, pred_prob_topk: ', np.sort(beta_prob[i])[-20:],
           #      '\npred_prob_smallk: ', np.sort(beta_prob[i])[0:20])
           #print('pos_prob', [beta_prob[i][k] for k in pos_l])
           selected_a = random.choices(action_set, weights = beta_prob[i], k=args.sample_size)
           selected_a_set = set(selected_a)
           distinct_sn += len(selected_a_set)
           for a in selected_a:
               lab = 1 if label[a]>0 else 0

               new_train_set.append([cur_x, a, lab, 1,beta_prob[i][a]]) # observed.

               # generate negative samples
               na = random.sample(action_set, k=1)[0]
               while na not in selected_a_set:
                   na = random.sample(action_set, k=1)[0]
               new_train_set.append([cur_x, na, 0, 0, 0]) # display=0的不更新pi
           # generate oracle train set
           for a in selected_a_set:
               lab = 1 if label[a]>0 else 0
               oracle_train_set.append([cur_x, a, lab, 1, beta_prob[i][a]])


       count_batch +=1
       print('count_batch: ', count_batch)
    for _, uij in DataInput(data_dict['valid_set'], train_batch_size):
       X_emb, beta_prob = model.run_eval(sess, uij, lr)
       for i in range(len(X_emb)):
           cur_x = X_emb[i]
           new_valid_set.append([cur_x,uij[1][i]])
    for _, uij in DataInput(data_dict['test_set'], train_batch_size):
       X_emb, beta_prob = model.run_eval(sess, uij, lr)
       for i in range(len(X_emb)):
           cur_x = X_emb[i]
           new_test_set.append([cur_x,uij[1][i]])




          

print(' new_train_set: ', len(new_train_set))
with open('data/syn_Wiki_%d_with_prob05.pkl'%(args.sample_size), 'wb') as f:
  pickle.dump(new_train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(new_valid_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(new_test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((data_dict['user_count'], data_dict['item_count'], data_dict['feature_count']), f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(data_dict['interaction'], f, pickle.HIGHEST_PROTOCOL)


print(' oracle_train_set: ', len(oracle_train_set), 'distinct_sn:', distinct_sn)
with open('data/syn_Wiki_%d_with_prob05_oracle.pkl'%(args.sample_size), 'wb') as f:
  pickle.dump(oracle_train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(new_valid_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(new_test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((data_dict['user_count'], data_dict['item_count'], data_dict['feature_count']), f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(data_dict['interaction'], f, pickle.HIGHEST_PROTOCOL)
   
