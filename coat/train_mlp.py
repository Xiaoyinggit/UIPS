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
from evaluate_mlp import *
from utils import *

from Model.offpolicy_capping import Capping
from Model.offpolicy_UIPS import UIPS





parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="no_off_policy")
parser.add_argument("--epochs", default='5')
parser.add_argument("--topN", default='[5]')
parser.add_argument("--lr", default=0.0001)
parser.add_argument("--train_batch_size", default='512')
parser.add_argument("--usedata", default='Kuai')
parser.add_argument("--capping", default='1')
parser.add_argument("--hyper", default='0.65')
parser.add_argument(
    '--UIPS_para',
    type=lambda x: {k:float(v) for k,v in (i.split('-') for i in x.split(','))},
    default='0-0',
    help='comma-separated field-position pairs, e.g. Date-0,Amount-2,Payee-5,Memo-9'
)
parser.add_argument("--random_seed", default='1234')



args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random_seed = eval(args.random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

train_batch_size = eval(args.train_batch_size)
topN = eval(args.topN)
lr = eval(args.lr)

hyperparameter = {}

print('[model name]',args.model_name)
print('[hyperparameter]',args.hyper)
print('[data]',args.usedata)
print('[epochs]',args.epochs)
print('[topN]',args.topN)
print('[learning rate]',args.lr)
print('[train batch size]',args.train_batch_size)
print('[if capping]',args.capping)


data_dict = loadData(args.usedata)
f_test = open('summary/seed_'+args.random_seed+'/model_'+args.model_name+'_test.txt','w')
f_valid = open('summary/seed_'+args.random_seed+'/model_'+args.model_name+'_valid.txt','w')



gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  summary_writer = tf.summary.FileWriter('./tf_summary/', sess.graph)

  if args.model_name == 'capping':
    model = Capping(data_dict['user_count'], data_dict['item_count'],data_dict['user_feature'],data_dict['item_feature'], eval(args.capping),random_seed=random_seed)
  elif args.model_name == 'UIPS':
    model = UIPS(user_count=data_dict['user_count'], item_count= data_dict['item_count'], user_features=data_dict['user_feature'], item_features=data_dict['item_feature'], para=args.UIPS_para,random_seed=random_seed) 
  else:
    print('[Main] Unknown model.')
    raise AssertionError


  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  output_value_list = [getInfo() for i in range(len(topN))]
  best_epoch = [0 for i in range(len(topN))]
  train_set = data_dict['train_set']

  for epoch in range(eval(args.epochs)):
    random.shuffle(train_set)
    loss_sum = 0.0
    loss_pi_sum = 0.0
    loss_beta_sum = 0.0
    count = 0
    start_time = time.time()
    if epoch == 0:
        for _, uij in DataInput(train_set, train_batch_size,args.model_name):
          count += 1
          loss = model.train(sess, uij, lr, summary_writer)
          loss_sum += loss
        print('Epoch {} Global_step {}\t''[joint]Train_loss: {}\t'.format(
        model.global_epoch_step.eval(), model.global_step.eval(),
        loss_sum / count))
        sys.stdout.flush()
    else:
        count = 0
        for _, uij in DataInput(train_set, train_batch_size, args.model_name):
          count += 1
          loss_beta = model.train_beta(sess, uij, lr, summary_writer)
          loss_beta_sum += loss_beta
        print('Epoch {} Global_step {}\t''[beta]Train_loss: {}\t'.format(
          model.global_epoch_step.eval(), model.global_step.eval(),
          loss_beta_sum / count))
        sys.stdout.flush()
        count = 0
        for _, uij in DataInput(train_set, train_batch_size, args.model_name):
          count+=1
          loss_pi = model.train_pi(sess, uij, lr, summary_writer)
          loss_pi_sum += loss_pi
        print('Epoch {} Global_step {}\t''[pi]Train_loss: {}\t'.format(
          model.global_epoch_step.eval(), model.global_step.eval(),
          loss_pi_sum / count))
        sys.stdout.flush()

    valid_pos_result, valid_user_pred = candidate_ranking(sess, model, data_dict['valid_set'][0:50], topN)  
    test_pos_result, test_user_pred = candidate_ranking(sess, model, data_dict['test_set'][50,:], topN)

    f_valid.write('epoch_'+str(model.global_epoch_step.eval())+'\t'+str(valid_pos_result[0][0])+'\t'+str(valid_pos_result[1][0])+'\t'+str(valid_pos_result[2][0])+'\n')
    f_test.write('epoch_'+str(model.global_epoch_step.eval())+'\t'+str(test_pos_result[0][0])+'\t'+str(test_pos_result[1][0])+'\t'+str(test_pos_result[2][0])+'\n')

    print('[Valid pos set] Precision: {}\tRecall: {}\tNDCG: {}\tuauc:{}'.format(
          valid_pos_result[0], valid_pos_result[1], valid_pos_result[2],valid_pos_result[3]))
    print('[Test pos set] Precision: {}\tRecall: {}\tNDCG: {}\tuauc:{}'.format(
          test_pos_result[0], test_pos_result[1], test_pos_result[2],test_pos_result[3]))
    sys.stdout.flush()

    model.global_epoch_step_op.eval()
    if args.model_name in ['UIPS']:
      model.initA.eval()

  f_valid.close()
  f_test.close()

