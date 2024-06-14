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
from pi.model.offpolicy_uncertainty import *
from pi.model.pi_capping import *


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="no_off_policy")
parser.add_argument("--optimal_beta_model_path", default="lr_0.00001_size_100/save_path/ckptTop20")
parser.add_argument("--epochs", default='5')
parser.add_argument("--topN", default='[5]')
parser.add_argument("--lr", default='0.0001')
parser.add_argument("--train_batch_size", default='512')
parser.add_argument("--usedata", default='Wiki_beta_hat')
parser.add_argument("--capping", default='1')
parser.add_argument("--hyper", default='0.65')
parser.add_argument("--random_seed", default='1234')
parser.add_argument(
    '--UIPS_para',
    type=lambda x: {k:float(v) for k,v in (i.split('-') for i in x.split(','))},
    default='0-0',
    help='comma-separated field-position pairs, e.g. Date-0,Amount-2,Payee-5,Memo-9'
)


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
print('[optimal_beta_model_path]', args.optimal_beta_model_path)
print('[hyperparameter]',args.hyper)
print('[data]', args.usedata)
print('[epochs]', args.epochs)
print('[topN]', args.topN)
print('[learning rate]', args.lr)
print('[train batch size]', args.train_batch_size)
print('[if capping]',args.capping)
print('[UIPS_para]', args.UIPS_para)

data_dict = loadData(args.usedata)

beta = ImportGraph(args.optimal_beta_model_path, data_dict)
f = open('summary/seed_'+args.random_seed+'/model_'+args.model_name+'.txt','w')


gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    summary_writer = tf.summary.FileWriter('./tf_summary/', sess.graph)
   
    if args.model_name == 'capping':
        model = Pi_capping(data_dict['user_count'], data_dict['item_count'], eval(args.capping),random_seed=random_seed)   
    elif args.model_name == 'UIPS':
        model = UIPS(data_dict['user_count'], data_dict['item_count'], para=args.UIPS_para,random_seed=random_seed)
    else:
        print('[Main] Unknown model.')
        raise AssertionError

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    output_value_list = [getInfo() for i in range(len(topN))]
    best_epoch = [0 for _ in range(len(topN))]
    train_set = data_dict['train_set']
    for epoch in range(eval(args.epochs)):
        random.shuffle(train_set)
        loss_sum = 0.0
        count = 0
        start_time = time.time()
        for _, uij in DataInputWithProb(train_set, train_batch_size):
            count += 1
            beta_prob, beta_uncertainty = beta.getProb(uij)  
            loss = model.train(sess, uij, lr,beta_prob[0],beta_uncertainty[:,0],summary_writer)
            loss_sum += loss
        print('Epoch {} Global_step {}\t''[joint]Train_loss: {}\t'.format(
            model.global_epoch_step.eval(), model.global_step.eval(),
            loss_sum / count))
        sys.stdout.flush()

        valid_pos_result, valid_user_pred = candidate_ranking(sess, model, data_dict['valid_set'], topN, data_dict['log_item'])
        test_pos_result, test_user_pred = candidate_ranking(sess, model,data_dict['test_set'], topN, data_dict['log_item'])
        f.write('epoch_'+str(model.global_epoch_step.eval())+'\t'+str(valid_pos_result[0][0])+'\t'+str(valid_pos_result[1][0])+'\t'+str(valid_pos_result[2][0])+'\n')
        f.write('epoch_'+str(model.global_epoch_step.eval())+'\t'+str(test_pos_result[0][0])+'\t'+str(test_pos_result[1][0])+'\t'+str(test_pos_result[2][0])+'\n')


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
    print('[hyperparameter]', args.hyper)
    print('[data]', args.usedata)
    print('[epochs]', args.epochs)
    print('[topN]', args.topN)
    print('[learning rate]', args.lr)
    print('[train batch size]', args.train_batch_size)
    print('[if capping]', args.capping)
    f.write('best res\t'+str(output_value_list[0]['test_pos_precision'])+'\t'+str(output_value_list[0]['test_pos_recall'])+'\t'+str(output_value_list[0]['test_pos_ndcg']))
    f.close()

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

