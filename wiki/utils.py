import pickle
import math
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow.compat.v1 as tf
from beta_hat.beta_hat_model import *
from beta_star.beta_star_model import *





def getInfo():
    output_value = {'best_valid_recall': -999,'best_valid_ndcg': -999,
                    'this_time_valid_precision': -999, 'this_time_valid_ndcg': -999, 'this_time_valid_recall': -999,
                    'test_recall': -999, 'test_precision': -999, 'test_ndcg': -999}
    return output_value

def save_to_json(ori_data,name,topN):
    ori_data = np.array(ori_data)
    ori_data = ori_data.astype(np.float)
    for top in topN:
        data = ori_data[:,:top]
        data = data.reshape(-1,)
        res = dict(Counter(data))
        with open(name+'top'+str(top)+'.json','w') as f:
            json.dump(res,f)

def loadData(dataset,sample_size='0'):
    interaction = {}
    train_set, valid_set, test_set = [], [], []
    user_count, item_count, feature_count= 0, 0, 0
    if dataset == 'Wiki_beta_star':
        data_path = 'data/full_Wiki.pkl'
        with open(data_path, 'rb') as f:
            train_set = pickle.load(f)
            valid_set = pickle.load(f)
            test_set = pickle.load(f)
            user_count, item_count, feature_count= pickle.load(f)
            interaction = pickle.load(f)
    elif dataset == 'Wiki_beta_hat':
#         data_path = 'wiki_syndata/syn_Wiki_100_1.pkl'
#         data_path = 'wiki_syndata/syn_Wiki_100_2.pkl'
        data_path = 'wiki_syndata/syn_Wiki_100_05.pkl'
        # data_path = 'wiki_syndata/syn_Wiki_100_01.pkl'
        with open(data_path, 'rb') as f:
            train_set = pickle.load(f)
            valid_set = pickle.load(f)
            test_set = pickle.load(f)
            user_count, item_count, feature_count= pickle.load(f)
            interaction = pickle.load(f)
    elif dataset == 'Wiki_beta_hat_percent':
        data_path = 'wiki_syndata/wiki_percent/temp_05_'+str(sample_size)+'.pkl'
        with open(data_path, 'rb') as f:
            train_set = pickle.load(f)
            valid_set = pickle.load(f)
            test_set = pickle.load(f)
            user_count, item_count, feature_count= pickle.load(f)
            interaction = pickle.load(f)
    elif dataset == 'new04data':
        data_path = 'wiki_syndata/wiki_percent/new_syn_Wiki_4p.pkl'
        with open(data_path, 'rb') as f:
            train_set = pickle.load(f)
            valid_set = pickle.load(f)
            test_set = pickle.load(f)
            user_count, item_count, feature_count= pickle.load(f)
            interaction = pickle.load(f)
    else:
        # direclty put path in 
        data_path = dataset
        print('[Loading] Loading From %s'%data_path)
        with open(data_path, 'rb') as f:
            train_set = pickle.load(f)
            valid_set = pickle.load(f)
            test_set = pickle.load(f)
            user_count, item_count, feature_count= pickle.load(f)
            interaction = pickle.load(f)
        print('[Loading] user_count: %d, item_count:%d, feature_count:%d, X_dim:%d'%(user_count, item_count, feature_count, len(train_set[0][0])))

    
    print('[Loading]len_interaction: ', len(interaction))
    log_item = {}
    for item in interaction.keys():
        log_num_iter = math.log(interaction[item])
        temp1 = math.floor(log_num_iter)
        temp2 = log_num_iter - temp1
        log_item[item] = temp1 + round(temp2) / 2 + 0.5  # 有些item变为0 了,加上0.5应该好一点
    return {'train_set': train_set, 'valid_set': valid_set, 'test_set': test_set,
            'user_count': user_count, 'item_count': item_count, 'feature_count': feature_count,
            'interaction': interaction, 'log_item': log_item}

class ImportBetaStarGraph():
    def __init__(self, path, data_dict, beta_star_para):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default() as sess:
            with self.graph.as_default():
#                 self.model = BetaStart(data_dict['user_count'],data_dict['item_count'],data_dict['feature_count'], para=beta_star_para)
                self.model = BetaStart(data_dict['user_count'],data_dict['item_count'],data_dict['feature_count'])
                sess.run(tf.global_variables_initializer())
                self.model.restore(sess, path)

    def getProb(self, uij):
        with self.sess.as_default() as sess:
            with self.graph.as_default():
                _, prob = self.model.run_eval(sess, uij[0], 0.0001)
        return prob

class ImportGraph():
    def __init__(self, path, data_dict):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default() as sess:
            with self.graph.as_default():
                self.model = BetaHat(data_dict['user_count'], data_dict['item_count'],hidden_dim=64)
                sess.run(tf.global_variables_initializer())
                self.model.restore(sess, path)

    def getProb(self, uij):
        with self.sess.as_default() as sess:
            with self.graph.as_default():
                prob = self.model.run_eval(sess, uij[0])
                uncertainty = self.model.getSUncertainty(sess, uij)
        return prob, uncertainty

    def getAllProb(self, uij):
        with self.sess.as_default() as sess:
            with self.graph.as_default():
                prob = self.model.run_eval(sess, uij[0])
                uncertainty = self.model.getAllUncertainty(sess, uij)
        return prob, uncertainty

def plot_scatter(value_dict, save_path):
    bucket = {}
    for item in value_dict.keys():
        p = value_dict[item]['pop']
        if p not in bucket.keys():
            bucket[p] = {'count':0,'uncertainty':0}
        bucket[p]['count']+=1
        bucket[p]['uncertainty'] += value_dict[item]['uncertainty']
    x,y,z = [],[],[]
    print(bucket)
    for key in bucket:
        x.append(key)
        y.append(bucket[key]['uncertainty']/bucket[key]['count'])
        z.append(bucket[key]['count'])
    plt.bar(x,y)
    plt.xlabel('popularity')
    plt.ylabel('uncertainty')
    plt.savefig(save_path + '22.png')
    plt.close()
    plt.bar(x,z)
    plt.xlabel('popularity')
    plt.ylabel('count')
    plt.savefig(save_path + '22.png')
    plt.close()
        
    
    
    

def loadUid_pos():
    data_path = 'wiki_syndata/uid_pos.pkl'
    with open(data_path, 'rb') as f:
        uid_pos = pickle.load(f)
    return uid_pos