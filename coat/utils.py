import pickle
import math
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow.compat.v1 as tf






def getInfo():
    output_value = {'best_valid_recall': -999,'best_test_recall':-999,'best_valid_ndcg': -999,
                    'this_time_valid_precision': -999, 'this_time_valid_ndcg': -999,
                    'test_pos_recall': -999, 'test_pos_precision': -999, 'test_pos_ndcg': -999,
                    'test_click_recall': -999, 'test_click_precision': -999, 'test_click_ndcg': -999,
                    'valid_uauc': -999, 'test_uauc': -999,
                    }
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
def save_to_json2(ori_data,name,topN):
    ori_data = np.array(ori_data)
    ori_data = ori_data.astype(np.float)
    for top in topN:
        data = ori_data[:,:top]
        np.savetxt(name+'top'+str(top)+".txt", data, fmt = '%d', delimiter = ',')

def loadData(dataset):
    data_path = 'data/coat_dataset.pkl'
    with open(data_path, 'rb') as f:
        train_set = pickle.load(f)
        valid_set = []
        #valid_set_old = pickle.load(f)
        test_set = pickle.load(f)
        a = pickle.load(f)
        print(a)
        user_count, item_count =a
        interaction = pickle.load(f)
        user_feature_list = pickle.load(f)
        item_feature_list =pickle.load(f)
    return {'train_set':train_set,'valid_set':valid_set,'test_set':test_set,
           'user_count':user_count, 'item_count':item_count, 'interaction':interaction,
            'user_feature':user_feature_list,'item_feature':item_feature_list}