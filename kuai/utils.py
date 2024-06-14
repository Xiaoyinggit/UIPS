import pickle
import math
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow.compat.v1 as tf





def getInfo():
    output_value = {'best_valid_recall': -999,'best_valid_ndcg':-999,
                    'this_time_valid_precision': -999, 'this_time_valid_ndcg': -999,'this_time_valid_recall': -999,
                    'test_pos_recall': -999, 'test_pos_precision': -999, 'test_pos_ndcg': -999,
                    'test_click_recall': -999, 'test_click_precision': -999, 'test_click_ndcg': -999,
                     'valid_uauc': -999, 'test_uauc': -999
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
   if dataset == 'Toutiao':
       data_path = 'toutiao_data/Toutiao_dataset.pkl'
       with open(data_path, 'rb') as f:
         train_set = pickle.load(f)
         valid_set = pickle.load(f)
         test_set = pickle.load(f)
         cate_list = pickle.load(f)
         user_count, item_count, cate_count = pickle.load(f)
         interaction = pickle.load(f)
         mask = pickle.load(f)  # 训练集中出现过的item
         all_item = pickle.load(f)  # 所有物品的id和category [[id,category],...], 有一点冗余了，可以通过cate_list得到categories
         mask_valid = pickle.load(f)
   else:
      # print('error data')
      if dataset == 'Kuai':
        data_path = 'data/Kuai_dataset.pkl'
      with open(data_path, 'rb') as f:
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)
        interaction = pickle.load(f)
        mask = pickle.load(f)  # 训练集中出现过的item
        all_item = pickle.load(f)  # 所有物品的id和category [[id,category],...], 有一点冗余了，可以通过cate_list得到categories
      # mask
      mask_valid = [] # 包含了valid的历史
      if dataset == 'Kuai':
        # mask_valid 不用动，因为用户是不一样的
        interaction[1225] = 1 # 少了一个没有互动过的item 设为1才可以去算log
    # bucket
   log_item = {}
   for item in interaction.keys():
       log_num_iter = math.log(interaction[item])
       temp1 = math.floor(log_num_iter)
       temp2 = log_num_iter - temp1
       log_item[item] = temp1 + round(temp2)/2 + 0.5  # 有些item变为0 了,加上0.5应该好一点
   return {'train_set':train_set,'valid_set':valid_set,'test_set':test_set,'cate_list':cate_list,
           'user_count':user_count, 'item_count':item_count, 'cate_count':cate_count,
           'interaction':interaction,'mask':mask,'mask_valid':mask_valid,
           'all_item':all_item,'log_item':log_item}


def plot_scatter2(value_dict, save_path):
    x = []
    y = []
    for item in value_dict.keys():
        count = value_dict[item]['count']
        x.append(value_dict[item]['pop'])
        y.append(value_dict[item]['beta'] / count)
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel('popularity')
    plt.ylabel('beta')
    plt.savefig(save_path + '.png')
    plt.close()


def plot_scatter(value_dict, save_path):
    x = []
    y = []
    color = []
    for item in value_dict.keys():
        count = value_dict[item]['count']
        x.append(value_dict[item]['pop'])
        y.append(value_dict[item]['uncertainty'] / count)
        color.append(value_dict[item]['beta'] / count)
    plt.scatter(x, y, c=color, alpha=0.6, cmap="Reds")
    plt.xlabel('popularity')
    plt.ylabel('uncertainty')
    plt.colorbar()
    plt.savefig(save_path + '.png')
    plt.close()

def save_ui(u,i,beta,uncertainty,ht,filename):
    with open(filename + '.txt', 'a') as file:
        for j in range(len(ht)):
            file.write(str(u[j][0]))
            file.write('\t')
            file.write(str(i[j][0]))
            file.write('\t')
            file.write(str(beta[j]))
            file.write('\t')
            file.write(str(uncertainty[j]))
            file.write('\t')
            file.write(str(ht[j]))
            file.write('\t')
            file.write('\n')