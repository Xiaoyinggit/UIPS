import numpy as np
import tensorflow.compat.v1 as tf
import math
import os
import time
import matplotlib.pyplot as plt
import sys
from sklearn import metrics


def candidate_ranking(sess, model, mask, test_set, all_item, topN, log_item,mask_valid,isTest=False):
    # 实现top,并排序
    def partition_arg_topK(matrix, K, axis=0):
        """
        perform topK based on np.argpartition
        :param matrix: to be sorted
        :param K: select and sort the top K items
        :param axis: 0 or 1. dimension to be sorted.
        :return:
        """
        a_part = np.argpartition(matrix, K, axis=axis)
        if axis == 0:
            row_index = np.arange(matrix.shape[1 - axis])
            a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
            return a_part[0:K, :][a_sec_argsort_K, row_index]
        else:
            column_index = np.arange(matrix.shape[1 - axis])[:, None]
            a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
            return a_part[:, 0:K][column_index, a_sec_argsort_K]

    user_pred = []
    groundTruth_all_click = []
    groundTruth_pos = []
    auc = []
    for u in range(len(test_set)):
        groundTruth_all_click.append(test_set[u][2])
        groundTruth_pos.append(test_set[u][3])

    batch_size = 512
    num_of_data = len(test_set)
    num_batch = num_of_data//batch_size
    # 最后还有一些不在num_batch*size中的
    for i in range(num_batch):
        user = [test_set[j][0] for j in range(i*batch_size,(i+1)*batch_size)]
        hist_t = [test_set[j][1] for j in range(i * batch_size, (i + 1) * batch_size)]
        sl_t = [len(test_set[j][1]) for j in range(i*batch_size,(i+1)*batch_size)]
        sl = [max(sl_t) for j in range(len(sl_t))]
        hist = [[0 for j in range(sl[0])]for u in range(batch_size)]
        gr_pos = [test_set[j][3] for j in range(i*batch_size, (i+1)*batch_size)]
        for u in range(batch_size):
            for item in range(len(hist_t[u])):
                hist[u][item] = hist_t[u][item]
        inTrainSet = [[0 for j in range(len(all_item))] for u in range(batch_size)]
        for u in range(batch_size):
            user_id = user[u]
            for j in range(len(mask[user_id])):
                inTrainSet[u][mask[user_id][j]] = -9999
        if isTest:
            for u in range(batch_size):
                user_id = user[u]
                if len(mask_valid)==0:
                    continue
                for j in range(len(mask_valid[user_id])):
                    inTrainSet[u][mask_valid[user_id][j]] = -9999
        prediction = model.run_evaluate_user(sess,user,hist,sl)[0] + inTrainSet # [1,512,item_count]
        result = partition_arg_topK(-prediction, topN[-1], axis=1)
        user_pred.extend(result.tolist())
    # 对于剩下来的user
    start = num_batch * batch_size
    end = len(test_set)
    user = [test_set[j][0] for j in range(start, end)]
    hist_t = [test_set[j][1] for j in range(start, end)]
    sl_t = [len(test_set[j][1]) for j in range(start, end)]
    sl = [max(sl_t) for j in range(len(sl_t))]
    hist = [[0 for j in range(sl[0])] for u in range(start, end)]
    gr_pos = [test_set[j][3] for j in range(start, end)]
    for u in range(len(hist_t)):
        for item in range(len(hist_t[u])):
            hist[u][item] = hist_t[u][item]
    inTrainSet = [[0 for j in range(len(all_item))] for u in range(start, end)]
    for u in range(len(user)):
        user_id = user[u]
        for j in range(len(mask[user_id])):
            inTrainSet[u][mask[user_id][j]] = -9999
    if isTest:
        for u in range(len(user)):
            user_id = user[u]
            if len(mask_valid) == 0:
                continue
            for j in range(len(mask_valid[user_id])):
                inTrainSet[u][mask_valid[user_id][j]] = -9999

    prediction = model.run_evaluate_user(sess,user,hist,sl)[0]+ inTrainSet
    result = partition_arg_topK(-prediction, topN[-1], axis=1)
    user_pred.extend(result.tolist())

    precision, recall, NDCG, test_popularity = computeTopNAccuracy(groundTruth_all_click, user_pred, topN, log_item)
    pos_precision, pos_recall, pos_NDCG, pos_test_popularity = computeTopNAccuracy(groundTruth_pos, user_pred, topN, log_item)
    return [precision, recall, NDCG, test_popularity], [pos_precision, pos_recall, pos_NDCG, pos_test_popularity], user_pred,0.0




def computeTopNAccuracy(GroundTruth, predictedIndices, topN, log_item):
    precision = []
    recall = []
    NDCG = []
    test_popularity = []


    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        test_popularity_top = {}
        temp_pool_ratio = {}
        temp_count_pool_ratio = {}
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:  # 集合中的关于这个user的数据
                userHit = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                for j in range(topN[index]):  # 10 20 50 100
                    item = predictedIndices[i][j]
                    if item not in log_item.keys():
                        key = 0.5
                    else:
                        key = log_item[item]

                    if key not in test_popularity_top.keys():
                        test_popularity_top[key] = 1
                    else:
                        test_popularity_top[key] += 1
                    if item in GroundTruth[i]:  # 预测的在这个里面
                        # if Hit!
                        dcg += 1.0 / math.log2(j + 2)  # 衰减
                        userHit += 1
                    if idcgCount > 0:  # 加的数量跟grandTruth有关
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1
                if (idcg != 0):
                    ndcg += (dcg / idcg)
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])
                sumForNdcg += ndcg

        for key in temp_pool_ratio.keys():
            temp_pool_ratio[key] /= temp_count_pool_ratio[key]

        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        test_popularity.append(test_popularity_top)

    return precision, recall, NDCG, test_popularity


def plot_popularity(popularity, path, topN):
  fig = plt.figure(figsize=(12,4))
  plt.title('popular distirbution')
  width = 0
  for j in range(len(popularity)):
    test_bucket = popularity[j]
    x = [i for i in test_bucket.keys()]
    x = list(np.sort(x))
    y = [test_bucket[i] for i in x]
    plt.xticks(x)
    plt.xlim(0,10)
    plt.xlabel('log iteraction')
    plt.ylabel('#impression')
    plt.bar(x,y,width = 0.3,alpha = 0.5,label='top'+str(topN[j]))
    width += 0.05
    plt.legend()
  plt.savefig('save_path/top'+str(topN)+path+'.png')
  plt.close()


def plot_extra_info(item, info, log_item, save_path, name,ifplotnum=False):
    print('[start plot{}]'.format(name))
    sys.stdout.flush()
    start_time = time.time()
    fig = plt.figure(figsize=(12, 4))
    plt.title(name)
    bucket = {}
    bucket_num = {}
    if len(item)!=len(info):
        print('error')
    for i in range(len(item)):
        if item[i] in log_item.keys():
            belong_bucket = log_item[item[i]]
        else:
            belong_bucket = 0.5
        if belong_bucket not in bucket.keys():
            bucket[belong_bucket] = info[i]
            bucket_num[belong_bucket] = 1
        else:
            bucket[belong_bucket] += info[i]
            bucket_num[belong_bucket] += 1
    x = [i for i in bucket.keys()]
    x = list(np.sort(x))
    if ifplotnum:
        y = [bucket_num[i] for i in x]
    else:
        y = [bucket[i]/bucket_num[i] for i in x]
    plt.xticks(x)
    plt.xlim(0, 10)
    plt.xlabel('log iteraction')
    plt.ylabel('sum of ' + name)
    plt.bar(x, y, width=0.3, alpha=0.5)
    if ifplotnum:
        plt.savefig('save_path/number_' + save_path+ '.png')
    else:
        plt.savefig('save_path/' + save_path+ '.png')
    plt.close()
    print('[end plot{}]'.format(name))
    print('cost:',time.time()-start_time)
    sys.stdout.flush()

