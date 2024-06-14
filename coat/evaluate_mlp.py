import numpy as np
import tensorflow.compat.v1 as tf
import math
import os
import time
import matplotlib.pyplot as plt
import sys
from sklearn import metrics


def candidate_ranking(sess, model, test_set, topN,batch_size=128):
    # 实现top,并排序
    def partition_arg_topK(matrix, K, axis=0):
        a_part = np.argpartition(matrix, K, axis=axis)
        if axis == 0:
            row_index = np.arange(matrix.shape[1 - axis])
            a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
            return a_part[0:K, :][a_sec_argsort_K, row_index]
        else:
            column_index = np.arange(matrix.shape[1 - axis])[:, None]
            a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
            return a_part[:, 0:K][column_index, a_sec_argsort_K]  # 返回的是index

    user_pred = []
    groundTruth_pos = []
    for u in range(len(test_set)):
        groundTruth_pos.append(test_set[u][3])
    auc = []
    batch_size = batch_size
    num_of_data = len(test_set)
    num_batch = num_of_data//batch_size
    # 最后还有一些不在num_batch*size中的
    for i in range(num_batch):
        user = [test_set[j][0] for j in range(i*batch_size,(i+1)*batch_size)]
        i16 = [test_set[j][1] for j in range(i*batch_size,(i+1)*batch_size)]
        y16 =  [test_set[j][2] for j in range(i*batch_size,(i+1)*batch_size)]
        prediction = model.run_evaluate_user(sess,user,i16)[0]
        index_result = partition_arg_topK(-prediction, topN[-1], axis=1)
        result = []
        for uu, indexs in enumerate(index_result):
            result.append([i16[uu][i] for i in indexs])
        user_pred.extend(result)
    # 对于剩下来的user
    start = num_batch * batch_size
    end = len(test_set)
    user = [test_set[j][0] for j in range(start, end)]
    i16 = [test_set[j][1] for j in range(start, end)]
    y16 = [test_set[j][2] for j in range(start, end)]
    prediction = model.run_evaluate_user(sess,user,i16)[0]
    index_result = partition_arg_topK(-prediction, topN[-1], axis=1)
    result = []
    for uu, indexs in enumerate(index_result):
        result.append([i16[uu][i] for i in indexs])
    user_pred.extend(result)
    pos_precision, pos_recall, pos_NDCG = computeTopNAccuracy(groundTruth_pos, user_pred, topN)
    return [pos_precision, pos_recall, pos_NDCG, 0.0], user_pred


def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:  # 集合中的关于这个user的数据
                userHit = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                for j in range(topN[index]):  # 10 20 50 100
                    item = predictedIndices[i][j]
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
        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))

    return precision, recall, NDCG