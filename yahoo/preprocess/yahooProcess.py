import numpy as np
import pandas as pd
import random
import pickle
import sys
import time
random.seed(1234)

root_path = 'Yahoo_R3/'
# features
user_features = {'uid':[],'Q1':[],'Q2':[],'Q3':[],'Q4':[],'Q5':[],'Q6':[],'Q7':[]}
with open(root_path+'ydata-ymusic-rating-study-v1_0-survey-answers.txt','r') as f:
     user = 0
     for line in f:
        line = line.replace('\n','')
        info = list(map(int,line.split('\t')))
        user+=1
        user_features['uid'].append(user) # 数据集uid是从1开始的
        for i in range(1,8):
            user_features['Q'+str(i)].append(info[i-1])
user_features = pd.DataFrame(user_features)

def toOneHot(x,length):
    newx = [0]*length
    newx[x-1] = 1
    return newx

for i in range(1,7):
    user_features['Q'+str(i)] = user_features['Q'+str(i)].map(lambda x: toOneHot(x,5))
user_features['Q7'] = user_features['Q7'].map(lambda x: toOneHot(x,2))
#train
ori_df = {'uid':[],'gid':[],'rating':[]}
with open(root_path+'ydata-ymusic-rating-study-v1_0-train.txt','r') as f:
    for line in f:
        line = line.replace('\n','')
        info = list(map(int,line.split('\t')))
        ori_df['uid'].append(info[0])
        ori_df['gid'].append(info[1])
        ori_df['rating'].append(info[2])
df = pd.DataFrame(ori_df)
# test
ori_df_test = {'uid':[],'gid':[],'rating':[]}
with open(root_path+'ydata-ymusic-rating-study-v1_0-test.txt','r') as f:
    for line in f:
        line = line.replace('\n','')
        info = list(map(int,line.split('\t')))
        ori_df_test['uid'].append(info[0])
        ori_df_test['gid'].append(info[1])
        ori_df_test['rating'].append(info[2])
df_test = pd.DataFrame(ori_df_test)

def transTo(df,threshold):
    temp = df[['uid','gid','rating']]
    pos_index = temp['rating']>threshold
    temp.loc[pos_index,'rating']=1
    temp.loc[~pos_index,'rating']=0
    return temp

# 训练集中只保留测试集中出现的user
uid_in_test = list(df_test['uid'].unique())
df = df[df['uid'].isin(uid_in_test)]
df = transTo(df, 3)  # >=3
df_test = transTo(df_test, 3)
# 去掉没有train和test里面没有正例的用户
uid_no_pos = set()
for uid in uid_in_test:
    vc = df[df['uid']==uid]['rating'].value_counts()
    if 1 not in vc:
        uid_no_pos.add(uid)
for uid in uid_in_test:
    vc = df_test[df_test['uid']==uid]['rating'].value_counts()
    if 1 not in vc:
        uid_no_pos.add(uid)
uid_no_pos = list(uid_no_pos)
df = df[~df['uid'].isin(uid_no_pos)]
df_test = df_test[~df_test['uid'].isin(uid_no_pos)]
user_features = user_features[~user_features['uid'].isin(uid_no_pos)]
# 对uid重新做一遍映射,从0开始
old_uid = list(df['uid'].value_counts().to_dict().keys())
old_uid.sort()
map_uid = dict(zip(old_uid, range(len(old_uid))))
df['uid'] = df['uid'].map(lambda x: map_uid[x])
df_test['uid'] = df_test['uid'].map(lambda x: map_uid[x])
user_features['uid'] = user_features['uid'].map(lambda x: map_uid[x])

user_count = len(df['uid'].unique())
item_count = len(df['gid'].unique())
print(user_count)
print(item_count)

# data set
# train set [uid,gid,rating,display]
# validate/test set [uid,[gid,gid,...,gid],[label,label,...,label]]

# evaluate的时候，用一个batch的 [uid,[gid,gid,...,gid],[label,label,...,label],[gid,gid,...(label=1,groundTruth)]] 输入，最后一个用来计算recall,precision,ndcg

train_set = []
for user,info in df.groupby('uid'):
    no_neg_sample_test = df_test[df_test['uid'] == user]['gid'].tolist()
    click_list = info['gid'].tolist()  # pv = 1
    rating_list = info['rating'].tolist()
    def gen_neg():
        neg = click_list[0]
        while neg in click_list or neg in no_neg_sample_test:
            neg = random.randint(0, item_count-1)
        return neg  # 负采样
    neg_list = [gen_neg() for i in range(len(click_list))]
    for i in range(len(click_list)):
        item = click_list[i]
        train_set.append((user, item, rating_list[i], 1))
        train_set.append((user, neg_list[i], 0, 0))

all_test_uid = list(df_test['uid'].unique())
# 随机选择5%的用户用做validate set
# valid_uid = random.sample(all_test_uid,int(0.05*len(all_test_uid)))
# 根据groundTruth的数目来分
user_gr = df_test[df_test['rating']==1]['uid'].value_counts()
gr_uid = {}
for user in user_gr.keys():
#     print(user,user_gr[user])
    if user_gr[user] not in gr_uid:
        gr_uid[user_gr[user]] = []
    gr_uid[user_gr[user]].append(user)

valid_uid = []
for k in gr_uid:
    len_gr = gr_uid[k]
    chose_uid = random.sample(len_gr,int(max(0.05*len(len_gr),1)))
    for i in chose_uid:
        valid_uid.append(i)

valid_set = []
test_set = []
for user,info in df_test.groupby('uid'):
    click_list = info['gid'].tolist()  # pv = 1
    rating_list = info['rating'].tolist()
    pos_list = info.loc[info['rating'] > 0, 'gid'].tolist()  # reward = 1
    if user in valid_uid:
        valid_set.append([user,list(info['gid']),list(info['rating']),pos_list])
    else:
        test_set.append([user,list(info['gid']),list(info['rating']),pos_list])

user_feature_list = [user_features['Q'+str(i)].to_list() for i in range(1,8)]
interaction = df['gid'].value_counts()

with open('data/yahoo_dataset_split.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count), f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(interaction, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(user_feature_list, f, pickle.HIGHEST_PROTOCOL)
with open('data/yahoo_dataset_info_split.txt', 'w') as f:
  f.write("num_user: "+str(user_count)+'\n')
  f.write("num_item: " + str(item_count)+'\n')
  f.write("num_train_set: "+str(len(train_set))+'\n')
  f.write("num_valid_set: " + str(len(valid_set)) + '\n')
  f.write("num_test_set: " + str(len(test_set))+'\n')
  f.write("num_test_set: " + str(len(test_set)) + '\n')