import numpy as np
import pandas as pd
import random
import pickle
import sys
import time
random.seed(1234)


def transTo(df,threshold):
    temp = df[['uid','gid','rating']]
    pos_index = temp['rating']>threshold
    temp.loc[pos_index,'rating']=1
    temp.loc[~pos_index,'rating']=0
    return temp

# train
user_rating = []
ori_df = {'uid':[],'gid':[],'rating':[]}
user = 0
with open('train.ascii','r') as f:
    for line in f:
        line = line.replace('\n','')
        info = list(map(int,line.split(' ')))
        for item,rating in enumerate(info):
            if rating!=0:
                ori_df['uid'].append(user)
                ori_df['gid'].append(item)
                ori_df['rating'].append(rating)
        user+=1
df = pd.DataFrame(ori_df)
user_count = len(df['uid'].unique())
item_count = len(df['gid'].unique())

# test
user_rating = []
ori_df_test = {'uid':[],'gid':[],'rating':[]}
user = -1
with open('test.ascii','r') as f:
    for line in f:
        user += 1
        line = line.replace('\n','')
        info = list(map(int,line.split(' ')))
        for item,rating in enumerate(info):
            if rating!=0:
                ori_df_test['uid'].append(user)
                ori_df_test['gid'].append(item)
                ori_df_test['rating'].append(rating)
df_test = pd.DataFrame(ori_df_test)

df = transTo(df,3)  #>=4
df_test = transTo(df_test,3)

start = time.time()
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
print('end build train dataset,cost {}'.format(time.time()-start))
sys.stdout.flush()


test_set = []
for user,info in df_test.groupby('uid'):
    click_list = info['gid'].tolist()  # pv = 1
    rating_list = info['rating'].tolist()
    pos_list = info.loc[info['rating'] > 0, 'gid'].tolist()  # reward = 1
    test_set.append([user,list(info['gid']),list(info['rating']),pos_list])


# user features
user_features = {'gender':[],'age':[],'location':[],'fashioninterest':[]}
user = -1
with open('user_item_features/user_features.ascii','r') as f:
     for line in f:
        user+=1
        line = line.replace('\n','')
        info = list(map(int,line.split(' ')))
        # user_features['uid'].append(user)  # list的index就是uid
        user_features['gender'].append(info[:2])
        user_features['age'].append(info[2:8])
        user_features['location'].append(info[8:11])
        user_features['fashioninterest'].append(info[11:])
user_features = pd.DataFrame(user_features)
# item features
item_features = {'gender':[],'jackettype':[],'color':[],'onfrontpage':[]}
item = -1
with open('user_item_features/item_features.ascii','r') as f:
     for line in f:
        item+=1
        line = line.replace('\n','')
        info = list(map(int,line.split(' ')))
        # item_features['gid'].append(item)
        item_features['gender'].append(info[:2])
        item_features['jackettype'].append(info[2:18])
        item_features['color'].append(info[18:31])
        item_features['onfrontpage'].append(info[31:])
item_features = pd.DataFrame(item_features)

interaction = df['gid'].value_counts()
user_feature_list = [user_features['gender'].to_list(),user_features['age'].to_list(),user_features['location'].to_list(),user_features['fashioninterest'].to_list()]
item_feature_list =[item_features['gender'].to_list(),item_features['jackettype'].to_list(),item_features['color'].to_list(),item_features['onfrontpage'].to_list()]


with open('data/coat_dataset.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  # pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count), f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(interaction, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(user_feature_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(item_feature_list, f, pickle.HIGHEST_PROTOCOL)

with open('data/coat_dataset_info.txt', 'w') as f:
  f.write("num_user: "+str(user_count)+'\n')
  f.write("num_item: " + str(item_count)+'\n')
  f.write("num_train_set: "+str(len(train_set))+'\n')
  # f.write("num_valid_set: " + str(len(valid_set)) + '\n')
  f.write("num_test_set: " + str(len(test_set))+'\n')
