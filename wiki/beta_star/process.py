import pickle
import random
import pandas as pd
import numpy as np
random.seed(1234)

train_data_count=feature_count=item_count=test_data_count=0
train_df = {'uid':[],'features':[],'groundTruth':[]}
valid_df = {'uid':[],'features':[],'groundTruth':[]}
test_df = {'uid':[],'features':[],'groundTruth':[]}
item_pop_df = {'gid':[]}

index = -1
def getOntHotGR(groundTruth,item_count):
    tmp = [0]*item_count
    for item in groundTruth:
        tmp[item] = 1
    count = sum(tmp)
    for i in range(len(tmp)):
        if tmp[i]:
            tmp[i]/=count
    return tmp

with open('wiki/train.txt','r') as f:
    for line in f:
        if index==-1:
            train_data_count,feature_count,item_count = map(int,line.replace('\n','').split(' '))
            valid_index = random.sample(list(range(train_data_count)),3000)
        elif index not in valid_index:
            data = line.replace('\n','').split(' ')
            groundTruth = list(map(int,data[0].split(',')))
            for item in groundTruth:
                item_pop_df['gid'].append(item)
            features = [0]*feature_count
            for i in range(1,len(data)):
                ind,val = data[i].split(':')
                features[int(ind)]=float(val)
            train_df['uid'].append(index)
            train_df['features'].append(features)
            train_df['groundTruth'].append(getOntHotGR(groundTruth,item_count))
        else:
            data = line.replace('\n','').split(' ')
            groundTruth = list(map(int,data[0].split(',')))
            features = [0]*feature_count
            for i in range(1,len(data)):
                ind,val = data[i].split(':')
                features[int(ind)]=float(val)
            valid_df['uid'].append(index)
            valid_df['features'].append(features)
            valid_df['groundTruth'].append(groundTruth)
        index+=1

cur = -1
with open('wiki/test.txt','r') as f:
    for line in f:
        if cur==-1:
            test_data_count,feature_count,item_count = map(int,line.replace('\n','').split(' '))
            cur+=1
        else:
            data = line.replace('\n','').split(' ')
            groundTruth = list(map(int,data[0].split(',')))
            features = [0]*feature_count
            for i in range(1,len(data)):
                ind,val = data[i].split(':')
                features[int(ind)]=float(val)
            test_df['uid'].append(index)
            test_df['features'].append(features)
            test_df['groundTruth'].append(groundTruth)
            index+=1

train_df = pd.DataFrame(train_df)
valid_df = pd.DataFrame(valid_df)
test_df = pd.DataFrame(test_df)
item_pop_df = pd.DataFrame(item_pop_df)

train_set = []
for user,hist in train_df.groupby('uid'):
    fea = hist['features'].tolist()[0]
    pos_list = hist['groundTruth'].tolist()[0]  # reward = 1
    train_set.append((fea,pos_list))
valid_set = []
for user,hist in valid_df.groupby('uid'):
    fea = hist['features'].tolist()[0]
    pos_list = hist['groundTruth'].tolist()[0]  # reward = 1
    valid_set.append((fea,pos_list))
test_set = []
for user,hist in test_df.groupby('uid'):
    fea = hist['features'].tolist()[0]
    pos_list = hist['groundTruth'].tolist()[0]  # reward = 1
    test_set.append((fea,pos_list))

user_count = 6616+14146
interaction = item_pop_df['gid'].value_counts()


with open('data/full_Wiki.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, feature_count), f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(interaction, f, pickle.HIGHEST_PROTOCOL)


with open('data/full_Wiki_info.txt', 'w') as f:
  f.write("num_user: "+str(user_count)+'\n')
  f.write("num_item: " + str(item_count)+'\n')
  f.write("num_train_set: "+str(len(train_set))+'\n')
  f.write("num_valid_set: " + str(len(valid_set)) + '\n')
  f.write("num_test_set: " + str(len(test_set))+'\n')

