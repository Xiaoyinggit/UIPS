import pickle
import random
import pandas as pd
import numpy as np
random.seed(1234)
import sys
import time
rootpath="../"

# read
print("Loading big matrix...")
big_matrix = pd.read_csv(rootpath + "raw_data/kuai/big_matrix.csv")
print("Loading small matrix...")
small_matrix = pd.read_csv(rootpath + "raw_data/kuai/small_matrix.csv")
print("Loading item features...")
item_feat = pd.read_csv(rootpath + "raw_data/kuai/item_feat.csv")
item_feat["feat"] = item_feat["feat"].map(eval)
print("All data loaded.")
sys.stdout.flush()

# count number
user_count = big_matrix['user_id'].nunique()
item_count = item_feat['video_id'].nunique()
temp_item = pd.DataFrame()
temp_item['feat'] = item_feat['feat'].map(lambda x:x[-1])
cate_count = temp_item['feat'].nunique()
print('#user:',user_count)
print('#item:',item_count)
print('#cate:',cate_count)
sys.stdout.flush()

# get need data
def getRatingInFo(df,threshold):
    user_video = df[['user_id','video_id','watch_ratio','timestamp']]
    user_video.loc[user_video['watch_ratio']<threshold,['reward']] = 0
    user_video.loc[user_video['watch_ratio']>=threshold,['reward']] = 1
    user_video = user_video[['user_id','video_id','reward','timestamp']]
    return user_video
threshold = 0.7
big_train = getRatingInFo(big_matrix,threshold)
small_test = getRatingInFo(small_matrix,threshold)


# build dataset
# 对于small中有的数据, 找出测试集验证集中的user，负采样的时候不要去采用到测试集里面的物品
start_time = time.time()
print('start build train dataset')
sys.stdout.flush()
user_in_small = small_test['user_id'].value_counts().index.tolist()
user_in_small.sort()
num_user_in_small = len(small_test['user_id'].value_counts())

# 根据big创建train
all_old_hist = {}
train_set = []
#mask = []  # 可以不用这个东西了,这里弄错了,mask只应该mask的是整个大矩阵-小矩阵
# hist_small = {}  # 用来放small中的user的历史
#hist_index = 0
for user,hist in big_train.groupby('user_id'):
    #mask_user = []
    pos_list = hist.loc[hist['reward'] > 0, 'video_id'].tolist()  # reward = 1
    click_list = hist['video_id'].tolist()  # pv = 1
    time_list = hist['timestamp'].tolist()
    rating_list = hist['reward'].tolist()
    # 创建测试集验证集历史,按照user的顺序去创建历史
    # if hist_index<num_user_in_small and user_in_small[hist_index] == user:
    #     hist_small[user] = pos_list
    #     hist_index+=1

    # 不去采样出现在small里面的item
    item_in_small_list = list(small_test.loc[small_test['user_id']==user]['video_id'])
    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list or neg in item_in_small_list:
            neg = random.randint(0, item_count-1)
        return neg   # 负采样
    neg_list = [gen_neg() for i in range(len(click_list))]

    hist_l = []
    pos_index = 0
    train_set.append((user, [], click_list[0], rating_list[0], 1,time_list[0]))
    train_set.append((user, [], neg_list[0], 0, 0,time_list[0]))
    for i in range(1,len(click_list)):
        this_time_click_list = click_list[:i]
        if pos_index< len(pos_list) and this_time_click_list[-1] == pos_list[pos_index]:
            pos_index += 1
            hist_l.append(this_time_click_list[-1])
        temp_hist = [item for item in hist_l]
        if len(temp_hist)>20:
            temp_hist = temp_hist[-20:]
        train_set.append((user, temp_hist, click_list[i], rating_list[i], 1,time_list[i]))
        train_set.append((user, temp_hist, neg_list[i], 0, 0,time_list[i]))
        # mask_user.append(click_list[i])

    # mask.append(mask_user)
    all_old_hist[user]=temp_hist

print('end build train dataset,cost {}'.format(time.time()-start_time))
sys.stdout.flush()

start_time = time.time()
print('start build test dataset')
sys.stdout.flush()
# 生成随机数200个index，作为valid_set
num_of_valid_set = 200
userin_small_id = small_test['user_id'].value_counts().keys().tolist()
valid_index = random.sample(userin_small_id,num_of_valid_set)
valid_index.sort()

# 这里生成mask
mask = []
# for i in range(1411):
#     id = small_matrix['user_id'].value_counts().keys().tolist()[i]
#     num_item = len(small_matrix.loc[small_matrix['user_id']==id,'video_id'].tolist())
#     if num_item == 3327:
#         print(id)
#         break
# 4681包含了所有的small matrix里面的item
item_in_small = small_matrix.loc[small_matrix['user_id']==4681,'video_id'].tolist()
mask_in_small = []
for i in range(item_count):
    if i not in item_in_small:
        mask_in_small.append(i)
for u in range(user_count):
    if u not in userin_small_id:
        mask.append(range(item_count))
    else:
        mask.append(mask_in_small)


index = 0
valid_set = []
test_set = []
for user, hist in small_test.groupby('user_id'):
    old_hist = all_old_hist[user]  # 训练集中的数据  # 训练集中的数据
    click_item = hist['video_id'].tolist()
    pos_item = hist['video_id'].loc[hist['reward']==1].tolist()
    if index<num_of_valid_set and user==valid_index[index]:
        valid_set.append([user,old_hist,click_item,pos_item])
        index+=1
    else:
        test_set.append([user,old_hist,click_item,pos_item])
print('end build test dataset,cost {}'.format(time.time()-start_time))
sys.stdout.flush()
# 对齐cate_list
# 不对齐tf.gather那块会报错


def MultiCate2(l,num_cate):
    newl = [0 for i in range(num_cate)]
    for i in l:
        newl[i] = 1/len(l)
    return newl
item_feat['feat'] = item_feat["feat"].map(lambda x:MultiCate2(x,cate_count))
cate_list = [item_feat['feat'][item] for item in item_feat['video_id']]

interaction = big_matrix['video_id'].value_counts()
all_item = item_feat.values  # 没什么用，后面就用到了他的长度，即cate_count
with open('../data/Kuai_dataset.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(interaction, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(mask, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(all_item, f, pickle.HIGHEST_PROTOCOL)

with open('../data/Kuai_dataset_info.txt', 'w') as f:
  f.write("num_user: "+str(user_count)+'\n')
  f.write("num_item: " + str(item_count)+'\n')
  f.write("num_train_set: "+str(len(train_set))+'\n')
  f.write("num_valid_set: " + str(len(valid_set)) + '\n')
  f.write("num_test_set: " + str(len(test_set))+'\n')
