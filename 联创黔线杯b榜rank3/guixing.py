# guixing.py
# Author:Andrew Li

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, roc_auc_score

import warnings
import random
from tqdm import tqdm
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
from datetime import datetime


# 读入所有原始数据
train_path_n = '/home/kesci/input/gzlt/train_set/201708n/'
train_path_q = '/home/kesci/input/gzlt/train_set/201708q/'
test_path = '/home/kesci/input/gzlt/test_set/201808/'
def read_data(file_list,names,label=False):
    data = pd.concat([pd.read_csv(f, names=names, sep='\t') for f in file_list])
    if label:
        return data
    else:
        return data.drop('label',axis=1)
# 用户身份属性
user_info = read_data([train_path_n+'201708n1.txt',train_path_q+'201708q1.txt'],['账期','虚拟id','出账收入','label'],True)
# 用户手机终端信息
user_app_info = read_data([train_path_n+'201708n2.txt',train_path_q+'201708q2.txt'],['虚拟id','品牌','终端型号','首次使用时间','末次使用时间','label'])
# 用户漫游行为
user_action = read_data([train_path_n+'201708n3.txt',train_path_q+'201708q3.txt'],['账期','虚拟id','联络圈规模','是否出省','是否出境','label'])
# 用户漫出省份
user_city = read_data([train_path_n+'201708n4.txt',train_path_q+'201708q4.txt'],['账期','虚拟id','漫出省份','label'])
# 用户是否去过景区
# user_label = read_data([train_path_n+'201708n5.txt',train_path_q+'201708q5.txt'],['账期','虚拟id','now_label','label'])
# 用户地理位置
user_geo = read_data([train_path_n+'201708n6.txt',train_path_q+'201708q6.txt'],['日期','时段','虚拟id','经度','纬度','label'])
# 用户app使用情况
user_app_use = read_data([train_path_n+'201708n7.txt',train_path_q+'201708q7.txt'],['账期','虚拟id','APP名称','流量','label'])

test_user_info = read_data([test_path+'2018_1.txt'],['账期','虚拟id','出账收入'],True)
test_user_app_info = read_data([test_path+'2018_2.txt'],['虚拟id','品牌','终端型号','首次使用时间','末次使用时间'],True)
test_user_action = read_data([test_path+'2018_3.txt'],['账期','虚拟id','联络圈规模','是否出省','是否出境'],True)
test_user_city = read_data([test_path+'2018_4.txt'],['账期','虚拟id','漫出省份'],True)
# test_user_label = read_data([test_path+'2018_5.txt'],['账期','虚拟id','now_label'],True)
test_user_geo = read_data([test_path+'2018_6.txt'],['日期','时段','虚拟id','经度','纬度'],True)
test_user_app_use = read_data([test_path+'2018_7.txt'],['账期','虚拟id','APP名称','流量'],True)

# 生成"虚拟id"的label标签
train_label = pd.Series(list(set(user_info.虚拟id))).to_frame()
train_label.columns = ['虚拟id']
print(train_label.shape)
test_label = pd.Series(list(set(test_user_info.虚拟id))).to_frame()
test_label.columns = ['虚拟id']
print(test_label.shape)
# (99000, 1)
# (50200, 1)

train_label.to_csv('./data/train_label.csv')
test_label.to_csv('./data/test_label.csv')

# 提统计特征函数
def ll_get_feature(ykt_jyrz, label):
    for feature in [i for i in ykt_jyrz.columns if i not in ['虚拟id', '账期','label']]:
        if ykt_jyrz[feature].dtype == 'object':
            label = label.merge(ykt_jyrz.groupby(by='虚拟id')[feature].count().reset_index().rename(columns = {feature:'count_'+ feature}), how='left', on='虚拟id')
            label = label.merge(ykt_jyrz.groupby(by='虚拟id')[feature].nunique().reset_index().rename(columns = {feature:'nunique_'+ feature}), how='left', on='虚拟id')
        else:
            label =label.merge(ykt_jyrz.groupby(['虚拟id'])[feature].count().reset_index().rename(columns = {feature:'count_'+ feature}),on='虚拟id',how='left')
            label =label.merge(ykt_jyrz.groupby(['虚拟id'])[feature].nunique().reset_index().rename(columns = {feature:'nunique_'+ feature}),on='虚拟id',how='left')
            label =label.merge(ykt_jyrz.groupby(['虚拟id'])[feature].mean().reset_index().rename(columns = {feature:'mean_'+ feature}),on='虚拟id',how='left')
            label =label.merge(ykt_jyrz.groupby(['虚拟id'])[feature].std().reset_index().rename(columns = {feature:'std_'+ feature}),on='虚拟id',how='left')
            label =label.merge(ykt_jyrz.groupby(['虚拟id'])[feature].max().reset_index().rename(columns = {feature:'max_'+ feature}),on='虚拟id',how='left')
            label =label.merge(ykt_jyrz.groupby(['虚拟id'])[feature].min().reset_index().rename(columns = {feature:'min_'+ feature}),on='虚拟id',how='left')
            label =label.merge(ykt_jyrz.groupby(['虚拟id'])[feature].sum().reset_index().rename(columns = {feature:'sum_'+ feature}),on='虚拟id',how='left')
            label =label.merge(ykt_jyrz.groupby(['虚拟id'])[feature].skew().reset_index().rename(columns = {feature:'skew_'+ feature}),on='虚拟id',how='left')
    return label

# 获取用户app使用情况特征
train_base = ll_get_feature(user_app_use, train_label)
print(train_base.shape)
# train_base.head()

# 获取用户手机终端特征
train_base_2 = ll_get_feature(user_app_info, train_label)
print(train_base_2.shape)
# train_base_2.head()

#用户漫游行为特征
train_base_3 = ll_get_feature(user_action, train_label)
print(train_base_3.shape)
# train_base_3.head()

# 用户漫出省份特征
train_base_4 = ll_get_feature(user_city, train_label)
print(train_base_4.shape)
# train_base_4.head()

# 用户是否去过景区特征
# train_base_5 = ll_get_feature(user_label, train_label)
# print(train_base_5.shape)
# train_base_5.head()

# 用户地理位置特征
train_base_6 = ll_get_feature(user_geo, train_label)
print(train_base_6.shape)
# train_base_6.head()

# 用户身份属性特征
train_base_7 = ll_get_feature(user_info, train_label)
print(train_base_7.shape)
# train_base_6.head()
train_base_list = [train_base, train_base_2, train_base_3, train_base_4,  train_base_6, train_base_7]
for df in train_base_list:
    train_label = train_label.merge(df, on='虚拟id', how='left')
print(train_label.shape)


# 获取训练集label标签，并与train_label拼接
label = user_info[['虚拟id', 'label']]
label.drop_duplicates(inplace=True)
print(train_label.shape) # (99000, 99)
train_label = train_label.merge(label, on='虚拟id', how='left')
print(train_label.shape)    # (99000, 100)


### 获取测试集统计特征
# 获取用户app使用情况特征
test_base = get_feature(test_user_app_use, test_label) 
print(test_base.shape)
# train_base.head()


# 获取用户手机终端特征
test_base_2 = get_feature(test_user_app_info, test_label)
print(test_base_2.shape)
# train_base_2.head()

#用户漫游行为特征
test_base_3 = get_feature(test_user_action, test_label)
print(test_base_3.shape)
# train_base_3.head()

# 用户漫出省份特征
test_base_4 = get_feature(test_user_city, test_label)
print(test_base_4.shape)
# train_base_4.head()

# 用户是否去过景区特征
# test_base_5 = get_feature(test_user_label, test_label)
# print(test_base_5.shape)
# train_base_5.head()

# 用户地理位置特征
test_base_6 = get_feature(test_user_geo, test_label)
print(test_base_6.shape)
# train_base_6.head()

# 用户身份属性特征
test_base_7 = get_feature(test_user_info, test_label)
print(test_base_7.shape)
# train_base_6.head()

test_base_list = [test_base, test_base_2, test_base_3, test_base_4, test_base_6, test_base_7]
for df in test_base_list:
    test_label = test_label.merge(df, on='虚拟id', how='left')
print(test_label.shape)


dataset1 = train_label
dataset2 = test_label


# 归一化
from sklearn.preprocessing import MinMaxScaler, StandardScaler
features = [i for i in dataset1.columns if i not in ['虚拟id', 'label']]
# 初始化缩放器
scaler = MinMaxScaler()
# data是要进行归一化的DataFrame对象
data_1 = dataset1[features]
# 拟合，计算缩放操作需要的个字段的最大和最小值
scaler.fit(data_1)
data_1 = scaler.transform(data_1)

data_2 = dataset2[features]
data_2 = scaler.transform(data_2)
print(data_1.shape)
print(data_2.shape)
# (99000, 90)
# (50200, 90)


# 导出统计特征训练集和测试集文件
dataset1.to_csv('./data/1lun_train.csv', index=None)
dataset2.to_csv('./data/1lun_test.csv', index=None)