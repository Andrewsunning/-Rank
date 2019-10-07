# user_app_info特征.py
# Author:Andrew Li

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import random
from tqdm import tqdm
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
from datetime import datetime
import time

# 读入原始数据
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
# 用户手机终端信息
user_app_info = read_data([train_path_n+'201708n2.txt',train_path_q+'201708q2.txt'],['虚拟id','品牌','终端型号','首次使用时间','末次使用时间','label'])

test_user_app_info = read_data([test_path+'2018_2.txt'],['虚拟id','品牌','终端型号','首次使用时间','末次使用时间'],True)

# 读入label文件（由guixing.py得到）
train_label = pd.read_csv('./data/train_label.csv')

test_label = pd.read_csv('./data/test_label.csv')

print(train_label.shape)
print(test_label.shape)
# (99000, 2)
# (50200, 2)

### 训练集特征
# 删除重复值
print(user_app_info.shape)
user_app_info.drop_duplicates(inplace=True)
print(user_app_info.shape)
# (1596811, 5)
# (1596771, 5)

# 1.提取末次使用和首次使用的时间差特征

user_app_info['首次使用时间'] = user_app_info.首次使用时间.astype('str')
user_app_info['末次使用时间'] = user_app_info.末次使用时间.astype('str')

user_app_info['timestyle_首次使用时间'] = user_app_info.首次使用时间.apply(lambda s:s[:4]+'-'+s[4:6]+'-'+s[6:8]+ ' '+s[8:10]+':'+s[10:12]+':'+s[12:14])
user_app_info['timestyle_末次使用时间'] = user_app_info.末次使用时间.apply(lambda s:s[:4]+'-'+s[4:6]+'-'+s[6:8]+ ' '+s[8:10]+':'+s[10:12]+':'+s[12:14])
print(user_app_info.shape)
# (1596771, 7)

# 将有效的时间格式转化为时间戳
tmp1 = user_app_info[user_app_info.timestyle_首次使用时间.apply(lambda x:False if len(x)<8 else True)]
# 将时间转化为时间戳
t1 = tmp1.timestyle_首次使用时间.apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))

t2 = tmp1.timestyle_末次使用时间.apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))

t3 = t2-t1
tmp1 = pd.concat([tmp1, t3], axis=1)
print(tmp1.shape)
# (1586188, 8)

tmp1.rename(columns={0:'shijiancha_使用时间'}, inplace=True)

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

tmp1 = tmp1[['虚拟id', 'shijiancha_使用时间']]
f3 = ll_get_feature(tmp1, train_label)
f3.drop(columns=['Unnamed: 0'], inplace=True)
print(f3.shape)
# (99000, 10)
f3.to_csv('./data/train_shijiancha_使用时间.csv', index=None)


### 测试集特征
# 删除重复值
print(test_user_app_info.shape)
test_user_app_info.drop_duplicates(inplace=True)
print(test_user_app_info.shape)
# (1586024, 5)
# (1585957, 5)

test_user_app_info['首次使用时间'] = test_user_app_info.首次使用时间.astype('str')
test_user_app_info['末次使用时间'] = test_user_app_info.末次使用时间.astype('str')

test_user_app_info['timestyle_首次使用时间'] = test_user_app_info.首次使用时间.apply(lambda s:s[:4]+'-'+s[4:6]+'-'+s[6:8]+ ' '+s[8:10]+':'+s[10:12]+':'+s[12:14])
test_user_app_info['timestyle_末次使用时间'] = test_user_app_info.末次使用时间.apply(lambda s:s[:4]+'-'+s[4:6]+'-'+s[6:8]+ ' '+s[8:10]+':'+s[10:12]+':'+s[12:14])
print(test_user_app_info.shape)
# (1585957, 7)

# 将有效的时间格式转化为时间戳
test_tmp1 = test_user_app_info[test_user_app_info.timestyle_首次使用时间.apply(lambda x:False if len(x)<8 else True)]

# 将时间转化为时间戳
t1 = test_tmp1.timestyle_首次使用时间.apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))

t2 = test_tmp1.timestyle_末次使用时间.apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))

t3 = t2-t1
test_tmp1 = pd.concat([test_tmp1, t3], axis=1)
print(test_tmp1.shape)
# (1585957, 8)

test_tmp1.rename(columns={0:'shijiancha_使用时间'}, inplace=True)

test_tmp1 = test_tmp1[['虚拟id', 'shijiancha_使用时间']]
test_f3 = ll_get_feature(test_tmp1, test_label)
test_f3.drop(columns=['Unnamed: 0'], inplace=True)
print(test_f3.shape)
# (50200, 9)

test_f3.to_csv('./data/test_shijiancha_使用时间.csv', index=None)