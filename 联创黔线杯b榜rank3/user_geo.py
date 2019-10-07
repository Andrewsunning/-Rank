# user_geo.py
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
        
# 用户地理位置
user_geo = read_data([train_path_n+'201708n6.txt',train_path_q+'201708q6.txt'],['日期','时段','虚拟id','经度','纬度','label'])

test_user_geo = read_data([test_path+'2018_6.txt'],['日期','时段','虚拟id','经度','纬度'],True)

# 去重
## 定义删除重复值的函数
def drop_dup(df):
    return df.drop_duplicates()
print(user_geo.shape)
user_geo = drop_dup(user_geo)
print(user_geo.shape)
print('\n')

print(test_user_geo.shape)
test_user_geo = drop_dup(test_user_geo)
print(test_user_geo.shape)
# (39245941, 5)
# (39241782, 5)

# (16558993, 5)
# (16558992, 5)

### 训练集特征
# 1. 用户在不同时段的记录数
print(set(user_geo.时段))
user_geo['时段'] = user_geo.时段.astype('int')
print(set(user_geo.时段))
# {0.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0}
# {0, 2, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22}

train_label = pd.read_csv('./data/train_label.csv', index_col=0 )
print(train_label.shape)
# (99000, 1)

tmp1 = user_geo[['时段', '虚拟id','经度', '纬度']]
tmp2 = tmp1.groupby(by=[ '虚拟id', '时段'], as_index=False)['经度'].count().rename(columns = {'经度':'count_时段'})

for feature in tqdm(list(set(user_geo.时段))):
    t = tmp2.loc[tmp2.时段==feature][['虚拟id', 'count_时段']].rename(columns={'count_时段':'count_时段'+str(feature)})
    train_label = train_label.merge(t,on='虚拟id', how='left')

train_label.fillna(0, inplace=True)
train_label.to_csv('./data/train_count_时段x.csv', index=None)

### 提取测试集特征
test_label = pd.read_csv('./data/test_label.csv', index_col=0)
print(test_label.shape)
# (50200, 1)

test_tmp1 = test_user_geo[['时段', '虚拟id','经度', '纬度']]
test_tmp2 = test_tmp1.groupby(by=[ '虚拟id', '时段'], as_index=False)['经度'].count().rename(columns = {'经度':'count_时段'})

for feature in tqdm(list(set(test_user_geo.时段))):
    t = test_tmp2.loc[test_tmp2.时段==feature][['虚拟id', 'count_时段']].rename(columns={'count_时段':'count_时段'+str(feature)})
    test_label = test_label.merge(t,on='虚拟id', how='left')
print(test_label.shape)
# (50200, 14)

test_label.fillna(0, inplace=True)
test_label.to_csv('./data/test_count_时段x.csv', index=None)