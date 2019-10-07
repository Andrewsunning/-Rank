# user_info特征.py
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

test_user_info = read_data([test_path+'2018_1.txt'],['账期','虚拟id','出账收入'],True)

# 定义删除重复值的函数
def drop_dup(df):
    return df.drop_duplicates()
    
# 删除重复值
print(user_info.shape)
user_info = drop_dup(user_info)
print(user_info.shape)

print(test_user_info.shape)
test_user_info = drop_dup(test_user_info)
print(test_user_info.shape)
# (198000, 4)
# (198000, 4)
# (100402, 3)
# (100400, 3)

# 1. 用户的出账收入是否为缺失值，以及取缺失值的次数，范围为【0，1，2】
tmp1 = user_info.出账收入.isna()
tmp1 = tmp1.apply(lambda x : 1 if x==True else 0)

user_info['isna_出账收入'] = tmp1

f1 = user_info.groupby(by=['虚拟id'], as_index=False)[['isna_出账收入']].sum()
f1.rename(columns={'isna_出账收入':'count_isna_出账收入'}, inplace=True)
print(f1.shape)
# (99000, 2)

# 2. 每一用户6、7月份的消费差
grouped = user_info.groupby(by=['虚拟id'])

id_list =  []
shourucha = []
for (name,group) in tqdm(grouped):
    id_list.append(name)
    value = (group[group.账期==201707]['出账收入'].values-group[group.账期==201706]['出账收入'].values)[0]
    shourucha.append(value)
print(len(shourucha))
print(shourucha[:5])
# 99000
# [55.06999999999999, -0.32, 26.18, 7.030000000000001, 33.599999999999994]

f2 = pd.DataFrame({'虚拟id':pd.Series(id_list),'chazhi_出账收入':pd.Series(shourucha)})
print(f2.shape)
# (99000, 2)

# 导出特征
f1.to_csv('./data/count_isna_出账收入.csv', index=None)
f2.to_csv('./data/chazhi_出账收入.csv', index=None)

### 测试集特征
# 1. 用户的出账收入是否为缺失值，以及取缺失值的次数，范围为【0，1，2】
tmp1 = test_user_info.出账收入.isna()
tmp1 = tmp1.apply(lambda x : 1 if x==True else 0)

test_user_info['isna_出账收入'] = tmp1

test_f1 = test_user_info.groupby(by=['虚拟id'], as_index=False)[['isna_出账收入']].sum()
test_f1.rename(columns={'isna_出账收入':'count_isna_出账收入'}, inplace=True)
print(test_f1.shape)
# (50200, 2)

# 2. 每一用户2018年6、7月份的消费差
grouped = test_user_info.groupby(by=['虚拟id'])

id_list =  []
shourucha = []
for (name,group) in tqdm(grouped):
    id_list.append(name)
    value = (group[group.账期==201807]['出账收入'].values-group[group.账期==201806]['出账收入'].values)[0]
    shourucha.append(value)
print(len(shourucha))
print(shourucha[:5])
# 50200
# [0.0, -15.699999999999989, -12.5, 0.0, 46.0]

test_f2 = pd.DataFrame({'虚拟id':pd.Series(id_list),'chazhi_出账收入':pd.Series(shourucha)})
print(test_f2.shape)
# (50200, 2)

# 导出测试集特征
test_f1.to_csv('./data/test_count_isna_出账收入.csv', index=None)
test_f2.to_csv('./data/test_chazhi_出账收入.csv', index=None)