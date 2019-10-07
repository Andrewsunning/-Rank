# 特征合并.py
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

# 读入label文件
train_label = pd.read_csv('./data/train_label.csv')

test_label = pd.read_csv('./data/test_label.csv')

print(train_label.shape)
print(test_label.shape)
# (99000, 2)
# (50200, 2)

train_df_0 = pd.read_csv('./data/1lun_train.csv')

train_df_1 = pd.read_csv('./data/chazhi_出账收入.csv')

train_df_2 = pd.read_csv('./data/count_isna_出账收入.csv')

train_df_3 = pd.read_csv('./data/train_shijiancha_使用时间.csv')

# train_df_4 = pd.read_csv('./data/train_is_-1_联络圈规模.csv')

# train_df_5 = pd.read_csv('./data/train_出省出境.csv')

# train_df_6 = pd.read_csv('./data/train_chazhi_漫出省份次数.csv')

# train_df_7 = pd.read_csv('./data/train_bili_漫出特定省份次数占漫出总次数比例特征.csv')

# train_df_8 = pd.read_csv('./data/train_6ge_user_app_use.csv')

train_df_9 = pd.read_csv('./data/train_count_时段x.csv')


# 合并特征
train_label = train_label.merge(train_df_0, on='虚拟id', how='left')
train_label = train_label.merge(train_df_1, on='虚拟id', how='left')
train_label = train_label.merge(train_df_2, on='虚拟id', how='left')
train_label = train_label.merge(train_df_3, on='虚拟id', how='left')
# train_label = train_label.merge(train_df_4, on='虚拟id', how='left')
# train_label = train_label.merge(train_df_5, on='虚拟id', how='left')
# train_label = train_label.merge(train_df_6, on='虚拟id', how='left')
# train_label = train_label.merge(train_df_7, on='虚拟id', how='left')
# train_label = train_label.merge(train_df_8, on='虚拟id', how='left')
train_label = train_label.merge(train_df_9, on='虚拟id', how='left')
train_label.drop(columns=['Unnamed: 0'], inplace=True)
print(train_label.shape)
# (99000, 115)

# 读入测试集特征
test_df_0 = pd.read_csv('./data/1lun_test.csv')

test_df_1 = pd.read_csv('./data/test_chazhi_出账收入.csv')

test_df_2 = pd.read_csv('./data/test_count_isna_出账收入.csv')

test_df_3 = pd.read_csv('./data/test_shijiancha_使用时间.csv')

# test_df_4 = pd.read_csv('./data/test_is_-1_联络圈规模.csv')

# test_df_5 = pd.read_csv('./data/test_出省出境.csv')

# test_df_6 = pd.read_csv('./data/test_chazhi_漫出省份次数.csv')

# test_df_7 = pd.read_csv('./data/test_bili_漫出特定省份次数占漫出总次数比例特征.csv')

# test_df_8 = pd.read_csv('./data/test_6ge_user_app_use.csv')

test_df_9 = pd.read_csv('./data/test_count_时段x.csv')

# 测试集特征合并
test_label = test_label.merge(test_df_0, on='虚拟id', how='left')
test_label = test_label.merge(test_df_1, on='虚拟id', how='left')
test_label = test_label.merge(test_df_2, on='虚拟id', how='left')
test_label = test_label.merge(test_df_3, on='虚拟id', how='left')

# test_label = test_label.merge(test_df_4, on='虚拟id', how='left')
# test_label = test_label.merge(test_df_5, on='虚拟id', how='left')

# test_label = test_label.merge(test_df_6, on='虚拟id', how='left')
# test_label = test_label.merge(test_df_7, on='虚拟id', how='left')
# test_label = test_label.merge(test_df_8, on='虚拟id', how='left')
test_label = test_label.merge(test_df_9, on='虚拟id', how='left')
test_label.drop(columns=['Unnamed: 0'], inplace=True)
print(test_label.shape)
# (50200, 114)


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
# (99000, 113)
# (50200, 113)


# dataset1是训练集+验证集
label = dataset1['label']
train = pd.DataFrame(data_1)

# dataset2是测试集
test_id = dataset2['虚拟id']
# test_label_tmp = dataset2['label']   # 在线提交时该行需要省略
test = pd.DataFrame(data_2)

skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
best_score = []

oof_preds = np.zeros(train.shape[0])
test_preds = np.zeros(test_id.shape[0])

model_lgb = lgb.LGBMClassifier(
    learning_rate=0.05,  # 重点
    colsample_bytree=0.85,
    num_leaves=100,    # 重点
    min_child_weight=6, # 防止过拟合
    min_data_in_leaf=6,
    subsample=0.7,
    n_estimators=185,  # 以上参数需要调参
    max_depth=-1,
    is_unbalance=True,
    boosting_type = 'gbdt', 
    gamma=0,
    objective='binary',
    random_state=2019,
    n_jobs = 2
    )

for index,(train_index, val_index) in enumerate(skf.split(train, label)):
    print('fold_{}'.format(index))
    model_lgb.fit(train.iloc[train_index], label.iloc[train_index], verbose=50, eval_metric='auc',
                  eval_set=[(train.iloc[train_index], label.iloc[train_index]), (train.iloc[val_index], label.iloc[val_index])], 
                  early_stopping_rounds=30)

    best_score.append(model_lgb.best_score_['valid_1']['auc'])
    print(best_score)
    
    oof_preds[val_index] = model_lgb.predict_proba(train.iloc[val_index], num_iteration=model_lgb.best_iteration_)[:,1]
    test_pred = model_lgb.predict_proba(test, num_iteration=model_lgb.best_iteration_)[:,1]
    test_preds += test_pred / 5.0

# 生成提交文件
sub = pd.concat([test_id, pd.Series(test_preds.round(5))], axis=1)
sub.columns = ['ID', 'Pred']
sub.to_csv('./data/sub22.csv')
