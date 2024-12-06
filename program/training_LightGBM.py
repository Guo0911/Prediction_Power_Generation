import os

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


num_of_epochs = 100000  # epochs
num_of_stopping = int(num_of_epochs/100)

train_X = pd.read_csv('../data/Dataset/train_X.csv')
train_y = pd.read_csv('../data/Dataset/train_y.csv')

valid_X = pd.read_csv('../data/Dataset/valid_X.csv')
valid_y = pd.read_csv('../data/Dataset/valid_y.csv')

params = {
    'objective': 'regression', # 設置迴歸任務
    'metric': 'mae',           # 評估指標
    # 'boosting_type': 'dart',   # 使用, 預設 gbdt(傳統梯度提升決策樹)，還有 rf(隨機森林), dart(Dropout 搭配 MART)
    
    # 'num_leaves': 31,          # 每棵樹的最大葉子節點數, 預設 31
    
    # 'learning_rate': 0.05,     # 學習率, 預設 0.1

    # 'drop_rate': 0.1,          # 每次迭代丟棄樹的比例, 預設 0.1
    # 'max_drop': 50,            # 最多丟棄樹的數量, 預設 50
    # 'skip_drop': 0.5,          # 跳過丟棄的概率, 預設 0.5
    
    # 'feature_fraction': 0.9,   # 每棵樹隨機選擇的特徵比例, 預設 1.0
    # 'bagging_fraction': 0.8,   # 每次迭代抽樣的數據比例, 預設 1.0
    # 'bagging_freq': 5,         # 每隔多少次迭代執行 bagging
    
    'verbose': 0,              # 不輸出訓練信息
    'random_state': 42,
}


# 5 fold cross validation 的部分，自行決定是否使用
"""
kf = KFold(n_splits=5, shuffle=True, random_state=42)

loss_folds = []
for train_idx, valid_idx in kf.split(train_X):    
    fold_train_X, fold_valid_X = train_X.iloc[train_idx], train_X.iloc[valid_idx]
    fold_train_y, fold_valid_y = train_y.iloc[train_idx], train_y.iloc[valid_idx]

    fold_train_data = lgb.Dataset(fold_train_X, label=fold_train_y)
    fold_valid_data = lgb.Dataset(fold_valid_X, label=fold_valid_y, reference=fold_train_data)

    model = lgb.train(
        params,
        fold_train_data,
        num_of_epochs,
        valid_sets=[fold_train_data, fold_valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=num_of_stopping),
        ]
    )
    
    predi_valid = model.predict(fold_valid_X)

    loss = mean_absolute_error(fold_valid_y, predi_valid)
    
    loss_folds.append(round(loss, 4))

    print(f'Fold {len(loss_folds)} loss : {round(loss, 4)}')

print(sum(loss_folds)/len(loss_folds))
"""


train_data = lgb.Dataset(train_X, label=train_y)
valid_data = lgb.Dataset(valid_X, label=valid_y, reference=train_data)

loss_history = {}
model = lgb.train(
    params,
    train_data,
    num_of_epochs,
    valid_sets=[train_data, valid_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=num_of_stopping),
        lgb.record_evaluation(loss_history),
    ]
)
predi = model.predict(valid_X)

loss = mean_absolute_error(valid_y, predi)
loss = round(loss, 4)

print(f'Loss: {loss}')


testdata_1min = pd.read_csv('../data/Dataset/testdata_1min.csv')

print(testdata_1min.info())

result_1min = model.predict(testdata_1min)

result_avg = [max(round(sum(result_1min[i:i+10])/10, 2), 0) for i in range(0, len(result_1min), 10)]

print(result_avg[:15])
print(result_avg[-15:])

testset = pd.read_csv('../data/Original/testset/upload(no answer).csv')

testset['序號'] = testset['序號'].astype('str')
testset['答案'] = result_avg


if not os.path.exists("../result"):
   os.makedirs("../result")


if not os.path.exists("../result/submission"):
   os.makedirs("../result/submission")

testset.to_csv('../result/submission/upload_1min.csv', index=False)
print('testset 1min saved')


if not os.path.exists("../result/model"):
   os.makedirs("../result/model")

model.save_model(f'../result/model/LGBM_{num_of_epochs}e_{loss}.json')  # 存成 JSON 格式
print('model saved')
