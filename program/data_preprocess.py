import os

import numpy as np
import pandas as pd

from datetime import datetime

only_train = True # 所有資料都會用在 training
time_range = True # 是否只使用 6:00 ~ 19:00 的資料
merge_10min = False # 把每筆資料合併成 10 分鐘 1 筆
fill_value_ = True # 是否要補值
train_code = [i for i in range(1, 18)] # 要使用哪些 Location code 的資料

# 選擇特徵，註解的是要使用的特徵
feature_removed = [
    # 'LocationCode',
    'DateTime',
    'WindSpeed(m/s)',
    'Pressure(hpa)',
    'Temperature(°C)',
    'Humidity(%)',
    'Sunlight(Lux)',
    'Power(mW)',
    'lat',
    'lon',
    # 'direction',
    'pres_cwb',
    # 'temp_cwb',
    'rh_cwb',
    'precp_cwb',
    # 'rad_cwb',
    'sun_cwb',
    'visb_cwb',
    'uvi_cwb',
    'cloud_cwb',
    # 'apparent_zenith',
    'zenith',
    # 'apparent_elevation',
    'elevation',
    # 'azimuth',
    # 'ghi',
    # 'dni',
    # 'dhi',
    'num_of_min',
    # 'day_of_year',
    'month',
    'day',
    # 'hour',
    # 'min',
    'hour_sin',
    'hour_cos',
]


def fill_value(data): # 補缺失值
    if fill_value_:
        return data.fillna({
            'pres_cwb': 0,
            'temp_cwb': 0,
            'rh_cwb': 0,
            'precp_cwb': 0,
            'rad_cwb': 0,
            'sun_cwb': 0,
            'visb_cwb': 0,
            'uvi_cwb': 0,
            'cloud_cwb': 0,
            
            'apparent_zenith': 0,
            'zenith': 0,
            'apparent_elevation': 0,
            'elevation': 0,
            'azimuth': 0,
            'ghi': 0,
            'dni': 0,
            'dhi': 0,
        })
    else:
        return data

def remove_feature(data):
    X = data.drop(feature_removed, axis=1) # features
    y = data.get(['Power(mW)']) # target
    
    X = fill_value(X)
    
    return X, y


dataset = []
for i in range(1, 18):
    dataset.append(pd.read_csv(f'../data/training set/{i}.csv'))


if time_range:
    for i in range(len(dataset)):
        dataset[i]['DateTime'] = pd.to_datetime(dataset[i]['DateTime'])

        # 設定 DateTime 為索引
        dataset[i].set_index('DateTime', inplace=True)
        dataset[i] = dataset[i].between_time('06:00', '19:00')

        # 重設索引，將 DateTime 變回一個欄位
        dataset[i].reset_index(inplace=True)
    
    print('只使用 06:00 ~ 19:00 的資料進行訓練')


if merge_10min:
    for i in range(len(dataset)):
        df = dataset[i].copy()

        df['DateTime'] = pd.to_datetime(df['DateTime'])

        df.set_index('DateTime', inplace=True)

        df_resampled = df.resample('10min').agg({
            'LocationCode': 'first',
            'WindSpeed(m/s)': 'mean',
            'Pressure(hpa)': 'mean',
            'Temperature(°C)': 'mean',
            'Humidity(%)': 'mean',
            'Sunlight(Lux)': 'mean',
            'Power(mW)': 'mean',
            'lat': 'first',
            'lon': 'first',
            'direction': 'first',
            'temp_cwb': 'mean',
            'rh_cwb': 'mean',
            'precp_cwb': 'mean',
            'rad_cwb': 'mean',
            'sun_cwb': 'mean'
        })

        df_resampled = df_resampled.dropna(subset=['LocationCode'])

        df_resampled.reset_index(inplace=True)

        df_resampled['LocationCode'] = df_resampled['LocationCode'].astype(int).astype('category')

        dataset[i] = df_resampled.round(2)
    
    print('將資料合併為 10 分鐘一筆')


train_dataset = pd.concat([dataset[i-1] for i in train_code], ignore_index=1)


print('training set: ', ', '.join(map(str, train_code)))
if only_train:
    print('train (only):')
    train = pd.concat([train_dataset.loc[train_dataset['LocationCode']==i] for i in train_code], ignore_index=1)
    valid = pd.concat([train_dataset.loc[train_dataset['LocationCode']==train_code[-1]]], ignore_index=1)
    
else:
    print('train (not only):')
    train = pd.concat([train_dataset.loc[train_dataset['LocationCode']==i] for i in train_code[:-1]], ignore_index=1)
    valid = pd.concat([train_dataset.loc[train_dataset['LocationCode']==i] for i in train_code[-1]], ignore_index=1)


print(train.info())
print(valid.info())


train_X, train_y = remove_feature(train)
valid_X, valid_y = remove_feature(valid)

print(train_X)
print(train_y)

print(train_X.info())
print(train_y.info())
print(valid_X.info())
print(valid_y.info())

if not os.path.exists("../data/Dataset"):
   os.makedirs("../data/Dataset")

train_X.to_csv('../data/Dataset/train_X.csv', index=False)
train_y.to_csv('../data/Dataset/train_y.csv', index=False)
valid_X.to_csv('../data/Dataset/valid_X.csv', index=False)
valid_y.to_csv('../data/Dataset/valid_y.csv', index=False)


testdata = pd.read_csv('../data/testset/testdata.csv')
testdata = testdata.drop(feature_removed, axis=1)
testdata = fill_value(testdata)
print(testdata.info())

testdata.to_csv('../data/Dataset/testdata.csv', index=False)


testdata_1min = pd.read_csv('../data/testset/testdata_1min.csv')
testdata_1min = testdata_1min.drop(feature_removed, axis=1)
testdata_1min = fill_value(testdata_1min)
print(testdata_1min.info())

testdata_1min.to_csv('../data/Dataset/testdata_1min.csv', index=False)
