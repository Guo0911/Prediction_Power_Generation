import os
import math
import pytz
import pvlib

import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime


tqdm.pandas()

additional = [2, 4, 7, 8, 9, 10, 12] # training set 中有額外資料的 Location code


def dms_to_decimal(degrees, minutes, seconds): # 把經緯度(度:角分:角秒)轉成十進位度數
    return round((degrees + (minutes/60) + (seconds/3600)), 4)

def get_distance_from_latlon_in_m(lat1, lon1, lat2, lon2): # Haversine function 計算兩個經緯度間距離(公尺)
    radius = 6371e3  # 地球半徑(m)
    
    # 轉換為弧度
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
        
    # Haversine 公式
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_lat/2) * math.sin(delta_lat/2) +         math.cos(lat1_rad) * math.cos(lat2_rad) *         math.sin(delta_lon/2) * math.sin(delta_lon/2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    distance = radius * c
    return distance


station_exp_info = { # Lat, Lon, 面板朝向, 第一個測站, 第二個測站
     1: ['23/53/58', '121/32/40',  181, 'C0T9E0', 'C0Z100'],
     2: ['23/53/59', '121/32/41',  175, 'C0T9E0', 'C0Z100'],
     3: ['23/53/59', '121/32/42',  180, 'C0T9E0', 'C0Z100'],
     4: ['23/53/58', '121/32/40',  161, 'C0T9E0', 'C0Z100'],
     5: ['23/53/58', '121/32/41',  208, 'C0T9E0', 'C0Z100'],
     6: ['23/53/58', '121/32/40',  208, 'C0T9E0', 'C0Z100'],
     7: ['23/53/58', '121/32/40',  172, 'C0T9E0', 'C0Z100'],
     8: ['23/53/59', '121/32/42',  219, 'C0T9E0', 'C0Z100'],
     9: ['23/53/58', '121/32/40',  151, 'C0T9E0', 'C0Z100'],
    10: ['23/53/58', '121/32/40',  223, 'C0T9E0', 'C0Z100'],
    11: ['23/53/59', '121/32/41',  131, 'C0T9E0', 'C0Z100'],
    12: ['23/53/59', '121/32/41',  298, 'C0T9E0', 'C0Z100'],
    13: ['23/53/52', '121/32/22',  249, 'C0T9E0', 'C0Z100'],
    14: ['23/53/52', '121/32/22',  197, 'C0T9E0', 'C0Z100'],
    15: ['24/00/33', '121/37/02',  127, '466990', '466990'],
    16: ['24/00/32', '121/37/02',   82, '466990', '466990'],
    17: [ 23.9751  ,  121.6133  , None, '466990', '466990'] # LC 17 的經緯度為觀測站數據
}

for id in station_exp_info.keys():
    if (id == 17):
        continue
    
    else:
        lat_d, lat_m, lat_s = map(int, station_exp_info[id][0].split('/'))
        lon_d, lon_m, lon_s = map(int, station_exp_info[id][1].split('/'))

        lat = dms_to_decimal(lat_d, lat_m, lat_s)
        lon = dms_to_decimal(lon_d, lon_m, lon_s)

        station_exp_info[id][0] = lat
        station_exp_info[id][1] = lon


cwb = pd.read_csv('../data/CWB data/cwb.csv')

temp = {}
for station_id in list(cwb['Station_ID'].unique()):
    temp[station_id] = cwb.loc[cwb['Station_ID']==station_id].reset_index(drop=True, inplace=False)
    
cwb = temp
print('cwb station id:', cwb.keys())


dataset = []
for station_id in range(1, 18):
    if station_id in additional:
        d1 = pd.read_csv(f'../data/Original/training set/L{station_id}_Train.csv')
        d2 = pd.read_csv(f'../data/Original/training set/L{station_id}_Train_2.csv')
        
        data = pd.concat([d1, d2], ignore_index=True)
    else:
        data = pd.read_csv(f'../data/Original/training set/L{station_id}_Train.csv')
    
    dataset.append(data)

dataset = pd.concat(dataset, ignore_index=True)


def get_Location_info(id):
    lat = station_exp_info[id][0]
    lon = station_exp_info[id][1]
    direction = station_exp_info[id][2]

    return lat, lon, direction

dataset[['lat', 'lon', 'direction']] = dataset.progress_apply(lambda x: get_Location_info(x['LocationCode']), axis=1, result_type='expand')

print('新增 lat, lon, direction')


def get_cwb_info(id, dt):
    station1  = station_exp_info[id][3]
    station2  = station_exp_info[id][4]
    
    date, time = dt.split(' ')
    m, d = map(int, date.split('-')[1:])
    h = int(time.split(':')[0])+1
    
    cwb3 = cwb['466990'][(cwb['466990']["Month"]==m) & (cwb['466990']["Day"]==d) & (cwb['466990']["Hour"]==h)]

    if station1 == '466990':
        cwb1 = cwb3
    else:
        cwb1 = cwb[station1][(cwb[station1]["Month"]==m) & (cwb[station1]["Day"]==d) & (cwb[station1]["Hour"]==h)]

    if station2 == '466990':
        cwb2 = cwb3
    else:
        cwb2 = cwb[station2][(cwb[station2]["Month"]==m) & (cwb[station2]["Day"]==d) & (cwb[station2]["Hour"]==h)]
    
    try:
        pres = float(cwb1['StnPres'].iloc[0])
        temp = float(cwb1['Temperature'].iloc[0])
        rh = float(cwb1['RH'].iloc[0])
        precp = float(cwb1['Precp'].iloc[0])
        rad = float(cwb2['GloblRad'].iloc[0])
        sun = float(cwb3['SunShine'].iloc[0])
        visb = float(cwb3['Visb'].iloc[0])
        uvi = float(cwb3['UVI'].iloc[0])
        cloud = float(cwb3['Cloud Amount'].iloc[0])
    except:
        print('sp:',cwb1['StnPres'])
        print('te:',cwb1['Temperature']) # t: Series([], Name: Temperature, dtype: float64)
        print('rh:',cwb1['RH'])
        print('pr:',cwb1['Precp'])
        print('gl:',cwb2['GloblRad'])
        print('ss:',cwb3['SunShine'])
        print('vi:',cwb3['Visb'])
        print('uv:',cwb3['UVI'])
        print('cl:',cwb3['Cloud'])
        print(id, dt, station1, station2, m, d, h)
    
    return (pres, temp, rh, precp, rad, sun, visb, uvi, cloud)

dataset[['pres_cwb', 'temp_cwb', 'rh_cwb', 'precp_cwb', 'rad_cwb', 'sun_cwb', 'visb_cwb', 'uvi_cwb', 'cloud_cwb']] = dataset.progress_apply(lambda x: get_cwb_info(x['LocationCode'], x['DateTime']), axis=1, result_type='expand')

print('新增 pres_cwb, temp_cwb, rh_cwb, precp_cwb, rad_cwb, sun_cwb, visb_cwb, uvi_cwb, cloud_cwb')


def get_solar_radiation(lat, lon, datetime, temperature=12, pres=1013.25, tz='Asia/Taipei'):
    location = pvlib.location.Location(
       latitude=lat,
       longitude=lon,
       tz=tz
    )
    
    time = pd.Timestamp(datetime, tz=tz)
    times = pd.DatetimeIndex([time])
    
    solar_position = location.get_solarposition(
       times,
       temperature=temperature,
       pressure=pres
    )
    
    clearsky = location.get_clearsky(
       times,
       pressure=pres
    )
    
    return (
        solar_position['apparent_zenith'].iloc[0],
        solar_position['zenith'].iloc[0],
        solar_position['apparent_elevation'].iloc[0],
        solar_position['elevation'].iloc[0],
        solar_position['azimuth'].iloc[0],
        clearsky['ghi'].iloc[0],
        clearsky['dni'].iloc[0],
        clearsky['dhi'].iloc[0]
    )

dataset[['apparent_zenith', 'zenith', 'apparent_elevation', 'elevation', 'azimuth', 'ghi', 'dni', 'dhi']] = dataset.progress_apply(
    lambda x: get_solar_radiation(x['lat'], x['lon'], x['DateTime'], x['temp_cwb'], x['pres_cwb']), axis=1, result_type='expand')

print('新增 apparent_zenith, zenith, apparent_elevation, elevation, azimuth, ghi, dni, dhi')


def get_minutes_of_day(datetime_str):
    datetime_str = str(datetime_str)
    dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    num_of_min = dt.hour * 60 + dt.minute
    
    return num_of_min # 直接回傳是第幾分鐘

dataset['DateTime'] = pd.to_datetime(dataset['DateTime'])
dataset['num_of_min'] = dataset['DateTime'].progress_apply(get_minutes_of_day)

print('計算資料為當天的第幾分鐘 num_of_min')


dataset['day_of_year'] = dataset['DateTime'].dt.dayofyear

print('計算資料為當年的第幾天 day_of_year')


def split_time_feature(datetime_str):
    datetime_str = str(datetime_str)
    dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    
    return (dt.month, dt.day, dt.hour, dt.minute)

dataset['DateTime'] = pd.to_datetime(dataset['DateTime'])
dataset[['month', 'day', 'hour', 'min']] = dataset.progress_apply(lambda x: split_time_feature(x['DateTime']), axis=1, result_type='expand')

print('拆分每個時間特徵 month, day, hour, min')


dataset['hour_sin'] = np.sin(dataset['hour'] * (2 * np.pi / 24))
dataset['hour_cos'] = np.cos(dataset['hour'] * (2 * np.pi / 24))

print('新增循環時間特徵 hour_sin, hour_cos')


print(dataset)
print(dataset.info())

if not os.path.exists("../data/training set"):
   os.makedirs("../data/training set")

for i in range(1, 18):
    temp = dataset.query(f'LocationCode=={i}')

    print(temp.info())
    
    temp.to_csv(f'../data/training set/{i}.csv', index=False)
