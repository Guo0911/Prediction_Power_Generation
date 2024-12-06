import os
import csv
import json
import time
import random
import requests

import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from openpyxl import Workbook
from datetime import datetime, timedelta


collection_start = "2024-01-01" # 資料開始時間
collection_end = "2024-11-17" # 資料結束時間

skip_remark = ['本站只有雷達觀測資料。'] # 備註中含有該文字則不抓取

response = requests.get("https://e-service.cwb.gov.tw/wdps/obs/state.htm", verify=False)
response.encoding = response.apparent_encoding

soup = BeautifulSoup(response.text, 'html.parser')

tables = soup.find_all('table')

data = [[header.text for header in tables[0].find_all('th')]]

cols = (tables[0].find_all('tr')[1:] + tables[1].find_all('tr')[1:])
for col in cols:
    data.append([row.text for row in col.find_all('td')])

wb = Workbook()
ws = wb.active

for row in data:
    if row[10] not in skip_remark:
        ws.append(row)

for row in ws.iter_rows():
    for cell in row:
        cell.number_format = '@'

if not os.path.exists("../data/CWB data"):
   os.makedirs("../data/CWB data")

wb.save('../data/CWB data/station_list.xlsx')


def generate_dates(start_collection, end_collection, station_start, station_end=None):
    # start_collection: 數據收集開始日期 (str: 'YYYY-MM-DD')
    # end_collection: 數據收集結束日期 (str: 'YYYY-MM-DD')
    # station_start: 觀測站啟用日期 (str: 'YYYY-MM-DD')
    # station_end: 觀測站停用日期 (str: 'YYYY-MM-DD')，如果仍在運作則用 None
    # return: 有效數據收集日期列表 (list)

    def parse_date(date_str):
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    
    start_collect = parse_date(start_collection)
    end_collect = parse_date(end_collection)
    station_start = parse_date(station_start)
    station_end = parse_date(station_end) if station_end else datetime.now().date()
    
    # 確定有效的開始和結束日期
    valid_start = max(start_collect, station_start)
    valid_end = min(end_collect, station_end)
    
    # 如果有效開始日期晚於有效結束日期，則沒有有效的數據收集日期
    if valid_start > valid_end:
        return []
    
    # 生成有效日期列表
    valid_dates = []
    current_date = valid_start
    while current_date <= valid_end:
        valid_dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    return valid_dates


def spider_cwb_data(station_id, date, lng, lat, altitude):
    payload = {
        "date": f"{date}T00:00:00.000+08:00",
        "type": "report_date",
        "stn_ID": station_id,
        "stn_type": "cwb" if station_id[:2] == '46' else "auto_C0",
        "more": "",
        "start": f"{date}T00:00:00",
        "end": f"{date}T23:59:59",
        "item": ""
    }

    response = requests.post("https://codis.cwa.gov.tw/api/station?", data=payload)

    soup = BeautifulSoup(response.text, 'html.parser')

    temp = json.loads(soup.contents[0])['data'][0]['dts']

    y, m, d = map(int, date.split('-'))

    station_data = []
    for i, t in enumerate(temp):
        station_data.append([
            station_id,
            lng,
            lat,
            altitude,
            y,
            m,
            d,
            i+1,
            t['StationPressure']['Instantaneous'] if t.get("StationPressure") is not None else None,
            t['SeaLevelPressure']['Instantaneous'] if t.get("SeaLevelPressure") is not None else None,
            t['AirTemperature']['Instantaneous'] if t.get("AirTemperature") is not None else None,
            t['DewPointTemperature']['Instantaneous'] if t.get("DewPointTemperature") is not None else None,
            t['RelativeHumidity']['Instantaneous'] if t.get("RelativeHumidity") is not None else None,
            t['WindSpeed']['Mean'] if t.get("WindSpeed") is not None else None,
            t['WindDirection']['Mean'] if t.get("WindDirection") is not None else None,
            t['PeakGust']['Maximum'] if t.get("PeakGust") is not None else None,
            t['PeakGust']['Direction'] if t.get("PeakGust") is not None else None,
            (t['Precipitation']['Accumulation'] if t['Precipitation']['Accumulation']>=0 else 0.1) if t.get("Precipitation") is not None else None, # < 0.5 if prec is T
            t['PrecipitationDuration']['Total'] if t.get("PrecipitationDuration") is not None else None,
            t['SunshineDuration']['Total'] if t.get("SunshineDuration") is not None else None,
            t['GlobalSolarRadiation']['Accumulation'] if t.get("GlobalSolarRadiation") is not None else None,
            t['Visibility']['Instantaneous'] if t.get("Visibility") is not None else None,
            t['UVIndex']['Accumulation'] if t.get("UVIndex") is not None else None,
            t['TotalCloudAmount']['Instantaneous'] if t.get("TotalCloudAmount") is not None else None,
            t['SoilTemperatureAt0cm']['Instantaneous'] if t.get("SoilTemperatureAt0cm") is not None else None,
            t['SoilTemperatureAt5cm']['Instantaneous'] if t.get("SoilTemperatureAt5cm") is not None else None,
            t['SoilTemperatureAt10cm']['Instantaneous'] if t.get("SoilTemperatureAt10cm") is not None else None,
            t['SoilTemperatureAt20cm']['Instantaneous'] if t.get("SoilTemperatureAt20cm") is not None else None,
            t['SoilTemperatureAt30cm']['Instantaneous'] if t.get("SoilTemperatureAt30cm") is not None else None,
            t['SoilTemperatureAt50cm']['Instantaneous'] if t.get("SoilTemperatureAt50cm") is not None else None,
            t['SoilTemperatureAt100cm']['Instantaneous'] if t.get("SoilTemperatureAt100cm") is not None else None
        ])

    return station_data

# 以下是 API 中的資料與 CWB 資料中對應的欄位名稱
# DataTime                                  =	ObsTime
# StationPressure['Instantaneous']          =	StnPres
# SeaLevelPressure['Instantaneous']         =	SeaPres
# AirTemperature['Instantaneous']           =	Temperature
# DewPointTemperature['Instantaneous']      =	Td dew point
# RelativeHumidity['Instantaneous']         =	RH
# WindSpeed['Mean']                         =	WS
# WindDirection['Mean']                     =	WD
# PeakGust['Maximum']                       =	WSGust
# PeakGust['Direction']                     =	WDGust
# Precipitation['Accumulation']             =	Precp
# PrecipitationDuration['Total']            =	PrecpHour
# SunshineDuration['Total']                 =	SunShine
# GlobalSolarRadiation['Accumulation']	    =	GloblRad
# Visibility['Instantaneous']               =	Visb
# UVIndex['Accumulation']                   =	UVI
# TotalCloudAmount['Instantaneous']         =	Cloud Amount
# SoilTemperatureAt0cm['Instantaneous']	    =	TxSoil0cm
# SoilTemperatureAt5cm['Instantaneous']	    =	TxSoil5cm
# SoilTemperatureAt10cm['Instantaneous']	=	TxSoil10cm
# SoilTemperatureAt20cm['Instantaneous']	=	TxSoil20cm
# SoilTemperatureAt30cm['Instantaneous']	=	TxSoil30cm
# SoilTemperatureAt50cm['Instantaneous']	=	TxSoil50cm
# SoilTemperatureAt100cm['Instantaneous']	=	TxSoil100cm

df = pd.read_excel('../data/CWB data/station_list.xlsx')

data = [[
    'Station_ID',
    'lng.', # 經度
    'lat.', # 緯度
    'altitude', # 海拔高度
    'Year',
    'Month',
    'Day',
    'Hour',
    'StnPres',
    'SeaPres',
    'Temperature',
    'Td dew point',
    'RH',
    'WS',
    'WD',
    'WSGust',
    'WDGust',
    'Precp',
    'PrecpHour',
    'SunShine',
    'GloblRad',
    'Visb',
    'UVI',
    'Cloud Amount',
    'TxSoil0cm',
    'TxSoil5cm',
    'TxSoil10cm',
    'TxSoil20cm',
    'TxSoil30cm',
    'TxSoil50cm',
    'TxSoil100cm',
]]


# "46"為中央氣象署地面氣象站
# "C0"為中央氣象署自動氣象站
# "C1"中央氣象署自動雨量站，其餘皆為農業觀測站
# station_list = df[df['站號'].str.startswith(tuple(['46']))] # 抓取所有 46 開頭的地面氣象站資料
station_list = df[df['站號'].str.startswith(tuple(['C0Z100', 'C0T9E0', '466990']))] # 抓取 C0Z100, C0T9E0, 466990 的資料，與 208 行擇一使用

for index, station in station_list.iterrows():
    station_id = station['站號']

    station_start = station['資料起始日期']
    station_end = station['撤站日期'] if not pd.isnull(station['撤站日期']) else None

    dates = generate_dates(collection_start, collection_end, station_start, station_end)

    for date in tqdm(dates):
        data += spider_cwb_data(station_id, date, station['經度'], station['緯度'], station['海拔高度(m)'])

        time.sleep(random.uniform(0.01, 0.3)) # 隨機暫停 0.01 ~ 0.3 秒.
    
    time.sleep(random.uniform(0.5, 1.5)) # 隨機暫停 0.5 ~ 1.5 秒.


with open('../data/CWB data/cwb.csv', 'w', newline= '') as f:
    write = csv.writer(f, delimiter= ',') 
    write.writerows(data)
