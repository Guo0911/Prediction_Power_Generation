# AI CUP 2024 根據區域微氣候資料預測發電量競賽

[競賽網站連結](https://tbrain.trendmicro.com.tw/Competitions/Details/36)


## 競賽說明
太陽能光電裝置的發電量與微氣候息息相關，除了太陽輻射之外，溫度與濕度會影響太陽能光電裝置光電效應的進行，而風力又會影響溫度與濕度，另外還需考慮落塵量和雨量的影響。由於花蓮幅員遼闊，擁有各種複雜的地形地貌，本計畫將在各種地形架設各種感測器與太陽能光電裝置，建立不同地區的微氣候資料集與收集太陽能光電裝置的發電量，做為本次比賽之資料使用。本競賽提供2024年間，在17個地點收集的區域微氣候資料與光電發電量作為訓練資料，比賽內容為預測指定之時間與地點的發電瓦數，並且以誤差值為排名依據。


## 競賽時程
本次競賽的重要時程：

- 2024/09/02 - 開放隊伍報名參加
- 2024/09/23 - 下載競賽訓練集
- 2024/11/18 - 競賽訓練集擴充，並發布測試集及開放提交答案
- 2024/12/02 - 公佈 Private Leaderboard 成績並上傳報告
- 2025/01/07 - 公佈競賽的最終名次


## 排名
本次參賽的 Leaderboard 最終名次及成績，競賽最佳成績為 Leaderboard 第一名：

|         |   排名   |  絕對誤差  |  競賽最佳成績  |
| :------ |  :----:  | :-------: | :-----------: |
| Public  | 3 / 500  | 326660.64 |   295665.98   |
| Private | 3 / 500  | 351729.70 |   310872.00   |


## 訓練與測試資料集
本次參賽除競賽提供的原始資料外，也另外擴充氣候觀測站資料<sup>[1]</sup>及 pvlib 套件<sup>[2]</sup>計算的太陽與輻射量特徵，下表為競賽期間使用的訓練集範例。由於 pvlib 的計算時間較長，因此提供[完整資料集下載連結](https://drive.google.com/file/d/1pwjorwPr3oMVKrbUFIHCWNZsIAzAkoBj/view?usp=sharing)，解壓縮後覆蓋原先的 `data` 資料夾即可。


| LocationCode | DateTime       | WindSpeed(m/s) | Pressure(hpa) | Temperature(簞C) | Humidity(%) | Sunlight(Lux) | Power(mW) | lat     | lon      | direction | pres_cwb | temp_cwb | rh_cwb | precp_cwb | rad_cwb | sun_cwb | visb_cwb | uvi_cwb | cloud_cwb | apparent_zenith | zenith      | apparent_elevation | elevation   | azimuth     | ghi         | dni         | dhi         | num_of_min | day_of_year | month | day | hour | min | hour_sin    | hour_cos      |
|--------------|----------------|----------------|---------------|-----------------|-------------|---------------|-----------|---------|----------|-----------|----------|----------|--------|-----------|---------|---------|----------|---------|-----------|-----------------|-------------|--------------------|-------------|-------------|-------------|-------------|-------------|------------|-------------|-------|-----|------|-----|-------------|---------------|
| 1            | 2024/1/1 10:58 | 0              | 1017.58       | 18.4            | 95          | 8361.67       | 14.66     | 23.8994 | 121.5444 | 181       | 976.3    | 14.7     | 90     | 0         | 0.2     | 0       | 12       | 1.61    | 10        | 49.07593769     | 49.07612268 | 40.92406231        | 40.92387732 | 161.9850634 | 673.0227681 | 881.9232636 | 95.31169172 | 658        | 1           | 1     | 1   | 10   | 58  | 0.5         | -0.866025404  |
| 1            | 2024/1/1 10:59 | 0              | 1017.55       | 18.4            | 95.7        | 8720          | 16.08     | 23.8994 | 121.5444 | 181       | 976.3    | 14.7     | 90     | 0         | 0.2     | 0       | 12       | 1.61    | 10        | 49.00577101     | 49.00595554 | 40.99422899        | 40.99404446 | 162.2750712 | 674.1364679 | 882.2701296 | 95.38223326 | 659        | 1           | 1     | 1   | 10   | 59  | 0.5         | -0.866025404  |
| 1            | 2024/1/1 11:00 | 0.12           | 1017.49       | 18.4            | 96.2        | 8798.33       | 16.23     | 23.8994 | 121.5444 | 181       | 975.2    | 15       | 88     | 0         | 0.2     | 0       |          | 1.67    |           | 48.93670774     | 48.93689143 | 41.06329226        | 41.06310857 | 162.5658499 | 675.2315415 | 882.6103677 | 95.45154285 | 660        | 1           | 1     | 1   | 11   | 0   | 0.258819045 | -0.965925826  |


<sup>[1]</sup> [CODiS 氣候觀測資料查詢服務](https://codis.cwa.gov.tw/StationData)  
<sup>[2]</sup> [pvlib-python | Github](https://github.com/pvlib/pvlib-python)

## 模型復現
本次參賽最佳成績是採用 LightGBM 模型，在 `notebook` 資料夾中存放 Kaggle 執行結果可供參考。

以下為本組最佳模型復現流程，若透過[連結](https://drive.google.com/file/d/1pwjorwPr3oMVKrbUFIHCWNZsIAzAkoBj/view?usp=sharing)下載完整資料集並覆蓋 `data` 資料夾，則可跳過 Step. 5 ~ 7 的部分：

### Step 1. 創建 Python 環境並安裝套件 (使用 Anaconda 進行)
```bash
conda create -n aicup2024 python==3.10.14 --y
```

### Step 2. 切換 Python 環境
```bash
conda activate aicup2024
```

### Step 3. 安裝後續步驟需要的所有套件
```bash
pip install -r requirements.txt
```

### Step 4. 切換至程式碼目錄
```bash
cd ./program
```

### Step 5. 蒐集 CWB 氣候觀測站資料 (執行時間約 8 ~ 16 分鐘，依照設備和網路狀況而定)
```bash
python web_crawler_for_CWB.py
```
程式的執行過程可參考 `notebook/[AI CUP] Web crawler for CWB.ipynb` 或 [Kaggle 執行紀錄](https://www.kaggle.com/code/guojhihrong/ai-cup-web-crawler-for-cwb)

### Step 6. 創建訓練集和測試集 (執行時間約 5 ~ 10 小時<sup>*</sup>，依照設備而定)
<sup>*</sup>此步驟會使用 `pvlib` 套件計算太陽和輻射量特徵，因此耗時較久，建議直接透過[連結](https://drive.google.com/file/d/1pwjorwPr3oMVKrbUFIHCWNZsIAzAkoBj/view?usp=sharing)下載完整資料集並覆蓋 `data` 資料夾

```bash
python create_training_dataset.py

python create_test_dataset.py
```
訓練集程式的執行過程可參考 `notebook/[AI CUP] Create training dataset.ipynb` 或 [Kaggle 執行紀錄](https://www.kaggle.com/code/guojhihrong/ai-cup-create-training-dataset)

測試集程式的執行過程可參考 `notebook/[AI CUP] Create test dataset.ipynb` 或 [Kaggle 執行紀錄](https://www.kaggle.com/code/guojhihrong/ai-cup-create-test-dataset)


### Step 7. 資料前處理 (執行時間約 30 ~ 90 秒，依照設備而定)
可以挑選所需特徵或資料處理方式，復現最佳模型請保持預設狀態即可
```bash
python data_preprocess.py
```
程式的執行過程可參考 `notebook/[AI CUP] Data Preprocess.ipynb` 或 [Kaggle 執行紀錄](https://www.kaggle.com/code/guojhihrong/ai-cup-data-preprocess)

### Step 8. 訓練 LightGBM 模型 (執行時間約 15 ~ 90 分鐘，依照設備而定)
預設不執行 5 Folds Cross Validation，若有需要請修改程式碼(調整 `"` 包覆部分)，使用 CV 會大幅增加執行所需時間
```bash
python training_LightGBM.py
```
程式的執行過程可參考 `notebook/[AI CUP] Training LightGBM.ipynb` 或 [Kaggle 執行紀錄](https://www.kaggle.com/code/guojhihrong/ai-cup-training-lightgbm)