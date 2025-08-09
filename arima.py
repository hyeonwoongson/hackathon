import os
import random
import glob
import re

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from tqdm import tqdm

# 추가 부분
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

LOOKBACK, PREDICT, BATCH_SIZE, EPOCHS = 28, 7, 16, 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_load_path = "/content/drive/MyDrive/hackathon_data/train.csv"
train_df = pd.read_csv(train_load_path)

# 추가 부분
print("데이터를 구글 드라이브에서 로드합니다.")
DATA_PATH = '/content/drive/MyDrive/hackathon_data'
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')
TEST_FILES = [os.path.join(DATA_PATH, f'TEST_{i:02d}.csv') for i in range(10)]

try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_dfs = [pd.read_csv(f) for f in TEST_FILES]
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # 학습 데이터와 테스트 데이터를 결합하여 전체 시계열을 만듭니다.
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print("데이터 로드 및 결합 완료.")
except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다. 구글 드라이브 마운트 및 경로를 확인해 주세요. 오류: {e}")
    # 파일이 없는 경우 코드 실행을 중단합니다.
    exit()

print("데이터 전처리 및 피처 엔지니어링을 수행합니다.")
combined_df['영업일자'] = pd.to_datetime(combined_df['영업일자'])
combined_df.set_index('영업일자', inplace=True)
combined_df.sort_index(inplace=True)

# 가장 많은 기록이 있는 메뉴를 선택하여 모델링 (예시)
menu_counts = combined_df['영업장명_메뉴명'].value_counts()
most_frequent_menu = menu_counts.idxmax()
print(f"가장 많은 기록이 있는 메뉴: '{most_frequent_menu}'를 선택했습니다.")
menu_df = combined_df[combined_df['영업장명_메뉴명'] == most_frequent_menu].copy()

# 일별 판매량으로 리샘플링하고 결측치를 0으로 채웁니다.
menu_df = menu_df['매출수량'].resample('D').sum().fillna(0).to_frame()

# 요일 및 월 피처 생성
menu_df['dayofweek'] = menu_df.index.dayofweek
menu_df['month'] = menu_df.index.month

# 'exog' 변수를 전체 데이터에 대해 한 번에 생성하여 컬럼 불일치 방지
all_exog_variables = pd.get_dummies(menu_df[['dayofweek']], columns=['dayofweek'], drop_first=True)
all_exog_variables = all_exog_variables.astype(int)
all_rf_features = menu_df[['dayofweek', 'month']]

# 학습/테스트 데이터 분리 (마지막 7일을 테스트 데이터로 사용)
train_end_date = menu_df.index[-8]
train_df = menu_df.loc[:train_end_date]
test_df = menu_df.loc[menu_df.index[-7]:]

exog_train = all_exog_variables.loc[:train_end_date]
exog_test = all_exog_variables.loc[menu_df.index[-7]:]

rf_features_train = all_rf_features.loc[:train_end_date]
rf_features_test = all_rf_features.loc[menu_df.index[-7]:]

# SARIMAX 모델 학습
print("SARIMAX 모델 학습을 시작합니다.")
try:
    sarimax_model = SARIMAX(train_df['매출수량'], exog=exog_train,
                            order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                            enforce_stationarity=False, enforce_invertibility=False)
    sarimax_fit = sarimax_model.fit(disp=False)
    print("SARIMAX 모델 학습이 완료되었습니다.")
except Exception as e:
    print(f"SARIMAX 모델 학습 중 오류 발생: {e}")

# Random Forest 모델 학습 (SARIMAX의 잔차를 예측)
print("Random Forest 모델 학습을 시작합니다.")
sarimax_train_pred = sarimax_fit.predict(start=train_df.index[0], end=train_df.index[-1], exog=exog_train, typ='levels')
residuals = train_df['매출수량'] - sarimax_train_pred

rf_data_for_train = pd.concat([rf_features_train, residuals], axis=1)
rf_data_for_train.columns = ['dayofweek', 'month', 'residuals']
rf_data_for_train_clean = rf_data_for_train.dropna()

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(rf_features_train, residuals)
print("Random Forest 모델 학습이 완료되었습니다.")

print("다음 7일 판매량을 예측합니다.")
sarimax_future_pred = sarimax_fit.predict(start=test_df.index[0], end=test_df.index[-1], exog=exog_test, typ='levels')
rf_future_residuals = rf_model.predict(rf_features_test)
final_predictions = sarimax_future_pred.values + rf_future_residuals

# SMAPE 계산 함수
def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

smape_value = smape(test_df['매출수량'].values, final_predictions)

print("\n--- 최종 예측 결과 ---")
print("SARIMAX + Random Forest를 활용한 다음 1주일 메뉴 판매량 예측:")
print(pd.Series(final_predictions, index=test_df.index))
print(f"\n테스트 데이터의 실제 판매량:\n{test_df['매출수량']}")
print(f"\n테스트 데이터에 대한 SMAPE: {smape_value:.2f}%")

plt.figure(figsize=(15, 7))
plt.plot(train_df.index, train_df['매출수량'], label='실제 판매량 (학습 데이터)', color='blue')
plt.plot(test_df.index, test_df['매출수량'], label='실제 판매량 (테스트 데이터)', color='green')
plt.plot(test_df.index, final_predictions, label='최종 예측 (SARIMAX+RF)', color='red', linestyle='--')
plt.title(f'{most_frequent_menu} 메뉴의 7일 판매량 예측')
plt.xlabel('날짜')
plt.ylabel('판매량')
plt.legend()
plt.grid(True)
plt.show()
