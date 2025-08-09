import os
import random
import glob
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def set_seed(seed=42):
    """
    재현성을 위해 랜덤 시드를 설정합니다.
    """
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

# 하이퍼파라미터 정의
# 훈련 데이터 전체를 LOOKBACK, 테스트 데이터 전체를 PREDICT 기간으로 설정
BATCH_SIZE = 16
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. 데이터 로드 및 병합 ---
print("1. 데이터를 구글 드라이브에서 로드하고 테스트 데이터를 병합합니다.")
DATA_PATH = '/content/drive/MyDrive/hackathon_data'

try:
    train_file = os.path.join(DATA_PATH, 'train.csv')
    test_files = sorted(glob.glob(os.path.join(DATA_PATH, 'TEST_*.csv')))

    train_raw_df = pd.read_csv(train_file, encoding='utf-8')
    test_dfs = [pd.read_csv(f, encoding='utf-8') for f in test_files]
    test_raw_df = pd.concat(test_dfs, ignore_index=True)

    print("훈련/테스트 데이터 로드 및 병합 완료.")
except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다. 구글 드라이브 마운트 및 경로를 확인해 주세요. 오류: {e}")
    exit()

# --- 2. 훈련 데이터 전처리 및 모델 학습 ---
print("\n2. 훈련 데이터 전처리 및 스케일링을 수행하고 모델을 학습합니다.")
train_raw_df['영업일자'] = pd.to_datetime(train_raw_df['영업일자'])
train_raw_df.set_index('영업일자', inplace=True)
train_raw_df.sort_index(inplace=True)

menu_counts = train_raw_df['영업장명_메뉴명'].value_counts()
most_frequent_menu = menu_counts.idxmax()
print(f"가장 많은 기록이 있는 메뉴: '{most_frequent_menu}'를 선택했습니다.")

# 영업장명과 메뉴명 분리
train_raw_df[['영업장명', '메뉴명']] = train_raw_df['영업장명_메뉴명'].str.split('_', n=1, expand=True)

# 계절성 특성 생성 함수
def get_seasonality(menu_name):
    summer_keywords = ['냉면', '비빔밥', '에이드', '아이스크림', '아이스']
    winter_keywords = ['찌개', '탕', '국밥', '우동', '핫']
    menu_name_lower = menu_name.lower()
    if any(keyword in menu_name_lower for keyword in summer_keywords):
        return 1  # 여름
    if any(keyword in menu_name_lower for keyword in winter_keywords):
        return 2  # 겨울
    return 0  # 계절 무관

train_raw_df['seasonality'] = train_raw_df['메뉴명'].apply(get_seasonality)

train_menu_df = train_raw_df[train_raw_df['영업장명_메뉴명'] == most_frequent_menu].copy()
train_menu_df = train_menu_df.resample('D').first().fillna(method='ffill').fillna(0)
train_menu_df = train_menu_df[['매출수량', 'seasonality']].fillna({'매출수량': 0, 'seasonality': 0})
train_menu_df.rename(columns={'매출수량': 'y'}, inplace=True)

# 병합된 테스트 데이터 전처리
test_raw_df['영업일자'] = pd.to_datetime(test_raw_df['영업일자'])
test_raw_df.set_index('영업일자', inplace=True)
test_raw_df.sort_index(inplace=True)

# 영업장명과 메뉴명 분리
test_raw_df[['영업장명', '메뉴명']] = test_raw_df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
test_raw_df['seasonality'] = test_raw_df['메뉴명'].apply(get_seasonality)

test_menu_df = test_raw_df[test_raw_df['영업장명_메뉴명'] == most_frequent_menu].copy()
test_menu_df = test_menu_df.resample('D').first().fillna(method='ffill').fillna(0)
test_menu_df = test_menu_df[['매출수량', 'seasonality']].fillna({'매출수량': 0, 'seasonality': 0})
test_menu_df.rename(columns={'매출수량': 'y'}, inplace=True)

# LOOKBACK과 PREDICT 기간 정의
LOOKBACK = len(train_menu_df)
PREDICT = len(test_menu_df)
print(f"LOOKBACK 기간: {LOOKBACK}일 (train.csv 전체)")
print(f"PREDICT 기간: {PREDICT}일 (TEST_00 ~ TEST_09 전체)")

# 훈련 데이터와 테스트 데이터 스케일링
y_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler.fit(train_menu_df['y'].values.reshape(-1, 1))
scaled_train_y = y_scaler.transform(train_menu_df['y'].values.reshape(-1, 1))
scaled_test_y = y_scaler.transform(test_menu_df['y'].values.reshape(-1, 1))

# 계절성 특성 스케일링 (값의 범위가 0, 1, 2이므로 스케일링을 적용)
seasonality_scaler = MinMaxScaler(feature_range=(0, 1))
seasonality_scaler.fit(train_menu_df['seasonality'].values.reshape(-1, 1))
scaled_train_seasonality = seasonality_scaler.transform(train_menu_df['seasonality'].values.reshape(-1, 1))
scaled_test_seasonality = seasonality_scaler.transform(test_menu_df['seasonality'].values.reshape(-1, 1))

# 단일 훈련 데이터셋 생성 (train 데이터를 보고 test 데이터를 예측하도록)
# X는 y와 seasonality를 모두 포함
X_train_np = np.concatenate((scaled_train_y, scaled_train_seasonality), axis=1).reshape(1, LOOKBACK, -1)
Y_train_np = scaled_test_y.reshape(1, PREDICT)

# NumPy 배열을 PyTorch 텐서로 변환
X_train = torch.FloatTensor(X_train_np).to(DEVICE)
Y_train = torch.FloatTensor(Y_train_np).to(DEVICE)

# --- 3. Transformer 모델 정의 및 학습 ---
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_encoder_layers=2, output_dim=PREDICT):
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model

        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = self.embedding(x)  # (batch_size, sequence_length, d_model)
        x = self.transformer_encoder(x) # (batch_size, sequence_length, d_model)
        return self.fc_out(x[:, -1, :]) # 마지막 토큰의 출력을 사용하여 예측

print("\n3. Transformer 모델 학습을 시작합니다.")
input_dim = 2 # 매출 수량(y) + 계절성(seasonality)
d_model = 64
nhead = 4
num_encoder_layers = 2
output_dim = PREDICT

model = TransformerPredictor(input_dim, d_model, nhead, num_encoder_layers, output_dim).to(DEVICE)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_function(y_pred, Y_train)
    loss.backward()
    optimizer.step()

print("Transformer 모델 학습이 완료되었습니다.")

# --- 4. 단일 예측 및 평가 ---
print("\n4. TEST_00.csv부터 TEST_09.csv 전체 데이터로 예측을 수행하고 평가합니다.")
# 예측을 위해 훈련 데이터 전체를 사용
train_sequence_for_pred = X_train

model.eval()
with torch.no_grad():
    transformer_predictions_scaled = model(train_sequence_for_pred).cpu().numpy().flatten()
    
    # 예측된 스케일링 값을 원래 스케일로 되돌림
    transformer_predictions = y_scaler.inverse_transform(transformer_predictions_scaled.reshape(-1, 1)).flatten()
    transformer_predictions[transformer_predictions < 0] = 0
    actual_test_values = test_menu_df['y'].values

# --- 5. 최종 평가 및 시각화 ---
def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

final_smape = smape(actual_test_values, transformer_predictions)

print("\n--- 최종 예측 결과 ---")
print(f"TEST_00.csv부터 TEST_09.csv 전체 예측에 대한 최종 SMAPE: {final_smape:.2f}%")

print("\n6. 예측 결과 시각화를 생성합니다.")
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(train_menu_df.index, train_menu_df['y'], label='실제 판매량 (훈련)', color='blue')
ax.plot(test_menu_df.index, actual_test_values, label='실제 판매량 (TEST_00 ~ TEST_09)', color='green')
ax.plot(test_menu_df.index, transformer_predictions, label='Transformer 예측', color='red', linestyle='--')

ax.set_title(f'{most_frequent_menu} 메뉴의 전체 테스트 기간 판매량 예측')
ax.set_xlabel('날짜')
ax.set_ylabel('판매량')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
