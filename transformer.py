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
    Sets a random seed to ensure reproducibility.
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

# Define hyperparameters
LOOKBACK = 28   # 예측을 위해 고려할 과거 데이터 기간
PREDICT = 7     # 예측할 미래 기간
BATCH_SIZE = 16
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. 데이터 로드 및 분리 ---
print("1. 데이터를 구글 드라이브에서 로드하고 훈련/테스트 데이터로 분리합니다.")
DATA_PATH = '/content/drive/MyDrive/hackathon_data'

try:
    train_file = os.path.join(DATA_PATH, 'train.csv')
    test_files = [os.path.join(DATA_PATH, f'TEST_{i:02d}.csv') for i in range(10)]

    train_raw_df = pd.read_csv(train_file)
    test_raw_dfs = [pd.read_csv(f) for f in test_files]
    test_raw_df = pd.concat(test_raw_dfs, ignore_index=True)

    print("훈련/테스트 데이터 로드 완료.")
except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다. 구글 드라이브 마운트 및 경로를 확인해 주세요. 오류: {e}")
    exit()

# --- 2. 훈련 데이터 전처리 ---
print("\n2. 훈련 데이터 전처리 및 스케일링을 수행합니다.")
train_raw_df['영업일자'] = pd.to_datetime(train_raw_df['영업일자'])
train_raw_df.set_index('영업일자', inplace=True)
train_raw_df.sort_index(inplace=True)

menu_counts = train_raw_df['영업장명_메뉴명'].value_counts()
most_frequent_menu = menu_counts.idxmax()
print(f"가장 많은 기록이 있는 메뉴: '{most_frequent_menu}'를 선택했습니다.")

train_menu_df = train_raw_df[train_raw_df['영업장명_메뉴명'] == most_frequent_menu].copy()
train_menu_df = train_menu_df['매출수량'].resample('D').sum().fillna(0).to_frame()
train_menu_df.rename(columns={'매출수량': 'y'}, inplace=True)

# 훈련 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_menu_df['y'].values.reshape(-1, 1))

# --- 3. Transformer 데이터셋 생성 ---
def create_dataset(dataset, lookback, predict):
    X, Y = [], []
    for i in range(len(dataset) - lookback - predict + 1):
        feature_sequence = dataset[i:(i + lookback), 0]
        label_sequence = dataset[(i + lookback):(i + lookback + predict), 0]
        X.append(feature_sequence)
        Y.append(label_sequence)
    return np.array(X), np.array(Y)

X_train_np, Y_train_np = create_dataset(scaled_train_data, LOOKBACK, PREDICT)

# NumPy 배열을 PyTorch 텐서로 변환
X_train = torch.FloatTensor(X_train_np).to(DEVICE)
Y_train = torch.FloatTensor(Y_train_np).to(DEVICE)

# --- 4. Transformer 모델 정의 및 학습 ---
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_encoder_layers=2, output_dim=7):
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
input_dim = 1
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
    y_pred = model(X_train.unsqueeze(2))
    loss = loss_function(y_pred, Y_train)
    loss.backward()
    optimizer.step()

print("Transformer 모델 학습이 완료되었습니다.")

# --- 5. 테스트 데이터 전처리 및 예측 ---
print("\n4. 테스트 데이터로 예측을 수행합니다.")
test_raw_df['영업일자'] = pd.to_datetime(test_raw_df['영업일자'])
test_raw_df.set_index('영업일자', inplace=True)
test_raw_df.sort_index(inplace=True)
test_menu_df = test_raw_df[test_raw_df['영업장명_메뉴명'] == most_frequent_menu].copy()
test_menu_df = test_menu_df['매출수량'].resample('D').sum().fillna(0).to_frame()
test_menu_df.rename(columns={'매출수량': 'y'}, inplace=True)

# 예측을 위해 훈련 데이터의 마지막 LOOKBACK 기간을 사용
last_train_sequence = scaled_train_data[-LOOKBACK:]
last_train_tensor = torch.FloatTensor(last_train_sequence).to(DEVICE).unsqueeze(0)

model.eval()
with torch.no_grad():
    transformer_predictions_scaled = model(last_train_tensor).cpu().numpy().flatten()

    # 예측된 스케일링 값을 원래 스케일로 되돌립니다.
    transformer_predictions = scaler.inverse_transform(transformer_predictions_scaled.reshape(-1, 1)).flatten()

transformer_predictions[transformer_predictions < 0] = 0

# --- 6. 평가 및 시각화 ---
def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

# 테스트 데이터의 첫 PREDICT 기간을 예측 대상 실제값으로 사용
actual_test_values = test_menu_df['y'].iloc[:PREDICT].values
smape_value = smape(actual_test_values, transformer_predictions)

print("\n--- 최종 예측 결과 ---")
print("Transformer 모델을 활용한 다음 1주일 메뉴 판매량 예측:")
print(pd.Series(transformer_predictions, index=test_menu_df.index[:PREDICT]))
print(f"\n테스트 데이터의 실제 판매량:\n{actual_test_values}")
print(f"\n테스트 데이터에 대한 SMAPE: {smape_value:.2f}%")

print("\n5. 예측 결과 시각화를 생성합니다.")
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(train_menu_df.index, train_menu_df['y'], label='실제 판매량 (훈련)', color='blue')
ax.plot(test_menu_df.index[:PREDICT], actual_test_values, label='실제 판매량 (테스트)', color='green')
ax.plot(test_menu_df.index[:PREDICT], transformer_predictions, label='Transformer 예측', color='red', linestyle='--')
ax.set_title(f'{most_frequent_menu} 메뉴의 7일 판매량 예측')
ax.set_xlabel('날짜')
ax.set_ylabel('판매량')
ax.legend()
ax.grid(True)
plt.show()
