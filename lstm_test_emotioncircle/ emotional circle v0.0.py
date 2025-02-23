import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 定义 LSTM 模型
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x 的尺寸为 (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        return out

## ------------------- 数据预处理 -------------------

# 1. 读取 CSV 文件
df = pd.read_csv("/Users/admin/Library/Mobile Documents/com~apple~CloudDocs/Source/Python3/lstm_test_emotioncircle/test_date1.csv")
print("原始数据：\n", df.head())

# 假设 CSV 中的列顺序不确定，这里统一命名为：
df.columns = ['delay', 'arousal', 'valence', 'stimulation']

# 2. 根据 delay 对数据排序（确保时间顺序正确）
df.sort_values('delay', inplace=True)
df.reset_index(drop=True, inplace=True)

# 3. 构造输入特征和目标
# 本示例中，我们使用所有4个字段构造时间序列的输入，
# 目标则设为下一时刻的 arousal, valence, stimulation（即“坐标变化”和 stimulation）
features = df[['delay', 'arousal', 'valence', 'stimulation']].values
target = df[['arousal', 'valence', 'stimulation']].values

# 4. 数据归一化
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

features_scaled = scaler_X.fit_transform(features)
target_scaled = scaler_Y.fit_transform(target)

features_scaled = np.round(features_scaled, decimals=1)
target_scaled = np.round(target_scaled, decimals=1)


# 5. 根据时间序列构造样本（滑动窗口）
seq_length = 4  # 过去4个时间步作为输入
X_seq = []
Y_seq = []
for i in range(len(features_scaled) - seq_length):
    # 输入：连续 seq_length 个时间步的所有特征
    X_seq.append(features_scaled[i : i + seq_length])
    # 目标：下一个时间步的 arousal, valence, stimulation
    Y_seq.append(target_scaled[i + seq_length])

# 转换为 PyTorch 张量
X_seq = torch.tensor(X_seq, dtype=torch.float32)
Y_seq = torch.tensor(Y_seq, dtype=torch.float32)

print("输入序列尺寸：", X_seq.shape)   # (样本数, seq_length, 4)
print("目标数据尺寸：", Y_seq.shape)     # (样本数, 3)

# ------------------- 构建并测试 LSTM 模型 -------------------

# 模型超参数
input_size = 4    # delay, arousal, valence, stimulation
hidden_size = 10   # 隐藏层维度
output_size = 3   # 预测 arousal, valence, stimulation
model = LSTMPredictor(input_size, hidden_size, output_size)

# 前向传播，查看模型输出
output = model(X_seq)

print("模型输出：\n", output)

