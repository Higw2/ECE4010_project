import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"D:\大二下\ECE4010\project\data\docker_metrics_partial_20260417_113301.csv")

features = ['cpu_percent', 'memory_mb', 'net_in_rate_mbs', 'net_out_rate_mbs']
container = 'qbittorrent-app-1'
container_df = df[df['container_name'] == container][features].copy()
X = np.nan_to_num(container_df.values, nan=0.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nStandardScaler 处理后 :")
print(pd.DataFrame(X_scaled[:8], columns=features).round(4).to_string(index=False))

SEQ_LEN = 120
sequences = np.array([X_scaled[i:i+SEQ_LEN] for i in range(len(X_scaled) - SEQ_LEN + 1)])
print(f"\n滑动窗口后 shape: {sequences.shape}  → (样本数, 时间步, 特征数)")
print(f"第1个窗口第1行: {sequences[0][0].round(4)}")
