"""
终极混合异常检测管道 (LSTM + Transformer)
=========================================
基于 LSTM 作为前端特征提取器，取代 Transformer 原始位置编码。

运行: python hybrid_pipeline.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import warnings
import json
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# ============================================================
# 1. 终极架构定义
# ============================================================

class LSTMTransformerAutoencoder(nn.Module):
    """
    终极混合架构：
    LSTM (局部特征 & 位置编码) -> Transformer Encoder (全局超长程联系) 
    -> Latent Space (降维压缩) 
    -> Transformer Decoder -> LSTM Decoder -> 重构输出
    """
    def __init__(self, input_size=4, hidden_size=32, latent_size=16, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # 1. LSTM 编码层: (Batch, Seq, Input) -> (Batch, Seq, Hidden)
        # 天然包含时间顺序信息，替代 Positional Encoding
        self.lstm_encoder = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        
        # 2. Transformer 编码层: 直接接收 lstm_out，无需加位置编码
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 潜在空间压缩 (Latent Space)
        self.encoder_dense = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size)
        )
        
        # 4. 潜在空间解压
        self.decoder_dense = nn.Linear(latent_size, hidden_size)
        
        # 5. Transformer 解码层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 6. LSTM 解码层
        self.lstm_decoder = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        
        # 7. 输出层
        self.output_dense = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # [编码阶段]
        lstm_out, _ = self.lstm_encoder(x)
        trans_enc_out = self.transformer_encoder(lstm_out)
        
        # [潜在空间压缩] 
        # 对时间步进行全局平均池化，压缩为单一高维状态
        z_input = trans_enc_out.mean(dim=1)  # (Batch, Hidden)
        z = self.encoder_dense(z_input)      # (Batch, Latent)
        
        # [解压阶段]
        z_expanded = self.decoder_dense(z)   # (Batch, Hidden)
        z_expanded = z_expanded.unsqueeze(1).expand(-1, seq_len, -1) # (Batch, Seq, Hidden)
        
        # [解码阶段]
        trans_dec_out = self.transformer_decoder(tgt=z_expanded, memory=trans_enc_out)
        lstm_dec_out, _ = self.lstm_decoder(trans_dec_out)
        
        # [重构输出]
        output = self.output_dense(lstm_dec_out)
        
        return output

# ============================================================
# 2. 数据处理与训练
# ============================================================

def preprocess_data(seq_len=120):
    print("\n" + "="*70 + "\n步骤 1: 数据预处理\n" + "="*70)
    
    data_files = sorted(Path('data').glob('docker_metrics_*.csv'))
    if not data_files:
        print("❌ 没有找到数据文件")
        return None
    
    df = pd.read_csv(data_files[-1])
    container_col = 'container_name' if 'container_name' in df.columns else 'container'
    processed_data = {}
    
    for container in df[container_col].unique():
        print(f"处理容器: {container}")
        container_df = df[df[container_col] == container].copy()
        
        features = ['cpu_percent', 'memory_mb', 'net_in_rate_mbs', 'net_out_rate_mbs']
        available_features = [f for f in features if f in container_df.columns]
        
        if len(available_features) == 0: continue
            
        X = np.nan_to_num(container_df[available_features].values, nan=0.0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        sequences = np.array([X_scaled[i:i+seq_len] for i in range(len(X_scaled) - seq_len + 1)])
        if len(sequences) < 2: continue
            
        n_train = max(1, int(len(sequences) * 0.7))
        processed_data[container] = {
            'train': sequences[:n_train],
            'test': sequences[n_train:],
            'scaler': scaler,
            'n_features': len(available_features)
        }
        print(f"  ✓ 窗口大小: {seq_len}, 训练集: {n_train}, 测试集: {len(sequences)-n_train}")
        
    return processed_data

def train_and_evaluate(processed_data):
    print("\n" + "="*70 + "\n步骤 2: 模型训练与评估\n" + "="*70)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Path('models').mkdir(exist_ok=True)
    Path('output').mkdir(exist_ok=True)
    
    config_results = {}
    
    for container_name, data in processed_data.items():
        print(f"\n🚀 开始训练: {container_name}")
        X_train = torch.FloatTensor(data['train'])
        loader = DataLoader(X_train, batch_size=32, shuffle=True)
        
        model = LSTMTransformerAutoencoder(input_size=data['n_features']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 训练
        epochs = 30
        model.train()
        start_time = time.time()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x = batch.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(x, output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.6f}")
                
        # 评估 & 阈值计算
        model.eval()
        X_test = torch.FloatTensor(data['test']).to(device)
        with torch.no_grad():
            output = model(X_test)
            errors = torch.mean((X_test - output) ** 2, dim=(1, 2)).cpu().numpy()
            
        mean_error, std_error = np.mean(errors), np.std(errors)
        threshold = mean_error + 3 * std_error  # 使用 3-Sigma 原则
        anomaly_rate = np.sum(errors > threshold) / len(errors)
        
        print(f"  ✓ 训练耗时: {time.time() - start_time:.1f}s")
        print(f"  ✓ 阈值设定 (Mean + 3σ): {threshold:.6f} | 测试集异常率: {anomaly_rate*100:.2f}%")
        
        # 保存资产
        torch.save(model.state_dict(), f'models/{container_name}_hybrid_model.pth')
        with open(f'models/{container_name}_scaler.pkl', 'wb') as f:
            pickle.dump(data['scaler'], f)
            
        config_results[container_name] = {
            'threshold': float(threshold),
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'anomaly_rate': float(anomaly_rate)
        }
        
    with open('output/hybrid_config.json', 'w') as f:
        json.dump(config_results, f, indent=2)
    print("\n✅ 所有训练完成！配置已保存至 output/hybrid_config.json")

if __name__ == '__main__':
    data = preprocess_data(seq_len=120)
    if data:
        train_and_evaluate(data)