"""
混合架构实时异常检测系统
======================
直接加载 hybrid_pipeline.py 训练出的模型与阈值。

运行: python hybrid_monitor.py
"""

import time
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from collections import deque
from pathlib import Path
import logging
import sys
import docker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 1. 架构定义 (需与 Pipeline 保持一致)
# ============================================================
class LSTMTransformerAutoencoder(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, latent_size=16, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm_encoder = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_dense = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, latent_size))
        self.decoder_dense = nn.Linear(latent_size, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lstm_decoder = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.output_dense = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        lstm_out, _ = self.lstm_encoder(x)
        trans_enc_out = self.transformer_encoder(lstm_out)
        z = self.encoder_dense(trans_enc_out.mean(dim=1))
        z_expanded = self.decoder_dense(z).unsqueeze(1).expand(-1, seq_len, -1)
        trans_dec_out = self.transformer_decoder(tgt=z_expanded, memory=trans_enc_out)
        lstm_dec_out, _ = self.lstm_decoder(trans_dec_out)
        return self.output_dense(lstm_dec_out)

# ============================================================
# 2. Docker 采集器 (保持原样)
# ============================================================
class DockerMetricsCollector:
    def __init__(self, nas_ip, nas_port):
        self.client = docker.DockerClient(base_url=f'tcp://{nas_ip}:{nas_port}', timeout=10)
        self.last_network_stats = {}
    
    def get_container_stats(self, container):
        try:
            stats = container.stats(stream=False)
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_delta / system_delta) * stats['cpu_stats'].get('online_cpus', 1) * 100.0 if system_delta > 0 else 0
            
            memory_mb = stats['memory_stats'].get('usage', 0) / (1024 * 1024)
            
            networks = stats.get('networks', {})
            net_in_bytes = sum(net.get('rx_bytes', 0) for net in networks.values())
            net_out_bytes = sum(net.get('tx_bytes', 0) for net in networks.values())
            
            c_id = container.id[:12]
            if c_id not in self.last_network_stats:
                self.last_network_stats[c_id] = {'in': net_in_bytes, 'out': net_out_bytes, 'ts': time.time()}
                net_in_rate = net_out_rate = 0
            else:
                last = self.last_network_stats[c_id]
                dt = time.time() - last['ts']
                net_in_rate = ((net_in_bytes - last['in']) / 1048576) / dt if dt > 0 else 0
                net_out_rate = ((net_out_bytes - last['out']) / 1048576) / dt if dt > 0 else 0
                self.last_network_stats[c_id] = {'in': net_in_bytes, 'out': net_out_bytes, 'ts': time.time()}
                
            return {'cpu_percent': max(0, cpu_percent), 'memory_mb': memory_mb, 
                    'net_in_rate_mbs': max(0, net_in_rate), 'net_out_rate_mbs': max(0, net_out_rate)}
        except Exception as e:
            return None

# ============================================================
# 3. 监控主进程
# ============================================================
def run_monitor(target_container="qbittorrent", nas_ip="192.168.3.2", nas_port=2375, interval=10):
    seq_len = 120 # 必须与训练时的序列长度严格一致
    features = ['cpu_percent', 'memory_mb', 'net_in_rate_mbs', 'net_out_rate_mbs']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载配置
    with open('output/hybrid_config.json', 'r') as f:
        config = json.load(f)[target_container]
    threshold = config['threshold']
    
    # 2. 加载模型与标化器
    with open(f'models/{target_container}_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    model = LSTMTransformerAutoencoder(input_size=len(features)).to(device)
    model.load_state_dict(torch.load(f'models/{target_container}_hybrid_model.pth', map_location=device))
    model.eval()
    
    # 3. 连接 Docker
    collector = DockerMetricsCollector(nas_ip, nas_port)
    container_obj = collector.client.containers.get(target_container)
    
    window = deque(maxlen=seq_len)
    logger.info(f"🚀 启动实时监控 | 容器: {target_container} | 阈值: {threshold:.6f}")
    logger.info(f"⏳ 正在收集初始数据 ({seq_len} 步)，预计需要 {seq_len * interval / 60:.1f} 分钟...")
    
    while True:
        try:
            stats = collector.get_container_stats(container_obj)
            if not stats: continue
                
            raw_data = np.array([stats[f] for f in features])
            scaled_data = scaler.transform([raw_data])[0]
            window.append(scaled_data)
            
            if len(window) == seq_len:
                tensor_input = torch.FloatTensor(np.array(window)).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(tensor_input)
                    mse_error = torch.mean((tensor_input - output) ** 2).item()
                
                if mse_error > threshold:
                    logger.warning(f"🚨 [异常告警] 误差: {mse_error:.6f} > {threshold:.6f} | CPU: {raw_data[0]:.1f}%, RAM: {raw_data[1]:.1f}MB")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] ✓ 正常 | 误差: {mse_error:.6f} | CPU: {raw_data[0]:.1f}% | NetIn: {raw_data[2]:.2f}MB/s")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 数据积累中... ({len(window)}/{seq_len})")
                
            time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("监控已手动停止。")
            break
        except Exception as e:
            logger.error(f"运行时错误: {e}")
            time.sleep(interval)

if __name__ == '__main__':
    run_monitor(target_container="qbittorrent-app-1")