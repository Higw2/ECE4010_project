"""
多模型实时异常检测系统
======================

自动从 best_models.json 读取最佳模型配置
支持 LSTM / Transformer / 混合模型

运行: python realtime_monitor_multimodel.py
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
from datetime import datetime

# ============================================================
# 配置日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('realtime_monitor_multimodel.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 1. 模型架构定义（与训练脚本保持一致）
# ============================================================

class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder"""
    def __init__(self, input_size=4, hidden_size=16, latent_size=8, num_layers=1):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.encoder_dense = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size)
        )
        self.decoder_dense = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output_dense = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        _, (h, _) = self.encoder_lstm(x)
        z = self.encoder_dense(h[-1])
        seq_len = x.shape[1]
        h_0 = self.decoder_dense(z).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        z_repeated = h_0.expand(seq_len, -1, -1).transpose(0, 1)
        output, _ = self.decoder_lstm(z_repeated, (h_0, c_0))
        output = self.output_dense(output)
        return output


class TransformerAutoencoder(nn.Module):
    """Transformer Autoencoder"""
    def __init__(self, input_size=4, hidden_size=32, num_heads=4, num_layers=2, 
                 latent_size=8, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # 位置编码
        self.positional_encoding = nn.Parameter(
            self._get_positional_encoding(seq_len=100, d_model=hidden_size),
            requires_grad=False
        )
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_dense = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size)
        )
        
        self.decoder_dense = nn.Linear(latent_size, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_size, input_size)
    
    def _get_positional_encoding(self, seq_len=100, d_model=32):
        """生成位置编码"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_proj = self.input_projection(x)
        pe = self.positional_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
        x_proj = x_proj + pe
        encoded = self.encoder(x_proj)
        z_input = encoded.mean(dim=1)
        z = self.encoder_dense(z_input)
        z_expanded = self.decoder_dense(z).unsqueeze(1).expand(-1, seq_len, -1)
        decoded = self.decoder(z_expanded, encoded)
        output = self.output_projection(decoded)
        return output


class LSTMAttentionAutoencoder(nn.Module):
    """LSTM + Attention"""
    def __init__(self, input_size=4, hidden_size=16, latent_size=8, num_heads=4):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.encoder_dense = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size)
        )
        self.decoder_dense = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.output_dense = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        lstm_out, (h, c) = self.encoder_lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        z = self.encoder_dense(h[-1])
        seq_len = x.shape[1]
        h_0 = self.decoder_dense(z).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        z_repeated = h_0.expand(seq_len, -1, -1).transpose(0, 1)
        lstm_dec, _ = self.decoder_lstm(z_repeated, (h_0, c_0))
        attn_dec, _ = self.decoder_attention(lstm_dec, lstm_dec, lstm_dec)
        output = self.output_dense(attn_dec)
        return output


# ============================================================
# 2. Docker 采集器
# ============================================================

class DockerMetricsCollector:
    """从 NAS Docker 采集性能指标"""
    def __init__(self, nas_ip='192.168.3.2', nas_port=2375, timeout=10):
        try:
            self.client = docker.DockerClient(
                base_url=f'tcp://{nas_ip}:{nas_port}',
                timeout=timeout
            )
            self.client.ping()
            logger.info(f"✓ 成功连接 NAS Docker: {nas_ip}:{nas_port}")
        except Exception as e:
            logger.error(f"✗ 无法连接 NAS Docker: {e}")
            raise
        
        self.last_network_stats = {}
    
    def get_container_stats(self, container):
        """获取容器统计信息"""
        try:
            stats = container.stats(stream=False)
            container_id = container.id[:12]
            
            # CPU 使用率
            cpu_delta = (stats['cpu_stats']['cpu_usage']['total_usage'] -
                        stats['precpu_stats']['cpu_usage']['total_usage'])
            system_delta = (stats['cpu_stats']['system_cpu_usage'] -
                           stats['precpu_stats']['system_cpu_usage'])
            cpu_percent = 0
            if system_delta > 0:
                cpu_count = stats['cpu_stats'].get('online_cpus',
                          len(stats['cpu_stats']['cpu_usage'].get('percpu_usage', [1])))
                cpu_percent = (cpu_delta / system_delta) * cpu_count * 100.0
            
            # 内存使用
            memory_usage = stats['memory_stats'].get('usage', 0)
            memory_mb = memory_usage / (1024 * 1024)
            
            # 网络 I/O
            networks = stats.get('networks', {})
            net_in_bytes = sum(net.get('rx_bytes', 0) for net in networks.values())
            net_out_bytes = sum(net.get('tx_bytes', 0) for net in networks.values())
            
            if container_id not in self.last_network_stats:
                self.last_network_stats[container_id] = {
                    'in_bytes': net_in_bytes,
                    'out_bytes': net_out_bytes,
                    'timestamp': time.time()
                }
                net_in_rate = 0
                net_out_rate = 0
            else:
                last = self.last_network_stats[container_id]
                time_delta = time.time() - last['timestamp']
                in_delta = net_in_bytes - last['in_bytes']
                out_delta = net_out_bytes - last['out_bytes']
                net_in_rate = (in_delta / (1024 * 1024)) / time_delta if time_delta > 0 else 0
                net_out_rate = (out_delta / (1024 * 1024)) / time_delta if time_delta > 0 else 0
                self.last_network_stats[container_id] = {
                    'in_bytes': net_in_bytes,
                    'out_bytes': net_out_bytes,
                    'timestamp': time.time()
                }
            
            return {
                'cpu_percent': max(0, cpu_percent),
                'memory_mb': memory_mb,
                'net_in_rate_mbs': max(0, net_in_rate),
                'net_out_rate_mbs': max(0, net_out_rate),
            }
        except Exception as e:
            logger.warning(f"获取容器统计失败: {e}")
            return None


# ============================================================
# 3. 多模型实时异常检测器
# ============================================================

class MultiModelAnomalyDetector:
    """支持多模型的实时异常检测器"""
    
    def __init__(self, target_container, model_choice='auto', 
                 nas_ip='192.168.3.2', nas_port=2375):
        """
        Args:
            target_container: 容器名称
            model_choice: 'auto'(自动选择最佳) / 'lstm' / 'transformer' / 'lstm_attention'
            nas_ip: NAS IP
            nas_port: Docker API 端口
        """
        self.container_name = target_container
        self.model_choice = model_choice
        self.features = ['cpu_percent', 'memory_mb', 'net_in_rate_mbs', 'net_out_rate_mbs']
        self.window = deque(maxlen=6)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.stats = {
            'total_samples': 0,
            'anomaly_count': 0,
            'max_error': 0,
            'min_error': float('inf'),
            'start_time': datetime.now()
        }
        
        # 初始化 Docker 采集器
        logger.info(f"正在连接 NAS Docker: {nas_ip}:{nas_port}...")
        self.collector = DockerMetricsCollector(nas_ip=nas_ip, nas_port=nas_port)
        
        # 获取容器对象
        try:
            self.container_obj = self.collector.client.containers.get(self.container_name)
            logger.info(f"✓ 成功找到容器: {self.container_name}")
        except Exception as e:
            logger.error(f"✗ 无法找到容器 '{self.container_name}': {e}")
            sys.exit(1)
        
        # 加载模型和配置
        self._load_models_and_config()
    
    def _load_models_and_config(self):
        """加载模型配置和选择最佳模型"""
        
        logger.info("\n" + "="*60)
        logger.info("加载模型配置...")
        logger.info("="*60)
        
        # 1. 从 JSON 读取最佳模型配置
        config_path = 'output/best_models.json'
        if not Path(config_path).exists():
            logger.error(f"✗ 找不到配置文件: {config_path}")
            logger.error("  请先运行 python enhanced_pipeline_multimodel.py")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            best_models_config = json.load(f)
        
        if self.container_name not in best_models_config:
            logger.error(f"✗ 配置文件中没有容器 '{self.container_name}'")
            logger.error(f"  可用的容器: {list(best_models_config.keys())}")
            sys.exit(1)
        
        container_config = best_models_config[self.container_name]
        
        # 2. 根据用户选择或自动选择模型
        if self.model_choice == 'auto':
            selected_model = container_config['best_model']
            logger.info(f"✓ 自动选择最佳模型: {selected_model.upper()}")
        else:
            selected_model = self.model_choice
            logger.info(f"✓ 使用指定模型: {selected_model.upper()}")
        
        self.selected_model = selected_model
        self.threshold = container_config['threshold']
        
        logger.info(f"  告警阈值: {self.threshold:.6f}")
        logger.info(f"  预期异常率: {container_config['anomaly_rate']*100:.2f}%\n")
        
        # 3. 加载 Scaler
        scaler_path = f'models/{self.container_name}_scaler.pkl'
        if not Path(scaler_path).exists():
            logger.error(f"✗ 标化器文件不存在: {scaler_path}")
            sys.exit(1)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"✓ 加载标化器")
        
        # 4. 加载选定的模型
        model_path = f'models/{self.container_name}_{selected_model}_model.pth'
        if not Path(model_path).exists():
            logger.error(f"✗ 模型文件不存在: {model_path}")
            sys.exit(1)
        
        # 根据模型类型创建模型实例
        model_configs = {
            'lstm': {
                'class': LSTMAutoencoder,
                'params': {'hidden_size': 16, 'latent_size': 8, 'num_layers': 1}
            },
            'transformer': {
                'class': TransformerAutoencoder,
                'params': {'hidden_size': 32, 'num_heads': 4, 'num_layers': 2, 'latent_size': 8}
            },
            'lstm_attention': {
                'class': LSTMAttentionAutoencoder,
                'params': {'hidden_size': 16, 'latent_size': 8, 'num_heads': 4}
            }
        }
        
        config = model_configs[selected_model]
        model_class = config['class']
        model_params = config['params'].copy()
        model_params['input_size'] = len(self.features)
        
        self.model = model_class(**model_params)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✓ 加载模型: {model_path}")
        logger.info(f"\n{'='*60}\n")
    
    def run_monitor(self, interval=10):
        """主监控循环"""
        
        logger.info("🚀 启动多模型实时异常检测")
        logger.info(f"  模型: {self.selected_model.upper()}")
        logger.info(f"  容器: {self.container_name}")
        logger.info(f"  阈值: {self.threshold:.6f}")
        logger.info("="*60 + "\n")
        
        try:
            while True:
                try:
                    # 获取统计信息
                    stats = self.collector.get_container_stats(self.container_obj)
                    if stats is None:
                        time.sleep(interval)
                        continue
                    
                    # 提取特征
                    raw_data = np.array([stats[feat] for feat in self.features])
                    
                    # 标化
                    scaled_data = self.scaler.transform([raw_data])[0]
                    
                    # 加入窗口
                    self.window.append(scaled_data)
                    
                    # 推理
                    if len(self.window) == 6:
                        tensor_input = torch.FloatTensor(
                            np.array(self.window)
                        ).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            output = self.model(tensor_input)
                            mse_error = torch.mean(
                                (tensor_input - output) ** 2
                            ).item()
                        
                        # 更新统计
                        self.stats['total_samples'] += 1
                        self.stats['max_error'] = max(self.stats['max_error'], mse_error)
                        self.stats['min_error'] = min(self.stats['min_error'], mse_error)
                        
                        # 判断异常
                        if mse_error > self.threshold:
                            self.stats['anomaly_count'] += 1
                            logger.warning(f"\n{'='*60}")
                            logger.warning(f"🚨 [异常告警] 容器 '{self.container_name}' 异常！")
                            logger.warning(f"{'='*60}")
                            logger.warning(f"  模型: {self.selected_model.upper()}")
                            logger.warning(f"  重构误差: {mse_error:.6f} > 阈值: {self.threshold:.6f}")
                            logger.warning(f"  超过幅度: {(mse_error - self.threshold):.6f}")
                            logger.warning(f"\n  性能指标:")
                            logger.warning(f"    ├─ CPU: {raw_data[0]:6.1f}%")
                            logger.warning(f"    ├─ 内存: {raw_data[1]:7.1f}MB")
                            logger.warning(f"    ├─ 网络入: {raw_data[2]:5.2f}MB/s")
                            logger.warning(f"    └─ 网络出: {raw_data[3]:5.2f}MB/s")
                            logger.warning(f"{'='*60}\n")
                        else:
                            print(f"[{time.strftime('%H:%M:%S')}] ✓ {self.container_name} | "
                                  f"模型: {self.selected_model.upper()} | "
                                  f"误差: {mse_error:.6f} | CPU: {raw_data[0]:5.1f}%")
                    else:
                        progress = len(self.window)
                        print(f"[{time.strftime('%H:%M:%S')}] 初始化中... ({progress}/6)")
                    
                    time.sleep(interval)
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"监控循环错误: {e}")
                    time.sleep(interval)
        
        finally:
            self._print_summary()
    
    def _print_summary(self):
        """打印统计摘要"""
        logger.info("\n" + "="*60)
        logger.info("📊 监控统计摘要")
        logger.info("="*60)
        
        duration = (datetime.now() - self.stats['start_time']).total_seconds()
        hours = duration / 3600
        
        logger.info(f"运行时间: {hours:.1f} 小时")
        logger.info(f"选用模型: {self.selected_model.upper()}")
        logger.info(f"采样总数: {self.stats['total_samples']}")
        logger.info(f"异常事件: {self.stats['anomaly_count']}")
        
        if self.stats['total_samples'] > 0:
            anomaly_rate = self.stats['anomaly_count'] / self.stats['total_samples'] * 100
            logger.info(f"异常率: {anomaly_rate:.2f}%")
        
        if self.stats['min_error'] != float('inf'):
            logger.info(f"误差范围: [{self.stats['min_error']:.6f}, {self.stats['max_error']:.6f}]")
        
        logger.info("="*60 + "\n")


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*8 + "🚀 多模型实时异常检测系统" + " "*25 + "║")
    print("╚" + "="*58 + "╝\n")
    
    # ================= 配置区 =================
    
    # 1. 监控的容器名称
    TARGET_CONTAINER = "qbittorrent"
    
    # 2. 模型选择
    # 'auto': 自动使用 best_models.json 中推荐的最佳模型
    # 'lstm': 强制使用 LSTM 模型
    # 'transformer': 强制使用 Transformer 模型
    # 'lstm_attention': 强制使用 LSTM + Attention 模型
    MODEL_CHOICE = 'auto'
    
    # 3. NAS 地址
    NAS_IP = '192.168.3.2'
    NAS_PORT = 2375
    
    # 4. 采样间隔（秒）
    SAMPLING_INTERVAL = 10
    
    # ==========================================
    
    logger.info(f"配置信息:")
    logger.info(f"  ├─ 监控容器: {TARGET_CONTAINER}")
    logger.info(f"  ├─ 模型选择: {MODEL_CHOICE}")
    logger.info(f"  ├─ NAS 地址: {NAS_IP}:{NAS_PORT}")
    logger.info(f"  └─ 采样间隔: {SAMPLING_INTERVAL}秒\n")
    
    # 创建检测器
    detector = MultiModelAnomalyDetector(
        target_container=TARGET_CONTAINER,
        model_choice=MODEL_CHOICE,
        nas_ip=NAS_IP,
        nas_port=NAS_PORT
    )
    
    # 启动监控
    try:
        detector.run_monitor(interval=SAMPLING_INTERVAL)
    except KeyboardInterrupt:
        logger.info("\n✓ 监控已停止")
    except Exception as e:
        logger.error(f"✗ 发生致命错误: {e}")
        import traceback
        traceback.print_exc()