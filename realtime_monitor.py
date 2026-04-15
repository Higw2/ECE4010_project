"""
实时容器监控与异常检测网关
结合 DockerMetricsCollector 与训练好的 LSTM Autoencoder
========================================================

使用流程：
1. 确保已经完成 collect_metrics.py 采集数据
2. 确保已经运行 complete_pipeline_fixed.py 训练模型
3. 在本脚本中修改 TARGET_CONTAINER 和 ALERT_THRESHOLD
4. 运行：python realtime_monitor.py

生成的告警会同时输出到：
- 控制台 (实时)
- realtime_monitor.log (持久化)
"""

import time
import torch
import torch.nn as nn
import numpy as np
import pickle
from collections import deque
from pathlib import Path
import logging
import sys
import json
from datetime import datetime

# ============================================================
# 必须项：导入 Docker 采集器类
# ============================================================
try:
    import docker
except ImportError:
    print("❌ 缺少 docker 库！请运行: pip install docker")
    sys.exit(1)

# ============================================================
# 配置日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('realtime_monitor.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 1. 模型架构（必须与训练时完全一致！）
# ============================================================

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder 用于时序异常检测
    
    架构：
    - 编码器：LSTM → Dense → Dense(潜在空间)
    - 解码器：Dense → LSTM → Dense
    
    输入形状: (batch, seq_len, n_features)
    输出形状: (batch, seq_len, n_features)
    """
    
    def __init__(self, input_size=4, hidden_size=16, latent_size=8):
        super().__init__()
        
        # 编码器：将时序数据压缩到潜在空间
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.encoder_dense = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size)
        )
        
        # 解码器：从潜在空间还原时序数据
        self.decoder_dense = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.output_dense = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        """
        前向传播
        x: (batch, seq_len, n_features)
        """
        # 编码：LSTM + Dense 压缩
        _, (h, _) = self.encoder_lstm(x)
        z = self.encoder_dense(h[-1])  # (batch, latent_size)
        
        # 解码：从潜在空间还原
        seq_len = x.shape[1]
        h_0 = self.decoder_dense(z).unsqueeze(0)  # (1, batch, hidden_size)
        c_0 = torch.zeros_like(h_0)
        
        # 把单个隐状态复制成完整序列
        z_repeated = h_0.expand(seq_len, -1, -1).transpose(0, 1)  # (batch, seq_len, hidden_size)
        
        output, _ = self.decoder_lstm(z_repeated, (h_0, c_0))
        output = self.output_dense(output)  # (batch, seq_len, n_features)
        
        return output


# ============================================================
# 2. Docker 指标采集器
# ============================================================

class DockerMetricsCollector:
    """从 NAS Docker 采集容器性能指标"""
    
    def __init__(self, nas_ip='192.168.3.2', nas_port=2375, timeout=10):
        self.nas_ip = nas_ip
        self.nas_port = nas_port
        
        # 连接到 NAS Docker
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
        
        # 用于计算网络速率
        self.last_network_stats = {}
    
    def get_container_stats(self, container):
        """获取单个容器的实时统计信息"""
        try:
            stats = container.stats(stream=False)
            container_id = container.id[:12]
            
            # ===== CPU 使用率 =====
            cpu_delta = (stats['cpu_stats']['cpu_usage']['total_usage'] -
                        stats['precpu_stats']['cpu_usage']['total_usage'])
            system_delta = (stats['cpu_stats']['system_cpu_usage'] -
                           stats['precpu_stats']['system_cpu_usage'])
            
            cpu_percent = 0
            if system_delta > 0:
                cpu_count = stats['cpu_stats'].get('online_cpus',
                          len(stats['cpu_stats']['cpu_usage'].get('percpu_usage', [1])))
                cpu_percent = (cpu_delta / system_delta) * cpu_count * 100.0
            
            # ===== 内存使用 =====
            memory_usage = stats['memory_stats'].get('usage', 0)
            memory_mb = memory_usage / (1024 * 1024)
            
            # ===== 网络 I/O =====
            networks = stats.get('networks', {})
            net_in_bytes = sum(net.get('rx_bytes', 0) for net in networks.values())
            net_out_bytes = sum(net.get('tx_bytes', 0) for net in networks.values())
            
            # 计算网络速率 (MB/s)
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
            logger.warning(f"获取容器统计信息失败: {e}")
            return None


# ============================================================
# 3. 实时异常检测器（核心）
# ============================================================

class RealTimeAnomalyDetector:
    """
    实时异常检测引擎
    
    工作流程：
    1. 从 Docker API 获取最新的性能数据
    2. 用保存的 scaler 进行标化
    3. 塞入滑动时间窗口
    4. 窗口满6个样本时，输入LSTM模型推理
    5. 计算重构误差，与阈值比较
    6. 超过阈值时发出告警
    """
    
    def __init__(self, target_container, threshold, 
                 seq_len=6, nas_ip='192.168.3.2', nas_port=2375):
        """
        Args:
            target_container: 要监控的容器名称，如 "qbittorrent"
            threshold: 异常告警阈值，从 output/report.txt 获取
            seq_len: 时间序列长度（与训练时必须一致）
            nas_ip: NAS IP 地址
            nas_port: Docker API 端口
        """
        self.container_name = target_container
        self.threshold = threshold
        self.seq_len = seq_len
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 特征列表（必须与训练时完全一致！）
        self.features = ['cpu_percent', 'memory_mb', 'net_in_rate_mbs', 'net_out_rate_mbs']
        
        # 滑动时间窗口（最多保留 seq_len 个最新的样本）
        self.window = deque(maxlen=seq_len)
        
        # 统计信息
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
        
        # 获取目标容器对象
        try:
            self.container_obj = self.collector.client.containers.get(self.container_name)
            logger.info(f"✓ 成功找到目标容器: {self.container_name}")
        except Exception as e:
            logger.error(f"✗ 无法找到容器 '{self.container_name}': {e}")
            logger.error(f"  可用的容器有: {[c.name for c in self.collector.client.containers.list()]}")
            sys.exit(1)
        
        # 加载训练好的模型和标化器
        self._load_ai_brain()
    
    def _load_ai_brain(self):
        """加载训练好的模型和标化器"""
        
        logger.info("\n" + "="*60)
        logger.info("加载训练好的模型和标化器...")
        logger.info("="*60)
        
        model_path = f'models/{self.container_name}_model.pth'
        scaler_path = f'models/{self.container_name}_scaler.pkl'
        
        # 检查文件是否存在
        if not Path(model_path).exists():
            logger.error(f"✗ 模型文件不存在: {model_path}")
            logger.error(f"  请先运行 'python complete_pipeline_fixed.py' 来训练模型")
            sys.exit(1)
        
        if not Path(scaler_path).exists():
            logger.error(f"✗ 标化器文件不存在: {scaler_path}")
            logger.error(f"  请先运行 'python complete_pipeline_fixed.py' 来训练模型")
            sys.exit(1)
        
        # 加载 Scaler（关键！）
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"✓ 成功加载数据标化器 (Scaler)")
            logger.info(f"  标化器参数: mean={self.scaler.mean_}, scale={self.scaler.scale_}")
        except Exception as e:
            logger.error(f"✗ 加载标化器失败: {e}")
            sys.exit(1)
        
        # 加载 LSTM 模型
        try:
            n_features = len(self.features)
            self.model = LSTMAutoencoder(
                input_size=n_features,
                hidden_size=16,
                latent_size=8
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()  # 切换到评估模式
            logger.info(f"✓ 成功加载 LSTM Autoencoder 模型")
            logger.info(f"  模型参数: input_size={n_features}, hidden_size=16, latent_size=8")
        except Exception as e:
            logger.error(f"✗ 加载模型失败: {e}")
            sys.exit(1)
        
        logger.info(f"\n🎯 检测引擎配置:")
        logger.info(f"  ├─ 监控容器: {self.container_name}")
        logger.info(f"  ├─ 告警阈值: {self.threshold:.6f}")
        logger.info(f"  ├─ 时间窗口: {self.seq_len} 个时间步 (60秒)")
        logger.info(f"  └─ 计算设备: {self.device}")
        logger.info("="*60 + "\n")
    
    def run_monitor(self, interval=10):
        """
        主监控循环
        
        Args:
            interval: 采样间隔（秒）
        """
        logger.info("🚀 启动实时异常检测引擎")
        logger.info("="*60)
        
        try:
            while True:
                try:
                    # 1. 从 Docker API 获取当前容器的性能数据
                    stats = self.collector.get_container_stats(self.container_obj)
                    
                    if stats is None:
                        time.sleep(interval)
                        continue
                    
                    # 2. 提取特征值
                    raw_data = np.array([stats[feat] for feat in self.features])
                    
                    # 3. 用保存的 scaler 进行标化（关键步骤！）
                    scaled_data = self.scaler.transform([raw_data])[0]
                    
                    # 4. 将标化后的数据加入滑动窗口
                    self.window.append(scaled_data)
                    
                    # 5. 当窗口内积累了 seq_len 个样本时，进行推理
                    if len(self.window) == self.seq_len:
                        # 将窗口数据转换为张量
                        tensor_input = torch.FloatTensor(
                            np.array(self.window)
                        ).unsqueeze(0).to(self.device)  # (1, seq_len, n_features)
                        
                        # 模型推理
                        with torch.no_grad():
                            output = self.model(tensor_input)
                            # 计算均方误差（这个时间窗口的重构误差）
                            mse_error = torch.mean(
                                (tensor_input - output) ** 2
                            ).item()
                        
                        # 更新统计信息
                        self.stats['total_samples'] += 1
                        self.stats['max_error'] = max(self.stats['max_error'], mse_error)
                        self.stats['min_error'] = min(self.stats['min_error'], mse_error)
                        
                        # 6. 与阈值比较，判断是否异常
                        if mse_error > self.threshold:
                            # 🚨 异常告警！
                            self.stats['anomaly_count'] += 1
                            logger.warning(f"\n{'='*60}")
                            logger.warning(f"🚨 [异常告警] 容器 '{self.container_name}' 状态异常！")
                            logger.warning(f"{'='*60}")
                            logger.warning(f"  重构误差: {mse_error:.6f} > 阈值: {self.threshold:.6f}")
                            logger.warning(f"  超过阈值: {(mse_error - self.threshold):.6f}")
                            logger.warning(f"\n  当前性能指标:")
                            logger.warning(f"    ├─ CPU: {raw_data[0]:6.1f}%")
                            logger.warning(f"    ├─ 内存: {raw_data[1]:7.1f}MB")
                            logger.warning(f"    ├─ 网络入: {raw_data[2]:5.2f}MB/s")
                            logger.warning(f"    └─ 网络出: {raw_data[3]:5.2f}MB/s")
                            logger.warning(f"{'='*60}\n")
                            
                            # TODO: 这里可以扩展
                            # - 发送企业微信告警
                            # - 发送邮件通知
                            # - 自动重启容器
                            # - 调用其他脚本处理
                            self._handle_alert(raw_data, mse_error)
                        
                        else:
                            # ✓ 正常运行
                            print(f"[{time.strftime('%H:%M:%S')}] ✓ {self.container_name} 运行平稳 | "
                                  f"误差: {mse_error:.6f} | CPU: {raw_data[0]:5.1f}% | "
                                  f"内存: {raw_data[1]:7.1f}MB")
                    
                    else:
                        # 窗口未满，继续收集数据
                        progress = len(self.window)
                        print(f"[{time.strftime('%H:%M:%S')}] 收集初始数据中... ({progress}/{self.seq_len})")
                    
                    # 7. 等待下一个采样周期
                    time.sleep(interval)
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"监控循环发生错误: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(interval)
        
        finally:
            self._print_summary()
    
    def _handle_alert(self, raw_data, mse_error):
        """处理告警（可扩展）"""
        # 这里可以添加：
        # 1. 发送企业微信通知
        # 2. 发送邮件
        # 3. 记录到数据库
        # 4. 调用自动恢复脚本
        pass
    
    def _print_summary(self):
        """打印运行统计摘要"""
        logger.info("\n" + "="*60)
        logger.info("📊 监控统计摘要")
        logger.info("="*60)
        
        duration = (datetime.now() - self.stats['start_time']).total_seconds()
        hours = duration / 3600
        
        logger.info(f"运行时间: {hours:.1f} 小时")
        logger.info(f"采样总数: {self.stats['total_samples']}")
        logger.info(f"异常事件: {self.stats['anomaly_count']}")
        if self.stats['total_samples'] > 0:
            anomaly_rate = self.stats['anomaly_count'] / self.stats['total_samples'] * 100
            logger.info(f"异常率: {anomaly_rate:.2f}%")
        
        if self.stats['min_error'] != float('inf'):
            logger.info(f"误差范围: [{self.stats['min_error']:.6f}, {self.stats['max_error']:.6f}]")
        
        logger.info("="*60 + "\n")


# ============================================================
# 4. 主程序入口
# ============================================================

if __name__ == '__main__':
    
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*8 + "🚀 实时容器异常检测系统 - 监控模式" + " "*16 + "║")
    print("╚" + "="*58 + "╝\n")
    
    # ================= 配置区域 (需要自己修改) =================
    
    # 1. 替换为你要监控的实际容器名称
    TARGET_CONTAINER = "qbittorrent"
    
    # 2. 从 output/report.txt 中找到的阈值
    #    示例：平均误差: 0.0234, 标准差: 0.0156, 阈值 (mean+2σ): 0.5500
    ALERT_THRESHOLD = 0.55
    
    # 3. NAS 地址和端口
    NAS_IP = '192.168.3.2'
    NAS_PORT = 2375
    
    # 4. 采样间隔（秒），建议 10 秒
    SAMPLING_INTERVAL = 10
    
    # =========================================================
    
    logger.info(f"配置信息:")
    logger.info(f"  ├─ 监控容器: {TARGET_CONTAINER}")
    logger.info(f"  ├─ 告警阈值: {ALERT_THRESHOLD}")
    logger.info(f"  ├─ NAS地址: {NAS_IP}:{NAS_PORT}")
    logger.info(f"  └─ 采样间隔: {SAMPLING_INTERVAL}秒\n")
    
    # 创建异常检测器
    detector = RealTimeAnomalyDetector(
        target_container=TARGET_CONTAINER,
        threshold=ALERT_THRESHOLD,
        seq_len=6,
        nas_ip=NAS_IP,
        nas_port=NAS_PORT
    )
    
    # 启动监控
    try:
        detector.run_monitor(interval=SAMPLING_INTERVAL)
    except KeyboardInterrupt:
        logger.info("\n✓ 监控已停止")
    except Exception as e:
        logger.error(f"✗ 发生超过阈值的错误: {e}")
        import traceback
        traceback.print_exc()
