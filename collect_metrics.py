"""
Docker 容器指标采集脚本 - 最终版本
支持绿联 DH4300 Plus NAS
"""

import docker
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import json
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('collection.log')
    ]
)
logger = logging.getLogger(__name__)

class DockerMetricsCollector:
    def __init__(self, nas_ip='192.168.3.2', nas_port=2375,
                 collection_interval=10, output_dir='./data'):
        """
        初始化采集器
        
        Args:
            nas_ip: NAS IP 地址
            nas_port: Docker API 端口
            collection_interval: 采集间隔（秒）
            output_dir: 数据保存目录
        """
        self.nas_ip = nas_ip
        self.nas_port = nas_port
        self.collection_interval = collection_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 连接到 NAS Docker
        self.connect()
        
        # 数据存储
        self.metrics_buffer = []
        self.last_cpu_stats = {}  # 用于计算 CPU 增量
        self.last_network_stats = {}  # 用于计算网络增量
    
    def connect(self):
        """连接到 NAS Docker"""
        try:
            self.client = docker.DockerClient(
                base_url=f'tcp://{self.nas_ip}:{self.nas_port}',
                timeout=10
            )
            self.client.ping()
            logger.info(f"✓ 成功连接到 NAS Docker: {self.nas_ip}:{self.nas_port}")
            
            # 获取版本信息
            version_info = self.client.version()
            logger.info(f"  Docker 版本: {version_info['Version']}")
            
        except Exception as e:
            logger.error(f"✗ 无法连接到 NAS Docker: {e}")
            sys.exit(1)
    
    def get_container_stats(self, container):
        """
        获取容器的详细统计信息
        
        Args:
            container: Docker 容器对象
            
        Returns:
            字典，包含 CPU、内存、网络等指标
        """
        try:
            stats = container.stats(stream=False)
            container_id = container.id[:12]
            
            # ===== CPU 使用率计算 =====
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
            memory_limit = stats['memory_stats'].get('limit', 1)
            memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0
            memory_mb = memory_usage / (1024 * 1024)
            
            # ===== 网络 I/O =====
            networks = stats.get('networks', {})
            net_in_bytes = sum(net.get('rx_bytes', 0) for net in networks.values())
            net_out_bytes = sum(net.get('tx_bytes', 0) for net in networks.values())
            
            # 计算网络速率（MB/s）
            net_in_mb = net_in_bytes / (1024 * 1024)
            net_out_mb = net_out_bytes / (1024 * 1024)
            
            # 如果是第一次，存储初始值用于计算速率
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
            
            # ===== 块设备 I/O =====
            blkio_stats = stats.get('blkio_stats', {})
            io_read_bytes = sum(
                item.get('value', 0) for item in 
                blkio_stats.get('io_service_bytes_recursive', [])
                if item.get('op') == 'Read'
            )
            io_write_bytes = sum(
                item.get('value', 0) for item in 
                blkio_stats.get('io_service_bytes_recursive', [])
                if item.get('op') == 'Write'
            )
            
            io_read_mb = io_read_bytes / (1024 * 1024)
            io_write_mb = io_write_bytes / (1024 * 1024)
            
            return {
                'cpu_percent': max(0, cpu_percent),
                'memory_mb': memory_mb,
                'memory_percent': max(0, memory_percent),
                'net_in_rate_mbs': max(0, net_in_rate),  # MB/s
                'net_out_rate_mbs': max(0, net_out_rate),
                'io_read_mb': io_read_mb,
                'io_write_mb': io_write_mb,
                'pids': stats.get('pids_stats', {}).get('current', 0)
            }
        
        except Exception as e:
            logger.warning(f"获取 {container.name} 的统计信息失败: {e}")
            return None
    
    def collect_once(self):
        """采集一次所有容器的指标"""
        timestamp = datetime.now().isoformat()
        
        try:
            containers = self.client.containers.list()
            
            if not containers:
                logger.warning("⚠ 未找到运行中的容器")
                return
            
            for container in containers:
                stats = self.get_container_stats(container)
                if stats is None:
                    continue
                
                record = {
                    'timestamp': timestamp,
                    'container_name': container.name,
                    'image': container.image.tags[0] if container.image.tags else 'unknown',
                    **stats
                }
                
                self.metrics_buffer.append(record)
            
            logger.info(f"✓ 采集了 {len(containers)} 个容器")
            
        except Exception as e:
            logger.error(f"采集过程出错: {e}")
    
    def run_continuous(self, duration_hours=3):
        """
        持续采集数据
        
        Args:
            duration_hours: 采集持续时间（小时）
        """
        total_seconds = duration_hours * 3600
        start_time = time.time()
        collection_count = 0
        
        logger.info(f"开始采集，目标持续时间: {duration_hours} 小时")
        logger.info(f"采集间隔: {self.collection_interval} 秒")
        logger.info(f"估计样本数: {int(total_seconds / self.collection_interval)}")
        
        try:
            while (time.time() - start_time) < total_seconds:
                self.collect_once()
                collection_count += 1
                
                # 定期保存数据（每 100 次采集）
                if collection_count % 100 == 0:
                    self.save_data(partial=True)
                
                time.sleep(self.collection_interval)
        
        except KeyboardInterrupt:
            logger.info("\n✓ 采集已停止（用户中断）")
        
        except Exception as e:
            logger.error(f"采集过程出现错误: {e}")
        
        finally:
            self.save_data()
    
    def save_data(self, partial=False):
        """保存采集的数据"""
        if not self.metrics_buffer:
            logger.warning("⚠ 没有数据可保存")
            return
        
        df = pd.DataFrame(self.metrics_buffer)
        
        # 文件名添加时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if partial:
            filename = f"docker_metrics_partial_{timestamp}.csv"
        else:
            filename = f"docker_metrics_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        
        logger.info(f"✓ 数据已保存: {filepath}")
        logger.info(f"  记录数: {len(df)}")
        logger.info(f"  容器数: {df['container_name'].nunique()}")
        logger.info(f"  时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        
        return filepath

# ============ 使用示例 ============
if __name__ == '__main__':
    # 创建采集器
    collector = DockerMetricsCollector(
        nas_ip='192.168.3.2',
        nas_port=2375,
        collection_interval=10,  # 每 10 秒采集一次
        output_dir='./data'
    )
    
    # 快速测试：采集 2 分钟
    #logger.info("\n开始快速测试 (2 分钟)...")
    #collector.run_continuous(duration_hours=2/60)  # 2 分钟
    
    # 采集 3 天的数据用于训练模型
    collector.run_continuous(duration_hours=72)