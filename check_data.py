"""
快速检查采集的数据
"""

import pandas as pd
from pathlib import Path

# 找最新的 CSV 文件
data_files = sorted(Path('data').glob('docker_metrics_*.csv'))

if data_files:
    latest_file = data_files[-1]
    print(f"✓ 加载: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    print(f"\n数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    print(f"\n数据预览:")
    print(df.head(10))
    
    print(f"\n容器列表:")
    print(df['container_name'].unique())
        
    print(f"\n统计信息:")
    for container in df['container_name'].unique():
        cdf = df[df['container_name'] == container]
        print(f"\n{container}:")
        print(f"  样本数: {len(cdf)}")
        print(f"  CPU: {cdf['cpu_percent'].mean():.2f}%")
        print(f"  内存: {cdf['memory_mb'].mean():.2f}MB")
else:
    print("❌ 没有找到数据文件")