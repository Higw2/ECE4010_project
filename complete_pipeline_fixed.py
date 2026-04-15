"""
完整的 NAS 异常检测管道 (修复版)
修复了列名不匹配的问题
数据预处理 → 模型训练 → 异常检测 → 可视化
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LSTM Autoencoder 模型定义
# ============================================================

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, latent_size=16):
        super().__init__()
        
        # 编码器
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.encoder_dense = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size)
        )
        
        # 解码器
        self.decoder_dense = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_dense = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        # 编码
        _, (h, _) = self.encoder_lstm(x)
        z = self.encoder_dense(h[-1])
        
        # 解码
        seq_len = x.shape[1]
        h_0 = self.decoder_dense(z).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        z_repeated = h_0.expand(seq_len, -1, -1).transpose(0, 1)
        
        output, _ = self.decoder_lstm(z_repeated, (h_0, c_0))
        output = self.output_dense(output)
        
        return output

# ============================================================
# 2. 数据预处理
# ============================================================

def preprocess_data():
    """预处理采集的数据"""
    
    print("\n" + "="*70)
    print("步骤 1: 数据预处理")
    print("="*70)
    
    # 加载数据
    data_files = sorted(Path('data').glob('docker_metrics_*.csv'))
    if not data_files:
        print("❌ 没有找到数据文件")
        return None
    
    latest_file = data_files[-1]
    print(f"✓ 加载数据: {latest_file.name}")
    
    df = pd.read_csv(latest_file)
    print(f"✓ 数据形状: {df.shape}")
    print(f"✓ 列名: {df.columns.tolist()}")
    
    # 按容器处理
    # 注意：列名是 container_name，不是 container
    container_col = 'container_name' if 'container_name' in df.columns else 'container'
    
    processed_data = {}
    
    for container in df[container_col].unique():
        print(f"\n  处理容器: {container}")
        
        container_df = df[df[container_col] == container].copy()
        print(f"    样本数: {len(container_df)}")
        
        # 提取特征
        # 选择数值特征（跳过 timestamp, container_name, image）
        features = ['cpu_percent', 'memory_mb', 'net_in_rate_mbs', 'net_out_rate_mbs']
        
        # 检查哪些特征存在
        available_features = [f for f in features if f in container_df.columns]
        print(f"    可用特征: {available_features}")
        
        if len(available_features) == 0:
            print(f"    ⚠️ 没有可用的数值特征，跳过")
            continue
        
        X = container_df[available_features].values
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 创建时间窗口
        seq_len = 6  # 6 个时间步（因为只有 11 个样本）
        sequences = []
        for i in range(len(X_scaled) - seq_len + 1):
            sequences.append(X_scaled[i:i+seq_len])
        
        sequences = np.array(sequences)
        print(f"    创建了 {len(sequences)} 个窗口 (seq_len={seq_len})")
        
        if len(sequences) < 2:
            print(f"    ⚠️ 数据太少，跳过")
            continue
        
        # 分割训练/测试集
        n_train = max(1, int(len(sequences) * 0.7))  # 70% 训练
        
        processed_data[container] = {
            'train': sequences[:n_train],
            'test': sequences[n_train:],
            'scaler': scaler,
            'all_sequences': sequences,
            'features': available_features,
            'n_features': len(available_features)
        }
        
        print(f"    训练集: {len(sequences[:n_train])}, 测试集: {len(sequences[n_train:])}")
    
    if not processed_data:
        print("\n❌ 没有可处理的容器")
        return None
    
    return processed_data

# ============================================================
# 3. 训练模型
# ============================================================

def train_models(processed_data):
    """训练模型"""
    
    print("\n" + "="*70)
    print("步骤 2: 训练 LSTM Autoencoder 模型")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ 使用设备: {device}")
    
    # 创建目录
    Path('models').mkdir(exist_ok=True)
    Path('output').mkdir(exist_ok=True)
    
    trained_models = {}
    
    for container_name, data in processed_data.items():
        print(f"\n  训练 {container_name}...")
        
        # 数据加载
        X_train = torch.FloatTensor(data['train'])
        
        # 如果训练集太小，不需要 batch
        batch_size = min(2, len(X_train))
        loader = DataLoader(X_train, batch_size=batch_size, shuffle=False)
        
        # 模型创建 - 根据实际特征数调整 input_size
        n_features = data['n_features']
        model = LSTMAutoencoder(input_size=n_features, hidden_size=16, latent_size=8)
        model = model.to(device)
        
        # 训练配置
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # 训练循环
        epochs = 30
        losses = []
        
        print(f"    数据形状: {X_train.shape}")
        print(f"    特征数: {n_features}")
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            for batch in loader:
                x = batch.to(device)
                
                # 前向传播
                output = model(x)
                loss = criterion(x, output)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:2d}/{epochs} - Loss: {avg_loss:.6f}")
        
        # 保存模型
        trained_models[container_name] = {
            'model': model,
            'losses': losses,
            'scaler': data['scaler'],
            'data': data
        }
        
        torch.save(model.state_dict(), f'models/{container_name}_model.pth')
        print(f"    ✓ 模型已保存")
        with open(f'models/{container_name}_scaler.pkl', 'wb') as f:
            pickle.dump(data['scaler'], f)
        
        # 绘制训练曲线
        plt.figure(figsize=(10, 5))
        plt.plot(losses, linewidth=2, color='blue')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title(f'{container_name} - Training Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'output/{container_name}_training_loss.png', dpi=100)
        plt.close()
        print(f"    ✓ 训练曲线已保存")
    
    return trained_models

# ============================================================
# 4. 异常检测
# ============================================================

def detect_anomalies(trained_models):
    """异常检测"""
    
    print("\n" + "="*70)
    print("步骤 3: 异常检测")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = {}
    
    for container_name, model_info in trained_models.items():
        print(f"\n  检测 {container_name}...")
        
        model = model_info['model']
        model.eval()
        
        # 测试集数据
        test_data = model_info['data']['test']
        
        if len(test_data) == 0:
            print(f"    ⚠️ 没有测试数据")
            continue
        
        X_test = torch.FloatTensor(test_data).to(device)
        
        # 计算重构误差
        with torch.no_grad():
            output = model(X_test)
            # 计算每个样本的平均误差
            errors = torch.mean((X_test - output) ** 2, dim=(1, 2)).cpu().numpy()
        
        # 计算统计指标
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        threshold = mean_error + 2 * std_error  # 用 2σ 而不是 3σ（因为数据少）
        
        # 检测异常
        anomalies = errors > threshold
        anomaly_count = np.sum(anomalies)
        anomaly_rate = anomaly_count / len(errors) if len(errors) > 0 else 0
        
        print(f"    样本数: {len(errors)}")
        print(f"    平均误差: {mean_error:.6f}")
        print(f"    标准差: {std_error:.6f}")
        print(f"    阈值 (mean+2σ): {threshold:.6f}")
        print(f"    异常数: {anomaly_count} / {len(errors)} ({anomaly_rate*100:.1f}%)")
        
        results[container_name] = {
            'errors': errors,
            'threshold': threshold,
            'anomalies': anomalies,
            'mean': mean_error,
            'std': std_error
        }
    
    return results

# ============================================================
# 5. 可视化
# ============================================================

def visualize_results(results):
    """可视化异常检测结果"""
    
    print("\n" + "="*70)
    print("步骤 4: 可视化结果")
    print("="*70)
    
    for container_name, result in results.items():
        print(f"\n  绘制 {container_name}...")
        
        errors = result['errors']
        threshold = result['threshold']
        anomalies = result['anomalies']
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 上图：重构误差时间序列
        ax1 = axes[0]
        x = range(len(errors))
        
        # 绘制误差曲线
        ax1.plot(x, errors, 'b-', alpha=0.7, linewidth=2, marker='o', markersize=6, label='Reconstruction Error')
        
        # 绘制阈值线
        ax1.axhline(threshold, color='r', linestyle='--', linewidth=2.5, label='Threshold (mean+2σ)')
        ax1.axhline(result['mean'], color='g', linestyle=':', linewidth=1.5, alpha=0.7, label='Mean Error')
        
        # 标记异常点
        anomaly_indices = np.where(anomalies)[0]
        if len(anomaly_indices) > 0:
            ax1.scatter(anomaly_indices, errors[anomaly_indices], 
                       color='red', s=200, marker='x', linewidths=3, 
                       label=f'Anomalies (n={len(anomaly_indices)})', zorder=5)
        
        ax1.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MSE (Reconstruction Error)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{container_name} - Reconstruction Error Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # 下图：错误分布直方图
        ax2 = axes[1]
        
        # 绘制所有错误的直方图
        ax2.hist(errors, bins=max(3, len(errors)//2), alpha=0.6, color='blue', 
                edgecolor='black', label='Error Distribution', density=False)
        
        # 标记阈值
        ax2.axvline(threshold, color='r', linestyle='--', linewidth=2.5, label='Threshold')
        ax2.axvline(result['mean'], color='g', linestyle=':', linewidth=1.5, alpha=0.7, label='Mean')
        
        ax2.set_xlabel('Reconstruction Error', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'output/{container_name}_anomaly_detection.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ 图表已保存: output/{container_name}_anomaly_detection.png")

# ============================================================
# 6. 生成摘要报告
# ============================================================

def generate_report(processed_data, trained_models, results):
    """生成摘要报告"""
    
    print("\n" + "="*70)
    print("项目完成摘要")
    print("="*70)
    
    report = []
    report.append(f"\n{'='*70}")
    report.append("NAS 异常检测项目 - 完成报告")
    report.append(f"{'='*70}\n")
    
    report.append(f"📊 数据统计:")
    for container_name, data in processed_data.items():
        report.append(f"  {container_name}:")
        report.append(f"    样本数: {len(data['all_sequences'])}")
        report.append(f"    训练集: {len(data['train'])}")
        report.append(f"    测试集: {len(data['test'])}")
        report.append(f"    特征数: {data['n_features']}")
    
    report.append(f"\n🧠 模型性能:")
    for container_name, result in results.items():
        report.append(f"  {container_name}:")
        report.append(f"    平均重构误差: {result['mean']:.8f}")
        report.append(f"    标准差: {result['std']:.8f}")
        report.append(f"    阈值 (mean+2σ): {result['threshold']:.8f}")
        anomaly_rate = np.sum(result['anomalies'])/len(result['errors'])*100
        report.append(f"    异常率: {anomaly_rate:.2f}%")
    
    report.append(f"\n{'='*70}")
    report.append("✓ 输出文件位置:")
    report.append("  models/                    - 训练的模型文件")
    report.append("  output/                    - 可视化结果和报告")
    report.append(f"{'='*70}\n")
    
    # 打印报告
    report_text = '\n'.join(report)
    print(report_text)
    
    # 保存报告
    with open('output/report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("✓ 报告已保存: output/report.txt")

# ============================================================
# 主程序
# ============================================================

def main():
    """主程序"""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "🚀 NAS 异常检测 - 完整管道 (修复版)" + " "*18 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        # 1. 数据预处理
        processed_data = preprocess_data()
        if processed_data is None:
            print("\n❌ 数据预处理失败")
            return
        
        # 2. 训练模型
        trained_models = train_models(processed_data)
        
        if not trained_models:
            print("\n❌ 模型训练失败")
            return
        
        # 3. 异常检测
        results = detect_anomalies(trained_models)
        
        if not results:
            print("\n❌ 异常检测失败")
            return
        
        # 4. 可视化
        visualize_results(results)
        
        # 5. 生成报告
        generate_report(processed_data, trained_models, results)
        
        print("\n✅ 项目运行完成！")
        print("\n📁 生成的文件:")
        print("  • models/                      - 训练的 LSTM 模型")
        print("  • output/                      - 可视化图表和报告")
        print("    - *_training_loss.png        - 训练曲线")
        print("    - *_anomaly_detection.png    - 异常检测结果")
        print("    - report.txt                 - 项目摘要")
        
        print("\n📊 下一步:")
        print("  1. 打开 output/ 目录查看 PNG 图表")
        print("  2. 读取 report.txt 了解详细结果")
        print("  3. 收集更多数据（建议采集至少 3-7 天）")
        print("  4. 准备演讲演示")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()