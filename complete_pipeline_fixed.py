"""
改进的完整异常检测管道 - 支持LSTM + Transformer + 混合模型
==========================================================

数据可以复用！只需要修改模型架构

运行: python enhanced_pipeline_multimodel.py
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
import json
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# ============================================================
# 1. 模型架构定义（3个模型）
# ============================================================

# ===== 模型 1: LSTM Autoencoder (原始版本) =====
class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder - 用于时序异常检测"""
    
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


# ===== 模型 2: Transformer Autoencoder (新增) =====
class TransformerAutoencoder(nn.Module):
    """
    Transformer Autoencoder - 平行注意力机制
    
    相比 LSTM 的优势：
    - 可以并行处理所有时间步（计算快）
    - 自注意力机制可以捕捉长期依赖
    - 在长序列上通常比 LSTM 更好
    """
    
    def __init__(self, input_size=4, hidden_size=32, num_heads=4, num_layers=2, 
                 latent_size=8, dropout=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入线性投影
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Positional Encoding (位置编码)
        self.positional_encoding = nn.Parameter(
            self._get_positional_encoding(seq_len=100, d_model=hidden_size),
            requires_grad=False
        )
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 编码器的密集层
        self.encoder_dense = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size)
        )
        
        # 解码器：从潜在表示重建序列
        self.decoder_dense = nn.Linear(latent_size, hidden_size)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出线性投影
        self.output_projection = nn.Linear(hidden_size, input_size)
    
    def _get_positional_encoding(self, seq_len=100, d_model=32):
        """生成位置编码 (Positional Encoding)"""
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
        
        # 投影输入到 hidden_size 维度
        x_proj = self.input_projection(x)  # (batch, seq_len, hidden_size)
        
        # 添加位置编码
        pe = self.positional_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
        x_proj = x_proj + pe
        
        # 编码
        encoded = self.encoder(x_proj)  # (batch, seq_len, hidden_size)
        
        # 全局平均池化来得到固定大小的表示
        z_input = encoded.mean(dim=1)  # (batch, hidden_size)
        z = self.encoder_dense(z_input)  # (batch, latent_size)
        
        # 解码：从潜在表示重复生成序列长度的向量
        z_expanded = self.decoder_dense(z)  # (batch, hidden_size)
        z_expanded = z_expanded.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, hidden_size)
        
        # 使用 memory 进行解码
        decoded = self.decoder(z_expanded, encoded)  # (batch, seq_len, hidden_size)
        
        # 投影回输出维度
        output = self.output_projection(decoded)  # (batch, seq_len, input_size)
        
        return output


# ===== 模型 3: 混合模型 (LSTM + Attention) =====
class LSTMAttentionAutoencoder(nn.Module):
    """
    LSTM 配合 Attention 机制
    结合 LSTM 的时间感知和 Attention 的长程依赖捕捉
    """
    
    def __init__(self, input_size=4, hidden_size=16, latent_size=8, num_heads=4):
        super().__init__()
        
        # 编码器：LSTM
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # 多头自注意力 (自注意编码器输出)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 编码器密集层
        self.encoder_dense = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size)
        )
        
        # 解码器
        self.decoder_dense = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # 解码后的注意力
        self.decoder_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.output_dense = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        # 编码
        lstm_out, (h, c) = self.encoder_lstm(x)
        
        # 自注意力
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 用注意力输出的最后一步作为潜在编码
        z = self.encoder_dense(h[-1])
        
        # 解码
        seq_len = x.shape[1]
        h_0 = self.decoder_dense(z).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        z_repeated = h_0.expand(seq_len, -1, -1).transpose(0, 1)
        
        lstm_dec, _ = self.decoder_lstm(z_repeated, (h_0, c_0))
        
        # 解码器注意力
        attn_dec, _ = self.decoder_attention(lstm_dec, lstm_dec, lstm_dec)
        
        output = self.output_dense(attn_dec)
        
        return output


# ============================================================
# 2. 数据预处理（与原有流程完全相同）
# ============================================================

def preprocess_data():
    """
    预处理采集的数据
    数据格式与原来完全相同，可以复用！
    """
    
    print("\n" + "="*70)
    print("步骤 1: 数据预处理（新数据或原有数据都可以）")
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
    
    container_col = 'container_name' if 'container_name' in df.columns else 'container'
    
    processed_data = {}
    
    for container in df[container_col].unique():
        print(f"\n  处理容器: {container}")
        
        container_df = df[df[container_col] == container].copy()
        print(f"    样本数: {len(container_df)}")
        
        # 特征提取（与原来相同）
        features = ['cpu_percent', 'memory_mb', 'net_in_rate_mbs', 'net_out_rate_mbs']
        available_features = [f for f in features if f in container_df.columns]
        
        if len(available_features) == 0:
            print(f"    ⚠️ 没有可用的数值特征，跳过")
            continue
        
        X = container_df[available_features].values
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 创建时间窗口
        # 创建时间窗口 (针对 10秒/次的采样频率)
        seq_len = 120  # 120个点 = 往前看 20 分钟的长期记忆
        
        # 针对 20000 条数据，如果显存/内存不够，可以加上步长(step)跳跃滑动
        # 但通常 20000 条数据完全没必要，步长为 1 即可
        step = 1 
        
        sequences = []
        for i in range(0, len(X_scaled) - seq_len + 1, step):
            sequences.append(X_scaled[i:i+seq_len])
        
        sequences = np.array(sequences)
        print(f"    创建了 {len(sequences)} 个窗口 (seq_len={seq_len})")
        
        if len(sequences) < 2:
            print(f"    ⚠️ 数据太少，跳过")
            continue
        
        n_train = max(1, int(len(sequences) * 0.7))
        
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
# 3. 多模型训练
# ============================================================

def train_multimodel(processed_data):
    """
    同时训练多个模型（LSTM、Transformer、混合）
    """
    
    print("\n" + "="*70)
    print("步骤 2: 多模型训练")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ 使用设备: {device}")
    
    Path('models').mkdir(exist_ok=True)
    Path('output').mkdir(exist_ok=True)
    
    # 定义要训练的模型
    model_configs = {
        'lstm': {
            'class': LSTMAutoencoder,
            'params': {'hidden_size': 16, 'latent_size': 8, 'num_layers': 1},
            'description': 'LSTM Autoencoder'
        },
        'transformer': {
            'class': TransformerAutoencoder,
            'params': {'hidden_size': 32, 'num_heads': 4, 'num_layers': 2, 'latent_size': 8},
            'description': 'Transformer Autoencoder'
        },
        'lstm_attention': {
            'class': LSTMAttentionAutoencoder,
            'params': {'hidden_size': 16, 'latent_size': 8, 'num_heads': 4},
            'description': 'LSTM + Attention'
        }
    }
    
    trained_models = {}
    results = {}
    
    for container_name, data in processed_data.items():
        print(f"\n{'='*70}")
        print(f"训练容器: {container_name}")
        print(f"{'='*70}")
        
        n_features = data['n_features']
        trained_models[container_name] = {}
        results[container_name] = {}
        
        for model_name, config in model_configs.items():
            print(f"\n  [{model_name.upper()}] {config['description']}...")
            
            X_train = torch.FloatTensor(data['train'])
            batch_size = min(2, len(X_train))
            loader = DataLoader(X_train, batch_size=batch_size, shuffle=False)
            
            # 创建模型
            model_class = config['class']
            model_params = config['params'].copy()
            model_params['input_size'] = n_features
            
            model = model_class(**model_params)
            model = model.to(device)
            
            # 训练配置
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            
            epochs = 30
            losses = []
            
            # 训练循环
            start_time = time.time()
            for epoch in range(epochs):
                total_loss = 0
                n_batches = 0
                
                for batch in loader:
                    x = batch.to(device)
                    output = model(x)
                    loss = criterion(x, output)
                    
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
            
            train_time = time.time() - start_time
            print(f"    ✓ 训练完成 (耗时: {train_time:.1f}秒)")
            
            # 保存模型
            trained_models[container_name][model_name] = {
                'model': model,
                'losses': losses,
                'train_time': train_time,
                'config': config
            }
            
            model_path = f'models/{container_name}_{model_name}_model.pth'
            torch.save(model.state_dict(), model_path)
            print(f"    ✓ 模型已保存: {model_path}")
            
            # 保存标化器（所有模型共用）
            if model_name == 'lstm':  # 只保存一次
                scaler_path = f'models/{container_name}_scaler.pkl'
                with open(scaler_path, 'wb') as f:
                    pickle.dump(data['scaler'], f)
                print(f"    ✓ 标化器已保存: {scaler_path}")
            
            # 绘制训练曲线
            plt.figure(figsize=(10, 5))
            plt.plot(losses, linewidth=2, color='blue', label=model_name.upper())
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('MSE Loss', fontsize=12)
            plt.title(f'{container_name} - {config["description"]} Training Loss', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'output/{container_name}_{model_name}_training_loss.png', dpi=100)
            plt.close()
    
    return trained_models, processed_data


# ============================================================
# 4. 异常检测 & 模型对比
# ============================================================

def detect_anomalies_multimodel(trained_models, processed_data):
    """
    用多个模型进行异常检测，并对比性能
    """
    
    print("\n" + "="*70)
    print("步骤 3: 多模型异常检测与性能对比")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    all_results = {}
    comparison_data = {}
    
    for container_name, models_dict in trained_models.items():
        print(f"\n{'='*60}")
        print(f"异常检测: {container_name}")
        print(f"{'='*60}")
        
        all_results[container_name] = {}
        comparison_data[container_name] = {}
        
        test_data = processed_data[container_name]['test']
        X_test = torch.FloatTensor(test_data).to(device)
        
        # 用每个模型进行推理
        for model_name, model_info in models_dict.items():
            print(f"\n  [{model_name.upper()}] {model_info['config']['description']}")
            
            model = model_info['model']
            model.eval()
            
            with torch.no_grad():
                output = model(X_test)
                errors = torch.mean((X_test - output) ** 2, dim=(1, 2)).cpu().numpy()
            
            # 计算统计指标
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            threshold = mean_error + 2 * std_error
            
            anomalies = errors > threshold
            anomaly_count = np.sum(anomalies)
            anomaly_rate = anomaly_count / len(errors) if len(errors) > 0 else 0
            
            print(f"    平均误差: {mean_error:.6f}")
            print(f"    标准差: {std_error:.6f}")
            print(f"    阈值 (mean+2σ): {threshold:.6f}")
            print(f"    异常数: {anomaly_count} / {len(errors)} ({anomaly_rate*100:.1f}%)")
            
            all_results[container_name][model_name] = {
                'errors': errors,
                'threshold': threshold,
                'anomalies': anomalies,
                'mean': mean_error,
                'std': std_error,
                'anomaly_rate': anomaly_rate
            }
            
            comparison_data[container_name][model_name] = {
                'mean_error': mean_error,
                'std_error': std_error,
                'threshold': threshold,
                'anomaly_rate': anomaly_rate,
                'train_time': model_info['train_time'],
                'config': model_info['config']
            }
    
    return all_results, comparison_data


# ============================================================
# 5. 可视化对比
# ============================================================

def visualize_comparison(all_results, comparison_data):
    """
    可视化多模型的对比结果
    """
    
    print("\n" + "="*70)
    print("步骤 4: 可视化对比结果")
    print("="*70)
    
    for container_name, models_results in all_results.items():
        print(f"\n  生成对比图表: {container_name}")
        
        # 1. 误差对比 (折线图)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        ax = axes[0, 0]
        for model_name, result in models_results.items():
            errors = result['errors']
            ax.plot(errors, label=model_name.upper(), alpha=0.7, linewidth=2)
            ax.axhline(result['threshold'], linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('MSE Error', fontsize=11)
        ax.set_title(f'{container_name} - Reconstruction Error Comparison', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 误差分布对比 (直方图)
        ax = axes[0, 1]
        for model_name, result in models_results.items():
            errors = result['errors']
            ax.hist(errors, bins=15, alpha=0.5, label=model_name.upper())
        
        ax.set_xlabel('Reconstruction Error', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Error Distribution', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 性能指标对比 (柱状图)
        ax = axes[1, 0]
        model_names = list(models_results.keys())
        means = [models_results[m]['mean'] for m in model_names]
        stds = [models_results[m]['std'] for m in model_names]
        
        x = np.arange(len(model_names))
        ax.bar(x - 0.2, means, 0.4, label='Mean', alpha=0.7)
        ax.bar(x + 0.2, stds, 0.4, label='Std', alpha=0.7)
        
        ax.set_ylabel('Error Value', fontsize=11)
        ax.set_title('Mean & Std Comparison', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in model_names])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 异常率对比 (饼图)
        ax = axes[1, 1]
        anomaly_rates = [models_results[m]['anomaly_rate'] * 100 for m in model_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        ax.bar(model_names, anomaly_rates, color=colors, alpha=0.7)
        ax.set_ylabel('Anomaly Rate (%)', fontsize=11)
        ax.set_title('Anomaly Rate Comparison', fontsize=12)
        ax.set_xticklabels([m.upper() for m in model_names])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'output/{container_name}_multimodel_comparison.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ 对比图已保存: output/{container_name}_multimodel_comparison.png")


# ============================================================
# 6. 生成详细报告
# ============================================================

def generate_detailed_report(comparison_data, trained_models):
    """
    生成多模型对比报告，推荐最佳模型
    """
    
    print("\n" + "="*70)
    print("步骤 5: 生成详细对比报告")
    print("="*70)
    
    report = []
    report.append(f"\n{'='*70}")
    report.append("多模型异常检测对比报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"{'='*70}\n")
    
    best_models = {}
    
    for container_name, models_data in comparison_data.items():
        report.append(f"\n{'─'*70}")
        report.append(f"容器: {container_name}")
        report.append(f"{'─'*70}\n")
        
        # 详细性能数据
        report.append("📊 模型性能详细对比:\n")
        report.append(f"{'模型':<20} {'平均误差':<15} {'标准差':<15} {'阈值':<15} {'异常率':<12} {'训练时间':<12}")
        report.append("─" * 90)
        
        best_model = None
        best_score = float('inf')
        
        for model_name, data in models_data.items():
            # 性能评分：综合考虑误差和异常率
            # 低误差和低异常率都是好的
            score = data['mean_error'] + data['anomaly_rate'] * 0.1
            
            report.append(
                f"{model_name.upper():<20} "
                f"{data['mean_error']:<15.6f} "
                f"{data['std_error']:<15.6f} "
                f"{data['threshold']:<15.6f} "
                f"{data['anomaly_rate']*100:<11.2f}% "
                f"{data['train_time']:<11.2f}s"
            )
            
            if score < best_score:
                best_score = score
                best_model = model_name
        
        report.append("")
        report.append(f"🏆 推荐模型: {best_model.upper()}")
        best_models[container_name] = {
            'model': best_model,
            'score': best_score,
            'data': models_data[best_model]
        }
        
        report.append(f"   - 理由: 平均误差最低或异常率最适中")
        report.append(f"   - 建议阈值: {models_data[best_model]['threshold']:.6f}")
        report.append("")
    
    # 模型特点说明
    report.append(f"\n{'='*70}")
    report.append("模型特点说明")
    report.append(f"{'='*70}\n")
    
    report.append("LSTM Autoencoder:")
    report.append("  - 优点: 轻量级、训练快、参数少")
    report.append("  - 缺点: 难以捕捉长期依赖")
    report.append("  - 适用: 短期模式识别、计算资源有限\n")
    
    report.append("Transformer Autoencoder:")
    report.append("  - 优点: 可并行处理、长期依赖捕捉能力强")
    report.append("  - 缺点: 参数多、训练较慢、需要更多数据")
    report.append("  - 适用: 长序列模式、有充足计算资源\n")
    
    report.append("LSTM + Attention:")
    report.append("  - 优点: 结合两者优势、性能均衡")
    report.append("  - 缺点: 参数介于两者之间")
    report.append("  - 适用: 平衡性能和效率的通用方案\n")
    
    # 使用建议
    report.append(f"\n{'='*70}")
    report.append("实时监控建议")
    report.append(f"{'='*70}\n")
    
    for container_name, best_info in best_models.items():
        report.append(f"{container_name}:")
        report.append(f"  - 推荐模型: {best_info['model']}")
        report.append(f"  - 模型文件: models/{container_name}_{best_info['model']}_model.pth")
        report.append(f"  - 标化器文件: models/{container_name}_scaler.pkl")
        report.append(f"  - 告警阈值: {best_info['data']['threshold']:.6f}")
        report.append(f"  - 预期异常率: {best_info['data']['anomaly_rate']*100:.2f}%")
        report.append("")
    
    report.append(f"\n{'='*70}")
    report.append("数据复用说明")
    report.append(f"{'='*70}\n")
    report.append("✅ 原有数据完全可以复用!")
    report.append("\n原因:")
    report.append("  1. 数据预处理完全相同（特征、标准化、窗口化）")
    report.append("  2. 不同模型使用相同的训练/测试集分割")
    report.append("  3. 只改变模型架构，数据流程不变")
    report.append("  4. 可以用同一份数据对比多个模型性能")
    report.append("\n数据流程:")
    report.append("  CSV 数据 → StandardScaler → 时间窗口 → [LSTM / Transformer / 混合]")
    report.append("                                          ↑")
    report.append("                           (三个模型共享同一份预处理数据)")
    
    report_text = '\n'.join(report)
    print(report_text)
    
    # 保存报告
    with open('output/multimodel_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n✓ 详细报告已保存: output/multimodel_report.txt")
    
    # 保存最佳模型信息到 JSON
    best_models_json = {}
    for container, info in best_models.items():
        best_models_json[container] = {
            'best_model': info['model'],
            'threshold': float(info['data']['threshold']),
            'mean_error': float(info['data']['mean_error']),
            'std_error': float(info['data']['std_error']),
            'anomaly_rate': float(info['data']['anomaly_rate'])
        }
    
    with open('output/best_models.json', 'w') as f:
        json.dump(best_models_json, f, indent=2)
    
    print("✓ 最佳模型信息已保存: output/best_models.json")
    
    return best_models


# ============================================================
# 主程序
# ============================================================

def main():
    """主程序"""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "🚀 多模型异常检测管道 (LSTM + Transformer)" + " "*17 + "║")
    print("╚" + "="*68 + "╝")
    print("\n✨ 特点：")
    print("  ✓ 原有数据完全可以复用")
    print("  ✓ 同时训练 3 个模型（LSTM / Transformer / 混合）")
    print("  ✓ 自动生成性能对比报告")
    print("  ✓ 推荐最佳模型用于实时监控\n")
    
    try:
        # 1. 数据预处理
        processed_data = preprocess_data()
        if processed_data is None:
            print("\n❌ 数据预处理失败")
            return
        
        # 2. 训练多个模型
        trained_models, processed_data = train_multimodel(processed_data)
        
        if not trained_models:
            print("\n❌ 模型训练失败")
            return
        
        # 3. 异常检测
        all_results, comparison_data = detect_anomalies_multimodel(trained_models, processed_data)
        
        if not all_results:
            print("\n❌ 异常检测失败")
            return
        
        # 4. 可视化对比
        visualize_comparison(all_results, comparison_data)
        
        # 5. 生成报告
        best_models = generate_detailed_report(comparison_data, trained_models)
        
        print("\n" + "="*70)
        print("✅ 多模型训练完成！")
        print("="*70)
        
        print("\n📁 生成的文件:")
        print("  模型文件:")
        print("    - models/*_lstm_model.pth          (LSTM 模型)")
        print("    - models/*_transformer_model.pth   (Transformer 模型)")
        print("    - models/*_lstm_attention_model.pth (混合模型)")
        print("    - models/*_scaler.pkl              (标化器，所有模型共用)")
        print("\n  结果文件:")
        print("    - output/multimodel_report.txt     (详细对比报告)")
        print("    - output/best_models.json          (最佳模型配置)")
        print("    - output/*_training_loss.png       (3 个训练曲线)")
        print("    - output/*_multimodel_comparison.png (4 个对比图表)")
        
        print("\n📊 下一步:")
        print("  1. 查看 output/multimodel_report.txt 了解模型对比结果")
        print("  2. 查看 output/best_models.json 获取最佳模型和阈值")
        print("  3. 在 realtime_monitor_multimodel.py 中选择最佳模型")
        print("  4. 运行 python realtime_monitor_multimodel.py 启动监控")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()