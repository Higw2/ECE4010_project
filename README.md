# 🚀 Docker-AIOps: 基于 LSTM-Transformer 混合架构的容器实时异常检测

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker API](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

## 项目简介

这是一个专门针对轻量级服务器和家庭 NAS（如绿联 NAS 等）环境设计的 **无侵入式 Docker 容器实时异常检测系统**。

传统的静态阈值告警往往过于死板，且在容器内安装 Agent 会带来额外的性能开销。本项目通过直接调用 Docker TCP API (Port 2375) 获取底层物理资源积分数据，并使用原生的 **LSTM-Transformer 混合自编码器 (Hybrid Autoencoder)** 捕捉容器运行的健康模式。当发生未知异常（如挖矿木马、内存泄漏、网络 CC 攻击）时，系统利用动态 3σ 均方误差 (MSE) 阈值实现毫秒级精准拦截与告警。

---

## 核心特性

- **纯无侵入式监控**：无需在容器（如 qBittorrent、Jellyfin 等）内部植入任何探针，依赖外部 API 轮询，开销极低。
- **抛弃传统位置编码**：创新性地使用 **LSTM** 作为前端特征提取器，利用其循环特性的天然时序感知能力，完美取代了 Transformer 原始生硬的数学位置编码 (Positional Encoding)。
- **上帝视角长程建模**：前端提取的特征送入 **Transformer Encoder**，打破 RNN 距离限制，利用自注意力机制 (Self-Attention) 建立长达 20 分钟 (120个时间步) 的全局状态关联。
- **动态 3σ 异常判定**：基于无监督学习，利用 Latent Space (潜在空间) 压缩与逆向重构计算 MSE 误差，自动生成符合当前容器特性的动态安全阈值。

---

## 🏗️ 系统架构

整个检测管道分为四个阶段：

1. **数据采集 (Data Collection)**：每 10 秒轮询采集 CPU 积分百分比、内存瞬时消耗、网络入站/出站 I/O 速率。
2. **预处理 (Preprocessing)**：使用 `StandardScaler` 消除量纲差异，应用 `seq_len=120` 的滑动窗口截取时序切片。
3. **混合模型 (Hybrid Autoencoder)**：
   `Input -> LSTM Encoder -> Transformer Encoder -> Latent Space Bottleneck -> Transformer Decoder -> LSTM Decoder -> Output`
4. **实时推断 (Inference)**：计算真实输入与模型重构输出的 MSE，一旦突破阈值即刻告警。

---

## 🛠️ 快速开始

### 1. 环境准备

建议使用 `conda` 或 `venv` 配置 Python 环境：

```bash
pip install torch pandas numpy scikit-learn docker matplotlib
