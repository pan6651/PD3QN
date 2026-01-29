# PD3QN: A Predictive Dueling Double Deep Q-Network
# 基于轻量级未来状态预测与自适应置信度门控的预测型 D3QN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

这是论文 **"PD3QN: A Predictive Dueling Double Deep Q-Network with Lightweight Future State Prediction and Adaptive Confidence Gating"** 的官方代码实现。

PD3QN 是一种新颖的深度强化学习（DRL）框架，旨在解决高动态、低容错环境（如 **带火球的 Flappy Bird**）中的**决策短视（Decision Myopia）**问题。通过将轻量级环境模型与 Model-free 骨干网络结合，实现了鲁棒的长程规划能力。

## 📖 目录
- [背景介绍](#-背景介绍)
- [核心特性](#-核心特性)
- [目录结构](#-目录结构)
- [环境安装](#-环境安装)
- [使用指南](#-使用指南)
  - [训练模型](#训练模型)
  - [测试模型](#测试模型)
  - [监控训练](#监控训练)
- [环境说明](#-环境说明)
- [致谢](#-致谢)

## 🧩 背景介绍

传统的 Model-free 算法（如 D3QN）在面对快节奏环境时往往缺乏前瞻性。PD3QN 通过引入以下机制解决了这一问题：

1.  **一步状态预测器 (OSSP):** 一个轻量级模块，用于预测下一帧画面及死亡风险。
2.  **自适应置信度门控 (CGN):** 动态调节对预测结果的信任程度，防止早期模型误差误导策略学习。
3.  **固定动作向量 (Fixed Action Vectors):** 替代传统的可学习 Embedding，在训练初期提供稳定的输入特征。

<p align="center">
  <img src="assets/framework.png" alt="PD3QN Framework" width="800">
  <br>
  <em>图：PD3QN 整体架构图（请确保 assets 目录下有 framework.png）</em>
</p>

## ✨ 核心特性

* **混合损失函数 (Hybrid Loss):** 结合 MSE、边缘检测 (Sobel) 和 SSIM，强制预测器关注物理结构而非背景噪声。
* **分层经验回放 (Stratified Experience Replay):** 平衡游戏早期（简单场景）和晚期（复杂场景）的样本分布，防止遗忘。
* **密集引导奖励 (Dense Guided Rewards):** 利用基于距离的奖励塑形，显著加速训练收敛。
* **高难度环境:** 在原版 Flappy Bird 基础上增加了**动态火球**障碍，考验智能体的动态避障能力。

## 📂 目录结构

基于本仓库的代码组织：

```text
PD3QN/
├── assets/                          # 游戏资源 (图片 sprites, 字体 fonts)
├── DDQN_Innovation_Research/
│   └── improvements/
│       └── experiments/
│           └── train_PD3QN.py       # 🚀 核心训练脚本
├── logs/                            # TensorBoard 日志文件
├── results/                         # 模型保存 (.pth) 和测试报告 (.json)
├── src/                             # 源代码库
│   ├── flappy_bird.py               # 游戏环境 (训练版)
│   ├── flappy_bird-test.py          # 游戏环境 (测试版 - 无头模式适配)
│   └── requirements.txt             # 项目依赖列表
├── test_PD3QN.py                    # 🧪 批量测试脚本
└── README.md                        # 项目说明文档
```

## ⚙️ 环境安装

1. **克隆仓库:**

   ```bash
   git clone [https://github.com/your-username/PD3QN.git](https://github.com/your-username/PD3QN.git)
   cd PD3QN
   ```

2. **创建虚拟环境 (推荐):**

   ```bash
   conda create -n pd3qn python=3.10
   conda activate pd3qn
   ```

3. **安装依赖:**
   注意：`requirements.txt` 位于 `src` 目录下。

   ```bash
   pip install -r src/requirements.txt
   ```
   *主要依赖包括: `torch`, `pygame`, `opencv-python`, `tensorboardX`, `numpy` 等。*

## 🚀 使用指南

### 训练模型
从头开始训练 PD3QN 模型。脚本使用 `StratifiedExperienceBuffer` 和 `FixedActionVectors`。

由于脚本路径较深，请在项目根目录下运行：

```bash
python DDQN_Innovation_Research/improvements/experiments/train_PD3QN.py
```

* **配置:** 您可以在 `train_PD3QN.py` 中直接修改超参数（Batch size, LR, Gamma 等）。
* **日志:** 训练日志将保存至 `logs/PD3QN_FixedVectors/`。
* **模型:** 检查点（Checkpoints）将保存至 `results/PD3QN_FixedVectors/`。

### 测试模型
用于评估已训练的模型。该脚本会执行批量测试，计算修剪后的平均分（去除最高/最低分），并生成详细报告。

```bash
python test_PD3QN.py
```

* 脚本会自动搜索 `results/PD3QN_FixedVectors/` 目录下的模型文件。
* 测试摘要和详细 JSON 报告将生成在 `results/test_reports/` 中。

### 监控训练
使用 TensorBoard 实时查看 Loss 和 Q值曲线：

```bash
tensorboard --logdir=logs/PD3QN_FixedVectors
```

## 🎮 环境说明: Flappy Bird with Fireballs

本环境是基于 `pygame` 修改的高难度版本：

* **状态 (State):** 连续 4 帧堆叠的灰度图像 (84x84)。
* **动作 (Actions):** 0 (什么都不做), 1 (跳跃/Flap)。
* **奖励 (Rewards):**
    * 存活每帧: +0.1
    * 通过管道: +1.0
    * 躲避火球: +0.5
    * 碰撞 (Terminal): -1.0
    * *密集引导:* 基于与管道中心距离的连续奖励/惩罚。



## 🤝 致谢 (Acknowledgments)

* **代码致谢 (Code Reference):**
    本项目的游戏环境代码参考并修改自 [SeVEnMY/ReinforcementLearningFlappyBird](https://github.com/SeVEnMY/ReinforcementLearningFlappyBird)。
   

* **基金支持 (Funding):**
    本研究工作得到了以下项目的资助：
    * **National Natural Science Foundation of China** (Grant No. 62471272, 61806107, 62201314)
    * **Opening Project of State Key Laboratory of Digital Publishing Technology**
    * **NSF of Shandong Province** (Grant No. ZR2025MS986)


