# PD3QN: A Predictive Dueling Double Deep Q-Network
# 基于轻量级未来状态预测与自适应置信度门控的预测型 D3QN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
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
- [环境说明](#-环境说明)
- [引用](#-引用)

## 🧩 背景介绍

传统的 Model-free 算法（如 D3QN）在面对快节奏环境时往往缺乏前瞻性。PD3QN 通过引入以下机制解决了这一问题：
1.  **一步状态预测器 (OSSP):** 一个轻量级模块，用于预测下一帧画面及死亡风险。
2.  **自适应置信度门控 (CGN):** 动态调节对预测结果的信任程度，防止早期模型误差误导策略学习。
3.  **固定动作向量 (Fixed Action Vectors):** 替代传统的可学习 Embedding，在训练初期提供稳定的输入特征。

<p align="center">
  <img src="assets/framework.png" alt="PD3QN Framework" width="800">
  <br>
  <em>图：PD3QN 整体架构图</em>
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
## ⚙️ 环境安装

### 克隆仓库:
