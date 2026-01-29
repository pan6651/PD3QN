# PD3QN: A Predictive Dueling Double Deep Q-Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

本仓库包含论文 **"PD3QN: A Predictive Dueling Double Deep Q-Network with Lightweight Future State Prediction and Adaptive Confidence Gating"** 的官方 PyTorch 代码实现。

---

## 📖 项目简介 (Introduction)

针对传统无模型（Model-free）算法在高动态、低容错环境下的“决策短视”问题，本项目提出了 **PD3QN** 框架。

该模型集成了轻量级的 **一步状态预测器 (OSSP)** 和 **自适应置信度门控网络 (CGN)**，在保证训练稳定性的同时，赋予智能体前瞻性的规划能力。实验在高度定制的 *Flappy Bird with Fireballs* 环境中进行，结果表明 PD3QN 在平均得分和最大得分上均显著优于 D3QN 基线。

### 核心特性
1.  **OSSP 预测器**: 使用混合损失函数 (MSE + Edge + SSIM) 捕捉环境物理动态。
2.  **CGN 门控**: 动态计算预测置信度权重 $w$，智能调节前瞻规划的影响力。
3.  **稳定性优化**: 采用固定动作向量 (Fixed Action Vectors) 和密集引导奖励 (Dense Guided Rewards)。

---

## 📂 代码结构 (File Structure)

```text
PD3QN/
├── train_PD3QN.py      # [核心] 训练脚本 (包含 PD3QN, OSSP, CGN 模型定义)
├── test_PD3QN.py       # [核心] 测试脚本 (支持批量评估与截尾统计)
├── flappy_bird.py      # 训练环境 (无渲染，高FPS优化)
├── flappy_bird-test.py # 测试环境 (带渲染窗口，用于可视化)
├── requirements.txt    # 依赖库列表
├── assets/             # 游戏素材 (图片/音频)
├── results/            # 输出目录 (保存模型 .pth 和测试报告)
└── logs/               # TensorBoard 日志目录
