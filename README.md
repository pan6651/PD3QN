# PD3QN: A Predictive Dueling Double Deep Q-Network with Lightweight Future State Prediction and Adaptive Confidence Gating

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

We propose **PD3QN**, a novel Predictive D3QN designed to address the problem of **Decision Myopia** in highly dynamic and low-fault-tolerant environments (such as Flappy Bird with fireballs).

To tackle the lack of foresight in traditional Model-free algorithms, we introduce a **Lightweight One-Step State Predictor (OSSP)** to infer future frames and potential risks, coupled with an **Adaptive Confidence Gating Network (CGN)** to dynamically adjust the weight of predictions during decision-making. Through this mechanism, PD3QN significantly enhances the agent's long-range planning capabilities and robustness while ensuring training stability.

<p align="center">
  <img src="framework.png" alt="PD3QN Framework" width="800">
  <br>
  <em>Figure: Overall Architecture of PD3QN</em>
</p>

This guide provides a step-by-step walkthrough for using this repository. 👇

## 📂 Directory Structure

Based on the project organization:

```text
PD3QN/
├── assets/                          # Game assets (sprites, fonts)
├── DDQN_Innovation_Research/
│   └── improvements/
│       └── experiments/
│           └── train_PD3QN.py       # 🚀 Core training script
├── logs/                            # TensorBoard log files
├── results/                         # Saved models (.pth)
├── src/                             # Source code library
│   ├── flappy_bird.py               # Game environment (Training version)
│   └── flappy_bird-test.py          # Game environment (Testing version)
├── requirements.txt                 # Project dependencies
├── test_PD3QN.py                    # 🧪 Batch testing script
└── README.md                        # Project documentation
```

## ⚙️ Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/pan6651/PD3QN.git
   cd PD3QN
   ```

2. **Create a Virtual Environment:**

   ```bash
   conda create -n pd3qn python=3.10
   conda activate pd3qn
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r src/requirements.txt
   ```
   *Main dependencies include: torch, pygame, opencv-python, tensorboardX, numpy, etc.*

## 🚀 Usage Guide

### Training the Model
Run the following command from the root directory to start training the PD3QN model from scratch:
```bash
python DDQN_Innovation_Research/improvements/experiments/train_PD3QN.py
```

* **Configuration:** You can modify hyperparameters (Batch size, LR, Gamma, etc.) directly in `train_PD3QN.py`.
* **Logs:** Training logs will be saved to `logs/PD3QN_FixedVectors/`.
* **Models:** Checkpoints (model files) will be saved to `results/PD3QN_FixedVectors/`.

### Testing the Model
Used for evaluating trained models. This script performs batch testing, calculates the trimmed mean score (removing the highest and lowest scores), and generates a detailed report.

```bash
python test_PD3QN.py
```

* The script automatically searches for model files in the `results/PD3QN_FixedVectors/` directory.
* Test summaries and detailed JSON reports will be generated in `results/test_reports/`.

### Testing the Model
Used for evaluating trained models. This script performs batch testing, calculates the trimmed mean score (removing the highest and lowest scores), and generates a detailed report.

```bash
tensorboard --logdir=logs/PD3QN_FixedVectors
```

## 🎮 Environment: Flappy Bird with Fireballs

This environment is a high-difficulty version modified based on `pygame`:

* **State:** 4 stacked consecutive grayscale frames (84x84).
* **Actions:** 0 (Idle/Do nothing), 1 (Flap/Jump).
* **Rewards:**
    * Survival per frame: +0.1
    * Passing through pipes: +1.0
    * Dodging fireballs: +0.5
    * Collision (Terminal state): -1.0
    * *Dense Reward Shaping:* Continuous reward/penalty based on the distance from the center of the pipe gap.



## 🤝 Acknowledgments

* **Code Reference:**
    The game environment is based on and extended from [SeVEnMY/ReinforcementLearningFlappyBird](https://github.com/SeVEnMY/ReinforcementLearningFlappyBird).

* **Funding:**
    This research work is supported by the following projects:
    * **National Natural Science Foundation of China** (Grant No. 62471272, 61806107, 62201314)
    * **Opening Project of State Key Laboratory of Digital Publishing Technology**
    * **NSF of Shandong Province** (Grant No. ZR2025MS986)


