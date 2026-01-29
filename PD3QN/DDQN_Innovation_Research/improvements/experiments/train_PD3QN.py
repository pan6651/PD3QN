#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, sample
from tensorboardX import SummaryWriter
from collections import deque
import json
from datetime import datetime

# Add original project path
current_file = os.path.abspath(__file__)
experiments_dir = os.path.dirname(current_file)
improvements_dir = os.path.dirname(experiments_dir)
ddqn_research_dir = os.path.dirname(improvements_dir)
project_root_dir = os.path.dirname(ddqn_research_dir)

# Add paths
src_path = os.path.join(project_root_dir, 'src')
ddqn_path = ddqn_research_dir

sys.path.append(src_path)
sys.path.append(ddqn_path)

# Switch to project root directory
os.chdir(project_root_dir)

print(f"Current file path: {current_file}")
print(f"Project root directory: {project_root_dir}")

from flappy_bird import FlappyBird

# 🌟 Define fixed action vectors
class FixedActionVectors:
    """
    Fixed Action Vectors Class
    Defines fixed vector representations for each action to ensure encoding consistency
    """
    def __init__(self, action_dim=2, vector_dim=64):
        self.action_dim = action_dim
        self.vector_dim = vector_dim
        
        if vector_dim == 64:
            # 🎮 64-dim fixed action vectors (for frame predictor)
            # Action 0 (Do nothing): Vector biased towards negative values, representing "passive" action
            action_0_vector = torch.tensor([
                -0.5, 0.3, -0.8, 0.2, -0.6, 0.1, -0.4, 0.7,
                -0.3, 0.5, -0.9, 0.0, -0.2, 0.4, -0.7, 0.6,
                -0.1, 0.8, -0.5, 0.3, -0.6, 0.2, -0.4, 0.9,
                -0.8, 0.1, -0.3, 0.7, -0.5, 0.4, -0.2, 0.6,
                -0.7, 0.0, -0.9, 0.5, -0.1, 0.8, -0.4, 0.3,
                -0.6, 0.2, -0.5, 0.7, -0.3, 0.1, -0.8, 0.4,
                -0.2, 0.9, -0.7, 0.0, -0.4, 0.6, -0.1, 0.5,
                -0.9, 0.3, -0.5, 0.8, -0.6, 0.2, -0.3, 0.7
            ], dtype=torch.float32)
            
            # Action 1 (Flap): Vector biased towards positive values, representing "active" action
            action_1_vector = torch.tensor([
                0.8, -0.2, 0.6, -0.1, 0.9, -0.3, 0.4, -0.5,
                0.7, -0.4, 0.5, -0.6, 0.8, -0.1, 0.3, -0.7,
                0.9, -0.0, 0.6, -0.2, 0.4, -0.8, 0.7, -0.3,
                0.5, -0.9, 0.8, -0.1, 0.2, -0.6, 0.9, -0.4,
                0.3, -0.7, 0.6, -0.0, 0.8, -0.5, 0.1, -0.9,
                0.4, -0.2, 0.7, -0.6, 0.9, -0.3, 0.0, -0.8,
                0.5, -0.1, 0.2, -0.7, 0.8, -0.4, 0.6, -0.9,
                0.3, -0.5, 0.7, -0.0, 0.1, -0.8, 0.9, -0.2
            ], dtype=torch.float32)
        elif vector_dim == 32:
            # 🎮 32-dim fixed action vectors (for weight network)
            # Action 0 (Do nothing): Passive
            action_0_vector = torch.tensor([
                -0.5, 0.3, -0.8, 0.2, -0.6, 0.1, -0.4, 0.7,
                -0.3, 0.5, -0.9, 0.0, -0.2, 0.4, -0.7, 0.6,
                -0.1, 0.8, -0.5, 0.3, -0.6, 0.2, -0.4, 0.9,
                -0.8, 0.1, -0.3, 0.7, -0.5, 0.4, -0.2, 0.6
            ], dtype=torch.float32)
            
            # Action 1 (Flap): Active
            action_1_vector = torch.tensor([
                0.8, -0.2, 0.6, -0.1, 0.9, -0.3, 0.4, -0.5,
                0.7, -0.4, 0.5, -0.6, 0.8, -0.1, 0.3, -0.7,
                0.9, -0.0, 0.6, -0.2, 0.4, -0.8, 0.7, -0.3,
                0.5, -0.9, 0.8, -0.1, 0.2, -0.6, 0.9, -0.4
            ], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported vector dimension: {vector_dim}, only 32 or 64 supported")
        
        # Store fixed vectors
        self.action_vectors = torch.stack([action_0_vector, action_1_vector], dim=0)
        print(f"🌟 Fixed action vectors initialized:")
        print(f"   Action dim: {action_dim}")
        print(f"   Vector dim: {vector_dim}")
        print(f"   Action 0 vector mean: {action_0_vector.mean().item():.3f}")
        print(f"   Action 1 vector mean: {action_1_vector.mean().item():.3f}")
    
    def get_action_vector(self, action, device='cpu'):
        """
        Get fixed vector for specified action
        Args:
            action: Single action index or batch action tensor
            device: Target device
        Returns:
            Corresponding fixed action vector
        """
        if isinstance(action, int):
            # Single action
            return self.action_vectors[action].to(device).unsqueeze(0)
        else:
            # Batch action
            batch_size = action.size(0)
            action_vectors = self.action_vectors.to(device)
            return action_vectors[action]  # Auto-broadcast
    
    def to(self, device):
        """Move vectors to specified device"""
        self.action_vectors = self.action_vectors.to(device)
        return self

def compute_edge_loss(pred_frame, real_frame):
    """Compute Edge Loss - Focus on image edge features"""
    # Calculate gradients using Sobel operator
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred_frame.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred_frame.device)
    
    # Calculate edges of predicted frame
    pred_edge_x = F.conv2d(pred_frame, sobel_x, padding=1)
    pred_edge_y = F.conv2d(pred_frame, sobel_y, padding=1)
    pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
    
    # Calculate edges of real frame
    real_edge_x = F.conv2d(real_frame, sobel_x, padding=1)
    real_edge_y = F.conv2d(real_frame, sobel_y, padding=1)
    real_edge = torch.sqrt(real_edge_x**2 + real_edge_y**2 + 1e-6)
    
    return F.mse_loss(pred_edge, real_edge)

def compute_ssim_loss(pred_frame, real_frame, window_size=11):
    """Compute SSIM Loss - Focus on image structural information"""
    def gaussian_window(size, sigma=1.5):
        """Create Gaussian window"""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)
    
    # Create Gaussian window
    window = gaussian_window(window_size).to(pred_frame.device)
    
    # Calculate mean
    mu1 = F.conv2d(pred_frame, window, padding=window_size//2)
    mu2 = F.conv2d(real_frame, window, padding=window_size//2)
    
    # Calculate variance and covariance
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred_frame**2, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(real_frame**2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(pred_frame * real_frame, window, padding=window_size//2) - mu1_mu2
    
    # SSIM constants
    C1 = 0.01**2
    C2 = 0.03**2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return 1 - ssim_map.mean()

def compute_multi_component_frame_loss(pred_frame, real_frame, mse_weight=1.0, edge_weight=0.1, ssim_weight=0.05):
    """Multi-component frame loss function"""
    mse_loss = F.mse_loss(pred_frame, real_frame)
    edge_loss = compute_edge_loss(pred_frame, real_frame)
    ssim_loss = compute_ssim_loss(pred_frame, real_frame)
    
    total_loss = mse_weight * mse_loss + edge_weight * edge_loss + ssim_weight * ssim_loss
    
    return total_loss, mse_loss, edge_loss, ssim_loss

def get_convergence_lr_schedule(current_iter, warmup_iters=50000, stable_iters=500000, total_iters=2000000, 
                               initial_lr=1e-4, warmup_lr=5e-5, stable_lr=1e-4, min_lr=1e-5):
    """Dynamic LR schedule - Promote loss convergence"""
    if current_iter < warmup_iters:
        # Warmup phase: gradually increase from low LR
        progress = current_iter / warmup_iters
        return warmup_lr + (stable_lr - warmup_lr) * progress
    elif current_iter < warmup_iters + stable_iters:
        # Stable phase: keep stable LR
        return stable_lr
    else:
        # Decay phase: gradually decrease LR
        decay_progress = (current_iter - warmup_iters - stable_iters) / (total_iters - warmup_iters - stable_iters)
        return stable_lr * (1 - decay_progress) + min_lr * decay_progress

class LossConvergenceFramePredictorFixed(nn.Module):
    def __init__(self):
        super(LossConvergenceFramePredictorFixed, self).__init__()
        
        # 🔧 State Encoder - Enhance feature extraction
        self.state_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.LayerNorm([32, 20, 20]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LayerNorm([64, 9, 9]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.LayerNorm([128, 4, 4]),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.ReLU()
        )
        
        # 🌟 Fixed Action Vectors (Not learnable parameters)
        self.fixed_action_vectors = FixedActionVectors(action_dim=2, vector_dim=64)
        
        # 🔄 Deeply Optimized Dynamics Network - Improve expressiveness
        self.dynamics = nn.Sequential(
            nn.Linear(256 + 64, 512),
            nn.LayerNorm(512),    # 🌟 Use LayerNorm instead of BatchNorm - Supports batch_size=1
            nn.ReLU(),
            nn.Dropout(0.3),      # 🌟 Prevent overfitting
            nn.Linear(512, 512),  # 🌟 Increase network depth
            nn.LayerNorm(512),    # 🌟 Use LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
        # 🔧 Refined Decoder - Improve reconstruction quality
        self.frame_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0),  # 1x1 -> 4x4
            nn.LayerNorm([128, 4, 4]),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 4x4 -> 8x8
            nn.LayerNorm([64, 8, 8]),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 8x8 -> 16x16
            nn.LayerNorm([32, 16, 16]),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 16x16 -> 32x32
            nn.LayerNorm([16, 32, 32]),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),     # 32x32 -> 64x64
            nn.LayerNorm([8, 64, 64]),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=1, padding=1),      # 64x64 -> 85x85
            nn.AdaptiveAvgPool2d((84, 84)),                                    # Ensure output is 84x84
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),               # Final refinement
            nn.Sigmoid()
        )
        
        # 🏁 Regularized Terminal Predictor
        self.terminal_predictor = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.LayerNorm(128),     # 🌟 Use LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),      # 🌟 Use LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state, action):
        """Forward pass"""
        batch_size = state.size(0)
        device = state.device
        
        # Encode current state
        state_features = self.state_encoder(state)
        state_features = state_features.view(batch_size, -1)
        
        # 🌟 Get fixed action vectors (no longer learnable)
        action_embed = self.fixed_action_vectors.get_action_vector(action, device)
        
        # Combine state and action features
        combined_features = torch.cat([state_features, action_embed], dim=1)
        
        # Predict latent representation of next timestep
        next_frame_latent = self.dynamics(combined_features)  # (batch, 256)
        
        # Decode the 4th frame of the next timestep
        next_frame_latent_reshaped = next_frame_latent.view(batch_size, 256, 1, 1)
        next_frame = self.frame_decoder(next_frame_latent_reshaped)
        
        # Predict terminal probability
        terminal_prob = self.terminal_predictor(combined_features)
        
        return next_frame, terminal_prob
    
    def to(self, device):
        """Override to method to ensure fixed vectors move to correct device"""
        super().to(device)
        self.fixed_action_vectors.to(device)
        return self

class StratifiedExperienceBuffer:
    """
    Stratified Experience Buffer
    🌟 Solves experience distribution shift:
    1. Early stage buffer: Stores experience from first 60 frames (fewer pipes, more black background) - 10%
    2. Late stage buffer: Stores all experience after 60 frames (includes various difficulties) - 90%
    """
    def __init__(self, total_size=46000, early_ratio=0.1):
        self.total_size = total_size
        self.early_ratio = early_ratio
        self.late_ratio = 1.0 - early_ratio
        
        self.early_size = int(total_size * early_ratio)
        self.late_size = total_size - self.early_size
        
        self.early_buffer = deque(maxlen=self.early_size)     # Early stage buffer (0-60 frames)
        self.late_buffer = deque(maxlen=self.late_size)       # Late stage buffer (60+ frames)
        
        self.game_step_count = 0  # Current game step counter
        print(f"🌟 Simplified Stratified Buffer Initialized:")
        print(f"   Early pool (0-60 frames): {self.early_size} (Ratio: {early_ratio:.1f})")
        print(f"   Late pool (60+ frames): {self.late_size} (Ratio: {self.late_ratio:.1f})")
    
    def add_experience(self, experience, is_terminal=False):
        """Add experience based on game progress"""
        if is_terminal:
            self.game_step_count = 0  # Game over, reset counter
        else:
            self.game_step_count += 1
        
        # Decide which buffer to store in based on steps
        if self.game_step_count <= 60:           # Early stage: first 60 frames
            self.early_buffer.append(experience)
        else:                                    # Late stage: after 60 frames
            self.late_buffer.append(experience)
    
    def get_adaptive_sample(self, batch_size, training_progress=0.0):
        """
        Stratified sampling strategy
        Always keep 10% early experience, 90% late experience
        """
        # Fixed sampling ratio: Early 10%, Late 90%
        early_weight = 0.1
        late_weight = 0.9
        
        # Calculate sample counts for each buffer
        early_samples = max(int(batch_size * early_weight), 1) if len(self.early_buffer) > 0 else 0
        late_samples = batch_size - early_samples
        late_samples = min(late_samples, len(self.late_buffer)) if len(self.late_buffer) > 0 else 0
        
        # Adjust sample counts to ensure total is batch_size
        total_available = early_samples + late_samples
        if total_available < batch_size:
            # If samples insufficient, prioritize filling from late buffer
            if len(self.late_buffer) >= len(self.early_buffer):
                late_samples += batch_size - total_available
            else:
                early_samples += batch_size - total_available
        
        # Execute sampling
        batch = []
        if early_samples > 0 and len(self.early_buffer) > 0:
            batch.extend(sample(list(self.early_buffer), min(early_samples, len(self.early_buffer))))
        if late_samples > 0 and len(self.late_buffer) > 0:
            batch.extend(sample(list(self.late_buffer), min(late_samples, len(self.late_buffer))))
        
        # If batch is still short, fill randomly from all buffers
        all_experiences = list(self.early_buffer) + list(self.late_buffer)
        while len(batch) < batch_size and len(all_experiences) > 0:
            remaining = sample(all_experiences, min(batch_size - len(batch), len(all_experiences)))
            batch.extend(remaining)
        
        return batch[:batch_size]
    
    def __len__(self):
        return len(self.early_buffer) + len(self.late_buffer)
    
    def get_stats(self):
        """Get buffer statistics"""
        return {
            'early': len(self.early_buffer),
            'late': len(self.late_buffer),
            'total': len(self)
        }

class SimplifiedWeightNetworkFixed(nn.Module):
    def __init__(self):
        super(SimplifiedWeightNetworkFixed, self).__init__()
        
        # State feature extraction
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 🌟 Fixed action vectors
        self.fixed_action_vectors = FixedActionVectors(action_dim=2, vector_dim=32)
        
        # Weight computation network
        linear_input_size = 7 * 7 * 64 + 32
        self.weight_network = nn.Sequential(
            nn.Linear(linear_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state, action):
        """Compute weight"""
        device = state.device
        
        # Extract state features
        x = state.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        state_features = x.view(x.size(0), -1)
        
        # 🌟 Get fixed action vectors
        action_embed = self.fixed_action_vectors.get_action_vector(action, device)
        
        # Combine features and compute weight
        combined_features = torch.cat([state_features, action_embed], dim=1)
        weight = self.weight_network(combined_features)
        
        return weight
    
    def to(self, device):
        """Override to method to ensure fixed vectors move to correct device"""
        super().to(device)
        self.fixed_action_vectors.to(device)
        return self

class PD3QN(nn.Module):
    """
    PD3QN (Predictive Dueling Double DQN)
    Based on D3QN, integrates frame predictor and dynamic weight network, using fixed action vectors for stability
    """
    def __init__(self):
        super(PD3QN, self).__init__()
        
        # D3QN Core Network (Dueling Architecture)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        linear_input_size = 7 * 7 * 64
        fc_output_size = 512
        
        # Value stream and Advantage stream
        self.fc_val = nn.Linear(linear_input_size, fc_output_size)
        self.fc_adv = nn.Linear(linear_input_size, fc_output_size)
        self.val = nn.Linear(fc_output_size, 1)
        self.adv = nn.Linear(fc_output_size, 2)
        
        # 🌟 PD3QN Enhanced Components: Fixed vector predictor and weight network
        self.frame_predictor = LossConvergenceFramePredictorFixed()
        self.weight_network = SimplifiedWeightNetworkFixed()
    
    def forward(self, x):
        """Standard D3QN forward pass (for getting current Q-values)"""
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x_flat = x.view(x.size(0), -1)
        x_val = F.relu(self.fc_val(x_flat))
        x_adv = F.relu(self.fc_adv(x_flat))
        
        val = self.val(x_val)
        adv = self.adv(x_adv)
        
        # Dueling aggregation: Q = V + (A - mean(A))
        q_values = val + adv - adv.mean(1, keepdim=True)
        return q_values
    
    def enhanced_q_values(self, state, gamma=0.99):
        """PD3QN Core: Compute Enhanced Q-Values"""
        batch_size = state.size(0)
        device = state.device
        
        # Compute base D3QN Q-values
        current_q_values = self.forward(state)
        enhanced_q_values = current_q_values.clone()
        
        # Compute enhancement term for each action
        for action_idx in range(2):
            action_tensor = torch.full((batch_size,), action_idx, dtype=torch.long, device=device)
            
            # 1. Predict next frame
            with torch.no_grad():
                predicted_4th_frame, terminal_prob = self.frame_predictor(state, action_tensor)
            
            # 2. Construct imaginary new state (State_{t+1})
            new_state = torch.cat([
                state[:, 1:, :, :],
                predicted_4th_frame
            ], dim=1)
            
            # 3. Compute dynamic weight
            weight = self.weight_network(state, action_tensor)
            
            # 4. Compute future state value V(s')
            with torch.no_grad():
                future_q_values = self.forward(new_state)
                state_value = torch.max(future_q_values, dim=1, keepdim=True)[0]
            
            # 5. Adjust for terminal state
            terminal_mask = (terminal_prob > 0.5).float()
            adjusted_state_value = state_value * (1 - terminal_mask) - terminal_mask * 1.0
            
            # 6. Compute enhancement and add
            value_enhancement = weight * gamma * adjusted_state_value
            enhanced_q_values[:, action_idx:action_idx+1] += value_enhancement
        
        return enhanced_q_values, current_q_values
    
    def to(self, device):
        """Override to method to ensure all components move to correct device"""
        super().to(device)
        self.frame_predictor.to(device)
        self.weight_network.to(device)
        return self

def pre_processing(image, width, height):
    """Image preprocessing"""
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    image = image.astype(np.float32) / 255.0
    return image[None, :, :]

def train_pd3qn_fixed():
    """Train PD3QN - Fixed Action Vector Version"""
    
    # Path setup
    current_file = os.path.abspath(__file__)
    experiments_dir = os.path.dirname(current_file)
    improvements_dir = os.path.dirname(experiments_dir)
    ddqn_research_dir = os.path.dirname(improvements_dir)
    project_root_dir = os.path.dirname(ddqn_research_dir)
    
    # Change save path to PD3QN
    saved_path = os.path.join(project_root_dir, "results", "PD3QN_FixedVectors")
    log_path = os.path.join(project_root_dir, "logs", "PD3QN_FixedVectors")
    
    print(f"🌟 PD3QN (Fixed Vector Version) Training Config")
    print(f"📁 Model save path: {saved_path}")
    print(f"📁 Log save path: {log_path}")
    
    # Training parameters
    image_size = 84
    lr = 1e-4
    num_iters = 2000000
    initial_epsilon = 0.1
    final_epsilon = 1e-4
    memory_buffer_size = 46000
    gamma = 0.99
    batch_size = 64
    target_update_freq = 1000
    
    # Create directories
    os.makedirs(saved_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 🌟 Model Initialization - PD3QN
    main_model = PD3QN().to(device)
    target_model = PD3QN().to(device)
    target_model.load_state_dict(main_model.state_dict())
    
    # Optimizer - Use AdamW with weight decay
    pd3qn_optimizer = torch.optim.AdamW([
        {'params': main_model.conv1.parameters()},
        {'params': main_model.conv2.parameters()},
        {'params': main_model.conv3.parameters()},
        {'params': main_model.bn1.parameters()},
        {'params': main_model.bn2.parameters()},
        {'params': main_model.bn3.parameters()},
        {'params': main_model.fc_val.parameters()},
        {'params': main_model.fc_adv.parameters()},
        {'params': main_model.val.parameters()},
        {'params': main_model.adv.parameters()},
        {'params': main_model.weight_network.parameters()}
    ], lr=lr, weight_decay=1e-4)
    
    # 🌟 Frame Predictor Optimizer
    initial_predictor_lr = 1e-4
    frame_predictor_optimizer = torch.optim.AdamW(
        main_model.frame_predictor.parameters(), 
        lr=initial_predictor_lr,
        weight_decay=1e-4
    )
    
    # 🌟 LR Scheduler
    frame_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        frame_predictor_optimizer, 
        mode='min',
        factor=0.5,
        patience=50000,
        min_lr=1e-6
    )
    
    criterion = nn.MSELoss()
    
    # TensorBoard
    writer = SummaryWriter(log_path)
    
    # 🌟 Simplified Stratified Experience Buffer
    memory_buffer = StratifiedExperienceBuffer(
        total_size=memory_buffer_size,
        early_ratio=0.1  # First 60 frames take 10%
    )
    visual_memory_buffer = StratifiedExperienceBuffer(
        total_size=memory_buffer_size,
        early_ratio=0.1  # First 60 frames take 10%
    )
    
    # Initialize game
    game_state = FlappyBird(fps=280)
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], image_size, image_size)
    image = torch.from_numpy(image).to(device)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
    
    iter_count = 0
    
    # 🌟 Loss Convergence Tracking
    frame_loss_history = deque(maxlen=1000)
    best_frame_loss = float('inf')
    convergence_threshold = 0.08  # Target convergence threshold
    warmup_period = 100000        # Warmup period, focus on frame reconstruction
    
    print("🚀 Starting PD3QN (Fixed Vector Version) Training...")
    print(f"🔧 Device: {device}")
    print(f"⚡ Game FPS: 280 FPS")
    print(f"📦 Batch size: {batch_size}")
    print(f"🎯 Target steps: {num_iters:,}")
    print(f"🔄 Target network update freq: {target_update_freq}")
    print(f"🌟 Features: PD3QN Architecture + Fixed Action Vector Encoding")
    print(f"🔧 Convergence target: Frame Loss < {convergence_threshold}")
    print(f"📊 TensorBoard: tensorboard --logdir={log_path}")
    print("-" * 70)
    
    try:
        while iter_count < num_iters:
            # Forward pass (use enhanced Q-values)
            enhanced_q_values, current_q_values = main_model.enhanced_q_values(state)
            
            # Epsilon-greedy strategy
            epsilon = final_epsilon + ((num_iters - iter_count) * (initial_epsilon - final_epsilon) / num_iters)
            iter_count += 1
            
            if random() <= epsilon:
                action = randint(0, 1)
            else:
                action = torch.argmax(enhanced_q_values[0]).item()
            
            # Execute action
            next_image, reward, terminal = game_state.next_frame(action)
            next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], image_size, image_size)
            next_image = torch.from_numpy(next_image).to(device)
            next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
            
            # 🌟 Store in stratified experience buffer
            experience = [state, action, reward, next_state, terminal]
            memory_buffer.add_experience(experience, terminal)
            visual_memory_buffer.add_experience(experience, terminal)
            
            # Training progress
            training_progress = iter_count / num_iters
            
            # Train PD3QN
            if len(memory_buffer) >= batch_size and iter_count % 4 == 0:
                batch = memory_buffer.get_adaptive_sample(batch_size, training_progress)
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

                state_batch = torch.cat(tuple(state for state in state_batch))
                action_batch = torch.tensor(action_batch, dtype=torch.long).to(device)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(device)
                next_state_batch = torch.cat(tuple(state for state in next_state_batch))
                terminal_batch = torch.tensor(terminal_batch, dtype=torch.bool).to(device)

                # Double DQN Algorithm
                with torch.no_grad():
                    next_q_values_main = main_model(next_state_batch)
                    next_actions = torch.argmax(next_q_values_main, dim=1)
                    
                    next_q_values_target = target_model(next_state_batch)
                    next_q_value = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    
                    target_q_value = reward_batch + (gamma * next_q_value * (~terminal_batch))

                current_q_values = main_model(state_batch)
                current_q_value = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                
                loss = criterion(current_q_value, target_q_value)
                
                pd3qn_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(main_model.parameters(), 10.0)
                pd3qn_optimizer.step()

            # 🌟 Train Frame Predictor
            if len(visual_memory_buffer) >= batch_size and iter_count % 1 == 0:
                vp_batch = visual_memory_buffer.get_adaptive_sample(batch_size, training_progress)
                vp_state_batch, vp_action_batch, vp_reward_batch, vp_next_state_batch, vp_terminal_batch = zip(*vp_batch)
                
                vp_state_batch = torch.cat(tuple(state for state in vp_state_batch))
                vp_action_batch = torch.tensor(vp_action_batch, dtype=torch.long).to(device)
                vp_next_state_batch = torch.cat(tuple(state for state in vp_next_state_batch))
                vp_terminal_batch = torch.tensor(vp_terminal_batch, dtype=torch.float).to(device).unsqueeze(1)
                
                # Extract real 4th frame
                real_4th_frame = vp_next_state_batch[:, -1:, :, :]
                
                # 🌟 Dynamic LR adjustment
                current_lr = get_convergence_lr_schedule(iter_count)
                for param_group in frame_predictor_optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # Predictor forward pass
                pred_4th_frame, pred_terminal = main_model.frame_predictor(vp_state_batch, vp_action_batch)
                
                # 🌟 Multi-component frame loss
                total_frame_loss, mse_loss, edge_loss, ssim_loss = compute_multi_component_frame_loss(
                    pred_4th_frame, real_4th_frame
                )
                
                # Terminal loss
                terminal_loss = F.binary_cross_entropy(pred_terminal, vp_terminal_batch)
                
                # 🌟 Dynamic weight adjustment
                if iter_count < warmup_period:
                    terminal_weight = 0.01  # Lower terminal loss weight during warmup
                else:
                    terminal_weight = 0.02
                
                total_loss = total_frame_loss + terminal_weight * terminal_loss
                
                frame_predictor_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(main_model.frame_predictor.parameters(), 10.0)
                frame_predictor_optimizer.step()
                
                # 🌟 LR Scheduler update
                frame_lr_scheduler.step(total_frame_loss.detach())
                
                # 🌟 Convergence monitoring
                frame_loss_history.append(total_frame_loss.item())
                if total_frame_loss.item() < best_frame_loss:
                    best_frame_loss = total_frame_loss.item()

            # Update target network
            if iter_count % target_update_freq == 0:
                target_model.load_state_dict(main_model.state_dict())
                print(f"🔄 Target network updated (step: {iter_count})")

            state = next_state
            
            # Log and save
            if iter_count % 2000 == 0:
                actual_fps = game_state.fps_clock.get_fps()
                progress = (iter_count / num_iters) * 100
                
                # Buffer stats
                buffer_stats = memory_buffer.get_stats()
                avg_frame_loss = np.mean(list(frame_loss_history)) if frame_loss_history else 0
                current_lr_display = frame_predictor_optimizer.param_groups[0]['lr']
                
                print(f"📈 PD3QN {iter_count:7d}/{num_iters} ({progress:5.1f}%) | FPS: {actual_fps:6.1f} | ε: {epsilon:.4f}")
                print(f"   🔧 Q Loss: {loss:.4f} | Frame Loss: {total_frame_loss:.4f} (Best: {best_frame_loss:.4f}) | LR: {current_lr_display:.6f}")
                print(f"   📊 Buffer Early:{buffer_stats['early']} Late:{buffer_stats['late']}")
                print(f"   🎯 Convergence Status: {'✅ Converged' if avg_frame_loss < convergence_threshold else '🔄 Converging'}")
                
                # TensorBoard logging
                if iter_count % 4000 == 0:
                    writer.add_scalar('Train/PD3QN_Loss', loss, iter_count)
                    writer.add_scalar('Train/Frame_Loss_Total', total_frame_loss, iter_count)
                    writer.add_scalar('Train/Frame_Loss_MSE', mse_loss, iter_count)
                    writer.add_scalar('Train/Frame_Loss_Edge', edge_loss, iter_count)
                    writer.add_scalar('Train/Frame_Loss_SSIM', ssim_loss, iter_count)
                    writer.add_scalar('Train/Terminal_Loss', terminal_loss, iter_count)
                    writer.add_scalar('Train/Epsilon', epsilon, iter_count)
                    writer.add_scalar('Train/Frame_Predictor_LR', current_lr_display, iter_count)
                    writer.add_scalar('Train/Best_Frame_Loss', best_frame_loss, iter_count)

            if iter_count % 100000 == 0:
                model_name = f"PD3QN_FixedVectors_{iter_count}.pth"
                torch.save(main_model.state_dict(), os.path.join(saved_path, model_name))
                print(f"💾 Model saved: {model_name}")
                
    except KeyboardInterrupt:
        print("⏹️  Training interrupted, saving model...")
        torch.save(main_model.state_dict(), os.path.join(saved_path, f"PD3QN_FixedVectors_interrupted_{iter_count}.pth"))
    
    # Save final model
    torch.save(main_model.state_dict(), os.path.join(saved_path, "PD3QN_FixedVectors_final.pth"))
    
    writer.close()
    print("✅ PD3QN (Fixed Vector Version) Training Completed!")

if __name__ == "__main__":
    train_pd3qn_fixed()