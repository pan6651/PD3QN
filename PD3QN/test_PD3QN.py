#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import json
import statistics
from collections import defaultdict
import glob
import importlib.util

current_file = os.path.abspath(__file__)
project_root_dir = os.path.dirname(current_file)

src_path = os.path.join(project_root_dir, 'src')
sys.path.append(src_path)

ddqn_research_path = os.path.join(project_root_dir, 'DDQN_Innovation_Research')
sys.path.append(ddqn_research_path)

os.chdir(project_root_dir)

spec = importlib.util.spec_from_file_location("flappy_bird_test", os.path.join(src_path, "flappy_bird-test.py"))
flappy_bird_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flappy_bird_test)
FlappyBird = flappy_bird_test.FlappyBird

class FixedActionVectors:
    def __init__(self, action_dim=2, vector_dim=64):
        self.action_dim = action_dim
        self.vector_dim = vector_dim
        
        if vector_dim == 64:
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
            action_0_vector = torch.tensor([
                -0.5, 0.3, -0.8, 0.2, -0.6, 0.1, -0.4, 0.7,
                -0.3, 0.5, -0.9, 0.0, -0.2, 0.4, -0.7, 0.6,
                -0.1, 0.8, -0.5, 0.3, -0.6, 0.2, -0.4, 0.9,
                -0.8, 0.1, -0.3, 0.7, -0.5, 0.4, -0.2, 0.6
            ], dtype=torch.float32)
            
            action_1_vector = torch.tensor([
                0.8, -0.2, 0.6, -0.1, 0.9, -0.3, 0.4, -0.5,
                0.7, -0.4, 0.5, -0.6, 0.8, -0.1, 0.3, -0.7,
                0.9, -0.0, 0.6, -0.2, 0.4, -0.8, 0.7, -0.3,
                0.5, -0.9, 0.8, -0.1, 0.2, -0.6, 0.9, -0.4
            ], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported vector dimension: {vector_dim}")
        
        self.action_vectors = torch.stack([action_0_vector, action_1_vector], dim=0)
    
    def get_action_vector(self, action, device='cpu'):
        if isinstance(action, int):
            return self.action_vectors[action].to(device).unsqueeze(0)
        else:
            return self.action_vectors.to(device)[action]
    
    def to(self, device):
        self.action_vectors = self.action_vectors.to(device)
        return self

class LossConvergenceFramePredictorFixed(nn.Module):
    def __init__(self):
        super(LossConvergenceFramePredictorFixed, self).__init__()
        
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
        
        self.fixed_action_vectors = FixedActionVectors(action_dim=2, vector_dim=64)
        
        self.dynamics = nn.Sequential(
            nn.Linear(256 + 64, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
        self.frame_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0),
            nn.LayerNorm([128, 4, 4]),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, 8, 8]),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([32, 16, 16]),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([16, 32, 32]),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([8, 64, 64]),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((84, 84)),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        self.terminal_predictor = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state, action):
        batch_size = state.size(0)
        device = state.device
        
        state_features = self.state_encoder(state)
        state_features = state_features.view(batch_size, -1)
        
        action_embed = self.fixed_action_vectors.get_action_vector(action, device)
        
        combined_features = torch.cat([state_features, action_embed], dim=1)
        
        next_frame_latent = self.dynamics(combined_features)
        next_frame_latent_reshaped = next_frame_latent.view(batch_size, 256, 1, 1)
        next_frame = self.frame_decoder(next_frame_latent_reshaped)
        
        terminal_prob = self.terminal_predictor(combined_features)
        
        return next_frame, terminal_prob
    
    def to(self, device):
        super().to(device)
        self.fixed_action_vectors.to(device)
        return self

class SimplifiedWeightNetworkFixed(nn.Module):
    def __init__(self):
        super(SimplifiedWeightNetworkFixed, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.fixed_action_vectors = FixedActionVectors(action_dim=2, vector_dim=32)
        
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
        device = state.device
        
        x = state.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        state_features = x.view(x.size(0), -1)
        
        action_embed = self.fixed_action_vectors.get_action_vector(action, device)
        combined_features = torch.cat([state_features, action_embed], dim=1)
        weight = self.weight_network(combined_features)
        
        return weight
    
    def to(self, device):
        super().to(device)
        self.fixed_action_vectors.to(device)
        return self

class PD3QN(nn.Module):
    def __init__(self):
        super(PD3QN, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        linear_input_size = 7 * 7 * 64
        fc_output_size = 512
        
        self.fc_val = nn.Linear(linear_input_size, fc_output_size)
        self.fc_adv = nn.Linear(linear_input_size, fc_output_size)
        self.val = nn.Linear(fc_output_size, 1)
        self.adv = nn.Linear(fc_output_size, 2)
        
        self.frame_predictor = LossConvergenceFramePredictorFixed()
        self.weight_network = SimplifiedWeightNetworkFixed()
    
    def forward(self, x):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x_flat = x.view(x.size(0), -1)
        x_val = F.relu(self.fc_val(x_flat))
        x_adv = F.relu(self.fc_adv(x_flat))
        
        val = self.val(x_val)
        adv = self.adv(x_adv)
        
        q_values = val + adv - adv.mean(1, keepdim=True)
        return q_values
    
    def enhanced_q_values(self, state, gamma=0.99):
        batch_size = state.size(0)
        device = state.device
        
        current_q_values = self.forward(state)
        enhanced_q_values = current_q_values.clone()
        
        for action_idx in range(2):
            action_tensor = torch.full((batch_size,), action_idx, dtype=torch.long, device=device)
            
            with torch.no_grad():
                predicted_4th_frame, terminal_prob = self.frame_predictor(state, action_tensor)
            
            new_state = torch.cat([
                state[:, 1:, :, :],
                predicted_4th_frame
            ], dim=1)
            
            weight = self.weight_network(state, action_tensor)
            
            with torch.no_grad():
                future_q_values = self.forward(new_state)
                state_value = torch.max(future_q_values, dim=1, keepdim=True)[0]
            
            terminal_mask = (terminal_prob > 0.5).float()
            adjusted_state_value = state_value * (1 - terminal_mask) - terminal_mask * 1.0
            
            value_enhancement = weight * gamma * adjusted_state_value
            enhanced_q_values[:, action_idx:action_idx+1] += value_enhancement
        
        return enhanced_q_values, current_q_values
    
    def to(self, device):
        super().to(device)
        self.frame_predictor.to(device)
        self.weight_network.to(device)
        return self

def pre_processing(image, width, height):
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    image = image.astype(np.float32) / 255.0
    return image[None, :, :]

def test_single_model(model_path, test_rounds=50, image_size=84):
    model_name = os.path.basename(model_path).replace('.pth', '')
    
    if 'final' in model_name:
        step_num = "Final"
    else:
        try:
            step_num = model_name.split('_')[-1]
            step_num = f"{int(step_num)//1000}k"
        except:
            step_num = "Unknown"
    print(f"🔍 Testing model at step {step_num}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PD3QN().to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model.eval()
    
    scores = []
    
    for round_idx in range(test_rounds):
        try:
            game_state = FlappyBird(fps=300)
            image, reward, terminal = game_state.next_frame(0)
            image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], image_size, image_size)
            image = torch.from_numpy(image).to(device)
            state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
            
            total_score = 0
            
            while True:
                with torch.no_grad():
                    enhanced_q_values, _ = model.enhanced_q_values(state)
                    action = torch.argmax(enhanced_q_values[0]).item()
                
                next_image, reward, terminal = game_state.next_frame(action)
                
                if terminal:
                    total_score = getattr(game_state, 'final_score', 0)
                    break
                
                next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], image_size, image_size)
                next_image = torch.from_numpy(next_image).to(device)
                next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
                state = next_state
            scores.append(total_score)
            
        except Exception as e:
            print(f"  Test round {round_idx+1} failed: {e}")
            scores.append(0)
    
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0
    success_rate = sum(1 for s in scores if s > 0) / len(scores) * 100 if scores else 0
    
    print(f"   ✅ Done | Avg: {avg_score:.1f} | Max: {max_score} | Success Rate: {success_rate:.1f}%")
    return scores

def calculate_trimmed_statistics(scores, trim_count=5):
    if len(scores) < 2 * trim_count:
        raise ValueError(f"Insufficient data to trim {trim_count} max and {trim_count} min values")
    
    sorted_scores = sorted(scores)
    trimmed_scores = sorted_scores[trim_count:-trim_count]
    
    stats = {
        'original_count': len(scores),
        'trimmed_count': len(trimmed_scores),
        'original_scores': scores,
        'trimmed_scores': trimmed_scores,
        'original_mean': statistics.mean(scores),
        'trimmed_mean': statistics.mean(trimmed_scores),
        'original_max': max(scores),
        'trimmed_max': max(trimmed_scores),
        'original_min': min(scores),
        'trimmed_min': min(trimmed_scores),
        'original_variance': statistics.variance(scores) if len(scores) > 1 else 0,
        'trimmed_variance': statistics.variance(trimmed_scores) if len(trimmed_scores) > 1 else 0,
        'original_stdev': statistics.stdev(scores) if len(scores) > 1 else 0,
        'trimmed_stdev': statistics.stdev(trimmed_scores) if len(trimmed_scores) > 1 else 0,
        'removed_max_values': sorted_scores[-trim_count:],
        'removed_min_values': sorted_scores[:trim_count]
    }
    
    return stats

def main():
    print("🎯 PD3QN (FixedVectors) Model Testing Started")
    print("=" * 70)
    
    # NOTE: Adjust this folder name if you renamed it differently
    target_folder_name = "PD3QN_FixedVectors-Dense Guided Reward" 
    model_dir = os.path.join(project_root_dir, "results", target_folder_name)
    report_dir = os.path.join(project_root_dir, "results", "test_reports")
    
    os.makedirs(report_dir, exist_ok=True)
    
    # Look for the new file naming convention: PD3QN_FixedVectors_*.pth
    model_pattern = os.path.join(model_dir, "PD3QN_FixedVectors_*.pth")
    all_model_files = glob.glob(model_pattern)
    
    model_files = [f for f in all_model_files if 'final' not in os.path.basename(f)]
    
    def extract_step_number(filename):
        basename = os.path.basename(filename)
        if 'final' in basename:
            return float('inf')
        try:
            step_str = basename.split('_')[-1].replace('.pth', '')
            return int(step_str)
        except:
            return 0
    
    model_files.sort(key=extract_step_number)
    
    print(f"🔍 Found {len(model_files)} model files in {target_folder_name}:")
    for i, model_file in enumerate(model_files):
        print(f"  {i+1:2d}. {os.path.basename(model_file)}")
    
    if len(model_files) == 0:
        print("❌ No model files found! Check the directory and file naming.")
        return
    
    print("\n🚀 Starting batch testing...")
    print("-" * 70)
    
    all_results = {}
    test_rounds = 50
    trim_count = 5
    
    for i, model_file in enumerate(model_files):
        scores = test_single_model(model_file, test_rounds)
        stats = calculate_trimmed_statistics(scores, trim_count)
        
        model_name = os.path.basename(model_file).replace('.pth', '')
        all_results[model_name] = stats
        
        try:
            step_num = model_name.split('_')[-1]
            step_display = f"{int(step_num)//1000}k steps"
        except:
            step_display = "Unknown steps"
            
        print(f"   📊 {step_display}: Trimmed Mean {stats['trimmed_mean']:.1f} | Max {stats['trimmed_max']} | Var {stats['trimmed_variance']:.1f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"PD3QN_FixedVectors_BatchTestReport_{timestamp}.json")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    readable_report_file = os.path.join(report_dir, f"PD3QN_FixedVectors_TestSummary_{timestamp}.txt")
    
    with open(readable_report_file, 'w', encoding='utf-8') as f:
        f.write("🌟 PD3QN (FixedVectors) Model Test Summary Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models Tested: {len(model_files)}\n")
        f.write(f"Rounds per Model: {test_rounds}\n")
        f.write(f"Outliers Removed: Max {trim_count} + Min {trim_count}\n")
        f.write(f"Stats based on remaining: {test_rounds - 2*trim_count} data points\n\n")
        
        sorted_models = sorted(all_results.items(), 
                             key=lambda x: x[1]['trimmed_mean'], 
                             reverse=True)
        
        f.write("📊 Model Performance Ranking (by Trimmed Mean):\n")
        f.write("-" * 70 + "\n")
        
        for rank, (model_name, stats) in enumerate(sorted_models, 1):
            f.write(f"{rank:2d}. {model_name}\n")
            f.write(f"    Trimmed Mean: {stats['trimmed_mean']:.2f}\n")
            f.write(f"    Trimmed Max: {stats['trimmed_max']}\n")
            f.write(f"    Trimmed Variance: {stats['trimmed_variance']:.2f}\n")
            f.write(f"    Trimmed Stdev: {stats['trimmed_stdev']:.2f}\n")
            f.write(f"    Original Mean: {stats['original_mean']:.2f}\n")
            f.write(f"    Removed Max Values: {stats['removed_max_values']}\n")
            f.write(f"    Removed Min Values: {stats['removed_min_values']}\n")
            f.write("\n")
        
        all_trimmed_means = [stats['trimmed_mean'] for stats in all_results.values()]
        all_trimmed_maxes = [stats['trimmed_max'] for stats in all_results.values()]
        
        f.write("🎯 Overall Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Trimmed Mean: {max(all_trimmed_means):.2f}\n")
        f.write(f"Worst Trimmed Mean: {min(all_trimmed_means):.2f}\n")
        f.write(f"Average Trimmed Mean: {statistics.mean(all_trimmed_means):.2f}\n")
        f.write(f"Highest Max Score: {max(all_trimmed_maxes)}\n")
        f.write(f"StdDev of Trimmed Means: {statistics.stdev(all_trimmed_means):.2f}\n")
    
    print(f"\n✅ Batch testing completed!")
    print(f"📄 Detailed Report: {report_file}")
    print(f"📄 Readable Summary: {readable_report_file}")
    print("\n🏆 Best Model (by Trimmed Mean):")
    
    best_model = max(all_results.items(), key=lambda x: x[1]['trimmed_mean'])
    print(f"   Model: {best_model[0]}")
    print(f"   Trimmed Mean: {best_model[1]['trimmed_mean']:.2f}")
    print(f"   Trimmed Max: {best_model[1]['trimmed_max']}")
    print(f"   Trimmed Var: {best_model[1]['trimmed_variance']:.2f}")

if __name__ == "__main__":
    main()