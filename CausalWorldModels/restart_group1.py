#!/usr/bin/env python3
"""
Restart Group 1 Interrupted Experiments
Restarts the 3 experiments that were interrupted: gaussian_256D, beta_vae_4.0, no_conv_normalization
"""

import subprocess
import time
import os
from pathlib import Path

def run_experiment(name, arch, epochs, batch_size, lr, extra_args=None):
    """Run a single experiment with proper logging"""
    cmd = [
        'python', 'experiments/train_causal_vae_experiment.py',
        '--architecture', arch,
        '--force_causal',
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--learning_rate', str(lr),
        '--data_dir', './data/causal_episodes',
        '--output_dir', f'./data/models/causal/{name}',
        '--experiment_name', f'causal_{name}'
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    # Create log file
    log_file = Path(f"./data/logs/phase1/{name}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Start process with logging
    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    return process

def main():
    print("🔄 RESTARTING GROUP 1 INTERRUPTED EXPERIMENTS")
    print("=" * 50)
    
    # Clear old logs for interrupted experiments
    interrupted_experiments = ['gaussian_256D', 'beta_vae_4.0', 'no_conv_normalization']
    
    for exp in interrupted_experiments:
        log_file = Path(f"./data/logs/phase1/{exp}.log")
        if log_file.exists():
            print(f"🗑️  Clearing old log: {exp}")
            log_file.unlink()
    
    print()
    
    # Restart the 3 interrupted experiments
    experiments = {
        'gaussian_256D': {
            'arch': 'gaussian_256D', 
            'epochs': 50, 
            'batch_size': 32, 
            'lr': 0.0001
        },
        'beta_vae_4.0': {
            'arch': 'beta_vae_4.0', 
            'epochs': 50, 
            'batch_size': 28, 
            'lr': 0.00009,
            'extra': ['--beta', '4.0']
        },
        'no_conv_normalization': {
            'arch': 'no_conv_normalization', 
            'epochs': 40, 
            'batch_size': 32, 
            'lr': 0.0001
        }
    }
    
    processes = []
    
    for exp_name, config in experiments.items():
        print(f"⚡ Starting {exp_name}")
        extra = config.get('extra', None)
        process = run_experiment(
            exp_name, config['arch'], config['epochs'], 
            config['batch_size'], config['lr'], extra
        )
        processes.append((exp_name, process))
        print(f"   ✅ {exp_name} started (PID: {process.pid})")
        time.sleep(2)  # Stagger starts
    
    print(f"\n🎯 GROUP 1 RESTART COMPLETE!")
    print(f"   3 experiments restarted and running in parallel")
    print(f"   Use 'python process_detective.py' to monitor progress")
    print(f"   Use 'python watch_training.py' for live monitoring")

if __name__ == "__main__":
    main()
