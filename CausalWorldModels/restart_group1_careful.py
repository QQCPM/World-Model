#!/usr/bin/env python3
"""
Careful Group 1 Restart - Using proven working method
Restarts Group 1 experiments with the exact same approach that worked in testing
"""

import subprocess
import time
import os
from pathlib import Path

def start_experiment_carefully(name, arch, epochs, batch_size, lr, extra_args=None):
    """Start experiment using the exact method that worked in testing"""
    
    # Build command exactly like the working test
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
    
    print(f"🚀 Starting {name}")
    print(f"   Command: {' '.join(cmd)}")
    
    # Create fresh log file
    log_file = Path(f"./data/logs/phase1/{name}.log")
    if log_file.exists():
        log_file.unlink()
        print(f"   🗑️  Cleared old log")
    
    # Start with output redirection (same as working method)
    full_cmd = f"{' '.join(cmd)} > {log_file} 2>&1 &"
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Get the PID from the background process
        time.sleep(1)  # Let process start
        pid_result = subprocess.run(['pgrep', '-f', f'--architecture {arch}'], 
                                  capture_output=True, text=True)
        if pid_result.stdout.strip():
            pid = pid_result.stdout.strip().split('\n')[-1]  # Get latest PID
            print(f"   ✅ {name} started (PID: {pid})")
            return pid
        else:
            print(f"   ⚠️  {name} started but PID not found")
            return None
    else:
        print(f"   ❌ {name} failed to start: {result.stderr}")
        return None

def main():
    print("🔄 CAREFUL GROUP 1 RESTART")
    print("Using the exact method that worked in testing")
    print("=" * 50)
    
    # Group 1 experiments (excluding completed baseline_32D)
    experiments = [
        {
            'name': 'gaussian_256D',
            'arch': 'gaussian_256D',
            'epochs': 50,
            'batch_size': 32,
            'lr': 0.0001
        },
        {
            'name': 'beta_vae_4.0',
            'arch': 'beta_vae_4.0', 
            'epochs': 50,
            'batch_size': 28,
            'lr': 0.00009,
            'extra': ['--beta', '4.0']
        },
        {
            'name': 'no_conv_normalization',
            'arch': 'no_conv_normalization',
            'epochs': 40,
            'batch_size': 32,
            'lr': 0.0001
        }
    ]
    
    started_pids = []
    
    for exp in experiments:
        extra = exp.get('extra', None)
        pid = start_experiment_carefully(
            exp['name'], exp['arch'], exp['epochs'],
            exp['batch_size'], exp['lr'], extra
        )
        if pid:
            started_pids.append((exp['name'], pid))
        
        # Stagger starts to avoid resource conflicts
        time.sleep(3)
    
    print(f"\n🎯 GROUP 1 RESTART SUMMARY:")
    print(f"   Started {len(started_pids)}/3 experiments")
    
    for name, pid in started_pids:
        print(f"   ✅ {name} (PID: {pid})")
    
    print(f"\n💡 Monitor with:")
    print(f"   python process_detective.py")
    print(f"   tail -f data/logs/phase1/gaussian_256D.log")

if __name__ == "__main__":
    main()
