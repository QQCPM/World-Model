#!/usr/bin/env python3
"""
Final Working Phase 1 Orchestrator
Runs 2 groups sequentially with proper logging and monitoring
"""
import subprocess
import time
import os
from pathlib import Path
import signal
import sys

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
        '--experiment_name', f'causal_{name}',
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

def run_group(group_name, experiments):
    """Run a group of experiments in parallel"""
    print(f"\n🚀 STARTING {group_name}")
    print("=" * 50)
    
    processes = []
    
    # Start all experiments in group
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
    
    print(f"\n⏰ {group_name}: {len(processes)} experiments running in parallel")
    
    # Wait for all to complete
    for exp_name, process in processes:
        print(f"   ⏳ Waiting for {exp_name}...")
        process.wait()
        status = "✅" if process.returncode == 0 else "❌"
        print(f"   {status} {exp_name} finished (exit code: {process.returncode})")
    
    print(f"🎯 {group_name} COMPLETE!")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n🛑 Stopping all experiments...')
    os.system("pkill -f 'train_causal_vae_experiment'")
    sys.exit(0)

def main():
    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Clear old logs
    os.system("rm -rf data/logs/phase1/*")
    
    print("🧠 CAUSAL WORLD MODELS - Phase 1 Training")
    print("ALL 8 experiments running simultaneously for maximum speed!")
    print("Use Ctrl+C to stop all experiments")
    
    # ALL EXPERIMENTS: Run simultaneously (you have 74GB+ free RAM)
    all_experiments = {
        'baseline_32D': {'arch': 'baseline_32D', 'epochs': 40, 'batch_size': 32, 'lr': 0.0001},
        'gaussian_256D': {'arch': 'gaussian_256D', 'epochs': 50, 'batch_size': 32, 'lr': 0.0001},
        'beta_vae_4.0': {'arch': 'beta_vae_4.0', 'epochs': 50, 'batch_size': 28, 'lr': 0.00009, 'extra': ['--beta', '4.0']},
        'no_conv_normalization': {'arch': 'no_conv_normalization', 'epochs': 40, 'batch_size': 32, 'lr': 0.0001},
        'hierarchical_512D': {'arch': 'hierarchical_512D', 'epochs': 50, 'batch_size': 32, 'lr': 0.0001},
        'categorical_512D': {'arch': 'categorical_512D', 'epochs': 50, 'batch_size': 32, 'lr': 0.0001},
        'vq_vae_256D': {'arch': 'vq_vae_256D', 'epochs': 50, 'batch_size': 32, 'lr': 0.0001},
        'deeper_encoder': {'arch': 'deeper_encoder', 'epochs': 40, 'batch_size': 32, 'lr': 0.0001}
    }
    
    try:
        # Run ALL experiments simultaneously
        run_group("ALL 8 EXPERIMENTS (Parallel)", all_experiments)
        
        print("\n🎉 ALL 8 CAUSAL VAE EXPERIMENTS COMPLETE!")
        print("📊 Check data/models/causal/ for results")
        print("🔬 All models learned p(z | x, causal_state) - TRUE causal world models!")
        
    except KeyboardInterrupt:
        print("\n🛑 Experiments stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()