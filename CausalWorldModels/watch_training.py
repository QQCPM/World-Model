#!/usr/bin/env python3
"""
Training Watcher - Simple continuous monitoring
Shows what's happening right now with each experiment
"""

import os
import time
import subprocess
from pathlib import Path
from datetime import datetime

def get_running_processes():
    """Find training processes with PIDs"""
    try:
        # Find python processes running our training script
        result = subprocess.run(['pgrep', '-f', 'train_causal_vae_experiment'], 
                              capture_output=True, text=True)
        pids = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        processes = {}
        for pid in pids:
            if pid:
                try:
                    # Get command line for this PID
                    cmd_result = subprocess.run(['ps', '-p', pid, '-o', 'args='], 
                                              capture_output=True, text=True)
                    if cmd_result.stdout:
                        cmdline = cmd_result.stdout.strip()
                        # Extract experiment name
                        for exp in ['baseline_32D', 'gaussian_256D', 'beta_vae_4.0', 'no_conv_normalization',
                                   'hierarchical_512D', 'categorical_512D', 'vq_vae_256D', 'deeper_encoder']:
                            if f'--architecture {exp}' in cmdline:
                                processes[exp] = pid
                                break
                except:
                    continue
        return processes
    except:
        return {}

def check_log_activity(exp_name):
    """Check if log file is being updated"""
    log_file = Path(f"./data/logs/phase1/{exp_name}.log")
    if not log_file.exists():
        return "❓ No log", 0
    
    try:
        # Get last modification time
        mod_time = log_file.stat().st_mtime
        now = time.time()
        seconds_ago = now - mod_time
        
        # Read last few lines to see current status
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return "📝 Empty log", seconds_ago
        
        last_line = lines[-1].strip()
        
        # Check for completion/failure
        if "✅ CAUSAL-CONDITIONED training completed" in last_line:
            return "✅ COMPLETED", seconds_ago
        elif "❌ Training failed:" in last_line:
            return "❌ FAILED", seconds_ago
        elif "Epoch" in last_line and "Train Loss" in last_line:
            # Extract epoch info
            import re
            match = re.search(r'Epoch\s+(\d+)/(\d+)', last_line)
            if match:
                return f"🔄 Epoch {match.group(1)}/{match.group(2)}", seconds_ago
        elif "Batch" in last_line and "Loss=" in last_line:
            return "🔄 Training", seconds_ago
        elif seconds_ago > 300:  # 5 minutes
            return "⚠️  STUCK?", seconds_ago
        else:
            return "🔄 Active", seconds_ago
            
    except Exception as e:
        return f"❌ Error: {str(e)[:20]}", 0

def watch_training():
    """Continuously watch training progress"""
    print("👀 TRAINING WATCHER - Press Ctrl+C to exit")
    print("=" * 60)
    
    try:
        while True:
            os.system('clear')
            print(f"👀 TRAINING WATCHER - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)
            
            processes = get_running_processes()
            experiments = ['baseline_32D', 'gaussian_256D', 'beta_vae_4.0', 'no_conv_normalization',
                          'hierarchical_512D', 'categorical_512D', 'vq_vae_256D', 'deeper_encoder']
            
            running_count = 0
            completed_count = 0
            stuck_count = 0
            
            for exp in experiments:
                status, seconds_ago = check_log_activity(exp)
                
                # Count statuses
                if "COMPLETED" in status:
                    completed_count += 1
                elif "STUCK" in status:
                    stuck_count += 1
                elif exp in processes:
                    running_count += 1
                
                # Format time ago
                if seconds_ago > 3600:  # > 1 hour
                    time_ago = f"{seconds_ago/3600:.1f}h ago"
                elif seconds_ago > 60:  # > 1 minute
                    time_ago = f"{seconds_ago/60:.0f}m ago"
                else:
                    time_ago = f"{seconds_ago:.0f}s ago"
                
                # Show process info
                pid_info = f"PID:{processes[exp]}" if exp in processes else "No PID"
                
                print(f"{exp:<20} {status:<15} {pid_info:<10} {time_ago}")
            
            print()
            print(f"📊 {completed_count} completed, {running_count} running, {stuck_count} stuck")
            
            # Show system resources
            try:
                # Get memory usage
                mem_result = subprocess.run(['free', '-h'], capture_output=True, text=True)
                if mem_result.stdout:
                    mem_lines = mem_result.stdout.split('\n')
                    if len(mem_lines) > 1:
                        mem_info = mem_lines[1].split()
                        if len(mem_info) >= 3:
                            print(f"💾 Memory: {mem_info[2]} used / {mem_info[1]} total")
            except:
                pass
            
            print(f"\n🔄 Refreshing every 5s... (Ctrl+C to exit)")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n👋 Watcher stopped")

if __name__ == "__main__":
    watch_training()
