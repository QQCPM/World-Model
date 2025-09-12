#!/usr/bin/env python3
"""
Live Causal VAE Training Monitor
Shows real-time training progress with epoch, step, loss, and progress bars

Usage:
python monitor_training_live.py [--refresh_seconds 3]
"""

import os
import sys
import time
import json
import re
import psutil
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from typing import Dict, Optional, Tuple


class LiveTrainingMonitor:
    """Real-time training progress monitor with detailed metrics"""
    
    def __init__(self, refresh_seconds=3):
        self.refresh_seconds = refresh_seconds
        self.status_file = Path("./data/models/causal/phase1_status.json")
        self.log_dir = Path("./data/logs/phase1")
        
        # Experiment configurations for progress calculation
        self.experiment_configs = {
            'baseline_32D': {'epochs': 40, 'priority': 'high'},
            'gaussian_256D': {'epochs': 50, 'priority': 'high'}, 
            'beta_vae_4.0': {'epochs': 50, 'priority': 'medium'},
            'no_conv_normalization': {'epochs': 40, 'priority': 'medium'},
            'hierarchical_512D': {'epochs': 50, 'priority': 'critical'},
            'categorical_512D': {'epochs': 50, 'priority': 'high'},
            'vq_vae_256D': {'epochs': 50, 'priority': 'medium'},
            'deeper_encoder': {'epochs': 40, 'priority': 'medium'}
        }
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def parse_training_progress(self, log_file: Path) -> Dict:
        """Parse training log file to extract current progress"""
        if not log_file.exists():
            return {'status': 'pending', 'epoch': 0, 'max_epochs': 0, 'loss': 0.0, 'progress': 0.0}
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return {'status': 'pending', 'epoch': 0, 'max_epochs': 0, 'loss': 0.0, 'progress': 0.0}
            
            # Look for training completion or failure
            for line in reversed(lines[-10:]):  # Check last 10 lines
                if "✅ CAUSAL-CONDITIONED training completed" in line:
                    return {'status': 'completed', 'epoch': 0, 'max_epochs': 0, 'loss': 0.0, 'progress': 100.0}
                elif "❌ Training failed:" in line:
                    return {'status': 'failed', 'epoch': 0, 'max_epochs': 0, 'loss': 0.0, 'progress': 0.0}
            
            # Find epoch progress pattern: "Epoch X/Y 🔗: Train Loss=..."
            epoch_pattern = r'Epoch\s+(\d+)/(\d+)\s+🔗:\s+Train Loss=([0-9.]+)'
            current_epoch = 0
            max_epochs = 0 
            current_loss = 0.0
            
            # Search backwards through recent lines for latest epoch info
            for line in reversed(lines[-20:]):
                match = re.search(epoch_pattern, line)
                if match:
                    current_epoch = int(match.group(1))
                    max_epochs = int(match.group(2))
                    current_loss = float(match.group(3))
                    break
            
            # If we found epoch info, calculate progress
            if max_epochs > 0:
                progress = (current_epoch / max_epochs) * 100
                status = 'training'
            else:
                # Look for batch training indicators
                batch_pattern = r'Batch \d+ \(🔗 CAUSAL\): Loss=([0-9.]+)'
                for line in reversed(lines[-5:]):
                    if re.search(batch_pattern, line):
                        match = re.search(batch_pattern, line)
                        if match:
                            current_loss = float(match.group(1))
                        status = 'training'
                        progress = 0.0
                        break
                else:
                    status = 'starting'
                    progress = 0.0
            
            return {
                'status': status,
                'epoch': current_epoch,
                'max_epochs': max_epochs,
                'loss': current_loss,
                'progress': progress
            }
            
        except Exception as e:
            return {'status': 'error', 'epoch': 0, 'max_epochs': 0, 'loss': 0.0, 'progress': 0.0, 'error': str(e)}
    
    def get_process_info(self, pid: int) -> Optional[Dict]:
        """Get process resource usage"""
        try:
            proc = psutil.Process(pid)
            return {
                'memory_mb': proc.memory_info().rss / 1024 / 1024,
                'cpu_percent': proc.cpu_percent(),
                'create_time': datetime.fromtimestamp(proc.create_time())
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def format_duration(self, start_time) -> str:
        """Format training duration"""
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        
        duration = datetime.now() - start_time
        total_minutes = int(duration.total_seconds() / 60)
        return f"{total_minutes}.{int((duration.total_seconds() % 60) / 6)}m"
    
    def create_progress_bar(self, progress: float, width: int = 10) -> str:
        """Create ASCII progress bar"""
        filled = int(progress / 10)  # Each char represents 10%
        bar = "█" * filled + "░" * (width - filled)
        return f"|{bar}|"
    
    def get_priority_icon(self, priority: str) -> str:
        """Get priority icon"""
        icons = {'critical': '🔥', 'high': '⚡', 'medium': '📊', 'low': '🔬'}
        return icons.get(priority, '🔬')
    
    def load_status(self) -> Dict:
        """Load orchestrator status"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            return {}
        except:
            return {}
    
    def show_live_progress(self):
        """Display live training progress"""
        status_data = self.load_status()
        current_time = datetime.now().strftime("%H:%M:%S")
        
        print("🧠 CAUSAL TRAINING PROGRESS -", current_time)
        print("=" * 72)
        
        # Get running processes and completed/failed sets
        running_processes = status_data.get('running_processes', {})
        completed = set(status_data.get('completed_experiments', []))
        failed = set(status_data.get('failed_experiments', []))
        
        # Group experiments
        group1 = ['baseline_32D', 'gaussian_256D', 'beta_vae_4.0', 'no_conv_normalization']
        group2 = ['hierarchical_512D', 'categorical_512D', 'vq_vae_256D', 'deeper_encoder']
        
        active_count = 0
        completed_count = len(completed)
        pending_count = 0
        
        # Process all experiments
        for exp_name in group1 + group2:
            config = self.experiment_configs.get(exp_name, {})
            priority_icon = self.get_priority_icon(config.get('priority', 'medium'))
            log_file = self.log_dir / f"{exp_name}.log"
            
            # Get training progress
            progress_info = self.parse_training_progress(log_file)
            
            # Determine status
            if exp_name in completed:
                status_text = "COMPLETED"
                duration_text = ""
                progress_bar = self.create_progress_bar(100.0)
                progress_pct = "100%"
                loss_text = ""
                epoch_text = ""
            elif exp_name in failed:
                status_text = "FAILED   "
                duration_text = ""
                progress_bar = self.create_progress_bar(0.0)
                progress_pct = "0%"
                loss_text = ""
                epoch_text = ""
            elif exp_name in running_processes:
                proc_info = running_processes[exp_name]
                start_time = proc_info.get('start_time')
                if start_time:
                    duration_text = self.format_duration(start_time)
                else:
                    duration_text = "0.0m"
                
                if progress_info['status'] == 'training' and progress_info['max_epochs'] > 0:
                    status_text = "TRAINING "
                    active_count += 1
                    epoch_text = f"Ep {progress_info['epoch']}/{progress_info['max_epochs']}"
                    progress_bar = self.create_progress_bar(progress_info['progress'])
                    progress_pct = f"{progress_info['progress']:.0f}%"
                    loss_text = f"Loss:{progress_info['loss']:.1f}"
                elif progress_info['status'] == 'training':
                    status_text = "TRAINING "
                    active_count += 1
                    epoch_text = "Starting"
                    progress_bar = self.create_progress_bar(5.0)  # Show some progress
                    progress_pct = "5%"
                    loss_text = f"Loss:{progress_info['loss']:.1f}" if progress_info['loss'] > 0 else ""
                else:
                    status_text = "STARTING "
                    active_count += 1
                    epoch_text = "Init"
                    progress_bar = self.create_progress_bar(0.0)
                    progress_pct = "0%"
                    loss_text = ""
            else:
                status_text = "PENDING  "
                duration_text = ""
                progress_bar = self.create_progress_bar(0.0)
                progress_pct = "0%"
                loss_text = ""
                epoch_text = ""
                pending_count += 1
            
            # Format the line exactly as requested
            name_padded = f"{exp_name:<20}"
            duration_padded = f"{duration_text:>5}"
            epoch_padded = f"{epoch_text:>10}"
            progress_padded = f"{progress_pct:>4}"
            loss_padded = f"{loss_text:>11}"
            
            print(f"{priority_icon} {name_padded} {status_text} {duration_padded} {epoch_padded} {progress_bar} {progress_padded} {loss_padded}")
        
        # Summary status
        print()
        print(f"📊 Status: {completed_count} completed, {active_count} active, {pending_count} pending")
        
        # System info
        system_memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        print(f"🖥️  System: {cpu_percent:.1f}% CPU, {system_memory.percent:.1f}% RAM")
        
        if completed_count == 8:
            print("🎉 ALL EXPERIMENTS COMPLETED! Ready for Phase 2A")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        print("🔄 Starting live training monitor...")
        print(f"Refresh rate: {self.refresh_seconds} seconds")
        print("Press Ctrl+C to exit\n")
        
        try:
            while True:
                self.clear_screen()
                self.show_live_progress()
                print(f"\n🔄 Refreshing every {self.refresh_seconds}s... (Ctrl+C to exit)")
                time.sleep(self.refresh_seconds)
                
        except KeyboardInterrupt:
            print("\n👋 Live monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitor error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Live Causal VAE Training Monitor')
    parser.add_argument('--refresh_seconds', type=int, default=3,
                       help='Refresh interval in seconds')
    
    args = parser.parse_args()
    
    monitor = LiveTrainingMonitor(refresh_seconds=args.refresh_seconds)
    monitor.monitor_loop()


if __name__ == "__main__":
    main()