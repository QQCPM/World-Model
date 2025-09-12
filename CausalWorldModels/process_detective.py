#!/usr/bin/env python3
"""
Process Detective - Smart training process tracker
Detects stuck processes, tracks timing, investigates issues
"""

import os
import re
import time
import psutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class ProcessDetective:
    """Intelligent process monitoring and issue detection"""
    
    def __init__(self):
        self.log_dir = Path("./data/logs/phase1")
        self.status_file = Path("./data/process_status.json")
        self.experiments = [
            'baseline_32D', 'gaussian_256D', 'beta_vae_4.0', 'no_conv_normalization',
            'hierarchical_512D', 'categorical_512D', 'vq_vae_256D', 'deeper_encoder'
        ]
        
    def find_training_processes(self) -> Dict[str, Dict]:
        """Find all running training processes"""
        processes = {}
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info', 'cpu_percent']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                
                # Look for our training processes
                if 'train_causal_vae_experiment.py' in cmdline:
                    # Extract experiment name from command line
                    for exp in self.experiments:
                        if f'--architecture {exp}' in cmdline:
                            processes[exp] = {
                                'pid': proc.info['pid'],
                                'start_time': datetime.fromtimestamp(proc.info['create_time']),
                                'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                                'cpu_percent': proc.cpu_percent(interval=0.1),
                                'cmdline': cmdline
                            }
                            break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return processes
    
    def analyze_log_progress(self, exp_name: str) -> Dict:
        """Analyze training progress from log file"""
        log_file = self.log_dir / f"{exp_name}.log"
        
        if not log_file.exists():
            return {'status': 'no_log', 'last_activity': None, 'stuck_duration': 0}
        
        try:
            # Get file modification time
            mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            time_since_update = (datetime.now() - mod_time).total_seconds()
            
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return {'status': 'empty_log', 'last_activity': mod_time, 'stuck_duration': time_since_update}
            
            # Check for completion/failure
            last_lines = lines[-10:]
            for line in reversed(last_lines):
                if "✅ CAUSAL-CONDITIONED training completed" in line:
                    return {'status': 'completed', 'last_activity': mod_time, 'stuck_duration': 0}
                elif "❌ Training failed:" in line:
                    error_match = re.search(r'❌ Training failed: (.+)', line)
                    error_msg = error_match.group(1) if error_match else "Unknown error"
                    return {'status': 'failed', 'last_activity': mod_time, 'stuck_duration': 0, 'error': error_msg}
            
            # Find latest epoch and batch info
            latest_epoch = None
            latest_batch = None
            latest_loss = None
            
            # Look for epoch pattern
            for line in reversed(lines[-20:]):
                if not latest_epoch:
                    epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)\s+🔗:\s+Train Loss=([0-9.]+)', line)
                    if epoch_match:
                        latest_epoch = (int(epoch_match.group(1)), int(epoch_match.group(2)))
                        latest_loss = float(epoch_match.group(3))
                
                if not latest_batch:
                    batch_match = re.search(r'Batch (\d+) \(🔗 CAUSAL\): Loss=([0-9.]+)', line)
                    if batch_match:
                        latest_batch = int(batch_match.group(1))
                        if not latest_loss:
                            latest_loss = float(batch_match.group(2))
            
            # Determine if stuck (no log updates for >10 minutes)
            is_stuck = time_since_update > 600  # 10 minutes
            
            status = 'stuck' if is_stuck else 'training'
            if latest_epoch is None and latest_batch is None:
                status = 'starting'
            
            return {
                'status': status,
                'last_activity': mod_time,
                'stuck_duration': time_since_update if is_stuck else 0,
                'latest_epoch': latest_epoch,
                'latest_batch': latest_batch,
                'latest_loss': latest_loss,
                'total_lines': len(lines)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'last_activity': None, 'stuck_duration': 0}
    
    def estimate_completion_time(self, exp_name: str, log_analysis: Dict) -> Optional[str]:
        """Estimate when training will complete"""
        if log_analysis['status'] not in ['training', 'stuck'] or not log_analysis.get('latest_epoch'):
            return None
        
        current_epoch, total_epochs = log_analysis['latest_epoch']
        if current_epoch == 0:
            return None
        
        # Find running process to get start time
        processes = self.find_training_processes()
        if exp_name not in processes:
            return None
        
        start_time = processes[exp_name]['start_time']
        elapsed = datetime.now() - start_time
        
        # Calculate average time per epoch
        avg_time_per_epoch = elapsed.total_seconds() / current_epoch
        remaining_epochs = total_epochs - current_epoch
        estimated_remaining = timedelta(seconds=avg_time_per_epoch * remaining_epochs)
        
        completion_time = datetime.now() + estimated_remaining
        return completion_time.strftime("%H:%M")
    
    def detect_issues(self, exp_name: str, process_info: Dict, log_analysis: Dict) -> List[str]:
        """Detect potential issues with training"""
        issues = []
        
        # Check if process is stuck
        if log_analysis['status'] == 'stuck':
            stuck_minutes = log_analysis['stuck_duration'] / 60
            issues.append(f"🚨 STUCK for {stuck_minutes:.1f} minutes")
        
        # Check memory usage
        if process_info['memory_mb'] > 16000:  # >16GB
            issues.append(f"⚠️  High memory: {process_info['memory_mb']:.0f}MB")
        
        # Check CPU usage (too low might indicate hanging)
        if process_info['cpu_percent'] < 5:
            issues.append("⚠️  Low CPU usage - possibly hanging")
        
        # Check if loss is exploding
        if log_analysis.get('latest_loss') and log_analysis['latest_loss'] > 10000:
            issues.append(f"⚠️  High loss: {log_analysis['latest_loss']:.0f}")
        
        # Check if training just started but no progress
        runtime = datetime.now() - process_info['start_time']
        if runtime.total_seconds() > 300 and log_analysis['status'] == 'starting':  # 5 minutes
            issues.append("⚠️  Started 5+ min ago but no training progress")
        
        return issues
    
    def investigate_experiment(self, exp_name: str) -> Dict:
        """Full investigation of a single experiment"""
        processes = self.find_training_processes()
        log_analysis = self.analyze_log_progress(exp_name)
        
        if exp_name not in processes:
            return {
                'experiment': exp_name,
                'process_status': 'not_running',
                'log_analysis': log_analysis,
                'issues': ['❌ Process not found - may have crashed or finished'],
                'estimated_completion': None
            }
        
        process_info = processes[exp_name]
        issues = self.detect_issues(exp_name, process_info, log_analysis)
        estimated_completion = self.estimate_completion_time(exp_name, log_analysis)
        
        return {
            'experiment': exp_name,
            'process_status': 'running',
            'process_info': process_info,
            'log_analysis': log_analysis,
            'issues': issues,
            'estimated_completion': estimated_completion
        }
    
    def full_investigation(self) -> Dict:
        """Investigate all experiments"""
        print("🔍 PROCESS DETECTIVE - Full Investigation")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        results = {}
        running_count = 0
        stuck_count = 0
        completed_count = 0
        failed_count = 0
        
        for exp_name in self.experiments:
            result = self.investigate_experiment(exp_name)
            results[exp_name] = result
            
            # Count statuses
            if result['log_analysis']['status'] == 'completed':
                completed_count += 1
            elif result['log_analysis']['status'] == 'failed':
                failed_count += 1
            elif result['log_analysis']['status'] == 'stuck':
                stuck_count += 1
            elif result['process_status'] == 'running':
                running_count += 1
            
            # Print investigation results
            status_icon = self.get_status_icon(result)
            print(f"{status_icon} {exp_name:<20}", end="")
            
            if result['process_status'] == 'running':
                proc = result['process_info']
                runtime = datetime.now() - proc['start_time']
                runtime_str = f"{int(runtime.total_seconds()/60)}m"
                
                print(f" PID:{proc['pid']:<6} Runtime:{runtime_str:<6}", end="")
                print(f" CPU:{proc['cpu_percent']:.0f}% RAM:{proc['memory_mb']:.0f}MB", end="")
                
                if result['log_analysis'].get('latest_epoch'):
                    epoch_info = result['log_analysis']['latest_epoch']
                    print(f" Epoch:{epoch_info[0]}/{epoch_info[1]}", end="")
                
                if result['estimated_completion']:
                    print(f" ETA:{result['estimated_completion']}", end="")
            
            elif result['log_analysis']['status'] == 'completed':
                print(" ✅ COMPLETED", end="")
            elif result['log_analysis']['status'] == 'failed':
                error = result['log_analysis'].get('error', 'Unknown')
                print(f" ❌ FAILED: {error[:30]}", end="")
            else:
                print(" ❓ NOT RUNNING", end="")
            
            print()
            
            # Show issues
            for issue in result['issues']:
                print(f"    {issue}")
        
        print()
        print(f"📊 SUMMARY: {completed_count} completed, {running_count} running, {stuck_count} stuck, {failed_count} failed")
        
        # Show most critical issues
        critical_issues = []
        for exp_name, result in results.items():
            for issue in result['issues']:
                if '🚨' in issue:
                    critical_issues.append(f"{exp_name}: {issue}")
        
        if critical_issues:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in critical_issues:
                print(f"  {issue}")
        
        return results
    
    def get_status_icon(self, result: Dict) -> str:
        """Get status icon for experiment"""
        if result['log_analysis']['status'] == 'completed':
            return '✅'
        elif result['log_analysis']['status'] == 'failed':
            return '❌'
        elif result['log_analysis']['status'] == 'stuck':
            return '🚨'
        elif result['process_status'] == 'running':
            return '🔄'
        else:
            return '❓'
    
    def kill_stuck_processes(self, confirm=True):
        """Kill processes that are stuck"""
        results = {}
        for exp_name in self.experiments:
            results[exp_name] = self.investigate_experiment(exp_name)
        
        stuck_processes = []
        for exp_name, result in results.items():
            if result['log_analysis']['status'] == 'stuck' and result['process_status'] == 'running':
                stuck_processes.append((exp_name, result['process_info']['pid']))
        
        if not stuck_processes:
            print("No stuck processes found.")
            return
        
        print(f"Found {len(stuck_processes)} stuck processes:")
        for exp_name, pid in stuck_processes:
            print(f"  {exp_name} (PID: {pid})")
        
        if confirm:
            response = input("\nKill these processes? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
        
        for exp_name, pid in stuck_processes:
            try:
                os.kill(pid, 9)  # SIGKILL
                print(f"✅ Killed {exp_name} (PID: {pid})")
            except ProcessLookupError:
                print(f"⚠️  Process {exp_name} (PID: {pid}) already dead")
            except PermissionError:
                print(f"❌ Permission denied killing {exp_name} (PID: {pid})")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process Detective - Training Monitor')
    parser.add_argument('--kill-stuck', action='store_true', 
                       help='Kill stuck processes')
    parser.add_argument('--experiment', 
                       help='Investigate specific experiment')
    
    args = parser.parse_args()
    
    detective = ProcessDetective()
    
    if args.kill_stuck:
        detective.kill_stuck_processes()
    elif args.experiment:
        result = detective.investigate_experiment(args.experiment)
        print(json.dumps(result, indent=2, default=str))
    else:
        detective.full_investigation()

if __name__ == "__main__":
    main()
