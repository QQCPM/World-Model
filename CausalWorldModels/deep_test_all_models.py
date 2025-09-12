#!/usr/bin/env python3
"""
Deep Test All 8 Models
Comprehensive validation that all architectures can train successfully
Estimates completion times and identifies potential issues
"""

import os
import re
import time
import psutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class ModelDeepTester:
    """Comprehensive testing of all 8 causal VAE architectures"""
    
    def __init__(self):
        self.experiments = {
            'baseline_32D': {'epochs': 40, 'batch_size': 32, 'lr': 0.0001, 'expected_hours': 1.5, 'memory_gb': 8},
            'gaussian_256D': {'epochs': 50, 'batch_size': 32, 'lr': 0.0001, 'expected_hours': 7.0, 'memory_gb': 10},
            'beta_vae_4.0': {'epochs': 50, 'batch_size': 28, 'lr': 0.00009, 'expected_hours': 8.0, 'memory_gb': 12, 'extra': ['--beta', '4.0']},
            'no_conv_normalization': {'epochs': 40, 'batch_size': 32, 'lr': 0.0001, 'expected_hours': 6.0, 'memory_gb': 15},
            'hierarchical_512D': {'epochs': 50, 'batch_size': 32, 'lr': 0.0001, 'expected_hours': 8.5, 'memory_gb': 15},
            'categorical_512D': {'epochs': 50, 'batch_size': 32, 'lr': 0.0001, 'expected_hours': 9.0, 'memory_gb': 16},
            'vq_vae_256D': {'epochs': 50, 'batch_size': 32, 'lr': 0.0001, 'expected_hours': 7.5, 'memory_gb': 14},
            'deeper_encoder': {'epochs': 40, 'batch_size': 32, 'lr': 0.0001, 'expected_hours': 6.5, 'memory_gb': 8}
        }
        self.log_dir = Path("./data/logs/phase1")
        
    def get_running_processes(self) -> Dict[str, Dict]:
        """Get all running training processes with detailed info"""
        processes = {}
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info', 'cpu_percent']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                
                if 'train_causal_vae_experiment.py' in cmdline:
                    for exp in self.experiments.keys():
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
    
    def analyze_training_progress(self, exp_name: str) -> Dict:
        """Analyze detailed training progress from log"""
        log_file = self.log_dir / f"{exp_name}.log"
        
        if not log_file.exists():
            return {'status': 'no_log', 'progress': 0, 'current_epoch': 0, 'total_epochs': 0}
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            if not content.strip():
                return {'status': 'empty_log', 'progress': 0, 'current_epoch': 0, 'total_epochs': 0}
            
            # Check for completion/failure
            if "✅ CAUSAL-CONDITIONED training completed" in content:
                # Extract final validation loss
                val_loss_match = re.search(r'Best validation loss: ([0-9.]+)', content)
                final_loss = float(val_loss_match.group(1)) if val_loss_match else None
                return {'status': 'completed', 'progress': 100, 'final_loss': final_loss}
            
            if "❌ Training failed:" in content:
                error_match = re.search(r'❌ Training failed: (.+)', content)
                error = error_match.group(1) if error_match else "Unknown error"
                return {'status': 'failed', 'progress': 0, 'error': error}
            
            # Find latest epoch info
            epoch_matches = re.findall(r'Epoch\s+(\d+)/(\d+)\s+🔗:\s+Train Loss=([0-9.]+),\s+Val Loss=([0-9.]+)', content)
            
            if epoch_matches:
                current_epoch, total_epochs, train_loss, val_loss = epoch_matches[-1]
                current_epoch = int(current_epoch)
                total_epochs = int(total_epochs)
                progress = (current_epoch / total_epochs) * 100
                
                # Check if training is progressing (loss decreasing)
                if len(epoch_matches) >= 3:
                    recent_losses = [float(match[3]) for match in epoch_matches[-3:]]
                    is_improving = recent_losses[-1] < recent_losses[0]
                else:
                    is_improving = True
                
                return {
                    'status': 'training',
                    'progress': progress,
                    'current_epoch': current_epoch,
                    'total_epochs': total_epochs,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'is_improving': is_improving,
                    'total_epochs_seen': len(epoch_matches)
                }
            
            # Check for batch training (early stages)
            batch_matches = re.findall(r'Batch \d+ \(🔗 CAUSAL\): Loss=([0-9.]+)', content)
            if batch_matches:
                return {
                    'status': 'training_batches',
                    'progress': 5,  # Early stage
                    'current_epoch': 0,
                    'total_epochs': self.experiments[exp_name]['epochs'],
                    'latest_batch_loss': float(batch_matches[-1]),
                    'total_batches_seen': len(batch_matches)
                }
            
            # Check if model creation succeeded
            if "✅ Created CAUSAL model:" in content:
                return {
                    'status': 'model_created',
                    'progress': 2,
                    'current_epoch': 0,
                    'total_epochs': self.experiments[exp_name]['epochs']
                }
            
            # Check if dataset loading succeeded
            if "✅ Dataset ready:" in content:
                return {
                    'status': 'dataset_loaded',
                    'progress': 1,
                    'current_epoch': 0,
                    'total_epochs': self.experiments[exp_name]['epochs']
                }
            
            return {
                'status': 'starting',
                'progress': 0,
                'current_epoch': 0,
                'total_epochs': self.experiments[exp_name]['epochs']
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'progress': 0}
    
    def estimate_completion_time(self, exp_name: str, process_info: Dict, progress_info: Dict) -> Optional[datetime]:
        """Estimate when training will complete"""
        if progress_info['status'] not in ['training', 'training_batches']:
            return None
        
        if progress_info['current_epoch'] == 0:
            # Use expected time from configuration
            expected_hours = self.experiments[exp_name]['expected_hours']
            return datetime.now() + timedelta(hours=expected_hours)
        
        # Calculate based on actual progress
        runtime = datetime.now() - process_info['start_time']
        progress_fraction = progress_info['progress'] / 100
        
        if progress_fraction > 0:
            total_estimated_time = runtime / progress_fraction
            remaining_time = total_estimated_time - runtime
            return datetime.now() + remaining_time
        
        return None
    
    def check_model_health(self, exp_name: str, process_info: Dict, progress_info: Dict) -> List[str]:
        """Check for potential issues with model training"""
        issues = []
        
        # Check memory usage
        memory_mb = process_info['memory_mb']
        expected_memory_gb = self.experiments[exp_name]['memory_gb']
        if memory_mb > expected_memory_gb * 1024 * 1.2:  # 20% over expected
            issues.append(f"⚠️  High memory: {memory_mb:.0f}MB (expected ~{expected_memory_gb}GB)")
        
        # Check CPU usage
        cpu_percent = process_info['cpu_percent']
        if cpu_percent < 10:
            issues.append("⚠️  Very low CPU usage - may be stuck")
        elif cpu_percent > 800:  # More than 8 cores
            issues.append(f"⚠️  Very high CPU usage: {cpu_percent:.0f}%")
        
        # Check training progress
        if progress_info['status'] == 'training':
            if not progress_info.get('is_improving', True):
                issues.append("⚠️  Validation loss not improving")
            
            if progress_info.get('val_loss', 0) > 1000:
                issues.append(f"⚠️  High validation loss: {progress_info['val_loss']:.1f}")
        
        # Check runtime vs expected
        runtime = datetime.now() - process_info['start_time']
        expected_hours = self.experiments[exp_name]['expected_hours']
        if runtime.total_seconds() > expected_hours * 3600 * 1.5:  # 50% over expected
            issues.append(f"⚠️  Runtime {runtime.total_seconds()/3600:.1f}h exceeds expected {expected_hours}h")
        
        return issues
    
    def deep_test_all(self) -> Dict:
        """Perform comprehensive test of all 8 models"""
        print("🔬 DEEP TEST - All 8 Causal VAE Models")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        processes = self.get_running_processes()
        results = {}
        
        # Test each experiment
        for exp_name, config in self.experiments.items():
            print(f"🧪 Testing {exp_name}...")
            
            progress_info = self.analyze_training_progress(exp_name)
            
            if exp_name in processes:
                process_info = processes[exp_name]
                completion_time = self.estimate_completion_time(exp_name, process_info, progress_info)
                health_issues = self.check_model_health(exp_name, process_info, progress_info)
                
                results[exp_name] = {
                    'process_status': 'running',
                    'process_info': process_info,
                    'progress_info': progress_info,
                    'completion_time': completion_time,
                    'health_issues': health_issues,
                    'config': config
                }
            else:
                results[exp_name] = {
                    'process_status': 'not_running',
                    'progress_info': progress_info,
                    'completion_time': None,
                    'health_issues': [],
                    'config': config
                }
        
        # Print detailed results
        self.print_detailed_results(results)
        
        return results
    
    def print_detailed_results(self, results: Dict):
        """Print comprehensive test results"""
        print("\n📊 DETAILED TEST RESULTS")
        print("=" * 60)
        
        running_count = 0
        completed_count = 0
        failed_count = 0
        total_issues = 0
        
        for exp_name, result in results.items():
            config = result['config']
            progress = result['progress_info']
            
            # Status icon
            if progress['status'] == 'completed':
                icon = "✅"
                completed_count += 1
            elif progress['status'] == 'failed':
                icon = "❌"
                failed_count += 1
            elif result['process_status'] == 'running':
                icon = "🔄"
                running_count += 1
            else:
                icon = "❓"
            
            print(f"\n{icon} {exp_name}")
            print(f"   Expected: {config['epochs']} epochs, ~{config['expected_hours']}h, {config['memory_gb']}GB")
            
            if result['process_status'] == 'running':
                proc = result['process_info']
                runtime = datetime.now() - proc['start_time']
                
                print(f"   Status: Running (PID: {proc['pid']})")
                print(f"   Runtime: {runtime.total_seconds()/3600:.1f}h")
                print(f"   Resources: {proc['cpu_percent']:.0f}% CPU, {proc['memory_mb']:.0f}MB RAM")
                
                if progress['status'] == 'training':
                    print(f"   Progress: {progress['current_epoch']}/{progress['total_epochs']} epochs ({progress['progress']:.1f}%)")
                    print(f"   Loss: Train={progress['train_loss']:.1f}, Val={progress['val_loss']:.1f}")
                    
                    if result['completion_time']:
                        eta = result['completion_time'].strftime("%H:%M")
                        print(f"   ETA: {eta}")
                elif progress['status'] == 'training_batches':
                    print(f"   Progress: Early training ({progress['total_batches_seen']} batches)")
                    print(f"   Latest batch loss: {progress['latest_batch_loss']:.1f}")
                else:
                    print(f"   Progress: {progress['status']}")
                
                # Health issues
                if result['health_issues']:
                    total_issues += len(result['health_issues'])
                    for issue in result['health_issues']:
                        print(f"   {issue}")
            
            elif progress['status'] == 'completed':
                final_loss = progress.get('final_loss', 'Unknown')
                print(f"   Status: ✅ COMPLETED")
                print(f"   Final validation loss: {final_loss}")
            
            elif progress['status'] == 'failed':
                error = progress.get('error', 'Unknown error')
                print(f"   Status: ❌ FAILED - {error}")
            
            else:
                print(f"   Status: Not running ({progress['status']})")
        
        # Summary
        print(f"\n📈 SUMMARY")
        print(f"   ✅ Completed: {completed_count}/8")
        print(f"   🔄 Running: {running_count}/8")
        print(f"   ❌ Failed: {failed_count}/8")
        print(f"   ⚠️  Health issues: {total_issues}")
        
        # Overall completion estimate
        if running_count > 0:
            max_completion = None
            for exp_name, result in results.items():
                if result['completion_time']:
                    if max_completion is None or result['completion_time'] > max_completion:
                        max_completion = result['completion_time']
            
            if max_completion:
                print(f"   🕐 All experiments ETA: {max_completion.strftime('%H:%M')} ({(max_completion - datetime.now()).total_seconds()/3600:.1f}h remaining)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        if total_issues > 0:
            print("   - Monitor experiments with health issues")
            print("   - Consider restarting stuck processes")
        if failed_count > 0:
            print("   - Investigate failed experiments")
        if running_count == 8:
            print("   - All experiments running successfully!")
            print("   - Use 'python watch_training.py' for live monitoring")

def main():
    tester = ModelDeepTester()
    results = tester.deep_test_all()
    
    # Save results
    import json
    results_file = Path("./data/deep_test_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert datetime objects to strings for JSON serialization
    json_results = {}
    for exp_name, result in results.items():
        json_result = result.copy()
        if 'process_info' in json_result and 'start_time' in json_result['process_info']:
            json_result['process_info']['start_time'] = json_result['process_info']['start_time'].isoformat()
        if 'completion_time' in json_result and json_result['completion_time']:
            json_result['completion_time'] = json_result['completion_time'].isoformat()
        json_results[exp_name] = json_result
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
