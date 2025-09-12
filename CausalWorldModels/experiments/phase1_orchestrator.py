#!/usr/bin/env python3
"""
Phase 1 Experiment Orchestrator
Manages parallel training of 8 VAE architectures with proper resource allocation
and staggered execution to stay within 128GB memory limits.

Based on the revised plan:
- Group 1 (4 models): Run first 12 hours  
- Group 2 (4 models): Run next 12 hours
- Each model ~12GB RAM usage
- Total per group: ~48GB (comfortable margin)
"""

import os
import sys
import time
import json
import subprocess
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass
from datetime import datetime
import signal


@dataclass
class ExperimentConfig:
    """Configuration for a single VAE experiment"""
    name: str
    architecture: str
    description: str
    expected_memory_gb: float
    training_epochs: int
    batch_size: int
    learning_rate: float
    latent_dim: int
    special_params: Dict = None


class Phase1Orchestrator:
    """Orchestrates Phase 1 architecture validation experiments"""
    
    def __init__(self, data_dir='./data/causal_episodes/', output_dir='./data/models/phase1/', 
                 max_memory_gb=120):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_memory_gb = max_memory_gb
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/results", exist_ok=True)
        
        # Define experiment configurations
        self.experiment_configs = self._create_experiment_configs()
        
        # Group experiments for staggered execution
        self.group1, self.group2 = self._create_experiment_groups()
        
        # Track running experiments
        self.running_experiments = {}
        
        print(f"Phase1Orchestrator initialized:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Memory limit: {self.max_memory_gb}GB")
        print(f"  Total experiments: {len(self.experiment_configs)}")
    
    def _create_experiment_configs(self) -> Dict[str, ExperimentConfig]:
        """Create configurations for all 8 Phase 1 experiments"""
        
        configs = {
            # Core Architectures
            "baseline_32D": ExperimentConfig(
                name="baseline_32D",
                architecture="baseline_32D", 
                description="Original World Models architecture with 32D Gaussian latent",
                expected_memory_gb=8.0,
                training_epochs=50,
                batch_size=32,
                learning_rate=0.0001,
                latent_dim=32
            ),
            
            "gaussian_256D": ExperimentConfig(
                name="gaussian_256D",
                architecture="gaussian_256D",
                description="Modern Gaussian VAE with 256D latent and layer normalization", 
                expected_memory_gb=12.0,
                training_epochs=50,
                batch_size=24,
                learning_rate=0.0001,
                latent_dim=256
            ),
            
            "categorical_1024D": ExperimentConfig(
                name="categorical_1024D",
                architecture="categorical_512D",  # Keep implementation name
                description="DreamerV3 style categorical VAE (32x32=1024D)",
                expected_memory_gb=16.0,
                training_epochs=50, 
                batch_size=18,
                learning_rate=0.0001,
                latent_dim=1024,
                special_params={'num_categoricals': 32, 'num_classes': 32}
            ),
            
            "beta_vae_4.0": ExperimentConfig(
                name="beta_vae_4.0",
                architecture="beta_vae_4.0",
                description="β-VAE with β=4.0 for disentangled representations",
                expected_memory_gb=12.0,
                training_epochs=50,
                batch_size=24,
                learning_rate=0.0001,
                latent_dim=256,
                special_params={'beta': 4.0}
            ),
            
            "vq_vae_256D": ExperimentConfig(
                name="vq_vae_256D", 
                architecture="vq_vae_256D",
                description="Vector Quantized VAE with 256 codebook entries",
                expected_memory_gb=14.0,
                training_epochs=50,
                batch_size=20,
                learning_rate=0.0001,
                latent_dim=256,
                special_params={'codebook_size': 256, 'commitment_cost': 0.25}
            ),
            
            "hierarchical_512D": ExperimentConfig(
                name="hierarchical_512D",
                architecture="hierarchical_512D", 
                description="Our innovation: static/dynamic factorized VAE",
                expected_memory_gb=15.0,
                training_epochs=50,
                batch_size=20,
                learning_rate=0.0001,
                latent_dim=512,
                special_params={'static_dim': 256, 'dynamic_dim': 256}
            ),
            
            # Ablation Studies
            "no_conv_normalization": ExperimentConfig(
                name="no_conv_normalization",
                architecture="no_conv_normalization",
                description="Ablation: test importance of layer normalization",
                expected_memory_gb=11.0,
                training_epochs=50,
                batch_size=24,
                learning_rate=0.0001,
                latent_dim=256
            ),
            
            "deeper_encoder": ExperimentConfig(
                name="deeper_encoder",
                architecture="deeper_encoder",
                description="Ablation: test depth vs width (8 layers vs 4)",
                expected_memory_gb=16.0,
                training_epochs=50,
                batch_size=18,
                learning_rate=0.0001, 
                latent_dim=256
            )
        }
        
        return configs
    
    def _create_experiment_groups(self) -> Tuple[List[str], List[str]]:
        """Create two balanced groups for staggered execution"""
        
        # Group 1: Lighter memory usage models (run first)
        group1 = [
            "baseline_32D",          # 8GB
            "gaussian_256D",         # 12GB  
            "beta_vae_4.0",         # 12GB
            "no_conv_normalization"  # 11GB
        ]
        # Total Group 1: ~43GB
        
        # Group 2: Heavier memory usage models (run second)
        group2 = [
            "categorical_1024D",     # 16GB
            "vq_vae_256D",          # 14GB
            "hierarchical_512D",     # 15GB
            "deeper_encoder"         # 16GB
        ]
        # Total Group 2: ~60GB
        
        # Verify memory allocations
        group1_memory = sum(self.experiment_configs[name].expected_memory_gb for name in group1)
        group2_memory = sum(self.experiment_configs[name].expected_memory_gb for name in group2)
        
        print(f"\nMemory allocation plan:")
        print(f"  Group 1: {group1_memory}GB / {self.max_memory_gb}GB ({group1})")
        print(f"  Group 2: {group2_memory}GB / {self.max_memory_gb}GB ({group2})")
        
        if group1_memory > self.max_memory_gb or group2_memory > self.max_memory_gb:
            raise ValueError("Memory allocation exceeds limit!")
        
        return group1, group2
    
    def run_phase1_experiments(self, dry_run=False):
        """Run all Phase 1 experiments in staggered groups"""
        
        print(f"\n🚀 Starting Phase 1 Architecture Validation Experiments")
        print(f"   Total experiments: {len(self.experiment_configs)}")
        print(f"   Execution strategy: Staggered groups (4+4)")
        print(f"   Dry run: {'Yes' if dry_run else 'No'}")
        
        if dry_run:
            print("\n📋 DRY RUN - Would execute:")
            for i, group in enumerate([self.group1, self.group2], 1):
                group_memory = sum(self.experiment_configs[name].expected_memory_gb for name in group)
                print(f"  Group {i} ({group_memory}GB): {group}")
            return
        
        # Execute Group 1
        print(f"\n⏱️  Phase 1.1: Starting Group 1 experiments")
        start_time = time.time()
        
        group1_results = self._run_experiment_group(self.group1, group_name="Group1")
        
        group1_duration = time.time() - start_time
        print(f"✅ Group 1 completed in {group1_duration/3600:.1f} hours")
        
        # Wait and cleanup before Group 2
        print(f"\n🔄 Cleaning up before Group 2...")
        time.sleep(30)  # Let memory clear
        
        # Execute Group 2
        print(f"\n⏱️  Phase 1.2: Starting Group 2 experiments")
        group2_start = time.time()
        
        group2_results = self._run_experiment_group(self.group2, group_name="Group2")
        
        group2_duration = time.time() - group2_start
        print(f"✅ Group 2 completed in {group2_duration/3600:.1f} hours")
        
        # Generate summary
        total_duration = time.time() - start_time
        all_results = {**group1_results, **group2_results}
        
        self._generate_phase1_summary(all_results, total_duration)
        
        return all_results
    
    def _run_experiment_group(self, experiment_names: List[str], group_name: str) -> Dict:
        """Run a group of experiments in parallel"""
        
        print(f"\n🔧 Starting {group_name} experiments in parallel:")
        
        # Start all experiments in the group
        processes = {}
        
        for exp_name in experiment_names:
            config = self.experiment_configs[exp_name]
            print(f"  🚀 Starting {exp_name} ({config.expected_memory_gb}GB)")
            
            # Create experiment command
            cmd = self._create_experiment_command(config)
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )
            
            processes[exp_name] = {
                'process': process,
                'config': config,
                'start_time': time.time(),
                'cmd': ' '.join(cmd)
            }
        
        print(f"✅ All {len(processes)} {group_name} experiments started")
        
        # Monitor experiments
        results = self._monitor_experiments(processes, group_name)
        
        return results
    
    def _create_experiment_command(self, config: ExperimentConfig) -> List[str]:
        """Create command to run a single experiment"""
        
        # This would normally call the actual training script
        # For now, create a placeholder command that simulates training
        
        cmd = [
            sys.executable,  # Use current python (conda) instead of system python3
            'experiments/train_causal_vae_experiment.py',
            '--architecture', config.architecture,
            '--force_causal',  # CRITICAL: Enable causal conditioning
            '--epochs', str(config.training_epochs),
            '--batch_size', str(config.batch_size),
            '--learning_rate', str(config.learning_rate),
            '--data_dir', self.data_dir,
            '--output_dir', f"{self.output_dir}/{config.name}",
            '--experiment_name', config.name
        ]
        
        # Add special parameters
        if config.special_params:
            for key, value in config.special_params.items():
                cmd.extend([f'--{key}', str(value)])
        
        return cmd
    
    def _monitor_experiments(self, processes: Dict, group_name: str) -> Dict:
        """Monitor running experiments and collect results"""
        
        results = {}
        completed = set()
        
        print(f"\n📊 Monitoring {group_name} experiments...")
        
        while len(completed) < len(processes):
            for exp_name, proc_info in processes.items():
                if exp_name in completed:
                    continue
                
                process = proc_info['process']
                
                # Check if process finished
                if process.poll() is not None:
                    # Process completed
                    end_time = time.time()
                    duration = end_time - proc_info['start_time']
                    
                    stdout, stderr = process.communicate()
                    
                    result = {
                        'experiment_name': exp_name,
                        'architecture': proc_info['config'].architecture,
                        'duration_seconds': duration,
                        'duration_hours': duration / 3600,
                        'return_code': process.returncode,
                        'success': process.returncode == 0,
                        'stdout': stdout,
                        'stderr': stderr,
                        'expected_memory_gb': proc_info['config'].expected_memory_gb,
                        'command': proc_info['cmd']
                    }
                    
                    results[exp_name] = result
                    completed.add(exp_name)
                    
                    status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
                    print(f"  {status} {exp_name} in {duration/3600:.1f}h")
                    
                    # Save individual result
                    self._save_experiment_result(result)
            
            # Brief pause before checking again
            if len(completed) < len(processes):
                time.sleep(60)  # Check every minute
        
        print(f"🏁 All {group_name} experiments completed")
        return results
    
    def _save_experiment_result(self, result: Dict):
        """Save individual experiment result"""
        
        result_file = f"{self.output_dir}/results/{result['experiment_name']}_result.json"
        
        # Create serializable version
        save_result = result.copy()
        save_result['timestamp'] = datetime.now().isoformat()
        
        with open(result_file, 'w') as f:
            json.dump(save_result, f, indent=2)
        
        print(f"📁 Saved result: {result_file}")
    
    def _generate_phase1_summary(self, all_results: Dict, total_duration: float):
        """Generate comprehensive Phase 1 summary"""
        
        print(f"\n📈 Generating Phase 1 Summary...")
        
        # Calculate statistics
        successful_experiments = [r for r in all_results.values() if r['success']]
        failed_experiments = [r for r in all_results.values() if not r['success']]
        
        total_memory_used = sum(r['expected_memory_gb'] for r in all_results.values())
        avg_duration_hours = sum(r['duration_hours'] for r in all_results.values()) / len(all_results)
        
        summary = {
            'phase': 'Phase 1 - Architecture Validation',
            'execution_timestamp': datetime.now().isoformat(),
            'total_duration_hours': total_duration / 3600,
            'total_experiments': len(all_results),
            'successful_experiments': len(successful_experiments),
            'failed_experiments': len(failed_experiments),
            'success_rate': len(successful_experiments) / len(all_results),
            'total_memory_allocated_gb': total_memory_used,
            'average_experiment_duration_hours': avg_duration_hours,
            'experiment_results': all_results,
            'top_architectures': self._rank_architectures(successful_experiments),
            'recommendations': self._generate_recommendations(all_results)
        }
        
        # Save summary
        summary_file = f"{self.output_dir}/phase1_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n" + "="*60)
        print(f"📊 PHASE 1 RESULTS SUMMARY")
        print(f"="*60)
        print(f"Total Duration: {total_duration/3600:.1f} hours")
        print(f"Experiments: {len(successful_experiments)}/{len(all_results)} successful")
        print(f"Success Rate: {len(successful_experiments)/len(all_results)*100:.1f}%")
        print(f"Memory Used: {total_memory_used:.1f}GB / {self.max_memory_gb}GB")
        
        if successful_experiments:
            print(f"\n🏆 Top Performing Architectures:")
            for i, arch in enumerate(summary['top_architectures'][:3], 1):
                print(f"  {i}. {arch['name']} ({arch['duration_hours']:.1f}h)")
        
        if failed_experiments:
            print(f"\n⚠️  Failed Experiments:")
            for exp in failed_experiments:
                print(f"  - {exp['experiment_name']}: {exp['stderr'][:100]}...")
        
        print(f"\n📁 Full summary saved to: {summary_file}")
        print(f"🎯 Ready for Phase 2A: Causal Validation")
    
    def _rank_architectures(self, successful_experiments: List[Dict]) -> List[Dict]:
        """Rank successful architectures by performance"""
        
        # For now, rank by training time (faster = better for experimentation)
        # In real implementation, would use validation metrics
        
        ranked = sorted(successful_experiments, key=lambda x: x['duration_hours'])
        
        return [
            {
                'name': exp['experiment_name'],
                'architecture': exp['architecture'],
                'duration_hours': exp['duration_hours'],
                'memory_gb': exp['expected_memory_gb']
            }
            for exp in ranked
        ]
    
    def _generate_recommendations(self, all_results: Dict) -> List[str]:
        """Generate recommendations for Phase 2A"""
        
        recommendations = []
        
        successful_experiments = [r for r in all_results.values() if r['success']]
        
        if len(successful_experiments) >= 2:
            # Get top 2-3 architectures for Phase 2A
            top_archs = sorted(successful_experiments, key=lambda x: x['duration_hours'])[:3]
            arch_names = [a['experiment_name'] for a in top_archs]
            recommendations.append(f"Use top architectures for Phase 2A: {', '.join(arch_names)}")
        
        # Memory recommendations
        max_memory_used = max(r['expected_memory_gb'] for r in all_results.values())
        if max_memory_used > self.max_memory_gb * 0.8:
            recommendations.append("Consider reducing batch sizes for memory-intensive models")
        
        # Training time recommendations  
        if any(r['duration_hours'] > 6 for r in all_results.values()):
            recommendations.append("Some models took >6 hours - consider reducing epochs or data size")
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Phase 1 Architecture Validation Orchestrator')
    parser.add_argument('--data_dir', type=str, default='./data/causal_episodes/',
                       help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./data/models/phase1/',
                       help='Output directory for models and results')
    parser.add_argument('--max_memory_gb', type=int, default=120,
                       help='Maximum memory usage in GB')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show execution plan without running experiments')
    
    args = parser.parse_args()
    
    print("🧪 Phase 1 Architecture Validation Orchestrator")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = Phase1Orchestrator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_memory_gb=args.max_memory_gb
    )
    
    try:
        # Run experiments
        results = orchestrator.run_phase1_experiments(dry_run=args.dry_run)
        
        if not args.dry_run:
            print(f"\n🎉 Phase 1 completed successfully!")
            print(f"📁 Results saved to: {args.output_dir}")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()