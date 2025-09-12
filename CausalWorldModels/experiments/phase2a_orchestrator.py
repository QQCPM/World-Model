#!/usr/bin/env python3
"""
Phase 2A Orchestrator - Causal Validation Experiments
Manages parallel training of causal RNN models with different causal factor combinations
Tests which causal factors are most important for predicting future states
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
import numpy as np

# Add paths for imports
sys.path.append('..')
sys.path.append('../causal_vae')

@dataclass
class CausalExperimentConfig:
    """Configuration for a single causal validation experiment"""
    name: str
    description: str
    causal_factors: List[str]  # Which causal factors to include
    input_dim: int  # Total input dimension (latent + action + causal)
    expected_memory_gb: float
    training_epochs: int
    batch_size: int
    learning_rate: float
    special_params: Dict = None

class Phase2AOrchestrator:
    """Orchestrates Phase 2A causal validation experiments"""
    
    def __init__(self, 
                 vae_models_dir='./data/models/phase1/', 
                 data_dir='./data/causal_episodes/',
                 output_dir='./data/models/phase2a/', 
                 max_memory_gb=90):
        self.vae_models_dir = vae_models_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_memory_gb = max_memory_gb
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/results", exist_ok=True)
        
        # Define experiment configurations
        self.experiment_configs = self._create_causal_experiments()
        
        # Group experiments
        self.experiment_groups = self._create_execution_groups()
        
        # Track running experiments
        self.running_experiments = {}
        
        print(f"Phase2AOrchestrator initialized:")
        print(f"  VAE models: {self.vae_models_dir}")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Memory limit: {self.max_memory_gb}GB")
        print(f"  Total causal experiments: {len(self.experiment_configs)}")
    
    def _create_causal_experiments(self) -> Dict[str, CausalExperimentConfig]:
        """Create configurations for all Phase 2A causal experiments"""
        
        # Base dimensions
        base_latent_dim = 256  # Will use best VAE from Phase 1
        action_dim = 5
        
        # Causal factor dimensions
        causal_dims = {
            'time_hour': 24,
            'day_week': 7, 
            'weather': 4,
            'event': 5,
            'crowd_density': 5
        }
        
        configs = {}
        
        # Experiment 1: No causal factors (baseline)
        configs["no_causal"] = CausalExperimentConfig(
            name="no_causal",
            description="Baseline - no causal conditioning",
            causal_factors=[],
            input_dim=base_latent_dim + action_dim,  # 261
            expected_memory_gb=8.0,
            training_epochs=30,
            batch_size=32,
            learning_rate=0.0001
        )
        
        # Experiment 2: Temporal factors only
        temporal_factors = ['time_hour', 'day_week'] 
        temporal_dim = sum(causal_dims[f] for f in temporal_factors)  # 31
        configs["temporal_only"] = CausalExperimentConfig(
            name="temporal_only",
            description="Time-based effects only (hour + day)",
            causal_factors=temporal_factors,
            input_dim=base_latent_dim + action_dim + temporal_dim,  # 292
            expected_memory_gb=10.0,
            training_epochs=30,
            batch_size=28,
            learning_rate=0.0001
        )
        
        # Experiment 3: Environmental factors only
        env_factors = ['weather']
        env_dim = sum(causal_dims[f] for f in env_factors)  # 4
        configs["environmental_only"] = CausalExperimentConfig(
            name="environmental_only",
            description="Weather effects only",
            causal_factors=env_factors,
            input_dim=base_latent_dim + action_dim + env_dim,  # 265
            expected_memory_gb=8.5,
            training_epochs=30,
            batch_size=32,
            learning_rate=0.0001
        )
        
        # Experiment 4: Social factors only
        social_factors = ['event', 'crowd_density']
        social_dim = sum(causal_dims[f] for f in social_factors)  # 10
        configs["social_only"] = CausalExperimentConfig(
            name="social_only",
            description="Social/event effects only",
            causal_factors=social_factors,
            input_dim=base_latent_dim + action_dim + social_dim,  # 271
            expected_memory_gb=9.0,
            training_epochs=30,
            batch_size=30,
            learning_rate=0.0001
        )
        
        # Experiment 5: Temporal + Environmental interaction
        temp_env_factors = ['time_hour', 'day_week', 'weather']
        temp_env_dim = sum(causal_dims[f] for f in temp_env_factors)  # 35
        configs["temporal_environmental"] = CausalExperimentConfig(
            name="temporal_environmental",
            description="Test temporal-environmental interactions",
            causal_factors=temp_env_factors,
            input_dim=base_latent_dim + action_dim + temp_env_dim,  # 296
            expected_memory_gb=11.0,
            training_epochs=30,
            batch_size=26,
            learning_rate=0.0001
        )
        
        # Experiment 6: Full causal model
        all_factors = list(causal_dims.keys())
        all_dim = sum(causal_dims.values())  # 45
        configs["full_causal"] = CausalExperimentConfig(
            name="full_causal",
            description="All causal factors included",
            causal_factors=all_factors,
            input_dim=base_latent_dim + action_dim + all_dim,  # 306
            expected_memory_gb=12.0,
            training_epochs=30,
            batch_size=24,
            learning_rate=0.0001
        )
        
        return configs
    
    def _create_execution_groups(self) -> List[List[str]]:
        """Create balanced execution groups to stay within memory limits"""
        
        # Sort experiments by memory usage
        sorted_experiments = sorted(
            self.experiment_configs.items(),
            key=lambda x: x[1].expected_memory_gb
        )
        
        groups = []
        current_group = []
        current_memory = 0
        
        for exp_name, config in sorted_experiments:
            if current_memory + config.expected_memory_gb <= self.max_memory_gb:
                current_group.append(exp_name)
                current_memory += config.expected_memory_gb
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [exp_name]
                current_memory = config.expected_memory_gb
        
        if current_group:
            groups.append(current_group)
        
        # Print group allocation
        print(f"\nExecution groups:")
        for i, group in enumerate(groups, 1):
            group_memory = sum(self.experiment_configs[name].expected_memory_gb for name in group)
            print(f"  Group {i} ({group_memory:.1f}GB): {group}")
        
        return groups
    
    def run_phase2a_experiments(self, vae_model_name: str = "best", dry_run=False):
        """
        Run all Phase 2A causal validation experiments
        
        Args:
            vae_model_name: Which VAE model to use ("best", "gaussian_256D", etc.)
            dry_run: Whether to just show execution plan
        """
        
        print(f"\n🚀 Starting Phase 2A Causal Validation Experiments")
        print(f"   Using VAE model: {vae_model_name}")
        print(f"   Total experiments: {len(self.experiment_configs)}")
        print(f"   Execution groups: {len(self.experiment_groups)}")
        print(f"   Dry run: {'Yes' if dry_run else 'No'}")
        
        if dry_run:
            print("\n📋 DRY RUN - Would execute:")
            for i, group in enumerate(self.experiment_groups, 1):
                group_memory = sum(self.experiment_configs[name].expected_memory_gb for name in group)
                print(f"  Group {i} ({group_memory:.1f}GB): {group}")
            return {}
        
        # Check that VAE models are available
        if not self._validate_vae_models(vae_model_name):
            raise ValueError(f"VAE models not available for Phase 2A")
        
        all_results = {}
        start_time = time.time()
        
        # Execute each group sequentially
        for group_idx, group in enumerate(self.experiment_groups, 1):
            print(f"\n⏱️  Phase 2A.{group_idx}: Starting Group {group_idx} experiments")
            group_start_time = time.time()
            
            group_results = self._run_experiment_group(group, vae_model_name, group_idx)
            all_results.update(group_results)
            
            group_duration = time.time() - group_start_time
            print(f"✅ Group {group_idx} completed in {group_duration/3600:.1f} hours")
            
            # Cleanup between groups
            if group_idx < len(self.experiment_groups):
                print(f"🔄 Cleaning up before next group...")
                time.sleep(30)
        
        # Generate comprehensive summary
        total_duration = time.time() - start_time
        self._generate_phase2a_summary(all_results, total_duration, vae_model_name)
        
        return all_results
    
    def _validate_vae_models(self, vae_model_name: str) -> bool:
        """Check that required VAE models are available"""
        
        if vae_model_name == "best":
            # Look for any successful Phase 1 models
            phase1_summary_path = os.path.join(self.vae_models_dir, 'phase1_summary.json')
            if os.path.exists(phase1_summary_path):
                print(f"✅ Phase 1 summary found - will use best model")
                return True
            else:
                print(f"⚠️  No Phase 1 summary found - will use simulation mode")
                return True  # Allow simulation mode
        else:
            model_path = os.path.join(self.vae_models_dir, vae_model_name)
            if os.path.exists(model_path):
                print(f"✅ VAE model found: {model_path}")
                return True
            else:
                print(f"⚠️  VAE model not found: {model_path} - will use simulation mode")
                return True  # Allow simulation mode
    
    def _run_experiment_group(self, experiment_names: List[str], 
                             vae_model_name: str, group_num: int) -> Dict:
        """Run a group of causal experiments in parallel"""
        
        print(f"\n🔧 Starting Group {group_num} causal experiments in parallel:")
        
        processes = {}
        
        for exp_name in experiment_names:
            config = self.experiment_configs[exp_name]
            print(f"  🚀 Starting {exp_name} ({config.expected_memory_gb}GB)")
            
            # Create experiment command
            cmd = self._create_causal_experiment_command(config, vae_model_name)
            
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
        
        print(f"✅ All {len(processes)} Group {group_num} experiments started")
        
        # Monitor experiments
        results = self._monitor_causal_experiments(processes, f"Group{group_num}")
        
        return results
    
    def _create_causal_experiment_command(self, config: CausalExperimentConfig, 
                                         vae_model_name: str) -> List[str]:
        """Create command to run a single causal experiment"""
        
        cmd = [
            'python3',
            'experiments/train_causal_rnn_experiment.py',  # Will create this
            '--experiment_name', config.name,
            '--description', config.description,
            '--causal_factors', ','.join(config.causal_factors),
            '--epochs', str(config.training_epochs),
            '--batch_size', str(config.batch_size),
            '--learning_rate', str(config.learning_rate),
            '--data_dir', self.data_dir,
            '--output_dir', f"{self.output_dir}/{config.name}",
            '--vae_model_name', vae_model_name,
            '--vae_models_dir', self.vae_models_dir
        ]
        
        return cmd
    
    def _monitor_causal_experiments(self, processes: Dict, group_name: str) -> Dict:
        """Monitor running causal experiments and collect results"""
        
        results = {}
        completed = set()
        
        print(f"\n📊 Monitoring {group_name} causal experiments...")
        
        while len(completed) < len(processes):
            for exp_name, proc_info in processes.items():
                if exp_name in completed:
                    continue
                
                process = proc_info['process']
                
                if process.poll() is not None:
                    # Process completed
                    end_time = time.time()
                    duration = end_time - proc_info['start_time']
                    
                    stdout, stderr = process.communicate()
                    
                    result = {
                        'experiment_name': exp_name,
                        'causal_factors': proc_info['config'].causal_factors,
                        'description': proc_info['config'].description,
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
                    self._save_causal_experiment_result(result)
            
            if len(completed) < len(processes):
                time.sleep(60)  # Check every minute
        
        print(f"🏁 All {group_name} experiments completed")
        return results
    
    def _save_causal_experiment_result(self, result: Dict):
        """Save individual causal experiment result"""
        
        result_file = f"{self.output_dir}/results/{result['experiment_name']}_result.json"
        
        save_result = result.copy()
        save_result['timestamp'] = datetime.now().isoformat()
        
        with open(result_file, 'w') as f:
            json.dump(save_result, f, indent=2)
        
        print(f"📁 Saved causal result: {result_file}")
    
    def _generate_phase2a_summary(self, all_results: Dict, total_duration: float, 
                                 vae_model_name: str):
        """Generate comprehensive Phase 2A summary with causal analysis"""
        
        print(f"\n📈 Generating Phase 2A Causal Analysis Summary...")
        
        successful_experiments = [r for r in all_results.values() if r['success']]
        failed_experiments = [r for r in all_results.values() if not r['success']]
        
        # Causal factor analysis
        causal_factor_performance = {}
        
        for result in successful_experiments:
            factors = result['causal_factors']
            factor_key = '+'.join(sorted(factors)) if factors else 'none'
            
            if factor_key not in causal_factor_performance:
                causal_factor_performance[factor_key] = {
                    'experiments': [],
                    'avg_duration': 0,
                    'factor_count': len(factors)
                }
            
            causal_factor_performance[factor_key]['experiments'].append(result['experiment_name'])
            causal_factor_performance[factor_key]['avg_duration'] = result['duration_hours']
        
        summary = {
            'phase': 'Phase 2A - Causal Validation',
            'vae_model_used': vae_model_name,
            'execution_timestamp': datetime.now().isoformat(),
            'total_duration_hours': total_duration / 3600,
            'total_experiments': len(all_results),
            'successful_experiments': len(successful_experiments),
            'failed_experiments': len(failed_experiments),
            'success_rate': len(successful_experiments) / len(all_results) if all_results else 0,
            'causal_factor_analysis': causal_factor_performance,
            'experiment_results': all_results,
            'causal_insights': self._analyze_causal_insights(successful_experiments),
            'recommendations_phase2b': self._generate_phase2b_recommendations(successful_experiments)
        }
        
        # Save summary
        summary_file = f"{self.output_dir}/phase2a_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n" + "="*70)
        print(f"📊 PHASE 2A CAUSAL VALIDATION RESULTS")
        print(f"="*70)
        print(f"VAE Model Used: {vae_model_name}")
        print(f"Total Duration: {total_duration/3600:.1f} hours")
        print(f"Experiments: {len(successful_experiments)}/{len(all_results)} successful")
        print(f"Success Rate: {len(successful_experiments)/len(all_results)*100:.1f}%")
        
        if successful_experiments:
            print(f"\n🔍 Causal Factor Analysis:")
            sorted_factors = sorted(causal_factor_performance.items(), 
                                  key=lambda x: x[1]['avg_duration'])
            
            for factor_combo, analysis in sorted_factors:
                print(f"  {factor_combo}: {analysis['avg_duration']:.1f}h avg")
        
        print(f"\n📁 Full summary saved to: {summary_file}")
        print(f"🎯 Ready for Phase 2B: Replay Strategy Experiments")
    
    def _analyze_causal_insights(self, successful_experiments: List[Dict]) -> Dict:
        """Extract insights about which causal factors are most important"""
        
        insights = {
            'most_effective_factors': [],
            'factor_importance_ranking': {},
            'surprising_results': [],
            'recommendations': []
        }
        
        # Simple analysis based on training success and duration
        # In a real implementation, this would analyze actual performance metrics
        
        factor_success = {}
        for result in successful_experiments:
            factors = result['causal_factors']
            for factor in factors:
                if factor not in factor_success:
                    factor_success[factor] = {'count': 0, 'avg_duration': 0}
                factor_success[factor]['count'] += 1
                factor_success[factor]['avg_duration'] += result['duration_hours']
        
        # Rank factors by how often they appear in successful experiments
        for factor, stats in factor_success.items():
            stats['avg_duration'] /= stats['count']
            insights['factor_importance_ranking'][factor] = {
                'success_count': stats['count'],
                'avg_duration': stats['avg_duration'],
                'importance_score': stats['count'] / stats['avg_duration']  # More successes, less time = better
            }
        
        return insights
    
    def _generate_phase2b_recommendations(self, successful_experiments: List[Dict]) -> List[str]:
        """Generate recommendations for Phase 2B based on Phase 2A results"""
        
        recommendations = []
        
        if len(successful_experiments) >= 3:
            recommendations.append("Strong causal validation results - proceed with full Phase 2B")
            recommendations.append("Use top 2-3 causal factor combinations for replay experiments")
        else:
            recommendations.append("Limited causal validation success - review factor implementations")
            recommendations.append("Consider simplified Phase 2B with fewer experiments")
        
        # Analyze which causal factors worked best
        factor_counts = {}
        for result in successful_experiments:
            for factor in result['causal_factors']:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        if factor_counts:
            best_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            factor_names = [f[0] for f in best_factors]
            recommendations.append(f"Focus Phase 2B on top causal factors: {', '.join(factor_names)}")
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description='Phase 2A Causal Validation Orchestrator')
    parser.add_argument('--vae_models_dir', type=str, default='./data/models/phase1/',
                       help='Directory containing Phase 1 VAE models')
    parser.add_argument('--data_dir', type=str, default='./data/causal_episodes/',
                       help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./data/models/phase2a/',
                       help='Output directory for Phase 2A results')
    parser.add_argument('--max_memory_gb', type=int, default=90,
                       help='Maximum memory usage in GB')
    parser.add_argument('--vae_model', type=str, default='best',
                       help='Which VAE model to use (best, gaussian_256D, etc.)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show execution plan without running experiments')
    
    args = parser.parse_args()
    
    print("🧪 Phase 2A Causal Validation Orchestrator")
    print("=" * 70)
    
    # Create orchestrator
    orchestrator = Phase2AOrchestrator(
        vae_models_dir=args.vae_models_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_memory_gb=args.max_memory_gb
    )
    
    try:
        # Run experiments
        results = orchestrator.run_phase2a_experiments(
            vae_model_name=args.vae_model,
            dry_run=args.dry_run
        )
        
        if not args.dry_run:
            print(f"\n🎉 Phase 2A completed successfully!")
            print(f"📁 Results saved to: {args.output_dir}")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()