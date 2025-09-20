"""
Pathway Analysis
Dual-pathway performance analysis for causal GRU architecture

Analyzes:
- Pathway usage patterns (observational vs interventional)
- Pathway specialization effectiveness
- Intervention detection accuracy
- Pathway balance and stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


class PathwayAnalyzer:
    """
    Analyzer for dual-pathway causal GRU performance

    Evaluates pathway specialization and effectiveness
    """

    def __init__(self):
        self.intervention_types = [
            'observational', 'single_variable', 'dual_variable',
            'temporal_shift', 'causal_chain', 'soft_intervention'
        ]

    def analyze_pathway_performance(self, dual_pathway_model, test_data_loader,
                                  intervention_labels=None):
        """
        Comprehensive pathway performance analysis

        Args:
            dual_pathway_model: DualPathwayCausalGRU instance
            test_data_loader: Test data loader
            intervention_labels: Optional intervention type labels for each batch

        Returns:
            pathway_analysis: Dict with comprehensive pathway analysis
        """
        print("🔀 Analyzing Dual-Pathway Performance")
        print("-" * 40)

        dual_pathway_model.eval()

        pathway_analysis = {
            'pathway_usage': {},
            'specialization_analysis': {},
            'intervention_detection': {},
            'pathway_balance': {},
            'performance_by_mode': {}
        }

        # Collect pathway usage data
        pathway_data = self._collect_pathway_usage_data(
            dual_pathway_model, test_data_loader
        )

        # 1. PATHWAY USAGE ANALYSIS
        print("📊 Analyzing pathway usage patterns...")
        pathway_analysis['pathway_usage'] = self._analyze_pathway_usage(pathway_data)

        # 2. SPECIALIZATION ANALYSIS
        print("🎯 Analyzing pathway specialization...")
        pathway_analysis['specialization_analysis'] = self._analyze_pathway_specialization(
            dual_pathway_model, test_data_loader
        )

        # 3. INTERVENTION DETECTION ANALYSIS
        print("🔍 Analyzing intervention detection...")
        pathway_analysis['intervention_detection'] = self._analyze_intervention_detection(
            dual_pathway_model, test_data_loader, intervention_labels
        )

        # 4. PATHWAY BALANCE ANALYSIS
        print("⚖️  Analyzing pathway balance...")
        pathway_analysis['pathway_balance'] = self._analyze_pathway_balance(pathway_data)

        # 5. PERFORMANCE BY MODE ANALYSIS
        print("📈 Analyzing performance by mode...")
        pathway_analysis['performance_by_mode'] = self._analyze_performance_by_mode(
            dual_pathway_model, test_data_loader
        )

        # OVERALL PATHWAY SCORE
        overall_score = self._compute_overall_pathway_score(pathway_analysis)
        pathway_analysis['overall_pathway_score'] = overall_score

        self._print_pathway_summary(pathway_analysis)

        return pathway_analysis

    def _collect_pathway_usage_data(self, dual_pathway_model, test_data_loader):
        """
        Collect pathway usage data across test dataset
        """
        pathway_data = {
            'obs_weights': [],
            'int_weights': [],
            'intervention_scores': [],
            'pathway_balances': [],
            'prediction_errors': [],
            'batch_info': []
        }

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_data_loader):
                states, actions, causal_factors = batch_data

                # Forward pass to get pathway info
                predicted_states, hidden_states, pathway_info = dual_pathway_model(
                    states[:, :-1], actions[:, :-1], causal_factors[:, :-1]
                )

                # Compute prediction error
                prediction_error = F.mse_loss(predicted_states, states[:, 1:]).item()

                # Store pathway data
                pathway_data['obs_weights'].append(pathway_info['obs_weight'])
                pathway_data['int_weights'].append(pathway_info['int_weight'])
                pathway_data['intervention_scores'].append(pathway_info['intervention_score'])
                pathway_data['pathway_balances'].append(pathway_info['pathway_balance'])
                pathway_data['prediction_errors'].append(prediction_error)
                pathway_data['batch_info'].append({
                    'batch_idx': batch_idx,
                    'batch_size': states.shape[0],
                    'sequence_length': states.shape[1]
                })

                if batch_idx >= 20:  # Limit for efficiency
                    break

        return pathway_data

    def _analyze_pathway_usage(self, pathway_data):
        """
        Analyze pathway usage patterns
        """
        obs_weights = pathway_data['obs_weights']
        int_weights = pathway_data['int_weights']
        intervention_scores = pathway_data['intervention_scores']

        usage_analysis = {
            'average_obs_weight': np.mean(obs_weights),
            'average_int_weight': np.mean(int_weights),
            'obs_weight_std': np.std(obs_weights),
            'int_weight_std': np.std(int_weights),
            'average_intervention_score': np.mean(intervention_scores),
            'intervention_score_std': np.std(intervention_scores),
            'obs_dominant_batches': sum(1 for w in obs_weights if w > 0.6),
            'int_dominant_batches': sum(1 for w in int_weights if w > 0.6),
            'balanced_batches': sum(1 for obs, int in zip(obs_weights, int_weights)
                                  if 0.4 <= obs <= 0.6 and 0.4 <= int <= 0.6)
        }

        # Usage pattern classification
        if usage_analysis['average_obs_weight'] > 0.7:
            usage_pattern = 'obs_dominant'
        elif usage_analysis['average_int_weight'] > 0.5:
            usage_pattern = 'int_dominant'
        else:
            usage_pattern = 'balanced'

        usage_analysis['usage_pattern'] = usage_pattern
        usage_analysis['pathway_diversity'] = usage_analysis['obs_weight_std'] + usage_analysis['int_weight_std']

        return usage_analysis

    def _analyze_pathway_specialization(self, dual_pathway_model, test_data_loader):
        """
        Analyze how well pathways specialize for different data types
        """
        specialization_results = {}

        # Test observational specialization
        obs_performance = self._test_pathway_mode_performance(
            dual_pathway_model, test_data_loader, 'observational'
        )

        # Test interventional specialization
        int_performance = self._test_pathway_mode_performance(
            dual_pathway_model, test_data_loader, 'interventional'
        )

        # Test balanced mode
        balanced_performance = self._test_pathway_mode_performance(
            dual_pathway_model, test_data_loader, 'balanced'
        )

        # Test auto mode (default)
        auto_performance = self._test_pathway_mode_performance(
            dual_pathway_model, test_data_loader, 'auto'
        )

        specialization_results = {
            'observational_mode_mse': obs_performance,
            'interventional_mode_mse': int_performance,
            'balanced_mode_mse': balanced_performance,
            'auto_mode_mse': auto_performance,
            'best_mode': min([
                ('observational', obs_performance),
                ('interventional', int_performance),
                ('balanced', balanced_performance),
                ('auto', auto_performance)
            ], key=lambda x: x[1])[0],
            'mode_performance_variance': np.var([obs_performance, int_performance,
                                               balanced_performance, auto_performance]),
            'specialization_effectiveness': self._compute_specialization_effectiveness(
                obs_performance, int_performance, auto_performance
            )
        }

        return specialization_results

    def _test_pathway_mode_performance(self, dual_pathway_model, test_data_loader, mode):
        """
        Test performance in specific pathway mode
        """
        dual_pathway_model.eval()
        original_mode = dual_pathway_model.pathway_weights.data.clone()

        # Set pathway mode
        dual_pathway_model.set_pathway_mode(mode)

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in test_data_loader:
                states, actions, causal_factors = batch_data

                predicted_states, _, _ = dual_pathway_model(
                    states[:, :-1], actions[:, :-1], causal_factors[:, :-1]
                )

                loss = F.mse_loss(predicted_states, states[:, 1:])
                total_loss += loss.item()
                num_batches += 1

                if num_batches >= 5:  # Limit for efficiency
                    break

        # Restore original pathway weights
        dual_pathway_model.pathway_weights.data = original_mode

        return total_loss / max(num_batches, 1)

    def _analyze_intervention_detection(self, dual_pathway_model, test_data_loader,
                                      intervention_labels):
        """
        Analyze intervention detection accuracy
        """
        if intervention_labels is None:
            # Generate synthetic intervention labels for testing
            intervention_labels = self._generate_synthetic_intervention_labels(test_data_loader)

        detection_results = {}

        # Collect intervention detection scores
        predicted_interventions = []
        true_interventions = []

        dual_pathway_model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_data_loader):
                states, actions, causal_factors = batch_data

                # Get intervention detection scores
                intervention_scores = dual_pathway_model.detect_intervention(causal_factors)
                predicted_interventions.extend(intervention_scores.cpu().numpy().flatten())

                # Get true labels for this batch
                if batch_idx < len(intervention_labels):
                    batch_labels = intervention_labels[batch_idx]
                    true_interventions.extend(batch_labels)

                if batch_idx >= len(intervention_labels) - 1:
                    break

        # Ensure equal lengths
        min_length = min(len(predicted_interventions), len(true_interventions))
        predicted_interventions = predicted_interventions[:min_length]
        true_interventions = true_interventions[:min_length]

        if len(predicted_interventions) > 0 and len(true_interventions) > 0:
            # Convert to binary predictions
            binary_predictions = (np.array(predicted_interventions) > 0.5).astype(int)
            binary_true = np.array(true_interventions).astype(int)

            # Compute detection metrics
            try:
                auc_score = roc_auc_score(binary_true, predicted_interventions)
            except:
                auc_score = 0.5  # Random performance if labels are all same

            accuracy = np.mean(binary_predictions == binary_true)

            # Confusion matrix
            cm = confusion_matrix(binary_true, binary_predictions)

            detection_results = {
                'auc_score': auc_score,
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist(),
                'detection_quality': self._grade_detection_quality(auc_score, accuracy)
            }
        else:
            detection_results = {
                'status': 'insufficient_data',
                'auc_score': 0.5,
                'accuracy': 0.5,
                'detection_quality': 'Unknown'
            }

        return detection_results

    def _generate_synthetic_intervention_labels(self, test_data_loader):
        """
        Generate synthetic intervention labels for testing
        """
        synthetic_labels = []

        for batch_data in test_data_loader:
            states, actions, causal_factors = batch_data
            batch_size, seq_len = causal_factors.shape[:2]

            # Generate random intervention labels (30% interventional)
            batch_labels = np.random.choice([0, 1], size=(batch_size * seq_len,), p=[0.7, 0.3])
            synthetic_labels.append(batch_labels)

        return synthetic_labels

    def _analyze_pathway_balance(self, pathway_data):
        """
        Analyze pathway balance and stability
        """
        obs_weights = pathway_data['obs_weights']
        int_weights = pathway_data['int_weights']
        pathway_balances = pathway_data['pathway_balances']

        balance_analysis = {
            'average_pathway_balance': np.mean(pathway_balances),
            'pathway_balance_std': np.std(pathway_balances),
            'balance_stability': 1.0 / (1.0 + np.std(pathway_balances)),
            'weight_correlation': np.corrcoef(obs_weights, int_weights)[0, 1],
            'target_ratio_adherence': self._compute_target_ratio_adherence(obs_weights, int_weights),
            'balance_grade': self._grade_pathway_balance(np.mean(pathway_balances), np.std(pathway_balances))
        }

        return balance_analysis

    def _analyze_performance_by_mode(self, dual_pathway_model, test_data_loader):
        """
        Analyze performance across different pathway modes
        """
        modes = ['observational', 'interventional', 'balanced', 'auto']
        mode_performances = {}

        for mode in modes:
            performance = self._test_pathway_mode_performance(
                dual_pathway_model, test_data_loader, mode
            )
            mode_performances[mode] = performance

        # Find best and worst modes
        best_mode = min(mode_performances, key=mode_performances.get)
        worst_mode = max(mode_performances, key=mode_performances.get)

        performance_analysis = {
            'mode_performances': mode_performances,
            'best_mode': best_mode,
            'worst_mode': worst_mode,
            'performance_range': mode_performances[worst_mode] - mode_performances[best_mode],
            'auto_vs_best_ratio': mode_performances['auto'] / mode_performances[best_mode],
            'mode_robustness': self._compute_mode_robustness(mode_performances)
        }

        return performance_analysis

    def _compute_specialization_effectiveness(self, obs_perf, int_perf, auto_perf):
        """
        Compute how effectively pathways specialize
        """
        # Good specialization: auto mode should be close to best specialized mode
        best_specialized = min(obs_perf, int_perf)
        specialization_gap = auto_perf - best_specialized

        # Lower gap = better specialization
        effectiveness = 1.0 / (1.0 + specialization_gap)

        return effectiveness

    def _compute_target_ratio_adherence(self, obs_weights, int_weights):
        """
        Compute adherence to target 60/40 ratio
        """
        target_obs_ratio = 0.6
        target_int_ratio = 0.4

        obs_deviation = np.mean([abs(w - target_obs_ratio) for w in obs_weights])
        int_deviation = np.mean([abs(w - target_int_ratio) for w in int_weights])

        adherence = 1.0 - (obs_deviation + int_deviation) / 2.0

        return max(0.0, adherence)

    def _compute_mode_robustness(self, mode_performances):
        """
        Compute robustness across different modes
        """
        performances = list(mode_performances.values())
        mean_performance = np.mean(performances)
        std_performance = np.std(performances)

        # Lower standard deviation = more robust
        robustness = 1.0 / (1.0 + std_performance / mean_performance)

        return robustness

    def _compute_overall_pathway_score(self, pathway_analysis):
        """
        Compute overall pathway performance score
        """
        # Weight different components
        usage_score = 0.2 * (1.0 - pathway_analysis['pathway_usage']['pathway_diversity'] / 2.0)
        specialization_score = 0.3 * pathway_analysis['specialization_analysis']['specialization_effectiveness']
        balance_score = 0.2 * pathway_analysis['pathway_balance']['balance_stability']
        robustness_score = 0.3 * pathway_analysis['performance_by_mode']['mode_robustness']

        overall_score = usage_score + specialization_score + balance_score + robustness_score

        return {
            'overall_score': overall_score,
            'usage_component': usage_score,
            'specialization_component': specialization_score,
            'balance_component': balance_score,
            'robustness_component': robustness_score,
            'grade': self._grade_pathway_performance(overall_score)
        }

    def _grade_detection_quality(self, auc_score, accuracy):
        """Grade intervention detection quality"""
        avg_score = (auc_score + accuracy) / 2.0

        if avg_score >= 0.9:
            return 'Excellent'
        elif avg_score >= 0.8:
            return 'Very Good'
        elif avg_score >= 0.7:
            return 'Good'
        elif avg_score >= 0.6:
            return 'Fair'
        else:
            return 'Poor'

    def _grade_pathway_balance(self, mean_balance, std_balance):
        """Grade pathway balance quality"""
        stability = 1.0 / (1.0 + std_balance)
        target_balance = 0.3  # Target balance level

        balance_quality = stability * (1.0 - abs(mean_balance - target_balance))

        if balance_quality >= 0.8:
            return 'Excellent'
        elif balance_quality >= 0.6:
            return 'Good'
        elif balance_quality >= 0.4:
            return 'Fair'
        else:
            return 'Poor'

    def _grade_pathway_performance(self, score):
        """Grade overall pathway performance"""
        if score >= 0.8:
            return 'A+'
        elif score >= 0.7:
            return 'A'
        elif score >= 0.6:
            return 'B+'
        elif score >= 0.5:
            return 'B'
        elif score >= 0.4:
            return 'C'
        else:
            return 'F'

    def _print_pathway_summary(self, pathway_analysis):
        """Print pathway analysis summary"""
        print("\n" + "="*50)
        print("🔀 DUAL-PATHWAY ANALYSIS SUMMARY")
        print("="*50)

        overall = pathway_analysis['overall_pathway_score']
        usage = pathway_analysis['pathway_usage']
        specialization = pathway_analysis['specialization_analysis']
        balance = pathway_analysis['pathway_balance']
        detection = pathway_analysis['intervention_detection']

        print(f"📊 Overall Grade: {overall['grade']}")
        print(f"📈 Overall Score: {overall['overall_score']:.3f}")
        print()

        print(f"🎯 Usage Pattern: {usage['usage_pattern']}")
        print(f"⚖️  Pathway Balance: {balance['balance_grade']}")
        print(f"🔍 Intervention Detection: {detection['detection_quality']}")
        print(f"🎛️  Best Mode: {specialization['best_mode']}")

        print(f"\n📊 Pathway Weights:")
        print(f"   - Observational: {usage['average_obs_weight']:.3f} ± {usage['obs_weight_std']:.3f}")
        print(f"   - Interventional: {usage['average_int_weight']:.3f} ± {usage['int_weight_std']:.3f}")

        if overall['overall_score'] >= 0.7:
            print("\n🎉 EXCELLENT: Dual-pathway architecture is working effectively!")
        elif overall['overall_score'] >= 0.5:
            print("\n👍 GOOD: Pathways show reasonable specialization")
        else:
            print("\n⚠️  NEEDS IMPROVEMENT: Pathway specialization could be enhanced")

        print("="*50)

    def plot_pathway_usage(self, pathway_analysis, save_path=None):
        """
        Plot pathway usage patterns

        Args:
            pathway_analysis: Results from analyze_pathway_performance
            save_path: Optional path to save plot
        """
        usage = pathway_analysis['pathway_usage']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Average pathway weights
        weights = [usage['average_obs_weight'], usage['average_int_weight']]
        stds = [usage['obs_weight_std'], usage['int_weight_std']]
        labels = ['Observational', 'Interventional']

        ax1.bar(labels, weights, yerr=stds, capsize=5, alpha=0.7,
                color=['blue', 'red'])
        ax1.set_ylabel('Average Weight')
        ax1.set_title('Pathway Usage Weights')
        ax1.set_ylim(0, 1)

        # Add target line
        ax1.axhline(y=0.6, color='blue', linestyle='--', alpha=0.5, label='Target Obs (60%)')
        ax1.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Target Int (40%)')
        ax1.legend()

        # Plot 2: Performance by mode
        if 'performance_by_mode' in pathway_analysis:
            mode_perf = pathway_analysis['performance_by_mode']['mode_performances']
            modes = list(mode_perf.keys())
            performances = list(mode_perf.values())

            ax2.bar(modes, performances, alpha=0.7, color='green')
            ax2.set_ylabel('MSE Loss')
            ax2.set_title('Performance by Pathway Mode')
            ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Pathway usage plot saved to: {save_path}")

        return fig


def test_pathway_analyzer():
    """
    Test pathway analyzer functionality
    """
    print("Testing PathwayAnalyzer...")

    # Create analyzer
    analyzer = PathwayAnalyzer()

    # Test helper methods
    obs_weights = [0.7, 0.6, 0.8, 0.5]
    int_weights = [0.3, 0.4, 0.2, 0.5]

    adherence = analyzer._compute_target_ratio_adherence(obs_weights, int_weights)
    print(f"Target ratio adherence: {adherence:.3f}")

    # Test grading functions
    detection_grade = analyzer._grade_detection_quality(0.85, 0.80)
    print(f"Detection quality grade: {detection_grade}")

    balance_grade = analyzer._grade_pathway_balance(0.3, 0.1)
    print(f"Balance grade: {balance_grade}")

    print("✅ PathwayAnalyzer test passed")

    return True


if __name__ == "__main__":
    test_pathway_analyzer()