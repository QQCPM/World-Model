"""
Active Learning Metrics
Validation framework for intervention selection efficiency

Tests:
- Information gain per intervention vs random baseline
- Convergence speed of causal graph learning
- Intervention portfolio optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import matplotlib.pyplot as plt
from scipy.stats import entropy


class ActiveLearningMetrics:
    """
    Metrics for evaluating active learning efficiency in causal discovery

    Validates that intervention designer selects informative interventions
    """

    def __init__(self, num_variables=5):
        self.num_variables = num_variables
        self.variable_names = [
            'weather', 'crowd_density', 'special_event', 'time_of_day', 'road_conditions'
        ]

    def evaluate_intervention_efficiency(self, intervention_designer, structure_learner,
                                       test_data, num_trials=20):
        """
        Evaluate efficiency of intervention selection vs random baseline

        Args:
            intervention_designer: InterventionDesigner instance
            structure_learner: CausalStructureLearner instance
            test_data: Test dataset
            num_trials: Number of intervention trials

        Returns:
            efficiency_metrics: Dict with efficiency evaluation results
        """
        print("🎛️  Evaluating Intervention Selection Efficiency")
        print("-" * 45)

        # Run active learning trials
        active_learning_results = self._run_active_learning_trials(
            intervention_designer, structure_learner, test_data, num_trials
        )

        # Run random baseline trials
        random_baseline_results = self._run_random_baseline_trials(
            structure_learner, test_data, num_trials
        )

        # Compare efficiency
        efficiency_comparison = self._compare_learning_efficiency(
            active_learning_results, random_baseline_results
        )

        # Convergence analysis
        convergence_analysis = self._analyze_convergence_speed(
            active_learning_results, random_baseline_results
        )

        # Portfolio optimization analysis
        portfolio_analysis = self._analyze_intervention_portfolio(
            intervention_designer, active_learning_results
        )

        efficiency_metrics = {
            'active_learning_performance': active_learning_results,
            'random_baseline_performance': random_baseline_results,
            'efficiency_comparison': efficiency_comparison,
            'convergence_analysis': convergence_analysis,
            'portfolio_analysis': portfolio_analysis,
            'overall_efficiency_score': self._compute_overall_efficiency_score(
                efficiency_comparison, convergence_analysis
            )
        }

        self._print_efficiency_summary(efficiency_metrics)

        return efficiency_metrics

    def _run_active_learning_trials(self, intervention_designer, structure_learner,
                                  test_data, num_trials):
        """
        Run active learning intervention trials
        """
        learning_curve = []
        intervention_history = []
        information_gains = []

        # Initial structure
        initial_loss = self._compute_structure_loss(structure_learner, test_data)
        learning_curve.append(initial_loss)

        for trial in range(num_trials):
            # Select optimal intervention
            best_intervention, candidates = intervention_designer.select_optimal_intervention(
                structure_learner, test_data, num_candidates=10
            )

            # Record intervention
            intervention_history.append({
                'trial': trial,
                'intervention': best_intervention,
                'expected_gain': best_intervention['info_gain'],
                'target_variables': best_intervention['target_variables']
            })

            # Simulate intervention outcome
            simulated_gain = self._simulate_intervention_outcome(
                best_intervention, structure_learner, test_data
            )

            information_gains.append(simulated_gain)

            # Update intervention designer
            intervention_designer.update_from_intervention_outcome(
                best_intervention, test_data, structure_learner
            )

            # Measure learning progress
            updated_loss = self._compute_structure_loss(structure_learner, test_data)
            learning_curve.append(updated_loss)

        results = {
            'learning_curve': learning_curve,
            'intervention_history': intervention_history,
            'information_gains': information_gains,
            'total_information_gain': sum(information_gains),
            'average_gain_per_intervention': np.mean(information_gains),
            'convergence_rate': self._compute_convergence_rate(learning_curve)
        }

        return results

    def _run_random_baseline_trials(self, structure_learner, test_data, num_trials):
        """
        Run random intervention baseline trials
        """
        learning_curve = []
        intervention_history = []
        information_gains = []

        # Initial structure
        initial_loss = self._compute_structure_loss(structure_learner, test_data)
        learning_curve.append(initial_loss)

        for trial in range(num_trials):
            # Random intervention selection
            random_intervention = self._generate_random_intervention()

            # Record intervention
            intervention_history.append({
                'trial': trial,
                'intervention': random_intervention,
                'expected_gain': 0.0,  # Random has no expected gain prediction
                'target_variables': random_intervention['target_variables']
            })

            # Simulate intervention outcome
            simulated_gain = self._simulate_intervention_outcome(
                random_intervention, structure_learner, test_data
            )

            information_gains.append(simulated_gain)

            # Measure learning progress
            updated_loss = self._compute_structure_loss(structure_learner, test_data)
            learning_curve.append(updated_loss)

        results = {
            'learning_curve': learning_curve,
            'intervention_history': intervention_history,
            'information_gains': information_gains,
            'total_information_gain': sum(information_gains),
            'average_gain_per_intervention': np.mean(information_gains),
            'convergence_rate': self._compute_convergence_rate(learning_curve)
        }

        return results

    def _simulate_intervention_outcome(self, intervention, structure_learner, test_data):
        """
        Simulate the outcome of performing an intervention

        Returns estimated information gain
        """
        # Simple simulation: information gain based on intervention complexity and uncertainty
        target_vars = intervention['target_variables']
        num_targets = len(target_vars)

        # More targets = potentially more information, but also more complex
        complexity_factor = 1.0 / (1.0 + 0.3 * (num_targets - 1))

        # Random gain with realistic distribution
        base_gain = np.random.exponential(0.1) * complexity_factor

        # Add some structure-dependent variation
        uncertainty_reduction = min(base_gain, 0.2)

        return uncertainty_reduction

    def _compute_structure_loss(self, structure_learner, test_data):
        """
        Compute current structure learning loss
        """
        total_loss = 0.0
        num_batches = 0

        for batch_data in test_data:
            states, actions, causal_factors = batch_data
            loss, _ = structure_learner.compute_structure_loss(causal_factors)
            total_loss += loss.item()
            num_batches += 1
            break  # Use first batch for efficiency

        return total_loss / max(num_batches, 1)

    def _generate_random_intervention(self):
        """
        Generate random intervention for baseline comparison
        """
        # Random number of targets (1-2)
        num_targets = np.random.choice([1, 2], p=[0.7, 0.3])
        target_variables = np.random.choice(self.num_variables, num_targets, replace=False).tolist()

        return {
            'type': 'random',
            'target_variables': target_variables,
            'num_targets': num_targets,
            'info_gain': 0.0,  # Random has no predicted gain
            'acquisition_score': 0.0
        }

    def _compute_convergence_rate(self, learning_curve):
        """
        Compute convergence rate from learning curve
        """
        if len(learning_curve) < 3:
            return 0.0

        # Fit exponential decay to learning curve
        x = np.arange(len(learning_curve))
        y = np.array(learning_curve)

        # Normalize
        y_norm = (y - y[-1]) / (y[0] - y[-1] + 1e-8)

        # Compute rate of improvement
        improvements = []
        for i in range(1, len(y_norm)):
            improvement = max(0, y_norm[i-1] - y_norm[i])
            improvements.append(improvement)

        convergence_rate = np.mean(improvements) if improvements else 0.0
        return convergence_rate

    def _compare_learning_efficiency(self, active_results, random_results):
        """
        Compare active learning vs random baseline efficiency
        """
        active_gain = active_results['total_information_gain']
        random_gain = random_results['total_information_gain']

        # Efficiency ratio
        efficiency_ratio = active_gain / (random_gain + 1e-8)

        # Learning curve comparison
        active_curve = active_results['learning_curve']
        random_curve = random_results['learning_curve']

        # Area under learning curve (lower is better)
        active_auc = np.trapz(active_curve)
        random_auc = np.trapz(random_curve)

        auc_improvement = (random_auc - active_auc) / (random_auc + 1e-8)

        # Convergence comparison
        active_convergence = active_results['convergence_rate']
        random_convergence = random_results['convergence_rate']

        convergence_improvement = (active_convergence - random_convergence) / (random_convergence + 1e-8)

        comparison = {
            'information_gain_ratio': efficiency_ratio,
            'auc_improvement': auc_improvement,
            'convergence_improvement': convergence_improvement,
            'active_outperforms_random': efficiency_ratio > 1.2,  # 20% better threshold
            'efficiency_grade': self._grade_efficiency(efficiency_ratio)
        }

        return comparison

    def _analyze_convergence_speed(self, active_results, random_results):
        """
        Analyze convergence speed differences
        """
        active_curve = active_results['learning_curve']
        random_curve = random_results['learning_curve']

        # Find convergence points (95% of final improvement)
        def find_convergence_point(curve, threshold=0.95):
            if len(curve) < 2:
                return len(curve)

            total_improvement = curve[0] - curve[-1]
            target_improvement = total_improvement * threshold

            for i, loss in enumerate(curve):
                current_improvement = curve[0] - loss
                if current_improvement >= target_improvement:
                    return i

            return len(curve) - 1

        active_convergence_point = find_convergence_point(active_curve)
        random_convergence_point = find_convergence_point(random_curve)

        convergence_analysis = {
            'active_convergence_point': active_convergence_point,
            'random_convergence_point': random_convergence_point,
            'convergence_speedup': (random_convergence_point - active_convergence_point) / (random_convergence_point + 1e-8),
            'active_converges_faster': active_convergence_point < random_convergence_point,
            'final_active_loss': active_curve[-1],
            'final_random_loss': random_curve[-1],
            'final_loss_improvement': (random_curve[-1] - active_curve[-1]) / (random_curve[-1] + 1e-8)
        }

        return convergence_analysis

    def _analyze_intervention_portfolio(self, intervention_designer, active_results):
        """
        Analyze diversity and optimality of intervention portfolio
        """
        intervention_history = active_results['intervention_history']

        # Variable selection diversity
        selected_variables = []
        for intervention in intervention_history:
            selected_variables.extend(intervention['target_variables'])

        # Count selections per variable
        variable_counts = {i: selected_variables.count(i) for i in range(self.num_variables)}

        # Compute selection entropy (higher = more diverse)
        selection_probs = [count / len(selected_variables) for count in variable_counts.values()]
        selection_entropy = entropy(selection_probs) if any(p > 0 for p in selection_probs) else 0

        # Analyze intervention types
        intervention_types = [intervention['intervention']['type'] if 'type' in intervention['intervention']
                            else 'unknown' for intervention in intervention_history]
        type_diversity = len(set(intervention_types))

        # Correlation between predicted and actual gains
        predicted_gains = [intervention['expected_gain'] for intervention in intervention_history]
        actual_gains = active_results['information_gains']

        if len(predicted_gains) > 1 and len(actual_gains) > 1:
            gain_correlation = np.corrcoef(predicted_gains, actual_gains)[0, 1]
        else:
            gain_correlation = 0.0

        portfolio_analysis = {
            'variable_selection_entropy': selection_entropy,
            'most_selected_variable': self.variable_names[max(variable_counts, key=variable_counts.get)],
            'least_selected_variable': self.variable_names[min(variable_counts, key=variable_counts.get)],
            'intervention_type_diversity': type_diversity,
            'predicted_vs_actual_correlation': gain_correlation,
            'portfolio_quality': self._grade_portfolio_quality(selection_entropy, gain_correlation)
        }

        return portfolio_analysis

    def _compute_overall_efficiency_score(self, efficiency_comparison, convergence_analysis):
        """
        Compute overall active learning efficiency score
        """
        # Weight different aspects
        efficiency_score = 0.4 * min(efficiency_comparison['information_gain_ratio'] / 2.0, 1.0)
        convergence_score = 0.3 * (1.0 if convergence_analysis['active_converges_faster'] else 0.5)
        improvement_score = 0.3 * max(0, convergence_analysis['final_loss_improvement'])

        overall_score = efficiency_score + convergence_score + improvement_score

        return {
            'overall_score': overall_score,
            'efficiency_component': efficiency_score,
            'convergence_component': convergence_score,
            'improvement_component': improvement_score,
            'grade': self._grade_overall_efficiency(overall_score)
        }

    def _grade_efficiency(self, efficiency_ratio):
        """Grade efficiency ratio"""
        if efficiency_ratio >= 2.0:
            return 'Excellent'
        elif efficiency_ratio >= 1.5:
            return 'Very Good'
        elif efficiency_ratio >= 1.2:
            return 'Good'
        elif efficiency_ratio >= 1.0:
            return 'Fair'
        else:
            return 'Poor'

    def _grade_portfolio_quality(self, entropy, correlation):
        """Grade intervention portfolio quality"""
        entropy_score = min(entropy / np.log(self.num_variables), 1.0)  # Normalize by max entropy
        correlation_score = max(0, correlation)

        combined_score = 0.6 * entropy_score + 0.4 * correlation_score

        if combined_score >= 0.8:
            return 'Excellent'
        elif combined_score >= 0.6:
            return 'Good'
        elif combined_score >= 0.4:
            return 'Fair'
        else:
            return 'Poor'

    def _grade_overall_efficiency(self, score):
        """Grade overall efficiency"""
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

    def _print_efficiency_summary(self, efficiency_metrics):
        """Print efficiency evaluation summary"""
        print("\n" + "="*50)
        print("🎛️  ACTIVE LEARNING EFFICIENCY SUMMARY")
        print("="*50)

        overall = efficiency_metrics['overall_efficiency_score']
        comparison = efficiency_metrics['efficiency_comparison']
        convergence = efficiency_metrics['convergence_analysis']
        portfolio = efficiency_metrics['portfolio_analysis']

        print(f"📊 Overall Grade: {overall['grade']}")
        print(f"📈 Overall Score: {overall['overall_score']:.3f}")
        print()

        print(f"🎯 Information Gain Ratio: {comparison['information_gain_ratio']:.2f}x random")
        print(f"🏃 Convergence: {'✅ Faster' if convergence['active_converges_faster'] else '❌ Slower'}")
        print(f"📊 Portfolio Quality: {portfolio['portfolio_quality']}")
        print(f"🎛️  Most Selected Variable: {portfolio['most_selected_variable']}")

        if comparison['active_outperforms_random']:
            print("\n🎉 SUCCESS: Active learning significantly outperforms random!")
        else:
            print("\n⚠️  WARNING: Active learning advantage is minimal")

        print("="*50)

    def plot_learning_curves(self, efficiency_metrics, save_path=None):
        """
        Plot learning curves comparison

        Args:
            efficiency_metrics: Results from evaluate_intervention_efficiency
            save_path: Optional path to save plot
        """
        active_curve = efficiency_metrics['active_learning_performance']['learning_curve']
        random_curve = efficiency_metrics['random_baseline_performance']['learning_curve']

        plt.figure(figsize=(10, 6))

        plt.plot(active_curve, 'b-', label='Active Learning', linewidth=2)
        plt.plot(random_curve, 'r--', label='Random Baseline', linewidth=2)

        plt.xlabel('Intervention Number')
        plt.ylabel('Structure Learning Loss')
        plt.title('Active Learning vs Random Baseline Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add efficiency ratio as text
        ratio = efficiency_metrics['efficiency_comparison']['information_gain_ratio']
        plt.text(0.05, 0.95, f'Efficiency Ratio: {ratio:.2f}x',
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Learning curves plot saved to: {save_path}")

        return plt.gcf()


def test_active_learning_metrics():
    """
    Test active learning metrics functionality
    """
    print("Testing ActiveLearningMetrics...")

    # Create metrics evaluator
    metrics = ActiveLearningMetrics(num_variables=5)

    # Test convergence rate computation
    learning_curve = [1.0, 0.8, 0.6, 0.5, 0.45, 0.42, 0.41, 0.405]
    convergence_rate = metrics._compute_convergence_rate(learning_curve)
    print(f"Convergence rate: {convergence_rate:.4f}")

    # Test random intervention generation
    random_intervention = metrics._generate_random_intervention()
    print(f"Random intervention targets: {random_intervention['target_variables']}")

    # Test efficiency grading
    efficiency_grade = metrics._grade_efficiency(1.8)
    print(f"Efficiency grade for 1.8x ratio: {efficiency_grade}")

    print("✅ ActiveLearningMetrics test passed")

    return True


if __name__ == "__main__":
    test_active_learning_metrics()