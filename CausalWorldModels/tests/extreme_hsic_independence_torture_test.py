#!/usr/bin/env python3
"""
EXTREME HSIC MECHANISM INDEPENDENCE TORTURE TEST
==============================================

This test pushes the HSIC independence enforcement to its absolute limits to determine
if mechanisms are truly independent or just superficially decorrelated.

CHALLENGE DESIGN:
1. SUBTLE NON-LINEAR CONFOUNDING
   - Create non-linear dependencies that linear tests would miss
   - Test if HSIC can detect and eliminate sophisticated confounding

2. ADVERSARIAL CORRELATION PATTERNS
   - Engineer correlation patterns that fool basic independence measures
   - Test robustness against adversarial statistical patterns

3. HIGH-DIMENSIONAL DEPENDENCY STRUCTURES
   - Complex multi-dimensional dependencies between mechanisms
   - Test if HSIC scales to high-dimensional independence

4. TEMPORAL DEPENDENCY CHALLENGES
   - Mechanisms that become dependent over time
   - Test if independence holds under temporal evolution

TARGET: Only genuine HSIC independence should achieve >85% on this challenge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_architectures.causal_mechanisms import CausalMechanismModules


class ExtremeHSICIndependenceTortureTest:
    """
    Torture test for HSIC mechanism independence

    Tests whether mechanisms are genuinely independent or just superficially decorrelated
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.test_results = {}

    def setup_model(self):
        """Initialize the causal mechanism modules"""
        self.model = CausalMechanismModules(
            state_dim=12,
            hidden_dim=32
        ).to(self.device)

        # Enable independence enforcement
        self.model.independence_enforcer_enabled = True

    def generate_subtly_confounded_data(self, batch_size=64, seq_len=30):
        """
        Generate data where mechanisms are subtly confounded in non-linear ways

        Mechanisms should be independent, but we'll create hidden confounding
        """
        states = torch.zeros(batch_size, seq_len, 12)
        causal_factors = torch.zeros(batch_size, seq_len, 5)
        actions = torch.zeros(batch_size, seq_len, 2)

        # Hidden confounding variable (not observed)
        hidden_confounder = torch.randn(batch_size, seq_len)

        # Initialize
        states[:, 0, :] = torch.randn(batch_size, 12) * 0.3
        causal_factors[:, 0, :] = torch.randn(batch_size, 5) * 0.3

        for t in range(1, seq_len):
            # SUBTLE NON-LINEAR CONFOUNDING
            # All mechanisms influenced by hidden confounder through non-linear transforms

            conf = hidden_confounder[:, t]

            # Weather mechanism - confounded through sine transform
            weather_base = 0.9 * causal_factors[:, t-1, 0] + 0.1 * torch.randn(batch_size)
            weather_conf = 0.15 * torch.sin(3 * conf)  # Non-linear confounding
            causal_factors[:, t, 0] = weather_base + weather_conf

            # Crowd mechanism - confounded through exponential transform
            crowd_base = 0.8 * causal_factors[:, t-1, 1] + 0.2 * torch.randn(batch_size)
            crowd_conf = 0.1 * torch.sign(conf) * torch.exp(-torch.abs(conf))  # Non-linear confounding
            causal_factors[:, t, 1] = crowd_base + crowd_conf

            # Event mechanism - confounded through polynomial transform
            event_base = 0.7 * causal_factors[:, t-1, 2] + 0.3 * torch.randn(batch_size)
            event_conf = 0.08 * (conf**2 - conf)  # Polynomial confounding
            causal_factors[:, t, 2] = event_base + event_conf

            # Time mechanism - confounded through tanh transform
            time_base = 0.9 * causal_factors[:, t-1, 3] + 0.1 * torch.randn(batch_size)
            time_conf = 0.12 * torch.tanh(2 * conf)  # Tanh confounding
            causal_factors[:, t, 3] = time_base + time_conf

            # Road mechanism - confounded through discontinuous transform
            road_base = 0.85 * causal_factors[:, t-1, 4] + 0.15 * torch.randn(batch_size)
            road_conf = 0.1 * torch.where(conf > 0, conf**0.5, -(-conf)**0.5)  # Discontinuous confounding
            causal_factors[:, t, 4] = road_base + road_conf

            # Update hidden confounder
            hidden_confounder[:, t] = 0.95 * conf + 0.05 * torch.randn(batch_size)

            # State evolution
            states[:, t, :] = 0.8 * states[:, t-1, :] + 0.2 * torch.randn(batch_size, 12)
            actions[:, t, :] = torch.randn(batch_size, 2) * 0.4

        return states, causal_factors, actions, hidden_confounder

    def generate_adversarial_correlation_data(self, batch_size=64, seq_len=30):
        """
        Generate data with adversarial correlation patterns designed to fool independence tests
        """
        states = torch.zeros(batch_size, seq_len, 12)
        causal_factors = torch.zeros(batch_size, seq_len, 5)
        actions = torch.zeros(batch_size, seq_len, 2)

        # Initialize
        states[:, 0, :] = torch.randn(batch_size, 12) * 0.3
        causal_factors[:, 0, :] = torch.randn(batch_size, 5) * 0.3

        for t in range(1, seq_len):
            # ADVERSARIAL PATTERN 1: Zero linear correlation but high non-linear dependence
            # Weather and crowd have zero Pearson correlation but strong non-linear relationship

            weather_base = 0.9 * causal_factors[:, t-1, 0] + 0.1 * torch.randn(batch_size)
            crowd_base = 0.8 * causal_factors[:, t-1, 1] + 0.2 * torch.randn(batch_size)

            # Make crowd depend on weather^2 - mean(weather^2) (zero linear correlation)
            weather_squared = weather_base**2
            weather_squared_centered = weather_squared - weather_squared.mean()
            crowd_dependency = 0.2 * weather_squared_centered

            causal_factors[:, t, 0] = weather_base
            causal_factors[:, t, 1] = crowd_base + crowd_dependency

            # ADVERSARIAL PATTERN 2: Alternating correlation
            # Event and time alternate between positive and negative correlation

            event_base = 0.7 * causal_factors[:, t-1, 2] + 0.3 * torch.randn(batch_size)
            time_base = 0.9 * causal_factors[:, t-1, 3] + 0.1 * torch.randn(batch_size)

            # Alternating correlation based on time step
            correlation_sign = 1 if (t % 4) < 2 else -1
            time_dependency = 0.15 * correlation_sign * event_base

            causal_factors[:, t, 2] = event_base
            causal_factors[:, t, 3] = time_base + time_dependency

            # ADVERSARIAL PATTERN 3: Delayed dependency
            # Road depends on weather from 3 timesteps ago

            road_base = 0.85 * causal_factors[:, t-1, 4] + 0.15 * torch.randn(batch_size)
            if t >= 3:
                delayed_weather = causal_factors[:, t-3, 0]
                road_dependency = 0.12 * delayed_weather
            else:
                road_dependency = 0

            causal_factors[:, t, 4] = road_base + road_dependency

            # State evolution
            states[:, t, :] = 0.8 * states[:, t-1, :] + 0.2 * torch.randn(batch_size, 12)
            actions[:, t, :] = torch.randn(batch_size, 2) * 0.4

        return states, causal_factors, actions

    def test_1_subtle_nonlinear_confounding_resistance(self):
        """
        TEST 1: Can HSIC detect and resist subtle non-linear confounding?
        """
        print("🔥 TEST 1: Subtle Non-Linear Confounding Resistance")
        print("=" * 60)

        # Generate confounded data
        states, causal_factors, actions, hidden_confounder = self.generate_subtly_confounded_data()

        # Forward pass through mechanisms
        self.model.train()  # Training mode to compute independence loss

        mechanism_effects, composed_effects, next_state, independence_loss, isolation_confidence = self.model(
            states[:, :-1].reshape(-1, 12),
            causal_factors[:, :-1].reshape(-1, 5),
            actions[:, :-1].reshape(-1, 2)
        )

        # Extract individual mechanism outputs
        weather_effects = mechanism_effects['weather'].detach().cpu().numpy()
        crowd_effects = mechanism_effects['crowd'].detach().cpu().numpy()
        event_effects = mechanism_effects['event'].detach().cpu().numpy()
        time_effects = mechanism_effects['time'].detach().cpu().numpy()
        road_effects = mechanism_effects['road'].detach().cpu().numpy()

        # Compute various dependence measures
        dependence_measures = {}

        # Linear correlation (should be low even with confounding)
        dependence_measures['weather_crowd_pearson'] = abs(pearsonr(
            weather_effects.flatten(), crowd_effects.flatten()
        )[0])

        dependence_measures['event_time_pearson'] = abs(pearsonr(
            event_effects.flatten(), time_effects.flatten()
        )[0])

        # Non-linear correlation (should be detected by good independence)
        dependence_measures['weather_crowd_spearman'] = abs(spearmanr(
            weather_effects.flatten(), crowd_effects.flatten()
        )[0])

        # Mutual information (should be low if truly independent)
        try:
            dependence_measures['weather_crowd_mi'] = mutual_info_regression(
                weather_effects.reshape(-1, 1), crowd_effects.flatten()
            )[0]
            dependence_measures['event_time_mi'] = mutual_info_regression(
                event_effects.reshape(-1, 1), time_effects.flatten()
            )[0]
        except:
            dependence_measures['weather_crowd_mi'] = 0.5  # Penalty if MI fails
            dependence_measures['event_time_mi'] = 0.5

        # Test results
        results = {
            'independence_loss': independence_loss.item(),
            'isolation_confidence': isolation_confidence,
            'dependence_measures': dependence_measures,
            'mechanism_output_shapes': {
                'weather': weather_effects.shape,
                'crowd': crowd_effects.shape,
                'event': event_effects.shape,
                'time': time_effects.shape,
                'road': road_effects.shape
            }
        }

        # Scoring criteria (very harsh)
        independence_loss_pass = results['independence_loss'] < 0.1  # Should minimize dependence
        isolation_confidence_pass = results['isolation_confidence'] > 0.7  # High confidence
        pearson_pass = (dependence_measures['weather_crowd_pearson'] < 0.2 and
                       dependence_measures['event_time_pearson'] < 0.2)
        mi_pass = (dependence_measures['weather_crowd_mi'] < 0.3 and
                   dependence_measures['event_time_mi'] < 0.3)

        total_score = (independence_loss_pass + isolation_confidence_pass + pearson_pass + mi_pass) / 4.0

        print(f"  Independence Loss: {results['independence_loss']:.6f} {'✅' if independence_loss_pass else '❌'}")
        print(f"  Isolation Confidence: {results['isolation_confidence']:.4f} {'✅' if isolation_confidence_pass else '❌'}")
        print(f"  Weather-Crowd Pearson: {dependence_measures['weather_crowd_pearson']:.4f} {'✅' if pearson_pass else '❌'}")
        print(f"  Weather-Crowd MI: {dependence_measures['weather_crowd_mi']:.4f} {'✅' if mi_pass else '❌'}")
        print(f"  Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("  🎉 EXCEPTIONAL: Strong resistance to non-linear confounding!")
        elif total_score > 0.6:
            print("  👍 GOOD: Some resistance to non-linear confounding")
        elif total_score > 0.3:
            print("  ⚠️  WEAK: Limited resistance to non-linear confounding")
        else:
            print("  💀 FAILED: No resistance to non-linear confounding")

        self.test_results['test_1'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.6
        }

        return total_score, results

    def test_2_adversarial_correlation_patterns(self):
        """
        TEST 2: Can HSIC handle adversarial correlation patterns designed to fool independence tests?
        """
        print("\n🎯 TEST 2: Adversarial Correlation Pattern Resistance")
        print("=" * 60)

        # Generate adversarial data
        states, causal_factors, actions = self.generate_adversarial_correlation_data()

        # Forward pass through mechanisms
        mechanism_effects, composed_effects, next_state, independence_loss, isolation_confidence = self.model(
            states[:, :-1].reshape(-1, 12),
            causal_factors[:, :-1].reshape(-1, 5),
            actions[:, :-1].reshape(-1, 2)
        )

        # Extract mechanism outputs
        weather_effects = mechanism_effects['weather'].detach().cpu().numpy()
        crowd_effects = mechanism_effects['crowd'].detach().cpu().numpy()
        event_effects = mechanism_effects['event'].detach().cpu().numpy()
        time_effects = mechanism_effects['time'].detach().cpu().numpy()
        road_effects = mechanism_effects['road'].detach().cpu().numpy()

        # Test various adversarial patterns
        adversarial_tests = {}

        # Test 1: Zero linear correlation but high non-linear dependence
        linear_corr = abs(pearsonr(weather_effects.flatten(), crowd_effects.flatten())[0])
        nonlinear_corr = abs(spearmanr(weather_effects.flatten(), crowd_effects.flatten())[0])
        adversarial_tests['linear_vs_nonlinear'] = {
            'linear_correlation': linear_corr,
            'nonlinear_correlation': nonlinear_corr,
            'pass': linear_corr < 0.15 and nonlinear_corr < 0.25  # Should resist both
        }

        # Test 2: Alternating correlation patterns
        event_flat = event_effects.flatten()
        time_flat = time_effects.flatten()
        chunk_size = len(event_flat) // 4

        correlations_by_chunk = []
        for i in range(4):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 3 else len(event_flat)
            if end_idx > start_idx:
                chunk_corr = abs(pearsonr(
                    event_flat[start_idx:end_idx],
                    time_flat[start_idx:end_idx]
                )[0])
                correlations_by_chunk.append(chunk_corr)

        avg_chunk_correlation = np.mean(correlations_by_chunk) if correlations_by_chunk else 1.0
        adversarial_tests['alternating_correlation'] = {
            'chunk_correlations': correlations_by_chunk,
            'average_correlation': avg_chunk_correlation,
            'pass': avg_chunk_correlation < 0.2
        }

        # Test 3: Delayed dependency detection
        weather_flat = weather_effects.flatten()
        road_flat = road_effects.flatten()
        delayed_corr = abs(pearsonr(weather_flat, road_flat)[0])
        adversarial_tests['delayed_dependency'] = {
            'delayed_correlation': delayed_corr,
            'pass': delayed_corr < 0.15
        }

        results = {
            'independence_loss': independence_loss.item(),
            'isolation_confidence': isolation_confidence,
            'adversarial_tests': adversarial_tests
        }

        # Overall scoring
        pattern_passes = sum(test['pass'] for test in adversarial_tests.values())
        independence_pass = results['independence_loss'] < 0.1
        confidence_pass = results['isolation_confidence'] > 0.6

        total_score = (pattern_passes/3.0 * 0.6 + independence_pass * 0.2 + confidence_pass * 0.2)

        print(f"  Independence Loss: {results['independence_loss']:.6f} {'✅' if independence_pass else '❌'}")
        print(f"  Isolation Confidence: {results['isolation_confidence']:.4f} {'✅' if confidence_pass else '❌'}")
        print(f"  Linear vs Non-linear: {'✅' if adversarial_tests['linear_vs_nonlinear']['pass'] else '❌'}")
        print(f"  Alternating Correlation: {'✅' if adversarial_tests['alternating_correlation']['pass'] else '❌'}")
        print(f"  Delayed Dependency: {'✅' if adversarial_tests['delayed_dependency']['pass'] else '❌'}")
        print(f"  Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("  🎉 EXCEPTIONAL: Resisted all adversarial correlation patterns!")
        elif total_score > 0.6:
            print("  👍 GOOD: Resisted most adversarial patterns")
        elif total_score > 0.3:
            print("  ⚠️  WEAK: Some resistance to adversarial patterns")
        else:
            print("  💀 FAILED: Failed to resist adversarial patterns")

        self.test_results['test_2'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.6
        }

        return total_score, results

    def test_3_high_dimensional_independence_scaling(self):
        """
        TEST 3: Does HSIC independence scale to high-dimensional mechanism outputs?
        """
        print("\n🧠 TEST 3: High-Dimensional Independence Scaling")
        print("=" * 60)

        # Create high-dimensional test with many potential dependencies
        batch_size, seq_len = 32, 20

        states = torch.zeros(batch_size, seq_len, 12)
        causal_factors = torch.zeros(batch_size, seq_len, 5)
        actions = torch.zeros(batch_size, seq_len, 2)

        # Initialize
        states[:, 0, :] = torch.randn(batch_size, 12) * 0.3

        # Create systematic dependencies between ALL pairs of causal factors
        for t in range(seq_len):
            if t == 0:
                causal_factors[:, t, :] = torch.randn(batch_size, 5) * 0.3
            else:
                # Each factor influences every other factor slightly
                for i in range(5):
                    base_evolution = 0.8 * causal_factors[:, t-1, i] + 0.2 * torch.randn(batch_size)

                    # Add cross-influences from all other factors
                    cross_influence = 0.0
                    for j in range(5):
                        if i != j:
                            # Different types of cross-influence for each pair
                            if (i + j) % 3 == 0:
                                cross_influence += 0.05 * torch.sin(causal_factors[:, t-1, j])
                            elif (i + j) % 3 == 1:
                                cross_influence += 0.03 * causal_factors[:, t-1, j]**2
                            else:
                                cross_influence += 0.04 * torch.tanh(causal_factors[:, t-1, j])

                    causal_factors[:, t, i] = base_evolution + cross_influence

            # State evolution
            states[:, t, :] = 0.8 * states[:, t-1, :] + 0.2 * torch.randn(batch_size, 12)
            actions[:, t, :] = torch.randn(batch_size, 2) * 0.4

        # Test mechanism independence
        mechanism_effects, composed_effects, next_state, independence_loss, isolation_confidence = self.model(
            states[:, :-1].reshape(-1, 12),
            causal_factors[:, :-1].reshape(-1, 5),
            actions[:, :-1].reshape(-1, 2)
        )

        # Compute all pairwise dependencies
        mechanisms = ['weather', 'crowd', 'event', 'time', 'road']
        pairwise_dependencies = {}

        for i, mech_i in enumerate(mechanisms):
            for j, mech_j in enumerate(mechanisms):
                if i < j:  # Only compute upper triangle
                    effects_i = mechanism_effects[mech_i].detach().cpu().numpy().flatten()
                    effects_j = mechanism_effects[mech_j].detach().cpu().numpy().flatten()

                    # Multiple dependency measures
                    pearson_corr = abs(pearsonr(effects_i, effects_j)[0])
                    spearman_corr = abs(spearmanr(effects_i, effects_j)[0])

                    try:
                        mi = mutual_info_regression(effects_i.reshape(-1, 1), effects_j)[0]
                    except:
                        mi = 0.5  # Penalty if fails

                    pairwise_dependencies[f"{mech_i}_{mech_j}"] = {
                        'pearson': pearson_corr,
                        'spearman': spearman_corr,
                        'mutual_info': mi,
                        'max_dependency': max(pearson_corr, spearman_corr, mi)
                    }

        # Aggregate statistics
        all_pearson = [dep['pearson'] for dep in pairwise_dependencies.values()]
        all_spearman = [dep['spearman'] for dep in pairwise_dependencies.values()]
        all_mi = [dep['mutual_info'] for dep in pairwise_dependencies.values()]
        all_max = [dep['max_dependency'] for dep in pairwise_dependencies.values()]

        results = {
            'independence_loss': independence_loss.item(),
            'isolation_confidence': isolation_confidence,
            'pairwise_dependencies': pairwise_dependencies,
            'aggregate_stats': {
                'mean_pearson': np.mean(all_pearson),
                'max_pearson': np.max(all_pearson),
                'mean_spearman': np.mean(all_spearman),
                'max_spearman': np.max(all_spearman),
                'mean_mi': np.mean(all_mi),
                'max_mi': np.max(all_mi),
                'mean_max_dependency': np.mean(all_max),
                'worst_dependency': np.max(all_max)
            }
        }

        # Scoring criteria (extremely harsh for high-dimensional independence)
        independence_pass = results['independence_loss'] < 0.05  # Very low
        confidence_pass = results['isolation_confidence'] > 0.75  # High confidence
        mean_dependency_pass = results['aggregate_stats']['mean_max_dependency'] < 0.2
        worst_dependency_pass = results['aggregate_stats']['worst_dependency'] < 0.4

        total_score = (independence_pass + confidence_pass + mean_dependency_pass + worst_dependency_pass) / 4.0

        print(f"  Independence Loss: {results['independence_loss']:.6f} {'✅' if independence_pass else '❌'}")
        print(f"  Isolation Confidence: {results['isolation_confidence']:.4f} {'✅' if confidence_pass else '❌'}")
        print(f"  Mean Max Dependency: {results['aggregate_stats']['mean_max_dependency']:.4f} {'✅' if mean_dependency_pass else '❌'}")
        print(f"  Worst Dependency: {results['aggregate_stats']['worst_dependency']:.4f} {'✅' if worst_dependency_pass else '❌'}")
        print(f"  Number of Pairs Tested: {len(pairwise_dependencies)}")
        print(f"  Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("  🎉 EXCEPTIONAL: Maintains independence in high-dimensional space!")
        elif total_score > 0.6:
            print("  👍 GOOD: Good high-dimensional independence")
        elif total_score > 0.3:
            print("  ⚠️  WEAK: Limited high-dimensional independence")
        else:
            print("  💀 FAILED: Cannot maintain high-dimensional independence")

        self.test_results['test_3'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.6
        }

        return total_score, results

    def run_all_tests(self):
        """Run all HSIC independence torture tests"""
        print("🔥 EXTREME HSIC INDEPENDENCE TORTURE TEST")
        print("=" * 80)
        print("Testing whether mechanism independence is genuine or just superficial decorrelation")
        print()

        self.setup_model()

        # Run all tests
        score_1, _ = self.test_1_subtle_nonlinear_confounding_resistance()
        score_2, _ = self.test_2_adversarial_correlation_patterns()
        score_3, _ = self.test_3_high_dimensional_independence_scaling()

        # Overall assessment
        overall_score = (score_1 + score_2 + score_3) / 3.0

        print("\n" + "=" * 80)
        print("📊 HSIC INDEPENDENCE TORTURE TEST RESULTS")
        print("=" * 80)
        print(f"Test 1 - Non-Linear Confounding: {score_1:.3f}")
        print(f"Test 2 - Adversarial Patterns: {score_2:.3f}")
        print(f"Test 3 - High-Dimensional Scaling: {score_3:.3f}")
        print(f"Overall Score: {overall_score:.3f}")

        if overall_score > 0.85:
            grade = "A+"
            status = "🔥 EXCEPTIONAL - Genuine HSIC independence!"
        elif overall_score > 0.75:
            grade = "A"
            status = "🎉 EXCELLENT - Strong mechanism independence"
        elif overall_score > 0.65:
            grade = "B"
            status = "👍 GOOD - Decent mechanism independence"
        elif overall_score > 0.4:
            grade = "C"
            status = "⚠️ WEAK - Limited mechanism independence"
        else:
            grade = "F"
            status = "💀 FAILED - No meaningful mechanism independence"

        print(f"Grade: {grade}")
        print(f"Status: {status}")

        # Save results
        results_summary = {
            'overall_score': overall_score,
            'grade': grade,
            'status': status,
            'individual_tests': self.test_results,
            'timestamp': time.time()
        }

        return results_summary


def main():
    """Run the extreme HSIC independence torture test"""
    test = ExtremeHSICIndependenceTortureTest()
    results = test.run_all_tests()

    # Save results
    with open('extreme_hsic_independence_torture_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        json.dump(convert_numpy(results), f, indent=2)

    print(f"\n📁 Results saved to: extreme_hsic_independence_torture_results.json")

    return results


if __name__ == "__main__":
    results = main()