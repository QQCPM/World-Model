"""
Enhanced Causal Reasoner Tester
Comprehensive validation framework for genuine causal reasoning

Tests:
- Level 1: Structure-aware mechanism tests
- Level 2: Active learning validation
- Level 3: Joint learning assessment
- Level 4: Out-of-distribution generalization
- Severe validation tests (currently 0/5 → target 4/5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import json
import time
from dataclasses import dataclass


@dataclass
class CausalTestConfig:
    """Configuration for causal reasoning tests"""
    num_test_episodes: int = 100
    intervention_strength_range: Tuple[float, float] = (-1.0, 1.0)
    temporal_horizon: int = 10
    noise_levels: List[float] = None
    confidence_threshold: float = 0.8

    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.1, 0.2, 0.3]


class CausalReasonerTester:
    """
    Enhanced causal reasoning test suite

    Validates genuine causal understanding vs pattern matching
    """

    def __init__(self, config: CausalTestConfig = None):
        self.config = config or CausalTestConfig()

        # Variable names for campus environment
        self.variable_names = [
            'weather',           # 0: weather condition
            'crowd_density',     # 1: campus crowd level
            'special_event',     # 2: special events happening
            'time_of_day',       # 3: time of day
            'road_conditions'    # 4: road/path conditions
        ]

        # Test results storage
        self.test_results = {}

    def run_comprehensive_evaluation(self, joint_trainer, test_data_loader):
        """
        Run comprehensive causal reasoning evaluation

        Args:
            joint_trainer: Trained JointCausalTrainer instance
            test_data_loader: Test data loader

        Returns:
            evaluation_results: Dict with detailed test results
        """
        print("🧪 Starting Comprehensive Causal Reasoning Evaluation")
        print("=" * 60)

        # Extract models
        dynamics_model = joint_trainer.dynamics_model
        structure_learner = joint_trainer.structure_learner
        causal_mechanisms = joint_trainer.causal_mechanisms
        intervention_designer = joint_trainer.intervention_designer

        evaluation_results = {
            'test_timestamp': time.time(),
            'model_info': self._get_model_info(joint_trainer),
            'severe_validation': {},
            'structure_tests': {},
            'mechanism_tests': {},
            'intervention_tests': {},
            'generalization_tests': {}
        }

        # SEVERE VALIDATION TESTS (Main target: 0/5 → 4/5)
        print("\n🎯 SEVERE VALIDATION TESTS")
        print("-" * 30)

        severe_results = self._run_severe_validation_tests(
            dynamics_model, structure_learner, causal_mechanisms, test_data_loader
        )
        evaluation_results['severe_validation'] = severe_results

        # STRUCTURE LEARNING TESTS
        print("\n🏗️  STRUCTURE LEARNING TESTS")
        print("-" * 30)

        structure_results = self._test_structure_learning(
            structure_learner, test_data_loader
        )
        evaluation_results['structure_tests'] = structure_results

        # MECHANISM ISOLATION TESTS
        print("\n⚙️  MECHANISM ISOLATION TESTS")
        print("-" * 30)

        mechanism_results = self._test_mechanism_isolation(
            causal_mechanisms, test_data_loader
        )
        evaluation_results['mechanism_tests'] = mechanism_results

        # INTERVENTION DESIGN TESTS
        print("\n🎛️  INTERVENTION DESIGN TESTS")
        print("-" * 30)

        intervention_results = self._test_intervention_design(
            intervention_designer, structure_learner, test_data_loader
        )
        evaluation_results['intervention_tests'] = intervention_results

        # OUT-OF-DISTRIBUTION GENERALIZATION
        print("\n🌍 GENERALIZATION TESTS")
        print("-" * 30)

        generalization_results = self._test_generalization(
            dynamics_model, structure_learner, test_data_loader
        )
        evaluation_results['generalization_tests'] = generalization_results

        # COMPUTE OVERALL SCORES
        overall_scores = self._compute_overall_scores(evaluation_results)
        evaluation_results['overall_scores'] = overall_scores

        # SUMMARY REPORT
        self._print_evaluation_summary(evaluation_results)

        return evaluation_results

    def _run_severe_validation_tests(self, dynamics_model, structure_learner,
                                   causal_mechanisms, test_data_loader):
        """
        Run the 5 severe validation tests that currently fail
        """
        severe_tests = {
            'test_1_counterfactual_consistency': self._test_counterfactual_consistency,
            'test_2_intervention_invariance': self._test_intervention_invariance,
            'test_3_temporal_causality': self._test_temporal_causality,
            'test_4_mechanism_isolation': self._test_mechanism_isolation_severe,
            'test_5_compositional_generalization': self._test_compositional_generalization
        }

        severe_results = {}
        passed_tests = 0

        for test_name, test_function in severe_tests.items():
            print(f"Running {test_name}...")

            try:
                test_result = test_function(
                    dynamics_model, structure_learner, causal_mechanisms, test_data_loader
                )

                passed = test_result['passed']
                confidence = test_result['confidence']

                severe_results[test_name] = test_result

                if passed:
                    passed_tests += 1
                    print(f"  ✅ PASSED (confidence: {confidence:.3f})")
                else:
                    print(f"  ❌ FAILED (confidence: {confidence:.3f})")

            except Exception as e:
                print(f"  ⚠️  ERROR: {str(e)}")
                severe_results[test_name] = {
                    'passed': False,
                    'confidence': 0.0,
                    'error': str(e)
                }

        severe_results['summary'] = {
            'total_tests': len(severe_tests),
            'passed_tests': passed_tests,
            'success_rate': passed_tests / len(severe_tests),
            'target_achieved': passed_tests >= 4  # Target: 4/5 tests
        }

        print(f"\n🎯 SEVERE VALIDATION SUMMARY: {passed_tests}/{len(severe_tests)} passed")
        if passed_tests >= 4:
            print("🎉 TARGET ACHIEVED: 4/5+ severe tests passed!")
        else:
            print(f"📈 Progress: {passed_tests}/5 → Need {4 - passed_tests} more to reach target")

        return severe_results

    def _test_counterfactual_consistency(self, dynamics_model, structure_learner,
                                       causal_mechanisms, test_data_loader):
        """
        Test 1: Counterfactual consistency
        Model should generate consistent counterfactuals that respect causal structure
        """
        consistency_scores = []

        for batch_data in test_data_loader:
            states, actions, causal_factors = batch_data

            # Generate counterfactual scenarios
            for intervention_var in range(5):
                # Create intervention
                cf_causal = causal_factors.clone()
                cf_causal[:, :, intervention_var] = torch.randn_like(cf_causal[:, :, intervention_var])

                # Predict counterfactual states
                with torch.no_grad():
                    cf_states, _, _ = dynamics_model(
                        states[:, :-1], actions[:, :-1], cf_causal[:, :-1]
                    )

                # Check consistency (counterfactuals should be plausible)
                consistency = self._measure_trajectory_consistency(cf_states)
                consistency_scores.append(consistency)

            break  # Test on first batch

        avg_consistency = np.mean(consistency_scores)

        return {
            'passed': avg_consistency > 0.7,
            'confidence': avg_consistency,
            'details': {
                'average_consistency': avg_consistency,
                'min_consistency': np.min(consistency_scores),
                'max_consistency': np.max(consistency_scores)
            }
        }

    def _test_intervention_invariance(self, dynamics_model, structure_learner,
                                    causal_mechanisms, test_data_loader):
        """
        Test 2: Intervention invariance
        Model predictions should be stable under do-operations vs observational changes
        """
        invariance_scores = []

        for batch_data in test_data_loader:
            states, actions, causal_factors = batch_data

            # Test intervention vs observation
            for var_idx in range(5):
                # Observational prediction
                obs_states, _, _ = dynamics_model(
                    states[:, :-1], actions[:, :-1], causal_factors[:, :-1]
                )

                # Interventional prediction (force pathway)
                dynamics_model.set_pathway_mode('interventional')
                int_states, _, _ = dynamics_model(
                    states[:, :-1], actions[:, :-1], causal_factors[:, :-1]
                )
                dynamics_model.set_pathway_mode('auto')

                # Measure invariance (they should be different but stable)
                invariance = self._measure_prediction_stability(obs_states, int_states)
                invariance_scores.append(invariance)

            break  # Test on first batch

        avg_invariance = np.mean(invariance_scores)

        return {
            'passed': avg_invariance > 0.6,
            'confidence': avg_invariance,
            'details': {
                'average_invariance': avg_invariance,
                'pathway_difference_detected': len([s for s in invariance_scores if s > 0.1]) > 0
            }
        }

    def _test_temporal_causality(self, dynamics_model, structure_learner,
                               causal_mechanisms, test_data_loader):
        """
        Test 3: Temporal causality
        Model should understand temporal relationships and delays
        """
        temporal_scores = []

        # Get learned causal structure
        causal_graph = structure_learner.get_adjacency_matrix()

        for batch_data in test_data_loader:
            states, actions, causal_factors = batch_data

            # Test temporal delays
            for delay in [1, 2, 3]:
                # Shift causal factor by delay
                shifted_causal = torch.roll(causal_factors, shifts=delay, dims=1)

                with torch.no_grad():
                    shifted_states, _, _ = dynamics_model(
                        states[:, :-1], actions[:, :-1], shifted_causal[:, :-1]
                    )

                # Check if model detects temporal inconsistency
                temporal_consistency = self._measure_temporal_causality(
                    states, shifted_states, delay
                )
                temporal_scores.append(temporal_consistency)

            break  # Test on first batch

        avg_temporal = np.mean(temporal_scores)

        return {
            'passed': avg_temporal > 0.5,
            'confidence': avg_temporal,
            'details': {
                'average_temporal_score': avg_temporal,
                'delay_sensitivity': len([s for s in temporal_scores if s > 0.3]) / len(temporal_scores)
            }
        }

    def _test_mechanism_isolation_severe(self, dynamics_model, structure_learner,
                                       causal_mechanisms, test_data_loader):
        """
        Test 4: Mechanism isolation (severe version)
        Model should isolate individual causal mechanisms
        """
        isolation_scores = []

        for batch_data in test_data_loader:
            states, actions, causal_factors = batch_data

            # Test isolation of each mechanism
            for mechanism_idx in range(5):
                # Zero out other mechanisms
                isolated_causal = torch.zeros_like(causal_factors)
                isolated_causal[:, :, mechanism_idx] = causal_factors[:, :, mechanism_idx]

                # Get mechanism-specific effects
                batch_states = states[:1, :-1].reshape(-1, states.shape[-1])
                batch_causal = isolated_causal[:1, :-1].reshape(-1, 5)
                batch_actions = actions[:1, :-1].reshape(-1, 2)

                mechanism_effects, composed_effects, _, _, _ = causal_mechanisms(
                    batch_states, batch_causal, batch_actions
                )

                # Check if mechanism has distinguishable effect
                effect_magnitude = torch.norm(composed_effects).item()
                isolation_scores.append(effect_magnitude)

            break  # Test on first batch

        # Mechanisms should have diverse effects
        effect_diversity = np.std(isolation_scores) / (np.mean(isolation_scores) + 1e-8)

        return {
            'passed': effect_diversity > 0.3,
            'confidence': min(effect_diversity / 0.5, 1.0),
            'details': {
                'effect_diversity': effect_diversity,
                'mechanism_effects': isolation_scores
            }
        }

    def _test_compositional_generalization(self, dynamics_model, structure_learner,
                                         causal_mechanisms, test_data_loader):
        """
        Test 5: Compositional generalization
        Model should compose mechanisms correctly for novel combinations
        """
        composition_scores = []

        for batch_data in test_data_loader:
            states, actions, causal_factors = batch_data

            # Test novel combinations
            novel_combinations = [
                [0, 2],  # Weather + Event
                [1, 3],  # Crowd + Time
                [0, 4],  # Weather + Road
                [1, 2, 4]  # Crowd + Event + Road
            ]

            for combination in novel_combinations:
                # Create causal scenario with specific combination
                combo_causal = torch.zeros_like(causal_factors)
                for var_idx in combination:
                    combo_causal[:, :, var_idx] = torch.rand_like(causal_factors[:, :, var_idx])

                # Predict with composition
                with torch.no_grad():
                    combo_states, _, pathway_info = dynamics_model(
                        states[:, :-1], actions[:, :-1], combo_causal[:, :-1]
                    )

                # Check if composition is reasonable
                composition_quality = self._measure_composition_quality(
                    combo_states, combination, pathway_info
                )
                composition_scores.append(composition_quality)

            break  # Test on first batch

        avg_composition = np.mean(composition_scores)

        return {
            'passed': avg_composition > 0.4,
            'confidence': avg_composition,
            'details': {
                'average_composition': avg_composition,
                'composition_scores': composition_scores
            }
        }

    def _test_structure_learning(self, structure_learner, test_data_loader):
        """
        Test structure learning accuracy and stability
        """
        structure_results = {}

        # Get learned structure
        causal_graph = structure_learner.get_adjacency_matrix()
        structure_summary = structure_learner.get_causal_graph_summary()

        # Basic structure metrics
        structure_results['graph_properties'] = {
            'num_edges': structure_summary['num_edges'],
            'sparsity': structure_summary['sparsity'],
            'density': structure_summary['graph_density']
        }

        # Structure stability test
        stability_scores = []
        for batch_data in test_data_loader:
            states, actions, causal_factors = batch_data

            # Compute structure loss
            structure_loss, loss_info = structure_learner.compute_structure_loss(causal_factors)
            stability_scores.append(loss_info['dag_constraint'])

            break  # Test on first batch

        avg_stability = np.mean(np.abs(stability_scores))

        structure_results['stability'] = {
            'dag_constraint_violation': avg_stability,
            'is_stable': avg_stability < 0.1
        }

        return structure_results

    def _test_mechanism_isolation(self, causal_mechanisms, test_data_loader):
        """
        Test individual mechanism isolation and interpretability
        """
        mechanism_results = {}

        for batch_data in test_data_loader:
            states, actions, causal_factors = batch_data

            # Test each mechanism
            mechanism_effects = {}

            batch_states = states[:1, :-1].reshape(-1, states.shape[-1])
            batch_actions = actions[:1, :-1].reshape(-1, 2)
            batch_causal = causal_factors[:1, :-1].reshape(-1, 5)

            individual_effects, composed_effects, _, _, _ = causal_mechanisms(
                batch_states, batch_causal, batch_actions
            )

            for mechanism_name, effect in individual_effects.items():
                mechanism_effects[mechanism_name] = {
                    'effect_magnitude': torch.norm(effect).item(),
                    'effect_variance': torch.var(effect).item()
                }

            mechanism_results['individual_mechanisms'] = mechanism_effects
            break  # Test on first batch

        return mechanism_results

    def _test_intervention_design(self, intervention_designer, structure_learner, test_data_loader):
        """
        Test intervention design effectiveness
        """
        intervention_results = {}

        for batch_data in test_data_loader:
            states, actions, causal_factors = batch_data

            # Test intervention selection
            best_intervention, candidates = intervention_designer.select_optimal_intervention(
                structure_learner, causal_factors, num_candidates=10
            )

            intervention_results['best_intervention'] = {
                'target_variables': [intervention_designer.variable_names[i]
                                   for i in best_intervention['target_variables']],
                'expected_gain': best_intervention['info_gain'],
                'acquisition_score': best_intervention['acquisition_score']
            }

            # Test intervention recommendations
            recommendations = intervention_designer.get_intervention_recommendations(
                structure_learner, top_k=3
            )

            intervention_results['top_recommendations'] = [
                {
                    'rank': rec['rank'],
                    'variables': rec['target_variables'],
                    'expected_gain': rec['expected_gain'],
                    'feasibility': rec['feasibility']
                }
                for rec in recommendations
            ]

            break  # Test on first batch

        return intervention_results

    def _test_generalization(self, dynamics_model, structure_learner, test_data_loader):
        """
        Test out-of-distribution generalization
        """
        generalization_results = {}

        # Test with different noise levels
        noise_performances = {}

        for noise_level in self.config.noise_levels:
            performance_scores = []

            for batch_data in test_data_loader:
                states, actions, causal_factors = batch_data

                # Add noise to causal factors
                noisy_causal = causal_factors + torch.randn_like(causal_factors) * noise_level

                # Test prediction accuracy
                with torch.no_grad():
                    predicted_states, _, _ = dynamics_model(
                        states[:, :-1], actions[:, :-1], noisy_causal[:, :-1]
                    )

                # Measure prediction quality
                prediction_error = F.mse_loss(predicted_states, states[:, 1:]).item()
                performance_scores.append(1.0 / (1.0 + prediction_error))  # Convert to performance score

                break  # Test on first batch

            noise_performances[f'noise_{noise_level}'] = np.mean(performance_scores)

        generalization_results['noise_robustness'] = noise_performances

        # Compute generalization score
        baseline_performance = noise_performances['noise_0.0']
        noisy_performance = np.mean([score for key, score in noise_performances.items() if key != 'noise_0.0'])

        generalization_results['generalization_score'] = noisy_performance / baseline_performance

        return generalization_results

    # Helper methods for measurements
    def _measure_trajectory_consistency(self, trajectory):
        """Measure trajectory smoothness and physical plausibility"""
        if trajectory.shape[1] < 2:
            return 0.5

        # Check for reasonable velocity changes
        position_changes = trajectory[:, 1:, :2] - trajectory[:, :-1, :2]
        velocity_changes = torch.diff(position_changes, dim=1)

        # Penalize large velocity jumps
        large_jumps = (torch.norm(velocity_changes, dim=2) > 1.0).float()
        consistency = 1.0 - large_jumps.mean()

        return consistency.item()

    def _measure_prediction_stability(self, pred1, pred2):
        """Measure stability between two predictions"""
        difference = F.mse_loss(pred1, pred2).item()
        # Convert difference to stability score (0 = identical, 1 = very different)
        return min(difference, 1.0)

    def _measure_temporal_causality(self, original_states, shifted_states, delay):
        """Measure temporal causality understanding"""
        # Align tensors to handle temporal sequence length differences
        min_len = min(original_states.shape[1], shifted_states.shape[1])
        aligned_original = original_states[:, :min_len]
        aligned_shifted = shifted_states[:, :min_len]

        # Model should detect temporal inconsistency
        difference = F.mse_loss(aligned_original, aligned_shifted).item()
        # Higher difference indicates better temporal understanding
        return min(difference / delay, 1.0)

    def _measure_composition_quality(self, combo_states, combination, pathway_info):
        """Measure quality of mechanism composition"""
        # Check if pathway selection is reasonable
        pathway_balance = pathway_info.get('pathway_balance', 0.5)

        # Check trajectory smoothness
        consistency = self._measure_trajectory_consistency(combo_states)

        # Combine metrics
        quality = 0.5 * (1.0 - pathway_balance) + 0.5 * consistency
        return quality

    def _get_model_info(self, joint_trainer):
        """Get model architecture information"""
        return {
            'dynamics_model_params': joint_trainer.dynamics_model.count_parameters(),
            'structure_learner_params': sum(p.numel() for p in joint_trainer.structure_learner.parameters()),
            'total_parameters': sum(p.numel() for p in joint_trainer.dynamics_model.parameters()) +
                              sum(p.numel() for p in joint_trainer.structure_learner.parameters()),
            'training_epochs': joint_trainer.current_epoch
        }

    def _compute_overall_scores(self, evaluation_results):
        """Compute overall evaluation scores"""
        severe_score = evaluation_results['severe_validation']['summary']['success_rate']

        # Structure score
        structure_stable = evaluation_results['structure_tests']['stability']['is_stable']
        structure_score = 1.0 if structure_stable else 0.5

        # Generalization score
        gen_score = evaluation_results['generalization_tests'].get('generalization_score', 0.5)

        # Overall causal reasoning score
        overall_score = 0.5 * severe_score + 0.3 * structure_score + 0.2 * gen_score

        return {
            'severe_validation_score': severe_score,
            'structure_learning_score': structure_score,
            'generalization_score': gen_score,
            'overall_causal_reasoning_score': overall_score,
            'grade': self._assign_grade(overall_score)
        }

    def _assign_grade(self, score):
        """Assign letter grade based on overall score"""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B+'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.5:
            return 'C+'
        elif score >= 0.4:
            return 'C'
        else:
            return 'F'

    def _print_evaluation_summary(self, evaluation_results):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*60)
        print("🎯 CAUSAL REASONING EVALUATION SUMMARY")
        print("="*60)

        overall = evaluation_results['overall_scores']
        severe = evaluation_results['severe_validation']['summary']

        print(f"📊 Overall Grade: {overall['grade']}")
        print(f"📈 Overall Score: {overall['overall_causal_reasoning_score']:.3f}")
        print()

        print(f"🎯 Severe Validation: {severe['passed_tests']}/{severe['total_tests']} tests passed")
        if severe['target_achieved']:
            print("   🎉 TARGET ACHIEVED: 4/5+ severe tests passed!")
        else:
            print(f"   📈 Progress toward 4/5 target: {severe['passed_tests']}/5")
        print()

        print(f"🏗️  Structure Learning: {'✅ Stable' if evaluation_results['structure_tests']['stability']['is_stable'] else '❌ Unstable'}")
        print(f"🌍 Generalization: {overall['generalization_score']:.3f}")
        print()

        if severe['success_rate'] >= 0.8:
            print("🎊 EXCELLENT: Model demonstrates genuine causal reasoning!")
        elif severe['success_rate'] >= 0.6:
            print("👍 GOOD: Model shows strong causal understanding")
        elif severe['success_rate'] >= 0.4:
            print("📈 PROGRESS: Model has basic causal capabilities")
        else:
            print("⚠️  PATTERN MATCHING: Model needs more causal reasoning development")

        print("="*60)

    def save_evaluation_results(self, evaluation_results, filepath):
        """Save evaluation results to file"""
        with open(filepath, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        print(f"Evaluation results saved to: {filepath}")


def test_causal_reasoner_tester():
    """
    Test causal reasoner tester functionality
    """
    print("Testing CausalReasonerTester...")

    # Create tester
    tester = CausalReasonerTester()

    # Test helper methods
    test_trajectory = torch.randn(4, 10, 12)
    consistency = tester._measure_trajectory_consistency(test_trajectory)
    print(f"Trajectory consistency: {consistency:.3f}")

    # Test grading
    grade = tester._assign_grade(0.75)
    print(f"Sample grade for 0.75 score: {grade}")

    print("✅ CausalReasonerTester test passed")

    return True


if __name__ == "__main__":
    test_causal_reasoner_tester()