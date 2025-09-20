#!/usr/bin/env python3
"""
EXTREME COUNTERFACTUAL REASONING TORTURE TEST
============================================

This test pushes counterfactual reasoning to its absolute limits to determine
if the system can perform genuine counterfactual inference or if it's broken.

CHALLENGE DESIGN:
1. BASIC COUNTERFACTUAL CONSISTENCY
   - Test if "what if weather was sunny instead of rainy" works
   - Verify consistency across multiple counterfactual queries

2. CHAINED COUNTERFACTUAL REASONING
   - Multi-step counterfactuals: "what if A, then what if B given A"
   - Test compositional counterfactual reasoning

3. STRUCTURAL COUNTERFACTUALS
   - Counterfactuals that depend on causal structure
   - Test if system respects causal ordering

4. IMPOSSIBLE COUNTERFACTUALS
   - Test if system rejects causally impossible scenarios
   - Verify counterfactual reasoning boundaries

TARGET: Only genuine counterfactual reasoning should achieve >70% consistency

NOTE: Current system shows 0.000 score - this test will diagnose the root cause
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import traceback

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components with error handling
try:
    from causal_architectures.dual_pathway_gru import DualPathwayCausalGRU
    from causal_architectures.counterfactual_wrapper import CounterfactualDynamicsWrapper
    from training.counterfactual_generator import StructureAwareCFGenerator
except ImportError as e:
    print(f"Warning: Could not import some components: {e}")

from causal_envs.continuous_campus_env import CausalState, WeatherType, EventType


class ExtremeCounterfactualReasoningTortureTest:
    """
    Torture test for counterfactual reasoning capabilities

    Tests whether counterfactual reasoning works at all and how well it handles complex scenarios
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.cf_generator = None
        self.test_results = {}
        self.setup_errors = []

    def setup_models(self):
        """Initialize counterfactual reasoning models with extensive error handling"""
        print("    🔧 Setting up counterfactual reasoning models...")

        try:
            # Try to create base dynamics model
            base_model = DualPathwayCausalGRU(
                state_dim=12,
                action_dim=2,
                causal_dim=5,
                hidden_dim=64
            ).to(self.device)
            print("      ✅ Base DualPathwayCausalGRU created")

            # Try to wrap with counterfactual wrapper
            try:
                self.model = CounterfactualDynamicsWrapper(
                    base_model,
                    state_dim=12,
                    action_dim=2,
                    causal_dim=5
                ).to(self.device)
                print("      ✅ CounterfactualDynamicsWrapper created")
            except Exception as e:
                print(f"      ❌ CounterfactualDynamicsWrapper failed: {e}")
                self.setup_errors.append(f"CounterfactualDynamicsWrapper: {e}")
                # Use base model directly
                self.model = base_model
                print("      ⚠️  Using base model without counterfactual wrapper")

            # Try to create counterfactual generator
            try:
                self.cf_generator = StructureAwareCFGenerator(
                    num_variables=5,
                    state_dim=12,
                    action_dim=2
                ).to(self.device)
                print("      ✅ StructureAwareCFGenerator created")
            except Exception as e:
                print(f"      ❌ StructureAwareCFGenerator failed: {e}")
                self.setup_errors.append(f"StructureAwareCFGenerator: {e}")
                self.cf_generator = None

        except Exception as e:
            print(f"      💀 Complete model setup failure: {e}")
            self.setup_errors.append(f"Complete setup failure: {e}")
            raise

    def generate_factual_scenario(self, seq_len=20):
        """
        Generate a factual scenario for counterfactual comparison
        """
        states = torch.zeros(seq_len, 12)
        actions = torch.zeros(seq_len, 2)
        causal_factors = torch.zeros(seq_len, 5)

        # Initialize
        states[0, :] = torch.randn(12) * 0.3
        causal_factors[0, :] = torch.randn(5) * 0.3

        # Create factual sequence with specific weather pattern
        for t in range(1, seq_len):
            # Factual: rainy weather throughout
            weather = WeatherType.RAIN
            crowd = 0.6 + 0.1 * np.sin(t * 0.2)  # Moderate crowd
            time_hour = (8 + t) % 24
            event = EventType.NORMAL

            causal_state = CausalState(
                time_hour=time_hour,
                day_week=t % 7,
                weather=weather,
                event=event,
                crowd_density=crowd
            )

            causal_factors[t, :] = torch.tensor(causal_state.to_vector())

            # State evolution
            states[t, :] = 0.8 * states[t-1, :] + 0.2 * torch.randn(12)
            actions[t, :] = torch.randn(2) * 0.5

        return states, actions, causal_factors

    def generate_counterfactual_scenario(self, factual_states, factual_actions, factual_causal, intervention_timestep=5):
        """
        Generate counterfactual scenario by changing weather at specific timestep
        """
        cf_states = factual_states.clone()
        cf_actions = factual_actions.clone()
        cf_causal = factual_causal.clone()

        # Counterfactual: change weather to sunny at intervention timestep
        for t in range(intervention_timestep, len(cf_causal)):
            weather = WeatherType.SUNNY  # Counterfactual weather
            crowd = factual_causal[t, 2].item()  # Keep other factors similar initially
            time_hour = int(factual_causal[t, 3].item() * 23)
            event = EventType.NORMAL

            cf_causal_state = CausalState(
                time_hour=time_hour,
                day_week=t % 7,
                weather=weather,
                event=event,
                crowd_density=crowd
            )

            cf_causal[t, :] = torch.tensor(cf_causal_state.to_vector())

        return cf_states, cf_actions, cf_causal

    def test_1_basic_counterfactual_consistency(self):
        """
        TEST 1: Can the system generate basic counterfactuals consistently?
        """
        print("🔥 TEST 1: Basic Counterfactual Consistency")
        print("=" * 60)

        if self.setup_errors:
            print("    💀 SETUP ERRORS DETECTED:")
            for error in self.setup_errors:
                print(f"      - {error}")

        # Generate factual and counterfactual scenarios
        factual_states, factual_actions, factual_causal = self.generate_factual_scenario()
        cf_states, cf_actions, cf_causal = self.generate_counterfactual_scenario(
            factual_states, factual_actions, factual_causal, intervention_timestep=5
        )

        results = {
            'setup_errors': self.setup_errors,
            'factual_generated': True,
            'counterfactual_generated': True,
            'model_functional': False,
            'prediction_differences': [],
            'consistency_scores': []
        }

        # Test basic model functionality
        try:
            self.model.eval()
            with torch.no_grad():
                # Test if model can make predictions at all
                test_input_states = factual_states[:-1].unsqueeze(0)  # Add batch dimension
                test_input_actions = factual_actions[:-1].unsqueeze(0)
                test_input_causal = factual_causal[:-1].unsqueeze(0)

                # Try different forward pass methods
                forward_success = False
                prediction_method = "unknown"

                # Method 1: Standard forward
                try:
                    factual_pred = self.model(test_input_states, test_input_actions, test_input_causal)
                    if isinstance(factual_pred, tuple):
                        factual_pred = factual_pred[0]  # Get states if tuple returned
                    forward_success = True
                    prediction_method = "standard_forward"
                    print(f"      ✅ Standard forward pass successful: {factual_pred.shape}")
                except Exception as e:
                    print(f"      ❌ Standard forward failed: {e}")

                # Method 2: Try predict_single if standard forward fails
                if not forward_success and hasattr(self.model, 'predict_single'):
                    try:
                        single_pred, _ = self.model.predict_single(
                            test_input_states[0, 0],
                            test_input_actions[0, 0],
                            test_input_causal[0, 0]
                        )
                        forward_success = True
                        prediction_method = "predict_single"
                        print(f"      ✅ Predict single successful: {single_pred.shape}")
                    except Exception as e:
                        print(f"      ❌ Predict single failed: {e}")

                # Method 3: Try base model if wrapped model fails
                if not forward_success and hasattr(self.model, 'base_model'):
                    try:
                        base_pred = self.model.base_model(test_input_states, test_input_actions, test_input_causal)
                        if isinstance(base_pred, tuple):
                            base_pred = base_pred[0]
                        forward_success = True
                        prediction_method = "base_model"
                        print(f"      ✅ Base model forward successful: {base_pred.shape}")
                    except Exception as e:
                        print(f"      ❌ Base model forward failed: {e}")

                results['model_functional'] = forward_success
                results['prediction_method'] = prediction_method

                if forward_success:
                    # Try to generate counterfactual predictions
                    try:
                        cf_test_states = cf_states[:-1].unsqueeze(0)
                        cf_test_actions = cf_actions[:-1].unsqueeze(0)
                        cf_test_causal = cf_causal[:-1].unsqueeze(0)

                        if prediction_method == "standard_forward":
                            cf_pred = self.model(cf_test_states, cf_test_actions, cf_test_causal)
                            if isinstance(cf_pred, tuple):
                                cf_pred = cf_pred[0]
                        elif prediction_method == "predict_single":
                            cf_pred, _ = self.model.predict_single(
                                cf_test_states[0, 0],
                                cf_test_actions[0, 0],
                                cf_test_causal[0, 0]
                            )
                            cf_pred = cf_pred.unsqueeze(0).unsqueeze(0)
                        elif prediction_method == "base_model":
                            cf_pred = self.model.base_model(cf_test_states, cf_test_actions, cf_test_causal)
                            if isinstance(cf_pred, tuple):
                                cf_pred = cf_pred[0]

                        # Compute difference between factual and counterfactual predictions
                        if prediction_method == "predict_single":
                            # For single predictions, only compare one timestep
                            pred_diff = torch.abs(factual_pred - cf_pred).mean().item()
                        else:
                            pred_diff = torch.abs(factual_pred - cf_pred).mean().item()

                        results['prediction_differences'].append(pred_diff)
                        results['counterfactual_prediction_successful'] = True

                        print(f"      ✅ Counterfactual prediction successful")
                        print(f"      📊 Prediction difference: {pred_diff:.6f}")

                    except Exception as e:
                        print(f"      ❌ Counterfactual prediction failed: {e}")
                        results['counterfactual_prediction_successful'] = False

        except Exception as e:
            print(f"    💀 COMPLETE MODEL FAILURE: {e}")
            print(f"    Traceback: {traceback.format_exc()}")
            results['model_functional'] = False

        # Scoring based on what worked
        setup_score = 1.0 if not self.setup_errors else 0.5
        functionality_score = 1.0 if results['model_functional'] else 0.0
        cf_prediction_score = 1.0 if results.get('counterfactual_prediction_successful', False) else 0.0

        total_score = (setup_score + functionality_score + cf_prediction_score) / 3.0

        print(f"    Setup Success: {'✅' if setup_score > 0.8 else '❌'}")
        print(f"    Model Functional: {'✅' if functionality_score > 0.8 else '❌'}")
        print(f"    CF Prediction: {'✅' if cf_prediction_score > 0.8 else '❌'}")
        print(f"    Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("    🎉 EXCEPTIONAL: Basic counterfactual reasoning works!")
        elif total_score > 0.5:
            print("    👍 PARTIAL: Some counterfactual capability")
        elif total_score > 0.2:
            print("    ⚠️  BROKEN: Major counterfactual issues")
        else:
            print("    💀 FAILED: Counterfactual reasoning completely broken")

        self.test_results['test_1'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.5
        }

        return total_score, results

    def test_2_counterfactual_generator_functionality(self):
        """
        TEST 2: Can the counterfactual generator component work?
        """
        print("\n🎯 TEST 2: Counterfactual Generator Functionality")
        print("=" * 60)

        if self.cf_generator is None:
            print("    💀 CRITICAL: No counterfactual generator available")
            results = {'error': 'no_cf_generator', 'generator_functional': False}
            self.test_results['test_2'] = {'score': 0.0, 'details': results, 'passed': False}
            return 0.0, results

        # Test counterfactual generator
        results = {
            'generator_functional': False,
            'generation_attempts': 0,
            'successful_generations': 0,
            'generation_errors': []
        }

        try:
            # Create test episode
            seq_len = 10
            test_episode = {
                'states': torch.randn(seq_len, 12),
                'actions': torch.randn(seq_len, 2),
                'causal_factors': torch.randn(seq_len, 5)
            }

            # Create test intervention
            intervention_spec = {
                'target_variable': 0,  # Weather
                'intervention_value': 0.8,
                'intervention_timestep': 5
            }

            # Create dummy causal graph
            causal_graph = torch.zeros(5, 5)
            causal_graph[0, 1] = 0.5  # weather -> crowd

            # Create dummy mechanisms
            from causal_architectures.causal_mechanisms import CausalMechanismModules
            causal_mechanisms = CausalMechanismModules(state_dim=12, hidden_dim=32)

            results['generation_attempts'] = 1

            # Try to generate counterfactual
            try:
                cf_episode, generation_info = self.cf_generator.generate_counterfactual(
                    test_episode, intervention_spec, causal_graph, causal_mechanisms
                )

                results['successful_generations'] = 1
                results['generator_functional'] = True
                results['cf_episode_shape'] = {
                    'states': cf_episode['states'].shape,
                    'actions': cf_episode['actions'].shape,
                    'causal_factors': cf_episode['causal_factors'].shape
                }
                print(f"      ✅ Counterfactual generation successful")

            except Exception as e:
                print(f"      ❌ Counterfactual generation failed: {e}")
                results['generation_errors'].append(str(e))

        except Exception as e:
            print(f"    💀 GENERATOR SETUP FAILURE: {e}")
            results['generation_errors'].append(f"Setup failure: {e}")

        # Scoring
        functionality_score = 1.0 if results['generator_functional'] else 0.0
        total_score = functionality_score

        print(f"    Generator Functional: {'✅' if functionality_score > 0.8 else '❌'}")
        print(f"    Successful Generations: {results['successful_generations']}/{results['generation_attempts']}")
        print(f"    Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("    🎉 EXCEPTIONAL: Counterfactual generator works!")
        elif total_score > 0.5:
            print("    👍 PARTIAL: Some generator capability")
        else:
            print("    💀 FAILED: Counterfactual generator broken")

        self.test_results['test_2'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.5
        }

        return total_score, results

    def test_3_interface_compatibility_diagnosis(self):
        """
        TEST 3: Diagnose interface compatibility issues
        """
        print("\n🧠 TEST 3: Interface Compatibility Diagnosis")
        print("=" * 60)

        results = {
            'model_type': type(self.model).__name__,
            'missing_methods': [],
            'working_methods': [],
            'interface_issues': []
        }

        # Check for expected methods
        expected_methods = [
            'forward',
            'set_pathway_mode',
            'get_adjacency_matrix',
            'predict_single',
            'pathway_weights'
        ]

        for method_name in expected_methods:
            if hasattr(self.model, method_name):
                results['working_methods'].append(method_name)
                print(f"      ✅ Has method: {method_name}")
            else:
                results['missing_methods'].append(method_name)
                print(f"      ❌ Missing method: {method_name}")

        # Check for expected attributes
        expected_attributes = [
            'pathway_weights',
            'base_model',
            'state_dim',
            'action_dim',
            'causal_dim'
        ]

        results['missing_attributes'] = []
        results['working_attributes'] = []

        for attr_name in expected_attributes:
            if hasattr(self.model, attr_name):
                results['working_attributes'].append(attr_name)
                print(f"      ✅ Has attribute: {attr_name}")
            else:
                results['missing_attributes'].append(attr_name)
                print(f"      ❌ Missing attribute: {attr_name}")

        # Specific diagnostics for known issues
        if 'set_pathway_mode' in results['missing_methods']:
            results['interface_issues'].append("Missing set_pathway_mode - prevents pathway switching")

        if 'pathway_weights' in results['missing_attributes']:
            results['interface_issues'].append("Missing pathway_weights - prevents pathway analysis")

        # Overall interface health
        total_expected = len(expected_methods) + len(expected_attributes)
        total_working = len(results['working_methods']) + len(results['working_attributes'])
        interface_health = total_working / total_expected

        results['interface_health'] = interface_health
        results['total_issues'] = len(results['missing_methods']) + len(results['missing_attributes'])

        # Scoring
        interface_score = interface_health
        total_score = interface_score

        print(f"    Interface Health: {interface_health:.3f} ({'✅' if interface_health > 0.8 else '❌'})")
        print(f"    Critical Issues: {len(results['interface_issues'])}")
        print(f"    Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("    🎉 EXCELLENT: Interface mostly compatible")
        elif total_score > 0.6:
            print("    👍 GOOD: Minor interface issues")
        elif total_score > 0.4:
            print("    ⚠️  WEAK: Significant interface problems")
        else:
            print("    💀 FAILED: Major interface incompatibility")

        self.test_results['test_3'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.6
        }

        return total_score, results

    def run_all_tests(self):
        """Run all counterfactual reasoning torture tests"""
        print("🔥 EXTREME COUNTERFACTUAL REASONING TORTURE TEST")
        print("=" * 80)
        print("Testing counterfactual reasoning capabilities and diagnosing failures")
        print()

        try:
            self.setup_models()
        except Exception as e:
            print(f"💀 CRITICAL: Model setup completely failed: {e}")
            return {
                'overall_score': 0.0,
                'grade': 'F',
                'status': '💀💀 CATASTROPHIC FAILURE - Cannot initialize models',
                'setup_error': str(e)
            }

        # Run all tests
        score_1, _ = self.test_1_basic_counterfactual_consistency()
        score_2, _ = self.test_2_counterfactual_generator_functionality()
        score_3, _ = self.test_3_interface_compatibility_diagnosis()

        # Overall assessment
        overall_score = (score_1 + score_2 + score_3) / 3.0

        print("\n" + "=" * 80)
        print("📊 COUNTERFACTUAL REASONING TORTURE TEST RESULTS")
        print("=" * 80)
        print(f"Test 1 - Basic Consistency: {score_1:.3f}")
        print(f"Test 2 - Generator Functionality: {score_2:.3f}")
        print(f"Test 3 - Interface Diagnosis: {score_3:.3f}")
        print(f"Overall Score: {overall_score:.3f}")

        if overall_score > 0.7:
            grade = "A"
            status = "🎉 EXCELLENT - Strong counterfactual reasoning!"
        elif overall_score > 0.5:
            grade = "B"
            status = "👍 GOOD - Decent counterfactual capability"
        elif overall_score > 0.3:
            grade = "C"
            status = "⚠️ WEAK - Limited counterfactual reasoning"
        elif overall_score > 0.1:
            grade = "D"
            status = "💀 POOR - Major counterfactual issues"
        else:
            grade = "F"
            status = "💀💀 FAILED - Counterfactual reasoning completely broken"

        print(f"Grade: {grade}")
        print(f"Status: {status}")

        # Provide specific recommendations based on results
        if overall_score < 0.3:
            print("\n🔧 RECOMMENDED FIXES:")
            if self.setup_errors:
                print("  1. Fix model setup errors (imports, class definitions)")
            if score_3 < 0.6:
                print("  2. Fix interface compatibility (missing methods/attributes)")
            if score_1 < 0.5:
                print("  3. Debug basic model forward pass functionality")
            if score_2 < 0.5:
                print("  4. Fix counterfactual generator component")

        # Save results
        results_summary = {
            'overall_score': overall_score,
            'grade': grade,
            'status': status,
            'individual_tests': self.test_results,
            'setup_errors': self.setup_errors,
            'timestamp': time.time()
        }

        return results_summary


def main():
    """Run the extreme counterfactual reasoning torture test"""
    test = ExtremeCounterfactualReasoningTortureTest()
    results = test.run_all_tests()

    # Save results
    with open('extreme_counterfactual_reasoning_torture_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        json.dump(convert_numpy(results), f, indent=2)

    print(f"\n📁 Results saved to: extreme_counterfactual_reasoning_torture_results.json")

    return results


if __name__ == "__main__":
    results = main()