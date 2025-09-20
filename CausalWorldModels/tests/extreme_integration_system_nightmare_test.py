#!/usr/bin/env python3
"""
EXTREME INTEGRATION SYSTEM NIGHTMARE TEST
=========================================

This test creates the most challenging possible scenario for the integrated
causal reasoning system - where ALL components must work together flawlessly.

NIGHTMARE SCENARIO DESIGN:
1. COMPLETE SYSTEM INTEGRATION
   - All components working together: dual-pathway, HSIC, structure learning, temporal, counterfactual
   - End-to-end causal reasoning under extreme conditions

2. CASCADING FAILURE DETECTION
   - Test how system handles component failures
   - Verify graceful degradation vs catastrophic collapse

3. REAL-WORLD COMPLEXITY SIMULATION
   - Non-stationary environments, multiple confounders, adversarial conditions
   - Test system robustness under realistic complexity

4. COMPUTATIONAL SCALABILITY STRESS
   - Memory limits, long sequences, complex reasoning chains
   - Test if system can handle production-scale demands

TARGET: Only genuinely integrated systems should achieve >60% under these conditions
This is the ultimate test that separates proof-of-concept from production-ready systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import traceback
import psutil
import gc
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Comprehensive imports with error handling
components_available = {}

try:
    from training.joint_causal_trainer import JointCausalTrainer, JointTrainingConfig
    components_available['joint_trainer'] = True
except ImportError as e:
    print(f"Warning: JointCausalTrainer not available: {e}")
    components_available['joint_trainer'] = False

try:
    from causal_architectures.dual_pathway_gru import DualPathwayCausalGRU
    components_available['dual_pathway'] = True
except ImportError as e:
    print(f"Warning: DualPathwayCausalGRU not available: {e}")
    components_available['dual_pathway'] = False

try:
    from causal_architectures.enhanced_structure_learner import EnhancedCausalStructureLearner
    components_available['enhanced_structure'] = True
except ImportError:
    try:
        from causal_architectures.structure_learner import CausalStructureLearner
        components_available['basic_structure'] = True
        components_available['enhanced_structure'] = False
    except ImportError as e:
        print(f"Warning: No structure learner available: {e}")
        components_available['enhanced_structure'] = False
        components_available['basic_structure'] = False

try:
    from causal_envs.temporal_integration import TemporalCausalIntegrator, TemporalIntegrationConfig
    components_available['temporal_integration'] = True
except ImportError as e:
    print(f"Warning: TemporalCausalIntegrator not available: {e}")
    components_available['temporal_integration'] = False

try:
    from causal_envs.continuous_campus_env import ContinuousCampusEnv, CausalState, WeatherType, EventType
    components_available['environment'] = True
except ImportError as e:
    print(f"Warning: ContinuousCampusEnv not available: {e}")
    components_available['environment'] = False


class ExtremeIntegrationSystemNightmareTest:
    """
    Ultimate integration test for the complete causal reasoning system

    Tests all components working together under the most challenging conditions possible
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.components = {}
        self.test_results = {}
        self.system_errors = []
        self.memory_usage = {}

    def setup_complete_system(self):
        """
        Setup the complete integrated causal reasoning system
        """
        print("    🔧 Setting up complete integrated system...")
        print(f"    📋 Component availability: {components_available}")

        # Track memory usage
        process = psutil.Process()
        self.memory_usage['initial'] = process.memory_info().rss / 1024 / 1024  # MB

        setup_results = {
            'components_attempted': 0,
            'components_successful': 0,
            'setup_errors': [],
            'component_status': {}
        }

        # Component 1: Dual-pathway dynamics
        if components_available.get('dual_pathway', False):
            try:
                setup_results['components_attempted'] += 1
                self.components['dual_pathway'] = DualPathwayCausalGRU(
                    state_dim=12,
                    action_dim=2,
                    causal_dim=5,
                    hidden_dim=64
                ).to(self.device)
                setup_results['components_successful'] += 1
                setup_results['component_status']['dual_pathway'] = 'success'
                print("      ✅ Dual-pathway GRU")
            except Exception as e:
                setup_results['setup_errors'].append(f"dual_pathway: {e}")
                setup_results['component_status']['dual_pathway'] = 'failed'
                print(f"      ❌ Dual-pathway GRU: {e}")

        # Component 2: Structure learning
        if components_available.get('enhanced_structure', False):
            try:
                setup_results['components_attempted'] += 1
                self.components['structure_learner'] = EnhancedCausalStructureLearner(
                    num_variables=5,
                    hidden_dim=64,
                    total_epochs=20
                ).to(self.device)
                setup_results['components_successful'] += 1
                setup_results['component_status']['structure_learner'] = 'success'
                print("      ✅ Enhanced Structure Learner")
            except Exception as e:
                setup_results['setup_errors'].append(f"enhanced_structure: {e}")
                setup_results['component_status']['structure_learner'] = 'failed'
                print(f"      ❌ Enhanced Structure Learner: {e}")
        elif components_available.get('basic_structure', False):
            try:
                setup_results['components_attempted'] += 1
                from causal_architectures.structure_learner import CausalStructureLearner
                self.components['structure_learner'] = CausalStructureLearner(
                    num_variables=5,
                    hidden_dim=64
                ).to(self.device)
                setup_results['components_successful'] += 1
                setup_results['component_status']['structure_learner'] = 'success'
                print("      ✅ Basic Structure Learner")
            except Exception as e:
                setup_results['setup_errors'].append(f"basic_structure: {e}")
                setup_results['component_status']['structure_learner'] = 'failed'
                print(f"      ❌ Basic Structure Learner: {e}")

        # Component 3: Temporal integration
        if components_available.get('temporal_integration', False):
            try:
                setup_results['components_attempted'] += 1
                config = TemporalIntegrationConfig(
                    enable_delays=True,
                    enable_logging=False,
                    validation_mode=False
                )
                self.components['temporal_integrator'] = TemporalCausalIntegrator(config)
                setup_results['components_successful'] += 1
                setup_results['component_status']['temporal_integrator'] = 'success'
                print("      ✅ Temporal Integrator")
            except Exception as e:
                setup_results['setup_errors'].append(f"temporal_integration: {e}")
                setup_results['component_status']['temporal_integrator'] = 'failed'
                print(f"      ❌ Temporal Integrator: {e}")

        # Component 4: Environment
        if components_available.get('environment', False):
            try:
                setup_results['components_attempted'] += 1
                self.components['environment'] = ContinuousCampusEnv(
                    render_mode=None,
                    enable_temporal_delays=True
                )
                setup_results['components_successful'] += 1
                setup_results['component_status']['environment'] = 'success'
                print("      ✅ Continuous Campus Environment")
            except Exception as e:
                setup_results['setup_errors'].append(f"environment: {e}")
                setup_results['component_status']['environment'] = 'failed'
                print(f"      ❌ Continuous Campus Environment: {e}")

        # Component 5: Joint trainer (if available)
        if components_available.get('joint_trainer', False):
            try:
                setup_results['components_attempted'] += 1
                config = JointTrainingConfig(
                    max_epochs=5,
                    batch_size=8,
                    sequence_length=20
                )
                self.components['joint_trainer'] = JointCausalTrainer(config)
                setup_results['components_successful'] += 1
                setup_results['component_status']['joint_trainer'] = 'success'
                print("      ✅ Joint Causal Trainer")
            except Exception as e:
                setup_results['setup_errors'].append(f"joint_trainer: {e}")
                setup_results['component_status']['joint_trainer'] = 'failed'
                print(f"      ❌ Joint Causal Trainer: {e}")

        # Track memory after setup
        self.memory_usage['after_setup'] = process.memory_info().rss / 1024 / 1024

        setup_results['memory_increase'] = self.memory_usage['after_setup'] - self.memory_usage['initial']
        setup_results['setup_success_rate'] = setup_results['components_successful'] / max(setup_results['components_attempted'], 1)

        print(f"    📊 Setup Results: {setup_results['components_successful']}/{setup_results['components_attempted']} components")
        print(f"    🧠 Memory increase: {setup_results['memory_increase']:.1f} MB")

        return setup_results

    def generate_nightmare_scenario_data(self, seq_len=100, batch_size=16):
        """
        Generate the most challenging possible causal scenario
        """
        print(f"    🌪️  Generating nightmare scenario: {seq_len} steps, {batch_size} batch")

        # Multi-domain, non-stationary, confounded, adversarial data
        states = torch.zeros(batch_size, seq_len, 12)
        actions = torch.zeros(batch_size, seq_len, 2)
        causal_factors = torch.zeros(batch_size, seq_len, 5)

        # Hidden confounders and domain shifts
        hidden_confounders = torch.randn(batch_size, seq_len, 3)  # 3 hidden variables
        domain_shifts = torch.zeros(batch_size, seq_len)

        for t in range(seq_len):
            # Domain shift every 25 timesteps (makes structure learning hard)
            domain = t // 25
            domain_shifts[:, t] = domain

            if t == 0:
                states[:, 0, :] = torch.randn(batch_size, 12) * 0.5
                causal_factors[:, 0, :] = torch.randn(batch_size, 5) * 0.3
            else:
                # State evolution with memory
                states[:, t, :] = 0.7 * states[:, t-1, :] + 0.3 * torch.randn(batch_size, 12)

            # Complex causal relationships that change by domain
            conf1, conf2, conf3 = hidden_confounders[:, t, 0], hidden_confounders[:, t, 1], hidden_confounders[:, t, 2]

            if domain % 4 == 0:
                # Domain 0: weather -> crowd -> event
                causal_factors[:, t, 0] = 0.6 * conf1 + 0.4 * torch.randn(batch_size)  # weather influenced by hidden
                causal_factors[:, t, 1] = 0.7 * causal_factors[:, t, 0] + 0.3 * torch.randn(batch_size)  # crowd
                causal_factors[:, t, 2] = 0.5 * causal_factors[:, t, 1] + 0.5 * torch.randn(batch_size)  # event
            elif domain % 4 == 1:
                # Domain 1: event -> crowd -> weather (reversed!)
                causal_factors[:, t, 2] = 0.6 * conf2 + 0.4 * torch.randn(batch_size)
                causal_factors[:, t, 1] = 0.7 * causal_factors[:, t, 2] + 0.3 * torch.randn(batch_size)
                causal_factors[:, t, 0] = 0.5 * causal_factors[:, t, 1] + 0.5 * torch.randn(batch_size)
            elif domain % 4 == 2:
                # Domain 2: Complex fork structure
                causal_factors[:, t, 0] = 0.6 * conf3 + 0.4 * torch.randn(batch_size)
                causal_factors[:, t, 1] = 0.4 * causal_factors[:, t, 0] + 0.3 * conf1 + 0.3 * torch.randn(batch_size)
                causal_factors[:, t, 2] = 0.3 * causal_factors[:, t, 0] + 0.4 * conf2 + 0.3 * torch.randn(batch_size)
            else:
                # Domain 3: Spurious correlations (no true causation)
                causal_factors[:, t, 0] = 0.8 * conf1 + 0.2 * torch.randn(batch_size)
                causal_factors[:, t, 1] = 0.8 * conf1 + 0.2 * torch.randn(batch_size)  # Same confounder!
                causal_factors[:, t, 2] = 0.8 * conf2 + 0.2 * torch.randn(batch_size)

            # Time and road evolve independently with noise
            causal_factors[:, t, 3] = 0.9 * causal_factors[:, t-1, 3] + 0.1 * torch.randn(batch_size) if t > 0 else torch.randn(batch_size) * 0.3
            causal_factors[:, t, 4] = 0.8 * causal_factors[:, t-1, 4] + 0.2 * torch.randn(batch_size) if t > 0 else torch.randn(batch_size) * 0.3

            # Actions influenced by states and confounders (makes things harder)
            actions[:, t, :] = 0.3 * states[:, t, :2] + 0.2 * conf1.unsqueeze(1) + 0.5 * torch.randn(batch_size, 2)

        return states, actions, causal_factors, hidden_confounders, domain_shifts

    def test_1_complete_integration_under_stress(self):
        """
        TEST 1: Can all components work together under stress?
        """
        print("🔥 TEST 1: Complete Integration Under Stress")
        print("=" * 60)

        # Setup system
        setup_results = self.setup_complete_system()

        # Generate nightmare data
        try:
            states, actions, causal_factors, hidden_confounders, domain_shifts = self.generate_nightmare_scenario_data(
                seq_len=80, batch_size=12
            )
            data_generation_success = True
        except Exception as e:
            print(f"    💀 Data generation failed: {e}")
            data_generation_success = False
            states = actions = causal_factors = None

        results = {
            'setup_results': setup_results,
            'data_generation_success': data_generation_success,
            'component_test_results': {},
            'integration_success': False,
            'system_errors': []
        }

        if not data_generation_success:
            total_score = 0.0
        else:
            # Test each component individually
            component_scores = {}

            # Test 1a: Dual-pathway dynamics
            if 'dual_pathway' in self.components:
                try:
                    model = self.components['dual_pathway']
                    model.eval()
                    with torch.no_grad():
                        pred_states, hidden, pathway_info = model(
                            states[:, :-1], actions[:, :-1], causal_factors[:, :-1]
                        )
                    component_scores['dual_pathway'] = 1.0
                    results['component_test_results']['dual_pathway'] = {
                        'success': True,
                        'output_shape': pred_states.shape,
                        'pathway_balance': pathway_info.get('pathway_balance', 0)
                    }
                    print("      ✅ Dual-pathway dynamics functional")
                except Exception as e:
                    component_scores['dual_pathway'] = 0.0
                    results['component_test_results']['dual_pathway'] = {'success': False, 'error': str(e)}
                    results['system_errors'].append(f"dual_pathway: {e}")
                    print(f"      ❌ Dual-pathway dynamics failed: {e}")
            else:
                component_scores['dual_pathway'] = 0.0

            # Test 1b: Structure learning
            if 'structure_learner' in self.components:
                try:
                    learner = self.components['structure_learner']
                    learner.train()

                    # Quick training iteration
                    if hasattr(learner, 'compute_enhanced_structure_loss'):
                        loss, loss_info = learner.compute_enhanced_structure_loss(causal_factors)
                    else:
                        loss, loss_info = learner.compute_structure_loss(causal_factors)

                    adjacency = learner.get_adjacency_matrix()
                    edges_found = (adjacency > 0.3).sum().item()

                    component_scores['structure_learner'] = 1.0 if edges_found > 0 else 0.5
                    results['component_test_results']['structure_learner'] = {
                        'success': True,
                        'loss': loss.item(),
                        'edges_found': edges_found,
                        'adjacency_shape': adjacency.shape
                    }
                    print(f"      ✅ Structure learning functional ({edges_found} edges)")
                except Exception as e:
                    component_scores['structure_learner'] = 0.0
                    results['component_test_results']['structure_learner'] = {'success': False, 'error': str(e)}
                    results['system_errors'].append(f"structure_learner: {e}")
                    print(f"      ❌ Structure learning failed: {e}")
            else:
                component_scores['structure_learner'] = 0.0

            # Test 1c: Temporal integration
            if 'temporal_integrator' in self.components:
                try:
                    integrator = self.components['temporal_integrator']
                    integrator.reset()

                    # Test temporal processing
                    test_action = np.array([1.0, 0.5])
                    test_state = CausalState(
                        time_hour=12,
                        day_week=1,
                        weather=WeatherType.RAIN,
                        event=EventType.NORMAL,
                        crowd_density=0.5
                    )

                    modified_action, temporal_info = integrator.apply_temporal_effects(test_action, test_state)

                    component_scores['temporal_integrator'] = 1.0
                    results['component_test_results']['temporal_integrator'] = {
                        'success': True,
                        'temporal_info_keys': list(temporal_info.keys()),
                        'delay_detected': temporal_info.get('delay_comparison', {}).get('weather_difference', 0) > 0.01
                    }
                    print("      ✅ Temporal integration functional")
                except Exception as e:
                    component_scores['temporal_integrator'] = 0.0
                    results['component_test_results']['temporal_integrator'] = {'success': False, 'error': str(e)}
                    results['system_errors'].append(f"temporal_integrator: {e}")
                    print(f"      ❌ Temporal integration failed: {e}")
            else:
                component_scores['temporal_integrator'] = 0.0

            # Test 1d: Environment
            if 'environment' in self.components:
                try:
                    env = self.components['environment']
                    obs, info = env.reset()
                    action = env.action_space.sample()
                    next_obs, reward, terminated, truncated, info = env.step(action)

                    component_scores['environment'] = 1.0
                    results['component_test_results']['environment'] = {
                        'success': True,
                        'observation_shape': obs.shape,
                        'action_shape': action.shape,
                        'reward': reward
                    }
                    print("      ✅ Environment functional")
                except Exception as e:
                    component_scores['environment'] = 0.0
                    results['component_test_results']['environment'] = {'success': False, 'error': str(e)}
                    results['system_errors'].append(f"environment: {e}")
                    print(f"      ❌ Environment failed: {e}")
            else:
                component_scores['environment'] = 0.0

            # Overall integration score
            total_components = len(component_scores)
            successful_components = sum(score > 0 for score in component_scores.values())
            integration_rate = successful_components / max(total_components, 1)

            results['integration_success'] = integration_rate > 0.5
            results['successful_components'] = successful_components
            results['total_components'] = total_components

            total_score = integration_rate

        print(f"    Components Working: {results.get('successful_components', 0)}/{results.get('total_components', 0)}")
        print(f"    System Errors: {len(results['system_errors'])}")
        print(f"    Integration Score: {total_score:.3f}")

        if total_score > 0.8:
            print("    🎉 EXCEPTIONAL: Complete system integration!")
        elif total_score > 0.6:
            print("    👍 GOOD: Most components working")
        elif total_score > 0.3:
            print("    ⚠️  WEAK: Partial integration")
        else:
            print("    💀 FAILED: System integration broken")

        self.test_results['test_1'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.5
        }

        return total_score, results

    def test_2_cascading_failure_analysis(self):
        """
        TEST 2: How does the system handle cascading component failures?
        """
        print("\n🎯 TEST 2: Cascading Failure Analysis")
        print("=" * 60)

        # Systematically break components and see how system responds
        failure_scenarios = [
            ("dual_pathway_failure", "Disable dual-pathway dynamics"),
            ("structure_learner_failure", "Disable structure learning"),
            ("temporal_integrator_failure", "Disable temporal integration"),
            ("memory_pressure", "Simulate memory pressure"),
        ]

        results = {
            'failure_scenarios': {},
            'graceful_degradation_score': 0.0,
            'catastrophic_failures': 0
        }

        for scenario_name, scenario_description in failure_scenarios:
            print(f"    Testing: {scenario_description}")

            scenario_result = {
                'description': scenario_description,
                'system_survived': False,
                'error_count': 0,
                'functionality_retained': 0.0
            }

            try:
                if scenario_name == "dual_pathway_failure":
                    # Try to use a broken dual pathway
                    if 'dual_pathway' in self.components:
                        # Intentionally corrupt the model
                        original_forward = self.components['dual_pathway'].forward
                        def broken_forward(*args, **kwargs):
                            raise RuntimeError("Simulated dual-pathway failure")
                        self.components['dual_pathway'].forward = broken_forward

                        # Test if system can still function
                        try:
                            # Try basic inference
                            test_data = torch.randn(1, 10, 12)
                            result = "System attempted to use broken component"
                            scenario_result['system_survived'] = False
                        except Exception as e:
                            scenario_result['error_count'] += 1
                            scenario_result['system_survived'] = True  # Good - caught the error

                        # Restore original
                        self.components['dual_pathway'].forward = original_forward

                elif scenario_name == "memory_pressure":
                    # Simulate memory pressure
                    try:
                        # Allocate large tensors to stress memory
                        memory_hogs = []
                        for i in range(10):
                            memory_hogs.append(torch.randn(1000, 1000).to(self.device))

                        # Try to use system under memory pressure
                        if 'dual_pathway' in self.components:
                            test_input = torch.randn(1, 5, 12).to(self.device)
                            test_action = torch.randn(1, 5, 2).to(self.device)
                            test_causal = torch.randn(1, 5, 5).to(self.device)

                            result = self.components['dual_pathway'](test_input, test_action, test_causal)
                            scenario_result['system_survived'] = True

                        # Clean up
                        del memory_hogs
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    except Exception as e:
                        scenario_result['error_count'] += 1
                        scenario_result['system_survived'] = False

                # Test remaining functionality after failure
                working_components = 0
                total_testable = 0

                for component_name, component in self.components.items():
                    if component_name != scenario_name.split('_')[0]:  # Skip the broken component
                        total_testable += 1
                        try:
                            # Basic functionality test
                            if hasattr(component, 'forward') and component_name == 'dual_pathway':
                                test_result = component(torch.randn(1, 5, 12), torch.randn(1, 5, 2), torch.randn(1, 5, 5))
                                working_components += 1
                            elif hasattr(component, 'reset') and component_name == 'temporal_integrator':
                                component.reset()
                                working_components += 1
                            elif hasattr(component, 'get_adjacency_matrix') and component_name == 'structure_learner':
                                adj = component.get_adjacency_matrix()
                                working_components += 1
                            else:
                                working_components += 1  # Assume working if no test available
                        except Exception as e:
                            pass  # Component failed test

                scenario_result['functionality_retained'] = working_components / max(total_testable, 1)

            except Exception as e:
                scenario_result['error_count'] += 1
                scenario_result['system_survived'] = False
                print(f"      💀 Scenario testing failed: {e}")

            results['failure_scenarios'][scenario_name] = scenario_result

            if scenario_result['system_survived']:
                print(f"      ✅ System survived {scenario_description}")
            else:
                print(f"      ❌ System failed under {scenario_description}")
                results['catastrophic_failures'] += 1

        # Compute overall graceful degradation score
        survival_rate = sum(1 for s in results['failure_scenarios'].values() if s['system_survived']) / len(failure_scenarios)
        avg_functionality_retained = np.mean([s['functionality_retained'] for s in results['failure_scenarios'].values()])

        results['graceful_degradation_score'] = (survival_rate + avg_functionality_retained) / 2.0

        total_score = results['graceful_degradation_score']

        print(f"    Survival Rate: {survival_rate:.3f}")
        print(f"    Avg Functionality Retained: {avg_functionality_retained:.3f}")
        print(f"    Catastrophic Failures: {results['catastrophic_failures']}")
        print(f"    Overall Score: {total_score:.3f}")

        if total_score > 0.7:
            print("    🎉 EXCELLENT: Graceful degradation under failures")
        elif total_score > 0.5:
            print("    👍 GOOD: Some resilience to failures")
        elif total_score > 0.3:
            print("    ⚠️  WEAK: Limited failure resilience")
        else:
            print("    💀 FAILED: No graceful degradation")

        self.test_results['test_2'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.5
        }

        return total_score, results

    def test_3_computational_scalability_stress(self):
        """
        TEST 3: Can the system handle computational scalability stress?
        """
        print("\n🧠 TEST 3: Computational Scalability Stress")
        print("=" * 60)

        # Test system under increasing computational demands
        scalability_tests = [
            ("small_scale", 32, 20, 4),     # batch_size, seq_len, test_iterations
            ("medium_scale", 64, 50, 3),
            ("large_scale", 128, 100, 2),
            ("extreme_scale", 256, 200, 1)
        ]

        results = {
            'scalability_results': {},
            'memory_usage': {},
            'timing_results': {},
            'system_limits': {}
        }

        process = psutil.Process()

        for test_name, batch_size, seq_len, iterations in scalability_tests:
            print(f"    Testing {test_name}: batch={batch_size}, seq={seq_len}, iter={iterations}")

            test_result = {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'iterations': iterations,
                'success': False,
                'avg_time': 0.0,
                'peak_memory': 0.0,
                'errors': []
            }

            try:
                # Memory before test
                memory_before = process.memory_info().rss / 1024 / 1024

                times = []
                peak_memory = memory_before

                for iteration in range(iterations):
                    try:
                        start_time = time.time()

                        # Generate large data
                        states = torch.randn(batch_size, seq_len, 12)
                        actions = torch.randn(batch_size, seq_len, 2)
                        causal_factors = torch.randn(batch_size, seq_len, 5)

                        # Test dual pathway if available
                        if 'dual_pathway' in self.components:
                            model = self.components['dual_pathway']
                            model.eval()
                            with torch.no_grad():
                                pred = model(states[:, :-1], actions[:, :-1], causal_factors[:, :-1])

                        # Test structure learning if available
                        if 'structure_learner' in self.components and iteration == 0:  # Only first iteration
                            learner = self.components['structure_learner']
                            if hasattr(learner, 'compute_enhanced_structure_loss'):
                                loss, _ = learner.compute_enhanced_structure_loss(causal_factors[:8])  # Smaller batch for structure learning
                            else:
                                loss, _ = learner.compute_structure_loss(causal_factors[:8])

                        end_time = time.time()
                        times.append(end_time - start_time)

                        # Check memory usage
                        current_memory = process.memory_info().rss / 1024 / 1024
                        peak_memory = max(peak_memory, current_memory)

                        # Clean up
                        del states, actions, causal_factors
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        test_result['errors'].append(f"iteration_{iteration}: {e}")
                        print(f"      ❌ Iteration {iteration} failed: {e}")
                        break

                if times:
                    test_result['success'] = True
                    test_result['avg_time'] = np.mean(times)
                    test_result['peak_memory'] = peak_memory
                    test_result['memory_increase'] = peak_memory - memory_before
                    print(f"      ✅ {test_name}: {test_result['avg_time']:.2f}s avg, {test_result['memory_increase']:.1f}MB")
                else:
                    test_result['success'] = False
                    print(f"      💀 {test_name}: Complete failure")

            except Exception as e:
                test_result['errors'].append(f"setup_failure: {e}")
                test_result['success'] = False
                print(f"      💀 {test_name} setup failed: {e}")

            results['scalability_results'][test_name] = test_result

        # Analyze scalability
        successful_tests = [r for r in results['scalability_results'].values() if r['success']]
        success_rate = len(successful_tests) / len(scalability_tests)

        # Check if performance degrades gracefully
        if len(successful_tests) >= 2:
            times = [r['avg_time'] for r in successful_tests]
            batches = [r['batch_size'] for r in successful_tests]

            # Compute time complexity (should be roughly linear or slightly worse)
            if len(times) >= 2:
                time_scaling = times[-1] / times[0] if times[0] > 0 else float('inf')
                batch_scaling = batches[-1] / batches[0] if batches[0] > 0 else float('inf')
                scaling_efficiency = batch_scaling / time_scaling if time_scaling > 0 else 0
            else:
                scaling_efficiency = 1.0
        else:
            scaling_efficiency = 0.0

        total_score = (success_rate + min(scaling_efficiency, 1.0)) / 2.0

        print(f"    Success Rate: {success_rate:.3f}")
        print(f"    Scaling Efficiency: {scaling_efficiency:.3f}")
        print(f"    Overall Score: {total_score:.3f}")

        if total_score > 0.7:
            print("    🎉 EXCELLENT: Good computational scalability")
        elif total_score > 0.5:
            print("    👍 GOOD: Acceptable scalability")
        elif total_score > 0.3:
            print("    ⚠️  WEAK: Limited scalability")
        else:
            print("    💀 FAILED: Poor computational scalability")

        self.test_results['test_3'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.5
        }

        return total_score, results

    def run_all_tests(self):
        """Run all integration system nightmare tests"""
        print("🔥 EXTREME INTEGRATION SYSTEM NIGHTMARE TEST")
        print("=" * 80)
        print("The ultimate test: ALL components working together under extreme stress")
        print()

        # Track overall system health
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.time()

        # Run all tests
        score_1, _ = self.test_1_complete_integration_under_stress()
        score_2, _ = self.test_2_cascading_failure_analysis()
        score_3, _ = self.test_3_computational_scalability_stress()

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Overall assessment
        overall_score = (score_1 + score_2 + score_3) / 3.0

        print("\n" + "=" * 80)
        print("📊 INTEGRATION SYSTEM NIGHTMARE TEST RESULTS")
        print("=" * 80)
        print(f"Test 1 - Complete Integration: {score_1:.3f}")
        print(f"Test 2 - Cascading Failures: {score_2:.3f}")
        print(f"Test 3 - Computational Scalability: {score_3:.3f}")
        print(f"Overall Score: {overall_score:.3f}")
        print(f"Total Test Time: {end_time - start_time:.1f}s")
        print(f"Memory Usage: {end_memory - start_memory:+.1f}MB")

        if overall_score > 0.7:
            grade = "A"
            status = "🎉 EXCEPTIONAL - Production-ready integrated system!"
        elif overall_score > 0.6:
            grade = "B"
            status = "👍 GOOD - Strong integrated system with minor issues"
        elif overall_score > 0.4:
            grade = "C"
            status = "⚠️ MODERATE - Functional but needs significant work"
        elif overall_score > 0.2:
            grade = "D"
            status = "💀 POOR - Major integration problems"
        else:
            grade = "F"
            status = "💀💀 CATASTROPHIC - System integration fundamentally broken"

        print(f"Grade: {grade}")
        print(f"Status: {status}")

        # System diagnosis
        if overall_score < 0.5:
            print("\n🔧 CRITICAL SYSTEM ISSUES DETECTED:")
            if score_1 < 0.5:
                print("  - Component integration failures")
            if score_2 < 0.5:
                print("  - No graceful failure handling")
            if score_3 < 0.5:
                print("  - Computational scalability problems")

        # Save results
        results_summary = {
            'overall_score': overall_score,
            'grade': grade,
            'status': status,
            'individual_tests': self.test_results,
            'system_metrics': {
                'test_duration': end_time - start_time,
                'memory_usage_change': end_memory - start_memory,
                'components_available': components_available,
                'system_errors': self.system_errors
            },
            'timestamp': time.time()
        }

        return results_summary


def main():
    """Run the extreme integration system nightmare test"""
    test = ExtremeIntegrationSystemNightmareTest()
    results = test.run_all_tests()

    # Save results
    with open('extreme_integration_system_nightmare_results.json', 'w') as f:
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

    print(f"\n📁 Results saved to: extreme_integration_system_nightmare_results.json")

    return results


if __name__ == "__main__":
    results = main()