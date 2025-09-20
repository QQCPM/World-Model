#!/usr/bin/env python3
"""
EXTREME CAUSAL REASONING CHALLENGE
The Ultimate Test for Genuine Causal Understanding

This test goes far beyond standard validation to probe the deepest aspects
of causal reasoning that distinguish genuine understanding from pattern matching.

Tests Include:
1. Multi-step temporal causal chains with confounders
2. Adversarial scenarios designed to fool correlational models
3. Cross-domain causal transfer and generalization
4. Mechanism isolation under extreme stress
5. Counterfactual reasoning with incomplete information
6. Causal invariance across distributional shifts
7. Compositional causal understanding
8. Meta-causal reasoning (causality about causality)

Target: True causal AI must achieve >80% on these extreme challenges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Import your components
from causal_architectures import DualPathwayCausalGRU, CausalMechanismModules, CausalStructureLearner
from training import JointCausalTrainer, JointTrainingConfig
from validation import CausalReasonerTester


@dataclass
class ExtremeChallengeConfig:
    """Configuration for extreme causal challenges"""
    num_test_scenarios: int = 50
    noise_levels: List[float] = None
    confounding_strength: float = 0.8
    temporal_complexity: int = 5  # Multi-step chains
    adversarial_strength: float = 0.9
    cross_domain_shift: float = 0.7

    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.2, 0.5, 0.8, 1.0]  # Extreme noise


class ExtremeCausalChallenger:
    """
    The ultimate causal reasoning challenge suite

    Tests genuine causal understanding through adversarial scenarios
    that would fool any system relying on correlational patterns
    """

    def __init__(self, config: ExtremeChallengeConfig = None):
        self.config = config or ExtremeChallengeConfig()

        # Challenge results
        self.challenge_results = {}

        # Extreme test scenarios
        self.scenario_generators = {
            'temporal_causal_chains': self._generate_temporal_chain_scenario,
            'confounding_adversarial': self._generate_confounding_scenario,
            'cross_domain_transfer': self._generate_cross_domain_scenario,
            'mechanism_stress_test': self._generate_mechanism_stress_scenario,
            'counterfactual_reasoning': self._generate_counterfactual_scenario,
            'causal_invariance': self._generate_invariance_scenario,
            'compositional_causality': self._generate_compositional_scenario,
            'meta_causal_reasoning': self._generate_meta_causal_scenario
        }

    def run_extreme_challenge(self, causal_system):
        """
        Run the complete extreme causal reasoning challenge

        Args:
            causal_system: Your trained causal system (joint trainer)

        Returns:
            challenge_report: Comprehensive assessment of causal capabilities
        """
        print("🔥 EXTREME CAUSAL REASONING CHALLENGE")
        print("=" * 60)
        print("Testing the limits of genuine causal understanding...")
        print("⚠️  WARNING: These tests are designed to break weak models!")

        start_time = time.time()

        # Run all extreme challenges
        challenge_scores = {}

        for challenge_name, generator in self.scenario_generators.items():
            print(f"\n🎯 Challenge: {challenge_name.upper()}")
            print("-" * 40)

            score = self._run_single_challenge(challenge_name, generator, causal_system)
            challenge_scores[challenge_name] = score

            print(f"   Result: {score:.3f} (Target: >0.8)")
            status = "✅ PASSED" if score > 0.8 else "❌ FAILED" if score < 0.5 else "⚠️  WEAK"
            print(f"   Status: {status}")

        # Compute overall challenge score
        overall_score = np.mean(list(challenge_scores.values()))
        elapsed_time = time.time() - start_time

        # Generate final assessment
        challenge_report = self._generate_challenge_report(
            challenge_scores, overall_score, elapsed_time
        )

        # Print final results
        self._print_final_assessment(challenge_report)

        return challenge_report

    def _run_single_challenge(self, challenge_name: str, generator, causal_system):
        """Run a single extreme challenge"""

        scenario_scores = []

        for i in range(self.config.num_test_scenarios):
            try:
                # Generate challenging scenario
                scenario = generator()

                # Test causal system on scenario
                score = self._evaluate_scenario(scenario, causal_system, challenge_name)
                scenario_scores.append(score)

                if i % 10 == 0:
                    print(f"   Scenario {i+1}/{self.config.num_test_scenarios}: {score:.3f}")

            except Exception as e:
                print(f"   ⚠️  Scenario {i+1} failed: {e}")
                scenario_scores.append(0.0)

        return np.mean(scenario_scores) if scenario_scores else 0.0

    def _generate_temporal_chain_scenario(self):
        """
        Generate multi-step temporal causal chains with delays
        Tests: Can the system handle complex temporal dependencies?
        """
        seq_len = 20
        chain_length = self.config.temporal_complexity

        # Create base scenario
        scenario = {
            'type': 'temporal_chain',
            'states': torch.randn(1, seq_len, 12),
            'actions': torch.randn(1, seq_len, 2),
            'causal_factors': torch.randn(1, seq_len, 5)
        }

        # Inject temporal causal chain: A(t) -> B(t+1) -> C(t+2) -> D(t+3) -> E(t+4)
        for t in range(seq_len - chain_length):
            # Weather affects crowd with 1-step delay
            scenario['causal_factors'][0, t+1, 1] = 0.7 * scenario['causal_factors'][0, t, 0] + 0.3 * torch.randn(1)

            # Crowd affects events with 1-step delay
            scenario['causal_factors'][0, t+2, 2] = 0.6 * scenario['causal_factors'][0, t+1, 1] + 0.4 * torch.randn(1)

            # Events affect time patterns with 1-step delay
            scenario['causal_factors'][0, t+3, 3] = 0.5 * scenario['causal_factors'][0, t+2, 2] + 0.5 * torch.randn(1)

            # Time affects road conditions with 1-step delay
            scenario['causal_factors'][0, t+4, 4] = 0.8 * scenario['causal_factors'][0, t+3, 3] + 0.2 * torch.randn(1)

        # Add challenge: System must detect this 5-step chain
        scenario['ground_truth'] = {
            'chain_length': chain_length,
            'causal_edges': [(0,1), (1,2), (2,3), (3,4)],  # Weather->Crowd->Event->Time->Road
            'delays': [1, 1, 1, 1]
        }

        return scenario

    def _generate_confounding_scenario(self):
        """
        Generate adversarial confounding scenario
        Tests: Can the system distinguish correlation from causation?
        """
        seq_len = 20

        scenario = {
            'type': 'confounding_adversarial',
            'states': torch.randn(1, seq_len, 12),
            'actions': torch.randn(1, seq_len, 2),
            'causal_factors': torch.randn(1, seq_len, 5)
        }

        # Create confounding scenario:
        # Hidden confounder Z affects both X and Y, creating spurious correlation
        hidden_confounder = torch.randn(seq_len)

        for t in range(seq_len):
            # Hidden confounder affects weather (X)
            scenario['causal_factors'][0, t, 0] = 0.8 * hidden_confounder[t] + 0.2 * torch.randn(1)

            # Same hidden confounder affects crowd (Y) - creating spurious correlation
            scenario['causal_factors'][0, t, 1] = 0.7 * hidden_confounder[t] + 0.3 * torch.randn(1)

            # But weather does NOT directly cause crowd (this is the test)
            # A correlational model would incorrectly infer Weather -> Crowd

        # The challenge: System must NOT infer direct causal relationship between weather and crowd
        scenario['ground_truth'] = {
            'spurious_correlation': (0, 1),  # Weather-Crowd correlation
            'true_causal_edges': [],  # No direct causation
            'confounded': True,
            'confounder_strength': self.config.confounding_strength
        }

        return scenario

    def _generate_cross_domain_scenario(self):
        """
        Generate cross-domain causal transfer scenario
        Tests: Does causal understanding generalize across domains?
        """
        seq_len = 15

        # Create scenario with different causal variable meanings
        scenario = {
            'type': 'cross_domain',
            'states': torch.randn(1, seq_len, 12),
            'actions': torch.randn(1, seq_len, 2),
            'causal_factors': torch.randn(1, seq_len, 5)
        }

        # Map causal variables to different domain:
        # 0: Stock market volatility (was weather)
        # 1: Trading volume (was crowd)
        # 2: News events (was campus events)
        # 3: Market hours (was time)
        # 4: Network latency (was road conditions)

        for t in range(seq_len - 2):
            # Financial domain causality: News -> Volatility -> Volume
            scenario['causal_factors'][0, t+1, 0] = 0.6 * scenario['causal_factors'][0, t, 2] + 0.4 * torch.randn(1)
            scenario['causal_factors'][0, t+2, 1] = 0.7 * scenario['causal_factors'][0, t+1, 0] + 0.3 * torch.randn(1)

        # Challenge: Can the system apply causal understanding to new domain?
        scenario['ground_truth'] = {
            'domain_shift': self.config.cross_domain_shift,
            'causal_structure': [(2,0), (0,1)],  # News->Volatility->Volume
            'domain': 'financial'
        }

        return scenario

    def _generate_mechanism_stress_scenario(self):
        """
        Generate mechanism isolation stress test
        Tests: Can mechanisms maintain independence under extreme conditions?
        """
        seq_len = 20
        batch_size = 8  # Stress test with larger batch

        scenario = {
            'type': 'mechanism_stress',
            'states': torch.randn(batch_size, seq_len, 12),
            'actions': torch.randn(batch_size, seq_len, 2),
            'causal_factors': torch.randn(batch_size, seq_len, 5)
        }

        # Create stress conditions:
        # 1. High noise
        noise_level = random.choice([0.5, 0.8, 1.0])
        scenario['causal_factors'] += noise_level * torch.randn_like(scenario['causal_factors'])

        # 2. Correlated inputs (challenge mechanism independence)
        for b in range(batch_size):
            correlation_strength = 0.9
            for t in range(seq_len):
                # Force high correlation between different factors
                base_factor = scenario['causal_factors'][b, t, 0]
                scenario['causal_factors'][b, t, 1] = correlation_strength * base_factor + (1-correlation_strength) * scenario['causal_factors'][b, t, 1]
                scenario['causal_factors'][b, t, 2] = correlation_strength * base_factor + (1-correlation_strength) * scenario['causal_factors'][b, t, 2]

        # 3. Extreme values
        scenario['causal_factors'] = torch.clamp(scenario['causal_factors'], -3.0, 3.0)

        scenario['ground_truth'] = {
            'stress_level': 'extreme',
            'noise_level': noise_level,
            'correlation_challenge': True,
            'target_independence': 0.8  # Must maintain high independence despite stress
        }

        return scenario

    def _generate_counterfactual_scenario(self):
        """
        Generate counterfactual reasoning under uncertainty
        Tests: Can the system reason about alternative histories?
        """
        seq_len = 15

        # Base scenario
        scenario = {
            'type': 'counterfactual_reasoning',
            'states': torch.randn(1, seq_len, 12),
            'actions': torch.randn(1, seq_len, 2),
            'causal_factors': torch.randn(1, seq_len, 5)
        }

        # Create factual trajectory with known causation
        for t in range(seq_len - 1):
            # Weather affects crowd density with delay
            scenario['causal_factors'][0, t+1, 1] = 0.8 * scenario['causal_factors'][0, t, 0] + 0.2 * torch.randn(1)

        # Define counterfactual intervention at timestep 10
        intervention_time = 10
        intervention = {
            'variable': 0,  # Weather
            'value': 1.5,   # Set to extreme value
            'time': intervention_time
        }

        # Generate counterfactual trajectory
        cf_causal_factors = scenario['causal_factors'].clone()
        cf_causal_factors[0, intervention_time, 0] = intervention['value']

        # Propagate counterfactual effects
        for t in range(intervention_time, seq_len - 1):
            cf_causal_factors[0, t+1, 1] = 0.8 * cf_causal_factors[0, t, 0] + 0.2 * torch.randn(1)

        scenario['counterfactual'] = {
            'intervention': intervention,
            'cf_trajectory': cf_causal_factors,
            'factual_trajectory': scenario['causal_factors']
        }

        scenario['ground_truth'] = {
            'consistency_required': True,
            'temporal_consistency': True,
            'intervention_propagation': True
        }

        return scenario

    def _generate_invariance_scenario(self):
        """
        Generate causal invariance under distributional shift
        Tests: Does causal understanding remain stable across contexts?
        """
        seq_len = 18

        scenario = {
            'type': 'causal_invariance',
            'states': torch.randn(1, seq_len, 12),
            'actions': torch.randn(1, seq_len, 2),
            'causal_factors': torch.randn(1, seq_len, 5)
        }

        # Create base causal relationship: Event -> Crowd
        base_relationship_strength = 0.7

        # Apply distributional shifts while maintaining causal relationship
        shift_types = ['scale', 'offset', 'noise', 'nonlinear']
        shift_type = random.choice(shift_types)

        for t in range(seq_len - 1):
            base_effect = base_relationship_strength * scenario['causal_factors'][0, t, 2]  # Event effect

            if shift_type == 'scale':
                # Scale shift - different magnitudes but same relationship
                scale_factor = 0.3 + 1.4 * (t / seq_len)  # Varying scale
                scenario['causal_factors'][0, t+1, 1] = scale_factor * base_effect + 0.3 * torch.randn(1)

            elif shift_type == 'offset':
                # Offset shift - different baseline but same relationship
                offset = 2.0 * torch.sin(torch.tensor(2 * np.pi * t / seq_len))  # Varying offset
                scenario['causal_factors'][0, t+1, 1] = base_effect + offset + 0.3 * torch.randn(1)

            elif shift_type == 'noise':
                # Noise shift - increasing noise but same relationship
                noise_level = 0.1 + 0.8 * (t / seq_len)  # Increasing noise
                scenario['causal_factors'][0, t+1, 1] = base_effect + noise_level * torch.randn(1)

            elif shift_type == 'nonlinear':
                # Nonlinear transformation but same causal direction
                transformed_effect = torch.tanh(base_effect * 2.0)  # Nonlinear transform
                scenario['causal_factors'][0, t+1, 1] = transformed_effect + 0.2 * torch.randn(1)

        scenario['ground_truth'] = {
            'invariant_relationship': (2, 1),  # Event -> Crowd must be detected despite shifts
            'shift_type': shift_type,
            'causal_strength': base_relationship_strength,
            'invariance_required': True
        }

        return scenario

    def _generate_compositional_scenario(self):
        """
        Generate compositional causal understanding test
        Tests: Can the system understand how multiple mechanisms compose?
        """
        seq_len = 20

        scenario = {
            'type': 'compositional_causality',
            'states': torch.randn(1, seq_len, 12),
            'actions': torch.randn(1, seq_len, 2),
            'causal_factors': torch.randn(1, seq_len, 5)
        }

        # Create compositional causal structure:
        # Weather + Time jointly affect Crowd
        # Event moderates the Weather->Crowd relationship

        for t in range(seq_len - 1):
            weather = scenario['causal_factors'][0, t, 0]
            time_factor = scenario['causal_factors'][0, t, 3]
            event = scenario['causal_factors'][0, t, 2]

            # Compositional effect: Weather and Time jointly determine Crowd
            joint_effect = 0.4 * weather + 0.3 * time_factor

            # Event acts as a moderator (interaction effect)
            moderation = event * 0.2 * weather  # Event strengthens weather effect

            # Final crowd density
            scenario['causal_factors'][0, t+1, 1] = joint_effect + moderation + 0.2 * torch.randn(1)

        scenario['ground_truth'] = {
            'compositional_structure': True,
            'joint_causes': [(0, 3), 1],  # Weather + Time -> Crowd
            'moderator': (2, 0, 1),       # Event moderates Weather -> Crowd
            'interaction_effect': True
        }

        return scenario

    def _generate_meta_causal_scenario(self):
        """
        Generate meta-causal reasoning test
        Tests: Can the system reason about causality itself?
        """
        seq_len = 16

        scenario = {
            'type': 'meta_causal',
            'states': torch.randn(1, seq_len, 12),
            'actions': torch.randn(1, seq_len, 2),
            'causal_factors': torch.randn(1, seq_len, 5)
        }

        # Create scenario where causal relationships change over time
        # First half: Weather -> Crowd
        # Second half: Time -> Crowd (relationship switches)

        switch_point = seq_len // 2

        for t in range(seq_len - 1):
            if t < switch_point:
                # Phase 1: Weather dominates
                scenario['causal_factors'][0, t+1, 1] = 0.8 * scenario['causal_factors'][0, t, 0] + 0.2 * torch.randn(1)
            else:
                # Phase 2: Time dominates (causal structure changes)
                scenario['causal_factors'][0, t+1, 1] = 0.8 * scenario['causal_factors'][0, t, 3] + 0.2 * torch.randn(1)

        scenario['ground_truth'] = {
            'causal_structure_change': True,
            'switch_point': switch_point,
            'phase1_causation': (0, 1),  # Weather -> Crowd
            'phase2_causation': (3, 1),  # Time -> Crowd
            'meta_reasoning_required': True
        }

        return scenario

    def _evaluate_scenario(self, scenario, causal_system, challenge_type):
        """Evaluate causal system on specific scenario"""

        if challenge_type == 'temporal_causal_chains':
            return self._evaluate_temporal_chains(scenario, causal_system)
        elif challenge_type == 'confounding_adversarial':
            return self._evaluate_confounding(scenario, causal_system)
        elif challenge_type == 'cross_domain_transfer':
            return self._evaluate_cross_domain(scenario, causal_system)
        elif challenge_type == 'mechanism_stress_test':
            return self._evaluate_mechanism_stress(scenario, causal_system)
        elif challenge_type == 'counterfactual_reasoning':
            return self._evaluate_counterfactual(scenario, causal_system)
        elif challenge_type == 'causal_invariance':
            return self._evaluate_invariance(scenario, causal_system)
        elif challenge_type == 'compositional_causality':
            return self._evaluate_compositional(scenario, causal_system)
        elif challenge_type == 'meta_causal_reasoning':
            return self._evaluate_meta_causal(scenario, causal_system)
        else:
            return 0.0

    def _evaluate_temporal_chains(self, scenario, causal_system):
        """Evaluate temporal causal chain detection"""
        try:
            # Test if system can detect multi-step causal chains
            states = scenario['states']
            actions = scenario['actions']
            causal_factors = scenario['causal_factors']

            # Use structure learner to infer causal graph
            structure_learner = causal_system.structure_learner
            loss, loss_info = structure_learner.compute_structure_loss(causal_factors)

            # Get learned adjacency matrix
            learned_graph = structure_learner.get_adjacency_matrix()

            # Check if learned graph captures the temporal chain
            ground_truth_edges = scenario['ground_truth']['causal_edges']
            chain_detection_score = 0.0

            for (cause, effect) in ground_truth_edges:
                edge_strength = learned_graph[cause, effect].item()
                if edge_strength > 0.3:  # Threshold for edge detection
                    chain_detection_score += edge_strength

            # Normalize by chain length
            return chain_detection_score / len(ground_truth_edges)

        except Exception as e:
            print(f"Error in temporal chain evaluation: {e}")
            return 0.0

    def _evaluate_confounding(self, scenario, causal_system):
        """Evaluate confounding resistance"""
        try:
            causal_factors = scenario['causal_factors']

            # Test if system incorrectly infers spurious relationship
            structure_learner = causal_system.structure_learner
            loss, loss_info = structure_learner.compute_structure_loss(causal_factors)
            learned_graph = structure_learner.get_adjacency_matrix()

            # Check spurious correlation edge (should be weak or absent)
            spurious_edge = scenario['ground_truth']['spurious_correlation']
            spurious_strength = learned_graph[spurious_edge[0], spurious_edge[1]].item()

            # Good causal reasoning should NOT infer strong spurious edge
            resistance_score = 1.0 - min(spurious_strength, 1.0)

            return resistance_score

        except Exception as e:
            print(f"Error in confounding evaluation: {e}")
            return 0.0

    def _evaluate_cross_domain(self, scenario, causal_system):
        """Evaluate cross-domain generalization"""
        try:
            # Test if causal understanding transfers to new domain
            states = scenario['states']
            actions = scenario['actions']
            causal_factors = scenario['causal_factors']

            # Forward through dynamics model
            dynamics_model = causal_system.dynamics_model
            predicted_states, _, pathway_info = dynamics_model(
                states[:, :-1], actions[:, :-1], causal_factors[:, :-1]
            )

            # Evaluate prediction quality (should maintain accuracy despite domain shift)
            target_states = states[:, 1:]
            prediction_error = F.mse_loss(predicted_states, target_states).item()

            # Lower error = better generalization
            generalization_score = 1.0 / (1.0 + prediction_error)

            return generalization_score

        except Exception as e:
            print(f"Error in cross-domain evaluation: {e}")
            return 0.0

    def _evaluate_mechanism_stress(self, scenario, causal_system):
        """Evaluate mechanism independence under stress"""
        try:
            states = scenario['states']
            actions = scenario['actions']
            causal_factors = scenario['causal_factors']

            # Test mechanism isolation under stress conditions
            causal_mechanisms = causal_system.causal_mechanisms

            batch_states = states.reshape(-1, states.shape[-1])
            batch_causal = causal_factors.reshape(-1, causal_factors.shape[-1])
            batch_actions = actions.reshape(-1, actions.shape[-1])

            mechanism_effects, composed_effects, next_state, independence_loss, isolation_confidence = causal_mechanisms(
                batch_states, batch_causal, batch_actions
            )

            # High isolation confidence under stress = good mechanism design
            stress_resistance = isolation_confidence

            return stress_resistance

        except Exception as e:
            print(f"Error in mechanism stress evaluation: {e}")
            return 0.0

    def _evaluate_counterfactual(self, scenario, causal_system):
        """Evaluate counterfactual reasoning consistency"""
        try:
            # Test counterfactual consistency
            factual_trajectory = scenario['causal_factors']
            cf_trajectory = scenario['counterfactual']['cf_trajectory']
            intervention = scenario['counterfactual']['intervention']

            # Generate counterfactual with system
            dynamics_model = causal_system.dynamics_model

            # Factual prediction
            factual_pred, _, _ = dynamics_model(
                factual_trajectory[:, :-1],
                scenario['actions'][:, :-1],
                factual_trajectory[:, :-1]
            )

            # Counterfactual prediction
            cf_pred, _, _ = dynamics_model(
                cf_trajectory[:, :-1],
                scenario['actions'][:, :-1],
                cf_trajectory[:, :-1]
            )

            # Measure consistency: predictions should differ appropriately
            intervention_time = intervention['time']

            # Before intervention: should be similar
            pre_diff = F.mse_loss(
                factual_pred[:, :intervention_time],
                cf_pred[:, :intervention_time]
            ).item()

            # After intervention: should differ
            post_diff = F.mse_loss(
                factual_pred[:, intervention_time:],
                cf_pred[:, intervention_time:]
            ).item()

            # Good counterfactual reasoning: low pre_diff, higher post_diff
            consistency_score = (1.0 / (1.0 + pre_diff)) * min(post_diff, 1.0)

            return consistency_score

        except Exception as e:
            print(f"Error in counterfactual evaluation: {e}")
            return 0.0

    def _evaluate_invariance(self, scenario, causal_system):
        """Evaluate causal invariance under distributional shift"""
        try:
            causal_factors = scenario['causal_factors']

            # Test if causal relationship is detected despite distributional shift
            structure_learner = causal_system.structure_learner
            loss, loss_info = structure_learner.compute_structure_loss(causal_factors)
            learned_graph = structure_learner.get_adjacency_matrix()

            # Check if invariant relationship is detected
            invariant_edge = scenario['ground_truth']['invariant_relationship']
            detected_strength = learned_graph[invariant_edge[0], invariant_edge[1]].item()

            # Should detect relationship despite distributional shifts
            invariance_score = min(detected_strength, 1.0)

            return invariance_score

        except Exception as e:
            print(f"Error in invariance evaluation: {e}")
            return 0.0

    def _evaluate_compositional(self, scenario, causal_system):
        """Evaluate compositional causal understanding"""
        try:
            states = scenario['states']
            actions = scenario['actions']
            causal_factors = scenario['causal_factors']

            # Test compositional mechanism understanding
            causal_mechanisms = causal_system.causal_mechanisms

            batch_states = states.reshape(-1, states.shape[-1])
            batch_causal = causal_factors.reshape(-1, causal_factors.shape[-1])
            batch_actions = actions.reshape(-1, actions.shape[-1])

            mechanism_effects, composed_effects, next_state, independence_loss, isolation_confidence = causal_mechanisms(
                batch_states, batch_causal, batch_actions
            )

            # Evaluate if composition produces reasonable effects
            composition_quality = 1.0 - F.mse_loss(
                composed_effects,
                torch.zeros_like(composed_effects)
            ).item()

            # Normalize composition score
            composition_score = min(composition_quality, 1.0)

            return composition_score

        except Exception as e:
            print(f"Error in compositional evaluation: {e}")
            return 0.0

    def _evaluate_meta_causal(self, scenario, causal_system):
        """Evaluate meta-causal reasoning"""
        try:
            causal_factors = scenario['causal_factors']
            switch_point = scenario['ground_truth']['switch_point']

            # Split data at switch point
            phase1_factors = causal_factors[:, :switch_point]
            phase2_factors = causal_factors[:, switch_point:]

            structure_learner = causal_system.structure_learner

            # Learn structure for each phase
            loss1, _ = structure_learner.compute_structure_loss(phase1_factors)
            graph1 = structure_learner.get_adjacency_matrix()

            loss2, _ = structure_learner.compute_structure_loss(phase2_factors)
            graph2 = structure_learner.get_adjacency_matrix()

            # Check if system detects different causal structures
            phase1_edge = scenario['ground_truth']['phase1_causation']
            phase2_edge = scenario['ground_truth']['phase2_causation']

            phase1_strength = graph1[phase1_edge[0], phase1_edge[1]].item()
            phase2_strength = graph2[phase2_edge[0], phase2_edge[1]].item()

            # Good meta-causal reasoning detects both phase-specific relationships
            meta_score = (min(phase1_strength, 1.0) + min(phase2_strength, 1.0)) / 2.0

            return meta_score

        except Exception as e:
            print(f"Error in meta-causal evaluation: {e}")
            return 0.0

    def _generate_challenge_report(self, challenge_scores, overall_score, elapsed_time):
        """Generate comprehensive challenge assessment report"""

        # Classify performance levels
        def classify_performance(score):
            if score >= 0.8:
                return "🔥 EXCEPTIONAL", "A+"
            elif score >= 0.7:
                return "✅ STRONG", "A"
            elif score >= 0.6:
                return "👍 GOOD", "B+"
            elif score >= 0.5:
                return "⚠️  ACCEPTABLE", "B"
            elif score >= 0.3:
                return "❌ WEAK", "C"
            else:
                return "💀 FAILED", "F"

        overall_status, overall_grade = classify_performance(overall_score)

        report = {
            'timestamp': time.time(),
            'test_type': 'extreme_causal_challenge',
            'overall_score': overall_score,
            'overall_grade': overall_grade,
            'overall_status': overall_status,
            'elapsed_time': elapsed_time,
            'challenge_breakdown': {},
            'summary': {},
            'recommendations': []
        }

        # Detailed breakdown
        for challenge_name, score in challenge_scores.items():
            status, grade = classify_performance(score)
            report['challenge_breakdown'][challenge_name] = {
                'score': score,
                'grade': grade,
                'status': status,
                'passed': score >= 0.8
            }

        # Generate summary
        passed_challenges = sum(1 for score in challenge_scores.values() if score >= 0.8)
        total_challenges = len(challenge_scores)

        report['summary'] = {
            'challenges_passed': passed_challenges,
            'total_challenges': total_challenges,
            'pass_rate': passed_challenges / total_challenges,
            'strongest_capability': max(challenge_scores.items(), key=lambda x: x[1]),
            'weakest_capability': min(challenge_scores.items(), key=lambda x: x[1]),
            'causal_understanding_level': self._assess_causal_level(challenge_scores)
        }

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(challenge_scores)

        return report

    def _assess_causal_level(self, challenge_scores):
        """Assess overall level of causal understanding"""
        strong_scores = [score for score in challenge_scores.values() if score >= 0.7]

        if len(strong_scores) >= 7:
            return "🧠 GENIUS: Near-human causal reasoning"
        elif len(strong_scores) >= 5:
            return "🎯 EXPERT: Strong causal understanding"
        elif len(strong_scores) >= 3:
            return "📈 COMPETENT: Developing causal capabilities"
        elif len(strong_scores) >= 1:
            return "🌱 NOVICE: Basic causal patterns"
        else:
            return "⚠️  CORRELATIONAL: Pattern matching only"

    def _generate_recommendations(self, challenge_scores):
        """Generate specific recommendations for improvement"""
        recommendations = []

        for challenge_name, score in challenge_scores.items():
            if score < 0.5:
                if challenge_name == 'temporal_causal_chains':
                    recommendations.append("Enhance temporal reasoning with longer sequence training")
                elif challenge_name == 'confounding_adversarial':
                    recommendations.append("Strengthen confounding resistance with adversarial training")
                elif challenge_name == 'mechanism_stress_test':
                    recommendations.append("Improve mechanism isolation with stronger independence constraints")
                # ... add more specific recommendations

        if len(recommendations) == 0:
            recommendations.append("🎉 Exceptional performance! System demonstrates genuine causal understanding")

        return recommendations

    def _print_final_assessment(self, report):
        """Print comprehensive final assessment"""
        print("\n" + "="*80)
        print("🔥 EXTREME CAUSAL CHALLENGE FINAL ASSESSMENT")
        print("="*80)

        print(f"\n📊 OVERALL PERFORMANCE")
        print(f"   Score: {report['overall_score']:.3f}")
        print(f"   Grade: {report['overall_grade']}")
        print(f"   Status: {report['overall_status']}")
        print(f"   Time: {report['elapsed_time']:.1f}s")

        print(f"\n🎯 CHALLENGE BREAKDOWN")
        print(f"   Passed: {report['summary']['challenges_passed']}/{report['summary']['total_challenges']}")
        print(f"   Pass Rate: {report['summary']['pass_rate']:.1%}")

        for challenge_name, results in report['challenge_breakdown'].items():
            status_icon = "✅" if results['passed'] else "❌"
            print(f"   {status_icon} {challenge_name}: {results['score']:.3f} ({results['grade']})")

        print(f"\n🧠 CAUSAL UNDERSTANDING LEVEL")
        print(f"   {report['summary']['causal_understanding_level']}")

        print(f"\n🎯 STRONGEST CAPABILITY")
        strongest = report['summary']['strongest_capability']
        print(f"   {strongest[0]}: {strongest[1]:.3f}")

        print(f"\n⚠️  IMPROVEMENT AREAS")
        for rec in report['recommendations'][:3]:  # Top 3 recommendations
            print(f"   • {rec}")

        print("\n" + "="*80)

        # Final verdict
        if report['overall_score'] >= 0.8:
            print("🔥 VERDICT: This system demonstrates GENUINE CAUSAL REASONING!")
            print("   It goes beyond pattern matching to true causal understanding.")
        elif report['overall_score'] >= 0.6:
            print("👍 VERDICT: Strong causal capabilities with room for improvement")
        elif report['overall_score'] >= 0.4:
            print("⚠️  VERDICT: Basic causal understanding, needs significant work")
        else:
            print("❌ VERDICT: Primarily correlational - not yet truly causal")

        print("="*80)


def main():
    """Run the extreme causal challenge"""
    print("🔥 PREPARING EXTREME CAUSAL REASONING CHALLENGE")
    print("This will test the absolute limits of your causal system...")

    # Initialize challenge
    config = ExtremeChallengeConfig(
        num_test_scenarios=20,  # Reduce for faster testing
        confounding_strength=0.9,
        temporal_complexity=4,
        adversarial_strength=0.8
    )

    challenger = ExtremeCausalChallenger(config)

    # Create and configure your causal system
    training_config = JointTrainingConfig(
        max_epochs=0,  # Don't train, just test current system
        batch_size=16
    )

    causal_system = JointCausalTrainer(training_config)

    # Run the extreme challenge
    challenge_report = challenger.run_extreme_challenge(causal_system)

    # Save results (convert non-serializable objects)
    def make_serializable(obj):
        if hasattr(obj, 'item'):  # torch tensor
            return obj.item()
        elif isinstance(obj, (bool, int, float, str, list, dict, type(None))):
            return obj
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return str(obj)

    # Clean the report for JSON serialization
    clean_report = json.loads(json.dumps(challenge_report, default=make_serializable))

    with open('extreme_causal_challenge_results.json', 'w') as f:
        json.dump(clean_report, f, indent=2)

    print(f"\n📄 Full results saved to: extreme_causal_challenge_results.json")

    return challenge_report


if __name__ == "__main__":
    main()