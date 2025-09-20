#!/usr/bin/env python3
"""
EXTREME DUAL-PATHWAY SPECIALIZATION TORTURE TEST
===============================================

This test pushes the dual-pathway architecture to its absolute limits to determine
if the pathways truly specialize or if they're just redundant representations.

CHALLENGE DESIGN:
1. ADVERSARIAL OBSERVATIONAL vs INTERVENTIONAL DATA
   - Create scenarios where observational and interventional patterns are OPPOSITE
   - Test if pathways can maintain specialization under extreme pressure

2. SUBTLE INTERVENTION DETECTION
   - Mix interventional data that looks observational
   - Test if intervention detector can distinguish subtle manipulations

3. PATHWAY CONSISTENCY UNDER DOMAIN SHIFT
   - Change data distribution but maintain causal structure
   - Test if pathway specialization transfers

4. COMPUTATIONAL TRACTABILITY STRESS
   - Long sequences with many pathway switches
   - Test if memory and computation scale appropriately

TARGET: Only genuine dual-pathway specialization should achieve >80% on this challenge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_architectures.dual_pathway_gru import DualPathwayCausalGRU, CausalLoss


class ExtremeDualPathwayTortureTest:
    """
    Torture test for dual-pathway specialization

    Tests whether the pathways genuinely specialize or just learn redundant representations
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.test_results = {}

    def setup_model(self):
        """Initialize the dual-pathway model"""
        self.model = DualPathwayCausalGRU(
            state_dim=12,
            action_dim=2,
            causal_dim=5,
            hidden_dim=64
        ).to(self.device)

        # Enable specialization for testing
        self.model.specialization_enabled = True

    def generate_adversarial_observational_data(self, batch_size=32, seq_len=50):
        """
        Generate observational data with specific causal patterns

        Pattern: Weather → Crowd (positive correlation)
        """
        states = torch.zeros(batch_size, seq_len, 12)
        actions = torch.zeros(batch_size, seq_len, 2)
        causal_factors = torch.zeros(batch_size, seq_len, 5)

        # Initialize random starting conditions
        states[:, 0, :] = torch.randn(batch_size, 12) * 0.5
        causal_factors[:, 0, :] = torch.randn(batch_size, 5) * 0.3

        for t in range(1, seq_len):
            # OBSERVATIONAL PATTERN: Weather → Crowd (POSITIVE correlation)
            weather = causal_factors[:, t-1, 0]
            causal_factors[:, t, 1] = 0.8 * weather + 0.2 * torch.randn(batch_size)

            # Other factors evolve naturally
            causal_factors[:, t, 0] = 0.9 * weather + 0.1 * torch.randn(batch_size)
            causal_factors[:, t, 2:] = 0.7 * causal_factors[:, t-1, 2:] + 0.3 * torch.randn(batch_size, 3)

            # State evolution based on causal factors
            weather_effect = causal_factors[:, t, 0].unsqueeze(1) * 0.3
            crowd_effect = causal_factors[:, t, 1].unsqueeze(1) * -0.2

            states[:, t, :2] = states[:, t-1, :2] + weather_effect + crowd_effect + 0.1 * torch.randn(batch_size, 2)
            states[:, t, 2:] = 0.95 * states[:, t-1, 2:] + 0.05 * torch.randn(batch_size, 10)

            # Random actions
            actions[:, t, :] = torch.randn(batch_size, 2) * 0.5

        return states, actions, causal_factors, torch.zeros(batch_size, seq_len, 1)  # No interventions

    def generate_adversarial_interventional_data(self, batch_size=32, seq_len=50):
        """
        Generate interventional data with OPPOSITE causal patterns

        Pattern: Weather intervention → Crowd (NEGATIVE correlation) - opposite of observational
        """
        states = torch.zeros(batch_size, seq_len, 12)
        actions = torch.zeros(batch_size, seq_len, 2)
        causal_factors = torch.zeros(batch_size, seq_len, 5)
        intervention_mask = torch.zeros(batch_size, seq_len, 1)

        # Initialize random starting conditions
        states[:, 0, :] = torch.randn(batch_size, 12) * 0.5
        causal_factors[:, 0, :] = torch.randn(batch_size, 5) * 0.3

        for t in range(1, seq_len):
            # INTERVENTIONAL PATTERN: Do(Weather) → Crowd (NEGATIVE correlation) - OPPOSITE!

            # Random weather interventions (30% of time)
            intervene = torch.rand(batch_size) < 0.3
            intervention_mask[:, t, 0] = intervene.float()

            if intervene.any():
                # Intervene on weather for some samples
                causal_factors[intervene, t, 0] = torch.randn(intervene.sum()) * 0.8

                # Under intervention, crowd responds NEGATIVELY (opposite of observational)
                weather_intervened = causal_factors[intervene, t, 0]
                causal_factors[intervene, t, 1] = -0.8 * weather_intervened + 0.2 * torch.randn(intervene.sum())

            # Non-intervened samples follow natural evolution
            non_intervene = ~intervene
            if non_intervene.any():
                weather = causal_factors[non_intervene, t-1, 0]
                causal_factors[non_intervene, t, 0] = 0.9 * weather + 0.1 * torch.randn(non_intervene.sum())
                causal_factors[non_intervene, t, 1] = 0.7 * causal_factors[non_intervene, t-1, 1] + 0.3 * torch.randn(non_intervene.sum())

            # Other factors evolve naturally
            causal_factors[:, t, 2:] = 0.7 * causal_factors[:, t-1, 2:] + 0.3 * torch.randn(batch_size, 3)

            # State evolution based on causal factors
            weather_effect = causal_factors[:, t, 0].unsqueeze(1) * 0.3
            crowd_effect = causal_factors[:, t, 1].unsqueeze(1) * -0.2

            states[:, t, :2] = states[:, t-1, :2] + weather_effect + crowd_effect + 0.1 * torch.randn(batch_size, 2)
            states[:, t, 2:] = 0.95 * states[:, t-1, 2:] + 0.05 * torch.randn(batch_size, 10)

            # Actions influenced by interventions
            actions[:, t, :] = torch.randn(batch_size, 2) * 0.5
            if intervene.any():
                actions[intervene, t, :] *= 1.5  # More aggressive actions under intervention

        return states, actions, causal_factors, intervention_mask

    def test_1_pathway_specialization_under_opposition(self):
        """
        TEST 1: Can pathways maintain specialization when obs/int patterns are opposite?

        This is the ultimate test of pathway specialization
        """
        print("🔥 TEST 1: Pathway Specialization Under Opposition")
        print("=" * 60)

        # Generate adversarial data
        obs_states, obs_actions, obs_causal, obs_mask = self.generate_adversarial_observational_data()
        int_states, int_actions, int_causal, int_mask = self.generate_adversarial_interventional_data()

        # Test model on both
        self.model.eval()

        with torch.no_grad():
            # Observational forward pass
            obs_pred, obs_hidden, obs_pathway_info = self.model(
                obs_states[:, :-1], obs_actions[:, :-1], obs_causal[:, :-1], obs_mask[:, :-1]
            )

            # Interventional forward pass
            int_pred, int_hidden, int_pathway_info = self.model(
                int_states[:, :-1], int_actions[:, :-1], int_causal[:, :-1], int_mask[:, :-1]
            )

        # CRITICAL TEST: Are pathway outputs genuinely different?
        specialization_score = self.model.compute_specialization_score(
            obs_pathway_info['obs_weight'], int_pathway_info['int_weight']
        )

        # CRITICAL TEST: Does intervention detection work?
        obs_intervention_score = obs_pathway_info['intervention_score']
        int_intervention_score = int_pathway_info['intervention_score']
        intervention_discrimination = abs(int_intervention_score - obs_intervention_score)

        # CRITICAL TEST: Are predictions meaningfully different?
        prediction_divergence = torch.mean(torch.abs(obs_pred - int_pred)).item()

        results = {
            'specialization_score': specialization_score.item() if hasattr(specialization_score, 'item') else specialization_score,
            'intervention_discrimination': intervention_discrimination,
            'prediction_divergence': prediction_divergence,
            'obs_intervention_score': obs_intervention_score,
            'int_intervention_score': int_intervention_score,
            'obs_pathway_balance': obs_pathway_info['pathway_balance'],
            'int_pathway_balance': int_pathway_info['pathway_balance']
        }

        # Scoring criteria (extremely harsh)
        specialization_pass = results['specialization_score'] > 0.6  # High bar
        discrimination_pass = results['intervention_discrimination'] > 0.3  # Must distinguish
        divergence_pass = results['prediction_divergence'] > 0.1  # Must predict differently

        total_score = (specialization_pass + discrimination_pass + divergence_pass) / 3.0

        print(f"  Specialization Score: {results['specialization_score']:.4f} {'✅' if specialization_pass else '❌'}")
        print(f"  Intervention Discrimination: {results['intervention_discrimination']:.4f} {'✅' if discrimination_pass else '❌'}")
        print(f"  Prediction Divergence: {results['prediction_divergence']:.4f} {'✅' if divergence_pass else '❌'}")
        print(f"  Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("  🎉 EXCEPTIONAL: Genuine pathway specialization under adversarial conditions!")
        elif total_score > 0.6:
            print("  👍 GOOD: Pathways show specialization but room for improvement")
        elif total_score > 0.3:
            print("  ⚠️  WEAK: Limited pathway specialization detected")
        else:
            print("  💀 FAILED: No meaningful pathway specialization")

        self.test_results['test_1'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.6
        }

        return total_score, results

    def test_2_subtle_intervention_detection(self):
        """
        TEST 2: Can the model detect subtle interventions that look observational?
        """
        print("\n🎯 TEST 2: Subtle Intervention Detection")
        print("=" * 60)

        batch_size, seq_len = 16, 30

        # Generate data with VERY subtle interventions
        states = torch.zeros(batch_size, seq_len, 12)
        actions = torch.zeros(batch_size, seq_len, 2)
        causal_factors = torch.zeros(batch_size, seq_len, 5)
        true_interventions = torch.zeros(batch_size, seq_len, 1)

        # Initialize
        states[:, 0, :] = torch.randn(batch_size, 12) * 0.3
        causal_factors[:, 0, :] = torch.randn(batch_size, 5) * 0.2

        for t in range(1, seq_len):
            # Subtle interventions: only 10% of time, tiny magnitude
            intervene = torch.rand(batch_size) < 0.1
            true_interventions[:, t, 0] = intervene.float()

            if intervene.any():
                # VERY subtle intervention - just small bias
                natural_evolution = 0.95 * causal_factors[intervene, t-1, 0] + 0.05 * torch.randn(intervene.sum())
                intervention_bias = torch.randn(intervene.sum()) * 0.1  # Very small
                causal_factors[intervene, t, 0] = natural_evolution + intervention_bias
            else:
                # Natural evolution
                causal_factors[~intervene, t, 0] = 0.95 * causal_factors[~intervene, t-1, 0] + 0.05 * torch.randn((~intervene).sum())

            # Other factors evolve naturally
            causal_factors[:, t, 1:] = 0.9 * causal_factors[:, t-1, 1:] + 0.1 * torch.randn(batch_size, 4)

            # State evolution
            states[:, t, :] = 0.9 * states[:, t-1, :] + 0.1 * torch.randn(batch_size, 12)
            actions[:, t, :] = torch.randn(batch_size, 2) * 0.3

        # Test intervention detection
        self.model.eval()

        with torch.no_grad():
            pred_states, hidden, pathway_info = self.model(
                states[:, :-1], actions[:, :-1], causal_factors[:, :-1]
            )

        # Extract intervention scores
        detected_interventions = pathway_info['intervention_score']
        true_intervention_rate = true_interventions[:, 1:].mean().item()

        # Compute detection accuracy
        threshold = 0.5
        predictions = (detected_interventions > threshold).float()
        accuracy = (predictions == true_interventions[:, 1:, 0].unsqueeze(-1)).float().mean().item()

        # Compute precision/recall for the rare positive class
        true_positives = ((predictions == 1) & (true_interventions[:, 1:, 0].unsqueeze(-1) == 1)).sum().item()
        predicted_positives = (predictions == 1).sum().item()
        actual_positives = (true_interventions[:, 1:, 0] == 1).sum().item()

        precision = true_positives / max(predicted_positives, 1)
        recall = true_positives / max(actual_positives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_intervention_rate': true_intervention_rate,
            'avg_detection_score': detected_interventions,
            'intervention_base_rate': true_intervention_rate
        }

        # Scoring (very harsh for subtle detection)
        f1_pass = f1 > 0.3  # Detecting subtle interventions is extremely hard
        accuracy_pass = accuracy > 0.85  # Should still be mostly accurate

        total_score = (f1_pass * 0.7 + accuracy_pass * 0.3)

        print(f"  Accuracy: {accuracy:.4f} {'✅' if accuracy_pass else '❌'}")
        print(f"  F1 Score: {f1:.4f} {'✅' if f1_pass else '❌'}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  True Intervention Rate: {true_intervention_rate:.3f}")
        print(f"  Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("  🎉 EXCEPTIONAL: Detected subtle interventions!")
        elif total_score > 0.5:
            print("  👍 GOOD: Some subtle intervention detection")
        else:
            print("  💀 FAILED: Cannot detect subtle interventions")

        self.test_results['test_2'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.5
        }

        return total_score, results

    def test_3_pathway_memory_consistency(self):
        """
        TEST 3: Do pathways maintain consistent specialization across long sequences?
        """
        print("\n🧠 TEST 3: Pathway Memory Consistency")
        print("=" * 60)

        # Generate very long sequence with pathway switches
        batch_size, seq_len = 8, 200  # Long sequence

        states = torch.zeros(batch_size, seq_len, 12)
        actions = torch.zeros(batch_size, seq_len, 2)
        causal_factors = torch.zeros(batch_size, seq_len, 5)
        intervention_mask = torch.zeros(batch_size, seq_len, 1)

        # Initialize
        states[:, 0, :] = torch.randn(batch_size, 12) * 0.3
        causal_factors[:, 0, :] = torch.randn(batch_size, 5) * 0.3

        # Create structured intervention pattern
        intervention_phases = []
        for t in range(1, seq_len):
            # Structured phases: 30 steps obs, 20 steps int, repeat
            phase = (t // 25) % 2
            is_intervention_phase = phase == 1

            if is_intervention_phase:
                # Intervention phase
                intervention_mask[:, t, 0] = 1.0
                causal_factors[:, t, 0] = torch.randn(batch_size) * 0.5  # Strong intervention
                causal_factors[:, t, 1:] = 0.8 * causal_factors[:, t-1, 1:] + 0.2 * torch.randn(batch_size, 4)
            else:
                # Observational phase
                causal_factors[:, t, :] = 0.9 * causal_factors[:, t-1, :] + 0.1 * torch.randn(batch_size, 5)

            intervention_phases.append(is_intervention_phase)

            # State evolution
            states[:, t, :] = 0.8 * states[:, t-1, :] + 0.2 * torch.randn(batch_size, 12)
            actions[:, t, :] = torch.randn(batch_size, 2) * 0.4

        # Test pathway consistency
        self.model.eval()

        with torch.no_grad():
            # Process in chunks to avoid memory issues
            chunk_size = 50
            pathway_weights = []
            intervention_scores = []

            for start_idx in range(0, seq_len - chunk_size, chunk_size // 2):
                end_idx = min(start_idx + chunk_size, seq_len)
                chunk_len = end_idx - start_idx - 1

                if chunk_len <= 0:
                    continue

                pred_chunk, hidden_chunk, pathway_info_chunk = self.model(
                    states[:, start_idx:end_idx-1],
                    actions[:, start_idx:end_idx-1],
                    causal_factors[:, start_idx:end_idx-1],
                    intervention_mask[:, start_idx:end_idx-1]
                )

                pathway_weights.append(pathway_info_chunk['obs_weight'])
                intervention_scores.append(pathway_info_chunk['intervention_score'])

        # Analyze consistency
        obs_phases = [i for i, is_int in enumerate(intervention_phases[1:]) if not is_int]
        int_phases = [i for i, is_int in enumerate(intervention_phases[1:]) if is_int]

        # Check if pathway weights are consistent within phases
        pathway_weights = np.array(pathway_weights)
        intervention_scores = np.array(intervention_scores)

        # Compute phase consistency
        if len(obs_phases) > 10 and len(int_phases) > 10:
            obs_weight_var = np.var(pathway_weights[obs_phases[:min(len(obs_phases), len(pathway_weights))]])
            int_weight_var = np.var(pathway_weights[int_phases[:min(len(int_phases), len(pathway_weights))]])

            obs_score_var = np.var(intervention_scores[obs_phases[:min(len(obs_phases), len(intervention_scores))]])
            int_score_var = np.var(intervention_scores[int_phases[:min(len(int_phases), len(intervention_scores))]])
        else:
            obs_weight_var = np.var(pathway_weights)
            int_weight_var = np.var(pathway_weights)
            obs_score_var = np.var(intervention_scores)
            int_score_var = np.var(intervention_scores)

        # Consistency scores (lower variance = better consistency)
        weight_consistency = 1.0 / (1.0 + obs_weight_var + int_weight_var)
        score_consistency = 1.0 / (1.0 + obs_score_var + int_score_var)

        results = {
            'weight_consistency': weight_consistency,
            'score_consistency': score_consistency,
            'obs_weight_variance': obs_weight_var,
            'int_weight_variance': int_weight_var,
            'obs_score_variance': obs_score_var,
            'int_score_variance': int_score_var,
            'sequence_length': seq_len
        }

        # Scoring
        weight_pass = weight_consistency > 0.7
        score_pass = score_consistency > 0.7

        total_score = (weight_pass + score_pass) / 2.0

        print(f"  Weight Consistency: {weight_consistency:.4f} {'✅' if weight_pass else '❌'}")
        print(f"  Score Consistency: {score_consistency:.4f} {'✅' if score_pass else '❌'}")
        print(f"  Obs Weight Variance: {obs_weight_var:.6f}")
        print(f"  Int Weight Variance: {int_weight_var:.6f}")
        print(f"  Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("  🎉 EXCEPTIONAL: Consistent pathway behavior across long sequences!")
        elif total_score > 0.6:
            print("  👍 GOOD: Generally consistent pathway behavior")
        else:
            print("  💀 FAILED: Inconsistent pathway behavior")

        self.test_results['test_3'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.6
        }

        return total_score, results

    def run_all_tests(self):
        """Run all dual-pathway torture tests"""
        print("🔥 EXTREME DUAL-PATHWAY SPECIALIZATION TORTURE TEST")
        print("=" * 80)
        print("Testing whether pathways genuinely specialize or just learn redundant representations")
        print()

        self.setup_model()

        # Run all tests
        score_1, _ = self.test_1_pathway_specialization_under_opposition()
        score_2, _ = self.test_2_subtle_intervention_detection()
        score_3, _ = self.test_3_pathway_memory_consistency()

        # Overall assessment
        overall_score = (score_1 + score_2 + score_3) / 3.0

        print("\n" + "=" * 80)
        print("📊 DUAL-PATHWAY TORTURE TEST RESULTS")
        print("=" * 80)
        print(f"Test 1 - Pathway Opposition: {score_1:.3f}")
        print(f"Test 2 - Subtle Detection: {score_2:.3f}")
        print(f"Test 3 - Memory Consistency: {score_3:.3f}")
        print(f"Overall Score: {overall_score:.3f}")

        if overall_score > 0.8:
            grade = "A+"
            status = "🔥 EXCEPTIONAL - Genuine dual-pathway specialization!"
        elif overall_score > 0.7:
            grade = "A"
            status = "🎉 EXCELLENT - Strong pathway specialization"
        elif overall_score > 0.6:
            grade = "B"
            status = "👍 GOOD - Decent pathway specialization"
        elif overall_score > 0.4:
            grade = "C"
            status = "⚠️ WEAK - Limited pathway specialization"
        else:
            grade = "F"
            status = "💀 FAILED - No meaningful pathway specialization"

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
    """Run the extreme dual-pathway torture test"""
    test = ExtremeDualPathwayTortureTest()
    results = test.run_all_tests()

    # Save results
    with open('extreme_dual_pathway_torture_results.json', 'w') as f:
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

    print(f"\n📁 Results saved to: extreme_dual_pathway_torture_results.json")

    return results


if __name__ == "__main__":
    results = main()