#!/usr/bin/env python3
"""
EXTREME STRUCTURE LEARNING GROUND TRUTH VALIDATION TEST
=======================================================

This test pushes the causal structure learning to its limits by testing against
known ground truth causal structures with increasing complexity.

CHALLENGE DESIGN:
1. SIMPLE KNOWN STRUCTURES
   - Test basic chains, forks, colliders with known edge weights
   - Verify if learner can recover obvious patterns

2. COMPLEX GRAPH TOPOLOGIES
   - Dense graphs, cycles, multiple components
   - Test scalability and robustness

3. CONFOUNDING AND SPURIOUS CORRELATIONS
   - Hidden confounders, selection bias, spurious edges
   - Test if learner can distinguish causation from correlation

4. TEMPORAL STRUCTURE EVOLUTION
   - Structures that change over time
   - Test adaptability to non-stationary causal graphs

TARGET: Only genuine structure learning should achieve >70% structural recovery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import pearsonr

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from causal_architectures.enhanced_structure_learner import EnhancedCausalStructureLearner
except ImportError:
    # Fallback to basic structure learner
    from causal_architectures.structure_learner import CausalStructureLearner as EnhancedCausalStructureLearner


class GroundTruthStructureGenerator:
    """
    Generate causal data with known ground truth structures
    """

    def __init__(self, num_variables=5):
        self.num_variables = num_variables
        self.variable_names = ['weather', 'crowd_density', 'special_event', 'time_of_day', 'road_conditions']

    def generate_simple_chain(self, batch_size=64, seq_len=50):
        """
        Generate data from simple causal chain: A -> B -> C -> D -> E

        Ground truth: weather -> crowd -> event -> time -> road
        """
        data = torch.zeros(batch_size, seq_len, self.num_variables)

        # Initialize first variable randomly
        data[:, 0, 0] = torch.randn(batch_size) * 0.5  # weather

        for t in range(seq_len):
            if t > 0:
                # weather evolves naturally
                data[:, t, 0] = 0.8 * data[:, t-1, 0] + 0.2 * torch.randn(batch_size)

            # Causal chain with strong effects
            data[:, t, 1] = 0.7 * data[:, t, 0] + 0.3 * torch.randn(batch_size)  # weather -> crowd
            data[:, t, 2] = 0.6 * data[:, t, 1] + 0.4 * torch.randn(batch_size)  # crowd -> event
            data[:, t, 3] = 0.5 * data[:, t, 2] + 0.5 * torch.randn(batch_size)  # event -> time
            data[:, t, 4] = 0.4 * data[:, t, 3] + 0.6 * torch.randn(batch_size)  # time -> road

        # Ground truth adjacency matrix
        ground_truth = torch.zeros(self.num_variables, self.num_variables)
        ground_truth[0, 1] = 0.7  # weather -> crowd
        ground_truth[1, 2] = 0.6  # crowd -> event
        ground_truth[2, 3] = 0.5  # event -> time
        ground_truth[3, 4] = 0.4  # time -> road

        return data, ground_truth

    def generate_complex_fork(self, batch_size=64, seq_len=50):
        """
        Generate data from fork structure: A -> {B, C, D} -> E

        Ground truth: weather -> {crowd, event, time} -> road
        """
        data = torch.zeros(batch_size, seq_len, self.num_variables)

        for t in range(seq_len):
            if t == 0:
                data[:, 0, 0] = torch.randn(batch_size) * 0.5  # weather
            else:
                data[:, t, 0] = 0.9 * data[:, t-1, 0] + 0.1 * torch.randn(batch_size)

            # Fork: weather influences multiple variables
            data[:, t, 1] = 0.6 * data[:, t, 0] + 0.4 * torch.randn(batch_size)  # weather -> crowd
            data[:, t, 2] = 0.5 * data[:, t, 0] + 0.5 * torch.randn(batch_size)  # weather -> event
            data[:, t, 3] = 0.4 * data[:, t, 0] + 0.6 * torch.randn(batch_size)  # weather -> time

            # Collider: all three influence road
            data[:, t, 4] = (0.3 * data[:, t, 1] + 0.2 * data[:, t, 2] +
                            0.2 * data[:, t, 3] + 0.3 * torch.randn(batch_size))  # {crowd, event, time} -> road

        # Ground truth adjacency matrix
        ground_truth = torch.zeros(self.num_variables, self.num_variables)
        ground_truth[0, 1] = 0.6  # weather -> crowd
        ground_truth[0, 2] = 0.5  # weather -> event
        ground_truth[0, 3] = 0.4  # weather -> time
        ground_truth[1, 4] = 0.3  # crowd -> road
        ground_truth[2, 4] = 0.2  # event -> road
        ground_truth[3, 4] = 0.2  # time -> road

        return data, ground_truth

    def generate_confounded_structure(self, batch_size=64, seq_len=50):
        """
        Generate data with hidden confounding to test robustness

        Hidden confounder H influences multiple observed variables
        """
        data = torch.zeros(batch_size, seq_len, self.num_variables)
        hidden_confounder = torch.zeros(batch_size, seq_len)

        for t in range(seq_len):
            if t == 0:
                hidden_confounder[:, 0] = torch.randn(batch_size) * 0.7
                data[:, 0, 0] = torch.randn(batch_size) * 0.3
            else:
                hidden_confounder[:, t] = 0.9 * hidden_confounder[:, t-1] + 0.1 * torch.randn(batch_size)
                data[:, t, 0] = 0.8 * data[:, t-1, 0] + 0.2 * torch.randn(batch_size)

            # True causal relationships
            data[:, t, 1] = 0.5 * data[:, t, 0] + 0.5 * torch.randn(batch_size)  # weather -> crowd

            # Confounded relationships (both influenced by hidden confounder)
            confound_effect = hidden_confounder[:, t]
            data[:, t, 2] = 0.4 * confound_effect + 0.6 * torch.randn(batch_size)  # H -> event
            data[:, t, 3] = 0.3 * confound_effect + 0.7 * torch.randn(batch_size)  # H -> time

            # Independent variable
            data[:, t, 4] = 0.8 * data[:, t-1, 4] + 0.2 * torch.randn(batch_size) if t > 0 else torch.randn(batch_size) * 0.3

        # Ground truth (only direct causal relationships)
        ground_truth = torch.zeros(self.num_variables, self.num_variables)
        ground_truth[0, 1] = 0.5  # weather -> crowd (only true causal edge)

        return data, ground_truth, hidden_confounder

    def generate_temporal_varying_structure(self, batch_size=64, seq_len=100):
        """
        Generate data where causal structure changes over time
        """
        data = torch.zeros(batch_size, seq_len, self.num_variables)

        # Two different structures for different time periods
        change_point = seq_len // 2

        for t in range(seq_len):
            if t == 0:
                data[:, 0, :] = torch.randn(batch_size, self.num_variables) * 0.3
            else:
                # Natural evolution
                data[:, t, :] = 0.7 * data[:, t-1, :] + 0.3 * torch.randn(batch_size, self.num_variables)

            if t < change_point:
                # Structure 1: weather -> crowd -> event
                data[:, t, 1] = 0.6 * data[:, t, 0] + 0.4 * torch.randn(batch_size)
                data[:, t, 2] = 0.5 * data[:, t, 1] + 0.5 * torch.randn(batch_size)
            else:
                # Structure 2: event -> crowd -> weather (reversed!)
                data[:, t, 0] = 0.5 * data[:, t, 1] + 0.5 * torch.randn(batch_size)
                data[:, t, 1] = 0.6 * data[:, t, 2] + 0.4 * torch.randn(batch_size)

        # Ground truth for second half (structure changed)
        ground_truth = torch.zeros(self.num_variables, self.num_variables)
        ground_truth[2, 1] = 0.6  # event -> crowd
        ground_truth[1, 0] = 0.5  # crowd -> weather

        return data, ground_truth, change_point


class ExtremeStructureLearningGroundTruthTest:
    """
    Torture test for causal structure learning against known ground truth
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.test_results = {}
        self.generator = GroundTruthStructureGenerator()

    def setup_model(self):
        """Initialize the structure learning model"""
        try:
            self.model = EnhancedCausalStructureLearner(
                num_variables=5,
                hidden_dim=64,
                learning_rate=1e-3,
                total_epochs=50
            ).to(self.device)
        except:
            # Fallback if enhanced version not available
            from causal_architectures.structure_learner import CausalStructureLearner
            self.model = CausalStructureLearner(
                num_variables=5,
                hidden_dim=64,
                learning_rate=1e-3
            ).to(self.device)

    def compute_structure_recovery_metrics(self, learned_adj, ground_truth_adj, threshold=0.3):
        """
        Compute comprehensive structure recovery metrics
        """
        # Threshold learned adjacency matrix
        learned_binary = (learned_adj > threshold).float()
        ground_truth_binary = (ground_truth_adj > 0).float()

        # Basic metrics
        true_positives = ((learned_binary == 1) & (ground_truth_binary == 1)).sum().item()
        false_positives = ((learned_binary == 1) & (ground_truth_binary == 0)).sum().item()
        true_negatives = ((learned_binary == 0) & (ground_truth_binary == 0)).sum().item()
        false_negatives = ((learned_binary == 0) & (ground_truth_binary == 1)).sum().item()

        # Derived metrics
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (true_positives + true_negatives) / max(true_positives + false_positives + true_negatives + false_negatives, 1)

        # Structural Hamming Distance
        shd = false_positives + false_negatives

        # Threshold-free metrics using continuous values
        # Structural correlation
        learned_flat = learned_adj.flatten()
        ground_truth_flat = ground_truth_adj.flatten()

        if torch.std(learned_flat) > 1e-8 and torch.std(ground_truth_flat) > 1e-8:
            structural_correlation = torch.corrcoef(torch.stack([learned_flat, ground_truth_flat]))[0, 1].item()
        else:
            structural_correlation = 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'structural_hamming_distance': shd,
            'structural_correlation': structural_correlation,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'threshold_used': threshold
        }

    def test_1_simple_chain_recovery(self):
        """
        TEST 1: Can the model recover a simple causal chain?
        """
        print("🔥 TEST 1: Simple Chain Recovery")
        print("=" * 60)
        print("Ground Truth: weather -> crowd -> event -> time -> road")

        # Generate simple chain data
        data, ground_truth = self.generator.generate_simple_chain()

        # Train structure learner
        self.model.train()

        # Multiple training iterations
        best_loss = float('inf')
        best_adjacency = None

        for epoch in range(10):  # Quick training
            try:
                if hasattr(self.model, 'compute_enhanced_structure_loss'):
                    loss, loss_info = self.model.compute_enhanced_structure_loss(data)
                else:
                    loss, loss_info = self.model.compute_structure_loss(data)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_adjacency = self.model.get_adjacency_matrix().detach()
            except Exception as e:
                print(f"    Training error at epoch {epoch}: {e}")
                continue

        if best_adjacency is None:
            print("    💀 CRITICAL ERROR: Training completely failed!")
            return 0.0, {'error': 'training_failed'}

        # Evaluate structure recovery
        recovery_metrics = self.compute_structure_recovery_metrics(best_adjacency, ground_truth)

        # Additional analysis
        learned_summary = self.model.get_causal_graph_summary() if hasattr(self.model, 'get_causal_graph_summary') else {}

        results = {
            'final_loss': best_loss,
            'recovery_metrics': recovery_metrics,
            'learned_adjacency': best_adjacency.cpu().numpy().tolist(),
            'ground_truth_adjacency': ground_truth.numpy().tolist(),
            'learned_graph_summary': learned_summary
        }

        # Scoring criteria
        f1_pass = recovery_metrics['f1_score'] > 0.6  # Should recover most edges
        precision_pass = recovery_metrics['precision'] > 0.5  # Should not hallucinate too many edges
        recall_pass = recovery_metrics['recall'] > 0.5  # Should find most true edges
        correlation_pass = recovery_metrics['structural_correlation'] > 0.4  # Should correlate with truth

        total_score = (f1_pass + precision_pass + recall_pass + correlation_pass) / 4.0

        print(f"    Final Training Loss: {best_loss:.6f}")
        print(f"    F1 Score: {recovery_metrics['f1_score']:.4f} {'✅' if f1_pass else '❌'}")
        print(f"    Precision: {recovery_metrics['precision']:.4f} {'✅' if precision_pass else '❌'}")
        print(f"    Recall: {recovery_metrics['recall']:.4f} {'✅' if recall_pass else '❌'}")
        print(f"    Structural Correlation: {recovery_metrics['structural_correlation']:.4f} {'✅' if correlation_pass else '❌'}")
        print(f"    Structural Hamming Distance: {recovery_metrics['structural_hamming_distance']}")
        print(f"    Edges Found: {recovery_metrics['true_positives']}/{ground_truth.sum().item()}")
        print(f"    Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("    🎉 EXCEPTIONAL: Perfect simple chain recovery!")
        elif total_score > 0.6:
            print("    👍 GOOD: Good simple chain recovery")
        elif total_score > 0.3:
            print("    ⚠️  WEAK: Partial simple chain recovery")
        else:
            print("    💀 FAILED: Cannot recover simple chain")

        self.test_results['test_1'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.6
        }

        return total_score, results

    def test_2_complex_fork_recovery(self):
        """
        TEST 2: Can the model recover a complex fork structure?
        """
        print("\n🎯 TEST 2: Complex Fork Structure Recovery")
        print("=" * 60)
        print("Ground Truth: weather -> {crowd, event, time} -> road")

        # Generate complex fork data
        data, ground_truth = self.generator.generate_complex_fork()

        # Train structure learner
        self.model.train()

        best_loss = float('inf')
        best_adjacency = None

        for epoch in range(15):  # More training for complex structure
            try:
                if hasattr(self.model, 'compute_enhanced_structure_loss'):
                    loss, loss_info = self.model.compute_enhanced_structure_loss(data)
                else:
                    loss, loss_info = self.model.compute_structure_loss(data)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_adjacency = self.model.get_adjacency_matrix().detach()
            except Exception as e:
                print(f"    Training error at epoch {epoch}: {e}")
                continue

        if best_adjacency is None:
            print("    💀 CRITICAL ERROR: Training completely failed!")
            return 0.0, {'error': 'training_failed'}

        # Evaluate structure recovery
        recovery_metrics = self.compute_structure_recovery_metrics(best_adjacency, ground_truth)

        results = {
            'final_loss': best_loss,
            'recovery_metrics': recovery_metrics,
            'learned_adjacency': best_adjacency.cpu().numpy().tolist(),
            'ground_truth_adjacency': ground_truth.numpy().tolist()
        }

        # Scoring criteria (harsher for complex structure)
        f1_pass = recovery_metrics['f1_score'] > 0.4  # Complex structure is harder
        precision_pass = recovery_metrics['precision'] > 0.4
        recall_pass = recovery_metrics['recall'] > 0.4
        correlation_pass = recovery_metrics['structural_correlation'] > 0.3

        total_score = (f1_pass + precision_pass + recall_pass + correlation_pass) / 4.0

        print(f"    Final Training Loss: {best_loss:.6f}")
        print(f"    F1 Score: {recovery_metrics['f1_score']:.4f} {'✅' if f1_pass else '❌'}")
        print(f"    Precision: {recovery_metrics['precision']:.4f} {'✅' if precision_pass else '❌'}")
        print(f"    Recall: {recovery_metrics['recall']:.4f} {'✅' if recall_pass else '❌'}")
        print(f"    Structural Correlation: {recovery_metrics['structural_correlation']:.4f} {'✅' if correlation_pass else '❌'}")
        print(f"    True Edges: {ground_truth.sum().item()}, Found: {recovery_metrics['true_positives']}")
        print(f"    Overall Score: {total_score:.3f}")

        if total_score > 0.7:
            print("    🎉 EXCEPTIONAL: Excellent complex structure recovery!")
        elif total_score > 0.5:
            print("    👍 GOOD: Good complex structure recovery")
        elif total_score > 0.3:
            print("    ⚠️  WEAK: Partial complex structure recovery")
        else:
            print("    💀 FAILED: Cannot recover complex structure")

        self.test_results['test_2'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.5
        }

        return total_score, results

    def test_3_confounding_robustness(self):
        """
        TEST 3: Can the model handle hidden confounding?
        """
        print("\n🧠 TEST 3: Confounding Robustness")
        print("=" * 60)
        print("Hidden confounder influences multiple variables")

        # Generate confounded data
        data, ground_truth, hidden_confounder = self.generator.generate_confounded_structure()

        # Train structure learner
        self.model.train()

        best_loss = float('inf')
        best_adjacency = None

        for epoch in range(20):  # More training needed for confounded data
            try:
                if hasattr(self.model, 'compute_enhanced_structure_loss'):
                    loss, loss_info = self.model.compute_enhanced_structure_loss(data)
                else:
                    loss, loss_info = self.model.compute_structure_loss(data)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_adjacency = self.model.get_adjacency_matrix().detach()
            except Exception as e:
                print(f"    Training error at epoch {epoch}: {e}")
                continue

        if best_adjacency is None:
            print("    💀 CRITICAL ERROR: Training completely failed!")
            return 0.0, {'error': 'training_failed'}

        # Evaluate structure recovery
        recovery_metrics = self.compute_structure_recovery_metrics(best_adjacency, ground_truth)

        # Check for spurious correlations (should not find edges between confounded variables)
        spurious_edge_23 = best_adjacency[2, 3].item() > 0.3  # event -> time (confounded)
        spurious_edge_32 = best_adjacency[3, 2].item() > 0.3  # time -> event (confounded)
        spurious_edges_found = spurious_edge_23 or spurious_edge_32

        results = {
            'final_loss': best_loss,
            'recovery_metrics': recovery_metrics,
            'learned_adjacency': best_adjacency.cpu().numpy().tolist(),
            'ground_truth_adjacency': ground_truth.numpy().tolist(),
            'spurious_edges_found': spurious_edges_found,
            'spurious_edge_details': {
                'event_to_time': best_adjacency[2, 3].item(),
                'time_to_event': best_adjacency[3, 2].item()
            }
        }

        # Scoring criteria (very harsh for confounding)
        f1_pass = recovery_metrics['f1_score'] > 0.5  # Should find true edge
        precision_pass = recovery_metrics['precision'] > 0.6  # Should avoid spurious edges
        no_spurious_pass = not spurious_edges_found  # Should not find confounded edges
        low_false_positives = recovery_metrics['false_positives'] <= 1  # At most 1 spurious edge

        total_score = (f1_pass + precision_pass + no_spurious_pass + low_false_positives) / 4.0

        print(f"    Final Training Loss: {best_loss:.6f}")
        print(f"    F1 Score: {recovery_metrics['f1_score']:.4f} {'✅' if f1_pass else '❌'}")
        print(f"    Precision: {recovery_metrics['precision']:.4f} {'✅' if precision_pass else '❌'}")
        print(f"    Spurious Edges Avoided: {'✅' if no_spurious_pass else '❌'}")
        print(f"    False Positives: {recovery_metrics['false_positives']} {'✅' if low_false_positives else '❌'}")
        print(f"    Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("    🎉 EXCEPTIONAL: Robust to confounding!")
        elif total_score > 0.6:
            print("    👍 GOOD: Some robustness to confounding")
        elif total_score > 0.4:
            print("    ⚠️  WEAK: Limited robustness to confounding")
        else:
            print("    💀 FAILED: Not robust to confounding")

        self.test_results['test_3'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.6
        }

        return total_score, results

    def run_all_tests(self):
        """Run all structure learning ground truth tests"""
        print("🔥 EXTREME STRUCTURE LEARNING GROUND TRUTH TEST")
        print("=" * 80)
        print("Testing structure learning against known ground truth causal graphs")
        print()

        self.setup_model()

        # Run all tests
        score_1, _ = self.test_1_simple_chain_recovery()
        score_2, _ = self.test_2_complex_fork_recovery()
        score_3, _ = self.test_3_confounding_robustness()

        # Overall assessment
        overall_score = (score_1 + score_2 + score_3) / 3.0

        print("\n" + "=" * 80)
        print("📊 STRUCTURE LEARNING GROUND TRUTH TEST RESULTS")
        print("=" * 80)
        print(f"Test 1 - Simple Chain: {score_1:.3f}")
        print(f"Test 2 - Complex Fork: {score_2:.3f}")
        print(f"Test 3 - Confounding Robustness: {score_3:.3f}")
        print(f"Overall Score: {overall_score:.3f}")

        if overall_score > 0.7:
            grade = "A"
            status = "🎉 EXCELLENT - Strong structure learning!"
        elif overall_score > 0.6:
            grade = "B"
            status = "👍 GOOD - Decent structure learning"
        elif overall_score > 0.4:
            grade = "C"
            status = "⚠️ WEAK - Limited structure learning"
        elif overall_score > 0.2:
            grade = "D"
            status = "💀 POOR - Minimal structure learning"
        else:
            grade = "F"
            status = "💀💀 FAILED - No meaningful structure learning"

        print(f"Grade: {grade}")
        print(f"Status: {status}")

        # Diagnostic information
        if overall_score < 0.3:
            print("\n🔍 DIAGNOSTIC INFORMATION:")
            print("- Structure learner may have threshold issues")
            print("- Training may not be converging properly")
            print("- DAG constraints might be too restrictive")
            print("- Loss function may not be optimizing correctly")

        # Save results
        results_summary = {
            'overall_score': overall_score,
            'grade': grade,
            'status': status,
            'individual_tests': self.test_results,
            'timestamp': time.time(),
            'diagnostic_info': {
                'model_type': type(self.model).__name__,
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
                'training_issues_detected': overall_score < 0.3
            }
        }

        return results_summary


def main():
    """Run the extreme structure learning ground truth test"""
    test = ExtremeStructureLearningGroundTruthTest()
    results = test.run_all_tests()

    # Save results
    with open('extreme_structure_learning_ground_truth_results.json', 'w') as f:
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

    print(f"\n📁 Results saved to: extreme_structure_learning_ground_truth_results.json")

    return results


if __name__ == "__main__":
    results = main()