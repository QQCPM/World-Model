"""
Structure Validator
Comprehensive validation framework for causal structure learning

Tests:
- Graph Recovery Accuracy vs ground truth
- Intervention Identification capability
- Structure Stability over training
- DAG constraint satisfaction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score


class StructureValidator:
    """
    Validation framework for causal structure learning accuracy

    Evaluates learned causal graphs against ground truth and stability metrics
    """

    def __init__(self, variable_names=None):
        self.variable_names = variable_names or [
            'weather', 'crowd_density', 'special_event', 'time_of_day', 'road_conditions'
        ]
        self.num_variables = len(self.variable_names)

    def validate_structure_learning(self, structure_learner, ground_truth_graph=None,
                                  validation_data=None, stability_epochs=10):
        """
        Comprehensive structure learning validation

        Args:
            structure_learner: CausalStructureLearner instance
            ground_truth_graph: Optional [num_vars, num_vars] true causal graph
            validation_data: Validation dataset
            stability_epochs: Number of epochs to test stability

        Returns:
            validation_results: Dict with comprehensive validation metrics
        """
        print("🏗️  Validating Causal Structure Learning")
        print("-" * 40)

        validation_results = {
            'graph_properties': {},
            'dag_constraints': {},
            'stability_analysis': {},
            'recovery_accuracy': {},
            'intervention_identification': {}
        }

        # Get current learned structure
        learned_graph = structure_learner.get_adjacency_matrix()
        structure_summary = structure_learner.get_causal_graph_summary()

        # 1. GRAPH PROPERTIES ANALYSIS
        print("📊 Analyzing graph properties...")
        validation_results['graph_properties'] = self._analyze_graph_properties(
            learned_graph, structure_summary
        )

        # 2. DAG CONSTRAINTS VALIDATION
        print("🔗 Validating DAG constraints...")
        validation_results['dag_constraints'] = self._validate_dag_constraints(
            structure_learner, learned_graph
        )

        # 3. STABILITY ANALYSIS
        print("📈 Analyzing structure stability...")
        if validation_data is not None:
            validation_results['stability_analysis'] = self._analyze_structure_stability(
                structure_learner, validation_data, stability_epochs
            )

        # 4. RECOVERY ACCURACY (if ground truth available)
        if ground_truth_graph is not None:
            print("🎯 Computing recovery accuracy...")
            validation_results['recovery_accuracy'] = self._compute_recovery_accuracy(
                learned_graph, ground_truth_graph
            )

        # 5. INTERVENTION IDENTIFICATION
        print("🎛️  Testing intervention identification...")
        validation_results['intervention_identification'] = self._test_intervention_identification(
            structure_learner, validation_data
        )

        # OVERALL STRUCTURE SCORE
        overall_score = self._compute_overall_structure_score(validation_results)
        validation_results['overall_structure_score'] = overall_score

        self._print_structure_summary(validation_results)

        return validation_results

    def _analyze_graph_properties(self, learned_graph, structure_summary):
        """
        Analyze basic properties of learned causal graph
        """
        # Convert to numpy for analysis
        adj_matrix = learned_graph.detach().cpu().numpy()

        # Basic graph metrics
        num_edges = structure_summary['num_edges']
        sparsity = structure_summary['sparsity']
        density = structure_summary['graph_density']

        # Network analysis using NetworkX
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

        # Connectivity analysis
        is_connected = nx.is_weakly_connected(G)
        num_components = nx.number_weakly_connected_components(G)

        # Centrality measures
        in_degree_centrality = nx.in_degree_centrality(G)
        out_degree_centrality = nx.out_degree_centrality(G)

        # Identify hubs (high centrality nodes)
        max_in_centrality = max(in_degree_centrality.values())
        max_out_centrality = max(out_degree_centrality.values())

        hub_nodes = []
        for node, centrality in in_degree_centrality.items():
            if centrality > 0.5:  # High centrality threshold
                hub_nodes.append(self.variable_names[node])

        # Path analysis
        try:
            avg_path_length = nx.average_shortest_path_length(G.to_undirected())
        except:
            avg_path_length = float('inf')  # Disconnected graph

        properties = {
            'num_edges': num_edges,
            'sparsity': sparsity,
            'density': density,
            'is_connected': is_connected,
            'num_components': num_components,
            'max_in_centrality': max_in_centrality,
            'max_out_centrality': max_out_centrality,
            'hub_nodes': hub_nodes,
            'average_path_length': avg_path_length,
            'has_cycles': not nx.is_directed_acyclic_graph(G),
            'topological_order_exists': nx.is_directed_acyclic_graph(G)
        }

        return properties

    def _validate_dag_constraints(self, structure_learner, learned_graph):
        """
        Validate DAG constraints and structure quality
        """
        # Test DAG constraint satisfaction
        dag_constraint = structure_learner.dag_constraint
        constraint_loss, constraint_info = dag_constraint(learned_graph)

        # Additional DAG validation
        adj_np = learned_graph.detach().cpu().numpy()

        # Check for self-loops
        has_self_loops = np.diag(adj_np).sum() > 0.01

        # Check for bidirectional edges (should be minimal in DAG)
        bidirectional_edges = 0
        for i in range(self.num_variables):
            for j in range(self.num_variables):
                if i != j and adj_np[i, j] > 0.1 and adj_np[j, i] > 0.1:
                    bidirectional_edges += 1

        # Matrix properties
        matrix_rank = np.linalg.matrix_rank(adj_np)
        spectral_radius = np.max(np.abs(np.linalg.eigvals(adj_np)))

        dag_validation = {
            'dag_constraint_violation': constraint_info['dag_constraint'],
            'is_valid_dag': abs(constraint_info['dag_constraint']) < 0.1,
            'has_self_loops': has_self_loops,
            'bidirectional_edges': bidirectional_edges,
            'matrix_rank': matrix_rank,
            'spectral_radius': spectral_radius,
            'sparsity_penalty': constraint_info['sparsity_penalty'],
            'max_edge_weight': constraint_info['max_edge_weight']
        }

        return dag_validation

    def _analyze_structure_stability(self, structure_learner, validation_data,
                                   stability_epochs):
        """
        Analyze stability of learned structure over time
        """
        if validation_data is None:
            return {'status': 'no_validation_data'}

        stability_metrics = {
            'adjacency_variance': [],
            'edge_flip_rate': [],
            'structure_distance': []
        }

        # Store initial structure
        initial_graph = structure_learner.get_adjacency_matrix().detach().clone()
        previous_graph = initial_graph.clone()

        # Test stability over multiple validation passes
        for epoch in range(stability_epochs):
            # Update structure with validation data
            for batch_data in validation_data:
                states, actions, causal_factors = batch_data

                with torch.no_grad():
                    structure_learner.update_structure_confidence(causal_factors)

                break  # Use first batch

            # Get updated structure
            current_graph = structure_learner.get_adjacency_matrix().detach()

            # Compute stability metrics
            adjacency_var = torch.var(current_graph).item()
            stability_metrics['adjacency_variance'].append(adjacency_var)

            # Edge flip rate (how many edges changed significantly)
            edge_changes = torch.abs(current_graph - previous_graph) > 0.1
            flip_rate = edge_changes.float().mean().item()
            stability_metrics['edge_flip_rate'].append(flip_rate)

            # Structure distance (Frobenius norm)
            structure_distance = torch.norm(current_graph - initial_graph).item()
            stability_metrics['structure_distance'].append(structure_distance)

            previous_graph = current_graph.clone()

        # Compute overall stability scores
        avg_variance = np.mean(stability_metrics['adjacency_variance'])
        avg_flip_rate = np.mean(stability_metrics['edge_flip_rate'])
        final_distance = stability_metrics['structure_distance'][-1]

        # Stability score (lower variance and flip rate = more stable)
        stability_score = 1.0 / (1.0 + avg_variance + avg_flip_rate)

        stability_analysis = {
            'stability_score': stability_score,
            'average_adjacency_variance': avg_variance,
            'average_edge_flip_rate': avg_flip_rate,
            'final_structure_distance': final_distance,
            'is_stable': stability_score > 0.7,
            'stability_trend': 'improving' if len(stability_metrics['edge_flip_rate']) > 1 and
                             stability_metrics['edge_flip_rate'][-1] < stability_metrics['edge_flip_rate'][0]
                             else 'stable'
        }

        return stability_analysis

    def _compute_recovery_accuracy(self, learned_graph, ground_truth_graph):
        """
        Compute structure recovery accuracy against ground truth
        """
        # Convert to binary adjacency matrices
        learned_binary = (learned_graph.detach().cpu().numpy() > 0.1).astype(int)
        truth_binary = (ground_truth_graph > 0.1).astype(int)

        # Flatten matrices for sklearn metrics
        learned_flat = learned_binary.flatten()
        truth_flat = truth_binary.flatten()

        # Edge-level metrics
        precision = precision_score(truth_flat, learned_flat, zero_division=0)
        recall = recall_score(truth_flat, learned_flat, zero_division=0)
        f1 = f1_score(truth_flat, learned_flat, zero_division=0)

        # Structural Hamming Distance
        hamming_distance = np.sum(learned_flat != truth_flat)
        total_possible_edges = self.num_variables * (self.num_variables - 1)
        structural_accuracy = 1.0 - (hamming_distance / len(truth_flat))

        # Parent set accuracy (for each variable, how well did we recover its parents)
        parent_accuracies = []
        for var in range(self.num_variables):
            true_parents = set(np.where(truth_binary[:, var] > 0)[0])
            learned_parents = set(np.where(learned_binary[:, var] > 0)[0])

            if len(true_parents) == 0 and len(learned_parents) == 0:
                parent_accuracy = 1.0
            elif len(true_parents) == 0:
                parent_accuracy = 0.0
            else:
                intersection = len(true_parents.intersection(learned_parents))
                union = len(true_parents.union(learned_parents))
                parent_accuracy = intersection / len(true_parents) if len(true_parents) > 0 else 0.0

            parent_accuracies.append(parent_accuracy)

        # Orientation accuracy (for edges that exist in both, are they oriented correctly?)
        orientation_accuracy = 0.0
        common_edges = 0

        for i in range(self.num_variables):
            for j in range(self.num_variables):
                if truth_binary[i, j] > 0 or learned_binary[i, j] > 0:
                    common_edges += 1
                    if truth_binary[i, j] == learned_binary[i, j]:
                        orientation_accuracy += 1

        if common_edges > 0:
            orientation_accuracy /= common_edges

        recovery_accuracy = {
            'edge_precision': precision,
            'edge_recall': recall,
            'edge_f1_score': f1,
            'structural_accuracy': structural_accuracy,
            'parent_set_accuracy': np.mean(parent_accuracies),
            'orientation_accuracy': orientation_accuracy,
            'hamming_distance': hamming_distance,
            'total_possible_edges': total_possible_edges,
            'recovery_grade': self._grade_recovery_accuracy(f1)
        }

        return recovery_accuracy

    def _test_intervention_identification(self, structure_learner, validation_data):
        """
        Test ability to identify interventional vs observational data
        """
        if validation_data is None:
            return {'status': 'no_validation_data'}

        identification_scores = []

        for batch_data in validation_data:
            states, actions, causal_factors = batch_data

            # Create mock intervention data
            intervention_causal = causal_factors.clone()
            intervention_var = torch.randint(0, self.num_variables, (1,)).item()
            intervention_causal[:, :, intervention_var] = torch.randn_like(
                intervention_causal[:, :, intervention_var]
            )

            # Test if structure learner can distinguish
            with torch.no_grad():
                obs_loss, obs_info = structure_learner.compute_structure_loss(causal_factors)
                int_loss, int_info = structure_learner.compute_structure_loss(intervention_causal)

            # Good structure learner should have different losses for interventional data
            loss_difference = abs(obs_loss.item() - int_loss.item())
            identification_score = min(loss_difference / obs_loss.item(), 1.0)
            identification_scores.append(identification_score)

            break  # Test on first batch

        avg_identification = np.mean(identification_scores)

        identification_results = {
            'identification_score': avg_identification,
            'can_distinguish_interventions': avg_identification > 0.1,
            'identification_grade': 'Good' if avg_identification > 0.2 else 'Poor'
        }

        return identification_results

    def _compute_overall_structure_score(self, validation_results):
        """
        Compute overall structure learning score
        """
        scores = []

        # DAG constraint score
        dag_valid = validation_results['dag_constraints']['is_valid_dag']
        scores.append(1.0 if dag_valid else 0.3)

        # Stability score
        if 'stability_score' in validation_results['stability_analysis']:
            stability = validation_results['stability_analysis']['stability_score']
            scores.append(stability)

        # Recovery accuracy (if available)
        if validation_results['recovery_accuracy']:
            f1_score = validation_results['recovery_accuracy'].get('edge_f1_score', 0.5)
            scores.append(f1_score)

        # Intervention identification
        if validation_results['intervention_identification']:
            id_score = validation_results['intervention_identification'].get('identification_score', 0.5)
            scores.append(id_score)

        # Graph properties (reasonable structure)
        props = validation_results['graph_properties']
        structure_reasonableness = 1.0
        if props['has_cycles']:
            structure_reasonableness *= 0.7  # Penalize cycles
        if props['density'] > 0.8:
            structure_reasonableness *= 0.8  # Penalize overly dense graphs
        scores.append(structure_reasonableness)

        overall_score = np.mean(scores)

        return {
            'overall_score': overall_score,
            'grade': self._grade_structure_score(overall_score),
            'component_scores': {
                'dag_constraints': scores[0],
                'stability': scores[1] if len(scores) > 1 else 0.5,
                'recovery_accuracy': scores[2] if len(scores) > 2 else 0.5,
                'intervention_identification': scores[3] if len(scores) > 3 else 0.5,
                'structure_reasonableness': scores[-1]
            }
        }

    def _grade_recovery_accuracy(self, f1_score):
        """Grade recovery accuracy based on F1 score"""
        if f1_score >= 0.9:
            return 'Excellent'
        elif f1_score >= 0.8:
            return 'Very Good'
        elif f1_score >= 0.7:
            return 'Good'
        elif f1_score >= 0.6:
            return 'Fair'
        else:
            return 'Poor'

    def _grade_structure_score(self, score):
        """Grade overall structure learning performance"""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B+'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.5:
            return 'C'
        else:
            return 'F'

    def _print_structure_summary(self, validation_results):
        """Print structure validation summary"""
        print("\n" + "="*50)
        print("🏗️  STRUCTURE LEARNING VALIDATION SUMMARY")
        print("="*50)

        overall = validation_results['overall_structure_score']
        props = validation_results['graph_properties']
        dag = validation_results['dag_constraints']

        print(f"📊 Overall Grade: {overall['grade']}")
        print(f"📈 Overall Score: {overall['overall_score']:.3f}")
        print()

        print(f"🔗 DAG Constraints: {'✅ Valid' if dag['is_valid_dag'] else '❌ Invalid'}")
        print(f"📊 Graph Properties:")
        print(f"   - Edges: {props['num_edges']}")
        print(f"   - Sparsity: {props['sparsity']:.3f}")
        print(f"   - Connected: {'✅' if props['is_connected'] else '❌'}")
        print(f"   - Is DAG: {'✅' if props['topological_order_exists'] else '❌'}")

        if validation_results['stability_analysis']:
            stability = validation_results['stability_analysis']
            print(f"📈 Stability: {'✅ Stable' if stability.get('is_stable', False) else '❌ Unstable'}")

        if validation_results['recovery_accuracy']:
            recovery = validation_results['recovery_accuracy']
            print(f"🎯 Recovery: {recovery['recovery_grade']} (F1: {recovery['edge_f1_score']:.3f})")

        print("="*50)

    def create_ground_truth_campus_graph(self):
        """
        Create ground truth causal graph for campus environment

        Returns:
            ground_truth: [5, 5] adjacency matrix representing true causal structure
        """
        # Define true causal relationships for campus environment
        # Variables: [weather, crowd_density, special_event, time_of_day, road_conditions]

        ground_truth = torch.zeros(5, 5)

        # Weather affects road conditions
        ground_truth[0, 4] = 0.8  # weather → road_conditions

        # Time of day affects crowd density
        ground_truth[3, 1] = 0.6  # time_of_day → crowd_density

        # Special events affect crowd density
        ground_truth[2, 1] = 0.9  # special_event → crowd_density

        # Weather affects crowd (people avoid bad weather)
        ground_truth[0, 1] = 0.4  # weather → crowd_density

        # Time affects road conditions (maintenance, usage)
        ground_truth[3, 4] = 0.3  # time_of_day → road_conditions

        return ground_truth


def test_structure_validator():
    """
    Test structure validator functionality
    """
    print("Testing StructureValidator...")

    # Create validator
    validator = StructureValidator()

    # Create mock learned graph
    learned_graph = torch.rand(5, 5) * 0.5

    # Test graph properties analysis
    from types import SimpleNamespace
    mock_summary = {
        'num_edges': 8,
        'sparsity': 0.6,
        'graph_density': 0.4
    }

    properties = validator._analyze_graph_properties(learned_graph, mock_summary)
    print(f"Graph properties: {len(properties)} metrics computed")

    # Test DAG validation
    class MockDAGConstraint:
        def __call__(self, graph):
            return torch.tensor(0.05), {
                'dag_constraint': 0.05,
                'sparsity_penalty': 0.1,
                'max_edge_weight': 0.8
            }

    class MockStructureLearner:
        def __init__(self):
            self.dag_constraint = MockDAGConstraint()

        def get_adjacency_matrix(self):
            return learned_graph

    mock_learner = MockStructureLearner()
    dag_validation = validator._validate_dag_constraints(mock_learner, learned_graph)
    print(f"DAG validation: {'Valid' if dag_validation['is_valid_dag'] else 'Invalid'}")

    # Test ground truth graph creation
    ground_truth = validator.create_ground_truth_campus_graph()
    print(f"Ground truth graph shape: {ground_truth.shape}")
    print(f"Ground truth edges: {(ground_truth > 0.1).sum().item()}")

    # Test recovery accuracy
    recovery = validator._compute_recovery_accuracy(learned_graph, ground_truth)
    print(f"Recovery F1 score: {recovery['edge_f1_score']:.3f}")

    print("✅ StructureValidator test passed")

    return True


if __name__ == "__main__":
    test_structure_validator()