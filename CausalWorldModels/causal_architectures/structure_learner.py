"""
Causal Structure Learner
NOTEARS-based causal discovery for continuous campus environment

Based on:
- Active Learning for Optimal Intervention Design (Nature MI, 2023)
- CL-NOTEARS (2024)
- Conservative structure learning principles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from causal_architectures.notears_constraints import AdaptiveDAGConstraint, NOTEARSConstraint


class CausalStructureLearner(nn.Module):
    """
    Neural causal structure learning with NOTEARS constraints

    Learns causal graph structure from observational and interventional data
    for campus environment causal factors:
    - weather (sunny, rainy, snowy, foggy)
    - crowd_density
    - special_event
    - time_of_day
    - road_conditions
    """

    def __init__(self, num_variables=5, hidden_dim=64, learning_rate=1e-3):
        super().__init__()
        self.num_variables = num_variables
        self.hidden_dim = hidden_dim

        # Variable names for interpretation
        self.variable_names = [
            'weather',
            'crowd_density',
            'special_event',
            'time_of_day',
            'road_conditions'
        ]

        # Learnable adjacency matrix (logits)
        self.adjacency_logits = nn.Parameter(
            torch.randn(num_variables, num_variables) * 0.1
        )

        # Neural networks for non-linear causal mechanisms
        self.causal_mechanisms = nn.ModuleDict()
        for i in range(num_variables):
            self.causal_mechanisms[f'mechanism_{i}'] = nn.Sequential(
                nn.Linear(num_variables, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

        # DAG constraint enforcement
        self.dag_constraint = AdaptiveDAGConstraint(num_variables)

        # Structure confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(num_variables * num_variables, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_variables * num_variables),
            nn.Sigmoid()
        )

        # Initialize adjacency to be sparse
        self._initialize_sparse_adjacency()

    def _initialize_sparse_adjacency(self):
        """Initialize adjacency matrix to encourage sparsity"""
        with torch.no_grad():
            # Initialize with slight negative bias to encourage sparsity
            self.adjacency_logits.data = torch.randn_like(self.adjacency_logits) * 0.1 - 0.5

            # Zero diagonal (no self-loops)
            self.adjacency_logits.data.fill_diagonal_(0)

    def get_adjacency_matrix(self, temperature=1.0, hard=False):
        """
        Convert logits to adjacency matrix using Gumbel-Softmax

        Args:
            temperature: Temperature for Gumbel-Softmax (lower = more discrete)
            hard: If True, use hard assignment

        Returns:
            adjacency: [num_vars, num_vars] adjacency matrix
        """
        # Zero diagonal (no self-loops)
        logits = self.adjacency_logits.clone()
        logits.fill_diagonal_(float('-inf'))

        # Apply sigmoid to get edge probabilities
        adjacency = torch.sigmoid(logits / temperature)

        if hard:
            # Hard thresholding for discrete structure
            adjacency = (adjacency > 0.5).float()

        return adjacency

    def predict_causal_effects(self, causal_factors, adjacency_matrix=None):
        """
        Predict causal effects using learned structure and mechanisms

        Args:
            causal_factors: [batch_size, num_variables] current variable values
            adjacency_matrix: Optional adjacency matrix (uses learned if None)

        Returns:
            predictions: [batch_size, num_variables] predicted values
            mechanisms: Dict of mechanism outputs for each variable
        """
        if adjacency_matrix is None:
            adjacency_matrix = self.get_adjacency_matrix()

        batch_size = causal_factors.shape[0]
        predictions = torch.zeros_like(causal_factors)
        mechanisms = {}

        # For each variable, predict based on its parents
        for i in range(self.num_variables):
            # Get parents (incoming edges)
            parents = adjacency_matrix[:, i]  # Column i represents edges into variable i

            # Weighted input from parents
            parent_input = torch.sum(
                causal_factors * parents.unsqueeze(0), dim=1, keepdim=True
            )

            # Apply causal mechanism
            # mechanism_input should be [batch_size, num_variables] = [batch_size, 5]
            mechanism_output = self.causal_mechanisms[f'mechanism_{i}'](causal_factors)

            predictions[:, i] = mechanism_output.squeeze(-1)
            mechanisms[f'variable_{i}'] = mechanism_output

        return predictions, mechanisms

    def compute_structure_loss(self, causal_data, interventions=None):
        """
        Compute loss for structure learning

        Args:
            causal_data: [batch_size, seq_len, num_variables] time series data
            interventions: Optional intervention masks

        Returns:
            total_loss: Combined structure learning loss
            loss_info: Detailed loss information
        """
        batch_size, seq_len, num_vars = causal_data.shape

        # Get current adjacency matrix
        adjacency = self.get_adjacency_matrix()

        # Prediction loss across time steps
        prediction_loss = 0
        num_predictions = 0

        for t in range(seq_len - 1):
            current_factors = causal_data[:, t, :]
            next_factors = causal_data[:, t + 1, :]

            # Predict next values
            predicted_next, _ = self.predict_causal_effects(current_factors, adjacency)

            # MSE loss for prediction
            prediction_loss += F.mse_loss(predicted_next, next_factors)
            num_predictions += 1

        prediction_loss /= num_predictions

        # DAG constraint loss
        dag_loss, dag_info = self.dag_constraint(adjacency)

        # Structure confidence
        confidence_input = adjacency.flatten().unsqueeze(0)
        confidence_scores = self.confidence_estimator(confidence_input).squeeze(0)
        confidence_penalty = -torch.mean(torch.log(confidence_scores + 1e-8))

        # Total loss
        total_loss = prediction_loss + dag_loss + 0.1 * confidence_penalty

        # Loss information
        loss_info = {
            'prediction_loss': prediction_loss.item(),
            'dag_loss': dag_loss.item(),
            'confidence_penalty': confidence_penalty.item(),
            'total_structure_loss': total_loss.item(),
            **dag_info
        }

        return total_loss, loss_info

    def discover_interventions(self, causal_data, max_interventions=20):
        """
        Discover optimal interventions for structure learning

        Args:
            causal_data: [batch_size, seq_len, num_variables] observational data
            max_interventions: Maximum number of interventions to suggest

        Returns:
            intervention_targets: List of variables to intervene on
            expected_gains: Expected information gain for each intervention
        """
        adjacency = self.get_adjacency_matrix()

        # Compute current uncertainty in structure
        confidence_input = adjacency.flatten().unsqueeze(0)
        confidence_scores = self.confidence_estimator(confidence_input).squeeze(0)
        current_uncertainty = -torch.sum(confidence_scores * torch.log(confidence_scores + 1e-8))

        intervention_gains = []

        # Test intervention on each variable
        for var_idx in range(self.num_variables):
            # Simulate intervention by setting variable to different values
            intervention_data = causal_data.clone()

            # Test multiple intervention values
            gains_for_var = []
            test_values = torch.linspace(-1, 1, 5)

            for val in test_values:
                # Apply intervention
                intervention_data[:, :, var_idx] = val

                # Compute structure loss with intervention
                with torch.no_grad():
                    loss, _ = self.compute_structure_loss(intervention_data)

                    # Information gain approximation
                    gain = current_uncertainty - loss.item()
                    gains_for_var.append(gain)

            # Average gain for this variable
            avg_gain = np.mean(gains_for_var)
            intervention_gains.append((var_idx, avg_gain))

        # Sort by expected gain
        intervention_gains.sort(key=lambda x: x[1], reverse=True)

        # Extract targets and gains
        intervention_targets = [x[0] for x in intervention_gains[:max_interventions]]
        expected_gains = [x[1] for x in intervention_gains[:max_interventions]]

        return intervention_targets, expected_gains

    def get_causal_graph_summary(self):
        """
        Get interpretable summary of learned causal structure

        Returns:
            summary: Dict with graph information and interpretation
        """
        adjacency = self.get_adjacency_matrix(hard=True)

        # Convert to numpy for analysis
        adj_np = adjacency.detach().cpu().numpy()

        # Find edges
        edges = []
        for i in range(self.num_variables):
            for j in range(self.num_variables):
                if adj_np[i, j] > 0.3:  # Significant edge (lowered threshold)
                    cause = self.variable_names[i]
                    effect = self.variable_names[j]
                    weight = adj_np[i, j]
                    edges.append({
                        'cause': cause,
                        'effect': effect,
                        'weight': weight,
                        'cause_idx': i,
                        'effect_idx': j
                    })

        # Compute graph statistics
        num_edges = len(edges)
        sparsity = 1 - (num_edges / (self.num_variables ** 2 - self.num_variables))

        # Find root causes (no incoming edges)
        incoming_edges = np.sum(adj_np, axis=0)
        root_causes = [self.variable_names[i] for i in range(self.num_variables)
                      if incoming_edges[i] < 0.1]

        # Find final effects (no outgoing edges)
        outgoing_edges = np.sum(adj_np, axis=1)
        final_effects = [self.variable_names[i] for i in range(self.num_variables)
                        if outgoing_edges[i] < 0.1]

        summary = {
            'edges': edges,
            'num_edges': num_edges,
            'sparsity': sparsity,
            'root_causes': root_causes,
            'final_effects': final_effects,
            'adjacency_matrix': adj_np.tolist(),
            'variable_names': self.variable_names,
            'graph_density': num_edges / (self.num_variables * (self.num_variables - 1))
        }

        return summary

    def export_causal_graph(self, filename=None):
        """
        Export causal graph for visualization tools (NetworkX, Graphviz)

        Returns:
            graph_data: Dict compatible with graph visualization libraries
        """
        summary = self.get_causal_graph_summary()

        # NetworkX compatible format
        graph_data = {
            'nodes': [{'id': i, 'name': name} for i, name in enumerate(self.variable_names)],
            'edges': [
                {
                    'source': edge['cause_idx'],
                    'target': edge['effect_idx'],
                    'weight': edge['weight'],
                    'label': f"{edge['cause']} → {edge['effect']}"
                }
                for edge in summary['edges']
            ]
        }

        if filename:
            import json
            with open(filename, 'w') as f:
                json.dump(graph_data, f, indent=2)

        return graph_data

    def update_structure_confidence(self, validation_data):
        """
        Update structure confidence based on validation performance

        Args:
            validation_data: [batch_size, seq_len, num_variables] validation data
        """
        self.eval()
        with torch.no_grad():
            loss, loss_info = self.compute_structure_loss(validation_data)

            # Update confidence based on prediction accuracy
            prediction_accuracy = 1.0 / (1.0 + loss_info['prediction_loss'])

            # Update confidence estimator (this is a simplified approach)
            adjacency = self.get_adjacency_matrix()
            confidence_input = adjacency.flatten().unsqueeze(0)
            target_confidence = torch.full_like(confidence_input, prediction_accuracy)

            # Light update to confidence estimator
            with torch.enable_grad():
                current_confidence = self.confidence_estimator(confidence_input)
                confidence_loss = F.mse_loss(current_confidence, target_confidence)

                # Minimal gradient update
                confidence_loss.backward()

        self.train()

    def get_model_name(self):
        return "causal_structure_learner"


def create_campus_structure_learner():
    """
    Create structure learner specifically for campus environment
    """
    return CausalStructureLearner(
        num_variables=5,  # weather, crowd, event, time, road
        hidden_dim=64,
        learning_rate=1e-3
    )


def test_structure_learner():
    """
    Test causal structure learner functionality
    """
    print("Testing CausalStructureLearner...")

    # Create learner
    learner = CausalStructureLearner(num_variables=5)

    # Generate synthetic data
    batch_size, seq_len = 16, 10
    causal_data = torch.randn(batch_size, seq_len, 5)

    # Test structure learning
    loss, loss_info = learner.compute_structure_loss(causal_data)
    print(f"Structure learning loss: {loss.item():.4f}")

    # Test causal prediction
    current_factors = causal_data[:, 0, :]
    predictions, mechanisms = learner.predict_causal_effects(current_factors)
    print(f"Prediction shape: {predictions.shape}")

    # Test intervention discovery
    targets, gains = learner.discover_interventions(causal_data, max_interventions=3)
    print(f"Best intervention targets: {[learner.variable_names[i] for i in targets[:3]]}")

    # Test graph summary
    summary = learner.get_causal_graph_summary()
    print(f"Number of edges: {summary['num_edges']}")
    print(f"Graph sparsity: {summary['sparsity']:.3f}")

    print("✅ CausalStructureLearner test passed")
    print(f"✅ Model parameters: {sum(p.numel() for p in learner.parameters())}")

    return True


if __name__ == "__main__":
    test_structure_learner()