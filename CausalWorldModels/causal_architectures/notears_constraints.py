"""
NOTEARS Constraints for Causal Structure Learning
Implementation of DAG constraints for neural causal discovery

Based on:
- NOTEARS: "DAGs with NO TEARS" (Zheng et al., 2018)
- Continuous Learning NOTEARS (CL-NOTEARS, 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class NOTEARSConstraint(nn.Module):
    """
    NOTEARS DAG constraint implementation for causal structure learning

    Enforces that learned adjacency matrix represents a valid DAG
    """

    def __init__(self, num_variables, constraint_weight=1.0, threshold=1e-3):
        super().__init__()
        self.num_variables = num_variables
        self.constraint_weight = constraint_weight
        self.threshold = threshold

    def compute_dag_constraint(self, adjacency_matrix):
        """
        Compute NOTEARS DAG constraint: Tr(e^(A ⊙ A)) - d = 0

        Args:
            adjacency_matrix: [num_vars, num_vars] weighted adjacency matrix

        Returns:
            constraint_value: Scalar constraint violation (0 = valid DAG)
        """
        # Element-wise square to ensure positive weights
        A_squared = adjacency_matrix * adjacency_matrix

        # Matrix exponential
        # For numerical stability, use eigendecomposition approach
        try:
            # Compute matrix exponential
            exp_A = torch.matrix_exp(A_squared)

            # DAG constraint: trace should equal number of variables
            constraint = torch.trace(exp_A) - self.num_variables

        except RuntimeError:
            # Fallback for numerical issues: use series approximation
            I = torch.eye(self.num_variables, device=adjacency_matrix.device)
            exp_A = I + A_squared

            # Add higher order terms for better approximation
            A_power = A_squared
            for k in range(2, 6):  # Up to A^5 / 5!
                A_power = torch.mm(A_power, A_squared) / k
                exp_A = exp_A + A_power

            constraint = torch.trace(exp_A) - self.num_variables

        return constraint

    def compute_sparsity_penalty(self, adjacency_matrix):
        """
        L1 sparsity penalty to encourage sparse graphs
        """
        return torch.sum(torch.abs(adjacency_matrix))

    def forward(self, adjacency_matrix):
        """
        Compute total constraint loss

        Args:
            adjacency_matrix: [num_vars, num_vars] adjacency matrix

        Returns:
            constraint_loss: Scalar loss combining DAG and sparsity constraints
            constraint_info: Dict with detailed constraint information
        """
        # DAG constraint
        dag_constraint = self.compute_dag_constraint(adjacency_matrix)
        dag_loss = self.constraint_weight * (dag_constraint ** 2)

        # Sparsity penalty
        sparsity_penalty = 0.1 * self.compute_sparsity_penalty(adjacency_matrix)

        # Total constraint loss
        total_loss = dag_loss + sparsity_penalty

        # Constraint information
        constraint_info = {
            'dag_constraint': dag_constraint.item(),
            'dag_violation': abs(dag_constraint.item()) > self.threshold,
            'sparsity_penalty': sparsity_penalty.item(),
            'total_constraint_loss': total_loss.item(),
            'num_edges': (torch.abs(adjacency_matrix) > 0.1).sum().item(),
            'max_edge_weight': torch.max(torch.abs(adjacency_matrix)).item()
        }

        return total_loss, constraint_info


class CausalGraphRegularizer(nn.Module):
    """
    Additional regularization for causal graph learning
    """

    def __init__(self, num_variables, temporal_consistency_weight=0.1):
        super().__init__()
        self.num_variables = num_variables
        self.temporal_consistency_weight = temporal_consistency_weight

    def temporal_consistency_loss(self, adjacency_t1, adjacency_t2):
        """
        Encourage temporal consistency in learned causal structure

        Args:
            adjacency_t1: Adjacency matrix at time t
            adjacency_t2: Adjacency matrix at time t+1

        Returns:
            consistency_loss: L2 penalty for graph changes
        """
        return F.mse_loss(adjacency_t1, adjacency_t2)

    def structural_prior_loss(self, adjacency_matrix, prior_structure=None):
        """
        Incorporate prior knowledge about causal structure

        Args:
            adjacency_matrix: Current adjacency matrix
            prior_structure: Prior adjacency matrix (if available)

        Returns:
            prior_loss: Penalty for deviating from prior structure
        """
        if prior_structure is None:
            return torch.tensor(0.0, device=adjacency_matrix.device)

        return F.mse_loss(adjacency_matrix, prior_structure)

    def forward(self, adjacency_matrix, prev_adjacency=None, prior_structure=None):
        """
        Compute regularization losses
        """
        total_loss = torch.tensor(0.0, device=adjacency_matrix.device)
        reg_info = {}

        # Temporal consistency
        if prev_adjacency is not None:
            temporal_loss = self.temporal_consistency_loss(adjacency_matrix, prev_adjacency)
            total_loss += self.temporal_consistency_weight * temporal_loss
            reg_info['temporal_consistency_loss'] = temporal_loss.item()

        # Structural prior
        if prior_structure is not None:
            prior_loss = self.structural_prior_loss(adjacency_matrix, prior_structure)
            total_loss += 0.05 * prior_loss  # Light weight on priors
            reg_info['structural_prior_loss'] = prior_loss.item()

        reg_info['total_regularization_loss'] = total_loss.item()

        return total_loss, reg_info


class AdaptiveDAGConstraint(nn.Module):
    """
    Adaptive DAG constraint with dynamic weighting during training
    """

    def __init__(self, num_variables, initial_weight=1.0, max_weight=10.0):
        super().__init__()
        self.num_variables = num_variables
        self.initial_weight = initial_weight
        self.max_weight = max_weight
        self.notears = NOTEARSConstraint(num_variables, initial_weight)
        self.regularizer = CausalGraphRegularizer(num_variables)

        # Adaptive weight parameters
        self.register_buffer('current_weight', torch.tensor(initial_weight))
        self.register_buffer('violation_history', torch.zeros(100))  # Track recent violations
        self.history_idx = 0

    def update_constraint_weight(self, constraint_violation):
        """
        Adaptively update constraint weight based on violation history
        """
        # Update violation history
        self.violation_history[self.history_idx] = abs(constraint_violation)
        self.history_idx = (self.history_idx + 1) % 100

        # Compute average recent violation
        avg_violation = self.violation_history.mean()

        # Increase weight if violations are persistent
        if avg_violation > 0.1:  # High violation threshold
            self.current_weight = torch.min(
                self.current_weight * 1.1,
                torch.tensor(self.max_weight)
            )
        elif avg_violation < 0.01:  # Low violation threshold
            self.current_weight = torch.max(
                self.current_weight * 0.95,
                torch.tensor(self.initial_weight)
            )

        # Update NOTEARS constraint weight
        self.notears.constraint_weight = self.current_weight.item()

    def forward(self, adjacency_matrix, prev_adjacency=None, prior_structure=None):
        """
        Compute adaptive DAG constraint
        """
        # NOTEARS constraint
        constraint_loss, constraint_info = self.notears(adjacency_matrix)

        # Update adaptive weight
        self.update_constraint_weight(constraint_info['dag_constraint'])

        # Additional regularization
        reg_loss, reg_info = self.regularizer(adjacency_matrix, prev_adjacency, prior_structure)

        # Combine losses
        total_loss = constraint_loss + reg_loss

        # Combined info
        combined_info = {**constraint_info, **reg_info}
        combined_info['adaptive_weight'] = self.current_weight.item()
        combined_info['avg_violation_history'] = self.violation_history.mean().item()

        return total_loss, combined_info


def test_notears_constraints():
    """
    Test NOTEARS constraint implementation
    """
    print("Testing NOTEARS constraints...")

    num_vars = 5

    # Test valid DAG (upper triangular)
    valid_dag = torch.triu(torch.randn(num_vars, num_vars), diagonal=1)

    # Test invalid graph (with cycles)
    invalid_graph = torch.randn(num_vars, num_vars)

    # Create constraint
    constraint = NOTEARSConstraint(num_vars)

    # Test valid DAG
    valid_loss, valid_info = constraint(valid_dag)
    print(f"Valid DAG constraint: {valid_info['dag_constraint']:.6f}")

    # Test invalid graph
    invalid_loss, invalid_info = constraint(invalid_graph)
    print(f"Invalid graph constraint: {invalid_info['dag_constraint']:.6f}")

    # Valid DAG should have lower constraint violation
    assert abs(valid_info['dag_constraint']) < abs(invalid_info['dag_constraint'])

    print("✅ NOTEARS constraint test passed")

    # Test adaptive constraint
    adaptive = AdaptiveDAGConstraint(num_vars)
    adaptive_loss, adaptive_info = adaptive(invalid_graph)
    print(f"✅ Adaptive constraint weight: {adaptive_info['adaptive_weight']:.3f}")

    return True


def visualize_adjacency_matrix(adjacency_matrix, variable_names=None):
    """
    Visualize adjacency matrix (for debugging)

    Args:
        adjacency_matrix: [num_vars, num_vars] adjacency matrix
        variable_names: List of variable names for labeling
    """
    import matplotlib.pyplot as plt

    if variable_names is None:
        variable_names = [f"Var_{i}" for i in range(adjacency_matrix.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert to numpy for visualization
    adj_np = adjacency_matrix.detach().cpu().numpy()

    # Create heatmap
    im = ax.imshow(adj_np, cmap='RdBu_r', vmin=-1, vmax=1)

    # Add labels
    ax.set_xticks(range(len(variable_names)))
    ax.set_yticks(range(len(variable_names)))
    ax.set_xticklabels(variable_names)
    ax.set_yticklabels(variable_names)

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Edge Weight')

    # Add edge weights as text
    for i in range(len(variable_names)):
        for j in range(len(variable_names)):
            if abs(adj_np[i, j]) > 0.1:  # Only show significant edges
                ax.text(j, i, f'{adj_np[i, j]:.2f}',
                       ha="center", va="center", color="white" if abs(adj_np[i, j]) > 0.5 else "black")

    ax.set_title('Causal Adjacency Matrix')
    ax.set_xlabel('Causes')
    ax.set_ylabel('Effects')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    test_notears_constraints()