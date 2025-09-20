"""
Dual-Pathway Causal GRU Architecture
Research-validated two-pathway system for genuine causal reasoning

Based on:
- SENA-discrepancy-VAE (2024)
- GraCE-VAE approaches (2024)
- Conservative training principles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union


class DualPathwayCausalGRU(nn.Module):
    """
    Research-validated dual-pathway architecture for causal reasoning

    Two pathways:
    1. Observational: Learn from observational data (correlations)
    2. Interventional: Learn from interventional data (causations)

    Maintains backward compatibility with existing GRUDynamics interface
    """

    def __init__(self, state_dim=12, action_dim=2, causal_dim=5, hidden_dim=64, num_layers=1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.causal_dim = causal_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input dimension: [state, action, causal_factors]
        input_dim = state_dim + action_dim + causal_dim

        # Observational pathway: standard dynamics learning
        self.observational_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Interventional pathway: do-operations and counterfactuals
        self.interventional_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Pathway selection mechanism
        self.pathway_selector = nn.Linear(input_dim, 2)  # [obs_weight, int_weight]

        # Shared output layers (preserve proven architecture from GRUDynamics)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, state_dim)
        )

        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.9))

        # Pathway mixture weights (learnable)
        self.pathway_weights = nn.Parameter(torch.tensor([0.7, 0.3]))  # [obs, int]

        # Causal intervention detector
        self.intervention_detector = nn.Sequential(
            nn.Linear(causal_dim, causal_dim * 2),
            nn.ReLU(),
            nn.Linear(causal_dim * 2, 1),
            nn.Sigmoid()
        )

        # MMD pathway specialization measurement
        self.specialization_enabled = True

    def compute_mmd_specialization_loss(self, obs_output, int_output, sigma=1.0):
        """
        Compute Maximum Mean Discrepancy to encourage pathway specialization
        Higher MMD = more specialized pathways
        """
        if not self.specialization_enabled:
            return torch.tensor(0.0, device=obs_output.device)

        batch_size = obs_output.shape[0]
        if batch_size < 2:  # Need at least 2 samples for MMD
            return torch.tensor(0.0, device=obs_output.device)

        # Flatten outputs for MMD computation (use reshape for non-contiguous tensors)
        obs_flat = obs_output.reshape(batch_size, -1)
        int_flat = int_output.reshape(batch_size, -1)

        # Compute MMD using Gaussian kernel
        mmd = self.compute_mmd(obs_flat, int_flat, sigma)

        # Return negative MMD as loss (minimize = maximize specialization)
        return -mmd

    def compute_mmd(self, x, y, sigma=1.0):
        """
        Maximum Mean Discrepancy for measuring distribution difference
        """
        xx = self.gaussian_kernel(x, x, sigma)
        yy = self.gaussian_kernel(y, y, sigma)
        xy = self.gaussian_kernel(x, y, sigma)

        mmd = torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
        return mmd

    def gaussian_kernel(self, x, y, sigma=1.0):
        """
        RBF kernel for MMD computation
        """
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))

    def compute_specialization_score(self, obs_output, int_output):
        """
        Compute pathway specialization score (0 = identical, 1 = maximally different)
        """
        # Compute pathway difference magnitude
        pathway_diff = torch.mean(torch.abs(obs_output - int_output))

        # Normalize by pathway output scales
        obs_scale = torch.std(obs_output) + 1e-8
        int_scale = torch.std(int_output) + 1e-8
        avg_scale = (obs_scale + int_scale) / 2

        specialization_score = pathway_diff / avg_scale

        # Clip to [0, 1] range
        return torch.clamp(specialization_score, 0.0, 1.0)

    def detect_intervention(self, causal_factors):
        """
        Detect if current input represents an intervention (do-operation)
        vs observational data

        Args:
            causal_factors: [batch_size, seq_len, causal_dim] or [batch_size, causal_dim]

        Returns:
            intervention_score: [batch_size, seq_len, 1] or [batch_size, 1]
        """
        return self.intervention_detector(causal_factors)

    def forward(self, state_sequence, action_sequence, causal_sequence,
                intervention_mask=None, hidden_obs=None, hidden_int=None):
        """
        Dual-pathway forward pass with automatic intervention detection

        Args:
            state_sequence: [batch_size, seq_len, state_dim]
            action_sequence: [batch_size, seq_len, action_dim]
            causal_sequence: [batch_size, seq_len, causal_dim]
            intervention_mask: [batch_size, seq_len, 1] optional manual intervention labels
            hidden_obs: Optional observational GRU hidden state
            hidden_int: Optional interventional GRU hidden state

        Returns:
            next_states: [batch_size, seq_len, state_dim]
            hidden_states: Tuple of (hidden_obs, hidden_int)
            pathway_info: Dict with pathway usage info
        """
        batch_size, seq_len = state_sequence.shape[:2]

        # Concatenate inputs
        x = torch.cat([state_sequence, action_sequence, causal_sequence], dim=-1)

        # Detect interventions if not provided
        if intervention_mask is None:
            intervention_score = self.detect_intervention(causal_sequence)
        else:
            intervention_score = intervention_mask

        # Observational pathway
        obs_out, hidden_obs = self.observational_gru(x, hidden_obs)

        # Interventional pathway
        int_out, hidden_int = self.interventional_gru(x, hidden_int)

        # Compute pathway specialization metrics
        specialization_loss = self.compute_mmd_specialization_loss(obs_out, int_out)
        specialization_score = self.compute_specialization_score(obs_out, int_out)

        # Adaptive pathway weighting based on intervention detection
        obs_weight = (1 - intervention_score) * self.pathway_weights[0]
        int_weight = intervention_score * self.pathway_weights[1]

        # Normalize weights
        total_weight = obs_weight + int_weight + 1e-8
        obs_weight = obs_weight / total_weight
        int_weight = int_weight / total_weight

        # Combine pathways
        combined_out = obs_weight * obs_out + int_weight * int_out

        # Output projection
        state_deltas = self.output_layers(combined_out)

        # Residual connection
        next_states = self.residual_weight * state_sequence + state_deltas

        # Pathway usage info
        pathway_info = {
            'intervention_score': intervention_score.mean().item(),
            'obs_weight': obs_weight.mean().item(),
            'int_weight': int_weight.mean().item(),
            'pathway_balance': abs(obs_weight.mean() - int_weight.mean()).item(),
            'specialization_loss': specialization_loss.item(),
            'specialization_score': specialization_score.item()
        }

        return next_states, (hidden_obs, hidden_int), pathway_info

    def predict_single(self, state, action, causal_factors,
                      intervention_mask=None, hidden_states=None):
        """
        Single step prediction (backward compatibility with GRUDynamics)

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            causal_factors: [batch_size, causal_dim]
            intervention_mask: [batch_size, 1] optional
            hidden_states: Optional tuple of (hidden_obs, hidden_int)

        Returns:
            next_state: [batch_size, state_dim]
            hidden_states: Tuple of (hidden_obs, hidden_int)
        """
        # Add sequence dimension
        state_seq = state.unsqueeze(1)
        action_seq = action.unsqueeze(1)
        causal_seq = causal_factors.unsqueeze(1)

        if intervention_mask is not None:
            intervention_mask = intervention_mask.unsqueeze(1)

        # Extract hidden states
        hidden_obs, hidden_int = hidden_states if hidden_states else (None, None)

        # Forward pass
        next_states, hidden_states, pathway_info = self.forward(
            state_seq, action_seq, causal_seq, intervention_mask, hidden_obs, hidden_int
        )

        # Remove sequence dimension
        next_state = next_states.squeeze(1)

        return next_state, hidden_states

    def observational_only(self, state_sequence, action_sequence, causal_sequence,
                          hidden=None):
        """
        Use only observational pathway (for baseline comparison)
        """
        x = torch.cat([state_sequence, action_sequence, causal_sequence], dim=-1)
        obs_out, hidden = self.observational_gru(x, hidden)
        state_deltas = self.output_layers(obs_out)
        next_states = self.residual_weight * state_sequence + state_deltas
        return next_states, hidden

    def interventional_only(self, state_sequence, action_sequence, causal_sequence,
                           hidden=None):
        """
        Use only interventional pathway (for counterfactual generation)
        """
        x = torch.cat([state_sequence, action_sequence, causal_sequence], dim=-1)
        int_out, hidden = self.interventional_gru(x, hidden)
        state_deltas = self.output_layers(int_out)
        next_states = self.residual_weight * state_sequence + state_deltas
        return next_states, hidden

    def set_pathway_mode(self, mode='auto'):
        """
        Set pathway operation mode

        Args:
            mode: 'auto' (adaptive), 'observational', 'interventional', 'balanced'
        """
        if mode == 'observational':
            self.pathway_weights.data = torch.tensor([1.0, 0.0])
        elif mode == 'interventional':
            self.pathway_weights.data = torch.tensor([0.0, 1.0])
        elif mode == 'balanced':
            self.pathway_weights.data = torch.tensor([0.5, 0.5])
        elif mode == 'auto':
            self.pathway_weights.data = torch.tensor([0.7, 0.3])  # Default adaptive
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_pathway_analysis(self, data_loader, num_batches=10):
        """
        Analyze pathway usage across dataset

        Returns:
            analysis: Dict with pathway statistics
        """
        self.eval()
        intervention_scores = []
        obs_weights = []
        int_weights = []

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break

                states, actions, causal_factors = batch
                _, _, pathway_info = self.forward(states, actions, causal_factors)

                intervention_scores.append(pathway_info['intervention_score'])
                obs_weights.append(pathway_info['obs_weight'])
                int_weights.append(pathway_info['int_weight'])

        return {
            'avg_intervention_score': np.mean(intervention_scores),
            'avg_obs_weight': np.mean(obs_weights),
            'avg_int_weight': np.mean(int_weights),
            'pathway_balance': np.mean([abs(o - i) for o, i in zip(obs_weights, int_weights)]),
            'intervention_detection_variance': np.var(intervention_scores)
        }

    def get_model_name(self):
        return "dual_pathway_causal_gru"

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_info(self):
        """Get detailed architecture information"""
        return {
            'model_name': self.get_model_name(),
            'total_parameters': self.count_parameters(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'causal_dim': self.causal_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'observational_params': sum(p.numel() for p in self.observational_gru.parameters()),
            'interventional_params': sum(p.numel() for p in self.interventional_gru.parameters()),
            'shared_params': sum(p.numel() for p in self.output_layers.parameters()),
            'pathway_weights': self.pathway_weights.data.tolist()
        }


class CausalLoss(nn.Module):
    """
    Specialized loss function for dual-pathway causal training
    """

    def __init__(self, mse_weight=1.0, pathway_balance_weight=0.1,
                 intervention_detection_weight=0.05):
        super().__init__()
        self.mse_weight = mse_weight
        self.pathway_balance_weight = pathway_balance_weight
        self.intervention_detection_weight = intervention_detection_weight

    def forward(self, predicted_states, target_states, pathway_info,
                true_intervention_mask=None):
        """
        Compute causal loss with pathway regularization

        Args:
            predicted_states: [batch_size, seq_len, state_dim]
            target_states: [batch_size, seq_len, state_dim]
            pathway_info: Dict with pathway usage information
            true_intervention_mask: [batch_size, seq_len, 1] ground truth interventions
        """
        # Standard MSE loss
        mse_loss = F.mse_loss(predicted_states, target_states)

        total_loss = self.mse_weight * mse_loss

        # Pathway balance regularization (prevent one pathway dominance)
        if self.pathway_balance_weight > 0:
            balance_penalty = (pathway_info['obs_weight'] - 0.6) ** 2  # Target 60% obs
            total_loss += self.pathway_balance_weight * balance_penalty

        # Intervention detection loss (if ground truth available)
        if true_intervention_mask is not None and self.intervention_detection_weight > 0:
            detection_loss = F.binary_cross_entropy(
                torch.tensor(pathway_info['intervention_score']).unsqueeze(0),
                true_intervention_mask.float().mean()
            )
            total_loss += self.intervention_detection_weight * detection_loss

        return total_loss, {
            'mse_loss': mse_loss.item(),
            'pathway_balance_penalty': balance_penalty if self.pathway_balance_weight > 0 else 0,
            'intervention_detection_loss': detection_loss.item() if true_intervention_mask is not None else 0,
            'total_loss': total_loss.item()
        }


def create_dual_pathway_model(config=None):
    """
    Factory function for creating dual-pathway causal GRU

    Args:
        config: Dict with model configuration or None for defaults

    Returns:
        model: DualPathwayCausalGRU instance
    """
    if config is None:
        config = {
            'state_dim': 12,
            'action_dim': 2,
            'causal_dim': 5,
            'hidden_dim': 64,
            'num_layers': 1
        }

    return DualPathwayCausalGRU(**config)


def test_dual_pathway_compatibility():
    """
    Test backward compatibility with existing GRUDynamics interface
    """
    print("Testing DualPathwayCausalGRU compatibility...")

    # Create model
    model = DualPathwayCausalGRU(state_dim=12, action_dim=2, causal_dim=5)

    # Test single prediction (GRUDynamics compatibility)
    batch_size = 10
    state = torch.randn(batch_size, 12)
    action = torch.randn(batch_size, 2)
    causal = torch.randn(batch_size, 5)

    # Test single prediction
    next_state, hidden = model.predict_single(state, action, causal)
    assert next_state.shape == (batch_size, 12), f"Expected {(batch_size, 12)}, got {next_state.shape}"

    # Test sequence prediction
    seq_len = 8
    state_seq = torch.randn(batch_size, seq_len, 12)
    action_seq = torch.randn(batch_size, seq_len, 2)
    causal_seq = torch.randn(batch_size, seq_len, 5)

    next_states, hidden_states, pathway_info = model(state_seq, action_seq, causal_seq)
    assert next_states.shape == (batch_size, seq_len, 12)

    print("✅ Backward compatibility test passed")
    print(f"✅ Model parameters: {model.count_parameters()}")
    print(f"✅ Pathway balance: {pathway_info['pathway_balance']:.3f}")

    return True


if __name__ == "__main__":
    test_dual_pathway_compatibility()