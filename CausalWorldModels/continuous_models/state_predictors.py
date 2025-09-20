"""
Continuous State Prediction Models
Models for predicting next state from current state, action, and causal factors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math


class LinearDynamics(nn.Module):
    """Simple linear state transition model with causal conditioning"""

    def __init__(self, state_dim=12, action_dim=2, causal_dim=5, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.causal_dim = causal_dim

        # Input: [state, action, causal] -> next_state
        input_dim = state_dim + action_dim + causal_dim

        self.dynamics = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.9))

    def forward(self, state, action, causal_factors):
        """
        Predict next state given current state, action, and causal factors

        Args:
            state: [batch_size, state_dim] current state
            action: [batch_size, action_dim] action taken
            causal_factors: [batch_size, causal_dim] causal context

        Returns:
            next_state: [batch_size, state_dim] predicted next state
        """
        # Concatenate inputs
        x = torch.cat([state, action, causal_factors], dim=-1)

        # Predict state change
        state_delta = self.dynamics(x)

        # Apply residual connection (most state remains unchanged)
        next_state = self.residual_weight * state + state_delta

        return next_state

    def get_model_name(self):
        return "linear_dynamics"


class LSTMPredictor(nn.Module):
    """LSTM-based sequence model for temporal state prediction"""

    def __init__(self, state_dim=12, action_dim=2, causal_dim=5, hidden_dim=128, num_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.causal_dim = causal_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input: [state, action, causal]
        input_dim = state_dim + action_dim + causal_dim

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, state_dim)
        )

        # Residual connection
        self.residual_weight = nn.Parameter(torch.tensor(0.85))

    def forward(self, state_sequence, action_sequence, causal_sequence, hidden=None):
        """
        Predict next states for a sequence

        Args:
            state_sequence: [batch_size, seq_len, state_dim]
            action_sequence: [batch_size, seq_len, action_dim]
            causal_sequence: [batch_size, seq_len, causal_dim]
            hidden: Optional LSTM hidden state

        Returns:
            next_states: [batch_size, seq_len, state_dim]
            hidden: LSTM hidden state
        """
        batch_size, seq_len = state_sequence.shape[:2]

        # Concatenate inputs
        x = torch.cat([state_sequence, action_sequence, causal_sequence], dim=-1)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)

        # Project to state space
        state_deltas = self.output_proj(lstm_out)

        # Apply residual connections
        next_states = self.residual_weight * state_sequence + state_deltas

        return next_states, hidden

    def predict_single(self, state, action, causal_factors, hidden=None):
        """Predict single next state (for online use)"""
        # Add sequence dimension
        state_seq = state.unsqueeze(1)
        action_seq = action.unsqueeze(1)
        causal_seq = causal_factors.unsqueeze(1)

        next_states, hidden = self.forward(state_seq, action_seq, causal_seq, hidden)

        # Remove sequence dimension
        next_state = next_states.squeeze(1)

        return next_state, hidden

    def get_model_name(self):
        return "lstm_predictor"


class GRUDynamics(nn.Module):
    """Lightweight GRU-based dynamics model"""

    def __init__(self, state_dim=12, action_dim=2, causal_dim=5, hidden_dim=96, num_layers=1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.causal_dim = causal_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        input_dim = state_dim + action_dim + causal_dim

        # GRU for dynamics
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, state_dim)
        )

        self.residual_weight = nn.Parameter(torch.tensor(0.9))

    def forward(self, state_sequence, action_sequence, causal_sequence, hidden=None):
        """Forward pass for sequence prediction"""
        # Concatenate inputs
        x = torch.cat([state_sequence, action_sequence, causal_sequence], dim=-1)

        # GRU forward
        gru_out, hidden = self.gru(x, hidden)

        # Output projection
        state_deltas = self.output_layers(gru_out)

        # Residual connection
        next_states = self.residual_weight * state_sequence + state_deltas

        return next_states, hidden

    def predict_single(self, state, action, causal_factors, hidden=None):
        """Single step prediction"""
        state_seq = state.unsqueeze(1)
        action_seq = action.unsqueeze(1)
        causal_seq = causal_factors.unsqueeze(1)

        next_states, hidden = self.forward(state_seq, action_seq, causal_seq, hidden)
        next_state = next_states.squeeze(1)

        return next_state, hidden

    def get_model_name(self):
        return "gru_dynamics"


class NeuralODE(nn.Module):
    """Neural ODE for continuous-time dynamics"""

    def __init__(self, state_dim=12, action_dim=2, causal_dim=5, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.causal_dim = causal_dim

        # ODE function f(t, y) where y = [state, action, causal]
        input_dim = state_dim + action_dim + causal_dim

        self.ode_func = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Time step for integration
        self.dt = nn.Parameter(torch.tensor(0.1))

    def forward(self, state, action, causal_factors, num_steps=1):
        """
        Integrate ODE for num_steps to predict future state

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            causal_factors: [batch_size, causal_dim]
            num_steps: Number of integration steps

        Returns:
            next_state: [batch_size, state_dim]
        """
        current_state = state

        for step in range(num_steps):
            # Concatenate current state with action and causal factors
            x = torch.cat([current_state, action, causal_factors], dim=-1)

            # Compute derivative
            state_derivative = self.ode_func(x)

            # Euler integration step
            current_state = current_state + self.dt * state_derivative

        return current_state

    def get_model_name(self):
        return "neural_ode"


class VAERNNHybrid(nn.Module):
    """Hybrid model: VAE for state compression + RNN for dynamics"""

    def __init__(self, state_dim=12, action_dim=2, causal_dim=5, latent_dim=8, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.causal_dim = causal_dim
        self.latent_dim = latent_dim

        # VAE encoder for state compression
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # VAE latent projections
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # VAE decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # RNN for latent dynamics
        latent_input_dim = latent_dim + action_dim + causal_dim
        self.rnn = nn.GRU(
            input_size=latent_input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def encode(self, state):
        """Encode state to latent representation"""
        h = self.encoder(state)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent to state"""
        return self.decoder(z)

    def forward(self, state_sequence, action_sequence, causal_sequence):
        """
        Forward pass: encode states, predict latent dynamics, decode

        Args:
            state_sequence: [batch_size, seq_len, state_dim]
            action_sequence: [batch_size, seq_len, action_dim]
            causal_sequence: [batch_size, seq_len, causal_dim]

        Returns:
            reconstructed_states: [batch_size, seq_len, state_dim]
            mu: [batch_size, seq_len, latent_dim]
            logvar: [batch_size, seq_len, latent_dim]
        """
        batch_size, seq_len = state_sequence.shape[:2]

        # Encode all states
        mu_list, logvar_list, z_list = [], [], []

        for t in range(seq_len):
            mu, logvar = self.encode(state_sequence[:, t])
            z = self.reparameterize(mu, logvar)

            mu_list.append(mu)
            logvar_list.append(logvar)
            z_list.append(z)

        # Stack latent sequences
        mu_seq = torch.stack(mu_list, dim=1)
        logvar_seq = torch.stack(logvar_list, dim=1)
        z_seq = torch.stack(z_list, dim=1)

        # RNN dynamics in latent space
        latent_input = torch.cat([z_seq, action_sequence, causal_sequence], dim=-1)
        rnn_out, _ = self.rnn(latent_input)

        # Project back to latent space
        next_z_seq = self.latent_proj(rnn_out)

        # Decode to state space
        reconstructed_states = []
        for t in range(seq_len):
            recon_state = self.decode(next_z_seq[:, t])
            reconstructed_states.append(recon_state)

        reconstructed_states = torch.stack(reconstructed_states, dim=1)

        return reconstructed_states, mu_seq, logvar_seq

    def predict_single(self, state, action, causal_factors):
        """Single step prediction"""
        # Encode current state
        mu, logvar = self.encode(state)
        z = self.reparameterize(mu, logvar)

        # Predict next latent state
        latent_input = torch.cat([z, action, causal_factors], dim=-1).unsqueeze(1)
        rnn_out, _ = self.rnn(latent_input)
        next_z = self.latent_proj(rnn_out.squeeze(1))

        # Decode to state
        next_state = self.decode(next_z)

        return next_state, mu, logvar

    def get_model_name(self):
        return "vae_rnn_hybrid"


def create_continuous_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function to create continuous state prediction models"""

    models = {
        'linear_dynamics': LinearDynamics,
        'lstm_predictor': LSTMPredictor,
        'gru_dynamics': GRUDynamics,
        'neural_ode': NeuralODE,
        'vae_rnn_hybrid': VAERNNHybrid
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> Dict:
    """Get model architecture information"""
    return {
        'name': model.get_model_name(),
        'parameters': count_parameters(model),
        'architecture': str(model),
        'device': str(next(model.parameters()).device),
        'dtype': str(next(model.parameters()).dtype)
    }


# Loss functions for different model types
class ContinuousStateLoss(nn.Module):
    """Loss function for continuous state prediction"""

    def __init__(self, mse_weight=1.0, consistency_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.consistency_weight = consistency_weight

    def forward(self, predicted_states, target_states, causal_factors=None):
        """
        Compute loss for state prediction

        Args:
            predicted_states: [batch_size, seq_len, state_dim]
            target_states: [batch_size, seq_len, state_dim]
            causal_factors: [batch_size, seq_len, causal_dim] for consistency loss
        """
        # MSE loss for state prediction
        mse_loss = F.mse_loss(predicted_states, target_states)

        total_loss = self.mse_weight * mse_loss

        # Optional causal consistency loss (ensure causal factors remain consistent)
        if causal_factors is not None and self.consistency_weight > 0:
            # Causal factors should be preserved in state transitions
            pred_causal = predicted_states[..., 7:12]  # Extract causal portion
            target_causal = target_states[..., 7:12]

            causal_loss = F.mse_loss(pred_causal, target_causal)
            total_loss += self.consistency_weight * causal_loss

        return total_loss, mse_loss


class VAELoss(nn.Module):
    """Loss function for VAE-RNN hybrid model"""

    def __init__(self, recon_weight=1.0, kl_weight=0.1):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

    def forward(self, reconstructed_states, target_states, mu, logvar):
        """VAE loss with reconstruction and KL divergence"""
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed_states, target_states)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (mu.shape[0] * mu.shape[1])  # Normalize by batch and sequence

        # Total loss
        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss

        return total_loss, recon_loss, kl_loss