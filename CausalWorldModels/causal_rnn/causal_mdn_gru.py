"""
Causal MDN-GRU Implementation
Extends World Models with causal state conditioning: p(z_t+1 | z_t, action, causal_state)

This is the core innovation for Phase 2A - instead of learning just p(z_t+1 | z_t, action),
we learn p(z_t+1 | z_t, action, causal_state) to capture how causal factors affect dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


class CausalMDNGRU(nn.Module):
    """
    Causal Mixture Density Network + GRU
    
    Architecture:
    1. Input: [z_t, action_t, causal_state_t] 
    2. GRU processes the concatenated input
    3. MDN head outputs mixture parameters for z_t+1
    4. Causal factors influence both the dynamics and the mixture components
    """
    
    def __init__(self, 
                 z_dim: int = 256,
                 action_dim: int = 5, 
                 causal_dim: int = 45,
                 hidden_dim: int = 512,
                 num_mixtures: int = 5,
                 layers: int = 1):
        super().__init__()
        
        self.z_dim = z_dim
        self.action_dim = action_dim  
        self.causal_dim = causal_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        self.layers = layers
        
        # Input is [z_t, action_t, causal_state_t]
        input_dim = z_dim + action_dim + causal_dim
        
        # GRU processes the temporal sequence
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim, 
            num_layers=layers,
            batch_first=True,
            dropout=0.1 if layers > 1 else 0
        )
        
        # Causal feature extractor - processes causal state for better dynamics
        self.causal_processor = nn.Sequential(
            nn.Linear(causal_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # MDN head outputs mixture parameters
        # Each mixture component needs: weight, mean, std for each z dimension
        mdn_input_dim = hidden_dim + 32  # GRU output + processed causal features
        
        self.mixture_weights = nn.Linear(mdn_input_dim, num_mixtures)
        self.mixture_means = nn.Linear(mdn_input_dim, num_mixtures * z_dim)
        self.mixture_stds = nn.Linear(mdn_input_dim, num_mixtures * z_dim)
        
        # Initialize mixture parameters properly
        self._init_mdn_parameters()
        
    def _init_mdn_parameters(self):
        """Initialize MDN parameters for stable training"""
        # Initialize mixture weights to be uniform
        nn.init.constant_(self.mixture_weights.bias, 0)
        
        # Initialize mixture means around zero with small variance
        nn.init.normal_(self.mixture_means.weight, 0, 0.1)
        nn.init.constant_(self.mixture_means.bias, 0)
        
        # Initialize mixture stds to reasonable values (around 1.0)
        nn.init.normal_(self.mixture_stds.weight, 0, 0.1)
        nn.init.constant_(self.mixture_stds.bias, 0.5)  # Will be passed through softplus
        
    def forward(self, z_sequence: torch.Tensor, 
                action_sequence: torch.Tensor,
                causal_sequence: torch.Tensor,
                hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through causal MDN-GRU
        
        Args:
            z_sequence: (batch_size, seq_len, z_dim) - latent states
            action_sequence: (batch_size, seq_len, action_dim) - actions (one-hot)
            causal_sequence: (batch_size, seq_len, causal_dim) - causal states
            hidden_state: Optional initial hidden state
            
        Returns:
            mixture_weights: (batch_size, seq_len, num_mixtures) - π_k
            mixture_means: (batch_size, seq_len, num_mixtures, z_dim) - μ_k
            mixture_stds: (batch_size, seq_len, num_mixtures, z_dim) - σ_k  
            hidden_state: Final hidden state
        """
        batch_size, seq_len = z_sequence.shape[:2]
        
        # Process causal features
        causal_features = self.causal_processor(causal_sequence)  # (batch, seq, 32)
        
        # Concatenate inputs: [z_t, action_t, causal_state_t]
        gru_input = torch.cat([z_sequence, action_sequence, causal_sequence], dim=-1)
        
        # GRU processes the sequence
        gru_output, hidden_state = self.gru(gru_input, hidden_state)  # (batch, seq, hidden_dim)
        
        # Combine GRU output with processed causal features
        mdn_input = torch.cat([gru_output, causal_features], dim=-1)  # (batch, seq, hidden_dim + 32)
        
        # Generate mixture parameters
        weights_logits = self.mixture_weights(mdn_input)  # (batch, seq, num_mixtures)
        mixture_weights = F.softmax(weights_logits, dim=-1)
        
        means_flat = self.mixture_means(mdn_input)  # (batch, seq, num_mixtures * z_dim)
        mixture_means = means_flat.view(batch_size, seq_len, self.num_mixtures, self.z_dim)
        
        stds_flat = self.mixture_stds(mdn_input)  # (batch, seq, num_mixtures * z_dim)
        mixture_stds = F.softplus(stds_flat).view(batch_size, seq_len, self.num_mixtures, self.z_dim)
        mixture_stds = torch.clamp(mixture_stds, min=1e-6, max=10.0)  # Numerical stability
        
        return mixture_weights, mixture_means, mixture_stds, hidden_state
    
    def sample(self, z_t: torch.Tensor, 
               action_t: torch.Tensor,
               causal_state_t: torch.Tensor, 
               hidden_state: Optional[torch.Tensor] = None,
               temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample z_t+1 from the learned causal dynamics
        
        Args:
            z_t: (batch_size, z_dim) - current latent state
            action_t: (batch_size, action_dim) - current action
            causal_state_t: (batch_size, causal_dim) - current causal state
            hidden_state: Optional hidden state
            temperature: Sampling temperature
            
        Returns:
            z_next: (batch_size, z_dim) - sampled next latent state
            hidden_state: Updated hidden state
        """
        # Add sequence dimension for single step
        z_seq = z_t.unsqueeze(1)  # (batch, 1, z_dim)
        action_seq = action_t.unsqueeze(1)  # (batch, 1, action_dim) 
        causal_seq = causal_state_t.unsqueeze(1)  # (batch, 1, causal_dim)
        
        # Forward pass
        weights, means, stds, hidden_state = self.forward(z_seq, action_seq, causal_seq, hidden_state)
        
        # Remove sequence dimension
        weights = weights.squeeze(1)  # (batch, num_mixtures)
        means = means.squeeze(1)  # (batch, num_mixtures, z_dim)
        stds = stds.squeeze(1) * temperature  # (batch, num_mixtures, z_dim)
        
        # Sample mixture component
        mixture_idx = torch.multinomial(weights, 1).squeeze(-1)  # (batch,)
        batch_idx = torch.arange(weights.shape[0], device=weights.device)
        
        # Sample from selected Gaussian component
        selected_means = means[batch_idx, mixture_idx]  # (batch, z_dim)
        selected_stds = stds[batch_idx, mixture_idx]  # (batch, z_dim)
        
        eps = torch.randn_like(selected_means)
        z_next = selected_means + selected_stds * eps
        
        return z_next, hidden_state
    
    def compute_loss(self, z_target: torch.Tensor,
                     mixture_weights: torch.Tensor, 
                     mixture_means: torch.Tensor,
                     mixture_stds: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for MDN
        
        Args:
            z_target: (batch_size, seq_len, z_dim) - target latent states z_t+1
            mixture_weights: (batch_size, seq_len, num_mixtures) - π_k
            mixture_means: (batch_size, seq_len, num_mixtures, z_dim) - μ_k
            mixture_stds: (batch_size, seq_len, num_mixtures, z_dim) - σ_k
            
        Returns:
            loss: Negative log-likelihood loss
        """
        batch_size, seq_len, z_dim = z_target.shape
        
        # Expand target to match mixture dimensions
        z_target_expanded = z_target.unsqueeze(2)  # (batch, seq, 1, z_dim)
        z_target_expanded = z_target_expanded.expand(-1, -1, self.num_mixtures, -1)  # (batch, seq, mixtures, z_dim)
        
        # Compute Gaussian log-probabilities for each component
        # log p(z | μ_k, σ_k) = -0.5 * log(2π) - log(σ_k) - 0.5 * ((z - μ_k) / σ_k)^2
        log_2pi = np.log(2 * np.pi)
        
        squared_diff = ((z_target_expanded - mixture_means) / mixture_stds) ** 2
        log_probs = -0.5 * log_2pi - torch.log(mixture_stds) - 0.5 * squared_diff
        log_probs = torch.sum(log_probs, dim=-1)  # Sum over z_dim: (batch, seq, mixtures)
        
        # Weight by mixture probabilities and compute log-sum-exp
        weighted_log_probs = log_probs + torch.log(mixture_weights + 1e-8)
        loss_per_step = -torch.logsumexp(weighted_log_probs, dim=-1)  # (batch, seq)
        
        return torch.mean(loss_per_step)
    
    def get_causal_attention_weights(self, causal_sequence: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights showing which causal factors are most important
        Useful for interpretability and analysis
        
        Args:
            causal_sequence: (batch_size, seq_len, causal_dim)
            
        Returns:
            attention_weights: (batch_size, seq_len, causal_dim) - normalized importance
        """
        causal_features = self.causal_processor[0](causal_sequence)  # First linear layer
        attention_logits = torch.sum(torch.abs(causal_features), dim=-1, keepdim=True)  # (batch, seq, 1)
        
        # Convert to attention over original causal dimensions
        causal_importance = torch.abs(causal_sequence) * attention_logits
        attention_weights = F.softmax(causal_importance, dim=-1)
        
        return attention_weights


class CausalTransitionPredictor(nn.Module):
    """
    High-level interface for training and using the causal dynamics model
    Handles data preprocessing, loss computation, and evaluation metrics
    """
    
    def __init__(self, 
                 z_dim: int = 256,
                 action_dim: int = 5,
                 causal_dim: int = 45,
                 **kwargs):
        super().__init__()
        
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.causal_dim = causal_dim
        
        self.causal_mdn_gru = CausalMDNGRU(
            z_dim=z_dim,
            action_dim=action_dim, 
            causal_dim=causal_dim,
            **kwargs
        )
        
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Training forward pass
        
        Args:
            batch_data: Dictionary with keys:
                - 'z_sequence': (batch, seq_len, z_dim) - latent states
                - 'action_sequence': (batch, seq_len, action_dim) - actions  
                - 'causal_sequence': (batch, seq_len, causal_dim) - causal states
                
        Returns:
            Dictionary with prediction outputs and loss
        """
        z_seq = batch_data['z_sequence']
        action_seq = batch_data['action_sequence'] 
        causal_seq = batch_data['causal_sequence']
        
        # Use z_t and action_t to predict z_t+1
        z_input = z_seq[:, :-1]  # (batch, seq_len-1, z_dim)
        action_input = action_seq[:, :-1]  # (batch, seq_len-1, action_dim)
        causal_input = causal_seq[:, :-1]  # (batch, seq_len-1, causal_dim)
        z_target = z_seq[:, 1:]  # (batch, seq_len-1, z_dim)
        
        # Forward through causal MDN-GRU
        mixture_weights, mixture_means, mixture_stds, _ = self.causal_mdn_gru(
            z_input, action_input, causal_input
        )
        
        # Compute loss
        loss = self.causal_mdn_gru.compute_loss(z_target, mixture_weights, mixture_means, mixture_stds)
        
        # Compute evaluation metrics
        with torch.no_grad():
            # Reconstruction accuracy - sample from model and compute MSE
            batch_size, seq_len = z_input.shape[:2]
            z_pred_samples = []
            
            for t in range(seq_len):
                z_t = z_input[:, t]  # (batch, z_dim)
                action_t = action_input[:, t]  # (batch, action_dim)
                causal_t = causal_input[:, t]  # (batch, causal_dim)
                
                # Sample from mixture
                weights_t = mixture_weights[:, t]  # (batch, num_mixtures)
                means_t = mixture_means[:, t]  # (batch, num_mixtures, z_dim)
                stds_t = mixture_stds[:, t]  # (batch, num_mixtures, z_dim)
                
                # Take mode (highest weight component) for evaluation
                best_mixture = torch.argmax(weights_t, dim=-1)  # (batch,)
                batch_idx = torch.arange(batch_size, device=z_t.device)
                z_pred = means_t[batch_idx, best_mixture]  # (batch, z_dim)
                z_pred_samples.append(z_pred)
            
            z_pred_sequence = torch.stack(z_pred_samples, dim=1)  # (batch, seq_len, z_dim)
            mse_loss = F.mse_loss(z_pred_sequence, z_target)
            
        return {
            'mixture_weights': mixture_weights,
            'mixture_means': mixture_means, 
            'mixture_stds': mixture_stds,
            'z_predictions': z_pred_sequence,
            'loss': loss,
            'mse_loss': mse_loss,
            'z_target': z_target
        }
    
    def generate_sequence(self, 
                         z_initial: torch.Tensor,
                         action_sequence: torch.Tensor, 
                         causal_sequence: torch.Tensor,
                         temperature: float = 1.0) -> torch.Tensor:
        """
        Generate a sequence of latent states using the learned causal dynamics
        
        Args:
            z_initial: (batch_size, z_dim) - initial latent state
            action_sequence: (batch_size, seq_len, action_dim) - planned actions
            causal_sequence: (batch_size, seq_len, causal_dim) - causal states over time
            temperature: Sampling temperature
            
        Returns:
            z_sequence: (batch_size, seq_len+1, z_dim) - generated latent sequence
        """
        batch_size, seq_len = action_sequence.shape[:2]
        device = z_initial.device
        
        z_sequence = [z_initial]
        hidden_state = None
        z_current = z_initial
        
        for t in range(seq_len):
            action_t = action_sequence[:, t]
            causal_t = causal_sequence[:, t]
            
            z_next, hidden_state = self.causal_mdn_gru.sample(
                z_current, action_t, causal_t, hidden_state, temperature
            )
            
            z_sequence.append(z_next)
            z_current = z_next
        
        return torch.stack(z_sequence, dim=1)  # (batch, seq_len+1, z_dim)


def create_causal_rnn(architecture: str = "causal_mdn_gru", **kwargs) -> nn.Module:
    """
    Factory function to create causal RNN architectures
    
    Args:
        architecture: Type of causal RNN to create
        **kwargs: Architecture-specific parameters
        
    Returns:
        Causal RNN model
    """
    architectures = {
        'causal_mdn_gru': CausalTransitionPredictor,
        'causal_predictor': CausalTransitionPredictor
    }
    
    if architecture not in architectures:
        raise ValueError(f"Unknown causal RNN architecture: {architecture}. Available: {list(architectures.keys())}")
    
    return architectures[architecture](**kwargs)


if __name__ == "__main__":
    # Test the causal MDN-GRU implementation
    print("🧪 Testing Causal MDN-GRU Implementation")
    print("=" * 50)
    
    # Create model
    model = create_causal_rnn("causal_mdn_gru", 
                             z_dim=256, 
                             action_dim=5, 
                             causal_dim=45,
                             hidden_dim=512,
                             num_mixtures=5)
    
    print(f"✅ Model created: {model.__class__.__name__}")
    
    # Test with dummy data
    batch_size, seq_len = 4, 10
    z_dim, action_dim, causal_dim = 256, 5, 45
    
    dummy_data = {
        'z_sequence': torch.randn(batch_size, seq_len, z_dim),
        'action_sequence': F.one_hot(torch.randint(0, action_dim, (batch_size, seq_len)), action_dim).float(),
        'causal_sequence': torch.randn(batch_size, seq_len, causal_dim)  # Should be one-hot in practice
    }
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_data)
        
        print(f"✅ Forward pass successful")
        print(f"   Loss: {output['loss'].item():.4f}")
        print(f"   MSE Loss: {output['mse_loss'].item():.4f}")
        print(f"   Predictions shape: {output['z_predictions'].shape}")
        print(f"   Mixture weights shape: {output['mixture_weights'].shape}")
        
        # Test sequence generation
        z_initial = torch.randn(batch_size, z_dim)
        action_seq = dummy_data['action_sequence']
        causal_seq = dummy_data['causal_sequence']
        
        generated_seq = model.generate_sequence(z_initial, action_seq, causal_seq)
        print(f"✅ Sequence generation successful")
        print(f"   Generated sequence shape: {generated_seq.shape}")
    
    print(f"\n🎉 Causal MDN-GRU implementation complete and tested!")
    print(f"📊 Ready for Phase 2A experiments")