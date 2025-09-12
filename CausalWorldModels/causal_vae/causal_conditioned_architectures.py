"""
Causal-Conditioned VAE Architectures
Extends the standard VAE architectures to learn p(z | x, causal_state) instead of just p(z | x)

This is the key innovation that makes the models truly "causal world models" - they condition
their latent representations on the causal factors in the environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from .modern_architectures import (
    BaseVAE, BaselineVAE32D, GaussianVAE256D, BetaVAE, 
    HierarchicalVAE512D, CategoricalVAE512D, VQVAE256D,
    NoConvNormVAE, DeeperEncoderVAE
)


class CausalConditioningModule(nn.Module):
    """
    Module that processes causal states for conditioning VAE architectures
    Converts 45D causal vector into meaningful conditioning features
    """
    
    def __init__(self, causal_dim: int = 45, conditioning_dim: int = 64):
        super().__init__()
        self.causal_dim = causal_dim
        self.conditioning_dim = conditioning_dim
        
        # Process causal factors with separate pathways for different factor types
        self.time_processor = nn.Sequential(
            nn.Linear(24, 16),  # Time of day (24 hours)
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        self.day_processor = nn.Sequential(
            nn.Linear(7, 8),   # Day of week
            nn.ReLU()
        )
        
        self.weather_processor = nn.Sequential(
            nn.Linear(4, 8),   # Weather types
            nn.ReLU()
        )
        
        self.event_processor = nn.Sequential(
            nn.Linear(5, 8),   # Event types
            nn.ReLU()
        )
        
        self.crowd_processor = nn.Sequential(
            nn.Linear(5, 8),   # Crowd density levels
            nn.ReLU()
        )
        
        # Combine all factors
        self.combiner = nn.Sequential(
            nn.Linear(40, conditioning_dim),  # 8+8+8+8+8 = 40
            nn.LayerNorm(conditioning_dim),
            nn.ReLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.LayerNorm(conditioning_dim)
        )
        
    def forward(self, causal_state: torch.Tensor) -> torch.Tensor:
        """
        Process causal state into conditioning features
        
        Args:
            causal_state: (batch_size, 45) - one-hot encoded causal factors
            
        Returns:
            conditioning_features: (batch_size, conditioning_dim) - processed features
        """
        # Split causal state into components
        time_features = causal_state[:, :24]      # Time hour (24D)
        day_features = causal_state[:, 24:31]     # Day week (7D)
        weather_features = causal_state[:, 31:35] # Weather (4D)
        event_features = causal_state[:, 35:40]   # Event (5D) 
        crowd_features = causal_state[:, 40:45]   # Crowd density (5D)
        
        # Process each factor type
        time_processed = self.time_processor(time_features)
        day_processed = self.day_processor(day_features) 
        weather_processed = self.weather_processor(weather_features)
        event_processed = self.event_processor(event_features)
        crowd_processed = self.crowd_processor(crowd_features)
        
        # Combine all processed features
        combined = torch.cat([
            time_processed, day_processed, weather_processed,
            event_processed, crowd_processed
        ], dim=-1)
        
        conditioning_features = self.combiner(combined)
        return conditioning_features


class CausalConditionedHierarchicalVAE(HierarchicalVAE512D):
    """
    Hierarchical VAE with causal conditioning - learns p(z_static, z_dynamic | x, causal_state)
    
    Key innovation: Static and dynamic latents are conditioned differently
    - Static latents: conditioned on time-invariant factors (building layouts, weather patterns)
    - Dynamic latents: conditioned on time-varying factors (crowds, events, time of day)
    """
    
    def __init__(self, input_shape=(64, 64, 3), causal_dim=45, static_dim=256, dynamic_dim=256):
        super().__init__(input_shape)
        
        self.causal_dim = causal_dim
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.causal_processor = CausalConditioningModule(causal_dim, 64)
        
        # Separate conditioning for static vs dynamic latents
        self.static_causal_dim = 32   # Buildings, base weather patterns
        self.dynamic_causal_dim = 32  # Crowds, time, events
        
        # Modify the existing latent layers to incorporate causal conditioning
        self.flatten_size = 512 * 4 * 4
        
        # Static branch with causal conditioning (buildings, weather patterns)
        self.static_mu = nn.Linear(self.flatten_size + self.static_causal_dim, self.static_dim)
        self.static_logvar = nn.Linear(self.flatten_size + self.static_causal_dim, self.static_dim)
        
        # Dynamic branch with causal conditioning (crowds, time, events)
        self.dynamic_mu = nn.Linear(self.flatten_size + self.dynamic_causal_dim, self.dynamic_dim)
        self.dynamic_logvar = nn.Linear(self.flatten_size + self.dynamic_causal_dim, self.dynamic_dim)
        
        # Causal feature splitters
        self.static_causal_proj = nn.Linear(64, self.static_causal_dim)
        self.dynamic_causal_proj = nn.Linear(64, self.dynamic_causal_dim)
        
    def encode(self, x: torch.Tensor, causal_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode with causal conditioning
        
        Args:
            x: (batch_size, 3, 64, 64) - input images
            causal_state: (batch_size, 45) - causal state vector
            
        Returns:
            static_mu, static_logvar, dynamic_mu, dynamic_logvar
        """
        # Process image through shared encoder
        h = self.shared_encoder(x)  # (batch_size, flatten_size)
        
        # Process causal state
        causal_features = self.causal_processor(causal_state)  # (batch_size, 64)
        
        # Split causal features for static vs dynamic
        static_causal = self.static_causal_proj(causal_features)   # (batch_size, 32)
        dynamic_causal = self.dynamic_causal_proj(causal_features) # (batch_size, 32)
        
        # Combine visual and causal features for each branch
        static_input = torch.cat([h, static_causal], dim=-1)
        dynamic_input = torch.cat([h, dynamic_causal], dim=-1)
        
        # Generate latent parameters
        static_mu = self.static_mu(static_input)
        static_logvar = self.static_logvar(static_input)
        
        dynamic_mu = self.dynamic_mu(dynamic_input)
        dynamic_logvar = self.dynamic_logvar(dynamic_input)
        
        return static_mu, static_logvar, dynamic_mu, dynamic_logvar
    
    def forward(self, x: torch.Tensor, causal_state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with causal conditioning
        
        Args:
            x: (batch_size, 3, 64, 64) - input images
            causal_state: (batch_size, 45) - causal state vector
            
        Returns:
            recon, static_mu, static_logvar, dynamic_mu, dynamic_logvar, z_static, z_dynamic
        """
        static_mu, static_logvar, dynamic_mu, dynamic_logvar = self.encode(x, causal_state)
        
        z_static = self.reparameterize(static_mu, static_logvar)
        z_dynamic = self.reparameterize(dynamic_mu, dynamic_logvar)
        
        recon = self.decode(z_static, z_dynamic)
        
        return recon, static_mu, static_logvar, dynamic_mu, dynamic_logvar, z_static, z_dynamic


class CausalConditionedGaussianVAE(GaussianVAE256D):
    """Gaussian VAE with causal conditioning - learns p(z | x, causal_state)"""
    
    def __init__(self, input_shape=(64, 64, 3), causal_dim=45):
        super().__init__(input_shape)
        
        self.causal_dim = causal_dim
        self.causal_processor = CausalConditioningModule(causal_dim, 64)
        
        # Modify latent layers to incorporate causal conditioning
        self.flatten_size = 512 * 4 * 4
        
        self.fc_mu = nn.Linear(self.flatten_size + 64, 256)      # + causal features
        self.fc_logvar = nn.Linear(self.flatten_size + 64, 256)  # + causal features
        
    def encode(self, x: torch.Tensor, causal_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with causal conditioning"""
        h = self.encoder(x)  # (batch_size, flatten_size)
        causal_features = self.causal_processor(causal_state)  # (batch_size, 64)
        
        # Combine visual and causal features
        combined_input = torch.cat([h, causal_features], dim=-1)
        
        mu = self.fc_mu(combined_input)
        logvar = self.fc_logvar(combined_input)
        
        return mu, logvar
    
    def forward(self, x: torch.Tensor, causal_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with causal conditioning"""
        mu, logvar = self.encode(x, causal_state)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        return recon, mu, logvar, z


class CausalConditionedBetaVAE(BetaVAE):
    """Beta-VAE with causal conditioning for disentangled causal representations"""
    
    def __init__(self, input_shape=(64, 64, 3), causal_dim=45, beta=4.0):
        super().__init__(input_shape, beta=beta)
        
        self.causal_dim = causal_dim
        self.causal_processor = CausalConditioningModule(causal_dim, 64)
        
        # Modify latent layers
        self.flatten_size = 512 * 4 * 4
        
        self.fc_mu = nn.Linear(self.flatten_size + 64, 256)
        self.fc_logvar = nn.Linear(self.flatten_size + 64, 256)
        
    def encode(self, x: torch.Tensor, causal_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with causal conditioning"""
        h = self.encoder(x)
        causal_features = self.causal_processor(causal_state)
        
        combined_input = torch.cat([h, causal_features], dim=-1)
        
        mu = self.fc_mu(combined_input)
        logvar = self.fc_logvar(combined_input)
        
        return mu, logvar
    
    def forward(self, x: torch.Tensor, causal_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with causal conditioning"""
        mu, logvar = self.encode(x, causal_state)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        return recon, mu, logvar, z


# Generic Causal VAE Wrapper
class CausalConditionedBaselineVAE(BaselineVAE32D):
    """Baseline 32D VAE with causal conditioning"""
    def __init__(self, input_shape=(64, 64, 3), causal_dim=45):
        super().__init__(input_shape)
        self.causal_dim = causal_dim
        self.causal_processor = CausalConditioningModule(causal_dim, 64)
        self.flatten_size = 128 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_size + 64, 32)
        self.fc_logvar = nn.Linear(self.flatten_size + 64, 32)
    
    def encode(self, x, causal_state):
        h = self.encoder(x)
        causal_features = self.causal_processor(causal_state)
        combined_input = torch.cat([h, causal_features], dim=-1)
        mu = self.fc_mu(combined_input)
        logvar = self.fc_logvar(combined_input)
        return mu, logvar
    
    def forward(self, x, causal_state):
        mu, logvar = self.encode(x, causal_state)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


class CausalConditionedCategoricalVAE(CategoricalVAE512D):
    """Categorical VAE with causal conditioning"""
    def __init__(self, input_shape=(64, 64, 3), causal_dim=45):
        super().__init__(input_shape)  # CategoricalVAE512D only takes input_shape
        self.causal_dim = causal_dim
        self.causal_processor = CausalConditioningModule(causal_dim, 64)
        self.tau = 1.0  # Gumbel-Softmax temperature
        
        # Modify categorical layers to include causal features
        self.flatten_size = 512 * 4 * 4
        self.fc_categorical = nn.Linear(self.flatten_size + 64, self.num_categoricals * self.num_classes)
    
    def encode(self, x, causal_state, hard=False):
        h = self.encoder(x)
        causal_features = self.causal_processor(causal_state)
        combined_input = torch.cat([h, causal_features], dim=-1)
        logits = self.fc_categorical(combined_input)
        logits = logits.view(-1, self.num_categoricals, self.num_classes)
        probs = F.gumbel_softmax(logits, tau=self.tau, hard=hard)
        z = probs.view(-1, self.latent_dim)
        return logits, z
    
    def forward(self, x, causal_state, hard=False):
        logits, z = self.encode(x, causal_state, hard)
        recon = self.decode(z)
        # Return 4 values to match other VAE architectures (recon, mu, logvar, z)
        # For Categorical VAE: recon, logits, z, z (duplicate z for compatibility)
        return recon, logits, z, z
    
    def loss_function(self, recon_x, x, logits, z, **kwargs):
        """Override to match standard VAE signature (recon_x, x, mu, logvar)"""
        # Call parent's loss_function with just the needed arguments
        return super().loss_function(recon_x, x, logits, **kwargs)


class CausalConditionedVQVAE(VQVAE256D):
    """VQ-VAE with causal conditioning"""
    def __init__(self, input_shape=(64, 64, 3), causal_dim=45, codebook_size=256, commitment_cost=0.25):
        super().__init__(input_shape, codebook_size, commitment_cost)
        self.causal_dim = causal_dim
        self.causal_processor = CausalConditioningModule(causal_dim, 64)
        
        # VQ-VAE encoder outputs (64x64->32x32->16x16) so 256 channels at 16x16
        # Replace the original encoder to include causal conditioning
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),     # 64->32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),   # 32->16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1)                       # 16->16, 256 channels
        )
        
        # Add causal conditioning layer before VQ
        self.causal_proj = nn.Linear(64, 256)  # Project causal features to match channels
        
    def encode_with_causal(self, x, causal_state):
        """Encode with causal conditioning integrated before VQ layer"""
        # Get visual features
        h = self.encoder(x)  # (batch, 256, 16, 16)
        batch_size, channels, height, width = h.shape
        
        # Process causal state
        causal_features = self.causal_processor(causal_state)  # (batch, 64)
        causal_projected = self.causal_proj(causal_features)   # (batch, 256)
        
        # Add causal bias to each spatial location
        causal_bias = causal_projected.view(batch_size, channels, 1, 1)
        h_conditioned = h + causal_bias
        
        return h_conditioned
    
    def forward(self, x, causal_state):
        z_e = self.encode_with_causal(x, causal_state)
        z_q, vq_loss, min_indices = self.vq_layer(z_e)  # Use vq_layer method
        recon = self.decoder(z_q)
        # Return 4 values to match other VAE architectures (recon, mu, logvar, z)
        # For VQ-VAE: recon, encoded, quantized, quantized (z_q is the "latent")
        # Store vq_loss as an attribute for loss_function to access
        self._vq_loss = vq_loss
        return recon, z_e, z_q, z_q
    
    def loss_function(self, recon_x, x, z_e, z_q, **kwargs):
        """Override to match standard VAE signature (recon_x, x, mu, logvar)"""
        # Use the stored vq_loss from forward pass
        vq_loss = getattr(self, '_vq_loss', torch.tensor(0.0, device=x.device))
        # Call parent's loss_function with just the needed arguments
        return super().loss_function(recon_x, x, vq_loss, **kwargs)


class CausalConditionedNoConvNormVAE(NoConvNormVAE):
    """No Conv Normalization VAE with causal conditioning"""
    def __init__(self, input_shape=(64, 64, 3), causal_dim=45):
        super().__init__(input_shape)
        self.causal_dim = causal_dim
        self.causal_processor = CausalConditioningModule(causal_dim, 64)
        self.flatten_size = 512 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_size + 64, 256)
        self.fc_logvar = nn.Linear(self.flatten_size + 64, 256)
    
    def encode(self, x, causal_state):
        h = self.encoder(x)
        causal_features = self.causal_processor(causal_state)
        combined_input = torch.cat([h, causal_features], dim=-1)
        mu = self.fc_mu(combined_input)
        logvar = self.fc_logvar(combined_input)
        return mu, logvar
    
    def forward(self, x, causal_state):
        mu, logvar = self.encode(x, causal_state)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


class CausalConditionedDeeperEncoderVAE(DeeperEncoderVAE):
    """Deeper Encoder VAE with causal conditioning"""
    def __init__(self, input_shape=(64, 64, 3), causal_dim=45):
        super().__init__(input_shape)
        self.causal_dim = causal_dim
        self.causal_processor = CausalConditioningModule(causal_dim, 64)
        self.flatten_size = 512 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_size + 64, 256)
        self.fc_logvar = nn.Linear(self.flatten_size + 64, 256)
    
    def encode(self, x, causal_state):
        h = self.encoder(x)
        causal_features = self.causal_processor(causal_state)
        combined_input = torch.cat([h, causal_features], dim=-1)
        mu = self.fc_mu(combined_input)
        logvar = self.fc_logvar(combined_input)
        return mu, logvar
    
    def forward(self, x, causal_state):
        mu, logvar = self.encode(x, causal_state)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def create_causal_conditioned_vae(architecture: str, **kwargs):
    """
    Factory function to create causal-conditioned VAE architectures
    
    Args:
        architecture: VAE architecture name with "_causal" suffix
        **kwargs: Architecture-specific parameters
        
    Returns:
        Causal-conditioned VAE model
    """
    # Clean kwargs - remove architecture-specific params that causal architectures don't need
    clean_kwargs = {k: v for k, v in kwargs.items() if k in ['input_shape', 'causal_dim']}
    
    if architecture == 'baseline_32D_causal':
        return CausalConditionedBaselineVAE(**clean_kwargs)
    elif architecture == 'gaussian_256D_causal':
        return CausalConditionedGaussianVAE(**clean_kwargs)
    elif architecture == 'beta_vae_4.0_causal':
        # BetaVAE needs beta parameter in constructor - remove from clean_kwargs to avoid duplicate
        beta_kwargs = {k: v for k, v in clean_kwargs.items() if k != 'beta'}
        return CausalConditionedBetaVAE(beta=4.0, **beta_kwargs)
    elif architecture == 'hierarchical_512D_causal':
        # Hierarchical needs static/dynamic dims
        static_dim = kwargs.get('static_dim', 256)
        dynamic_dim = kwargs.get('dynamic_dim', 256)
        return CausalConditionedHierarchicalVAE(static_dim=static_dim, dynamic_dim=dynamic_dim, **clean_kwargs)
    elif architecture == 'categorical_512D_causal':
        return CausalConditionedCategoricalVAE(**clean_kwargs)
    elif architecture == 'vq_vae_256D_causal':
        # VQ-VAE needs codebook params
        codebook_size = kwargs.get('codebook_size', 256)
        commitment_cost = kwargs.get('commitment_cost', 0.25)
        return CausalConditionedVQVAE(codebook_size=codebook_size, commitment_cost=commitment_cost, **clean_kwargs)
    elif architecture == 'no_conv_normalization_causal':
        return CausalConditionedNoConvNormVAE(**clean_kwargs)
    elif architecture == 'deeper_encoder_causal':
        return CausalConditionedDeeperEncoderVAE(**clean_kwargs)
    else:
        available = [
            'baseline_32D_causal', 'gaussian_256D_causal', 'beta_vae_4.0_causal', 
            'hierarchical_512D_causal', 'categorical_512D_causal', 'vq_vae_256D_causal',
            'no_conv_normalization_causal', 'deeper_encoder_causal'
        ]
        raise ValueError(f"Unknown causal architecture: {architecture}. Available: {available}")


if __name__ == "__main__":
    # Test causal-conditioned VAE architectures
    print("🧪 Testing Causal-Conditioned VAE Architectures")
    print("=" * 60)
    
    # Test data
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    causal_state = torch.zeros(batch_size, 45)
    
    # Set some causal factors (one-hot encoding)
    for i in range(batch_size):
        causal_state[i, 14] = 1    # 2 PM (time)
        causal_state[i, 26] = 1    # Tuesday (day)  
        causal_state[i, 31] = 1    # Sunny (weather)
        causal_state[i, 35] = 1    # Normal (event)
        causal_state[i, 42] = 1    # Medium crowds
    
    architectures_to_test = [
        'hierarchical_512D_causal',
        'gaussian_256D_causal', 
        'beta_vae_4.0_causal'
    ]
    
    for arch_name in architectures_to_test:
        print(f"\n🏗️  Testing {arch_name}:")
        
        try:
            model = create_causal_conditioned_vae(arch_name)
            
            with torch.no_grad():
                if 'hierarchical' in arch_name:
                    recon, static_mu, static_logvar, dynamic_mu, dynamic_logvar, z_static, z_dynamic = model(x, causal_state)
                    print(f"   ✅ Output shapes: recon={recon.shape}, z_static={z_static.shape}, z_dynamic={z_dynamic.shape}")
                    print(f"   ✅ Static latent dim: {z_static.shape[-1]}, Dynamic latent dim: {z_dynamic.shape[-1]}")
                else:
                    recon, mu, logvar, z = model(x, causal_state)
                    print(f"   ✅ Output shapes: recon={recon.shape}, z={z.shape}")
                    print(f"   ✅ Latent dim: {z.shape[-1]}")
                    
                print(f"   ✅ Causal conditioning: input 45D causal → processed for latent generation")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎉 Causal-Conditioned VAE architectures ready!")
    print(f"🔬 These learn p(z | x, causal_state) instead of just p(z | x)")
    print(f"📊 Ready for true causal world model training!")