"""
Modern VAE Architectures for Phase 1 Experiments
Implements all 8 architecture variants from the complete project plan:
1. baseline_32D - Original World Models style 
2. gaussian_256D - Bigger Gaussian VAE
3. categorical_512D - DreamerV3 style (reduced from 1024D)
4. beta_vae_4.0 - Disentangled β-VAE
5. vq_vae_256D - Vector Quantized VAE
6. hierarchical_512D - Our static/dynamic innovation
7. no_conv_normalization - Ablation study
8. deeper_encoder - Depth vs width test

Uses PyTorch for modern implementation with 2024 best practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from abc import ABC, abstractmethod


# Base VAE Interface
class BaseVAE(nn.Module, ABC):
    """Base class for all VAE architectures"""
    
    def __init__(self, input_shape=(64, 64, 3), latent_dim=256):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def encode(self, x):
        pass
    
    @abstractmethod
    def decode(self, z):
        pass
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def loss_function(self, recon_x, x, **kwargs):
        pass


# 1. Baseline 32D - Original World Models style
class BaselineVAE32D(BaseVAE):
    """Original World Models architecture with 32D Gaussian latent"""
    
    def __init__(self, input_shape=(64, 64, 3)):
        super().__init__(input_shape, latent_dim=32)
        
        # Encoder - matches original conv structure
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 64->32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32->16  
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), # 16->8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 8->4
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size
        self.flatten_size = 128 * 4 * 4  # 2048
        
        # Latent layers
        self.fc_mu = nn.Linear(self.flatten_size, 32)
        self.fc_logvar = nn.Linear(self.flatten_size, 32)
        
        # Decoder
        self.fc_decode = nn.Linear(32, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 4->8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 8->16
            nn.ReLU(), 
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),    # 32->64
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def loss_function(self, recon_x, x, mu, logvar, **kwargs):
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss, recon_loss, kl_loss


# 2. Gaussian 256D - Modern bigger version
class GaussianVAE256D(BaseVAE):
    """Modern Gaussian VAE with 256D latent and layer normalization"""
    
    def __init__(self, input_shape=(64, 64, 3)):
        super().__init__(input_shape, latent_dim=256)
        
        # Modern encoder with layer normalization
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([128, 16, 16]), 
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([512, 4, 4]),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.flatten_size = 512 * 4 * 4  # 8192
        
        self.fc_mu = nn.Linear(self.flatten_size, 256)
        self.fc_logvar = nn.Linear(self.flatten_size, 256)
        
        # Modern decoder
        self.fc_decode = nn.Linear(256, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        # Free bits (DreamerV3 technique)
        self.free_bits = 1.0
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def loss_function(self, recon_x, x, mu, logvar, **kwargs):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL loss with free bits
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.clamp(kl_loss, min=self.free_bits)  # Free bits technique
        kl_loss = kl_loss.sum()
        
        return recon_loss + kl_loss, recon_loss, kl_loss


# 3. Categorical 512D - DreamerV3 style (reduced from 1024D)
class CategoricalVAE512D(BaseVAE):
    """DreamerV3 style categorical VAE with 16x32 = 512D latent space"""
    
    def __init__(self, input_shape=(64, 64, 3)):
        super().__init__(input_shape, latent_dim=512)
        
        self.num_categoricals = 16
        self.num_classes = 32
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([512, 4, 4]),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.flatten_size = 512 * 4 * 4
        
        # Categorical latent layers
        self.fc_categorical = nn.Linear(self.flatten_size, self.num_categoricals * self.num_classes)
        
        # Decoder
        self.fc_decode = nn.Linear(self.latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        logits = self.fc_categorical(h)
        logits = logits.view(-1, self.num_categoricals, self.num_classes)
        return logits
    
    def sample_categorical(self, logits, hard=False):
        # Gumbel-Softmax for differentiable sampling
        if hard:
            # Hard sampling for inference
            indices = torch.argmax(logits, dim=-1)
            categorical = F.one_hot(indices, self.num_classes).float()
        else:
            # Soft sampling for training
            categorical = F.gumbel_softmax(logits, tau=1.0, hard=False)
        
        return categorical
    
    def decode(self, categorical):
        # Flatten categorical representation
        z = categorical.view(-1, self.latent_dim)
        h = self.fc_decode(z)
        return self.decoder(h)
    
    def forward(self, x, hard=False):
        logits = self.encode(x)
        categorical = self.sample_categorical(logits, hard=hard)
        recon = self.decode(categorical)
        return recon, logits, categorical
    
    def loss_function(self, recon_x, x, logits, **kwargs):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # Categorical KL loss
        q = F.softmax(logits, dim=-1)
        log_q = F.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.tensor(1.0 / self.num_classes))
        
        kl_loss = torch.sum(q * (log_q - log_uniform))
        
        return recon_loss + kl_loss, recon_loss, kl_loss


# 4. Beta VAE - Disentangled representation learning
class BetaVAE(BaseVAE):
    """β-VAE for disentangled representation learning"""
    
    def __init__(self, input_shape=(64, 64, 3), beta=4.0):
        super().__init__(input_shape, latent_dim=256)
        
        self.beta = beta
        
        # Same architecture as GaussianVAE256D but with beta weighting
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([512, 4, 4]),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.flatten_size = 512 * 4 * 4
        
        self.fc_mu = nn.Linear(self.flatten_size, 256)
        self.fc_logvar = nn.Linear(self.flatten_size, 256)
        
        self.fc_decode = nn.Linear(256, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def loss_function(self, recon_x, x, mu, logvar, **kwargs):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL loss with beta weighting for disentanglement
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss


# 5. Vector Quantized VAE - Discrete representation
class VQVAE256D(BaseVAE):
    """Vector Quantized VAE with discrete codebook"""
    
    def __init__(self, input_shape=(64, 64, 3), codebook_size=256, commitment_cost=0.25):
        super().__init__(input_shape, latent_dim=256)
        
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.embedding_dim = 256
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.embedding_dim, kernel_size=1)
        )
        
        # Vector Quantization layer
        self.vq_layer = VectorQuantization(self.codebook_size, self.embedding_dim, self.commitment_cost)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embedding_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        encoded = self.encode(x)
        quantized, vq_loss, perplexity = self.vq_layer(encoded)
        decoded = self.decode(quantized)
        
        return decoded, encoded, quantized, vq_loss, perplexity
    
    def loss_function(self, recon_x, x, vq_loss, **kwargs):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        return recon_loss + vq_loss, recon_loss, vq_loss


class VectorQuantization(nn.Module):
    """Vector Quantization layer for VQ-VAE"""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Get closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view_as(inputs)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='mean')
        q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction='mean')
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity


# 6. Hierarchical VAE - Our static/dynamic innovation
class HierarchicalVAE512D(BaseVAE):
    """Hierarchical VAE with static/dynamic factorization"""
    
    def __init__(self, input_shape=(64, 64, 3)):
        super().__init__(input_shape, latent_dim=512)
        
        self.static_dim = 256   # Building layouts (shouldn't change)
        self.dynamic_dim = 256  # Crowds, weather effects
        
        # Shared encoder backbone
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([512, 4, 4]),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.flatten_size = 512 * 4 * 4
        
        # Static branch (buildings, paths)
        self.static_mu = nn.Linear(self.flatten_size, self.static_dim)
        self.static_logvar = nn.Linear(self.flatten_size, self.static_dim)
        
        # Dynamic branch (crowds, weather effects)
        self.dynamic_mu = nn.Linear(self.flatten_size, self.dynamic_dim)
        self.dynamic_logvar = nn.Linear(self.flatten_size, self.dynamic_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(self.latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.shared_encoder(x)
        
        # Static encoding
        static_mu = self.static_mu(h)
        static_logvar = self.static_logvar(h)
        
        # Dynamic encoding
        dynamic_mu = self.dynamic_mu(h)
        dynamic_logvar = self.dynamic_logvar(h)
        
        return static_mu, static_logvar, dynamic_mu, dynamic_logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z_static, z_dynamic):
        z = torch.cat([z_static, z_dynamic], dim=1)
        h = self.fc_decode(z)
        return self.decoder(h)
    
    def forward(self, x):
        static_mu, static_logvar, dynamic_mu, dynamic_logvar = self.encode(x)
        
        z_static = self.reparameterize(static_mu, static_logvar)
        z_dynamic = self.reparameterize(dynamic_mu, dynamic_logvar)
        
        recon = self.decode(z_static, z_dynamic)
        
        return recon, static_mu, static_logvar, dynamic_mu, dynamic_logvar, z_static, z_dynamic
    
    def loss_function(self, recon_x, x, static_mu, static_logvar, dynamic_mu, dynamic_logvar, **kwargs):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # Static KL loss
        static_kl = -0.5 * torch.sum(1 + static_logvar - static_mu.pow(2) - static_logvar.exp())
        
        # Dynamic KL loss  
        dynamic_kl = -0.5 * torch.sum(1 + dynamic_logvar - dynamic_mu.pow(2) - dynamic_logvar.exp())
        
        return recon_loss + static_kl + dynamic_kl, recon_loss, static_kl + dynamic_kl


# 7 & 8. Ablation Studies
class NoConvNormVAE(BaseVAE):
    """Ablation: Remove layer normalization to test importance"""
    
    def __init__(self, input_shape=(64, 64, 3)):
        super().__init__(input_shape, latent_dim=256)
        
        # Same architecture but WITHOUT layer normalization
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # No LayerNorm
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # No LayerNorm
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # No LayerNorm
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # No LayerNorm
            nn.Flatten()
        )
        
        self.flatten_size = 512 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_size, 256)
        self.fc_logvar = nn.Linear(self.flatten_size, 256)
        
        self.fc_decode = nn.Linear(256, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # No LayerNorm
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # No LayerNorm
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # No LayerNorm
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def loss_function(self, recon_x, x, mu, logvar, **kwargs):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss


class DeeperEncoderVAE(BaseVAE):
    """Ablation: Test depth vs width - 8 layers instead of 4"""
    
    def __init__(self, input_shape=(64, 64, 3)):
        super().__init__(input_shape, latent_dim=256)
        
        # Deeper encoder with 8 layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),     # 64->64
            nn.LayerNorm([32, 64, 64]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),    # 64->32
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),    # 32->32
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),   # 32->16
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 16->16
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16->8
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 8->8
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8->4
            nn.LayerNorm([512, 4, 4]),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.flatten_size = 512 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_size, 256)
        self.fc_logvar = nn.Linear(self.flatten_size, 256)
        
        # Standard decoder (don't make decoder deeper too)
        self.fc_decode = nn.Linear(256, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def loss_function(self, recon_x, x, mu, logvar, **kwargs):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss


# Factory function to create VAE architectures
def create_vae_architecture(arch_name: str, **kwargs):
    """Factory function to create VAE architectures for experiments"""
    
    architectures = {
        'baseline_32D': BaselineVAE32D,
        'gaussian_256D': GaussianVAE256D,
        'categorical_512D': CategoricalVAE512D,
        'beta_vae_4.0': lambda **kw: BetaVAE(beta=4.0, **kw),
        'vq_vae_256D': VQVAE256D,
        'hierarchical_512D': HierarchicalVAE512D,
        'no_conv_normalization': NoConvNormVAE,
        'deeper_encoder': DeeperEncoderVAE
    }
    
    if arch_name not in architectures:
        raise ValueError(f"Unknown architecture: {arch_name}. Available: {list(architectures.keys())}")
    
    return architectures[arch_name](**kwargs)


# Utility functions for training
def symlog_transform(x):
    """DreamerV3 symlog transform for stable training"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp_transform(x):
    """Inverse of symlog transform"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


if __name__ == "__main__":
    # Test all architectures
    print("🧪 Testing Modern VAE Architectures")
    print("=" * 50)
    
    test_input = torch.randn(4, 3, 64, 64)  # Batch of 4 images
    
    architectures_to_test = [
        'baseline_32D',
        'gaussian_256D', 
        'categorical_512D',
        'beta_vae_4.0',
        'vq_vae_256D',
        'hierarchical_512D',
        'no_conv_normalization',
        'deeper_encoder'
    ]
    
    for arch_name in architectures_to_test:
        print(f"\n🔧 Testing {arch_name}...")
        
        try:
            model = create_vae_architecture(arch_name)
            model.eval()
            
            with torch.no_grad():
                if 'categorical' in arch_name:
                    output = model(test_input, hard=True)
                    print(f"   ✅ Output shapes: recon={output[0].shape}, logits={output[1].shape}")
                elif 'vq_vae' in arch_name:
                    output = model(test_input)
                    print(f"   ✅ Output shapes: recon={output[0].shape}, vq_loss={output[3].item():.3f}")
                elif 'hierarchical' in arch_name:
                    output = model(test_input)
                    print(f"   ✅ Output shapes: recon={output[0].shape}, static_z={output[5].shape}, dynamic_z={output[6].shape}")
                else:
                    output = model(test_input)
                    print(f"   ✅ Output shapes: recon={output[0].shape}, z={output[3].shape}")
                    
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            print(f"   📊 Parameters: {param_count:,}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎉 Architecture testing completed!")
    print(f"📋 Ready for Phase 1 parallel training experiments")