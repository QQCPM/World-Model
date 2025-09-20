"""
Domain-Invariant Causal Learning
PHASE 2 ENHANCEMENT: Cross-domain causal transfer capabilities

Research Inspiration:
- Domain-Adversarial Neural Networks (DANN) for domain adaptation
- CORAL: Deep CORAL for domain adaptation in deep neural networks
- Learning Transferable Features with Deep Adaptation Networks

Core Concept:
Extract domain-invariant causal features that generalize across different
environments while preserving the essential causal relationships.

Key Components:
1. Domain-Invariant Feature Extractor
2. Causal Relationship Abstraction Layer
3. Domain Adaptation mechanisms
4. Cross-domain validation framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DomainAdaptationConfig:
    """Configuration for domain adaptation"""
    feature_dim: int = 64
    num_domains: int = 3  # campus, urban, rural environments
    adaptation_weight: float = 0.1
    adversarial_weight: float = 0.05
    coral_weight: float = 0.02

    # Causal abstraction
    abstract_causal_dim: int = 8
    concrete_causal_dim: int = 5

    # Training parameters
    domain_classifier_hidden: int = 32
    gradient_reversal_lambda: float = 1.0


class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal layer for domain adversarial training"""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_val
        return output, None


class DomainInvariantFeatureExtractor(nn.Module):
    """
    Extract domain-invariant features from causal factors

    Uses adversarial training to learn features that are:
    1. Informative for causal reasoning
    2. Invariant across different domains
    """

    def __init__(self, config: DomainAdaptationConfig):
        super().__init__()
        self.config = config

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.concrete_causal_dim, config.feature_dim),
            nn.ReLU(),
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.ReLU(),
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.abstract_causal_dim)
        )

        # Domain classifier (for adversarial training)
        self.domain_classifier = nn.Sequential(
            nn.Linear(config.abstract_causal_dim, config.domain_classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.domain_classifier_hidden, config.num_domains)
        )

        # Causal relationship abstractor
        self.causal_abstractor = nn.Sequential(
            nn.Linear(config.abstract_causal_dim, config.feature_dim),
            nn.ReLU(),
            nn.Linear(config.feature_dim, config.abstract_causal_dim)
        )

    def forward(self, causal_factors, domain_id=None, lambda_val=1.0):
        """
        Extract domain-invariant features

        Args:
            causal_factors: [batch, seq_len, concrete_causal_dim] input features
            domain_id: Optional domain identifier for training
            lambda_val: Gradient reversal strength

        Returns:
            abstract_features: [batch, seq_len, abstract_causal_dim] domain-invariant features
            domain_prediction: Domain classification logits (if domain_id provided)
        """
        # Extract abstract features
        abstract_features = self.feature_extractor(causal_factors)

        # Apply causal relationship abstraction
        abstract_features = self.causal_abstractor(abstract_features)

        domain_prediction = None
        if domain_id is not None:
            # Gradient reversal for adversarial training
            reversed_features = GradientReversalLayer.apply(abstract_features, lambda_val)
            domain_prediction = self.domain_classifier(reversed_features)

        return abstract_features, domain_prediction


class CausalRelationshipAbstractor(nn.Module):
    """
    Learn abstract causal relationships that transfer across domains

    Maps concrete domain-specific causal patterns to abstract patterns
    that maintain causal semantics but are domain-invariant.
    """

    def __init__(self, config: DomainAdaptationConfig):
        super().__init__()
        self.config = config

        # Abstract causal relationship learning
        self.relationship_encoder = nn.Sequential(
            nn.Linear(2, config.feature_dim // 4),  # Pairwise relationships (2 features at a time)
            nn.ReLU(),
            nn.Linear(config.feature_dim // 4, config.feature_dim // 8),
            nn.ReLU(),
            nn.Linear(config.feature_dim // 8, 1)  # Relationship strength
        )

        # Temporal abstraction for causal delays
        self.temporal_abstractor = nn.Sequential(
            nn.Linear(config.abstract_causal_dim, config.feature_dim),
            nn.ReLU(),
            nn.Linear(config.feature_dim, config.abstract_causal_dim)
        )

        # Abstract adjacency matrix learning
        self.adjacency_learner = nn.Linear(
            config.abstract_causal_dim * config.abstract_causal_dim,
            config.abstract_causal_dim * config.abstract_causal_dim
        )

    def forward(self, abstract_features):
        """
        Learn abstract causal relationships

        Args:
            abstract_features: [batch, seq_len, abstract_causal_dim]

        Returns:
            abstract_adjacency: [batch, abstract_causal_dim, abstract_causal_dim]
            temporal_abstract: [batch, seq_len, abstract_causal_dim]
        """
        batch_size, seq_len, feature_dim = abstract_features.shape

        # Compute pairwise relationships
        relationships = []
        for i in range(feature_dim):
            for j in range(feature_dim):
                if i != j:
                    pair_features = torch.cat([
                        abstract_features[:, :, i:i+1],
                        abstract_features[:, :, j:j+1]
                    ], dim=-1)  # [batch, seq_len, 2]

                    # Average over sequence length for relationship strength
                    pair_avg = torch.mean(pair_features, dim=1)  # [batch, 2]
                    relationship_strength = self.relationship_encoder(pair_avg)  # [batch, 1]
                    relationships.append(relationship_strength)

        # Construct abstract adjacency matrix
        adjacency_vec = torch.cat(relationships, dim=-1)  # [batch, abstract_dim^2 - abstract_dim]

        # Pad diagonal (self-relationships are zero)
        full_adjacency = torch.zeros(batch_size, feature_dim * feature_dim, device=abstract_features.device)
        non_diag_indices = [i * feature_dim + j for i in range(feature_dim) for j in range(feature_dim) if i != j]
        full_adjacency[:, non_diag_indices] = adjacency_vec

        # Learn refined adjacency
        refined_adjacency = self.adjacency_learner(full_adjacency)
        abstract_adjacency = refined_adjacency.view(batch_size, feature_dim, feature_dim)

        # Apply temporal abstraction
        temporal_abstract = self.temporal_abstractor(abstract_features)

        return abstract_adjacency, temporal_abstract


class DomainAdaptationLayer(nn.Module):
    """
    Adapt abstract causal features to specific domains

    Takes domain-invariant abstract features and adapts them
    to work effectively in specific target domains.
    """

    def __init__(self, config: DomainAdaptationConfig):
        super().__init__()
        self.config = config

        # Domain-specific adaptation layers
        self.domain_adapters = nn.ModuleDict()
        for domain_id in range(config.num_domains):
            self.domain_adapters[str(domain_id)] = nn.Sequential(
                nn.Linear(config.abstract_causal_dim, config.feature_dim),
                nn.ReLU(),
                nn.LayerNorm(config.feature_dim),
                nn.Linear(config.feature_dim, config.concrete_causal_dim)
            )

        # CORAL (CORrelation ALignment) loss components
        self.coral_regularizer = nn.Parameter(torch.ones(1))

    def forward(self, abstract_features, target_domain_id):
        """
        Adapt abstract features to target domain

        Args:
            abstract_features: [batch, seq_len, abstract_causal_dim]
            target_domain_id: Target domain identifier

        Returns:
            adapted_features: [batch, seq_len, concrete_causal_dim]
        """
        adapter = self.domain_adapters[str(target_domain_id)]
        adapted_features = adapter(abstract_features)
        return adapted_features

    def compute_coral_loss(self, source_features, target_features):
        """
        Compute CORAL loss for domain adaptation

        Aligns the covariance of source and target feature distributions
        """
        # Flatten batch and sequence dimensions
        source_flat = source_features.view(-1, source_features.size(-1))
        target_flat = target_features.view(-1, target_features.size(-1))

        # Compute covariance matrices
        source_cov = torch.cov(source_flat.T)
        target_cov = torch.cov(target_flat.T)

        # CORAL loss (Frobenius norm of covariance difference)
        coral_loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        return coral_loss * self.coral_regularizer


class DomainInvariantCausalLearner(nn.Module):
    """
    Complete domain-invariant causal learning system

    Integrates all components for cross-domain causal transfer:
    1. Domain-invariant feature extraction
    2. Abstract causal relationship learning
    3. Domain adaptation for target environments
    """

    def __init__(self, config: DomainAdaptationConfig = None):
        super().__init__()
        self.config = config or DomainAdaptationConfig()

        # Core components
        self.feature_extractor = DomainInvariantFeatureExtractor(self.config)
        self.relationship_abstractor = CausalRelationshipAbstractor(self.config)
        self.domain_adapter = DomainAdaptationLayer(self.config)

        # Training state
        self.training_domains = set()
        self.domain_statistics = {}

        logger.info("DomainInvariantCausalLearner initialized for cross-domain transfer")

    def forward(self, causal_factors, source_domain_id=0, target_domain_id=None, training=True):
        """
        Complete forward pass for domain-invariant causal learning

        Args:
            causal_factors: [batch, seq_len, concrete_causal_dim]
            source_domain_id: Source domain identifier
            target_domain_id: Target domain (if different from source)
            training: Whether in training mode

        Returns:
            results: Dict containing all outputs and intermediate representations
        """
        results = {}

        # Step 1: Extract domain-invariant features
        lambda_val = self.config.gradient_reversal_lambda if training else 0.0
        abstract_features, domain_prediction = self.feature_extractor(
            causal_factors, source_domain_id if training else None, lambda_val
        )

        results['abstract_features'] = abstract_features
        results['domain_prediction'] = domain_prediction

        # Step 2: Learn abstract causal relationships
        abstract_adjacency, temporal_abstract = self.relationship_abstractor(abstract_features)

        results['abstract_adjacency'] = abstract_adjacency
        results['temporal_abstract'] = temporal_abstract

        # Step 3: Domain adaptation (if target domain specified)
        if target_domain_id is not None:
            adapted_features = self.domain_adapter(abstract_features, target_domain_id)
            results['adapted_features'] = adapted_features

            # Compute CORAL loss if we have source features for comparison
            if training and hasattr(self, '_source_features_cache'):
                coral_loss = self.domain_adapter.compute_coral_loss(
                    self._source_features_cache, abstract_features
                )
                results['coral_loss'] = coral_loss

        return results

    def compute_domain_adaptation_losses(self, results, true_domain_id):
        """
        Compute all domain adaptation losses

        Args:
            results: Forward pass results
            true_domain_id: True domain label

        Returns:
            losses: Dict of loss components
        """
        losses = {}

        # Domain classification loss (adversarial)
        if results['domain_prediction'] is not None:
            domain_targets = torch.full(
                (results['domain_prediction'].size(0),),
                true_domain_id,
                dtype=torch.long,
                device=results['domain_prediction'].device
            )
            domain_loss = F.cross_entropy(results['domain_prediction'], domain_targets)
            losses['domain_adversarial_loss'] = domain_loss * self.config.adversarial_weight

        # CORAL loss (if available)
        if 'coral_loss' in results:
            losses['coral_loss'] = results['coral_loss'] * self.config.coral_weight

        return losses

    def evaluate_cross_domain_transfer(self, source_data, target_data, source_domain_id, target_domain_id):
        """
        Evaluate cross-domain transfer performance

        Args:
            source_data: Source domain causal factors
            target_data: Target domain causal factors
            source_domain_id: Source domain ID
            target_domain_id: Target domain ID

        Returns:
            transfer_metrics: Dict of transfer performance metrics
        """
        self.eval()

        with torch.no_grad():
            # Extract features from source domain
            source_results = self.forward(source_data, source_domain_id, training=False)

            # Extract features from target domain
            target_results = self.forward(target_data, target_domain_id, training=False)

            # Compute transfer metrics
            source_abstract = source_results['abstract_features']
            target_abstract = target_results['abstract_features']

            # Feature similarity (how similar are the abstract representations)
            feature_similarity = F.cosine_similarity(
                source_abstract.mean(dim=1).mean(dim=0),
                target_abstract.mean(dim=1).mean(dim=0),
                dim=0
            ).item()

            # Adjacency consistency (how similar are the learned causal structures)
            source_adj = source_results['abstract_adjacency'].mean(dim=0)
            target_adj = target_results['abstract_adjacency'].mean(dim=0)

            adj_consistency = F.cosine_similarity(
                source_adj.flatten(),
                target_adj.flatten(),
                dim=0
            ).item()

            transfer_metrics = {
                'feature_similarity': feature_similarity,
                'adjacency_consistency': adj_consistency,
                'transfer_score': (feature_similarity + adj_consistency) / 2.0
            }

        return transfer_metrics


def create_domain_invariant_learner(config=None):
    """
    Factory function for domain-invariant causal learner

    Args:
        config: Optional DomainAdaptationConfig

    Returns:
        learner: DomainInvariantCausalLearner instance
    """
    return DomainInvariantCausalLearner(config)


# Test the domain-invariant learner
if __name__ == "__main__":
    print("🌍 Testing Domain-Invariant Causal Learner")

    # Create test configuration
    config = DomainAdaptationConfig(
        feature_dim=32,
        num_domains=3,
        abstract_causal_dim=6,
        concrete_causal_dim=5
    )

    # Create learner
    learner = create_domain_invariant_learner(config)

    # Test with sample data
    batch_size, seq_len = 4, 10
    source_data = torch.randn(batch_size, seq_len, 5)  # Campus environment
    target_data = torch.randn(batch_size, seq_len, 5)  # Urban environment

    print(f"✅ Created domain-invariant learner: {learner.__class__.__name__}")
    print(f"✅ Test data shapes: source={source_data.shape}, target={target_data.shape}")

    # Test forward pass
    results = learner(source_data, source_domain_id=0, target_domain_id=1)

    print(f"✅ Forward pass successful:")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
        else:
            print(f"   {key}: {value}")

    # Test cross-domain evaluation
    transfer_metrics = learner.evaluate_cross_domain_transfer(
        source_data, target_data, source_domain_id=0, target_domain_id=1
    )

    print(f"✅ Cross-domain transfer metrics:")
    for key, value in transfer_metrics.items():
        print(f"   {key}: {value:.4f}")

    print("\n🚀 Domain-Invariant Causal Learner ready for Phase 2 integration!")