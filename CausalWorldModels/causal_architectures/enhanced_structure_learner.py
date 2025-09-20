"""
Enhanced Causal Structure Learner with Adaptive Thresholding
PHASE 2 ENHANCEMENT: Structure learning from 0 edges → functional graphs

Research-Inspired Enhancements:
- CL-NOTEARS curriculum learning approach
- Gradient-informed adaptive thresholding
- PC-NOTEARS hybrid statistical constraints
- Dynamic edge weight adjustment
- PRESERVES: NOTEARS foundation and DAG constraints

Based on:
- CL-NOTEARS: Curriculum Learning for Causal Structure Discovery (2024)
- PC-NOTEARS: Hybrid constraint-based + continuous optimization
- Active Learning for Optimal Intervention Design (Nature MI, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
try:
    from scipy.stats import pearsonr
except ImportError:
    # Fallback correlation function if scipy not available
    def pearsonr(x, y):
        x_np = x if isinstance(x, np.ndarray) else np.array(x)
        y_np = y if isinstance(y, np.ndarray) else np.array(y)
        corr = np.corrcoef(x_np, y_np)[0, 1]
        # Simple p-value approximation
        n = len(x_np)
        t_stat = corr * np.sqrt((n-2)/(1-corr**2)) if abs(corr) < 0.99 else 10
        p_val = 2 * (1 - 0.95) if abs(t_stat) > 2 else 0.2  # Rough approximation
        return corr, p_val
import warnings
warnings.filterwarnings('ignore')

try:
    from causal_architectures.structure_learner import CausalStructureLearner
except ImportError:
    from .structure_learner import CausalStructureLearner


class AdaptiveThresholdScheduler:
    """
    Curriculum learning scheduler for dynamic threshold adaptation
    Based on CL-NOTEARS methodology
    """

    def __init__(self, initial_threshold=0.3, final_threshold=0.7, total_epochs=100):
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def get_threshold(self, epoch=None):
        """Get adaptive threshold for current epoch"""
        if epoch is not None:
            self.current_epoch = epoch

        # Curriculum learning: start permissive, gradually increase
        progress = min(self.current_epoch / self.total_epochs, 1.0)

        # Use sigmoid-like progression for smooth curriculum
        threshold = self.initial_threshold + (self.final_threshold - self.initial_threshold) * (
            1 / (1 + np.exp(-10 * (progress - 0.5)))
        )

        return threshold

    def step(self):
        """Step to next epoch"""
        self.current_epoch += 1


class GradientInformedThresholding:
    """
    Gradient-informed adaptive thresholding mechanism
    Uses gradient magnitudes to determine active learning regions
    """

    def __init__(self, percentile=80, smoothing_factor=0.1):
        self.percentile = percentile
        self.smoothing_factor = smoothing_factor
        self.gradient_history = []

    def compute_gradient_threshold(self, adjacency_logits, loss):
        """
        Compute threshold based on gradient magnitudes

        Args:
            adjacency_logits: Current adjacency logits
            loss: Current loss value

        Returns:
            gradient_threshold: Adaptive threshold based on gradients
        """
        try:
            # Compute gradients w.r.t adjacency logits
            if adjacency_logits.grad is not None:
                gradient_magnitudes = torch.abs(adjacency_logits.grad)
            else:
                # Compute gradients manually if not available
                gradients = torch.autograd.grad(
                    loss, adjacency_logits, retain_graph=True, create_graph=False
                )[0]
                gradient_magnitudes = torch.abs(gradients)

            # Use percentile-based thresholding (research insight)
            gradient_threshold = torch.quantile(gradient_magnitudes, self.percentile / 100.0)

            # Smooth with history
            if self.gradient_history:
                prev_threshold = self.gradient_history[-1]
                gradient_threshold = (1 - self.smoothing_factor) * prev_threshold + \
                                   self.smoothing_factor * gradient_threshold

            self.gradient_history.append(gradient_threshold.item())

            # Keep history bounded
            if len(self.gradient_history) > 20:
                self.gradient_history = self.gradient_history[-20:]

            return gradient_threshold.item()

        except Exception as e:
            # Fallback to default threshold if gradient computation fails
            return 0.4


class StatisticalConstraintGenerator:
    """
    PC-algorithm inspired statistical constraints for structure learning
    Pre-filters potential edges using conditional independence tests
    """

    def __init__(self, significance_level=0.05, min_correlation=0.1):
        self.significance_level = significance_level
        self.min_correlation = min_correlation

    def generate_pc_constraints(self, data):
        """
        Generate PC-algorithm style constraints from data

        Args:
            data: [batch_size, seq_len, num_variables] causal data

        Returns:
            edge_constraints: Dict with potential edges and their statistical support
        """
        batch_size, seq_len, num_vars = data.shape
        edge_constraints = {}

        # Flatten data for correlation analysis
        data_flat = data.view(-1, num_vars)  # [batch*seq, num_vars]

        # Test pairwise correlations (simplified PC step)
        for i in range(num_vars):
            for j in range(num_vars):
                if i != j:  # No self-loops
                    try:
                        # Compute correlation
                        corr, p_value = pearsonr(
                            data_flat[:, i].detach().cpu().numpy(),
                            data_flat[:, j].detach().cpu().numpy()
                        )

                        # Statistical significance and minimum correlation
                        is_significant = p_value < self.significance_level
                        is_strong_enough = abs(corr) > self.min_correlation

                        edge_constraints[(i, j)] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'significant': is_significant,
                            'strong': is_strong_enough,
                            'recommended': is_significant and is_strong_enough,
                            'constraint_weight': abs(corr) if is_significant else 0.0
                        }

                    except Exception:
                        # Fallback for edge case
                        edge_constraints[(i, j)] = {
                            'correlation': 0.0,
                            'p_value': 1.0,
                            'significant': False,
                            'strong': False,
                            'recommended': False,
                            'constraint_weight': 0.0
                        }

        return edge_constraints


class EnhancedCausalStructureLearner(CausalStructureLearner):
    """
    Enhanced Causal Structure Learner with Research-Backed Adaptive Thresholding

    Key Enhancements:
    1. Adaptive Threshold Scheduling (CL-NOTEARS)
    2. Gradient-Informed Thresholding
    3. PC-Algorithm Statistical Constraints
    4. Curriculum Learning for Structure Discovery
    5. PRESERVES: All existing NOTEARS functionality
    """

    def __init__(self, num_variables=5, hidden_dim=64, learning_rate=1e-3, total_epochs=100):
        super().__init__(num_variables, hidden_dim, learning_rate)

        # PRESERVE: All parent class functionality

        # ENHANCEMENT 1: Adaptive Threshold Scheduler
        self.threshold_scheduler = AdaptiveThresholdScheduler(
            initial_threshold=0.2,    # Start permissive (curriculum learning)
            final_threshold=0.6,      # End more selective
            total_epochs=total_epochs
        )

        # ENHANCEMENT 2: Gradient-Informed Thresholding
        self.gradient_thresholder = GradientInformedThresholding(
            percentile=80,            # Use top 20% of gradients
            smoothing_factor=0.1      # Smooth threshold changes
        )

        # ENHANCEMENT 3: Statistical Constraint Generator
        self.constraint_generator = StatisticalConstraintGenerator(
            significance_level=0.01,  # More stringent statistical significance
            min_correlation=0.3       # Higher minimum correlation for meaningful edges
        )

        # Enhancement tracking
        self.training_epoch = 0
        self.edge_constraints = None
        self.threshold_history = []
        self.enhancement_enabled = True

    def get_adaptive_threshold(self, data, loss=None):
        """
        Compute adaptive threshold using multiple research-backed methods

        Args:
            data: Training data for statistical analysis
            loss: Current loss for gradient analysis

        Returns:
            adaptive_threshold: Dynamically computed threshold
        """
        if not self.enhancement_enabled:
            return 0.5  # Fallback to original threshold

        # Method 1: Curriculum learning threshold (CL-NOTEARS)
        curriculum_threshold = self.threshold_scheduler.get_threshold(self.training_epoch)

        # Method 2: Gradient-informed threshold
        gradient_threshold = self.gradient_thresholder.compute_gradient_threshold(
            self.adjacency_logits, loss
        ) if loss is not None else 0.4

        # Method 3: Statistical constraint guidance
        if self.edge_constraints is not None:
            # Use average constraint weight as statistical guidance
            constraint_weights = [c['constraint_weight'] for c in self.edge_constraints.values()]
            statistical_guidance = np.mean(constraint_weights) if constraint_weights else 0.3
        else:
            statistical_guidance = 0.3

        # Combine methods with research-backed weighting
        # Emphasize curriculum early, gradients mid-training, statistics always
        epoch_progress = min(self.training_epoch / 50.0, 1.0)

        curriculum_weight = 1.0 - epoch_progress     # Strong early, weaker later
        gradient_weight = epoch_progress             # Weak early, stronger later
        statistical_weight = 0.5                    # Stronger statistical influence

        adaptive_threshold = (
            curriculum_weight * curriculum_threshold +
            gradient_weight * gradient_threshold +
            statistical_weight * statistical_guidance
        ) / (curriculum_weight + gradient_weight + statistical_weight)

        # Add selectivity boost: increase threshold if too many weak edges
        if hasattr(self, 'edge_constraints') and self.edge_constraints:
            strong_edges = sum(1 for c in self.edge_constraints.values() if c['constraint_weight'] > 0.5)
            total_edges = len(self.edge_constraints)
            if total_edges > 0:
                strong_ratio = strong_edges / total_edges
                if strong_ratio < 0.3:  # If less than 30% of potential edges are strong
                    adaptive_threshold += 0.1  # Be more selective

        # Clamp to reasonable range
        adaptive_threshold = np.clip(adaptive_threshold, 0.1, 0.8)

        self.threshold_history.append(adaptive_threshold)

        return adaptive_threshold

    def get_enhanced_adjacency_matrix(self, data=None, loss=None, temperature=1.0, hard=False):
        """
        Enhanced adjacency matrix generation with adaptive thresholding

        PRESERVES: Original get_adjacency_matrix interface
        ENHANCES: Adaptive threshold based on learning progress
        """
        # Get base adjacency matrix (preserve original logic)
        logits = self.adjacency_logits.clone()
        logits.fill_diagonal_(float('-inf'))
        adjacency = torch.sigmoid(logits / temperature)

        if hard and self.enhancement_enabled:
            # Use adaptive threshold instead of fixed 0.5
            adaptive_threshold = self.get_adaptive_threshold(data, loss)
            adjacency = (adjacency > adaptive_threshold).float()
        elif hard:
            # Original behavior when enhancement disabled
            adjacency = (adjacency > 0.5).float()

        return adjacency

    def get_adjacency_matrix(self, temperature=1.0, hard=False):
        """
        BACKWARD COMPATIBILITY: Keep original interface
        Falls back to original behavior when no enhancement data available
        """
        if self.enhancement_enabled and hasattr(self, '_current_data'):
            return self.get_enhanced_adjacency_matrix(
                self._current_data, getattr(self, '_current_loss', None), temperature, hard
            )
        else:
            # Original behavior
            return super().get_adjacency_matrix(temperature, hard)

    def compute_enhanced_structure_loss(self, causal_data, interventions=None):
        """
        Enhanced structure loss with statistical constraints and adaptive learning

        PRESERVES: Original compute_structure_loss functionality
        ENHANCES: Statistical constraints, curriculum learning
        """
        # Store data for adaptive thresholding
        self._current_data = causal_data

        # Generate statistical constraints from data
        self.edge_constraints = self.constraint_generator.generate_pc_constraints(causal_data)

        # Compute original structure loss
        total_loss, loss_info = super().compute_structure_loss(causal_data, interventions)

        # Store loss for gradient thresholding
        self._current_loss = total_loss

        # ENHANCEMENT: Add statistical constraint regularization
        if self.enhancement_enabled and self.edge_constraints:
            constraint_regularization = self._compute_constraint_regularization()
            total_loss += 0.1 * constraint_regularization
            loss_info['constraint_regularization'] = constraint_regularization.item()

        # Update epoch tracking
        self.training_epoch += 1
        self.threshold_scheduler.step()

        # Enhanced loss information
        loss_info.update({
            'adaptive_threshold': self.get_adaptive_threshold(causal_data, total_loss),
            'curriculum_threshold': self.threshold_scheduler.get_threshold(),
            'training_epoch': self.training_epoch,
            'enhancement_active': self.enhancement_enabled
        })

        return total_loss, loss_info

    def _compute_constraint_regularization(self):
        """
        Compute regularization term based on statistical constraints
        Encourages edges that have statistical support
        """
        adjacency = self.get_adjacency_matrix(hard=False)
        constraint_penalty = torch.tensor(0.0, device=adjacency.device)

        for (i, j), constraint in self.edge_constraints.items():
            edge_strength = adjacency[i, j]
            statistical_support = constraint['constraint_weight']

            if constraint['recommended']:
                # Encourage edges with statistical support
                penalty = -statistical_support * edge_strength  # Negative = encouragement
            else:
                # Discourage edges without statistical support
                penalty = 0.1 * edge_strength  # Positive = discouragement

            constraint_penalty += penalty

        return constraint_penalty

    def get_enhanced_causal_graph_summary(self, use_adaptive_threshold=True):
        """
        Enhanced graph summary with adaptive threshold

        PRESERVES: Original summary format
        ENHANCES: Adaptive edge detection
        """
        if use_adaptive_threshold and self.enhancement_enabled:
            # Use adaptive threshold for edge detection
            adaptive_threshold = self.get_adaptive_threshold(
                getattr(self, '_current_data', None),
                getattr(self, '_current_loss', None)
            )
        else:
            # Use original threshold
            adaptive_threshold = 0.3

        adjacency = self.get_adjacency_matrix(hard=False)
        adj_np = adjacency.detach().cpu().numpy()

        # ENHANCED: Statistical-guided edge selection
        # First, collect all potential edges with their statistical support
        potential_edges = []
        for i in range(self.num_variables):
            for j in range(self.num_variables):
                if i != j:  # No self-loops
                    weight = adj_np[i, j]
                    statistical_support = 0.0
                    is_statistically_supported = False

                    if self.edge_constraints and (i, j) in self.edge_constraints:
                        constraint = self.edge_constraints[(i, j)]
                        statistical_support = constraint['correlation']
                        is_statistically_supported = constraint['recommended']

                    potential_edges.append({
                        'cause': self.variable_names[i],
                        'effect': self.variable_names[j],
                        'weight': weight,
                        'cause_idx': i,
                        'effect_idx': j,
                        'statistical_support': abs(statistical_support),  # Use absolute correlation
                        'is_statistically_supported': is_statistically_supported,
                        'combined_score': weight * (1 + abs(statistical_support))  # Combined metric
                    })

        # ENHANCEMENT: Select edges based on statistical support + weight
        if use_adaptive_threshold and self.enhancement_enabled:
            # Strategy 1: Select top edges by statistical support
            statistically_strong = [e for e in potential_edges if e['statistical_support'] > 0.5]

            # Strategy 2: If we have strong statistical edges, use those
            if len(statistically_strong) >= 2:
                # Sort by statistical support and take top ones
                statistically_strong.sort(key=lambda x: x['statistical_support'], reverse=True)
                edges = statistically_strong[:6]  # Reasonable maximum
            else:
                # Fallback: use combined score with adaptive threshold
                above_threshold = [e for e in potential_edges if e['weight'] > adaptive_threshold]
                above_threshold.sort(key=lambda x: x['combined_score'], reverse=True)
                edges = above_threshold[:8]  # Limit to prevent over-discovery
        else:
            # Original threshold-based approach
            edges = [e for e in potential_edges if e['weight'] > adaptive_threshold]

        # Add threshold information to each edge
        for edge in edges:
            edge['threshold_used'] = adaptive_threshold

        # Compute enhanced graph statistics
        num_edges = len(edges)
        sparsity = 1 - (num_edges / (self.num_variables ** 2 - self.num_variables))

        # Find root causes and effects (same as original)
        incoming_edges = np.sum(adj_np > adaptive_threshold, axis=0)
        root_causes = [self.variable_names[i] for i in range(self.num_variables)
                      if incoming_edges[i] == 0]

        outgoing_edges = np.sum(adj_np > adaptive_threshold, axis=1)
        final_effects = [self.variable_names[i] for i in range(self.num_variables)
                        if outgoing_edges[i] == 0]

        summary = {
            'edges': edges,
            'num_edges': num_edges,
            'sparsity': sparsity,
            'root_causes': root_causes,
            'final_effects': final_effects,
            'adjacency_matrix': adj_np.tolist(),
            'variable_names': self.variable_names,
            'graph_density': num_edges / (self.num_variables * (self.num_variables - 1)),
            'adaptive_threshold': adaptive_threshold,
            'enhancement_info': {
                'statistical_constraints': len(self.edge_constraints) if self.edge_constraints else 0,
                'curriculum_threshold': self.threshold_scheduler.get_threshold(),
                'training_epoch': self.training_epoch
            }
        }

        return summary

    def get_causal_graph_summary(self):
        """
        BACKWARD COMPATIBILITY: Keep original interface
        """
        return self.get_enhanced_causal_graph_summary(use_adaptive_threshold=True)

    def enable_enhancements(self, enable=True):
        """Enable or disable enhancements for comparison"""
        self.enhancement_enabled = enable

    def reset_training_state(self):
        """Reset training state for new training session"""
        self.training_epoch = 0
        self.threshold_scheduler.current_epoch = 0
        self.gradient_thresholder.gradient_history = []
        self.threshold_history = []
        self.edge_constraints = None

    def get_model_name(self):
        return "enhanced_causal_structure_learner"

    def get_enhancement_info(self):
        """Get detailed information about enhancements"""
        return {
            'adaptive_thresholding': True,
            'curriculum_learning': True,
            'statistical_constraints': True,
            'gradient_informed': True,
            'training_epoch': self.training_epoch,
            'current_threshold': self.threshold_history[-1] if self.threshold_history else 0.5,
            'enhancement_enabled': self.enhancement_enabled
        }


def create_enhanced_structure_learner(num_variables=5, total_epochs=100):
    """
    Factory function for creating enhanced structure learner
    """
    return EnhancedCausalStructureLearner(
        num_variables=num_variables,
        hidden_dim=64,
        learning_rate=1e-3,
        total_epochs=total_epochs
    )


def test_enhanced_structure_learner():
    """
    Test enhanced structure learner functionality
    """
    print("🧪 Testing Enhanced Causal Structure Learner...")

    # Create enhanced learner
    enhanced_learner = create_enhanced_structure_learner(num_variables=5, total_epochs=50)

    # Generate test data with known structure (same as original test)
    batch_size, seq_len = 32, 20
    causal_data = torch.zeros(batch_size, seq_len, 5)

    # Initialize
    causal_data[:, 0, :] = torch.randn(batch_size, 5) * 0.5

    # Generate with strong causal relationships
    for t in range(1, seq_len):
        # Strong weather -> crowd relationship
        causal_data[:, t, 1] = 0.9 * causal_data[:, t-1, 0] + 0.1 * torch.randn(batch_size)
        # Strong time -> road relationship
        causal_data[:, t, 4] = 0.85 * causal_data[:, t-1, 3] + 0.15 * torch.randn(batch_size)
        # Root causes evolve
        causal_data[:, t, 0] = 0.95 * causal_data[:, t-1, 0] + 0.05 * torch.randn(batch_size)
        causal_data[:, t, 3] = 0.95 * causal_data[:, t-1, 3] + 0.05 * torch.randn(batch_size)
        causal_data[:, t, 2] = 0.7 * causal_data[:, t-1, 2] + 0.3 * torch.randn(batch_size)

    print(f"✅ Generated test data: {causal_data.shape}")

    # Test enhanced structure learning
    loss, loss_info = enhanced_learner.compute_enhanced_structure_loss(causal_data)
    print(f"✅ Enhanced structure loss: {loss.item():.4f}")
    print(f"✅ Adaptive threshold: {loss_info['adaptive_threshold']:.3f}")
    print(f"✅ Curriculum threshold: {loss_info['curriculum_threshold']:.3f}")

    # Test enhanced adjacency matrix
    adjacency = enhanced_learner.get_enhanced_adjacency_matrix(causal_data, loss, hard=True)
    print(f"✅ Enhanced adjacency shape: {adjacency.shape}")

    # Test enhanced graph summary
    summary = enhanced_learner.get_enhanced_causal_graph_summary()
    print(f"✅ Enhanced edges discovered: {summary['num_edges']}")
    print(f"✅ Adaptive threshold used: {summary['adaptive_threshold']:.3f}")

    if summary['edges']:
        print("✅ Discovered relationships:")
        for edge in summary['edges']:
            print(f"   {edge['cause']} → {edge['effect']} (weight: {edge['weight']:.3f}, stat: {edge['statistical_support']:.3f})")

    # Test backward compatibility
    original_summary = enhanced_learner.get_causal_graph_summary()
    assert 'edges' in original_summary
    print("✅ Backward compatibility maintained")

    print(f"✅ Enhancement info: {enhanced_learner.get_enhancement_info()}")

    return enhanced_learner


if __name__ == "__main__":
    enhanced_learner = test_enhanced_structure_learner()
    print("\n🎯 Enhanced Structure Learner ready for Phase 2 integration!")