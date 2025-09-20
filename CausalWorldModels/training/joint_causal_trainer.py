"""
Joint Causal Trainer
Integrated training pipeline for structure + dynamics learning

Combines:
- Dual-pathway causal GRU for dynamics
- Structure learning with NOTEARS
- Conservative training curriculum (60/40)
- Structure-aware counterfactual generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import time
import os
from dataclasses import dataclass

from causal_architectures import (
    DualPathwayCausalGRU, CausalStructureLearner, InterventionDesigner,
    CausalMechanismModules, CausalLoss, CounterfactualDynamicsWrapper
)
from training.training_curriculum import ConservativeTrainingCurriculum
from training.counterfactual_generator import StructureAwareCFGenerator


@dataclass
class JointTrainingConfig:
    """Configuration for joint causal training"""
    # Model dimensions
    state_dim: int = 12
    action_dim: int = 2
    causal_dim: int = 5
    hidden_dim: int = 64

    # Training parameters
    learning_rate: float = 1e-3
    structure_learning_rate: float = 5e-4
    batch_size: int = 32
    max_epochs: int = 100
    sequence_length: int = 20

    # Loss weights
    dynamics_loss_weight: float = 1.0
    structure_loss_weight: float = 0.5
    counterfactual_loss_weight: float = 0.3
    dag_constraint_weight: float = 1.0

    # Conservative curriculum
    observational_ratio: float = 0.6
    counterfactual_ratio: float = 0.4

    # Stability monitoring
    patience: int = 10
    min_improvement: float = 1e-4

    # Checkpointing
    save_interval: int = 10
    checkpoint_dir: str = "checkpoints"


class JointCausalTrainer:
    """
    Joint training system for causal structure + dynamics learning

    Integrates all causal components into unified training pipeline
    """

    def __init__(self, config: JointTrainingConfig = None):
        self.config = config or JointTrainingConfig()

        # Initialize components
        self._initialize_models()
        self._initialize_optimizers()
        self._initialize_training_components()

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'dynamics_loss': [],
            'structure_loss': [],
            'counterfactual_loss': [],
            'total_loss': [],
            'dag_constraint': [],
            'pathway_balance': []
        }

    def _initialize_models(self):
        """Initialize all model components"""
        # Enhanced dual-pathway dynamics model with counterfactual wrapper
        base_dynamics_model = DualPathwayCausalGRU(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            causal_dim=self.config.causal_dim,
            hidden_dim=self.config.hidden_dim
        )

        # Wrap with counterfactual dynamics wrapper for enhanced counterfactual reasoning
        self.dynamics_model = CounterfactualDynamicsWrapper(
            base_dynamics_model,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            causal_dim=self.config.causal_dim
        )

        # Structure learning model
        self.structure_learner = CausalStructureLearner(
            num_variables=self.config.causal_dim,
            hidden_dim=self.config.hidden_dim
        )

        # Causal mechanism modules
        self.causal_mechanisms = CausalMechanismModules(
            state_dim=self.config.state_dim,
            hidden_dim=self.config.hidden_dim // 2
        )

        # Intervention designer
        self.intervention_designer = InterventionDesigner(
            num_variables=self.config.causal_dim
        )

        # Counterfactual generator
        self.cf_generator = StructureAwareCFGenerator(
            num_variables=self.config.causal_dim,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim
        )

        # Loss functions
        self.dynamics_loss = CausalLoss(
            mse_weight=1.0,
            pathway_balance_weight=0.1,
            intervention_detection_weight=0.05
        )

    def _initialize_optimizers(self):
        """Initialize optimizers for different components"""
        # Dynamics optimizer
        self.dynamics_optimizer = optim.Adam(
            list(self.dynamics_model.parameters()) +
            list(self.causal_mechanisms.parameters()),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )

        # Structure learning optimizer
        self.structure_optimizer = optim.Adam(
            self.structure_learner.parameters(),
            lr=self.config.structure_learning_rate,
            weight_decay=1e-5
        )

        # Counterfactual generator optimizer
        self.cf_optimizer = optim.Adam(
            self.cf_generator.parameters(),
            lr=self.config.learning_rate * 0.5,
            weight_decay=1e-5
        )

        # Intervention designer optimizer
        self.intervention_optimizer = optim.Adam(
            self.intervention_designer.parameters(),
            lr=self.config.learning_rate * 0.3,
            weight_decay=1e-5
        )

    def _initialize_training_components(self):
        """Initialize training support components"""
        # Conservative curriculum
        self.curriculum = ConservativeTrainingCurriculum()

        # Learning rate schedulers
        self.dynamics_scheduler = optim.lr_scheduler.StepLR(
            self.dynamics_optimizer, step_size=30, gamma=0.8
        )
        self.structure_scheduler = optim.lr_scheduler.StepLR(
            self.structure_optimizer, step_size=40, gamma=0.9
        )

    def train_epoch(self, data_loader, validation_loader=None):
        """
        Train for one epoch with joint learning

        Args:
            data_loader: Training data loader
            validation_loader: Optional validation data loader

        Returns:
            epoch_metrics: Dict with training metrics
        """
        self.dynamics_model.train()
        self.structure_learner.train()
        self.causal_mechanisms.train()
        self.cf_generator.train()

        epoch_losses = {
            'dynamics_loss': 0.0,
            'structure_loss': 0.0,
            'counterfactual_loss': 0.0,
            'total_loss': 0.0,
            'dag_constraint': 0.0
        }

        num_batches = 0

        for batch_idx, batch_data in enumerate(data_loader):
            # Extract batch data
            states, actions, causal_factors = batch_data

            # Get batch composition from curriculum
            batch_composition = self.curriculum.get_batch_composition(states.shape[0])

            # Split into observational and counterfactual data
            obs_indices = torch.randperm(states.shape[0])[:batch_composition['observational']]
            cf_indices = torch.randperm(states.shape[0])[:batch_composition['counterfactual']]

            # Observational training
            if len(obs_indices) > 0:
                obs_losses = self._train_observational_batch(
                    states[obs_indices], actions[obs_indices], causal_factors[obs_indices]
                )
                for key, value in obs_losses.items():
                    epoch_losses[key] += value

            # Counterfactual training
            if len(cf_indices) > 0:
                cf_losses = self._train_counterfactual_batch(
                    states[cf_indices], actions[cf_indices], causal_factors[cf_indices]
                )
                for key, value in cf_losses.items():
                    epoch_losses[key] += value

            num_batches += 1

            # Structure learning (every few batches)
            if batch_idx % 3 == 0:
                structure_loss = self._train_structure_learning(causal_factors)
                epoch_losses['structure_loss'] += structure_loss

            # Intervention designer update (less frequent)
            if batch_idx % 5 == 0:
                self._update_intervention_designer(causal_factors)

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        # Validation
        val_metrics = {}
        if validation_loader is not None:
            val_metrics = self.validate(validation_loader)

        # Update curriculum
        stability_score = self._compute_stability_score()
        self.curriculum.update_epoch(
            self.current_epoch, epoch_losses['total_loss'], stability_score
        )

        # Learning rate scheduling
        self.dynamics_scheduler.step()
        self.structure_scheduler.step()

        # Combine metrics
        epoch_metrics = {
            'train': epoch_losses,
            'validation': val_metrics,
            'curriculum': self.curriculum.get_curriculum_status(),
            'stability_score': stability_score
        }

        return epoch_metrics

    def _train_observational_batch(self, states, actions, causal_factors):
        """
        Train on observational data (normal dynamics)

        Args:
            states: [batch_size, seq_len, state_dim]
            actions: [batch_size, seq_len, action_dim]
            causal_factors: [batch_size, seq_len, causal_dim]

        Returns:
            losses: Dict with loss components
        """
        self.dynamics_optimizer.zero_grad()

        # Forward pass through dual-pathway model
        predicted_states, hidden_states, pathway_info = self.dynamics_model(
            states[:, :-1], actions[:, :-1], causal_factors[:, :-1]
        )

        # Dynamics loss
        target_states = states[:, 1:]
        dynamics_loss, loss_components = self.dynamics_loss(
            predicted_states, target_states, pathway_info
        )

        # Add pathway specialization loss
        specialization_loss_weight = 0.05  # Conservative weight to start
        specialization_loss = pathway_info.get('specialization_loss', 0.0)
        dynamics_loss += specialization_loss_weight * specialization_loss

        # Apply causal mechanisms
        mechanism_effects, composed_effects, mechanism_predictions, independence_loss, isolation_confidence = self.causal_mechanisms(
            states[:, :-1].reshape(-1, self.config.state_dim),
            causal_factors[:, :-1].reshape(-1, self.config.causal_dim),
            actions[:, :-1].reshape(-1, self.config.action_dim)
        )

        # Mechanism consistency loss
        mechanism_predictions = mechanism_predictions.reshape(predicted_states.shape)
        mechanism_loss = F.mse_loss(mechanism_predictions, target_states)

        # Add independence loss to encourage mechanism isolation
        independence_loss_weight = 0.1  # Conservative weight to start
        mechanism_loss += independence_loss_weight * independence_loss

        # Total observational loss
        total_loss = (
            self.config.dynamics_loss_weight * dynamics_loss +
            0.2 * mechanism_loss
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), 1.0)
        self.dynamics_optimizer.step()

        return {
            'dynamics_loss': dynamics_loss.item(),
            'mechanism_loss': mechanism_loss.item(),
            'total_loss': total_loss.item()
        }

    def _train_counterfactual_batch(self, states, actions, causal_factors):
        """
        Train on counterfactual data

        Args:
            states: [batch_size, seq_len, state_dim]
            actions: [batch_size, seq_len, action_dim]
            causal_factors: [batch_size, seq_len, causal_dim]

        Returns:
            losses: Dict with loss components
        """
        # Get current causal graph
        causal_graph = self.structure_learner.get_adjacency_matrix()

        # Generate interventions for counterfactual training
        batch_size = states.shape[0]
        counterfactual_losses = []

        for i in range(batch_size):
            # Create base episode
            base_episode = {
                'states': states[i],
                'actions': actions[i],
                'causal_factors': causal_factors[i]
            }

            # Generate intervention specification
            intervention_spec = self.curriculum.generate_intervention_specification()

            # Generate counterfactual episode
            cf_episode, gen_info = self.cf_generator.generate_counterfactual(
                base_episode, intervention_spec, causal_graph, self.causal_mechanisms
            )

            # Train on counterfactual
            cf_states = cf_episode['states'].unsqueeze(0)
            cf_actions = cf_episode['actions'].unsqueeze(0)
            cf_causal = cf_episode['causal_factors'].unsqueeze(0)

            # Forward pass with interventional pathway emphasis
            self.dynamics_model.set_pathway_mode('interventional')
            predicted_cf_states, _, pathway_info = self.dynamics_model(
                cf_states[:, :-1], cf_actions[:, :-1], cf_causal[:, :-1]
            )

            # Counterfactual loss
            cf_loss = F.mse_loss(predicted_cf_states, cf_states[:, 1:])
            counterfactual_losses.append(cf_loss)

        # Reset pathway mode
        self.dynamics_model.set_pathway_mode('auto')

        # Average counterfactual loss
        avg_cf_loss = torch.stack(counterfactual_losses).mean()

        # Backward pass
        self.cf_optimizer.zero_grad()
        avg_cf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cf_generator.parameters(), 1.0)
        self.cf_optimizer.step()

        return {
            'counterfactual_loss': avg_cf_loss.item(),
            'total_loss': avg_cf_loss.item()
        }

    def _train_structure_learning(self, causal_factors):
        """
        Train causal structure learning

        Args:
            causal_factors: [batch_size, seq_len, causal_dim]

        Returns:
            structure_loss: Scalar loss value
        """
        self.structure_optimizer.zero_grad()

        # Structure learning loss
        structure_loss, loss_info = self.structure_learner.compute_structure_loss(causal_factors)

        structure_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.structure_learner.parameters(), 1.0)
        self.structure_optimizer.step()

        return structure_loss.item()

    def _update_intervention_designer(self, causal_factors):
        """
        Update intervention designer based on recent data

        Args:
            causal_factors: [batch_size, seq_len, causal_dim]
        """
        # Select optimal intervention
        best_intervention, candidates = self.intervention_designer.select_optimal_intervention(
            self.structure_learner, causal_factors
        )

        # Simulate intervention outcome (simplified)
        simulated_gain = best_intervention['info_gain'] * 0.8
        success = simulated_gain > 0.1

        # Update intervention designer
        self.intervention_designer.track_intervention_outcome(
            best_intervention, success
        )

    def validate(self, validation_loader):
        """
        Validate model performance

        Args:
            validation_loader: Validation data loader

        Returns:
            val_metrics: Dict with validation metrics
        """
        self.dynamics_model.eval()
        self.structure_learner.eval()

        val_losses = {
            'dynamics_loss': 0.0,
            'structure_loss': 0.0,
            'total_loss': 0.0
        }

        num_batches = 0

        with torch.no_grad():
            for batch_data in validation_loader:
                states, actions, causal_factors = batch_data

                # Dynamics validation
                predicted_states, _, pathway_info = self.dynamics_model(
                    states[:, :-1], actions[:, :-1], causal_factors[:, :-1]
                )

                dynamics_loss, _ = self.dynamics_loss(
                    predicted_states, states[:, 1:], pathway_info
                )

                # Structure validation
                structure_loss, _ = self.structure_learner.compute_structure_loss(causal_factors)

                val_losses['dynamics_loss'] += dynamics_loss.item()
                val_losses['structure_loss'] += structure_loss.item()
                val_losses['total_loss'] += (dynamics_loss + structure_loss).item()

                num_batches += 1

        # Average validation losses
        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)

        return val_losses

    def _compute_stability_score(self):
        """
        Compute model stability score for curriculum advancement

        Returns:
            stability_score: Scalar stability measure
        """
        if len(self.training_history['total_loss']) < 10:
            return 0.8  # Default moderate stability

        # Compute loss variance (lower = more stable)
        recent_losses = self.training_history['total_loss'][-10:]
        loss_variance = np.var(recent_losses)
        stability = 1.0 / (1.0 + loss_variance)

        return min(stability, 0.99)

    def train(self, train_loader, validation_loader=None, save_path=None):
        """
        Full training loop

        Args:
            train_loader: Training data loader
            validation_loader: Optional validation data loader
            save_path: Optional path to save best model

        Returns:
            training_history: Dict with complete training history
        """
        print(f"Starting joint causal training for {self.config.max_epochs} epochs...")

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train epoch
            epoch_metrics = self.train_epoch(train_loader, validation_loader)

            # Update history
            self.training_history['dynamics_loss'].append(epoch_metrics['train']['dynamics_loss'])
            self.training_history['structure_loss'].append(epoch_metrics['train'].get('structure_loss', 0))
            self.training_history['total_loss'].append(epoch_metrics['train']['total_loss'])

            # Progress logging
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.config.max_epochs} ({epoch_time:.2f}s)")
            print(f"  Train Loss: {epoch_metrics['train']['total_loss']:.4f}")
            if validation_loader:
                print(f"  Val Loss: {epoch_metrics['validation']['total_loss']:.4f}")
            print(f"  Curriculum: {epoch_metrics['curriculum']['current_complexity']} complexity")
            print(f"  Stability: {epoch_metrics['stability_score']:.3f}")

            # Early stopping check
            current_loss = epoch_metrics['validation']['total_loss'] if validation_loader else epoch_metrics['train']['total_loss']

            if current_loss < self.best_loss - self.config.min_improvement:
                self.best_loss = current_loss
                self.patience_counter = 0

                # Save best model
                if save_path:
                    self.save_checkpoint(save_path)

            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Periodic checkpointing
            if (epoch + 1) % self.config.save_interval == 0 and save_path:
                checkpoint_path = f"{save_path}_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path)

        print("Training completed!")
        return self.training_history

    def save_checkpoint(self, path):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'dynamics_model': self.dynamics_model.state_dict(),
            'structure_learner': self.structure_learner.state_dict(),
            'causal_mechanisms': self.causal_mechanisms.state_dict(),
            'cf_generator': self.cf_generator.state_dict(),
            'optimizers': {
                'dynamics': self.dynamics_optimizer.state_dict(),
                'structure': self.structure_optimizer.state_dict(),
                'cf': self.cf_optimizer.state_dict()
            },
            'training_history': self.training_history,
            'config': self.config
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path)

        self.dynamics_model.load_state_dict(checkpoint['dynamics_model'])
        self.structure_learner.load_state_dict(checkpoint['structure_learner'])
        self.causal_mechanisms.load_state_dict(checkpoint['causal_mechanisms'])
        self.cf_generator.load_state_dict(checkpoint['cf_generator'])

        self.dynamics_optimizer.load_state_dict(checkpoint['optimizers']['dynamics'])
        self.structure_optimizer.load_state_dict(checkpoint['optimizers']['structure'])
        self.cf_optimizer.load_state_dict(checkpoint['optimizers']['cf'])

        self.training_history = checkpoint['training_history']
        self.current_epoch = checkpoint['epoch']

        print(f"Checkpoint loaded: {path}")

    def get_causal_analysis(self):
        """
        Get comprehensive causal analysis from trained models

        Returns:
            analysis: Dict with causal insights
        """
        # Structure analysis
        structure_summary = self.structure_learner.get_causal_graph_summary()

        # Intervention analysis
        intervention_analysis = self.intervention_designer.get_intervention_history_analysis()

        # Pathway analysis
        pathway_analysis = {
            'pathway_weights': self.dynamics_model.pathway_weights.data.tolist(),
            'intervention_detector_accuracy': 'placeholder'  # Would compute from validation
        }

        return {
            'causal_structure': structure_summary,
            'intervention_insights': intervention_analysis,
            'pathway_analysis': pathway_analysis,
            'training_convergence': {
                'final_dynamics_loss': self.training_history['dynamics_loss'][-1] if self.training_history['dynamics_loss'] else 0,
                'final_structure_loss': self.training_history['structure_loss'][-1] if self.training_history['structure_loss'] else 0,
                'epochs_trained': len(self.training_history['total_loss'])
            }
        }


def test_joint_trainer():
    """
    Test joint causal trainer functionality
    """
    print("Testing JointCausalTrainer...")

    # Create trainer
    config = JointTrainingConfig(max_epochs=3, batch_size=8)
    trainer = JointCausalTrainer(config)

    # Create synthetic data loader
    class SyntheticDataLoader:
        def __init__(self, num_batches=5):
            self.num_batches = num_batches
            self.current = 0

        def __iter__(self):
            self.current = 0
            return self

        def __next__(self):
            if self.current >= self.num_batches:
                raise StopIteration

            batch_data = (
                torch.randn(8, 20, 12),  # states
                torch.randn(8, 20, 2),   # actions
                torch.randn(8, 20, 5)    # causal_factors
            )
            self.current += 1
            return batch_data

    train_loader = SyntheticDataLoader()

    # Test single epoch
    epoch_metrics = trainer.train_epoch(train_loader)
    print(f"Epoch metrics keys: {list(epoch_metrics.keys())}")

    # Test causal analysis
    analysis = trainer.get_causal_analysis()
    print(f"Analysis keys: {list(analysis.keys())}")

    print("✅ JointCausalTrainer test passed")

    return True


if __name__ == "__main__":
    test_joint_trainer()