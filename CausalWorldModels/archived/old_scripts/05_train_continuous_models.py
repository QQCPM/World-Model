#!/usr/bin/env python3
"""
Continuous State Prediction Model Training
Train models to predict next state from current state, action, and causal factors

Usage:
python 05_train_continuous_models.py --model_type lstm_predictor --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import argparse
import json
import time
from typing import Dict, List, Tuple, Optional
import glob
from collections import defaultdict

# Add continuous models to path
sys.path.append('continuous_models')
from state_predictors import (
    create_continuous_model,
    ContinuousStateLoss,
    VAELoss,
    get_model_info
)


class ContinuousEpisodeDataset(Dataset):
    """Dataset for continuous state prediction training"""

    def __init__(self, episode_files: List[str], sequence_length: int = 20, overlap: int = 10):
        """
        Initialize dataset from episode files

        Args:
            episode_files: List of .npz episode file paths
            sequence_length: Length of sequences for training
            overlap: Overlap between consecutive sequences
        """
        self.episode_files = episode_files
        self.sequence_length = sequence_length
        self.overlap = overlap

        # Load and process all episodes
        self.sequences = []
        self._load_episodes()

        print(f"Loaded {len(self.sequences)} sequences from {len(episode_files)} episodes")
        print(f"Sequence length: {sequence_length}, overlap: {overlap}")

    def _load_episodes(self):
        """Load episodes and extract training sequences"""
        for episode_file in self.episode_files:
            try:
                episode_data = np.load(episode_file, allow_pickle=True)

                # Extract data
                observations = episode_data['obs']  # [episode_length, 12]
                actions = episode_data['action']     # [episode_length, 2]
                causal_factors = episode_data['causal']  # [episode_length, 5]

                episode_length = len(observations)

                # Skip episodes that are too short
                if episode_length < self.sequence_length + 1:
                    continue

                # Extract overlapping sequences
                step = max(1, self.sequence_length - self.overlap)
                for start_idx in range(0, episode_length - self.sequence_length, step):
                    end_idx = start_idx + self.sequence_length

                    # Input sequences (current states, actions, causal factors)
                    states = observations[start_idx:end_idx]
                    seq_actions = actions[start_idx:end_idx]
                    seq_causal = causal_factors[start_idx:end_idx]

                    # Target sequences (next states)
                    target_states = observations[start_idx + 1:end_idx + 1]

                    self.sequences.append({
                        'states': states.astype(np.float32),
                        'actions': seq_actions.astype(np.float32),
                        'causal_factors': seq_causal.astype(np.float32),
                        'target_states': target_states.astype(np.float32),
                        'episode_file': episode_file
                    })

            except Exception as e:
                print(f"Error loading episode {episode_file}: {e}")
                continue

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return (
            torch.from_numpy(sequence['states']),
            torch.from_numpy(sequence['actions']),
            torch.from_numpy(sequence['causal_factors']),
            torch.from_numpy(sequence['target_states'])
        )


class ContinuousModelTrainer:
    """Trainer for continuous state prediction models"""

    def __init__(self, model_type: str, device: str = 'auto', **model_kwargs):
        """
        Initialize trainer

        Args:
            model_type: Type of model to train
            device: Device to use ('auto', 'cpu', 'cuda')
            **model_kwargs: Additional arguments for model creation
        """
        self.model_type = model_type
        self.device = self._get_device(device)
        self.model_kwargs = model_kwargs

        # Create model
        self.model = create_continuous_model(model_type, **model_kwargs)
        self.model.to(self.device)

        # Create appropriate loss function
        if model_type == 'vae_rnn_hybrid':
            self.criterion = VAELoss()
        else:
            self.criterion = ContinuousStateLoss()

        # Training history
        self.training_history = defaultdict(list)

        print(f"Created {model_type} model with {self._count_parameters()} parameters")
        print(f"Training on device: {self.device}")

    def _get_device(self, device: str) -> torch.device:
        """Determine training device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)

    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                   epochs: int = 100, lr: float = 1e-3, patience: int = 15):
        """
        Train the continuous model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            patience: Early stopping patience
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Learning rate: {lr}, Patience: {patience}")

        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer)

            # Validation phase
            val_loss = self._validate_epoch(val_loader)

            # Update learning rate
            scheduler.step(val_loss)

            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['epoch'].append(epoch)

            # Progress reporting
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}/{epochs}: Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self._save_checkpoint('best')
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (patience: {patience})")
                break

            # Save periodic checkpoint
            if epoch % 25 == 0 and epoch > 0:
                self._save_checkpoint(f'epoch_{epoch}')

        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return self.training_history

    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (states, actions, causal_factors, target_states) in enumerate(train_loader):
            # Move to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            causal_factors = causal_factors.to(self.device)
            target_states = target_states.to(self.device)

            optimizer.zero_grad()

            # Forward pass
            if self.model_type == 'vae_rnn_hybrid':
                predictions, mu, logvar = self.model(states, actions, causal_factors)
                loss, recon_loss, kl_loss = self.criterion(predictions, target_states, mu, logvar)
            elif self.model_type in ['lstm_predictor', 'gru_dynamics']:
                predictions, _ = self.model(states, actions, causal_factors)
                loss, mse_loss = self.criterion(predictions, target_states)
            elif self.model_type == 'neural_ode':
                # For Neural ODE, predict one step at a time
                batch_size, seq_len = states.shape[:2]
                predictions = []

                for t in range(seq_len):
                    pred = self.model(states[:, t], actions[:, t], causal_factors[:, t])
                    predictions.append(pred)

                predictions = torch.stack(predictions, dim=1)
                loss, mse_loss = self.criterion(predictions, target_states)
            else:  # linear_dynamics
                # For linear dynamics, predict one step at a time
                batch_size, seq_len = states.shape[:2]
                predictions = []

                for t in range(seq_len):
                    pred = self.model(states[:, t], actions[:, t], causal_factors[:, t])
                    predictions.append(pred)

                predictions = torch.stack(predictions, dim=1)
                loss, mse_loss = self.criterion(predictions, target_states)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for states, actions, causal_factors, target_states in val_loader:
                # Move to device
                states = states.to(self.device)
                actions = actions.to(self.device)
                causal_factors = causal_factors.to(self.device)
                target_states = target_states.to(self.device)

                # Forward pass (same as training)
                if self.model_type == 'vae_rnn_hybrid':
                    predictions, mu, logvar = self.model(states, actions, causal_factors)
                    loss, _, _ = self.criterion(predictions, target_states, mu, logvar)
                elif self.model_type in ['lstm_predictor', 'gru_dynamics']:
                    predictions, _ = self.model(states, actions, causal_factors)
                    loss, _ = self.criterion(predictions, target_states)
                else:  # neural_ode, linear_dynamics
                    batch_size, seq_len = states.shape[:2]
                    predictions = []

                    for t in range(seq_len):
                        pred = self.model(states[:, t], actions[:, t], causal_factors[:, t])
                        predictions.append(pred)

                    predictions = torch.stack(predictions, dim=1)
                    loss, _ = self.criterion(predictions, target_states)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        os.makedirs('models', exist_ok=True)

        checkpoint_path = f"models/{self.model_type}_{checkpoint_name}.pth"

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'model_kwargs': self.model_kwargs,
            'training_history': dict(self.training_history),
            'device': str(self.device)
        }, checkpoint_path)

    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0.0
        mse_losses = []
        predictions_list = []
        targets_list = []

        with torch.no_grad():
            for states, actions, causal_factors, target_states in test_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                causal_factors = causal_factors.to(self.device)
                target_states = target_states.to(self.device)

                # Forward pass
                if self.model_type == 'vae_rnn_hybrid':
                    predictions, mu, logvar = self.model(states, actions, causal_factors)
                    loss, mse_loss, _ = self.criterion(predictions, target_states, mu, logvar)
                elif self.model_type in ['lstm_predictor', 'gru_dynamics']:
                    predictions, _ = self.model(states, actions, causal_factors)
                    loss, mse_loss = self.criterion(predictions, target_states)
                else:
                    batch_size, seq_len = states.shape[:2]
                    pred_list = []

                    for t in range(seq_len):
                        pred = self.model(states[:, t], actions[:, t], causal_factors[:, t])
                        pred_list.append(pred)

                    predictions = torch.stack(pred_list, dim=1)
                    loss, mse_loss = self.criterion(predictions, target_states)

                total_loss += loss.item()
                mse_losses.append(mse_loss.item() if hasattr(mse_loss, 'item') else mse_loss)

                # Store for analysis
                predictions_list.append(predictions.cpu().numpy())
                targets_list.append(target_states.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(test_loader)
        avg_mse = np.mean(mse_losses)

        # Concatenate all predictions and targets
        all_predictions = np.concatenate(predictions_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)

        # Compute per-dimension MSE
        per_dim_mse = np.mean((all_predictions - all_targets) ** 2, axis=(0, 1))

        return {
            'test_loss': avg_loss,
            'test_mse': avg_mse,
            'per_dimension_mse': per_dim_mse.tolist(),
            'num_samples': len(all_predictions),
            'model_type': self.model_type
        }


def create_data_loaders(episode_dir: str = 'data/causal_episodes/',
                       train_ratio: float = 0.7, val_ratio: float = 0.15,
                       batch_size: int = 32, sequence_length: int = 20) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test data loaders"""

    # Find all episode files
    episode_files = glob.glob(os.path.join(episode_dir, 'episode_*.npz'))
    episode_files.sort()

    if len(episode_files) == 0:
        raise ValueError(f"No episode files found in {episode_dir}")

    print(f"Found {len(episode_files)} episode files")

    # Split episodes
    n_train = int(len(episode_files) * train_ratio)
    n_val = int(len(episode_files) * val_ratio)

    train_files = episode_files[:n_train]
    val_files = episode_files[n_train:n_train + n_val]
    test_files = episode_files[n_train + n_val:]

    print(f"Train episodes: {len(train_files)}")
    print(f"Validation episodes: {len(val_files)}")
    print(f"Test episodes: {len(test_files)}")

    # Create datasets
    train_dataset = ContinuousEpisodeDataset(train_files, sequence_length=sequence_length)
    val_dataset = ContinuousEpisodeDataset(val_files, sequence_length=sequence_length)
    test_dataset = ContinuousEpisodeDataset(test_files, sequence_length=sequence_length)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Train continuous state prediction models')
    parser.add_argument('--model_type', type=str, default='lstm_predictor',
                       choices=['linear_dynamics', 'lstm_predictor', 'gru_dynamics', 'neural_ode', 'vae_rnn_hybrid'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=20, help='Sequence length for training')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--device', type=str, default='auto', help='Training device')
    parser.add_argument('--episode_dir', type=str, default='data/causal_episodes/', help='Episode data directory')

    args = parser.parse_args()

    print("🚀 Continuous State Prediction Model Training")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Hidden dimension: {args.hidden_dim}")

    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            episode_dir=args.episode_dir,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length
        )

        # Create trainer
        model_kwargs = {'hidden_dim': args.hidden_dim}
        if args.model_type == 'vae_rnn_hybrid':
            model_kwargs['latent_dim'] = 8

        trainer = ContinuousModelTrainer(
            model_type=args.model_type,
            device=args.device,
            **model_kwargs
        )

        # Train model
        start_time = time.time()
        training_history = trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr
        )
        training_time = time.time() - start_time

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = trainer.evaluate_model(test_loader)

        # Save training results
        results = {
            'model_type': args.model_type,
            'training_args': vars(args),
            'training_history': training_history,
            'test_results': test_results,
            'training_time': training_time,
            'model_info': get_model_info(trainer.model)
        }

        results_file = f"results/{args.model_type}_training_results.json"
        os.makedirs('results', exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n🎉 Training completed successfully!")
        print(f"Training time: {training_time:.1f} seconds")
        print(f"Test MSE: {test_results['test_mse']:.6f}")
        print(f"Results saved to: {results_file}")
        print(f"Model saved to: models/{args.model_type}_best.pth")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())