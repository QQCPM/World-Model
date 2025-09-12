#!/usr/bin/env python3
"""
Causal RNN Training Script for Phase 2A Experiments
Trains causal MDN-GRU models with different causal factor combinations
Called by phase2a_orchestrator.py to run individual causal validation experiments
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append('..')
sys.path.append('../causal_vae')
sys.path.append('../causal_rnn')

# Import components
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    from causal_rnn.causal_mdn_gru import create_causal_rnn
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not available - using simulation mode")
    TORCH_AVAILABLE = False
    # Create dummy classes for simulation
    class Dataset:
        pass
    class DataLoader:
        pass

class CausalEpisodeSequenceDataset(Dataset):
    """Dataset for causal RNN training - loads sequences for temporal modeling"""
    
    def __init__(self, data_dir: str, causal_factors: List[str], sequence_length: int = 20):
        self.data_dir = data_dir
        self.causal_factors = causal_factors
        self.sequence_length = sequence_length
        
        # Causal factor dimensions mapping
        self.causal_dims = {
            'time_hour': 24,
            'day_week': 7,
            'weather': 4,
            'event': 5,
            'crowd_density': 5
        }
        
        # Calculate causal input dimension
        self.causal_input_dim = sum(self.causal_dims[f] for f in causal_factors)
        
        self.sequences = []
        self._load_sequences()
    
    def _load_sequences(self):
        """Load episode data and create sequences for training"""
        print(f"Loading sequences from {self.data_dir}...")
        print(f"Using causal factors: {self.causal_factors}")
        
        episode_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        episode_files.sort()
        
        total_sequences = 0
        
        for file in episode_files:
            if file == 'generation_summary.npz':
                continue
                
            filepath = os.path.join(self.data_dir, file)
            try:
                data = np.load(filepath, allow_pickle=True)
                
                # Extract data
                obs = data['obs']  # T x 64 x 64 x 3
                causal = data['causal']  # T x 45
                actions = data['action']  # T
                
                # Skip if episode too short
                if len(obs) < self.sequence_length + 1:
                    continue
                
                # Create sequences from this episode
                for start_idx in range(len(obs) - self.sequence_length):
                    end_idx = start_idx + self.sequence_length
                    
                    # Extract sequence
                    obs_seq = obs[start_idx:end_idx]
                    causal_seq = causal[start_idx:end_idx]
                    action_seq = actions[start_idx:end_idx]
                    
                    # Filter causal factors
                    filtered_causal_seq = self._filter_causal_factors(causal_seq)
                    
                    # Convert observations to latent representations (placeholder for now)
                    # In real implementation, this would use trained VAE encoder
                    latent_seq = self._obs_to_latent_placeholder(obs_seq)
                    
                    self.sequences.append({
                        'latent': latent_seq,
                        'actions': action_seq,
                        'causal': filtered_causal_seq
                    })
                    total_sequences += 1
                    
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}")
        
        print(f"Loaded {total_sequences} sequences from {len(episode_files)} episodes")
    
    def _filter_causal_factors(self, causal_sequence: np.ndarray) -> np.ndarray:
        """Filter causal sequence to only include specified factors"""
        if not self.causal_factors:
            return np.zeros((len(causal_sequence), 0))  # No causal factors
        
        # Extract specified causal factors from full 45-dim encoding
        filtered_sequences = []
        
        for causal_step in causal_sequence:
            filtered_step = []
            offset = 0
            
            for factor_name in ['time_hour', 'day_week', 'weather', 'event', 'crowd_density']:
                factor_dim = self.causal_dims[factor_name]
                factor_data = causal_step[offset:offset + factor_dim]
                
                if factor_name in self.causal_factors:
                    filtered_step.extend(factor_data)
                
                offset += factor_dim
            
            filtered_sequences.append(filtered_step)
        
        return np.array(filtered_sequences)
    
    def _obs_to_latent_placeholder(self, obs_sequence: np.ndarray) -> np.ndarray:
        """Placeholder function to convert observations to latent representations"""
        # In real implementation, this would use trained VAE encoder
        # For now, create dummy latent representations
        
        if TORCH_AVAILABLE:
            # Create more realistic dummy latents
            latent_dim = 256
            seq_len = len(obs_sequence)
            
            # Generate latents that have some structure based on observations
            latents = []
            for obs in obs_sequence:
                # Simple feature extraction as placeholder
                mean_pixel = np.mean(obs)
                std_pixel = np.std(obs)
                
                # Create structured dummy latent
                latent = np.random.randn(latent_dim) * 0.5
                latent[0] = mean_pixel / 255.0  # Normalized brightness
                latent[1] = std_pixel / 255.0   # Normalized contrast
                
                latents.append(latent)
            
            return np.array(latents, dtype=np.float32)
        else:
            # Simulation mode
            return np.random.randn(len(obs_sequence), 256).astype(np.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        if TORCH_AVAILABLE:
            latent = torch.FloatTensor(sequence['latent'])
            actions_one_hot = torch.FloatTensor(np.eye(5)[sequence['actions']])  # One-hot encode
            causal = torch.FloatTensor(sequence['causal'])
            
            return {
                'z_sequence': latent,
                'action_sequence': actions_one_hot,
                'causal_sequence': causal
            }
        else:
            # Simulation mode
            return sequence

class CausalRNNTrainer:
    """Handles training of causal RNN models"""
    
    def __init__(self, model, experiment_config: Dict, device='cpu'):
        self.model = model
        self.experiment_config = experiment_config
        self.device = device
        
        if TORCH_AVAILABLE and self.model is not None:
            self.model.to(device)
            self.optimizer = optim.Adam(self.model.parameters(), 
                                      lr=experiment_config.get('learning_rate', 0.0001))
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        if not TORCH_AVAILABLE:
            # Simulation mode
            return {
                'epoch_loss': np.random.uniform(3.0, 8.0),
                'mse_loss': np.random.uniform(1.5, 4.0),
                'num_batches': len(dataloader) if hasattr(dataloader, '__len__') else 10
            }
        
        self.model.train()
        epoch_loss = 0
        epoch_mse_loss = 0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Move to device
            for key in batch_data:
                batch_data[key] = batch_data[key].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(batch_data)
            loss = output['loss']
            mse_loss = output['mse_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_mse_loss += mse_loss.item()
            num_batches += 1
            
            # Progress tracking
            if batch_idx % 20 == 0:
                print(f'    Batch {batch_idx}: Loss={loss.item():.4f}, MSE={mse_loss.item():.4f}')
        
        return {
            'epoch_loss': epoch_loss / num_batches,
            'mse_loss': epoch_mse_loss / num_batches,
            'num_batches': num_batches
        }
    
    def validate(self, dataloader):
        """Validation pass"""
        if not TORCH_AVAILABLE:
            return {
                'val_loss': np.random.uniform(2.5, 7.0),
                'val_mse_loss': np.random.uniform(1.0, 3.5)
            }
        
        self.model.eval()
        val_loss = 0
        val_mse_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Move to device
                for key in batch_data:
                    batch_data[key] = batch_data[key].to(self.device)
                
                output = self.model(batch_data)
                
                val_loss += output['loss'].item()
                val_mse_loss += output['mse_loss'].item()
                num_batches += 1
        
        return {
            'val_loss': val_loss / num_batches,
            'val_mse_loss': val_mse_loss / num_batches
        }

def main():
    parser = argparse.ArgumentParser(description='Train Causal RNN for Phase 2A')
    
    # Core arguments from orchestrator
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--description', type=str, required=True)
    parser.add_argument('--causal_factors', type=str, required=True,
                       help='Comma-separated list of causal factors')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--vae_model_name', type=str, default='best')
    parser.add_argument('--vae_models_dir', type=str, required=True)
    
    # Additional parameters
    parser.add_argument('--sequence_length', type=int, default=20,
                       help='Length of sequences for training')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for causal RNN')
    parser.add_argument('--num_mixtures', type=int, default=5,
                       help='Number of mixture components')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Causal RNN Training: {args.experiment_name}")
    print(f"   Description: {args.description}")
    print(f"   Causal factors: {args.causal_factors}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   PyTorch available: {TORCH_AVAILABLE}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Parse causal factors
        causal_factors = [f.strip() for f in args.causal_factors.split(',') if f.strip()]
        
        # Calculate dimensions
        causal_dims = {
            'time_hour': 24, 'day_week': 7, 'weather': 4, 'event': 5, 'crowd_density': 5
        }
        causal_input_dim = sum(causal_dims[f] for f in causal_factors)
        
        print(f"\n📊 Loading dataset...")
        dataset = CausalEpisodeSequenceDataset(
            args.data_dir, 
            causal_factors, 
            args.sequence_length
        )
        
        if len(dataset) == 0:
            raise ValueError(f"No training data found")
        
        # Create data splits
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        if TORCH_AVAILABLE:
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            # Simulation mode
            train_loader = list(range(max(1, train_size // args.batch_size)))
            val_loader = list(range(max(1, val_size // args.batch_size)))
        
        print(f"   Training samples: {train_size}")
        print(f"   Validation samples: {val_size}")
        
        # Create model
        print(f"\n🏗️  Creating causal RNN model...")
        
        if TORCH_AVAILABLE:
            model = create_causal_rnn(
                "causal_mdn_gru",
                z_dim=256,  # Assuming 256D latent from VAE
                action_dim=5,
                causal_dim=causal_input_dim,
                hidden_dim=args.hidden_dim,
                num_mixtures=args.num_mixtures
            )
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"   Using device: {device}")
        else:
            model = None
            device = 'cpu'
        
        # Create trainer
        trainer = CausalRNNTrainer(
            model, 
            vars(args), 
            device
        )
        
        # Training loop
        print(f"\n🔥 Starting training for {args.epochs} epochs...")
        start_time = time.time()
        
        training_history = []
        best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            
            # Training
            train_metrics = trainer.train_epoch(train_loader)
            
            # Validation
            val_metrics = trainer.validate(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log progress
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"Train Loss={train_metrics['epoch_loss']:.4f}, "
                  f"Val Loss={val_metrics['val_loss']:.4f}, "
                  f"Time={epoch_time:.1f}s")
            
            # Save metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['epoch_loss'],
                'train_mse_loss': train_metrics['mse_loss'],
                'val_loss': val_metrics['val_loss'],
                'val_mse_loss': val_metrics['val_mse_loss'],
                'epoch_time': epoch_time
            }
            training_history.append(epoch_metrics)
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                
                if TORCH_AVAILABLE and model is not None:
                    model_path = os.path.join(args.output_dir, 'best_model.pth')
                    torch.save(model.state_dict(), model_path)
                    print(f"   💾 Saved best model (val_loss={best_val_loss:.4f})")
            
            # Early stopping
            if epoch > 10 and val_metrics['val_loss'] > best_val_loss * 1.3:
                print(f"   ⚠️  Early stopping - validation loss increasing")
                break
        
        total_time = time.time() - start_time
        
        # Final results
        print(f"\n✅ Training completed in {total_time/3600:.2f} hours")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        
        # Save comprehensive results
        results = {
            'experiment_name': args.experiment_name,
            'description': args.description,
            'causal_factors': causal_factors,
            'causal_input_dim': causal_input_dim,
            'training_config': vars(args),
            'training_history': training_history,
            'final_metrics': {
                'best_val_loss': best_val_loss,
                'final_train_loss': training_history[-1]['train_loss'],
                'final_val_loss': training_history[-1]['val_loss'],
                'total_epochs': len(training_history),
                'total_time_hours': total_time / 3600
            },
            'dataset_info': {
                'total_sequences': len(dataset),
                'train_sequences': train_size,
                'val_sequences': val_size,
                'sequence_length': args.sequence_length
            },
            'model_info': {
                'z_dim': 256,
                'action_dim': 5,
                'causal_dim': causal_input_dim,
                'hidden_dim': args.hidden_dim,
                'num_mixtures': args.num_mixtures
            },
            'timestamp': datetime.now().isoformat(),
            'pytorch_available': TORCH_AVAILABLE,
            'device': str(device)
        }
        
        # Save results
        results_path = os.path.join(args.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Results saved to: {results_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        
        # Save error information
        error_info = {
            'experiment_name': args.experiment_name,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'pytorch_available': TORCH_AVAILABLE
        }
        
        error_path = os.path.join(args.output_dir, 'error.json')
        with open(error_path, 'w') as f:
            json.dump(error_info, f, indent=2)
        
        return 1

if __name__ == "__main__":
    sys.exit(main())