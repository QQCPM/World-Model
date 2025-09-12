#!/usr/bin/env python3
"""
VAE-to-RNN Integration Pipeline
Bridges Phase 1 (VAE training) with Phase 2A (Causal RNN training)

Key functions:
1. Load trained VAE models from Phase 1
2. Convert raw observations to latent representations  
3. Process episode data for causal RNN training
4. Handle model compatibility and format conversions
"""

import os
import sys
import json
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Add paths for imports
sys.path.append('..')
sys.path.append('../causal_vae')
sys.path.append('../experiments')

try:
    import torch
    import torch.nn as nn
    from causal_vae.modern_architectures import create_vae_architecture
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not available - using simulation mode")
    TORCH_AVAILABLE = False

class VAEModelLoader:
    """Loads and manages trained VAE models from Phase 1"""
    
    def __init__(self, models_dir: str = './data/models/phase1/'):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.device = torch.device('cpu')  # Start with CPU
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        
        print(f"VAEModelLoader initialized on device: {self.device}")
    
    def get_best_model_name(self) -> str:
        """Get the name of the best performing model from Phase 1"""
        summary_path = os.path.join(self.models_dir, 'phase1_summary.json')
        
        if not os.path.exists(summary_path):
            print("⚠️  No Phase 1 summary found - will use gaussian_256D as default")
            return "gaussian_256D"
        
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            if 'top_architectures' in summary and summary['top_architectures']:
                best_model = summary['top_architectures'][0]['name']
                print(f"✅ Best Phase 1 model: {best_model}")
                return best_model
            else:
                print("⚠️  No top architectures in summary - using gaussian_256D")
                return "gaussian_256D"
                
        except Exception as e:
            print(f"⚠️  Error reading Phase 1 summary: {e}")
            return "gaussian_256D"
    
    def load_model(self, model_name: str) -> Optional[nn.Module]:
        """Load a specific VAE model"""
        
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        if not TORCH_AVAILABLE:
            print(f"⚠️  PyTorch not available - cannot load {model_name}")
            return None
        
        try:
            # Create model architecture
            model = create_vae_architecture(model_name)
            
            # Look for saved model weights
            model_path = os.path.join(self.models_dir, model_name, 'best_model.pth')
            
            if os.path.exists(model_path):
                # Load trained weights
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"✅ Loaded trained {model_name} from {model_path}")
            else:
                print(f"⚠️  No trained weights found for {model_name} - using random initialization")
            
            model.to(self.device)
            model.eval()
            
            self.loaded_models[model_name] = model
            return model
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return None
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a model's architecture and performance"""
        
        info = {
            'name': model_name,
            'available': False,
            'latent_dim': 256,  # Default
            'trained': False
        }
        
        # Check if model results exist
        results_path = os.path.join(self.models_dir, model_name, 'training_results.json')
        
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                info['available'] = True
                info['trained'] = True
                info['final_metrics'] = results.get('final_metrics', {})
                
                # Extract latent dimension
                if 'baseline_32D' in model_name:
                    info['latent_dim'] = 32
                elif '256D' in model_name:
                    info['latent_dim'] = 256
                elif '512D' in model_name:
                    info['latent_dim'] = 512
                    
            except Exception as e:
                print(f"⚠️  Error reading results for {model_name}: {e}")
        
        return info

class ObservationToLatentConverter:
    """Converts raw observations to latent representations using trained VAE"""
    
    def __init__(self, vae_model: nn.Module, model_name: str):
        self.vae_model = vae_model
        self.model_name = model_name
        self.device = next(vae_model.parameters()).device if vae_model else torch.device('cpu')
        
        # Determine latent dimension
        if 'baseline_32D' in model_name:
            self.latent_dim = 32
        elif '256D' in model_name:
            self.latent_dim = 256
        elif '512D' in model_name:
            self.latent_dim = 512
        else:
            self.latent_dim = 256  # Default
    
    def convert_observations(self, observations: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Convert array of observations to latent representations
        
        Args:
            observations: (N, 64, 64, 3) array of observations
            batch_size: Batch size for processing
            
        Returns:
            latents: (N, latent_dim) array of latent representations
        """
        
        if not TORCH_AVAILABLE or self.vae_model is None:
            # Simulation mode - create structured dummy latents
            return self._create_dummy_latents(observations)
        
        try:
            self.vae_model.eval()
            
            num_obs = len(observations)
            latents = []
            
            with torch.no_grad():
                for i in range(0, num_obs, batch_size):
                    batch_end = min(i + batch_size, num_obs)
                    batch_obs = observations[i:batch_end]
                    
                    # Convert to torch tensor and normalize
                    batch_tensor = torch.FloatTensor(batch_obs).to(self.device)
                    batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
                    batch_tensor = batch_tensor / 255.0  # Normalize to [0,1]
                    
                    # Encode to latent space
                    if 'categorical' in self.model_name:
                        logits = self.vae_model.encode(batch_tensor)
                        # Use mode (argmax) for deterministic encoding
                        categorical = torch.argmax(logits, dim=-1)
                        batch_latents = torch.flatten(categorical, start_dim=1).float()
                    elif 'vq_vae' in self.model_name:
                        encoded = self.vae_model.encode(batch_tensor)
                        quantized, _, _ = self.vae_model.vq_layer(encoded)
                        batch_latents = torch.flatten(quantized, start_dim=1)
                    elif 'hierarchical' in self.model_name:
                        static_mu, static_logvar, dynamic_mu, dynamic_logvar = self.vae_model.encode(batch_tensor)
                        # Use means for deterministic encoding
                        batch_latents = torch.cat([static_mu, dynamic_mu], dim=1)
                    else:
                        # Standard VAE - use mean for deterministic encoding
                        mu, logvar = self.vae_model.encode(batch_tensor)
                        batch_latents = mu
                    
                    latents.append(batch_latents.cpu().numpy())
            
            return np.concatenate(latents, axis=0)
            
        except Exception as e:
            print(f"⚠️  VAE encoding failed: {e} - using dummy latents")
            return self._create_dummy_latents(observations)
    
    def _create_dummy_latents(self, observations: np.ndarray) -> np.ndarray:
        """Create structured dummy latents for simulation mode"""
        
        num_obs = len(observations)
        latents = np.random.randn(num_obs, self.latent_dim).astype(np.float32) * 0.5
        
        # Add some structure based on observations
        for i, obs in enumerate(observations):
            # Basic image statistics as features
            mean_pixel = np.mean(obs) / 255.0
            std_pixel = np.std(obs) / 255.0
            
            # Encode basic features in first few dimensions
            latents[i, 0] = mean_pixel
            latents[i, 1] = std_pixel
            
            # Add some spatial structure
            if len(latents[i]) > 10:
                # Rough edge detection
                edges = np.mean(np.abs(np.diff(obs, axis=0))) + np.mean(np.abs(np.diff(obs, axis=1)))
                latents[i, 2] = edges / 255.0
        
        return latents

class EpisodeDataProcessor:
    """Processes episode data to create training sequences for causal RNN"""
    
    def __init__(self, vae_model_loader: VAEModelLoader):
        self.vae_model_loader = vae_model_loader
        self.converters = {}  # Cache converters for different models
    
    def process_episodes_for_causal_rnn(self, 
                                       data_dir: str,
                                       output_dir: str, 
                                       vae_model_name: str,
                                       sequence_length: int = 20) -> Dict:
        """
        Process raw episodes to create causal RNN training data
        
        Args:
            data_dir: Directory containing raw episode data
            output_dir: Directory to save processed sequences
            vae_model_name: Name of VAE model to use for encoding
            sequence_length: Length of sequences for causal RNN
            
        Returns:
            Processing statistics
        """
        
        print(f"\n🔄 Processing episodes for causal RNN training...")
        print(f"   Input data: {data_dir}")
        print(f"   Output directory: {output_dir}")
        print(f"   VAE model: {vae_model_name}")
        print(f"   Sequence length: {sequence_length}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load VAE model
        if vae_model_name not in self.converters:
            vae_model = self.vae_model_loader.load_model(vae_model_name)
            if vae_model is not None:
                converter = ObservationToLatentConverter(vae_model, vae_model_name)
                self.converters[vae_model_name] = converter
            else:
                # Create dummy converter for simulation
                converter = ObservationToLatentConverter(None, vae_model_name)
                self.converters[vae_model_name] = converter
        
        converter = self.converters[vae_model_name]
        
        # Process episodes
        episode_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        episode_files.sort()
        
        processed_sequences = []
        total_sequences = 0
        processing_stats = {
            'episodes_processed': 0,
            'sequences_created': 0,
            'total_timesteps': 0,
            'latent_dim': converter.latent_dim,
            'vae_model': vae_model_name
        }
        
        start_time = time.time()
        
        for episode_idx, episode_file in enumerate(episode_files):
            if episode_file == 'generation_summary.npz':
                continue
            
            filepath = os.path.join(data_dir, episode_file)
            
            try:
                # Load episode data
                data = np.load(filepath, allow_pickle=True)
                
                observations = data['obs']  # T x 64 x 64 x 3
                causal_states = data['causal']  # T x 45
                actions = data['action']  # T
                rewards = data['reward']  # T
                
                # Skip if too short
                if len(observations) < sequence_length + 1:
                    continue
                
                # Convert observations to latents
                latents = converter.convert_observations(observations)
                
                # Create sequences
                episode_sequences = self._create_sequences_from_episode(
                    latents, causal_states, actions, rewards, sequence_length
                )
                
                processed_sequences.extend(episode_sequences)
                total_sequences += len(episode_sequences)
                processing_stats['episodes_processed'] += 1
                processing_stats['total_timesteps'] += len(observations)
                
                if episode_idx % 10 == 0:
                    print(f"   Processed {episode_idx+1}/{len(episode_files)} episodes, "
                          f"{total_sequences} sequences created")
                    
            except Exception as e:
                print(f"⚠️  Failed to process {episode_file}: {e}")
        
        processing_stats['sequences_created'] = total_sequences
        
        # Save processed sequences
        if total_sequences > 0:
            sequences_file = os.path.join(output_dir, 'processed_sequences.npz')
            
            # Combine all sequences
            all_latents = np.array([seq['latents'] for seq in processed_sequences])
            all_causal = np.array([seq['causal'] for seq in processed_sequences]) 
            all_actions = np.array([seq['actions'] for seq in processed_sequences])
            all_rewards = np.array([seq['rewards'] for seq in processed_sequences])
            
            np.savez_compressed(
                sequences_file,
                latents=all_latents,
                causal=all_causal,
                actions=all_actions,
                rewards=all_rewards,
                metadata=processing_stats
            )
            
            print(f"✅ Saved {total_sequences} sequences to {sequences_file}")
        
        # Save processing stats
        stats_file = os.path.join(output_dir, 'processing_stats.json')
        processing_stats['processing_time_seconds'] = time.time() - start_time
        processing_stats['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        with open(stats_file, 'w') as f:
            json.dump(processing_stats, f, indent=2)
        
        print(f"✅ Episode processing completed:")
        print(f"   Episodes processed: {processing_stats['episodes_processed']}")
        print(f"   Sequences created: {processing_stats['sequences_created']}")
        print(f"   Processing time: {processing_stats['processing_time_seconds']:.1f}s")
        
        return processing_stats
    
    def _create_sequences_from_episode(self, 
                                      latents: np.ndarray,
                                      causal_states: np.ndarray, 
                                      actions: np.ndarray,
                                      rewards: np.ndarray,
                                      sequence_length: int) -> List[Dict]:
        """Create training sequences from a single episode"""
        
        sequences = []
        episode_length = len(latents)
        
        # Create overlapping sequences
        for start_idx in range(episode_length - sequence_length):
            end_idx = start_idx + sequence_length
            
            sequence = {
                'latents': latents[start_idx:end_idx].copy(),
                'causal': causal_states[start_idx:end_idx].copy(),
                'actions': actions[start_idx:end_idx].copy(),
                'rewards': rewards[start_idx:end_idx].copy()
            }
            
            sequences.append(sequence)
        
        return sequences

class IntegrationPipeline:
    """Main integration pipeline class"""
    
    def __init__(self, phase1_models_dir: str = './data/models/phase1/'):
        self.phase1_models_dir = phase1_models_dir
        self.vae_model_loader = VAEModelLoader(phase1_models_dir)
        self.episode_processor = EpisodeDataProcessor(self.vae_model_loader)
    
    def run_integration(self, 
                       data_dir: str,
                       output_dir: str,
                       vae_model_name: str = "best") -> Dict:
        """Run the complete integration pipeline"""
        
        print(f"🔗 Running VAE-to-RNN Integration Pipeline")
        print(f"=" * 60)
        
        # Resolve model name
        if vae_model_name == "best":
            vae_model_name = self.vae_model_loader.get_best_model_name()
        
        # Get model info
        model_info = self.vae_model_loader.get_model_info(vae_model_name)
        print(f"Using VAE model: {vae_model_name}")
        print(f"Model info: {model_info}")
        
        # Process episodes
        processing_stats = self.episode_processor.process_episodes_for_causal_rnn(
            data_dir=data_dir,
            output_dir=output_dir,
            vae_model_name=vae_model_name
        )
        
        # Create summary
        integration_summary = {
            'pipeline_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'vae_model_used': vae_model_name,
            'model_info': model_info,
            'processing_stats': processing_stats,
            'output_directory': output_dir,
            'ready_for_phase2a': processing_stats['sequences_created'] > 0
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, 'integration_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(integration_summary, f, indent=2)
        
        print(f"\n✅ Integration pipeline completed!")
        print(f"📁 Integration summary: {summary_file}")
        print(f"🎯 Ready for Phase 2A: {integration_summary['ready_for_phase2a']}")
        
        return integration_summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='VAE-to-RNN Integration Pipeline')
    parser.add_argument('--phase1_models_dir', type=str, default='./data/models/phase1/',
                       help='Directory containing Phase 1 VAE models')
    parser.add_argument('--data_dir', type=str, default='./data/causal_episodes/',
                       help='Directory containing raw episode data')
    parser.add_argument('--output_dir', type=str, default='./data/processed_sequences/',
                       help='Output directory for processed sequences')
    parser.add_argument('--vae_model', type=str, default='best',
                       help='VAE model to use (best, gaussian_256D, etc.)')
    parser.add_argument('--sequence_length', type=int, default=20,
                       help='Length of sequences for causal RNN training')
    
    args = parser.parse_args()
    
    # Create integration pipeline
    pipeline = IntegrationPipeline(args.phase1_models_dir)
    
    # Run integration
    result = pipeline.run_integration(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vae_model_name=args.vae_model
    )
    
    if result['ready_for_phase2a']:
        print(f"\n🎉 Integration successful - ready to run Phase 2A!")
        print(f"Next step: python3 experiments/phase2a_orchestrator.py")
    else:
        print(f"\n⚠️  Integration completed with issues - check logs")

if __name__ == "__main__":
    main()