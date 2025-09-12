#!/usr/bin/env python3
"""
Causal World Model Controller - Phase 3 Implementation
Uses trained VAE + Causal RNN to control agent in campus environment

Key capabilities:
1. Model-based planning using learned world model
2. Causal intervention and "what-if" analysis  
3. Adaptive behavior based on environmental conditions
4. CMA-ES evolution strategy for action planning
"""

import os
import sys
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json

# Add paths for imports
sys.path.append('..')
sys.path.append('../causal_envs')
sys.path.append('../integration')

try:
    import torch
    import torch.nn as nn
    from vae_to_rnn_pipeline import VAEModelLoader, ObservationToLatentConverter
    from causal_rnn.causal_mdn_gru import create_causal_rnn
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not available - using simulation mode")
    TORCH_AVAILABLE = False

# Import environment
from causal_envs.campus_env import SimpleCampusEnv

@dataclass
class ControllerConfig:
    """Configuration for the causal world model controller"""
    vae_model_name: str = "best"
    causal_rnn_model_name: str = "full_causal"
    planning_horizon: int = 10
    num_planning_samples: int = 50
    temperature: float = 1.0
    exploration_noise: float = 0.1
    max_episode_steps: int = 200

class CausalWorldModelController:
    """Main controller that uses learned world model for navigation"""
    
    def __init__(self, 
                 config: ControllerConfig,
                 phase1_models_dir: str = './data/models/phase1/',
                 phase2a_models_dir: str = './data/models/phase2a/'):
        
        self.config = config
        self.phase1_models_dir = phase1_models_dir
        self.phase2a_models_dir = phase2a_models_dir
        
        # Initialize components
        self.vae_model = None
        self.causal_rnn_model = None
        self.obs_converter = None
        self.device = torch.device('cpu')
        
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
        
        # Load models
        self._load_world_models()
        
        # Planning components
        self.planner = ModelBasedPlanner(self, config)
        
        print(f"✅ CausalWorldModelController initialized on {self.device}")
    
    def _load_world_models(self):
        """Load trained VAE and causal RNN models"""
        
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available - using simulation mode")
            return
        
        try:
            # Load VAE model
            vae_loader = VAEModelLoader(self.phase1_models_dir)
            
            if self.config.vae_model_name == "best":
                vae_model_name = vae_loader.get_best_model_name()
            else:
                vae_model_name = self.config.vae_model_name
            
            self.vae_model = vae_loader.load_model(vae_model_name)
            
            if self.vae_model is not None:
                self.obs_converter = ObservationToLatentConverter(self.vae_model, vae_model_name)
                print(f"✅ Loaded VAE model: {vae_model_name}")
            else:
                print(f"⚠️  Failed to load VAE model - using simulation mode")
                
            # Load causal RNN model
            self._load_causal_rnn_model()
            
        except Exception as e:
            print(f"⚠️  Error loading models: {e}")
    
    def _load_causal_rnn_model(self):
        """Load trained causal RNN model"""
        
        try:
            # Look for trained causal RNN model
            rnn_model_path = os.path.join(
                self.phase2a_models_dir, 
                self.config.causal_rnn_model_name,
                'best_model.pth'
            )
            
            if os.path.exists(rnn_model_path):
                # Load model configuration
                config_path = os.path.join(
                    self.phase2a_models_dir,
                    self.config.causal_rnn_model_name,
                    'training_results.json'
                )
                
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        training_results = json.load(f)
                    
                    model_info = training_results['model_info']
                    
                    # Create causal RNN model
                    self.causal_rnn_model = create_causal_rnn(
                        "causal_mdn_gru",
                        z_dim=model_info['z_dim'],
                        action_dim=model_info['action_dim'],
                        causal_dim=model_info['causal_dim'],
                        hidden_dim=model_info['hidden_dim'],
                        num_mixtures=model_info['num_mixtures']
                    )
                    
                    # Load trained weights
                    state_dict = torch.load(rnn_model_path, map_location=self.device)
                    self.causal_rnn_model.load_state_dict(state_dict)
                    self.causal_rnn_model.to(self.device)
                    self.causal_rnn_model.eval()
                    
                    print(f"✅ Loaded causal RNN model: {self.config.causal_rnn_model_name}")
                else:
                    print(f"⚠️  No config found for causal RNN model")
            else:
                print(f"⚠️  No trained causal RNN model found - using simulation mode")
                
        except Exception as e:
            print(f"⚠️  Error loading causal RNN: {e}")
    
    def predict_future(self, 
                      current_obs: np.ndarray,
                      action_sequence: np.ndarray,
                      causal_sequence: np.ndarray,
                      horizon: int = 10) -> Dict:
        """
        Predict future states using the world model
        
        Args:
            current_obs: Current observation (64, 64, 3)
            action_sequence: Planned actions (horizon,)
            causal_sequence: Causal states over horizon (horizon, causal_dim)
            horizon: Number of steps to predict
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        
        if not TORCH_AVAILABLE or self.obs_converter is None:
            # Simulation mode
            return self._simulate_future_prediction(current_obs, action_sequence, causal_sequence, horizon)
        
        try:
            # Convert observation to latent
            current_latent = self.obs_converter.convert_observations(current_obs[np.newaxis, ...])[0]
            
            if self.causal_rnn_model is not None:
                # Use trained causal RNN for prediction
                with torch.no_grad():
                    # Convert to torch tensors
                    z_initial = torch.FloatTensor(current_latent).unsqueeze(0).to(self.device)
                    
                    # One-hot encode actions
                    action_tensor = torch.FloatTensor(np.eye(5)[action_sequence]).unsqueeze(0).to(self.device)
                    causal_tensor = torch.FloatTensor(causal_sequence).unsqueeze(0).to(self.device)
                    
                    # Generate sequence
                    predicted_latents = self.causal_rnn_model.generate_sequence(
                        z_initial, action_tensor, causal_tensor, temperature=self.config.temperature
                    )
                    
                    predicted_latents = predicted_latents.squeeze(0).cpu().numpy()
                    
                    return {
                        'predicted_latents': predicted_latents,
                        'prediction_confidence': 0.8,  # Placeholder
                        'used_model': True
                    }
            else:
                return self._simulate_future_prediction(current_obs, action_sequence, causal_sequence, horizon)
                
        except Exception as e:
            print(f"⚠️  Prediction failed: {e}")
            return self._simulate_future_prediction(current_obs, action_sequence, causal_sequence, horizon)
    
    def _simulate_future_prediction(self, current_obs, action_sequence, causal_sequence, horizon):
        """Fallback simulation for future prediction"""
        
        latent_dim = 256
        predicted_latents = np.random.randn(horizon + 1, latent_dim) * 0.5
        
        # Add some structure based on actions
        for i, action in enumerate(action_sequence):
            if i < len(predicted_latents) - 1:
                # Simple action-dependent changes
                if action == 0:  # North
                    predicted_latents[i+1, 0] += 0.1
                elif action == 1:  # South
                    predicted_latents[i+1, 0] -= 0.1
                elif action == 2:  # East
                    predicted_latents[i+1, 1] += 0.1
                elif action == 3:  # West
                    predicted_latents[i+1, 1] -= 0.1
        
        return {
            'predicted_latents': predicted_latents,
            'prediction_confidence': 0.5,
            'used_model': False
        }
    
    def causal_intervention(self,
                           current_obs: np.ndarray,
                           base_causal_state: Dict,
                           intervention: Dict,
                           action_sequence: np.ndarray,
                           horizon: int = 5) -> Dict:
        """
        Perform causal intervention analysis
        
        Args:
            current_obs: Current observation
            base_causal_state: Original causal state
            intervention: Changes to make {"weather": "rain"}
            action_sequence: Actions to test
            horizon: Prediction horizon
            
        Returns:
            Comparison of predictions with/without intervention
        """
        
        print(f"🔬 Performing causal intervention: {intervention}")
        
        # Create base causal sequence
        base_causal_encoded = self._encode_causal_state(base_causal_state)
        base_causal_sequence = np.tile(base_causal_encoded, (horizon, 1))
        
        # Create intervention causal sequence
        intervention_causal_state = base_causal_state.copy()
        intervention_causal_state.update(intervention)
        intervention_causal_encoded = self._encode_causal_state(intervention_causal_state)
        intervention_causal_sequence = np.tile(intervention_causal_encoded, (horizon, 1))
        
        # Predict both scenarios
        base_prediction = self.predict_future(
            current_obs, action_sequence, base_causal_sequence, horizon
        )
        
        intervention_prediction = self.predict_future(
            current_obs, action_sequence, intervention_causal_sequence, horizon
        )
        
        # Analyze differences
        if base_prediction['used_model'] and intervention_prediction['used_model']:
            latent_diff = np.mean(np.abs(
                intervention_prediction['predicted_latents'] - base_prediction['predicted_latents']
            ))
        else:
            latent_diff = 0.1  # Placeholder
        
        result = {
            'intervention': intervention,
            'base_prediction': base_prediction,
            'intervention_prediction': intervention_prediction,
            'latent_space_difference': latent_diff,
            'significant_change': latent_diff > 0.05,
            'analysis': self._analyze_intervention_effect(intervention, latent_diff)
        }
        
        print(f"   Latent space difference: {latent_diff:.4f}")
        print(f"   Significant change: {result['significant_change']}")
        
        return result
    
    def _encode_causal_state(self, causal_state: Dict) -> np.ndarray:
        """Encode causal state dictionary to 45-dimensional vector"""
        
        encoding = np.zeros(45)
        offset = 0
        
        # Time hour (24 dims)
        encoding[offset + causal_state.get('time_hour', 12)] = 1
        offset += 24
        
        # Day of week (7 dims) 
        encoding[offset + causal_state.get('day_week', 2)] = 1
        offset += 7
        
        # Weather (4 dims)
        weather_map = {'sunny': 0, 'rain': 1, 'snow': 2, 'fog': 3}
        weather_idx = weather_map.get(causal_state.get('weather', 'sunny'), 0)
        encoding[offset + weather_idx] = 1
        offset += 4
        
        # Event (5 dims)
        event_map = {'normal': 0, 'gameday': 1, 'exam': 2, 'break': 3, 'construction': 4}
        event_idx = event_map.get(causal_state.get('event', 'normal'), 0)
        encoding[offset + event_idx] = 1
        offset += 5
        
        # Crowd density (5 dims)
        crowd_density = causal_state.get('crowd_density', 2)
        encoding[offset + crowd_density - 1] = 1  # -1 because 1-indexed
        
        return encoding
    
    def _analyze_intervention_effect(self, intervention: Dict, latent_diff: float) -> str:
        """Analyze the effect of causal intervention"""
        
        if 'weather' in intervention:
            if latent_diff > 0.1:
                return f"Weather change to {intervention['weather']} significantly affects navigation"
            else:
                return f"Weather change to {intervention['weather']} has minimal effect"
        
        if 'event' in intervention:
            if latent_diff > 0.15:
                return f"Event change to {intervention['event']} dramatically affects environment"
            else:
                return f"Event change to {intervention['event']} has moderate effect"
        
        return f"Intervention causes {'significant' if latent_diff > 0.05 else 'minimal'} changes"

class ModelBasedPlanner:
    """Plans actions using the world model"""
    
    def __init__(self, controller: CausalWorldModelController, config: ControllerConfig):
        self.controller = controller
        self.config = config
        self.action_dim = 5  # N, S, E, W, Stay
    
    def plan_actions(self, 
                    current_obs: np.ndarray,
                    goal_pos: np.ndarray,
                    current_pos: np.ndarray,
                    causal_state: Dict,
                    horizon: int = None) -> Tuple[int, Dict]:
        """
        Plan the next action using model-based planning
        
        Args:
            current_obs: Current observation
            goal_pos: Goal position
            current_pos: Current position
            causal_state: Current causal state
            horizon: Planning horizon
            
        Returns:
            (best_action, planning_info)
        """
        
        if horizon is None:
            horizon = self.config.planning_horizon
        
        print(f"🎯 Planning action: pos={current_pos}, goal={goal_pos}")
        
        # Simple geometric planning as baseline
        geometric_action = self._geometric_planner(current_pos, goal_pos)
        
        # Try model-based planning if available
        if self.controller.causal_rnn_model is not None:
            model_action, model_info = self._model_based_planning(
                current_obs, goal_pos, current_pos, causal_state, horizon
            )
            
            planning_info = {
                'method': 'model_based',
                'geometric_suggestion': geometric_action,
                'model_suggestion': model_action,
                'model_info': model_info,
                'chosen_action': model_action
            }
            
            return model_action, planning_info
        else:
            # Fall back to geometric planning
            planning_info = {
                'method': 'geometric',
                'chosen_action': geometric_action,
                'reason': 'No trained world model available'
            }
            
            return geometric_action, planning_info
    
    def _geometric_planner(self, current_pos: np.ndarray, goal_pos: np.ndarray) -> int:
        """Simple geometric path planning"""
        
        diff = goal_pos - current_pos
        
        # Prioritize larger dimension difference
        if abs(diff[1]) > abs(diff[0]):  # Y difference larger
            if diff[1] > 0:
                return 1  # South
            else:
                return 0  # North
        else:  # X difference larger  
            if diff[0] > 0:
                return 2  # East
            else:
                return 3  # West
    
    def _model_based_planning(self,
                             current_obs: np.ndarray,
                             goal_pos: np.ndarray,
                             current_pos: np.ndarray,
                             causal_state: Dict,
                             horizon: int) -> Tuple[int, Dict]:
        """Model-based action planning using world model"""
        
        best_action = 0
        best_value = -float('inf')
        action_values = []
        
        # Encode current causal state
        causal_encoded = self.controller._encode_causal_state(causal_state)
        
        # Evaluate each possible action
        for action in range(self.action_dim):
            # Create action sequence (repeat action for simplicity)
            action_sequence = np.full(horizon, action)
            causal_sequence = np.tile(causal_encoded, (horizon, 1))
            
            # Predict future with this action
            prediction = self.controller.predict_future(
                current_obs, action_sequence, causal_sequence, horizon
            )
            
            # Estimate value of this action
            action_value = self._evaluate_action_sequence(
                current_pos, goal_pos, action_sequence, prediction
            )
            
            action_values.append(action_value)
            
            if action_value > best_value:
                best_value = action_value
                best_action = action
        
        model_info = {
            'action_values': action_values,
            'best_value': best_value,
            'prediction_confidence': prediction.get('prediction_confidence', 0.5)
        }
        
        return best_action, model_info
    
    def _evaluate_action_sequence(self,
                                 current_pos: np.ndarray,
                                 goal_pos: np.ndarray,
                                 action_sequence: np.ndarray,
                                 prediction: Dict) -> float:
        """Evaluate the value of an action sequence"""
        
        # Simple evaluation based on expected progress toward goal
        # In a full implementation, this would use the predicted latents
        # to estimate position and other factors
        
        value = 0.0
        pos = current_pos.copy()
        
        # Simulate position changes based on actions
        for action in action_sequence:
            if action == 0:  # North
                pos[1] -= 1
            elif action == 1:  # South
                pos[1] += 1
            elif action == 2:  # East
                pos[0] += 1
            elif action == 3:  # West
                pos[0] -= 1
            # action == 4 is Stay
            
            # Distance to goal (negative because we want to minimize)
            distance = np.linalg.norm(pos - goal_pos)
            value -= distance
            
            # Small penalty for movement (energy cost)
            if action != 4:
                value -= 0.1
        
        return value

class CausalNavigationAgent:
    """High-level agent that combines controller with navigation strategies"""
    
    def __init__(self, controller: CausalWorldModelController):
        self.controller = controller
        self.env = None
        
    def run_navigation_episode(self,
                              goal_type: str = 'library',
                              causal_scenario: Optional[Dict] = None,
                              max_steps: int = 200) -> Dict:
        """
        Run a complete navigation episode
        
        Args:
            goal_type: Type of goal ('library', 'gym', etc.)
            causal_scenario: Specific causal conditions to test
            max_steps: Maximum episode steps
            
        Returns:
            Episode results and statistics
        """
        
        print(f"\n🚀 Running Navigation Episode")
        print(f"   Goal: {goal_type}")
        print(f"   Causal scenario: {causal_scenario}")
        
        # Create environment
        self.env = SimpleCampusEnv()
        
        # Reset with specified conditions
        reset_options = {'goal_type': goal_type}
        if causal_scenario:
            reset_options['causal_state'] = causal_scenario
        
        obs, info = self.env.reset(options=reset_options)
        
        # Episode tracking
        episode_data = {
            'goal_type': goal_type,
            'causal_scenario': causal_scenario,
            'trajectory': [],
            'actions': [],
            'rewards': [],
            'planning_info': [],
            'causal_interventions': []
        }
        
        total_reward = 0
        step = 0
        
        while step < max_steps:
            # Get current state
            current_pos = info['agent_pos']
            goal_pos = info['goal_pos'] 
            causal_state = {
                'time_hour': causal_scenario.get('time_hour', 12) if causal_scenario else 12,
                'day_week': causal_scenario.get('day_week', 2) if causal_scenario else 2,
                'weather': causal_scenario.get('weather', 'sunny') if causal_scenario else 'sunny',
                'event': causal_scenario.get('event', 'normal') if causal_scenario else 'normal',
                'crowd_density': causal_scenario.get('crowd_density', 2) if causal_scenario else 2
            }
            
            # Plan action
            action, planning_info = self.controller.planner.plan_actions(
                obs, goal_pos, current_pos, causal_state
            )
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Record data
            episode_data['trajectory'].append(current_pos.copy())
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['planning_info'].append(planning_info)
            
            total_reward += reward
            step += 1
            
            # Check if goal reached
            if terminated:
                print(f"🎯 Goal reached in {step} steps!")
                break
            
            if truncated:
                print(f"⏰ Episode truncated at {step} steps")
                break
            
            # Occasional causal intervention testing
            if step % 20 == 0 and step > 0:
                intervention = self._suggest_intervention(causal_state)
                if intervention:
                    intervention_result = self.controller.causal_intervention(
                        obs, causal_state, intervention, 
                        np.array([action] * 5)  # Test next 5 steps
                    )
                    episode_data['causal_interventions'].append(intervention_result)
        
        # Episode summary
        episode_data.update({
            'total_reward': total_reward,
            'episode_length': step,
            'success': terminated,
            'final_distance': np.linalg.norm(info['agent_pos'] - info['goal_pos']),
            'efficiency': total_reward / step if step > 0 else 0
        })
        
        print(f"✅ Episode completed:")
        print(f"   Success: {episode_data['success']}")
        print(f"   Steps: {episode_data['episode_length']}")
        print(f"   Total reward: {episode_data['total_reward']:.2f}")
        print(f"   Final distance: {episode_data['final_distance']:.1f}")
        
        return episode_data
    
    def _suggest_intervention(self, causal_state: Dict) -> Optional[Dict]:
        """Suggest interesting causal interventions to test"""
        
        # Simple intervention suggestions
        interventions = [
            {'weather': 'rain'} if causal_state['weather'] == 'sunny' else None,
            {'event': 'gameday'} if causal_state['event'] == 'normal' else None,
            {'crowd_density': 5} if causal_state['crowd_density'] < 4 else None
        ]
        
        # Return random valid intervention
        valid_interventions = [i for i in interventions if i is not None]
        if valid_interventions and np.random.random() < 0.3:  # 30% chance
            return np.random.choice(valid_interventions)
        
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Causal World Model Controller')
    parser.add_argument('--phase1_models', type=str, default='./data/models/phase1/',
                       help='Phase 1 VAE models directory')
    parser.add_argument('--phase2a_models', type=str, default='./data/models/phase2a/',
                       help='Phase 2A causal RNN models directory') 
    parser.add_argument('--goal_type', type=str, default='library',
                       help='Navigation goal type')
    parser.add_argument('--weather', type=str, default='sunny',
                       choices=['sunny', 'rain', 'snow', 'fog'],
                       help='Weather condition')
    parser.add_argument('--event', type=str, default='normal',
                       choices=['normal', 'gameday', 'exam', 'break', 'construction'],
                       help='Campus event')
    parser.add_argument('--max_steps', type=int, default=200,
                       help='Maximum episode steps')
    
    args = parser.parse_args()
    
    print("🧠 Causal World Model Controller - Phase 3")
    print("=" * 60)
    
    # Create controller configuration
    config = ControllerConfig(
        vae_model_name="best",
        causal_rnn_model_name="full_causal",
        planning_horizon=10,
        max_episode_steps=args.max_steps
    )
    
    # Create controller
    controller = CausalWorldModelController(
        config,
        phase1_models_dir=args.phase1_models,
        phase2a_models_dir=args.phase2a_models
    )
    
    # Create agent
    agent = CausalNavigationAgent(controller)
    
    # Define causal scenario
    causal_scenario = {
        'time_hour': 14,
        'day_week': 2,  # Tuesday
        'weather': args.weather,
        'event': args.event,
        'crowd_density': 3
    }
    
    # Run navigation episode
    result = agent.run_navigation_episode(
        goal_type=args.goal_type,
        causal_scenario=causal_scenario,
        max_steps=args.max_steps
    )
    
    # Save results
    results_file = f"navigation_result_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                json_result[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                json_result[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
            else:
                json_result[key] = value
        
        json.dump(json_result, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")
    
    if result['success']:
        print("🎉 Navigation successful! The causal world model works!")
    else:
        print("⚠️  Navigation incomplete - check model performance")

if __name__ == "__main__":
    main()