#!/usr/bin/env python3
"""
Causal Intervention Testing Framework
Scientific validation of genuine causal reasoning vs pattern matching

This framework tests three critical aspects of causal understanding:
1. Weather Intervention: Can models adapt when causal factors change mid-episode?
2. Factor Ablation: Which causal factors are most important for prediction?
3. Counterfactual Reasoning: Can models predict "what-if" scenarios?

Design Principles:
- Use REAL physics simulation as ground truth
- Test GENUINE causal understanding vs correlation learning
- Measure QUANTITATIVE adaptation and coherence metrics
- Validate against AUTHENTIC environment mechanics
"""

import torch
import numpy as np
import json
import time
import sys
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt

# Add project imports
sys.path.append('continuous_models')
sys.path.append('causal_envs')

from state_predictors import create_continuous_model, get_model_info
from continuous_campus_env import ContinuousCampusEnv, WeatherType, EventType, CausalState


class InterventionType(Enum):
    """Types of causal interventions for testing"""
    WEATHER_CHANGE = "weather_change"
    EVENT_CHANGE = "event_change"
    FACTOR_ABLATION = "factor_ablation"
    COUNTERFACTUAL = "counterfactual"


@dataclass
class InterventionSpec:
    """Specification for a causal intervention test"""
    intervention_type: InterventionType
    target_factor: str  # 'weather', 'event', 'crowd', 'time'
    intervention_timestep: int
    original_value: Union[float, str]
    intervention_value: Union[float, str]
    duration: int = -1  # -1 means permanent intervention


@dataclass
class InterventionResult:
    """Results from a causal intervention experiment"""
    intervention_spec: InterventionSpec
    pre_intervention_mse: float
    post_intervention_mse: float
    adaptation_rate: float
    causal_effect_magnitude: float
    model_prediction_shift: float
    physics_prediction_shift: float
    coherence_score: float


class CausalInterventionTester:
    """Framework for rigorous testing of causal understanding in world models"""

    def __init__(self, model_path: str, model_type: str, device: str = 'auto'):
        """
        Initialize the causal intervention tester

        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model (gru_dynamics, lstm_predictor, etc.)
            device: Device for testing ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_type = model_type
        self.model_path = model_path
        self.device = self._get_device(device)

        # Load trained model
        print(f"🧪 Loading {model_type} model from {model_path}")
        self.model = self._load_model()
        self.model.eval()

        # Initialize environment for ground truth generation
        self.env = ContinuousCampusEnv()

        # Map weather and event types to their numeric representations
        self.weather_mapping = {
            WeatherType.SUNNY: 0.0,
            WeatherType.RAIN: 1/3,
            WeatherType.SNOW: 2/3,
            WeatherType.FOG: 1.0
        }

        self.event_mapping = {
            EventType.NORMAL: 0.0,
            EventType.GAMEDAY: 0.25,
            EventType.EXAM: 0.5,
            EventType.BREAK: 0.75,
            EventType.CONSTRUCTION: 1.0
        }

        # Results storage
        self.intervention_results = []

        print(f"✅ Causal Intervention Tester initialized")
        print(f"   Model: {model_type}")
        print(f"   Device: {self.device}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters())}")

    def _get_device(self, device: str) -> torch.device:
        """Determine testing device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)

    def _load_model(self):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Get model kwargs from checkpoint
        model_kwargs = checkpoint.get('model_kwargs', {'hidden_dim': 64})

        # Create and load model
        model = create_continuous_model(self.model_type, **model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        return model

    def _causal_dict_to_vector(self, causal_dict: Dict) -> np.ndarray:
        """Convert causal state dictionary to numeric vector"""
        # Map weather string to numeric value
        weather_map = {
            'sunny': 0.0,
            'rain': 1/3,
            'snow': 2/3,
            'fog': 1.0
        }

        # Map event string to numeric value
        event_map = {
            'normal': 0.0,
            'gameday': 0.25,
            'exam': 0.5,
            'break': 0.75,
            'construction': 1.0
        }

        weather_norm = weather_map.get(causal_dict['weather'], 0.0)
        event_norm = event_map.get(causal_dict['event'], 0.0)
        crowd_norm = causal_dict['crowd_density']
        time_norm = causal_dict['time_hour'] / 23.0
        day_norm = causal_dict['day_week'] / 6.0

        return np.array([weather_norm, event_norm, crowd_norm, time_norm, day_norm], dtype=np.float32)

    def test_weather_interventions(self, num_episodes: int = 20,
                                 intervention_timestep: int = 40) -> List[InterventionResult]:
        """
        Test 1: Weather Intervention Testing
        Change weather mid-episode and measure model adaptation vs physics

        This tests whether the model truly understands causal relationships
        or just learned correlational patterns.
        """
        print(f"\n🌦️  WEATHER INTERVENTION TESTING")
        print(f"   Episodes: {num_episodes}")
        print(f"   Intervention at timestep: {intervention_timestep}")
        print(f"   Testing causal adaptation vs pattern matching")

        results = []

        # Test all weather transitions
        weather_pairs = [
            (WeatherType.SUNNY, WeatherType.RAIN),
            (WeatherType.SUNNY, WeatherType.SNOW),
            (WeatherType.RAIN, WeatherType.SUNNY),
            (WeatherType.RAIN, WeatherType.SNOW),
            (WeatherType.SNOW, WeatherType.SUNNY),
            (WeatherType.SNOW, WeatherType.RAIN)
        ]

        for original_weather, target_weather in weather_pairs:
            print(f"\n   Testing: {original_weather.value} → {target_weather.value}")

            episode_results = []

            for episode in range(num_episodes):
                # Generate baseline episode with consistent weather
                baseline_trajectory = self._generate_consistent_episode(
                    weather=original_weather,
                    episode_length=intervention_timestep * 2
                )

                # Generate intervention episode with weather change
                intervention_trajectory = self._generate_intervention_episode(
                    baseline_trajectory, intervention_timestep, 'weather', target_weather
                )

                # Test model predictions vs physics ground truth
                result = self._evaluate_intervention(
                    baseline_trajectory, intervention_trajectory, intervention_timestep
                )

                # Create intervention specification
                intervention_spec = InterventionSpec(
                    intervention_type=InterventionType.WEATHER_CHANGE,
                    target_factor='weather',
                    intervention_timestep=intervention_timestep,
                    original_value=original_weather.value,
                    intervention_value=target_weather.value
                )

                episode_results.append(InterventionResult(
                    intervention_spec=intervention_spec,
                    **result
                ))

            # Average results across episodes
            avg_result = self._average_results(episode_results)
            results.append(avg_result)

            print(f"      Pre-intervention MSE: {avg_result.pre_intervention_mse:.6f}")
            print(f"      Post-intervention MSE: {avg_result.post_intervention_mse:.6f}")
            print(f"      Adaptation rate: {avg_result.adaptation_rate:.3f}")
            print(f"      Coherence score: {avg_result.coherence_score:.3f}")

        self.intervention_results.extend(results)

        # Overall weather intervention summary
        avg_adaptation = np.mean([r.adaptation_rate for r in results])
        avg_coherence = np.mean([r.coherence_score for r in results])

        print(f"\n   🎯 WEATHER INTERVENTION SUMMARY:")
        print(f"      Average adaptation rate: {avg_adaptation:.3f}")
        print(f"      Average coherence score: {avg_coherence:.3f}")
        print(f"      Causal understanding: {'STRONG' if avg_coherence > 0.7 else 'MODERATE' if avg_coherence > 0.5 else 'WEAK'}")

        return results

    def test_factor_ablation(self, num_episodes: int = 15) -> List[InterventionResult]:
        """
        Test 2: Factor Ablation Analysis
        Remove each causal factor and measure prediction impact

        This identifies which factors the model considers most important
        and validates causal factor discovery capabilities.
        """
        print(f"\n🔬 FACTOR ABLATION ANALYSIS")
        print(f"   Episodes: {num_episodes}")
        print(f"   Testing causal factor importance ranking")

        results = []
        factor_names = ['weather', 'event', 'crowd_density', 'time_hour']

        for factor_name in factor_names:
            print(f"\n   Ablating factor: {factor_name}")

            episode_results = []

            for episode in range(num_episodes):
                # Generate complete episode with all factors
                complete_trajectory = self._generate_random_episode(episode_length=80)

                # Create ablated version (neutralize the factor)
                ablated_trajectory = self._create_ablated_trajectory(
                    complete_trajectory, factor_name
                )

                # Compare model predictions: complete vs ablated
                complete_predictions = self._predict_trajectory(complete_trajectory)
                ablated_predictions = self._predict_trajectory(ablated_trajectory)

                # Measure prediction impact of factor removal
                prediction_impact = np.mean((complete_predictions - ablated_predictions) ** 2)

                # Also measure physics ground truth impact
                physics_impact = np.mean((complete_trajectory['states'][1:] -
                                        ablated_trajectory['states'][1:]) ** 2)

                # Calculate factor importance score
                importance_score = prediction_impact / (physics_impact + 1e-8)

                result = {
                    'pre_intervention_mse': np.mean((complete_predictions -
                                                   complete_trajectory['states'][1:]) ** 2),
                    'post_intervention_mse': np.mean((ablated_predictions -
                                                    ablated_trajectory['states'][1:]) ** 2),
                    'adaptation_rate': importance_score,  # Use as importance measure
                    'causal_effect_magnitude': prediction_impact,
                    'model_prediction_shift': prediction_impact,
                    'physics_prediction_shift': physics_impact,
                    'coherence_score': min(1.0, importance_score)
                }

                intervention_spec = InterventionSpec(
                    intervention_type=InterventionType.FACTOR_ABLATION,
                    target_factor=factor_name,
                    intervention_timestep=0,
                    original_value='active',
                    intervention_value='ablated'
                )

                episode_results.append(InterventionResult(
                    intervention_spec=intervention_spec,
                    **result
                ))

            # Average results for this factor
            avg_result = self._average_results(episode_results)
            results.append(avg_result)

            print(f"      Factor importance: {avg_result.causal_effect_magnitude:.6f}")
            print(f"      Model sensitivity: {avg_result.model_prediction_shift:.6f}")

        self.intervention_results.extend(results)

        # Rank factors by importance
        factor_ranking = [(r.intervention_spec.target_factor, r.causal_effect_magnitude)
                         for r in results]
        factor_ranking.sort(key=lambda x: x[1], reverse=True)

        print(f"\n   🏆 FACTOR IMPORTANCE RANKING:")
        for i, (factor, importance) in enumerate(factor_ranking):
            print(f"      {i+1}. {factor}: {importance:.6f}")

        return results

    def test_counterfactual_reasoning(self, num_episodes: int = 10) -> List[InterventionResult]:
        """
        Test 3: Counterfactual Reasoning
        "What if this episode had different causal conditions?"

        This tests the model's ability to reason about alternative scenarios
        and predict outcomes under different causal configurations.
        """
        print(f"\n🔮 COUNTERFACTUAL REASONING TEST")
        print(f"   Episodes: {num_episodes}")
        print(f"   Testing 'what-if' scenario prediction")

        results = []

        for episode in range(num_episodes):
            print(f"\n   Episode {episode + 1}/{num_episodes}")

            # Generate original episode
            original_trajectory = self._generate_random_episode(episode_length=60)

            # Create counterfactual: opposite weather conditions
            original_weather_val = original_trajectory['causal_factors'][0, 0]  # Weather factor

            # Map to opposite weather
            if original_weather_val < 0.25:  # SUNNY
                counterfactual_weather = WeatherType.SNOW
            elif original_weather_val < 0.5:  # RAIN
                counterfactual_weather = WeatherType.SUNNY
            elif original_weather_val < 0.75:  # SNOW
                counterfactual_weather = WeatherType.RAIN
            else:  # FOG
                counterfactual_weather = WeatherType.SUNNY

            # Generate counterfactual trajectory
            counterfactual_trajectory = self._generate_counterfactual_episode(
                original_trajectory, counterfactual_weather
            )

            # Test model predictions on both scenarios
            original_predictions = self._predict_trajectory(original_trajectory)
            counterfactual_predictions = self._predict_trajectory(counterfactual_trajectory)

            # Measure prediction differences
            model_difference = np.mean((original_predictions - counterfactual_predictions) ** 2)
            physics_difference = np.mean((original_trajectory['states'][1:] -
                                        counterfactual_trajectory['states'][1:]) ** 2)

            # Calculate coherence: how well model differences match physics differences
            if physics_difference > 1e-8:
                coherence_score = 1.0 - abs(model_difference - physics_difference) / physics_difference
                coherence_score = max(0.0, min(1.0, coherence_score))
            else:
                coherence_score = 1.0 if model_difference < 1e-6 else 0.0

            result = {
                'pre_intervention_mse': np.mean((original_predictions - original_trajectory['states'][1:]) ** 2),
                'post_intervention_mse': np.mean((counterfactual_predictions - counterfactual_trajectory['states'][1:]) ** 2),
                'adaptation_rate': coherence_score,
                'causal_effect_magnitude': model_difference,
                'model_prediction_shift': model_difference,
                'physics_prediction_shift': physics_difference,
                'coherence_score': coherence_score
            }

            intervention_spec = InterventionSpec(
                intervention_type=InterventionType.COUNTERFACTUAL,
                target_factor='weather',
                intervention_timestep=0,
                original_value=f"weather_{original_weather_val:.2f}",
                intervention_value=counterfactual_weather.value
            )

            results.append(InterventionResult(
                intervention_spec=intervention_spec,
                **result
            ))

            print(f"      Coherence score: {coherence_score:.3f}")

        self.intervention_results.extend(results)

        # Overall counterfactual reasoning summary
        avg_coherence = np.mean([r.coherence_score for r in results])
        print(f"\n   🎯 COUNTERFACTUAL REASONING SUMMARY:")
        print(f"      Average coherence: {avg_coherence:.3f}")
        print(f"      Causal reasoning: {'EXCELLENT' if avg_coherence > 0.8 else 'GOOD' if avg_coherence > 0.6 else 'POOR'}")

        return results

    def _generate_consistent_episode(self, weather: WeatherType, episode_length: int) -> Dict:
        """Generate episode with consistent causal conditions"""
        self.env.reset()

        # Set consistent causal state
        causal_state = CausalState(
            time_hour=12,  # Noon
            day_week=2,    # Wednesday
            weather=weather,
            event=EventType.NORMAL,
            crowd_density=0.3
        )
        self.env.causal_state = causal_state

        states = []
        actions = []
        causal_factors = []

        obs, _ = self.env.reset()
        states.append(obs)

        for step in range(episode_length):
            # Simple goal-directed policy
            goal_x, goal_y = obs[2], obs[3]
            agent_x, agent_y = obs[0], obs[1]

            dx, dy = goal_x - agent_x, goal_y - agent_y
            dist = np.sqrt(dx*dx + dy*dy)

            if dist > 0.1:
                action = np.array([dx/dist, dy/dist]) * 0.5
            else:
                action = np.array([0.0, 0.0])

            # Add small random component
            action += np.random.normal(0, 0.1, 2)
            action = np.clip(action, -1, 1)

            obs, reward, terminated, truncated, info = self.env.step(action)

            states.append(obs)
            actions.append(action)
            # Convert dict causal_state to vector
            causal_dict = info['causal_state']
            causal_vector = self._causal_dict_to_vector(causal_dict)
            causal_factors.append(causal_vector)

            if terminated or truncated:
                break

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'causal_factors': np.array(causal_factors)
        }

    def _generate_intervention_episode(self, baseline_trajectory: Dict,
                                     intervention_timestep: int,
                                     factor_name: str,
                                     new_value) -> Dict:
        """Generate episode with intervention applied at specific timestep"""

        # Copy baseline trajectory up to intervention point
        states = baseline_trajectory['states'][:intervention_timestep + 1].copy()
        actions = baseline_trajectory['actions'][:intervention_timestep].copy()
        causal_factors = baseline_trajectory['causal_factors'][:intervention_timestep].copy()

        # Continue simulation from intervention point with modified causal factor
        self.env.reset()

        # Set agent to intervention state
        obs = states[intervention_timestep]

        # Apply intervention to causal state
        if factor_name == 'weather':
            weather_val = self.weather_mapping[new_value]
            modified_causal_state = CausalState(
                time_hour=12,
                day_week=2,
                weather=new_value,
                event=EventType.NORMAL,
                crowd_density=0.3
            )

        self.env.causal_state = modified_causal_state

        # Continue episode with modified causal factors
        remaining_actions = baseline_trajectory['actions'][intervention_timestep:]

        for i, action in enumerate(remaining_actions):
            obs, reward, terminated, truncated, info = self.env.step(action)

            states = np.append(states, [obs], axis=0)
            actions = np.append(actions, [action], axis=0)
            # Convert dict causal_state to vector
            causal_dict = info['causal_state']
            causal_vector = self._causal_dict_to_vector(causal_dict)
            causal_factors = np.append(causal_factors, [causal_vector], axis=0)

            if terminated or truncated:
                break

        return {
            'states': states,
            'actions': actions,
            'causal_factors': causal_factors
        }

    def _generate_random_episode(self, episode_length: int) -> Dict:
        """Generate episode with random causal conditions"""
        self.env.reset()

        states = []
        actions = []
        causal_factors = []

        obs, _ = self.env.reset()
        states.append(obs)

        for step in range(episode_length):
            # Random policy for ablation testing
            action = self.env.action_space.sample()

            obs, reward, terminated, truncated, info = self.env.step(action)

            states.append(obs)
            actions.append(action)
            # Convert dict causal_state to vector
            causal_dict = info['causal_state']
            causal_vector = self._causal_dict_to_vector(causal_dict)
            causal_factors.append(causal_vector)

            if terminated or truncated:
                break

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'causal_factors': np.array(causal_factors)
        }

    def _generate_counterfactual_episode(self, original_trajectory: Dict,
                                       counterfactual_weather: WeatherType) -> Dict:
        """Generate counterfactual episode with different weather"""

        # Use same initial state and actions, but different weather
        self.env.reset()

        # Set counterfactual causal state
        counterfactual_causal_state = CausalState(
            time_hour=12,
            day_week=2,
            weather=counterfactual_weather,
            event=EventType.NORMAL,
            crowd_density=0.3
        )
        self.env.causal_state = counterfactual_causal_state

        states = []
        actions = original_trajectory['actions']
        causal_factors = []

        obs, _ = self.env.reset()
        states.append(obs)

        for action in actions:
            obs, reward, terminated, truncated, info = self.env.step(action)

            states.append(obs)
            # Convert dict causal_state to vector
            causal_dict = info['causal_state']
            causal_vector = self._causal_dict_to_vector(causal_dict)
            causal_factors.append(causal_vector)

            if terminated or truncated:
                break

        return {
            'states': np.array(states),
            'actions': actions,
            'causal_factors': np.array(causal_factors)
        }

    def _create_ablated_trajectory(self, trajectory: Dict, ablated_factor: str) -> Dict:
        """Create trajectory with specified factor ablated (neutralized)"""

        ablated_trajectory = {
            'states': trajectory['states'].copy(),
            'actions': trajectory['actions'].copy(),
            'causal_factors': trajectory['causal_factors'].copy()
        }

        # Neutralize the specified factor
        factor_mapping = {
            'weather': 0,      # Weather is first causal factor
            'event': 1,        # Event is second
            'crowd_density': 2, # Crowd is third
            'time_hour': 3     # Time is fourth
        }

        if ablated_factor in factor_mapping:
            factor_idx = factor_mapping[ablated_factor]
            # Set factor to neutral value (middle of range)
            ablated_trajectory['causal_factors'][:, factor_idx] = 0.5

        return ablated_trajectory

    def _predict_trajectory(self, trajectory: Dict) -> np.ndarray:
        """Use trained model to predict state trajectory"""

        states = trajectory['states'][:-1]  # All states except last
        actions = trajectory['actions']
        causal_factors = trajectory['causal_factors']

        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.FloatTensor(actions).to(self.device)
            causal_tensor = torch.FloatTensor(causal_factors).to(self.device)

            # Add batch dimension if needed
            if len(states_tensor.shape) == 2:
                states_tensor = states_tensor.unsqueeze(0)
                actions_tensor = actions_tensor.unsqueeze(0)
                causal_tensor = causal_tensor.unsqueeze(0)

            # Model-specific prediction
            if self.model_type in ['lstm_predictor', 'gru_dynamics']:
                predictions, _ = self.model(states_tensor, actions_tensor, causal_tensor)
                return predictions.squeeze(0).cpu().numpy()
            else:  # linear_dynamics, neural_ode
                batch_size, seq_len = states_tensor.shape[:2]
                predictions = []
                for t in range(seq_len):
                    pred = self.model(states_tensor[:, t], actions_tensor[:, t], causal_tensor[:, t])
                    predictions.append(pred)
                predictions = torch.stack(predictions, dim=1)
                return predictions.squeeze(0).cpu().numpy()

    def _evaluate_intervention(self, baseline_trajectory: Dict,
                             intervention_trajectory: Dict,
                             intervention_timestep: int) -> Dict:
        """Evaluate model performance before/after intervention"""

        # Pre-intervention evaluation
        pre_states = baseline_trajectory['states'][:intervention_timestep]
        pre_actions = baseline_trajectory['actions'][:intervention_timestep-1]
        pre_causal = baseline_trajectory['causal_factors'][:intervention_timestep-1]

        pre_trajectory = {
            'states': pre_states,
            'actions': pre_actions,
            'causal_factors': pre_causal
        }

        pre_predictions = self._predict_trajectory(pre_trajectory)
        pre_mse = np.mean((pre_predictions - pre_states[1:]) ** 2)

        # Post-intervention evaluation
        post_length = min(20, len(intervention_trajectory['states']) - intervention_timestep - 1)
        post_start = intervention_timestep
        post_end = post_start + post_length

        post_states = intervention_trajectory['states'][post_start:post_end + 1]
        post_actions = intervention_trajectory['actions'][post_start:post_end]
        post_causal = intervention_trajectory['causal_factors'][post_start:post_end]

        post_trajectory = {
            'states': post_states,
            'actions': post_actions,
            'causal_factors': post_causal
        }

        post_predictions = self._predict_trajectory(post_trajectory)
        post_mse = np.mean((post_predictions - post_states[1:]) ** 2)

        # Calculate adaptation rate
        adaptation_rate = max(0.0, 1.0 - (post_mse / (pre_mse + 1e-8)))

        # Calculate prediction shifts
        if len(pre_predictions) >= 5 and len(post_predictions) >= 5:
            model_shift = np.mean((pre_predictions[-5:] - post_predictions[:5]) ** 2)
            physics_shift = np.mean((baseline_trajectory['states'][intervention_timestep-5:intervention_timestep] -
                                   intervention_trajectory['states'][intervention_timestep:intervention_timestep+5]) ** 2)
        else:
            model_shift = abs(post_mse - pre_mse)
            physics_shift = model_shift

        # Calculate coherence score
        if physics_shift > 1e-8:
            coherence_score = 1.0 - abs(model_shift - physics_shift) / physics_shift
            coherence_score = max(0.0, min(1.0, coherence_score))
        else:
            coherence_score = 1.0 if model_shift < 1e-6 else 0.0

        return {
            'pre_intervention_mse': pre_mse,
            'post_intervention_mse': post_mse,
            'adaptation_rate': adaptation_rate,
            'causal_effect_magnitude': model_shift,
            'model_prediction_shift': model_shift,
            'physics_prediction_shift': physics_shift,
            'coherence_score': coherence_score
        }

    def _average_results(self, results: List[InterventionResult]) -> InterventionResult:
        """Average multiple intervention results"""
        if not results:
            return None

        return InterventionResult(
            intervention_spec=results[0].intervention_spec,
            pre_intervention_mse=np.mean([r.pre_intervention_mse for r in results]),
            post_intervention_mse=np.mean([r.post_intervention_mse for r in results]),
            adaptation_rate=np.mean([r.adaptation_rate for r in results]),
            causal_effect_magnitude=np.mean([r.causal_effect_magnitude for r in results]),
            model_prediction_shift=np.mean([r.model_prediction_shift for r in results]),
            physics_prediction_shift=np.mean([r.physics_prediction_shift for r in results]),
            coherence_score=np.mean([r.coherence_score for r in results])
        )

    def generate_report(self, output_path: str = "results/causal_intervention_report.json"):
        """Generate comprehensive causal reasoning validation report"""

        # Organize results by intervention type
        results_by_type = {}
        for result in self.intervention_results:
            intervention_type = result.intervention_spec.intervention_type.value
            if intervention_type not in results_by_type:
                results_by_type[intervention_type] = []
            results_by_type[intervention_type].append(result)

        # Calculate summary statistics
        summary = {}
        for intervention_type, results in results_by_type.items():
            summary[intervention_type] = {
                'num_tests': len(results),
                'avg_adaptation_rate': np.mean([r.adaptation_rate for r in results]),
                'avg_coherence_score': np.mean([r.coherence_score for r in results]),
                'model_physics_alignment': np.mean([r.coherence_score for r in results]),
                'causal_understanding_level': self._assess_causal_understanding(results)
            }

        # Overall causal reasoning assessment
        all_coherence_scores = [r.coherence_score for r in self.intervention_results]
        overall_coherence = np.mean(all_coherence_scores)

        overall_assessment = {
            'overall_coherence_score': overall_coherence,
            'causal_reasoning_level': self._assess_overall_causal_reasoning(overall_coherence),
            'production_ready': overall_coherence > 0.6,
            'scientific_validity': overall_coherence > 0.5
        }

        # Generate final report
        report = {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'model_info': get_model_info(self.model),
            'test_timestamp': str(np.datetime64('now')),
            'summary_statistics': summary,
            'overall_assessment': overall_assessment,
            'detailed_results': [asdict(r) for r in self.intervention_results]
        }

        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary
        print(f"\n📊 CAUSAL INTERVENTION TESTING REPORT")
        print("=" * 60)
        print(f"Model: {self.model_type}")
        print(f"Total tests conducted: {len(self.intervention_results)}")

        for intervention_type, stats in summary.items():
            print(f"\n{intervention_type.upper()}:")
            print(f"   Tests: {stats['num_tests']}")
            print(f"   Adaptation Rate: {stats['avg_adaptation_rate']:.3f}")
            print(f"   Coherence Score: {stats['avg_coherence_score']:.3f}")
            print(f"   Understanding Level: {stats['causal_understanding_level']}")

        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   Coherence Score: {overall_coherence:.3f}")
        print(f"   Causal Reasoning: {overall_assessment['causal_reasoning_level']}")
        print(f"   Production Ready: {overall_assessment['production_ready']}")
        print(f"   Scientific Validity: {overall_assessment['scientific_validity']}")

        print(f"\nFull report saved to: {output_path}")
        return report

    def _assess_causal_understanding(self, results: List[InterventionResult]) -> str:
        """Assess level of causal understanding for specific intervention type"""
        avg_coherence = np.mean([r.coherence_score for r in results])

        if avg_coherence > 0.8:
            return "EXCELLENT"
        elif avg_coherence > 0.6:
            return "GOOD"
        elif avg_coherence > 0.4:
            return "MODERATE"
        else:
            return "POOR"

    def _assess_overall_causal_reasoning(self, overall_coherence: float) -> str:
        """Assess overall causal reasoning capability"""
        if overall_coherence > 0.8:
            return "ADVANCED CAUSAL REASONING"
        elif overall_coherence > 0.6:
            return "SOLID CAUSAL UNDERSTANDING"
        elif overall_coherence > 0.4:
            return "BASIC CAUSAL AWARENESS"
        else:
            return "LIMITED CAUSAL REASONING"

    def run_complete_test_suite(self) -> Dict:
        """Run all causal intervention tests"""
        print("🧪 STARTING COMPLETE CAUSAL INTERVENTION TEST SUITE")
        print("=" * 60)

        # Run all tests
        weather_results = self.test_weather_interventions(num_episodes=10, intervention_timestep=30)
        ablation_results = self.test_factor_ablation(num_episodes=8)
        counterfactual_results = self.test_counterfactual_reasoning(num_episodes=6)

        # Generate comprehensive report
        report = self.generate_report()

        print("\n🎉 CAUSAL INTERVENTION TESTING COMPLETED!")
        print("   Results demonstrate model's causal reasoning capabilities")

        return report


def main():
    """Run causal intervention testing on a trained model"""
    import argparse

    parser = argparse.ArgumentParser(description='Causal Intervention Testing Framework')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['linear_dynamics', 'gru_dynamics', 'lstm_predictor', 'neural_ode', 'vae_rnn_hybrid'],
                       help='Type of model')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for testing')
    parser.add_argument('--quick', action='store_true', help='Run quick test suite')

    args = parser.parse_args()

    print("🧪 Causal Intervention Testing Framework")
    print("=" * 60)

    # Initialize tester
    tester = CausalInterventionTester(args.model_path, args.model_type, args.device)

    # Run tests
    if args.quick:
        print("Running quick test suite...")
        tester.test_weather_interventions(num_episodes=3, intervention_timestep=20)
        tester.test_factor_ablation(num_episodes=3)
        tester.generate_report()
    else:
        print("Running complete test suite...")
        tester.run_complete_test_suite()

    return 0


if __name__ == "__main__":
    exit(main())