#!/usr/bin/env python3
"""
Causal Intervention Testing Framework
Tests whether trained models understand causality vs just pattern matching

This is the critical validation step that determines if we have true causal reasoning.
"""

import torch
import numpy as np
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add imports for our models and environment
sys.path.append('continuous_models')
sys.path.append('causal_envs')

from state_predictors import create_continuous_model, get_model_info
from continuous_campus_env import ContinuousCampusEnv, WeatherType, EventType


class InterventionType(Enum):
    """Types of causal interventions to test"""
    WEATHER_CHANGE = "weather_change"
    EVENT_ONSET = "event_onset"
    FACTOR_ABLATION = "factor_ablation"
    COUNTERFACTUAL = "counterfactual"


@dataclass
class InterventionSpec:
    """Specification for a causal intervention"""
    intervention_type: InterventionType
    target_factor: str  # 'weather', 'crowd_density', etc.
    intervention_timestep: int
    original_value: float
    intervention_value: float
    duration: int = -1  # -1 means permanent


@dataclass
class InterventionResult:
    """Results from causal intervention experiment"""
    intervention_spec: InterventionSpec
    pre_intervention_mse: float
    post_intervention_mse: float
    adaptation_rate: float
    causal_effect_magnitude: float
    model_prediction_drift: float
    physics_prediction_drift: float


class CausalInterventionTester:
    """Framework for testing causal understanding in trained models"""

    def __init__(self, model_path: str, model_type: str, device: str = 'auto'):
        """
        Initialize the intervention tester

        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model (gru_dynamics, lstm_predictor, etc.)
            device: Device to run tests on
        """
        self.model_type = model_type
        self.device = self._get_device(device)

        # Load trained model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Initialize environment for ground truth generation
        self.env = ContinuousCampusEnv()

        # Results storage
        self.intervention_results = []

        print(f"🧪 Causal Intervention Tester initialized")
        print(f"Model: {model_type}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters())}")

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

    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Get model kwargs from checkpoint or use defaults
        model_kwargs = checkpoint.get('model_kwargs', {'hidden_dim': 64})

        # Create and load model
        model = create_continuous_model(self.model_type, **model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        return model

    def test_weather_intervention(self, num_episodes: int = 50,
                                intervention_timestep: int = 50) -> List[InterventionResult]:
        """
        Test 1: Weather Intervention
        Change weather mid-episode and test model adaptation
        """
        print(f"\n🌦️  Testing Weather Interventions")
        print(f"Episodes: {num_episodes}, Intervention at timestep: {intervention_timestep}")

        results = []
        weather_types = [WeatherType.SUNNY, WeatherType.RAINY, WeatherType.SNOWY]

        for original_weather in weather_types:
            for target_weather in weather_types:
                if original_weather == target_weather:
                    continue

                print(f"  Testing: {original_weather.name} → {target_weather.name}")

                episode_results = []

                for episode in range(num_episodes):
                    # Generate baseline episode with original weather
                    baseline_states, baseline_actions, baseline_causal = self._generate_episode(
                        weather=original_weather, max_steps=intervention_timestep * 2
                    )

                    # Generate intervention episode: change weather at timestep
                    intervention_states, intervention_actions, intervention_causal = \
                        self._generate_intervention_episode(
                            baseline_states, baseline_actions, baseline_causal,
                            intervention_timestep, 'weather', target_weather.value
                        )

                    # Test model predictions
                    result = self._evaluate_intervention(
                        baseline_states, baseline_actions, baseline_causal,
                        intervention_states, intervention_actions, intervention_causal,
                        intervention_timestep
                    )

                    intervention_spec = InterventionSpec(
                        intervention_type=InterventionType.WEATHER_CHANGE,
                        target_factor='weather',
                        intervention_timestep=intervention_timestep,
                        original_value=original_weather.value,
                        intervention_value=target_weather.value
                    )

                    intervention_result = InterventionResult(
                        intervention_spec=intervention_spec,
                        **result
                    )

                    episode_results.append(intervention_result)

                # Average results across episodes
                avg_result = self._average_intervention_results(episode_results)
                results.append(avg_result)

                print(f"    Pre-intervention MSE: {avg_result.pre_intervention_mse:.6f}")
                print(f"    Post-intervention MSE: {avg_result.post_intervention_mse:.6f}")
                print(f"    Adaptation rate: {avg_result.adaptation_rate:.3f}")

        self.intervention_results.extend(results)
        return results

    def test_factor_ablation(self, num_episodes: int = 30) -> List[InterventionResult]:
        """
        Test 2: Factor Ablation
        Remove each causal factor and measure prediction impact
        """
        print(f"\n🔬 Testing Factor Ablation")
        print(f"Episodes: {num_episodes}")

        results = []
        causal_factors = ['weather', 'crowd_density', 'event_strength', 'campus_time', 'road_condition']

        for factor_idx, factor_name in enumerate(causal_factors):
            print(f"  Ablating factor: {factor_name}")

            episode_results = []

            for episode in range(num_episodes):
                # Generate complete episode
                states, actions, causal = self._generate_episode(max_steps=100)

                # Create ablated version (set factor to zero/neutral)
                ablated_causal = causal.copy()
                ablated_causal[:, factor_idx] = 0.0  # Zero out the factor

                # Compare model predictions
                complete_predictions = self._predict_sequence(states[:-1], actions[:-1], causal[:-1])
                ablated_predictions = self._predict_sequence(states[:-1], actions[:-1], ablated_causal[:-1])

                # Measure prediction difference
                prediction_impact = np.mean((complete_predictions - ablated_predictions) ** 2)

                # Measure ground truth difference (if possible)
                ablated_states = self._simulate_with_ablated_factor(states[0], actions, factor_idx)
                ground_truth_impact = np.mean((states[1:] - ablated_states[1:]) ** 2)

                result = {
                    'pre_intervention_mse': np.mean((complete_predictions - states[1:]) ** 2),
                    'post_intervention_mse': np.mean((ablated_predictions - states[1:]) ** 2),
                    'adaptation_rate': 1.0,  # Not applicable for ablation
                    'causal_effect_magnitude': prediction_impact,
                    'model_prediction_drift': prediction_impact,
                    'physics_prediction_drift': ground_truth_impact
                }

                intervention_spec = InterventionSpec(
                    intervention_type=InterventionType.FACTOR_ABLATION,
                    target_factor=factor_name,
                    intervention_timestep=0,
                    original_value=1.0,
                    intervention_value=0.0
                )

                episode_results.append(InterventionResult(
                    intervention_spec=intervention_spec,
                    **result
                ))

            # Average results
            avg_result = self._average_intervention_results(episode_results)
            results.append(avg_result)

            print(f"    Factor importance score: {avg_result.causal_effect_magnitude:.6f}")
            print(f"    Model sensitivity: {avg_result.model_prediction_drift:.6f}")

        self.intervention_results.extend(results)
        return results

    def test_counterfactual_reasoning(self, num_episodes: int = 20) -> List[InterventionResult]:
        """
        Test 3: Counterfactual Reasoning
        "What if this episode had different causal factors?"
        """
        print(f"\n🔮 Testing Counterfactual Reasoning")
        print(f"Episodes: {num_episodes}")

        results = []

        for episode in range(num_episodes):
            # Generate original episode
            original_states, original_actions, original_causal = self._generate_episode(max_steps=80)

            # Create counterfactual: different weather for entire episode
            counterfactual_causal = original_causal.copy()

            # Change weather to opposite (sunny<->rainy, snowy->sunny)
            original_weather = original_causal[0, 0]  # Weather is first causal factor
            if original_weather == WeatherType.SUNNY.value:
                new_weather = WeatherType.RAINY.value
            elif original_weather == WeatherType.RAINY.value:
                new_weather = WeatherType.SNOWY.value
            else:
                new_weather = WeatherType.SUNNY.value

            counterfactual_causal[:, 0] = new_weather

            # Generate counterfactual ground truth
            counterfactual_states = self._simulate_with_modified_causal(
                original_states[0], original_actions, counterfactual_causal
            )

            # Test model predictions
            original_predictions = self._predict_sequence(
                original_states[:-1], original_actions[:-1], original_causal[:-1]
            )
            counterfactual_predictions = self._predict_sequence(
                original_states[:-1], original_actions[:-1], counterfactual_causal[:-1]
            )

            # Measure counterfactual coherence
            model_difference = np.mean((original_predictions - counterfactual_predictions) ** 2)
            physics_difference = np.mean((original_states[1:] - counterfactual_states[1:]) ** 2)

            coherence_score = 1.0 - abs(model_difference - physics_difference) / max(model_difference, physics_difference)

            result = {
                'pre_intervention_mse': np.mean((original_predictions - original_states[1:]) ** 2),
                'post_intervention_mse': np.mean((counterfactual_predictions - counterfactual_states[1:]) ** 2),
                'adaptation_rate': coherence_score,
                'causal_effect_magnitude': model_difference,
                'model_prediction_drift': model_difference,
                'physics_prediction_drift': physics_difference
            }

            intervention_spec = InterventionSpec(
                intervention_type=InterventionType.COUNTERFACTUAL,
                target_factor='weather',
                intervention_timestep=0,
                original_value=original_weather,
                intervention_value=new_weather
            )

            results.append(InterventionResult(
                intervention_spec=intervention_spec,
                **result
            ))

        avg_result = self._average_intervention_results(results)
        print(f"  Average counterfactual coherence: {avg_result.adaptation_rate:.3f}")
        print(f"  Model-physics alignment: {1.0 - abs(avg_result.model_prediction_drift - avg_result.physics_prediction_drift):.3f}")

        self.intervention_results.extend(results)
        return results

    def _generate_episode(self, weather: WeatherType = None, max_steps: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a single episode with specified weather"""
        if weather is None:
            weather = np.random.choice(list(WeatherType))

        # Set environment weather
        self.env.reset()
        self.env.current_weather = weather

        states = []
        actions = []
        causal_factors = []

        obs, _ = self.env.reset()
        states.append(obs)

        for step in range(max_steps):
            # Simple random action policy for testing
            action = self.env.action_space.sample()

            obs, reward, terminated, truncated, info = self.env.step(action)

            states.append(obs)
            actions.append(action)
            causal_factors.append(info['causal_state'].to_vector())

            if terminated or truncated:
                break

        return np.array(states), np.array(actions), np.array(causal_factors)

    def _generate_intervention_episode(self, baseline_states: np.ndarray,
                                     baseline_actions: np.ndarray,
                                     baseline_causal: np.ndarray,
                                     intervention_timestep: int,
                                     factor_name: str,
                                     new_value: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate episode with intervention applied at specific timestep"""
        # Create modified causal factors
        modified_causal = baseline_causal.copy()

        # Apply intervention
        factor_mapping = {'weather': 0, 'crowd_density': 1, 'event_strength': 2, 'campus_time': 3, 'road_condition': 4}
        factor_idx = factor_mapping[factor_name]

        modified_causal[intervention_timestep:, factor_idx] = new_value

        # Simulate with modified causal factors
        new_states = self._simulate_with_modified_causal(
            baseline_states[0], baseline_actions, modified_causal
        )

        return new_states, baseline_actions, modified_causal

    def _simulate_with_modified_causal(self, initial_state: np.ndarray,
                                     actions: np.ndarray,
                                     causal_factors: np.ndarray) -> np.ndarray:
        """Simulate episode with modified causal factors using environment"""
        # This would require extending the environment to accept causal factor overrides
        # For now, return modified version based on known physics

        states = [initial_state]
        current_state = initial_state.copy()

        for i, (action, causal) in enumerate(zip(actions, causal_factors)):
            # Apply simple physics based on causal factors
            weather_effect = 1.0 - 0.3 * (causal[0] == WeatherType.SNOWY.value)  # Snow reduces movement
            crowd_effect = 1.0 - 0.2 * causal[1]  # Crowds reduce movement

            # Update position based on action and causal effects
            movement_scale = weather_effect * crowd_effect
            current_state[0] += action[0] * movement_scale * 0.1  # x position
            current_state[1] += action[1] * movement_scale * 0.1  # y position

            # Add some realistic state evolution
            current_state[2] = action[0] * movement_scale  # velocity_x
            current_state[3] = action[1] * movement_scale  # velocity_y

            # Keep within reasonable bounds
            current_state[0] = np.clip(current_state[0], -10, 10)
            current_state[1] = np.clip(current_state[1], -10, 10)

            states.append(current_state.copy())

        return np.array(states)

    def _simulate_with_ablated_factor(self, initial_state: np.ndarray,
                                    actions: np.ndarray,
                                    ablated_factor_idx: int) -> np.ndarray:
        """Simulate episode with one causal factor ablated"""
        # This is a simplified version - in practice would use environment
        states = [initial_state]
        current_state = initial_state.copy()

        for action in actions:
            # Simple physics without the ablated factor
            if ablated_factor_idx == 0:  # Weather ablated
                movement_scale = 1.0  # No weather effects
            elif ablated_factor_idx == 1:  # Crowd ablated
                movement_scale = 1.0  # No crowd effects
            else:
                movement_scale = 0.8  # Other factors have some generic effect

            current_state[0] += action[0] * movement_scale * 0.1
            current_state[1] += action[1] * movement_scale * 0.1
            current_state[2] = action[0] * movement_scale
            current_state[3] = action[1] * movement_scale

            current_state[0] = np.clip(current_state[0], -10, 10)
            current_state[1] = np.clip(current_state[1], -10, 10)

            states.append(current_state.copy())

        return np.array(states)

    def _predict_sequence(self, states: np.ndarray, actions: np.ndarray,
                         causal_factors: np.ndarray) -> np.ndarray:
        """Use trained model to predict state sequence"""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.FloatTensor(actions).to(self.device)
            causal_tensor = torch.FloatTensor(causal_factors).to(self.device)

            if len(states_tensor.shape) == 2:
                states_tensor = states_tensor.unsqueeze(0)
                actions_tensor = actions_tensor.unsqueeze(0)
                causal_tensor = causal_tensor.unsqueeze(0)

            # Model-specific prediction
            if self.model_type in ['lstm_predictor', 'gru_dynamics']:
                predictions, _ = self.model(states_tensor, actions_tensor, causal_tensor)
                return predictions.squeeze(0).cpu().numpy()
            elif self.model_type == 'neural_ode':
                batch_size, seq_len = states_tensor.shape[:2]
                predictions = []
                for t in range(seq_len):
                    pred = self.model(states_tensor[:, t], actions_tensor[:, t], causal_tensor[:, t])
                    predictions.append(pred)
                predictions = torch.stack(predictions, dim=1)
                return predictions.squeeze(0).cpu().numpy()
            else:  # linear_dynamics
                batch_size, seq_len = states_tensor.shape[:2]
                predictions = []
                for t in range(seq_len):
                    pred = self.model(states_tensor[:, t], actions_tensor[:, t], causal_tensor[:, t])
                    predictions.append(pred)
                predictions = torch.stack(predictions, dim=1)
                return predictions.squeeze(0).cpu().numpy()

    def _evaluate_intervention(self, baseline_states: np.ndarray, baseline_actions: np.ndarray,
                             baseline_causal: np.ndarray, intervention_states: np.ndarray,
                             intervention_actions: np.ndarray, intervention_causal: np.ndarray,
                             intervention_timestep: int) -> Dict:
        """Evaluate model performance before/after intervention"""

        # Pre-intervention predictions
        pre_predictions = self._predict_sequence(
            baseline_states[:intervention_timestep],
            baseline_actions[:intervention_timestep-1],
            baseline_causal[:intervention_timestep-1]
        )
        pre_mse = np.mean((pre_predictions - baseline_states[1:intervention_timestep]) ** 2)

        # Post-intervention predictions
        post_predictions = self._predict_sequence(
            intervention_states[intervention_timestep:intervention_timestep+20],
            intervention_actions[intervention_timestep:intervention_timestep+19],
            intervention_causal[intervention_timestep:intervention_timestep+19]
        )
        post_mse = np.mean((post_predictions - intervention_states[intervention_timestep+1:intervention_timestep+21]) ** 2)

        # Adaptation rate (how quickly model adjusts)
        adaptation_rate = max(0, 1.0 - (post_mse / (pre_mse + 1e-8)))

        # Causal effect magnitude
        model_diff = np.mean((pre_predictions[-10:] - post_predictions[:10]) ** 2)
        physics_diff = np.mean((baseline_states[intervention_timestep-10:intervention_timestep] -
                               intervention_states[intervention_timestep:intervention_timestep+10]) ** 2)

        return {
            'pre_intervention_mse': pre_mse,
            'post_intervention_mse': post_mse,
            'adaptation_rate': adaptation_rate,
            'causal_effect_magnitude': model_diff,
            'model_prediction_drift': model_diff,
            'physics_prediction_drift': physics_diff
        }

    def _average_intervention_results(self, results: List[InterventionResult]) -> InterventionResult:
        """Average multiple intervention results"""
        if not results:
            return None

        avg_result = InterventionResult(
            intervention_spec=results[0].intervention_spec,
            pre_intervention_mse=np.mean([r.pre_intervention_mse for r in results]),
            post_intervention_mse=np.mean([r.post_intervention_mse for r in results]),
            adaptation_rate=np.mean([r.adaptation_rate for r in results]),
            causal_effect_magnitude=np.mean([r.causal_effect_magnitude for r in results]),
            model_prediction_drift=np.mean([r.model_prediction_drift for r in results]),
            physics_prediction_drift=np.mean([r.physics_prediction_drift for r in results])
        )

        return avg_result

    def generate_report(self, output_path: str = "results/causal_intervention_report.json"):
        """Generate comprehensive report of intervention testing results"""

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
                'avg_causal_effect_magnitude': np.mean([r.causal_effect_magnitude for r in results]),
                'model_physics_alignment': np.mean([
                    1.0 - abs(r.model_prediction_drift - r.physics_prediction_drift) /
                    max(r.model_prediction_drift, r.physics_prediction_drift, 1e-8)
                    for r in results
                ])
            }

        # Generate final report
        report = {
            'model_type': self.model_type,
            'model_info': get_model_info(self.model),
            'test_timestamp': str(np.datetime64('now')),
            'summary_statistics': summary,
            'detailed_results': [
                {
                    'intervention_type': r.intervention_spec.intervention_type.value,
                    'target_factor': r.intervention_spec.target_factor,
                    'original_value': r.intervention_spec.original_value,
                    'intervention_value': r.intervention_spec.intervention_value,
                    'adaptation_rate': r.adaptation_rate,
                    'causal_effect_magnitude': r.causal_effect_magnitude,
                    'model_physics_alignment': 1.0 - abs(r.model_prediction_drift - r.physics_prediction_drift) /
                                               max(r.model_prediction_drift, r.physics_prediction_drift, 1e-8)
                }
                for r in self.intervention_results
            ]
        }

        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print(f"\n📊 CAUSAL INTERVENTION TESTING REPORT")
        print(f"Model: {self.model_type}")
        print(f"Total tests conducted: {len(self.intervention_results)}")

        for intervention_type, stats in summary.items():
            print(f"\n{intervention_type.upper()}:")
            print(f"  Tests: {stats['num_tests']}")
            print(f"  Adaptation Rate: {stats['avg_adaptation_rate']:.3f}")
            print(f"  Model-Physics Alignment: {stats['model_physics_alignment']:.3f}")

        print(f"\nFull report saved to: {output_path}")

        return report


def main():
    """Run complete causal intervention testing suite"""
    import argparse

    parser = argparse.ArgumentParser(description='Causal Intervention Testing')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model_type', type=str, required=True, help='Type of model')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--quick_test', action='store_true', help='Run reduced test suite')

    args = parser.parse_args()

    print("🧪 Starting Causal Intervention Testing")
    print("=" * 60)

    # Initialize tester
    tester = CausalInterventionTester(args.model_path, args.model_type, args.device)

    # Run test suite
    if args.quick_test:
        print("Running quick test suite...")
        tester.test_weather_intervention(num_episodes=10)
        tester.test_factor_ablation(num_episodes=5)
    else:
        print("Running complete test suite...")
        tester.test_weather_intervention(num_episodes=50)
        tester.test_factor_ablation(num_episodes=30)
        tester.test_counterfactual_reasoning(num_episodes=20)

    # Generate report
    report = tester.generate_report()

    print("\n🎉 Causal Intervention Testing completed!")
    return 0


if __name__ == "__main__":
    exit(main())