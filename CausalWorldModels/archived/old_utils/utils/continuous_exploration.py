"""
Continuous Exploration Strategy
Goal-directed exploration for continuous action space with causal awareness
"""

import numpy as np
from typing import Dict, Tuple, Optional

class ContinuousExploration:
    """Exploration strategy for continuous 2D velocity control"""

    def __init__(self, goal_bias=0.7, exploration_noise=0.3, causal_adaptation=True):
        """
        Initialize continuous exploration strategy

        Args:
            goal_bias: Probability of moving toward goal (vs random)
            exploration_noise: Gaussian noise scale for actions
            causal_adaptation: Whether to adapt behavior based on causal factors
        """
        self.goal_bias = goal_bias
        self.exploration_noise = exploration_noise
        self.causal_adaptation = causal_adaptation

        # Causal adaptation parameters
        self.weather_speed_multipliers = {
            'sunny': 1.0,
            'rain': 0.8,
            'snow': 0.6,
            'fog': 0.9
        }

        self.event_behavior_modifiers = {
            'normal': 1.0,
            'gameday': 0.9,     # More crowded, slower
            'exam': 1.1,        # More urgent
            'break': 0.8,       # More relaxed
            'construction': 0.7  # Obstacles, slower
        }

    def select_action(self, observation: np.ndarray, causal_state: Dict, step: int) -> np.ndarray:
        """
        Select continuous action based on observation and causal context

        Args:
            observation: 12D continuous state vector
            causal_state: Dictionary with causal factors
            step: Current episode step

        Returns:
            2D velocity command in range [-1, 1] x [-1, 1]
        """
        # Extract positions from observation
        agent_pos = observation[0:2]
        goal_pos = observation[2:4]
        current_velocity = observation[4:6]
        goal_distance = observation[6]

        # Base goal-directed action
        if goal_distance > 0.1 and np.random.random() < self.goal_bias:
            # Move toward goal
            direction = goal_pos - agent_pos
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 0:
                goal_action = direction / direction_norm
            else:
                goal_action = np.array([0.0, 0.0])
        else:
            # Random exploration
            goal_action = np.random.uniform(-1.0, 1.0, size=2)

        # Apply causal adaptations
        if self.causal_adaptation:
            goal_action = self._apply_causal_modifiers(goal_action, causal_state, step)

        # Add exploration noise
        noise = np.random.normal(0, self.exploration_noise, size=2)
        action = goal_action + noise

        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)

        return action

    def _apply_causal_modifiers(self, action: np.ndarray, causal_state: Dict, step: int) -> np.ndarray:
        """Apply causal factors to modify action selection"""

        modified_action = action.copy()

        # Weather effects on movement intensity
        weather = causal_state.get('weather', 'sunny')
        weather_multiplier = self.weather_speed_multipliers.get(weather, 1.0)
        modified_action *= weather_multiplier

        # Event effects on urgency/behavior
        event = causal_state.get('event', 'normal')
        event_multiplier = self.event_behavior_modifiers.get(event, 1.0)
        modified_action *= event_multiplier

        # Time-of-day effects
        time_hour = causal_state.get('time_hour', 12)
        if time_hour >= 22 or time_hour <= 6:  # Night time
            modified_action *= 0.8  # Slower at night

        # Crowd density effects
        crowd_density = causal_state.get('crowd_density', 1)
        if crowd_density >= 4:  # High crowd
            modified_action *= 0.7  # Slower in crowds

        # Temporal exploration decay (less random over time)
        exploration_decay = max(0.5, 1.0 - step / 1000.0)
        if np.random.random() > exploration_decay:
            # Reduce randomness as episode progresses
            modified_action = action * 0.8 + modified_action * 0.2

        return modified_action

    def get_exploration_stats(self) -> Dict:
        """Return current exploration strategy parameters"""
        return {
            'goal_bias': self.goal_bias,
            'exploration_noise': self.exploration_noise,
            'causal_adaptation': self.causal_adaptation,
            'weather_multipliers': self.weather_speed_multipliers,
            'event_modifiers': self.event_behavior_modifiers
        }


class AdaptiveExploration(ContinuousExploration):
    """Advanced exploration that adapts parameters during episode generation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.episode_success_history = []
        self.adaptation_rate = 0.05

    def update_strategy(self, episode_reward: float, episode_success: bool):
        """Update exploration parameters based on episode outcomes"""

        self.episode_success_history.append(episode_success)

        # Keep only recent history
        if len(self.episode_success_history) > 20:
            self.episode_success_history = self.episode_success_history[-20:]

        # Adapt goal bias based on recent success rate
        if len(self.episode_success_history) >= 5:
            recent_success_rate = np.mean(self.episode_success_history[-5:])

            if recent_success_rate < 0.1:  # Too low success rate
                self.goal_bias = min(0.9, self.goal_bias + self.adaptation_rate)
                self.exploration_noise = max(0.1, self.exploration_noise - self.adaptation_rate)
            elif recent_success_rate > 0.4:  # Good success rate
                self.goal_bias = max(0.5, self.goal_bias - self.adaptation_rate)
                self.exploration_noise = min(0.5, self.exploration_noise + self.adaptation_rate)

    def get_adaptation_stats(self) -> Dict:
        """Return adaptation statistics"""
        stats = self.get_exploration_stats()
        stats.update({
            'episode_history_length': len(self.episode_success_history),
            'recent_success_rate': np.mean(self.episode_success_history[-5:]) if len(self.episode_success_history) >= 5 else 0.0,
            'adaptation_rate': self.adaptation_rate
        })
        return stats