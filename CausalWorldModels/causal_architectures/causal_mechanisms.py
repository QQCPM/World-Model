"""
Causal Mechanism Modules
Physics-based causal mechanisms for campus environment

Models explicit causal relationships:
- Weather → Movement dynamics
- Crowd density → Path availability
- Events → Crowd patterns
- Time → Visibility and crowd
- Road conditions → Movement friction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import math


class CausalMechanismModules(nn.Module):
    """
    Collection of physics-based causal mechanism modules

    Each mechanism models a specific causal relationship with interpretable parameters
    """

    def __init__(self, state_dim=12, hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Individual causal mechanisms
        self.weather_mechanism = WeatherMovementMechanism(hidden_dim)
        self.crowd_mechanism = CrowdDensityMechanism(hidden_dim)
        self.event_mechanism = SpecialEventMechanism(hidden_dim)
        self.time_mechanism = TimeOfDayMechanism(hidden_dim)
        self.road_mechanism = RoadConditionMechanism(hidden_dim)

        # Mechanism composition network
        self.composition_network = nn.Sequential(
            nn.Linear(5 * hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4)  # Movement effects: [velocity_scale, friction, visibility, path_width]
        )

        # Effect integration weights
        self.effect_weights = nn.Parameter(torch.ones(5) * 0.2)  # Equal initial weights

        # HSIC Independence enforcer for mechanism isolation
        self.independence_enforcer_enabled = True

    def forward(self, state, causal_factors, action):
        """
        Apply causal mechanisms to predict state transitions

        Args:
            state: [batch_size, state_dim] current state
            causal_factors: [batch_size, 5] causal variables [weather, crowd, event, time, road]
            action: [batch_size, 2] agent action

        Returns:
            mechanism_effects: Dict of individual mechanism outputs
            composed_effects: [batch_size, 4] composed movement effects
            next_state: [batch_size, state_dim] predicted next state
        """
        batch_size = state.shape[0]

        # Extract individual causal factors
        weather = causal_factors[:, 0:1]      # Weather condition
        crowd = causal_factors[:, 1:2]        # Crowd density
        event = causal_factors[:, 2:3]        # Special event
        time_day = causal_factors[:, 3:4]     # Time of day
        road = causal_factors[:, 4:5]         # Road conditions

        # Apply individual mechanisms
        weather_effect = self.weather_mechanism(weather, state, action)
        crowd_effect = self.crowd_mechanism(crowd, state, action)
        event_effect = self.event_mechanism(event, state, action)
        time_effect = self.time_mechanism(time_day, state, action)
        road_effect = self.road_mechanism(road, state, action)

        # Store individual effects
        mechanism_effects = {
            'weather': weather_effect,
            'crowd': crowd_effect,
            'event': event_effect,
            'time': time_effect,
            'road': road_effect
        }

        # Compose mechanisms
        all_effects = torch.cat([
            weather_effect, crowd_effect, event_effect, time_effect, road_effect
        ], dim=1)

        # Get composed movement effects
        composed_effects = self.composition_network(all_effects)

        # Apply effects to state transition
        next_state = self._apply_effects_to_state(state, action, composed_effects)

        # Compute mechanism independence metrics
        independence_loss = self.compute_hsic_independence_loss(mechanism_effects)
        isolation_confidence = self.compute_isolation_confidence(mechanism_effects)

        return mechanism_effects, composed_effects, next_state, independence_loss, isolation_confidence

    def _apply_effects_to_state(self, state, action, effects):
        """
        Apply composed causal effects to state transition

        Args:
            state: [batch_size, state_dim] current state
            action: [batch_size, 2] action [velocity_x, velocity_y]
            effects: [batch_size, 4] [velocity_scale, friction, visibility, path_width]

        Returns:
            next_state: [batch_size, state_dim] next state
        """
        # Extract state components
        pos_x, pos_y = state[:, 0:1], state[:, 1:2]
        vel_x, vel_y = state[:, 2:3], state[:, 3:4]
        goal_x, goal_y = state[:, 4:5], state[:, 5:6]

        # Extract effects
        velocity_scale = torch.sigmoid(effects[:, 0:1]) * 2.0  # 0-2x velocity
        friction = torch.sigmoid(effects[:, 1:2])              # 0-1 friction
        visibility = torch.sigmoid(effects[:, 2:3])            # 0-1 visibility
        path_width = torch.sigmoid(effects[:, 3:4]) + 0.5      # 0.5-1.5 path width

        # Apply velocity scaling and friction
        intended_velocity = action * velocity_scale
        actual_velocity = intended_velocity * (1 - friction)

        # Update position
        new_pos_x = pos_x + actual_velocity[:, 0:1] * 0.1  # dt = 0.1
        new_pos_y = pos_y + actual_velocity[:, 1:2] * 0.1

        # Update velocity (with momentum)
        new_vel_x = 0.8 * vel_x + 0.2 * actual_velocity[:, 0:1]
        new_vel_y = 0.8 * vel_y + 0.2 * actual_velocity[:, 1:2]

        # Reconstruct state
        next_state = torch.cat([
            new_pos_x, new_pos_y,           # Position
            new_vel_x, new_vel_y,           # Velocity
            goal_x, goal_y,                 # Goal (unchanged)
            state[:, 6:7],                  # Goal distance (will be recomputed)
            state[:, 7:]                    # Causal factors (preserved)
        ], dim=1)

        # Recompute goal distance
        goal_distance = torch.sqrt((new_pos_x - goal_x)**2 + (new_pos_y - goal_y)**2)
        next_state[:, 6:7] = goal_distance

        return next_state

    def compute_hsic_independence_loss(self, mechanism_effects):
        """
        Compute HSIC independence loss between mechanism effects
        Based on Hilbert-Schmidt Independence Criterion
        """
        if not self.independence_enforcer_enabled:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        mechanism_outputs = [
            mechanism_effects['weather'],
            mechanism_effects['crowd'],
            mechanism_effects['event'],
            mechanism_effects['time'],
            mechanism_effects['road']
        ]

        num_mechanisms = len(mechanism_outputs)
        independence_loss = torch.tensor(0.0, device=mechanism_outputs[0].device)

        for i in range(num_mechanisms):
            for j in range(i + 1, num_mechanisms):
                # Compute HSIC between mechanism i and j
                hsic = self.hsic_loss(mechanism_outputs[i], mechanism_outputs[j])
                independence_loss += hsic

        # Normalize by number of pairs
        return independence_loss / (num_mechanisms * (num_mechanisms - 1) / 2)

    def hsic_loss(self, x, y, sigma=1.0):
        """
        Hilbert-Schmidt Independence Criterion for measuring dependence
        Lower values indicate more independence
        """
        batch_size = x.shape[0]

        # Flatten to ensure proper shape
        x_flat = x.view(batch_size, -1)
        y_flat = y.view(batch_size, -1)

        # Compute Gram matrices using Gaussian kernel
        K_x = self.gaussian_kernel(x_flat, x_flat, sigma)
        K_y = self.gaussian_kernel(y_flat, y_flat, sigma)

        # Center the Gram matrices
        H = torch.eye(batch_size, device=x.device) - torch.ones(batch_size, batch_size, device=x.device) / batch_size

        K_x_centered = H @ K_x @ H
        K_y_centered = H @ K_y @ H

        # HSIC statistic
        if batch_size > 1:
            hsic = torch.trace(K_x_centered @ K_y_centered) / (batch_size - 1) ** 2
        else:
            hsic = torch.tensor(0.0, device=x.device)

        return hsic

    def gaussian_kernel(self, x, y, sigma=1.0):
        """
        Gaussian RBF kernel for HSIC computation
        """
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))

    def compute_isolation_confidence(self, mechanism_effects):
        """
        Compute mechanism isolation confidence based on independence
        """
        independence_loss = self.compute_hsic_independence_loss(mechanism_effects)

        # Convert independence loss to confidence score
        # Lower independence loss = higher isolation confidence
        isolation_confidence = 1.0 / (1.0 + independence_loss.item() * 10.0)

        return isolation_confidence

    def get_mechanism_explanations(self, causal_factors):
        """
        Get human-readable explanations of active mechanisms

        Args:
            causal_factors: [batch_size, 5] causal variables

        Returns:
            explanations: List of mechanism explanations
        """
        explanations = []

        # Average causal factors for explanation
        avg_factors = causal_factors.mean(dim=0)

        # Weather explanation
        weather_val = avg_factors[0].item()
        if weather_val > 0.5:
            explanations.append(f"Clear weather conditions facilitate normal movement (factor: {weather_val:.2f})")
        elif weather_val > 0:
            explanations.append(f"Moderate weather slightly impedes movement (factor: {weather_val:.2f})")
        else:
            explanations.append(f"Adverse weather significantly reduces movement speed (factor: {weather_val:.2f})")

        # Crowd explanation
        crowd_val = avg_factors[1].item()
        if crowd_val > 0.7:
            explanations.append(f"High crowd density creates significant path congestion (density: {crowd_val:.2f})")
        elif crowd_val > 0.3:
            explanations.append(f"Moderate crowds require some path planning (density: {crowd_val:.2f})")
        else:
            explanations.append(f"Low crowd density allows free movement (density: {crowd_val:.2f})")

        # Event explanation
        event_val = avg_factors[2].item()
        if event_val > 0.5:
            explanations.append(f"Special event is creating unusual crowd patterns (intensity: {event_val:.2f})")

        # Time explanation
        time_val = avg_factors[3].item()
        if 0.7 < time_val < 0.9:
            explanations.append(f"Peak hours creating increased campus activity (time: {time_val:.2f})")
        elif time_val < 0.2 or time_val > 0.9:
            explanations.append(f"Off-peak hours with reduced visibility/activity (time: {time_val:.2f})")

        # Road explanation
        road_val = avg_factors[4].item()
        if road_val < 0.3:
            explanations.append(f"Poor road conditions increase movement friction (condition: {road_val:.2f})")

        return explanations

    def intervene_on_mechanism(self, mechanism_name, intervention_value):
        """
        Perform intervention on specific causal mechanism

        Args:
            mechanism_name: Name of mechanism to intervene on
            intervention_value: New value to set

        Returns:
            intervention_context: Dict describing the intervention
        """
        valid_mechanisms = ['weather', 'crowd', 'event', 'time', 'road']
        assert mechanism_name in valid_mechanisms, f"Unknown mechanism: {mechanism_name}"

        # Store original state for restoration
        original_state = {}

        if mechanism_name == 'weather':
            # Intervention: Force specific weather condition
            original_state['weather_params'] = self.weather_mechanism.get_parameters()
            self.weather_mechanism.set_intervention(intervention_value)

        elif mechanism_name == 'crowd':
            # Intervention: Set crowd density
            original_state['crowd_params'] = self.crowd_mechanism.get_parameters()
            self.crowd_mechanism.set_intervention(intervention_value)

        # Add other mechanisms as needed...

        intervention_context = {
            'mechanism': mechanism_name,
            'intervention_value': intervention_value,
            'original_state': original_state,
            'active': True
        }

        return intervention_context

    def restore_mechanism(self, intervention_context):
        """
        Restore mechanism to pre-intervention state
        """
        if not intervention_context['active']:
            return

        mechanism_name = intervention_context['mechanism']
        original_state = intervention_context['original_state']

        if mechanism_name == 'weather':
            self.weather_mechanism.restore_parameters(original_state['weather_params'])

        # Mark intervention as inactive
        intervention_context['active'] = False

    def get_model_name(self):
        return "causal_mechanism_modules"


class WeatherMovementMechanism(nn.Module):
    """
    Weather → Movement dynamics causal mechanism

    Models how weather conditions affect movement speed and friction
    """

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Weather interpretation network
        self.weather_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Physics parameters (learnable)
        self.sunny_multiplier = nn.Parameter(torch.tensor(1.0))      # Normal conditions
        self.rain_multiplier = nn.Parameter(torch.tensor(0.8))       # Slightly slower
        self.snow_multiplier = nn.Parameter(torch.tensor(0.6))       # Much slower
        self.fog_multiplier = nn.Parameter(torch.tensor(0.9))        # Reduced visibility

        self.intervention_active = False
        self.intervention_value = None

    def forward(self, weather_factor, state, action):
        """
        Apply weather effects to movement

        Args:
            weather_factor: [batch_size, 1] weather condition (-1 to 1)
            state: [batch_size, state_dim] current state
            action: [batch_size, 2] intended action

        Returns:
            weather_effect: [batch_size, hidden_dim] weather effect representation
        """
        if self.intervention_active:
            # Use intervention value instead of observed
            weather_factor = torch.full_like(weather_factor, self.intervention_value)

        # Interpret weather condition
        weather_effect = self.weather_net(weather_factor)

        return weather_effect

    def get_parameters(self):
        """Get current mechanism parameters"""
        return {
            'sunny_multiplier': self.sunny_multiplier.data.clone(),
            'rain_multiplier': self.rain_multiplier.data.clone(),
            'snow_multiplier': self.snow_multiplier.data.clone(),
            'fog_multiplier': self.fog_multiplier.data.clone()
        }

    def set_intervention(self, value):
        """Set intervention value"""
        self.intervention_active = True
        self.intervention_value = value

    def restore_parameters(self, params):
        """Restore mechanism parameters"""
        self.sunny_multiplier.data = params['sunny_multiplier']
        self.rain_multiplier.data = params['rain_multiplier']
        self.snow_multiplier.data = params['snow_multiplier']
        self.fog_multiplier.data = params['fog_multiplier']
        self.intervention_active = False


class CrowdDensityMechanism(nn.Module):
    """
    Crowd density → Path availability causal mechanism
    """

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.crowd_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.intervention_active = False
        self.intervention_value = None

    def forward(self, crowd_factor, state, action):
        if self.intervention_active:
            crowd_factor = torch.full_like(crowd_factor, self.intervention_value)

        crowd_effect = self.crowd_net(crowd_factor)
        return crowd_effect

    def get_parameters(self):
        return {'crowd_params': 'placeholder'}

    def set_intervention(self, value):
        self.intervention_active = True
        self.intervention_value = value

    def restore_parameters(self, params):
        self.intervention_active = False


class SpecialEventMechanism(nn.Module):
    """
    Special events → Crowd patterns causal mechanism
    """

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.event_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, event_factor, state, action):
        return self.event_net(event_factor)


class TimeOfDayMechanism(nn.Module):
    """
    Time of day → Visibility and activity causal mechanism
    """

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.time_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, time_factor, state, action):
        return self.time_net(time_factor)


class RoadConditionMechanism(nn.Module):
    """
    Road conditions → Movement friction causal mechanism
    """

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.road_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, road_factor, state, action):
        return self.road_net(road_factor)


def test_causal_mechanisms():
    """
    Test causal mechanism modules
    """
    print("Testing CausalMechanismModules...")

    # Create mechanism modules
    mechanisms = CausalMechanismModules(state_dim=12, hidden_dim=32)

    # Test data
    batch_size = 8
    state = torch.randn(batch_size, 12)
    causal_factors = torch.randn(batch_size, 5)
    action = torch.randn(batch_size, 2)

    # Test forward pass
    mechanism_effects, composed_effects, next_state, independence_loss, isolation_confidence = mechanisms(state, causal_factors, action)

    print(f"Mechanism effects keys: {list(mechanism_effects.keys())}")
    print(f"Composed effects shape: {composed_effects.shape}")
    print(f"Next state shape: {next_state.shape}")
    print(f"Independence loss: {independence_loss.item():.6f}")
    print(f"Isolation confidence: {isolation_confidence:.4f}")

    # Test explanations
    explanations = mechanisms.get_mechanism_explanations(causal_factors)
    print(f"Generated {len(explanations)} explanations")

    # Test intervention
    intervention_ctx = mechanisms.intervene_on_mechanism('weather', 0.8)
    print(f"Intervention context: {intervention_ctx['mechanism']}")

    # Test restoration
    mechanisms.restore_mechanism(intervention_ctx)

    print("✅ CausalMechanismModules test passed")
    print(f"✅ Model parameters: {sum(p.numel() for p in mechanisms.parameters())}")

    return True


if __name__ == "__main__":
    test_causal_mechanisms()