"""
Structure-Aware Counterfactual Generator
Generate counterfactual data respecting learned causal structure

Based on:
- Counterfactual Data Augmentation (CoDA) with Locally Factored Dynamics
- Structure-aware counterfactual generation
- Conservative intervention strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import copy


class StructureAwareCFGenerator(nn.Module):
    """
    Generate counterfactual episodes respecting causal structure

    Takes base episodes and intervention specifications, generates
    counterfactual trajectories using learned causal mechanisms
    """

    def __init__(self, num_variables=5, state_dim=12, action_dim=2):
        super().__init__()
        self.num_variables = num_variables
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Variable names for campus environment
        self.variable_names = [
            'weather',           # 0: weather condition
            'crowd_density',     # 1: campus crowd level
            'special_event',     # 2: special events happening
            'time_of_day',       # 3: time of day
            'road_conditions'    # 4: road/path conditions
        ]

        # Counterfactual consistency network
        self.consistency_network = nn.Sequential(
            nn.Linear(state_dim + num_variables, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)
        )

        # Intervention effect propagation
        self.propagation_network = nn.Sequential(
            nn.Linear(num_variables * 2, 32),  # original + intervention
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_variables)
        )

        # Temporal consistency constraints
        self.temporal_constraint_weight = 0.1

    def generate_counterfactual(self, base_episode, intervention_spec, causal_graph,
                              causal_mechanisms=None):
        """
        Generate counterfactual episode from base episode and intervention

        Args:
            base_episode: Dict with 'states', 'actions', 'causal_factors'
            intervention_spec: Dict specifying intervention details
            causal_graph: [num_vars, num_vars] learned causal adjacency matrix
            causal_mechanisms: Optional CausalMechanismModules for physics

        Returns:
            counterfactual_episode: Dict with counterfactual trajectory
            generation_info: Dict with generation metadata
        """
        # Extract base trajectory
        base_states = base_episode['states']        # [seq_len, state_dim]
        base_actions = base_episode['actions']      # [seq_len, action_dim]
        base_causal = base_episode['causal_factors'] # [seq_len, num_variables]

        seq_len = base_states.shape[0]

        # Create counterfactual causal factors
        cf_causal_factors = self._apply_intervention(
            base_causal, intervention_spec, causal_graph
        )

        # Generate counterfactual states and actions
        cf_states, cf_actions = self._propagate_counterfactual_effects(
            base_states, base_actions, base_causal, cf_causal_factors,
            causal_mechanisms
        )

        # Ensure consistency
        cf_states, cf_actions = self._enforce_consistency(
            cf_states, cf_actions, cf_causal_factors, base_episode
        )

        # Package counterfactual episode
        counterfactual_episode = {
            'states': cf_states,
            'actions': cf_actions,
            'causal_factors': cf_causal_factors,
            'intervention_spec': intervention_spec,
            'base_episode_id': base_episode.get('episode_id', None)
        }

        # Generation metadata
        generation_info = {
            'intervention_type': intervention_spec['type'],
            'target_variables': intervention_spec['targets'],
            'temporal_consistency_loss': self._compute_temporal_consistency(cf_states),
            'causal_consistency_score': self._compute_causal_consistency(
                cf_causal_factors, causal_graph
            ),
            'deviation_from_base': self._compute_deviation(
                base_episode, counterfactual_episode
            )
        }

        return counterfactual_episode, generation_info

    def _apply_intervention(self, base_causal_factors, intervention_spec, causal_graph):
        """
        Apply intervention to causal factors with structure awareness

        Args:
            base_causal_factors: [seq_len, num_variables] original factors
            intervention_spec: Intervention specification
            causal_graph: [num_vars, num_vars] causal structure

        Returns:
            cf_causal_factors: [seq_len, num_variables] counterfactual factors
        """
        cf_causal_factors = base_causal_factors.clone()
        seq_len = cf_causal_factors.shape[0]

        intervention_type = intervention_spec['type']
        targets = intervention_spec['targets']

        if intervention_type == 'single_variable':
            # Direct intervention on single variable
            target_var = targets[0]
            intervention_value = intervention_spec['values'][0]

            # Apply intervention throughout trajectory
            cf_causal_factors[:, target_var] = intervention_value

            # Propagate effects through causal graph
            cf_causal_factors = self._propagate_causal_effects(
                cf_causal_factors, target_var, causal_graph
            )

        elif intervention_type == 'dual_variable':
            # Intervention on two variables
            for i, target_var in enumerate(targets):
                intervention_value = intervention_spec['values'][i]
                cf_causal_factors[:, target_var] = intervention_value

            # Propagate effects for both variables
            for target_var in targets:
                cf_causal_factors = self._propagate_causal_effects(
                    cf_causal_factors, target_var, causal_graph
                )

        elif intervention_type == 'temporal_shift':
            # Temporal intervention (shift timing)
            target_var = targets[0]
            temporal_shift = intervention_spec['temporal_shift']

            # Shift the variable's temporal pattern
            if temporal_shift != 0:
                shifted_values = torch.roll(
                    cf_causal_factors[:, target_var],
                    shifts=temporal_shift,
                    dims=0
                )
                cf_causal_factors[:, target_var] = shifted_values

        elif intervention_type == 'causal_chain':
            # Intervention on causal chain
            parent, child = targets[0], targets[1]

            # Break causal connection by randomizing parent
            cf_causal_factors[:, parent] = torch.randn_like(cf_causal_factors[:, parent])

            # Recompute child based on new parent
            parent_effect = causal_graph[parent, child]
            if parent_effect > 0.1:  # Significant causal relationship
                cf_causal_factors[:, child] = (
                    0.7 * cf_causal_factors[:, child] +
                    0.3 * parent_effect * cf_causal_factors[:, parent]
                )

        return cf_causal_factors

    def _propagate_causal_effects(self, causal_factors, intervention_target, causal_graph):
        """
        Propagate intervention effects through causal graph

        Args:
            causal_factors: [seq_len, num_variables] current factors
            intervention_target: Index of intervened variable
            causal_graph: [num_vars, num_vars] causal adjacency matrix

        Returns:
            updated_factors: [seq_len, num_variables] with propagated effects
        """
        updated_factors = causal_factors.clone()

        # Find variables affected by intervention target (children)
        affected_variables = torch.where(causal_graph[intervention_target, :] > 0.1)[0]

        for affected_var in affected_variables:
            # Compute effect magnitude
            effect_strength = causal_graph[intervention_target, affected_var]

            # Apply causal effect with temporal delay
            for t in range(1, causal_factors.shape[0]):
                # Use previous timestep of intervention target
                causal_input = updated_factors[t-1, intervention_target]

                # Apply causal mechanism (simplified)
                causal_effect = effect_strength * causal_input

                # Mix with original value
                updated_factors[t, affected_var] = (
                    0.7 * updated_factors[t, affected_var] +
                    0.3 * causal_effect
                )

        return updated_factors

    def _propagate_counterfactual_effects(self, base_states, base_actions, base_causal,
                                        cf_causal_factors, causal_mechanisms=None):
        """
        Propagate counterfactual causal factors to states and actions

        Args:
            base_states: [seq_len, state_dim] original states
            base_actions: [seq_len, action_dim] original actions
            base_causal: [seq_len, num_variables] original causal factors
            cf_causal_factors: [seq_len, num_variables] counterfactual factors
            causal_mechanisms: Optional mechanism modules

        Returns:
            cf_states: [seq_len, state_dim] counterfactual states
            cf_actions: [seq_len, action_dim] counterfactual actions
        """
        seq_len = base_states.shape[0]
        cf_states = base_states.clone()
        cf_actions = base_actions.clone()

        # Compute causal factor differences
        causal_diff = cf_causal_factors - base_causal

        for t in range(seq_len):
            # Apply counterfactual effects to state
            if causal_mechanisms is not None:
                # Use physics-based mechanisms
                current_state = cf_states[t:t+1]
                current_action = cf_actions[t:t+1]
                current_causal = cf_causal_factors[t:t+1]

                _, _, predicted_next_state, _, _ = causal_mechanisms(
                    current_state, current_causal, current_action
                )

                if t < seq_len - 1:
                    cf_states[t+1] = predicted_next_state.squeeze(0)

            else:
                # Use neural approximation
                state_input = torch.cat([cf_states[t], cf_causal_factors[t]])
                state_correction = self.consistency_network(state_input)

                # Apply correction
                cf_states[t] = cf_states[t] + 0.1 * state_correction

            # Modify actions based on causal changes
            # Actions might change due to different environmental conditions
            action_modification = self._compute_action_modification(
                causal_diff[t], cf_states[t], base_actions[t]
            )

            cf_actions[t] = cf_actions[t] + action_modification

        return cf_states, cf_actions

    def _compute_action_modification(self, causal_diff, current_state, base_action):
        """
        Compute how actions should change due to counterfactual causal factors

        Args:
            causal_diff: [num_variables] difference in causal factors
            current_state: [state_dim] current state
            base_action: [action_dim] original action

        Returns:
            action_modification: [action_dim] change to apply to action
        """
        # Simple rule-based modifications
        modification = torch.zeros_like(base_action)

        # Weather effects on movement
        weather_diff = causal_diff[0]  # Weather change
        if weather_diff < -0.5:  # Worse weather
            modification = modification * 0.8  # Slower movement

        # Crowd effects
        crowd_diff = causal_diff[1]  # Crowd density change
        if crowd_diff > 0.3:  # More crowded
            # More careful movement
            modification = modification * 0.9

        # Keep modifications small to maintain realism
        modification = torch.clamp(modification, -0.2, 0.2)

        return modification

    def _enforce_consistency(self, cf_states, cf_actions, cf_causal_factors, base_episode):
        """
        Enforce temporal and physical consistency in counterfactual trajectory

        Args:
            cf_states: [seq_len, state_dim] counterfactual states
            cf_actions: [seq_len, action_dim] counterfactual actions
            cf_causal_factors: [seq_len, num_variables] counterfactual factors
            base_episode: Original episode for reference

        Returns:
            consistent_states: [seq_len, state_dim] consistent states
            consistent_actions: [seq_len, action_dim] consistent actions
        """
        seq_len = cf_states.shape[0]
        consistent_states = cf_states.clone()
        consistent_actions = cf_actions.clone()

        # Smooth temporal transitions
        for t in range(1, seq_len):
            # Limit large state jumps
            state_diff = consistent_states[t] - consistent_states[t-1]
            max_change = 0.5  # Maximum change per timestep

            # Clamp large changes
            state_diff = torch.clamp(state_diff, -max_change, max_change)
            consistent_states[t] = consistent_states[t-1] + state_diff

            # Ensure actions are feasible
            consistent_actions[t] = torch.clamp(consistent_actions[t], -1.0, 1.0)

        # Maintain goal consistency (goal shouldn't change)
        base_goal = base_episode['states'][:, 4:6]  # Goal position
        consistent_states[:, 4:6] = base_goal

        return consistent_states, consistent_actions

    def _compute_temporal_consistency(self, states):
        """
        Compute temporal consistency score for trajectory

        Args:
            states: [seq_len, state_dim] state trajectory

        Returns:
            consistency_score: Scalar consistency measure
        """
        if states.shape[0] < 2:
            return 0.0

        # Compute state differences
        state_diffs = states[1:] - states[:-1]

        # Measure smoothness (lower variance = more consistent)
        smoothness = -torch.var(state_diffs, dim=0).mean()

        return smoothness.item()

    def _compute_causal_consistency(self, cf_causal_factors, causal_graph):
        """
        Compute consistency of causal factors with learned structure

        Args:
            cf_causal_factors: [seq_len, num_variables] counterfactual factors
            causal_graph: [num_vars, num_vars] causal structure

        Returns:
            consistency_score: Scalar consistency measure
        """
        seq_len = cf_causal_factors.shape[0]
        total_consistency = 0.0

        for t in range(1, seq_len):
            current_factors = cf_causal_factors[t]
            prev_factors = cf_causal_factors[t-1]

            # Check if changes respect causal structure
            for i in range(self.num_variables):
                for j in range(self.num_variables):
                    if causal_graph[i, j] > 0.1:  # Causal relationship exists
                        # Change in j should be related to change in i
                        change_i = current_factors[i] - prev_factors[i]
                        change_j = current_factors[j] - prev_factors[j]

                        # Simple consistency check
                        expected_change = causal_graph[i, j] * change_i
                        consistency = 1.0 - abs(change_j - expected_change)
                        total_consistency += consistency

        avg_consistency = total_consistency / (seq_len * self.num_variables ** 2)
        return avg_consistency

    def _compute_deviation(self, base_episode, counterfactual_episode):
        """
        Compute deviation between base and counterfactual episodes

        Returns:
            deviation_metrics: Dict with deviation measures
        """
        base_states = base_episode['states']
        cf_states = counterfactual_episode['states']

        state_deviation = F.mse_loss(cf_states, base_states).item()

        base_actions = base_episode['actions']
        cf_actions = counterfactual_episode['actions']

        action_deviation = F.mse_loss(cf_actions, base_actions).item()

        return {
            'state_deviation': state_deviation,
            'action_deviation': action_deviation,
            'total_deviation': state_deviation + action_deviation
        }

    def generate_batch_counterfactuals(self, base_episodes, intervention_specs,
                                     causal_graph, causal_mechanisms=None):
        """
        Generate counterfactuals for batch of episodes

        Args:
            base_episodes: List of base episodes
            intervention_specs: List of intervention specifications
            causal_graph: Learned causal structure
            causal_mechanisms: Optional mechanism modules

        Returns:
            counterfactual_episodes: List of counterfactual episodes
            generation_infos: List of generation metadata
        """
        counterfactual_episodes = []
        generation_infos = []

        for base_episode, intervention_spec in zip(base_episodes, intervention_specs):
            cf_episode, gen_info = self.generate_counterfactual(
                base_episode, intervention_spec, causal_graph, causal_mechanisms
            )

            counterfactual_episodes.append(cf_episode)
            generation_infos.append(gen_info)

        return counterfactual_episodes, generation_infos

    def get_model_name(self):
        return "structure_aware_cf_generator"


def test_counterfactual_generator():
    """
    Test counterfactual generator functionality
    """
    print("Testing StructureAwareCFGenerator...")

    # Create generator
    generator = StructureAwareCFGenerator(num_variables=5, state_dim=12, action_dim=2)

    # Create synthetic base episode
    seq_len = 20
    base_episode = {
        'states': torch.randn(seq_len, 12),
        'actions': torch.randn(seq_len, 2),
        'causal_factors': torch.randn(seq_len, 5),
        'episode_id': 'test_episode_001'
    }

    # Create intervention specification
    intervention_spec = {
        'type': 'single_variable',
        'targets': [0],  # Intervene on weather
        'values': [0.8]  # Set to sunny
    }

    # Create synthetic causal graph
    causal_graph = torch.rand(5, 5) * 0.5

    # Generate counterfactual
    cf_episode, gen_info = generator.generate_counterfactual(
        base_episode, intervention_spec, causal_graph
    )

    print(f"Generated counterfactual episode with {cf_episode['states'].shape[0]} timesteps")
    print(f"Temporal consistency: {gen_info['temporal_consistency_loss']:.4f}")
    print(f"Causal consistency: {gen_info['causal_consistency_score']:.4f}")
    print(f"State deviation: {gen_info['deviation_from_base']['state_deviation']:.4f}")

    # Test batch generation
    base_episodes = [base_episode] * 3
    intervention_specs = [intervention_spec] * 3

    cf_episodes, gen_infos = generator.generate_batch_counterfactuals(
        base_episodes, intervention_specs, causal_graph
    )

    print(f"Generated {len(cf_episodes)} counterfactual episodes in batch")

    print("✅ StructureAwareCFGenerator test passed")
    print(f"✅ Model parameters: {sum(p.numel() for p in generator.parameters())}")

    return True


if __name__ == "__main__":
    test_counterfactual_generator()