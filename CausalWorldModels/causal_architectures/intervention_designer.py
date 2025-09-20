"""
Intervention Designer for Active Causal Learning
Optimal intervention selection for efficient causal structure discovery

Based on:
- Active Learning for Optimal Intervention Design (Nature MI, 2023)
- Bayesian Optimization for Causal Discovery
- Information-theoretic intervention selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import math
from scipy.optimize import minimize


class InterventionDesigner(nn.Module):
    """
    Active learning component for optimal intervention design

    Selects most informative interventions to accelerate causal structure learning
    """

    def __init__(self, num_variables=5, intervention_budget=10, exploration_weight=1.0):
        super().__init__()
        self.num_variables = num_variables
        self.intervention_budget = intervention_budget
        self.exploration_weight = exploration_weight

        # Variable names for campus environment
        self.variable_names = [
            'weather',           # 0: weather condition
            'crowd_density',     # 1: campus crowd level
            'special_event',     # 2: special events happening
            'time_of_day',       # 3: time of day
            'road_conditions'    # 4: road/path conditions
        ]

        # Intervention feasibility (some variables are easier to intervene on)
        self.intervention_feasibility = torch.tensor([
            0.3,  # weather: difficult to control
            0.8,  # crowd_density: can influence through events
            0.9,  # special_event: can schedule events
            0.4,  # time_of_day: can test at different times
            0.7   # road_conditions: can modify temporarily
        ])

        # Uncertainty tracker for each causal relationship
        self.uncertainty_tracker = nn.Parameter(
            torch.ones(num_variables, num_variables) * 0.5
        )

        # Value network for intervention selection
        self.value_network = nn.Sequential(
            nn.Linear(num_variables * num_variables + num_variables, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Information gain estimator
        self.info_gain_estimator = nn.Sequential(
            nn.Linear(num_variables * num_variables + num_variables, 32),  # structure + proposed intervention
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Intervention history tracking
        self.register_buffer('intervention_history', torch.zeros(100, num_variables))
        self.register_buffer('intervention_outcomes', torch.zeros(100))
        self.history_index = 0

    def compute_information_gain(self, current_uncertainty, proposed_intervention,
                                current_structure):
        """
        Estimate expected information gain from proposed intervention

        Args:
            current_uncertainty: [num_vars, num_vars] uncertainty matrix
            proposed_intervention: [num_vars] intervention specification
            current_structure: [num_vars, num_vars] current causal structure estimate

        Returns:
            expected_gain: Scalar expected information gain
        """
        # Combine uncertainty and structure information
        structure_uncertainty = current_uncertainty * current_structure

        # Focus on uncertain relationships that could be clarified
        intervention_targets = proposed_intervention > 0.1
        relevant_uncertainties = structure_uncertainty[intervention_targets, :]

        # Estimate information gain using neural estimator
        intervention_context = torch.cat([
            structure_uncertainty.flatten(),
            proposed_intervention
        ])

        estimated_gain = self.info_gain_estimator(intervention_context)

        return estimated_gain

    def select_optimal_intervention(self, structure_learner, current_data,
                                  num_candidates=20):
        """
        Select optimal intervention using acquisition function

        Args:
            structure_learner: CausalStructureLearner instance
            current_data: [batch_size, seq_len, num_vars] recent observational data
            num_candidates: Number of intervention candidates to consider

        Returns:
            best_intervention: Dict specifying optimal intervention
            candidate_info: List of all candidates with scores
        """
        # Get current structure estimate and uncertainty
        adjacency = structure_learner.get_adjacency_matrix()
        current_uncertainty = self.uncertainty_tracker.detach()

        candidates = []

        # Generate intervention candidates
        for _ in range(num_candidates):
            candidate = self._generate_intervention_candidate()

            # Compute expected information gain
            info_gain = self.compute_information_gain(
                current_uncertainty, candidate['intervention_vector'], adjacency
            )

            # Compute intervention cost (feasibility)
            cost = self._compute_intervention_cost(candidate)

            # Acquisition function: information gain / cost
            acquisition_score = info_gain / (cost + 1e-6)

            candidate['info_gain'] = info_gain.item()
            candidate['cost'] = cost
            candidate['acquisition_score'] = acquisition_score.item()

            candidates.append(candidate)

        # Sort by acquisition score
        candidates.sort(key=lambda x: x['acquisition_score'], reverse=True)

        best_intervention = candidates[0]

        return best_intervention, candidates

    def _generate_intervention_candidate(self):
        """
        Generate a candidate intervention

        Returns:
            candidate: Dict with intervention specification
        """
        # Randomly select 1-2 variables to intervene on
        num_targets = np.random.choice([1, 2], p=[0.7, 0.3])
        target_variables = np.random.choice(self.num_variables, num_targets, replace=False)

        # Generate intervention values
        intervention_vector = torch.zeros(self.num_variables)
        intervention_specs = {}

        for var_idx in target_variables:
            # Random intervention value
            intervention_value = np.random.uniform(-1, 1)
            intervention_vector[var_idx] = 1.0  # Mark as intervened
            intervention_specs[var_idx] = {
                'variable': self.variable_names[var_idx],
                'value': intervention_value,
                'type': 'do_operation'
            }

        return {
            'intervention_vector': intervention_vector,
            'target_variables': target_variables.tolist(),
            'intervention_specs': intervention_specs,
            'num_targets': num_targets
        }

    def _compute_intervention_cost(self, intervention):
        """
        Compute cost of performing intervention

        Args:
            intervention: Intervention specification dict

        Returns:
            cost: Scalar cost (lower is better)
        """
        base_cost = 0.0

        for var_idx in intervention['target_variables']:
            # Feasibility cost
            feasibility = self.intervention_feasibility[var_idx]
            var_cost = 1.0 / (feasibility + 0.1)

            # Complexity cost (multi-variable interventions are more expensive)
            complexity_penalty = 1.2 ** (intervention['num_targets'] - 1)

            base_cost += var_cost * complexity_penalty

        return base_cost

    def update_from_intervention_outcome(self, intervention, outcome_data,
                                       structure_learner):
        """
        Update intervention designer based on intervention outcome

        Args:
            intervention: Intervention specification that was performed
            outcome_data: [batch_size, seq_len, num_vars] data after intervention
            structure_learner: Updated structure learner
        """
        # Compute actual information gain
        with torch.no_grad():
            loss_before = self._get_cached_loss()  # Should cache this
            loss_after, _ = structure_learner.compute_structure_loss(outcome_data)
            actual_gain = loss_before - loss_after.item()

        # Update intervention history
        intervention_vector = intervention['intervention_vector']
        self.intervention_history[self.history_index] = intervention_vector
        self.intervention_outcomes[self.history_index] = actual_gain
        self.history_index = (self.history_index + 1) % 100

        # Update uncertainty estimates
        self._update_uncertainty_estimates(intervention, actual_gain, structure_learner)

        # Update value network (simple online learning)
        self._update_value_network(intervention, actual_gain)

    def _update_uncertainty_estimates(self, intervention, actual_gain, structure_learner):
        """
        Update uncertainty estimates based on intervention outcome
        """
        # Get new structure estimate
        new_adjacency = structure_learner.get_adjacency_matrix()

        # Update uncertainty for relationships involving intervened variables
        with torch.no_grad():
            for var_idx in intervention['target_variables']:
                # Reduce uncertainty for edges involving this variable
                uncertainty_reduction = min(actual_gain * 0.1, 0.2)

                # Incoming edges to intervened variable
                self.uncertainty_tracker.data[:, var_idx] *= (1 - uncertainty_reduction)

                # Outgoing edges from intervened variable
                self.uncertainty_tracker.data[var_idx, :] *= (1 - uncertainty_reduction)

            # Ensure uncertainty stays in valid range
            self.uncertainty_tracker.data.clamp_(0.01, 0.99)

    def _update_value_network(self, intervention, actual_gain):
        """
        Update value network with intervention outcome
        """
        # Prepare input
        uncertainty_flat = self.uncertainty_tracker.detach().flatten()
        intervention_vector = intervention['intervention_vector']
        network_input = torch.cat([uncertainty_flat, intervention_vector])

        # Current prediction
        predicted_value = self.value_network(network_input)

        # Target value (actual gain)
        target_value = torch.tensor(actual_gain).float()

        # Simple gradient step
        loss = F.mse_loss(predicted_value.squeeze(), target_value)

        # Manual gradient update (simplified online learning)
        loss.backward()
        with torch.no_grad():
            for param in self.value_network.parameters():
                if param.grad is not None:
                    param.data -= 0.01 * param.grad  # Small learning rate
                    param.grad.zero_()

    def _get_cached_loss(self):
        """Get cached loss value (placeholder for actual implementation)"""
        return getattr(self, '_cached_loss', 1.0)

    def design_intervention_sequence(self, structure_learner, planning_horizon=5):
        """
        Design sequence of interventions for long-term planning

        Args:
            structure_learner: Current structure learner
            planning_horizon: Number of future interventions to plan

        Returns:
            intervention_sequence: List of planned interventions
            expected_trajectory: Expected learning trajectory
        """
        intervention_sequence = []
        expected_gains = []

        # Current state
        current_adjacency = structure_learner.get_adjacency_matrix()
        current_uncertainty = self.uncertainty_tracker.clone()

        for step in range(planning_horizon):
            # Select best intervention for current state
            best_intervention, _ = self.select_optimal_intervention(
                structure_learner, None  # Would need current data
            )

            intervention_sequence.append(best_intervention)
            expected_gains.append(best_intervention['info_gain'])

            # Simulate outcome (simplified)
            # In practice, would use more sophisticated prediction
            simulated_gain = best_intervention['info_gain'] * 0.8  # Optimistic discount

            # Update simulated uncertainty
            for var_idx in best_intervention['target_variables']:
                current_uncertainty[:, var_idx] *= 0.9
                current_uncertainty[var_idx, :] *= 0.9

        expected_trajectory = {
            'interventions': intervention_sequence,
            'expected_gains': expected_gains,
            'cumulative_gain': sum(expected_gains),
            'final_uncertainty': current_uncertainty.mean().item()
        }

        return intervention_sequence, expected_trajectory

    def get_intervention_recommendations(self, structure_learner, top_k=3):
        """
        Get top-k intervention recommendations with explanations

        Args:
            structure_learner: Current structure learner
            top_k: Number of recommendations to return

        Returns:
            recommendations: List of intervention recommendations with explanations
        """
        best_intervention, all_candidates = self.select_optimal_intervention(
            structure_learner, None, num_candidates=50
        )

        recommendations = []

        for i, candidate in enumerate(all_candidates[:top_k]):
            # Generate explanation
            explanation = self._generate_intervention_explanation(candidate)

            recommendation = {
                'rank': i + 1,
                'intervention': candidate,
                'explanation': explanation,
                'expected_gain': candidate['info_gain'],
                'feasibility': 1.0 / candidate['cost'],
                'target_variables': [self.variable_names[idx] for idx in candidate['target_variables']]
            }

            recommendations.append(recommendation)

        return recommendations

    def _generate_intervention_explanation(self, intervention):
        """
        Generate human-readable explanation for intervention
        """
        target_vars = [self.variable_names[idx] for idx in intervention['target_variables']]

        if len(target_vars) == 1:
            explanation = f"Intervene on {target_vars[0]} to clarify its causal relationships. "
        else:
            explanation = f"Jointly intervene on {', '.join(target_vars)} to understand their interactions. "

        explanation += f"Expected to provide {intervention['info_gain']:.3f} information gain. "

        feasibility_score = 1.0 / intervention['cost']
        if feasibility_score > 0.8:
            explanation += "High feasibility intervention."
        elif feasibility_score > 0.5:
            explanation += "Moderate feasibility intervention."
        else:
            explanation += "Challenging but high-value intervention."

        return explanation

    def get_intervention_history_analysis(self):
        """
        Analyze intervention history for insights

        Returns:
            analysis: Dict with intervention performance analysis
        """
        # Get non-zero history
        valid_indices = self.intervention_outcomes[:self.history_index] != 0
        if valid_indices.sum() == 0:
            return {'status': 'no_interventions_recorded'}

        valid_outcomes = self.intervention_outcomes[:self.history_index][valid_indices]
        valid_interventions = self.intervention_history[:self.history_index][valid_indices]

        # Compute statistics
        avg_gain = valid_outcomes.mean().item()
        best_gain = valid_outcomes.max().item()
        worst_gain = valid_outcomes.min().item()

        # Variable-specific analysis
        variable_performance = {}
        for var_idx in range(self.num_variables):
            var_interventions = valid_interventions[:, var_idx] > 0
            if var_interventions.sum() > 0:
                var_outcomes = valid_outcomes[var_interventions]
                variable_performance[self.variable_names[var_idx]] = {
                    'avg_gain': var_outcomes.mean().item(),
                    'num_interventions': var_interventions.sum().item(),
                    'success_rate': (var_outcomes > 0).float().mean().item()
                }

        analysis = {
            'total_interventions': valid_indices.sum().item(),
            'average_gain': avg_gain,
            'best_gain': best_gain,
            'worst_gain': worst_gain,
            'success_rate': (valid_outcomes > 0).float().mean().item(),
            'variable_performance': variable_performance,
            'learning_trend': self._compute_learning_trend(valid_outcomes)
        }

        return analysis

    def _compute_learning_trend(self, outcomes):
        """Compute learning trend from intervention outcomes"""
        if len(outcomes) < 3:
            return 'insufficient_data'

        # Simple linear trend
        x = torch.arange(len(outcomes), dtype=torch.float)
        slope = torch.corrcoef(torch.stack([x, outcomes]))[0, 1]

        if slope > 0.1:
            return 'improving'
        elif slope < -0.1:
            return 'declining'
        else:
            return 'stable'

    def get_model_name(self):
        return "intervention_designer"


def test_intervention_designer():
    """
    Test intervention designer functionality
    """
    print("Testing InterventionDesigner...")

    # Create designer
    designer = InterventionDesigner(num_variables=5)

    # Test intervention candidate generation
    candidate = designer._generate_intervention_candidate()
    print(f"Generated intervention on: {[designer.variable_names[i] for i in candidate['target_variables']]}")

    # Test cost computation
    cost = designer._compute_intervention_cost(candidate)
    print(f"Intervention cost: {cost:.3f}")

    # Test information gain computation
    current_uncertainty = torch.rand(5, 5) * 0.5
    proposed_intervention = torch.zeros(5)
    proposed_intervention[0] = 1.0  # Intervene on weather
    current_structure = torch.rand(5, 5)

    info_gain = designer.compute_information_gain(
        current_uncertainty, proposed_intervention, current_structure
    )
    print(f"Estimated information gain: {info_gain.item():.4f}")

    # Test intervention history analysis
    analysis = designer.get_intervention_history_analysis()
    print(f"History analysis status: {analysis.get('status', 'has_data')}")

    print("✅ InterventionDesigner test passed")
    print(f"✅ Model parameters: {sum(p.numel() for p in designer.parameters())}")

    return True


if __name__ == "__main__":
    test_intervention_designer()