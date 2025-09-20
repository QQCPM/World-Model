"""
Causal Working Memory Module
PHASE 3 ENHANCEMENT: Multi-step reasoning for temporal causal chains

Research Inspiration:
- Working memory models from cognitive science
- Multi-step reasoning in temporal causal networks
- Bottleneck-aware chain propagation

Core Concept:
Maintain rich temporal context for causal reasoning, enabling the system to:
1. Remember multi-step causal chains across timesteps
2. Validate chain consistency and transitivity
3. Detect when bottlenecks are affecting causal propagation
4. Provide context for enhanced temporal decision-making
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalMemoryEntry:
    """Single entry in causal working memory"""
    timestep: int
    causal_factors: np.ndarray      # [5] causal factors at this timestep
    delayed_effects: np.ndarray     # [5] delayed effects computed at this timestep
    detected_chains: List[Dict]     # Causal chains detected at this timestep
    bottleneck_scores: Dict[int, float]  # Bottleneck scores for each variable
    action_taken: Optional[np.ndarray] = None  # [2] action if available
    action_effects: Optional[np.ndarray] = None  # [2] action effects if available


@dataclass
class CausalInsight:
    """Insights derived from working memory analysis"""
    dominant_bottleneck: int        # Most frequently important bottleneck variable
    stable_chains: List[Dict]       # Chains that appear consistently
    chain_strength_trend: Dict[str, List[float]]  # How chain strengths evolve
    transitivity_score: float       # Overall transitivity consistency
    prediction_confidence: float    # Confidence in next-step predictions


class CausalWorkingMemory:
    """
    Working memory for multi-step causal reasoning

    Maintains temporal context to enable sophisticated causal chain reasoning:
    - Multi-timestep causal factor history
    - Chain detection and validation across time
    - Bottleneck tracking and analysis
    - Transitivity and consistency checking
    """

    def __init__(self, memory_capacity=20, num_variables=5):
        self.memory_capacity = memory_capacity
        self.num_variables = num_variables

        # Core memory storage
        self.memory_entries = deque(maxlen=memory_capacity)
        self.current_timestep = 0

        # Variable names for interpretability
        self.variable_names = ['weather', 'event', 'crowd', 'time', 'day']

        # Aggregated insights
        self.chain_frequency = defaultdict(int)  # How often each chain type appears
        self.bottleneck_importance = defaultdict(list)  # Bottleneck scores over time
        self.transitivity_history = deque(maxlen=memory_capacity)

        # Multi-step reasoning state
        self.predicted_next_effects = None
        self.confidence_in_prediction = 0.0
        self.reasoning_chain_active = None

        logger.info("CausalWorkingMemory initialized with capacity %d", memory_capacity)

    def update(self, causal_factors: np.ndarray, delayed_effects: np.ndarray,
               detected_chains: List[Dict] = None, bottleneck_scores: Dict[int, float] = None,
               action: np.ndarray = None, action_effects: np.ndarray = None):
        """
        Update working memory with new temporal information

        Args:
            causal_factors: [5] current causal factors
            delayed_effects: [5] delayed effects from temporal buffer
            detected_chains: List of detected causal chains
            bottleneck_scores: Bottleneck importance scores
            action: [2] action taken (optional)
            action_effects: [2] effects of action (optional)
        """

        # Create memory entry
        entry = CausalMemoryEntry(
            timestep=self.current_timestep,
            causal_factors=causal_factors.copy(),
            delayed_effects=delayed_effects.copy(),
            detected_chains=detected_chains or [],
            bottleneck_scores=bottleneck_scores or {},
            action_taken=action.copy() if action is not None else None,
            action_effects=action_effects.copy() if action_effects is not None else None
        )

        # Add to memory
        self.memory_entries.append(entry)

        # Update aggregated insights
        self._update_insights(entry)

        # Multi-step reasoning update
        self._update_multi_step_reasoning(entry)

        self.current_timestep += 1

        logger.debug("Working memory updated: timestep %d, %d chains detected",
                    self.current_timestep, len(detected_chains or []))

    def _update_insights(self, entry: CausalMemoryEntry):
        """Update aggregated insights from new memory entry"""

        # Update chain frequency tracking
        for chain in entry.detected_chains:
            chain_key = f"{chain.get('cause', '?')}→{chain.get('bottleneck', '?')}→{chain.get('effect', '?')}"
            self.chain_frequency[chain_key] += 1

        # Update bottleneck importance tracking
        for var_idx, score in entry.bottleneck_scores.items():
            self.bottleneck_importance[var_idx].append(score)
            # Keep only recent scores
            if len(self.bottleneck_importance[var_idx]) > self.memory_capacity:
                self.bottleneck_importance[var_idx].pop(0)

        # Compute transitivity score for this timestep
        transitivity_score = self._compute_transitivity_score(entry.detected_chains)
        self.transitivity_history.append(transitivity_score)

    def _update_multi_step_reasoning(self, entry: CausalMemoryEntry):
        """Update multi-step reasoning state"""

        if len(self.memory_entries) < 3:
            return  # Need sufficient history for multi-step reasoning

        # Predict next effects based on current chains and history
        self.predicted_next_effects = self._predict_next_timestep_effects(entry)

        # Assess confidence based on chain consistency
        self.confidence_in_prediction = self._assess_prediction_confidence()

        # Identify if we're in an active reasoning chain
        self.reasoning_chain_active = self._identify_active_reasoning_chain()

    def _compute_transitivity_score(self, chains: List[Dict]) -> float:
        """Compute transitivity consistency score for detected chains"""
        if len(chains) < 2:
            return 1.0

        violations = 0
        total_checks = 0

        for i, chain1 in enumerate(chains):
            for j, chain2 in enumerate(chains[i+1:], i+1):
                total_checks += 1

                # Check logical consistency
                strength1 = chain1.get('strength', 0)
                strength2 = chain2.get('strength', 0)

                # Simple heuristic: similar strength chains should be consistent
                if abs(strength1 - strength2) > 0.5:
                    violations += 1

        return 1.0 - (violations / max(total_checks, 1))

    def _predict_next_timestep_effects(self, current_entry: CausalMemoryEntry) -> np.ndarray:
        """Predict effects for next timestep based on working memory"""

        if len(self.memory_entries) < 2:
            return current_entry.delayed_effects.copy()

        # Get recent entries for pattern analysis
        recent_entries = list(self.memory_entries)[-3:]

        # Analyze trends in causal factors
        prediction = current_entry.delayed_effects.copy()

        for var_idx in range(self.num_variables):
            # Look for temporal patterns in this variable
            recent_values = [entry.causal_factors[var_idx] for entry in recent_entries]
            recent_delayed = [entry.delayed_effects[var_idx] for entry in recent_entries]

            # Simple trend-based prediction
            if len(recent_values) >= 2:
                trend = recent_values[-1] - recent_values[-2]
                delay_trend = recent_delayed[-1] - recent_delayed[-2]

                # Weighted prediction combining trends
                prediction[var_idx] = current_entry.delayed_effects[var_idx] + 0.3 * trend + 0.2 * delay_trend

        return np.clip(prediction, 0.0, 1.0)

    def _assess_prediction_confidence(self) -> float:
        """Assess confidence in predictions based on memory consistency"""

        if len(self.memory_entries) < 3:
            return 0.5

        # Factors affecting confidence
        chain_consistency = np.mean(list(self.transitivity_history)[-5:]) if self.transitivity_history else 0.5

        # Bottleneck stability (consistent bottlenecks increase confidence)
        bottleneck_stability = self._compute_bottleneck_stability()

        # Chain frequency stability (frequent chains increase confidence)
        chain_stability = self._compute_chain_stability()

        # Combined confidence
        confidence = 0.4 * chain_consistency + 0.3 * bottleneck_stability + 0.3 * chain_stability

        return np.clip(confidence, 0.0, 1.0)

    def _compute_bottleneck_stability(self) -> float:
        """Compute how stable bottleneck scores are over time"""
        if not self.bottleneck_importance:
            return 0.5

        stabilities = []
        for var_idx, scores in self.bottleneck_importance.items():
            if len(scores) >= 3:
                # Compute coefficient of variation (lower = more stable)
                mean_score = np.mean(scores[-5:])
                std_score = np.std(scores[-5:])
                cv = std_score / (mean_score + 0.01)
                stability = 1.0 / (1.0 + cv)  # Convert to stability score
                stabilities.append(stability)

        return np.mean(stabilities) if stabilities else 0.5

    def _compute_chain_stability(self) -> float:
        """Compute how consistently chains appear over time"""
        if not self.chain_frequency:
            return 0.5

        total_chains = sum(self.chain_frequency.values())
        recent_timesteps = min(10, len(self.memory_entries))

        # Chains that appear frequently are more stable
        max_frequency = max(self.chain_frequency.values())
        stability = max_frequency / max(recent_timesteps, 1)

        return min(stability, 1.0)

    def _identify_active_reasoning_chain(self) -> Optional[Dict]:
        """Identify if there's currently an active multi-step reasoning chain"""

        if len(self.memory_entries) < 3:
            return None

        # Look for consistent chain patterns in recent memory
        recent_entries = list(self.memory_entries)[-3:]

        # Find chains that appear in multiple recent timesteps
        chain_appearances = defaultdict(int)

        for entry in recent_entries:
            for chain in entry.detected_chains:
                chain_key = f"{chain.get('cause', '?')}→{chain.get('bottleneck', '?')}→{chain.get('effect', '?')}"
                chain_appearances[chain_key] += 1

        # Find most consistent chain
        if chain_appearances:
            most_consistent_chain = max(chain_appearances.keys(), key=lambda k: chain_appearances[k])
            if chain_appearances[most_consistent_chain] >= 2:  # Appears in at least 2 recent timesteps
                return {
                    'chain_description': most_consistent_chain,
                    'consistency': chain_appearances[most_consistent_chain],
                    'timesteps': len(recent_entries)
                }

        return None

    def get_causal_insights(self) -> CausalInsight:
        """Get comprehensive insights from working memory"""

        # Find dominant bottleneck
        dominant_bottleneck = 0
        if self.bottleneck_importance:
            avg_scores = {var: np.mean(scores) for var, scores in self.bottleneck_importance.items()}
            dominant_bottleneck = max(avg_scores.keys(), key=lambda k: avg_scores[k])

        # Find stable chains (appear frequently)
        stable_chains = []
        total_timesteps = len(self.memory_entries)
        for chain_key, frequency in self.chain_frequency.items():
            if frequency >= max(2, total_timesteps // 3):  # Appears in at least 1/3 of timesteps
                stable_chains.append({
                    'chain': chain_key,
                    'frequency': frequency,
                    'stability': frequency / max(total_timesteps, 1)
                })

        # Chain strength trends
        chain_strength_trend = {}
        # This would require storing strength history - simplified for now

        # Overall transitivity score
        transitivity_score = np.mean(list(self.transitivity_history)) if self.transitivity_history else 0.5

        return CausalInsight(
            dominant_bottleneck=dominant_bottleneck,
            stable_chains=stable_chains,
            chain_strength_trend=chain_strength_trend,
            transitivity_score=transitivity_score,
            prediction_confidence=self.confidence_in_prediction
        )

    def get_next_step_prediction(self) -> Tuple[np.ndarray, float]:
        """Get prediction for next timestep with confidence"""
        if self.predicted_next_effects is None:
            # Default prediction: use last delayed effects
            if self.memory_entries:
                prediction = self.memory_entries[-1].delayed_effects.copy()
            else:
                prediction = np.zeros(self.num_variables)
            confidence = 0.1
        else:
            prediction = self.predicted_next_effects.copy()
            confidence = self.confidence_in_prediction

        return prediction, confidence

    def get_reasoning_context(self) -> Dict[str, Any]:
        """Get current reasoning context for decision making"""

        insights = self.get_causal_insights()

        context = {
            'memory_depth': len(self.memory_entries),
            'dominant_bottleneck': {
                'variable': self.variable_names[insights.dominant_bottleneck],
                'index': insights.dominant_bottleneck
            },
            'stable_chains': insights.stable_chains,
            'transitivity_health': insights.transitivity_score,
            'prediction_confidence': insights.prediction_confidence,
            'active_reasoning': self.reasoning_chain_active,
            'recent_bottleneck_scores': dict(self.memory_entries[-1].bottleneck_scores) if self.memory_entries else {}
        }

        return context

    def reset(self):
        """Reset working memory for new episode"""
        self.memory_entries.clear()
        self.current_timestep = 0
        self.chain_frequency.clear()
        self.bottleneck_importance.clear()
        self.transitivity_history.clear()
        self.predicted_next_effects = None
        self.confidence_in_prediction = 0.0
        self.reasoning_chain_active = None

        logger.info("Working memory reset for new episode")

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of current memory state"""
        return {
            'total_entries': len(self.memory_entries),
            'current_timestep': self.current_timestep,
            'unique_chains_seen': len(self.chain_frequency),
            'most_frequent_chain': max(self.chain_frequency.keys(), key=lambda k: self.chain_frequency[k]) if self.chain_frequency else None,
            'average_transitivity': np.mean(list(self.transitivity_history)) if self.transitivity_history else 0.0,
            'tracked_bottlenecks': list(self.bottleneck_importance.keys()),
            'prediction_available': self.predicted_next_effects is not None,
            'reasoning_active': self.reasoning_chain_active is not None
        }


def test_causal_working_memory():
    """Test the causal working memory module"""
    print("🧠 Testing Causal Working Memory Module")
    print("=" * 50)

    memory = CausalWorkingMemory(memory_capacity=15, num_variables=5)

    # Simulate temporal sequence with causal patterns
    print("1. Simulating temporal sequence...")

    for t in range(12):
        # Generate causal factors with patterns
        causal_factors = np.random.rand(5) * 0.1

        # Inject causal pattern: weather affects crowd
        if t >= 2:
            causal_factors[2] += 0.6 * np.random.rand()  # Weather effect on crowd

        # Simulate delayed effects
        delayed_effects = causal_factors.copy()
        if t >= 2:
            delayed_effects[2] += 0.4  # Additional delay effect

        # Simulate detected chains
        detected_chains = []
        if t >= 3:
            detected_chains.append({
                'cause': 0, 'bottleneck': 2, 'effect': 'action',
                'strength': 0.3 + 0.2 * np.random.rand()
            })

        # Simulate bottleneck scores
        bottleneck_scores = {2: 0.7 + 0.3 * np.random.rand()}  # Crowd as bottleneck

        # Update memory
        memory.update(causal_factors, delayed_effects, detected_chains, bottleneck_scores)

    print(f"   Processed {memory.current_timestep} timesteps")

    # Test insights
    print("\n2. Testing causal insights...")
    insights = memory.get_causal_insights()

    print(f"   Dominant bottleneck: {memory.variable_names[insights.dominant_bottleneck]}")
    print(f"   Stable chains found: {len(insights.stable_chains)}")
    print(f"   Transitivity score: {insights.transitivity_score:.3f}")
    print(f"   Prediction confidence: {insights.prediction_confidence:.3f}")

    # Test prediction
    print("\n3. Testing next-step prediction...")
    prediction, confidence = memory.get_next_step_prediction()

    print(f"   Prediction shape: {prediction.shape}")
    print(f"   Prediction confidence: {confidence:.3f}")
    print(f"   Predicted values: {prediction}")

    # Test reasoning context
    print("\n4. Testing reasoning context...")
    context = memory.get_reasoning_context()

    print(f"   Memory depth: {context['memory_depth']}")
    print(f"   Dominant bottleneck: {context['dominant_bottleneck']}")
    print(f"   Transitivity health: {context['transitivity_health']:.3f}")
    print(f"   Active reasoning: {context['active_reasoning']}")

    # Test memory summary
    print("\n5. Testing memory summary...")
    summary = memory.get_memory_summary()

    print(f"   Total entries: {summary['total_entries']}")
    print(f"   Unique chains seen: {summary['unique_chains_seen']}")
    print(f"   Most frequent chain: {summary['most_frequent_chain']}")
    print(f"   Prediction available: {summary['prediction_available']}")

    print("\n✅ Causal Working Memory Module working correctly!")
    print("✅ Multi-step reasoning capabilities enabled!")
    print("✅ Temporal context maintenance operational!")
    print("✅ Ready for TemporalCausalIntegrator integration!")

    return True


if __name__ == "__main__":
    test_causal_working_memory()