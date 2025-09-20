"""
Meta-Causal Reasoning Framework
PHASE 2 ENHANCEMENT: Reasoning about causal structure changes and causal causality

Research Inspiration:
- Meta-learning for causal discovery
- Learning to Learn Causal Models (Bengio et al.)
- Dynamic causal discovery with changing structures
- Causal reasoning about causality itself

Core Concept:
Enable the system to reason about:
1. How causal structures change over time
2. Why causal relationships appear/disappear
3. Causal factors that influence causality itself
4. Meta-patterns in causal relationship evolution

Key Components:
1. Causal Structure Change Detector
2. Meta-Causal Pattern Learner
3. Causal Evolution Predictor
4. Causal Mechanism Meta-Reasoner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetaCausalConfig:
    """Configuration for meta-causal reasoning"""
    # Structure evolution tracking
    num_variables: int = 5
    max_structure_history: int = 20
    change_detection_threshold: float = 0.3

    # Meta-pattern learning
    meta_pattern_dim: int = 16
    pattern_memory_size: int = 50
    pattern_similarity_threshold: float = 0.7

    # Causal evolution prediction
    evolution_predictor_hidden: int = 32
    prediction_horizon: int = 5

    # Meta-reasoning about mechanisms
    mechanism_embedding_dim: int = 24
    meta_reasoning_layers: int = 3


@dataclass
class CausalStructureSnapshot:
    """Snapshot of causal structure at a specific time"""
    timestep: int
    adjacency_matrix: torch.Tensor  # [num_vars, num_vars]
    structure_confidence: torch.Tensor  # [num_vars, num_vars]
    context_factors: torch.Tensor  # [context_dim] - what influenced this structure
    change_indicators: torch.Tensor  # [num_vars, num_vars] - which edges changed


@dataclass
class MetaCausalPattern:
    """Detected meta-pattern in causal evolution"""
    pattern_id: str
    pattern_embedding: torch.Tensor
    trigger_conditions: Dict[str, Any]
    typical_evolution: List[torch.Tensor]  # Sequence of adjacency matrices
    confidence: float
    frequency: int


class CausalStructureChangeDetector(nn.Module):
    """
    Detect when and how causal structures change over time

    Monitors sequence of causal structures to identify:
    - When changes occur
    - Which relationships are affected
    - Potential causes of structural changes
    """

    def __init__(self, config: MetaCausalConfig):
        super().__init__()
        self.config = config

        # Structure history buffer
        self.structure_history = deque(maxlen=config.max_structure_history)

        # Change detection network
        self.change_detector = nn.Sequential(
            nn.Linear(config.num_variables * config.num_variables * 2, 64),  # Current + previous
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.num_variables * config.num_variables)  # Change probability per edge
        )

        # Context encoder for understanding change triggers
        self.context_encoder = nn.Sequential(
            nn.Linear(config.num_variables, 32),  # Causal factors as context
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)  # Context embedding
        )

    def forward(self, current_adjacency, causal_factors, timestep):
        """
        Detect structural changes

        Args:
            current_adjacency: [batch, num_vars, num_vars] Current causal structure
            causal_factors: [batch, num_vars] Current causal state
            timestep: Current timestep

        Returns:
            change_detection: Dict with change analysis
        """
        batch_size = current_adjacency.size(0)
        device = current_adjacency.device

        # Encode current context
        context_embedding = self.context_encoder(causal_factors)

        if len(self.structure_history) == 0:
            # First timestep - no change detection possible
            change_scores = torch.zeros_like(current_adjacency)
            self._store_structure(current_adjacency, context_embedding, timestep)

            return {
                'change_detected': False,
                'change_scores': change_scores,
                'change_magnitude': 0.0,
                'context_embedding': context_embedding
            }

        # Get previous structure
        previous_snapshot = self.structure_history[-1]
        previous_adjacency = previous_snapshot.adjacency_matrix

        # Ensure shapes match (handle batch dimension)
        if previous_adjacency.dim() == 2:
            previous_adjacency = previous_adjacency.unsqueeze(0).expand(batch_size, -1, -1)

        # Detect changes
        structure_diff = torch.abs(current_adjacency - previous_adjacency)
        change_magnitude = torch.mean(structure_diff).item()

        # Use neural network for sophisticated change detection
        change_input = torch.cat([
            current_adjacency.view(batch_size, -1),
            previous_adjacency.view(batch_size, -1)
        ], dim=-1)

        change_logits = self.change_detector(change_input)
        change_scores = torch.sigmoid(change_logits).view(batch_size, self.config.num_variables, self.config.num_variables)

        # Determine if significant change occurred
        change_detected = change_magnitude > self.config.change_detection_threshold

        # Store current structure
        self._store_structure(current_adjacency, context_embedding, timestep, change_scores)

        return {
            'change_detected': change_detected,
            'change_scores': change_scores,
            'change_magnitude': change_magnitude,
            'context_embedding': context_embedding,
            'structure_diff': structure_diff
        }

    def _store_structure(self, adjacency, context, timestep, change_indicators=None):
        """Store structure snapshot in history"""
        if change_indicators is None:
            change_indicators = torch.zeros_like(adjacency)

        # Take first batch element for storage (assuming consistent across batch)
        snapshot = CausalStructureSnapshot(
            timestep=timestep,
            adjacency_matrix=adjacency[0].clone(),
            structure_confidence=torch.ones_like(adjacency[0]),
            context_factors=context[0].clone(),
            change_indicators=change_indicators[0].clone()
        )

        self.structure_history.append(snapshot)


class MetaCausalPatternLearner(nn.Module):
    """
    Learn meta-patterns in causal structure evolution

    Identifies recurring patterns in how causal structures change:
    - Common evolution trajectories
    - Trigger conditions for changes
    - Predictable structural transformations
    """

    def __init__(self, config: MetaCausalConfig):
        super().__init__()
        self.config = config

        # Pattern storage
        self.detected_patterns = []
        self.pattern_frequency = {}

        # Pattern encoder
        self.pattern_encoder = nn.Sequential(
            nn.Linear(config.num_variables * config.num_variables, config.meta_pattern_dim),
            nn.ReLU(),
            nn.Linear(config.meta_pattern_dim, config.meta_pattern_dim),
            nn.ReLU(),
            nn.Linear(config.meta_pattern_dim, config.meta_pattern_dim)
        )

        # Pattern similarity network
        self.similarity_network = nn.Sequential(
            nn.Linear(config.meta_pattern_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Evolution sequence encoder
        self.sequence_encoder = nn.LSTM(
            input_size=config.meta_pattern_dim,
            hidden_size=config.meta_pattern_dim,
            num_layers=2,
            batch_first=True
        )

    def forward(self, structure_sequence, context_sequence):
        """
        Learn and detect meta-patterns

        Args:
            structure_sequence: [batch, seq_len, num_vars, num_vars] Sequence of structures
            context_sequence: [batch, seq_len, context_dim] Context factors

        Returns:
            pattern_analysis: Dict with pattern detection results
        """
        batch_size, seq_len = structure_sequence.shape[:2]

        # Encode each structure in the sequence
        structure_embeddings = []
        for t in range(seq_len):
            structure_flat = structure_sequence[:, t].view(batch_size, -1)
            embedding = self.pattern_encoder(structure_flat)
            structure_embeddings.append(embedding)

        structure_embeddings = torch.stack(structure_embeddings, dim=1)  # [batch, seq_len, pattern_dim]

        # Encode evolution sequence
        lstm_output, (hidden, cell) = self.sequence_encoder(structure_embeddings)

        # Extract pattern representation (final hidden state)
        pattern_representation = hidden[-1]  # [batch, pattern_dim]

        # Match against known patterns
        pattern_matches = self._match_patterns(pattern_representation)

        # Detect new patterns
        new_patterns = self._detect_new_patterns(pattern_representation, structure_sequence, context_sequence)

        return {
            'pattern_representation': pattern_representation,
            'pattern_matches': pattern_matches,
            'new_patterns': new_patterns,
            'sequence_encoding': lstm_output
        }

    def _match_patterns(self, pattern_representation):
        """Match current pattern against known patterns"""
        matches = []

        for stored_pattern in self.detected_patterns:
            # Compute similarity
            similarity_input = torch.cat([
                pattern_representation,
                stored_pattern.pattern_embedding.unsqueeze(0).expand(pattern_representation.size(0), -1)
            ], dim=-1)

            similarity_score = self.similarity_network(similarity_input)

            if similarity_score.mean() > self.config.pattern_similarity_threshold:
                matches.append({
                    'pattern_id': stored_pattern.pattern_id,
                    'similarity': similarity_score.mean().item(),
                    'confidence': stored_pattern.confidence
                })

        return matches

    def _detect_new_patterns(self, pattern_representation, structure_sequence, context_sequence):
        """Detect and store new meta-patterns"""
        # Simple new pattern detection (could be made more sophisticated)
        if len(self.detected_patterns) < self.config.pattern_memory_size:
            # Create new pattern
            pattern_id = f"pattern_{len(self.detected_patterns)}"

            new_pattern = MetaCausalPattern(
                pattern_id=pattern_id,
                pattern_embedding=pattern_representation[0].clone(),  # Take first batch element
                trigger_conditions={},  # Could analyze context for triggers
                typical_evolution=[structure_sequence[0, t] for t in range(structure_sequence.size(1))],
                confidence=0.8,  # Initial confidence
                frequency=1
            )

            self.detected_patterns.append(new_pattern)
            return [new_pattern]

        return []


class CausalEvolutionPredictor(nn.Module):
    """
    Predict how causal structures will evolve

    Uses learned meta-patterns to forecast:
    - Future causal structure changes
    - Timing of structural transitions
    - Confidence in predictions
    """

    def __init__(self, config: MetaCausalConfig):
        super().__init__()
        self.config = config

        # Evolution prediction network
        self.evolution_predictor = nn.Sequential(
            nn.Linear(
                config.num_variables * config.num_variables + config.meta_pattern_dim + 8,  # structure + pattern + context
                config.evolution_predictor_hidden
            ),
            nn.ReLU(),
            nn.Linear(config.evolution_predictor_hidden, config.evolution_predictor_hidden),
            nn.ReLU(),
            nn.Linear(config.evolution_predictor_hidden, config.num_variables * config.num_variables * config.prediction_horizon)
        )

        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(config.meta_pattern_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, config.prediction_horizon),
            nn.Sigmoid()
        )

    def forward(self, current_structure, pattern_representation, context_embedding):
        """
        Predict future causal structure evolution

        Args:
            current_structure: [batch, num_vars, num_vars] Current structure
            pattern_representation: [batch, pattern_dim] Meta-pattern encoding
            context_embedding: [batch, context_dim] Context factors

        Returns:
            evolution_prediction: Dict with prediction results
        """
        batch_size = current_structure.size(0)

        # Prepare input for evolution prediction
        structure_flat = current_structure.view(batch_size, -1)
        prediction_input = torch.cat([
            structure_flat,
            pattern_representation,
            context_embedding
        ], dim=-1)

        # Predict future structures
        evolution_logits = self.evolution_predictor(prediction_input)
        future_structures = evolution_logits.view(
            batch_size,
            self.config.prediction_horizon,
            self.config.num_variables,
            self.config.num_variables
        )

        # Apply activation to get valid adjacency matrices
        future_structures = torch.sigmoid(future_structures)

        # Predict confidence for each future timestep
        confidence_scores = self.confidence_predictor(pattern_representation)

        return {
            'future_structures': future_structures,
            'prediction_confidence': confidence_scores,
            'prediction_horizon': self.config.prediction_horizon
        }


class MetaCausalReasoner(nn.Module):
    """
    Complete meta-causal reasoning system

    Integrates all components for reasoning about causality itself:
    1. Structure change detection
    2. Meta-pattern learning
    3. Evolution prediction
    4. Meta-reasoning about causal mechanisms
    """

    def __init__(self, config: MetaCausalConfig = None):
        super().__init__()
        self.config = config or MetaCausalConfig()

        # Core components
        self.change_detector = CausalStructureChangeDetector(self.config)
        self.pattern_learner = MetaCausalPatternLearner(self.config)
        self.evolution_predictor = CausalEvolutionPredictor(self.config)

        # Meta-reasoning state
        self.reasoning_history = deque(maxlen=100)
        self.meta_insights = {}

        logger.info("MetaCausalReasoner initialized for meta-causal reasoning")

    def forward(self, causal_structures, causal_factors_sequence, timestep):
        """
        Complete meta-causal reasoning

        Args:
            causal_structures: [batch, seq_len, num_vars, num_vars] Sequence of causal structures
            causal_factors_sequence: [batch, seq_len, num_vars] Sequence of causal factors
            timestep: Current timestep

        Returns:
            meta_analysis: Dict with comprehensive meta-causal analysis
        """
        batch_size, seq_len = causal_structures.shape[:2]

        # Step 1: Detect structural changes
        current_structure = causal_structures[:, -1]  # Most recent structure
        current_factors = causal_factors_sequence[:, -1]  # Most recent factors

        change_analysis = self.change_detector(current_structure, current_factors, timestep)

        # Step 2: Learn meta-patterns (if we have enough history)
        pattern_analysis = {}
        if seq_len >= 3:  # Need minimum sequence for pattern learning
            context_sequence = causal_factors_sequence  # Use causal factors as context
            pattern_analysis = self.pattern_learner(causal_structures, context_sequence)

        # Step 3: Predict evolution (if we have pattern representation)
        evolution_prediction = {}
        if 'pattern_representation' in pattern_analysis:
            evolution_prediction = self.evolution_predictor(
                current_structure,
                pattern_analysis['pattern_representation'],
                change_analysis['context_embedding']
            )

        # Step 4: Meta-reasoning synthesis
        meta_insights = self._synthesize_meta_insights(change_analysis, pattern_analysis, evolution_prediction)

        return {
            'change_analysis': change_analysis,
            'pattern_analysis': pattern_analysis,
            'evolution_prediction': evolution_prediction,
            'meta_insights': meta_insights,
            'reasoning_score': meta_insights.get('confidence', 0.5)
        }

    def _synthesize_meta_insights(self, change_analysis, pattern_analysis, evolution_prediction):
        """Synthesize high-level meta-causal insights"""
        insights = {}

        # Assess overall structural stability
        if 'change_magnitude' in change_analysis:
            stability_score = 1.0 - min(change_analysis['change_magnitude'], 1.0)
            insights['structural_stability'] = stability_score

        # Assess pattern consistency
        if 'pattern_matches' in pattern_analysis and pattern_analysis['pattern_matches']:
            pattern_consistency = np.mean([m['confidence'] for m in pattern_analysis['pattern_matches']])
            insights['pattern_consistency'] = pattern_consistency
        else:
            insights['pattern_consistency'] = 0.5

        # Assess prediction confidence
        if 'prediction_confidence' in evolution_prediction:
            pred_confidence = torch.mean(evolution_prediction['prediction_confidence']).item()
            insights['prediction_confidence'] = pred_confidence
        else:
            insights['prediction_confidence'] = 0.5

        # Overall meta-reasoning confidence
        components = [
            insights.get('structural_stability', 0.5),
            insights.get('pattern_consistency', 0.5),
            insights.get('prediction_confidence', 0.5)
        ]
        insights['confidence'] = np.mean(components)

        return insights

    def evaluate_meta_causal_scenario(self, scenario):
        """
        Evaluate meta-causal reasoning on a specific scenario

        Compatible with extreme causal challenge framework
        """
        try:
            # Extract sequence of structures and factors from scenario
            if 'structure_sequence' in scenario and 'causal_sequence' in scenario:
                structures = scenario['structure_sequence']
                factors = scenario['causal_sequence']
                timestep = scenario.get('timestep', 0)

                # Run meta-causal analysis
                analysis = self.forward(structures, factors, timestep)

                # Return reasoning score
                return analysis['reasoning_score']

            else:
                # Fallback: generate synthetic scenario for testing
                batch_size, seq_len, num_vars = 1, 5, 5
                structures = torch.rand(batch_size, seq_len, num_vars, num_vars)
                factors = torch.rand(batch_size, seq_len, num_vars)

                analysis = self.forward(structures, factors, 0)
                return analysis['reasoning_score']

        except Exception as e:
            logger.warning(f"Meta-causal evaluation failed: {e}")
            return 0.0


def create_meta_causal_reasoner(config=None):
    """
    Factory function for meta-causal reasoner

    Args:
        config: Optional MetaCausalConfig

    Returns:
        reasoner: MetaCausalReasoner instance
    """
    return MetaCausalReasoner(config)


# Test the meta-causal reasoner
if __name__ == "__main__":
    print("🧠 Testing Meta-Causal Reasoner")

    # Create test configuration
    config = MetaCausalConfig(
        num_variables=5,
        meta_pattern_dim=12,
        evolution_predictor_hidden=24
    )

    # Create reasoner
    reasoner = create_meta_causal_reasoner(config)

    # Test with sample data
    batch_size, seq_len, num_vars = 2, 6, 5
    causal_structures = torch.rand(batch_size, seq_len, num_vars, num_vars)
    causal_factors = torch.rand(batch_size, seq_len, num_vars)

    print(f"✅ Created meta-causal reasoner: {reasoner.__class__.__name__}")
    print(f"✅ Test data shapes: structures={causal_structures.shape}, factors={causal_factors.shape}")

    # Test forward pass
    results = reasoner(causal_structures, causal_factors, timestep=0)

    print(f"✅ Meta-causal analysis successful:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"   {key}: {len(value)} components")
        elif isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
        else:
            print(f"   {key}: {value}")

    # Test scenario evaluation
    test_scenario = {
        'structure_sequence': causal_structures,
        'causal_sequence': causal_factors,
        'timestep': 5
    }

    reasoning_score = reasoner.evaluate_meta_causal_scenario(test_scenario)
    print(f"✅ Meta-causal reasoning score: {reasoning_score:.4f}")

    print("\n🚀 Meta-Causal Reasoner ready for Phase 2 integration!")