# Causal Architectures Module
# Research-validated neural architectures for genuine causal reasoning

# Original components (for backward compatibility)
from .dual_pathway_gru import DualPathwayCausalGRU as OriginalDualPathwayCausalGRU, CausalLoss
from .structure_learner import CausalStructureLearner as OriginalCausalStructureLearner

# Enhanced components (now the defaults)
from .enhanced_dual_pathway_gru import EnhancedDualPathwayCausalGRU, create_enhanced_dual_pathway_model
from .enhanced_structure_learner import EnhancedCausalStructureLearner, create_enhanced_structure_learner

# Counterfactual wrapper (newly integrated)
from .counterfactual_wrapper import CounterfactualDynamicsWrapper, InputDimensionAdapter

# Phase 2 Advanced Components
from .domain_invariant_learner import (
    DomainInvariantCausalLearner, DomainInvariantFeatureExtractor,
    CausalRelationshipAbstractor, DomainAdaptationLayer,
    DomainAdaptationConfig, create_domain_invariant_learner
)
from .meta_causal_reasoner import (
    MetaCausalReasoner, CausalStructureChangeDetector,
    MetaCausalPatternLearner, CausalEvolutionPredictor,
    MetaCausalConfig, create_meta_causal_reasoner
)

# Other components
from .intervention_designer import InterventionDesigner
from .causal_mechanisms import CausalMechanismModules
from .notears_constraints import NOTEARSConstraint, AdaptiveDAGConstraint

# Make enhanced versions the default exports (backward compatible)
DualPathwayCausalGRU = EnhancedDualPathwayCausalGRU
CausalStructureLearner = EnhancedCausalStructureLearner
create_dual_pathway_model = create_enhanced_dual_pathway_model
create_campus_structure_learner = create_enhanced_structure_learner

__all__ = [
    # Default exports (enhanced versions)
    'DualPathwayCausalGRU',
    'CausalStructureLearner',
    'InterventionDesigner',
    'CausalMechanismModules',
    'NOTEARSConstraint',
    'AdaptiveDAGConstraint',
    'CausalLoss',
    'create_dual_pathway_model',
    'create_campus_structure_learner',

    # Enhanced component explicit exports
    'EnhancedDualPathwayCausalGRU',
    'EnhancedCausalStructureLearner',
    'create_enhanced_dual_pathway_model',
    'create_enhanced_structure_learner',

    # Counterfactual wrapper
    'CounterfactualDynamicsWrapper',
    'InputDimensionAdapter',

    # Phase 2 Advanced Components
    'DomainInvariantCausalLearner',
    'DomainInvariantFeatureExtractor',
    'CausalRelationshipAbstractor',
    'DomainAdaptationLayer',
    'DomainAdaptationConfig',
    'create_domain_invariant_learner',
    'MetaCausalReasoner',
    'CausalStructureChangeDetector',
    'MetaCausalPatternLearner',
    'CausalEvolutionPredictor',
    'MetaCausalConfig',
    'create_meta_causal_reasoner',

    # Original components (for explicit access)
    'OriginalDualPathwayCausalGRU',
    'OriginalCausalStructureLearner'
]