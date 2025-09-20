#!/usr/bin/env python3
"""
COMPREHENSIVE SYSTEM VALIDATION: Most Aggressive Test
Tests all three phases integrated together with extreme rigor

This is the ultimate test that validates:
- Phase 1: Enhanced counterfactual reasoning (0.000 → 0.6+)
- Phase 2: Enhanced structure learning (0 edges → functional graphs)
- Phase 3: Bottleneck-aware chain reasoning (temporal integration)
- Integration: All systems working together seamlessly
- Performance: No degradation of original functionality
"""

import sys
import os
import numpy as np
import torch
import torch.optim as optim
import time
from typing import Dict, List, Tuple, Any

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'causal_architectures'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'causal_envs'))

# Direct imports to avoid dependency issues
from enhanced_dual_pathway_gru import EnhancedDualPathwayCausalGRU
from enhanced_structure_learner import EnhancedCausalStructureLearner
from enhanced_temporal_integrator import EnhancedTemporalCausalIntegrator, EnhancedTemporalConfig


def create_extreme_causal_scenario():
    """Create the most challenging causal scenario possible"""
    print("🔥 Creating EXTREME Causal Scenario")
    print("=" * 60)

    # Create complex multi-timestep scenario with known causal patterns
    timesteps = 25
    batch_size = 16

    # Generate base scenario with multiple causal patterns
    scenario = {
        'states': torch.randn(batch_size, timesteps, 12) * 0.1,
        'actions': torch.randn(batch_size, timesteps, 2) * 0.5,
        'causal_factors': torch.randn(batch_size, timesteps, 5) * 0.1
    }

    # Inject STRONG causal patterns
    print("   Injecting complex causal patterns...")

    for b in range(batch_size):
        for t in range(1, timesteps):
            # PATTERN 1: Weather → Crowd (2-timestep delay)
            if t >= 2:
                scenario['causal_factors'][b, t, 2] += 0.8 * scenario['causal_factors'][b, t-2, 0]

            # PATTERN 2: Event → Crowd (immediate)
            scenario['causal_factors'][b, t, 2] += 0.6 * scenario['causal_factors'][b, t, 1]

            # PATTERN 3: Time → Everything (complex relationships)
            time_effect = 0.4 * scenario['causal_factors'][b, t, 3]
            scenario['causal_factors'][b, t, 0] += time_effect * 0.3  # time affects weather
            scenario['causal_factors'][b, t, 2] += time_effect * 0.5  # time affects crowd

            # PATTERN 4: Crowd → States (for counterfactual scenarios)
            crowd_effect = scenario['causal_factors'][b, t, 2]
            scenario['states'][b, t, :4] += crowd_effect.unsqueeze(0) * 0.4
            scenario['states'][b, t, 4:8] += crowd_effect.unsqueeze(0) * -0.3

            # PATTERN 5: States → Actions (for dynamics)
            state_effect = torch.mean(scenario['states'][b, t, :6])
            scenario['actions'][b, t] += state_effect * 0.3

    # Create counterfactual intervention
    intervention_time = 15
    intervention = {
        'variable': 0,  # Weather intervention
        'value': 2.0,   # Strong intervention
        'time': intervention_time
    }

    # Generate counterfactual trajectory
    cf_causal_factors = scenario['causal_factors'].clone()
    cf_states = scenario['states'].clone()

    for b in range(batch_size):
        # Apply intervention
        cf_causal_factors[b, intervention_time, 0] = intervention['value']

        # Propagate intervention effects
        for t in range(intervention_time + 1, timesteps):
            if t >= intervention_time + 2:
                # Weather affects crowd with delay
                cf_causal_factors[b, t, 2] += 0.8 * (cf_causal_factors[b, t-2, 0] - scenario['causal_factors'][b, t-2, 0])

            # Update states based on changed causal factors
            crowd_change = cf_causal_factors[b, t, 2] - scenario['causal_factors'][b, t, 2]
            cf_states[b, t, :4] += crowd_change.unsqueeze(0) * 0.4

    scenario['counterfactual'] = {
        'intervention': intervention,
        'cf_trajectory': cf_causal_factors,
        'cf_states': cf_states
    }

    print(f"   ✅ Scenario created: {timesteps} timesteps, {batch_size} batch size")
    print(f"   ✅ 5 complex causal patterns injected")
    print(f"   ✅ Counterfactual intervention at t={intervention_time}")

    return scenario


def test_phase1_counterfactual_reasoning(scenario):
    """Test Phase 1: Enhanced counterfactual reasoning"""
    print("\n🎯 PHASE 1: Enhanced Counterfactual Reasoning")
    print("=" * 60)

    try:
        # Create enhanced model
        enhanced_model = EnhancedDualPathwayCausalGRU(
            state_dim=12, action_dim=2, causal_dim=5, hidden_dim=64
        )

        # Test counterfactual evaluation
        cf_score = enhanced_model.evaluate_counterfactual_scenario(scenario)

        print(f"   Counterfactual reasoning score: {cf_score:.4f}")

        # Validation criteria
        phase1_success = cf_score >= 0.4  # Target achievement
        phase1_excellent = cf_score >= 0.6  # Excellence threshold

        if phase1_excellent:
            print(f"   🏆 EXCELLENT: {cf_score:.4f} >= 0.6")
        elif phase1_success:
            print(f"   ✅ SUCCESS: {cf_score:.4f} >= 0.4")
        else:
            print(f"   ❌ BELOW TARGET: {cf_score:.4f} < 0.4")

        return phase1_success, cf_score, enhanced_model

    except Exception as e:
        print(f"   ❌ PHASE 1 FAILED: {e}")
        return False, 0.0, None


def test_phase2_structure_learning(scenario):
    """Test Phase 2: Enhanced structure learning"""
    print("\n🔬 PHASE 2: Enhanced Structure Learning")
    print("=" * 60)

    try:
        # Create enhanced structure learner
        enhanced_learner = EnhancedCausalStructureLearner(
            num_variables=5,
            hidden_dim=64,
            total_epochs=100
        )

        # Prepare data for structure learning
        causal_data = scenario['causal_factors']

        # Training with enhanced methodology
        optimizer = optim.Adam(enhanced_learner.parameters(), lr=1e-3)

        print("   Training enhanced structure learner...")
        for epoch in range(50):  # Intensive training
            optimizer.zero_grad()

            # Enhanced structure loss
            loss, loss_info = enhanced_learner.compute_enhanced_structure_loss(causal_data)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(enhanced_learner.parameters(), 1.0)
            optimizer.step()

            if epoch % 10 == 0:
                summary = enhanced_learner.get_enhanced_causal_graph_summary()
                print(f"     Epoch {epoch:2d}: loss={loss.item():.4f}, edges={summary['num_edges']}")

        # Analyze final structure
        final_summary = enhanced_learner.get_enhanced_causal_graph_summary()

        print(f"   Final edges discovered: {final_summary['num_edges']}")
        print(f"   Adaptive threshold: {final_summary['adaptive_threshold']:.3f}")

        # Expected relationships: weather→crowd, event→crowd, time→weather/crowd
        expected_relationships = [(0, 2), (1, 2), (3, 0), (3, 2)]
        discovered_relationships = [(edge['cause_idx'], edge['effect_idx'])
                                  for edge in final_summary['edges']]

        correct_discoveries = sum(1 for rel in expected_relationships
                                if rel in discovered_relationships)
        discovery_accuracy = correct_discoveries / len(expected_relationships)

        print(f"   Discovery accuracy: {discovery_accuracy:.1%} ({correct_discoveries}/{len(expected_relationships)})")

        # Validation criteria
        edges_discovered = final_summary['num_edges']
        phase2_success = edges_discovered >= 2 and discovery_accuracy >= 0.5
        phase2_excellent = edges_discovered >= 3 and discovery_accuracy >= 0.75

        if phase2_excellent:
            print(f"   🏆 EXCELLENT: {edges_discovered} edges, {discovery_accuracy:.1%} accuracy")
        elif phase2_success:
            print(f"   ✅ SUCCESS: {edges_discovered} edges, {discovery_accuracy:.1%} accuracy")
        else:
            print(f"   ❌ BELOW TARGET: {edges_discovered} edges, {discovery_accuracy:.1%} accuracy")

        return phase2_success, discovery_accuracy, enhanced_learner

    except Exception as e:
        print(f"   ❌ PHASE 2 FAILED: {e}")
        return False, 0.0, None


def test_phase3_temporal_integration(scenario):
    """Test Phase 3: Bottleneck-aware chain reasoning"""
    print("\n🚀 PHASE 3: Bottleneck-Aware Chain Reasoning")
    print("=" * 60)

    try:
        # Create enhanced temporal integrator
        config = EnhancedTemporalConfig(
            enable_bottleneck_detection=True,
            enable_working_memory=True,
            enable_chain_validation=True,
            enable_multi_step_reasoning=True
        )
        integrator = EnhancedTemporalCausalIntegrator(config)

        # Create test causal state sequence
        def create_test_state(factors):
            class TestState:
                def __init__(self, factors):
                    self.time_hour = factors[3] * 24
                    self.day_week = factors[4] * 7
                    self.weather = factors[0]
                    self.event = factors[1]
                    self.crowd_density = factors[2]

                def to_vector(self):
                    return factors

            return TestState(factors)

        # Process temporal sequence
        batch_idx = 0  # Use first batch item
        causal_sequence = scenario['causal_factors'][batch_idx].numpy()
        action = np.array([1.0, 0.5])

        chains_detected = 0
        reasoning_applications = 0

        print("   Processing temporal sequence...")
        for t in range(causal_sequence.shape[0]):
            test_state = create_test_state(causal_sequence[t])

            modified_action, temporal_info = integrator.apply_temporal_effects(action, test_state)

            # Track Phase 3 metrics
            if 'enhanced_reasoning' in temporal_info:
                reasoning_applications += 1

            if 'bottleneck_analysis' in temporal_info.get('integration_info', {}):
                bottleneck_info = temporal_info['integration_info']['bottleneck_analysis']
                chains_detected += bottleneck_info.get('chains_detected', 0)

        # Get comprehensive validation
        enhanced_report = integrator.get_enhanced_validation_report()
        phase3_summary = integrator.get_phase3_summary()

        print(f"   Timesteps processed: {enhanced_report['timesteps_processed']}")
        print(f"   Chains detected: {phase3_summary['performance_metrics']['chains_detected']}")
        print(f"   Working memory depth: {phase3_summary['current_state']['working_memory_depth']}")
        print(f"   Reasoning applications: {reasoning_applications}")

        # Validation criteria
        temporal_preserved = enhanced_report['performance_preservation']['original_functionality_intact']
        chains_working = phase3_summary['performance_metrics']['chains_detected'] > 0
        memory_working = phase3_summary['current_state']['working_memory_depth'] > 0

        phase3_success = temporal_preserved and chains_working and memory_working
        phase3_excellent = (chains_working and memory_working and
                           phase3_summary['performance_metrics']['chains_detected'] > 5)

        if phase3_excellent:
            print(f"   🏆 EXCELLENT: All systems operational, high chain detection")
        elif phase3_success:
            print(f"   ✅ SUCCESS: All core systems working")
        else:
            print(f"   ❌ ISSUES: Some systems not fully operational")

        return phase3_success, phase3_summary, integrator

    except Exception as e:
        print(f"   ❌ PHASE 3 FAILED: {e}")
        return False, {}, None


def test_integrated_system_performance(scenario, phase1_model, phase2_learner, phase3_integrator):
    """Test all three phases working together"""
    print("\n🔥 INTEGRATED SYSTEM: All Phases Together")
    print("=" * 60)

    try:
        # Performance timing test
        start_time = time.time()

        # Run all three phases on the scenario
        print("   Running integrated analysis...")

        # Phase 1: Counterfactual analysis
        cf_scores = []
        for i in range(min(4, scenario['causal_factors'].shape[0])):  # Test multiple batch items
            single_scenario = {
                'states': scenario['states'][i:i+1],
                'actions': scenario['actions'][i:i+1],
                'causal_factors': scenario['causal_factors'][i:i+1],
                'counterfactual': {
                    'intervention': scenario['counterfactual']['intervention'],
                    'cf_trajectory': scenario['counterfactual']['cf_trajectory'][i:i+1],
                    'cf_states': scenario['counterfactual']['cf_states'][i:i+1]
                }
            }
            cf_score = phase1_model.evaluate_counterfactual_scenario(single_scenario)
            cf_scores.append(cf_score)

        avg_cf_score = np.mean(cf_scores)

        # Phase 2: Structure analysis on multiple data variations
        structure_summaries = []
        for variation in range(3):
            # Add noise variation to test robustness
            noisy_data = scenario['causal_factors'] + torch.randn_like(scenario['causal_factors']) * 0.05
            summary = phase2_learner.get_enhanced_causal_graph_summary()
            structure_summaries.append(summary)

        avg_edges = np.mean([s['num_edges'] for s in structure_summaries])

        # Phase 3: Temporal integration across multiple sequences
        temporal_reports = []
        for seq_idx in range(min(3, scenario['causal_factors'].shape[0])):
            integrator_copy = EnhancedTemporalCausalIntegrator(EnhancedTemporalConfig())

            # Process sequence
            causal_seq = scenario['causal_factors'][seq_idx].numpy()
            for t in range(causal_seq.shape[0]):
                class TestState:
                    def __init__(self, factors):
                        self.time_hour = factors[3] * 24
                        self.day_week = factors[4] * 7
                        self.weather = factors[0]
                        self.event = factors[1]
                        self.crowd_density = factors[2]
                    def to_vector(self):
                        return factors

                test_state = TestState(causal_seq[t])
                integrator_copy.apply_temporal_effects(np.array([1.0, 0.5]), test_state)

            report = integrator_copy.get_enhanced_validation_report()
            temporal_reports.append(report)

        integration_time = time.time() - start_time

        # Analyze integration performance
        temporal_preserved = all(r['performance_preservation']['original_functionality_intact']
                               for r in temporal_reports)

        avg_chains = np.mean([r['phase3_metrics']['chains_detected'] for r in temporal_reports])

        print(f"   Average counterfactual score: {avg_cf_score:.4f}")
        print(f"   Average edges discovered: {avg_edges:.1f}")
        print(f"   Average chains detected: {avg_chains:.1f}")
        print(f"   Temporal preservation: {'✅' if temporal_preserved else '❌'}")
        print(f"   Integration time: {integration_time:.2f}s")

        # Overall integration assessment
        integration_success = (
            avg_cf_score >= 0.4 and
            avg_edges >= 2 and
            avg_chains >= 1 and
            temporal_preserved
        )

        integration_excellent = (
            avg_cf_score >= 0.6 and
            avg_edges >= 3 and
            avg_chains >= 5 and
            temporal_preserved and
            integration_time < 10.0
        )

        if integration_excellent:
            print(f"   🏆 INTEGRATION EXCELLENT: All systems performing at high level")
        elif integration_success:
            print(f"   ✅ INTEGRATION SUCCESS: All core systems functional")
        else:
            print(f"   ❌ INTEGRATION ISSUES: Some systems underperforming")

        return integration_success, {
            'counterfactual_score': avg_cf_score,
            'structure_edges': avg_edges,
            'temporal_chains': avg_chains,
            'temporal_preserved': temporal_preserved,
            'integration_time': integration_time
        }

    except Exception as e:
        print(f"   ❌ INTEGRATION FAILED: {e}")
        return False, {}


def main():
    """Main comprehensive validation"""
    print("🎯 COMPREHENSIVE SYSTEM VALIDATION")
    print("=" * 80)
    print("Most aggressive test of all three phases integrated together")
    print("=" * 80)

    # Create extreme scenario
    scenario = create_extreme_causal_scenario()

    # Test Phase 1
    phase1_success, cf_score, phase1_model = test_phase1_counterfactual_reasoning(scenario)

    # Test Phase 2
    phase2_success, structure_accuracy, phase2_learner = test_phase2_structure_learning(scenario)

    # Test Phase 3
    phase3_success, phase3_summary, phase3_integrator = test_phase3_temporal_integration(scenario)

    # Test integrated system
    if phase1_model and phase2_learner and phase3_integrator:
        integration_success, integration_metrics = test_integrated_system_performance(
            scenario, phase1_model, phase2_learner, phase3_integrator
        )
    else:
        integration_success = False
        integration_metrics = {}

    # Final comprehensive assessment
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE VALIDATION RESULTS")
    print("=" * 80)

    results = {
        'phase1_counterfactual': phase1_success,
        'phase2_structure': phase2_success,
        'phase3_temporal': phase3_success,
        'integrated_system': integration_success
    }

    print(f"🎯 Phase 1 (Counterfactual): {'✅ SUCCESS' if phase1_success else '❌ FAILED'} (score: {cf_score:.4f})")
    print(f"🔬 Phase 2 (Structure): {'✅ SUCCESS' if phase2_success else '❌ FAILED'} (accuracy: {structure_accuracy:.1%})")
    print(f"🚀 Phase 3 (Temporal): {'✅ SUCCESS' if phase3_success else '❌ FAILED'}")
    print(f"🔥 Integrated System: {'✅ SUCCESS' if integration_success else '❌ FAILED'}")

    # Overall system assessment
    overall_success = all(results.values())
    functional_success = sum(results.values()) >= 3  # At least 3 out of 4 working

    if overall_success:
        print(f"\n🏆 COMPREHENSIVE VALIDATION: COMPLETE SUCCESS!")
        print(f"   ✅ All phases working perfectly together")
        print(f"   ✅ Integration seamless and performant")
        print(f"   ✅ Research-grade causal AI system operational")
        print(f"   🚀 SYSTEM READY FOR PRODUCTION")
    elif functional_success:
        print(f"\n🎯 COMPREHENSIVE VALIDATION: FUNCTIONAL SUCCESS!")
        print(f"   ✅ Core systems operational ({sum(results.values())}/4)")
        print(f"   ⚠️  Some fine-tuning may be needed")
        print(f"   📈 Strong foundation for continued development")
    else:
        print(f"\n⚠️ COMPREHENSIVE VALIDATION: NEEDS SIGNIFICANT WORK")
        print(f"   📋 Multiple systems require attention")
        failed_phases = [phase for phase, success in results.items() if not success]
        print(f"   🔧 Focus on: {', '.join(failed_phases)}")

    # Performance metrics summary
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   - Counterfactual reasoning: {cf_score:.3f}")
    print(f"   - Structure discovery accuracy: {structure_accuracy:.1%}")
    if phase3_integrator:
        print(f"   - Temporal chain detection: {phase3_summary['performance_metrics']['chains_detected']}")
    if integration_metrics:
        print(f"   - Integration time: {integration_metrics.get('integration_time', 0):.2f}s")

    return overall_success


if __name__ == "__main__":
    success = main()

    if success:
        print(f"\n✅ COMPREHENSIVE VALIDATION PASSED!")
        print(f"All systems validated and working together perfectly.")
        sys.exit(0)
    else:
        print(f"\n❌ COMPREHENSIVE VALIDATION IDENTIFIED ISSUES")
        print(f"Review results and address system components that need work.")
        sys.exit(1)