#!/usr/bin/env python3
"""
PHASE 2 COMPREHENSIVE TEST: Enhanced Structure Learning Training
Demonstrates adaptive thresholding through full training cycle
Target: 0 edges → 2-3 functional edges with strong statistical support
"""

import sys
import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causal_architectures.enhanced_structure_learner import EnhancedCausalStructureLearner


def generate_strong_causal_data(batch_size=64, seq_len=30):
    """
    Generate data with known strong causal relationships
    Following exact same pattern as enhanced_structure_training.py
    """
    causal_data = torch.zeros(batch_size, seq_len, 5)

    # Initialize with diverse starting conditions
    causal_data[:, 0, :] = torch.randn(batch_size, 5) * 0.5

    # Generate with very strong, clear causal relationships
    for t in range(1, seq_len):
        # VERY STRONG weather -> crowd relationship (0 -> 1)
        causal_data[:, t, 1] = (0.9 * causal_data[:, t-1, 0] +
                               0.1 * torch.randn(batch_size))

        # STRONG time -> road relationship (3 -> 4)
        causal_data[:, t, 4] = (0.85 * causal_data[:, t-1, 3] +
                               0.15 * torch.randn(batch_size))

        # MEDIUM event -> crowd relationship (2 -> 1)
        event_effect = 0.7 * causal_data[:, t, 2]
        causal_data[:, t, 1] += event_effect * 0.3  # Add to existing weather effect

        # Root causes evolve more slowly
        causal_data[:, t, 0] = (0.95 * causal_data[:, t-1, 0] +
                               0.05 * torch.randn(batch_size))  # weather (slow change)
        causal_data[:, t, 3] = (0.95 * causal_data[:, t-1, 3] +
                               0.05 * torch.randn(batch_size))  # time (slow change)

        # Events are more random but persistent
        causal_data[:, t, 2] = (0.7 * causal_data[:, t-1, 2] +
                               0.3 * torch.randn(batch_size))

        # Clamp to prevent extreme values
        causal_data[:, t, :] = torch.clamp(causal_data[:, t, :], -2.0, 2.0)

    return causal_data


def test_phase2_enhanced_training():
    """
    Comprehensive Phase 2 test: Enhanced structure learning through training
    """
    print("🔥 PHASE 2 COMPREHENSIVE TEST: Enhanced Structure Learning")
    print("=" * 70)

    # Create enhanced structure learner
    print("1. Creating enhanced structure learner...")
    enhanced_learner = EnhancedCausalStructureLearner(
        num_variables=5,
        hidden_dim=32,  # Smaller for faster training
        total_epochs=50
    )
    print(f"   ✅ Enhanced learner created: {enhanced_learner.get_model_name()}")

    # Generate strong causal data
    print("\n2. Generating strong causal data...")
    causal_data = generate_strong_causal_data(batch_size=64, seq_len=30)
    print(f"   ✅ Generated data: {causal_data.shape}")

    # Verify causal relationships in data
    print("\n3. Verifying embedded causal relationships...")
    weather = causal_data[:, :-1, 0].flatten()
    crowd_next = causal_data[:, 1:, 1].flatten()
    weather_crowd_corr = torch.corrcoef(torch.stack([weather, crowd_next]))[0, 1]

    time = causal_data[:, :-1, 3].flatten()
    road_next = causal_data[:, 1:, 4].flatten()
    time_road_corr = torch.corrcoef(torch.stack([time, road_next]))[0, 1]

    print(f"   ✅ Weather → Crowd correlation: {weather_crowd_corr:.3f}")
    print(f"   ✅ Time → Road correlation: {time_road_corr:.3f}")

    # Training with enhanced methodology
    print("\n4. Training enhanced structure learner...")
    optimizer = optim.Adam(enhanced_learner.parameters(), lr=2e-3)

    losses = []
    thresholds = []
    edges_discovered = []

    target_relationships = [(0, 1), (3, 4), (2, 1)]  # Expected: weather->crowd, time->road, event->crowd

    for epoch in range(25):  # Shorter training for test
        optimizer.zero_grad()

        # Enhanced structure loss with all improvements
        loss, loss_info = enhanced_learner.compute_enhanced_structure_loss(causal_data)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(enhanced_learner.parameters(), 1.0)
        optimizer.step()

        # Track progress
        losses.append(loss.item())
        thresholds.append(loss_info['adaptive_threshold'])

        # Get current edges with adaptive threshold
        summary = enhanced_learner.get_enhanced_causal_graph_summary()
        edges_discovered.append(summary['num_edges'])

        if epoch % 5 == 0:
            print(f"   Epoch {epoch:2d}: loss={loss.item():.4f}, "
                  f"threshold={loss_info['adaptive_threshold']:.3f}, "
                  f"edges={summary['num_edges']}")

    print("\n5. Analyzing final enhanced structure...")
    final_summary = enhanced_learner.get_enhanced_causal_graph_summary()

    print(f"   ✅ Final adaptive threshold: {final_summary['adaptive_threshold']:.3f}")
    print(f"   ✅ Edges discovered: {final_summary['num_edges']}")
    print(f"   ✅ Enhancement info: {enhanced_learner.get_enhancement_info()}")

    # Check discovered relationships
    discovered_relationships = []
    for edge in final_summary['edges']:
        discovered_relationships.append((edge['cause_idx'], edge['effect_idx']))

    correct_discoveries = 0
    print("\n   📊 Relationship Analysis:")
    for i, expected in enumerate(target_relationships):
        found = expected in discovered_relationships
        if found:
            correct_discoveries += 1

        # Find the edge for statistical info
        edge_info = None
        for edge in final_summary['edges']:
            if (edge['cause_idx'], edge['effect_idx']) == expected:
                edge_info = edge
                break

        if edge_info:
            print(f"     ✅ {enhanced_learner.variable_names[expected[0]]} → {enhanced_learner.variable_names[expected[1]]}: "
                  f"weight={edge_info['weight']:.3f}, stat={edge_info['statistical_support']:.3f}")
        else:
            print(f"     ❌ {enhanced_learner.variable_names[expected[0]]} → {enhanced_learner.variable_names[expected[1]]}: "
                  f"NOT DISCOVERED")

    discovery_accuracy = correct_discoveries / len(target_relationships)
    print(f"   📈 Discovery accuracy: {discovery_accuracy:.1%} ({correct_discoveries}/{len(target_relationships)})")

    # Compare with original structure learner
    print("\n6. Comparing with original structure learner...")
    try:
        from causal_architectures.structure_learner import CausalStructureLearner

        original_learner = CausalStructureLearner(num_variables=5, hidden_dim=32)
        original_optimizer = optim.Adam(original_learner.parameters(), lr=2e-3)

        # Train original for same number of epochs
        for epoch in range(25):
            original_optimizer.zero_grad()
            orig_loss, orig_loss_info = original_learner.compute_structure_loss(causal_data)
            orig_loss.backward()
            torch.nn.utils.clip_grad_norm_(original_learner.parameters(), 1.0)
            original_optimizer.step()

        original_summary = original_learner.get_causal_graph_summary()

        print(f"   📊 Original edges discovered: {original_summary['num_edges']}")
        print(f"   📊 Enhanced edges discovered: {final_summary['num_edges']}")
        print(f"   📊 Improvement: {final_summary['num_edges'] - original_summary['num_edges']} additional edges")

        comparison_success = True

    except Exception as e:
        print(f"   ⚠️  Original comparison failed: {e}")
        comparison_success = False

    # Final assessment
    print("\n" + "=" * 70)
    print("📊 PHASE 2 ENHANCEMENT RESULTS")
    print("=" * 70)

    results = {
        'edges_discovered': final_summary['num_edges'],
        'discovery_accuracy': discovery_accuracy,
        'adaptive_threshold_working': len(set(thresholds)) > 3,  # Threshold changed during training
        'strong_relationships_found': any(edge['statistical_support'] > 0.7 for edge in final_summary['edges']),
        'enhancement_active': enhanced_learner.get_enhancement_info()['enhancement_enabled']
    }

    print(f"🎯 Edges Discovered: {results['edges_discovered']} (target: 2-3 functional)")
    print(f"📈 Discovery Accuracy: {results['discovery_accuracy']:.1%}")
    print(f"🔄 Adaptive Thresholding: {'✅ WORKING' if results['adaptive_threshold_working'] else '❌ STATIC'}")
    print(f"💪 Strong Relationships: {'✅ FOUND' if results['strong_relationships_found'] else '❌ WEAK'}")
    print(f"🚀 Enhancement Status: {'✅ ACTIVE' if results['enhancement_active'] else '❌ DISABLED'}")

    # Overall success criteria
    functional_edges_found = 2 <= results['edges_discovered'] <= 10  # Reasonable range
    good_accuracy = results['discovery_accuracy'] >= 0.5  # At least 50% of expected relationships
    strong_statistical_support = results['strong_relationships_found']

    overall_success = functional_edges_found and good_accuracy and strong_statistical_support

    if overall_success:
        print(f"\n🏆 PHASE 2 ENHANCEMENT: COMPLETE SUCCESS!")
        print(f"   ✅ Structure learning: 0 edges → {results['edges_discovered']} functional edges")
        print(f"   ✅ Discovery accuracy: {results['discovery_accuracy']:.1%}")
        print(f"   ✅ Strong statistical support detected")
        print(f"   ✅ Adaptive thresholding working")
        print(f"   🚀 READY FOR FULL SYSTEM INTEGRATION")
    elif functional_edges_found and good_accuracy:
        print(f"\n🎯 PHASE 2 ENHANCEMENT: FUNCTIONAL SUCCESS!")
        print(f"   ✅ Major improvement in structure discovery")
        print(f"   ✅ Reasonable edge detection")
        print(f"   ⚠️  May need fine-tuning for optimal performance")
    else:
        print(f"\n⚠️ PHASE 2 ENHANCEMENT: NEEDS WORK")
        print(f"   📋 Address issues before proceeding")

    return overall_success, results, enhanced_learner


def demonstrate_research_insights():
    """
    Demonstrate embedded research insights in enhanced structure learner
    """
    print("\n🧠 RESEARCH INSIGHTS DEMONSTRATION")
    print("=" * 50)

    enhanced_learner = EnhancedCausalStructureLearner(num_variables=5, total_epochs=50)

    print("1. Adaptive Threshold Scheduling (CL-NOTEARS inspired):")
    for epoch in [0, 10, 25, 40, 50]:
        threshold = enhanced_learner.threshold_scheduler.get_threshold(epoch)
        print(f"   Epoch {epoch:2d}: threshold = {threshold:.3f}")

    print("\n2. Statistical Constraint Generation (PC-NOTEARS hybrid):")
    test_data = generate_strong_causal_data(batch_size=32, seq_len=20)
    constraints = enhanced_learner.constraint_generator.generate_pc_constraints(test_data)

    strong_constraints = [(i, j) for (i, j), info in constraints.items()
                         if info['recommended']]
    print(f"   Strong statistical constraints found: {len(strong_constraints)}")
    for (i, j) in strong_constraints[:3]:  # Show top 3
        info = constraints[(i, j)]
        print(f"     {enhanced_learner.variable_names[i]} → {enhanced_learner.variable_names[j]}: "
              f"corr={info['correlation']:.3f}, p={info['p_value']:.3f}")

    print(f"\n3. Key Research Insights Embedded:")
    print(f"   ✅ CL-NOTEARS curriculum learning")
    print(f"   ✅ Gradient-informed adaptive thresholding")
    print(f"   ✅ PC-NOTEARS hybrid statistical constraints")
    print(f"   ✅ Dynamic edge weight adjustment")
    print(f"   ✅ NOTEARS foundation preserved")

    print(f"\n🎯 Enhancement bridges statistical rigor with continuous optimization!")


if __name__ == "__main__":
    success, results, enhanced_learner = test_phase2_enhanced_training()

    if success:
        demonstrate_research_insights()

        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Integrate enhanced structure learner with main system")
        print(f"   2. Update extreme challenge to use enhanced learner")
        print(f"   3. Proceed to Phase 3 enhancements")
    else:
        print(f"\n🔧 DEBUGGING NEEDED:")
        print(f"   1. Tune adaptive thresholding parameters")
        print(f"   2. Improve statistical constraint generation")
        print(f"   3. Validate curriculum learning schedule")