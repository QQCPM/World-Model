#!/usr/bin/env python3
"""
EXTREME TEMPORAL CONSISTENCY STRESS TEST
========================================

This test pushes the temporal integration system to its absolute limits to determine
if temporal delays are genuine and consistent or just superficial buffering.

CHALLENGE DESIGN:
1. DELAY CONSISTENCY UNDER DOMAIN SHIFT
   - Test if 2-timestep weather delays hold across different scenarios
   - Verify delay robustness to distribution changes

2. TEMPORAL MEMORY CAPACITY STRESS
   - Long sequences with complex delay patterns
   - Test if delay buffer maintains consistency over time

3. CONFLICTING TEMPORAL SIGNALS
   - Multiple variables with different delay requirements
   - Test if system can handle conflicting temporal demands

4. TEMPORAL CAUSALITY VIOLATIONS
   - Test with future-looking dependencies (should be rejected)
   - Verify system maintains causal ordering

TARGET: Only genuine temporal integration should achieve >80% consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_envs.temporal_integration import TemporalCausalIntegrator, TemporalIntegrationConfig
from causal_envs.continuous_campus_env import CausalState, WeatherType, EventType


class ExtremeTemporalConsistencyStressTest:
    """
    Stress test for temporal consistency and delay reliability

    Tests whether temporal delays are genuine and consistent across challenging scenarios
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.integrator = None
        self.test_results = {}

    def setup_integrator(self, enable_validation=True):
        """Initialize temporal integrator with validation enabled"""
        config = TemporalIntegrationConfig(
            enable_delays=True,
            enable_logging=enable_validation,
            validation_mode=enable_validation,
            fallback_to_immediate=False  # No fallback for stress testing
        )
        self.integrator = TemporalCausalIntegrator(config)

    def generate_structured_weather_sequence(self, seq_len=200):
        """
        Generate structured weather sequence with known patterns for delay testing
        """
        weather_sequence = []
        causal_states = []

        # Create structured pattern: 3 cycles of [sunny, rain, snow, fog]
        weather_cycle = [WeatherType.SUNNY, WeatherType.RAIN, WeatherType.SNOW, WeatherType.FOG]
        base_crowd = 0.3
        base_time = 12

        for t in range(seq_len):
            # Weather follows structured pattern
            weather = weather_cycle[t % len(weather_cycle)]

            # Other factors evolve naturally
            crowd_density = base_crowd + 0.3 * np.sin(t * 0.1) + np.random.normal(0, 0.1)
            crowd_density = np.clip(crowd_density, 0, 1)

            time_hour = int((base_time + t * 0.2) % 24)

            causal_state = CausalState(
                time_hour=time_hour,
                day_week=t % 7,
                weather=weather,
                event=EventType.NORMAL,
                crowd_density=crowd_density
            )

            weather_sequence.append(weather)
            causal_states.append(causal_state)

        return causal_states, weather_sequence

    def test_1_delay_consistency_across_scenarios(self):
        """
        TEST 1: Do weather delays remain consistent across different scenarios?
        """
        print("🔥 TEST 1: Delay Consistency Across Scenarios")
        print("=" * 60)

        self.setup_integrator(enable_validation=True)

        # Test across multiple scenarios
        scenarios = [
            ("Normal Campus", 100, 0.3, EventType.NORMAL),
            ("High Crowd Density", 100, 0.8, EventType.GAMEDAY),
            ("Low Activity", 100, 0.1, EventType.BREAK),
            ("Construction Zone", 100, 0.5, EventType.CONSTRUCTION)
        ]

        scenario_results = {}

        for scenario_name, seq_len, base_crowd, event_type in scenarios:
            print(f"  Testing scenario: {scenario_name}")

            # Reset integrator for each scenario
            self.integrator.reset()

            # Generate scenario-specific data
            causal_states = []
            weather_pattern = [WeatherType.SUNNY, WeatherType.RAIN, WeatherType.SUNNY, WeatherType.RAIN] * (seq_len // 4)

            for t in range(seq_len):
                weather = weather_pattern[t] if t < len(weather_pattern) else WeatherType.SUNNY
                crowd = base_crowd + np.random.normal(0, 0.1)
                crowd = np.clip(crowd, 0, 1)

                causal_state = CausalState(
                    time_hour=(8 + t // 4) % 24,
                    day_week=t % 7,
                    weather=weather,
                    event=event_type,
                    crowd_density=crowd
                )
                causal_states.append(causal_state)

            # Process sequence and track delay differences
            delay_differences = []
            weather_changes = []

            for t, causal_state in enumerate(causal_states):
                test_action = np.array([1.0, 0.5])
                modified_action, temporal_info = self.integrator.apply_temporal_effects(test_action, causal_state)

                if t >= 2:  # After buffer fills
                    delay_diff = temporal_info['delay_comparison']['weather_difference']
                    delay_differences.append(delay_diff)

                    # Track weather changes
                    if t > 0:
                        weather_changed = causal_states[t].weather != causal_states[t-1].weather
                        weather_changes.append(weather_changed)

            # Analyze delay consistency
            avg_delay_diff = np.mean(delay_differences) if delay_differences else 0
            delay_variance = np.var(delay_differences) if delay_differences else 0
            delay_consistency = 1.0 / (1.0 + delay_variance)

            # Check if delays are actually detected during weather changes
            weather_change_indices = [i for i, changed in enumerate(weather_changes) if changed]
            if weather_change_indices:
                delay_during_changes = [delay_differences[i] for i in weather_change_indices if i < len(delay_differences)]
                avg_delay_during_changes = np.mean(delay_during_changes) if delay_during_changes else 0
            else:
                avg_delay_during_changes = 0

            scenario_results[scenario_name] = {
                'avg_delay_difference': avg_delay_diff,
                'delay_variance': delay_variance,
                'delay_consistency': delay_consistency,
                'avg_delay_during_changes': avg_delay_during_changes,
                'weather_changes_detected': len(weather_change_indices),
                'sequence_length': seq_len
            }

        # Overall consistency across scenarios
        all_consistencies = [r['delay_consistency'] for r in scenario_results.values()]
        all_delay_diffs = [r['avg_delay_difference'] for r in scenario_results.values()]

        cross_scenario_consistency = 1.0 - np.var(all_delay_diffs)
        avg_consistency = np.mean(all_consistencies)

        results = {
            'scenario_results': scenario_results,
            'cross_scenario_consistency': max(0, cross_scenario_consistency),
            'average_consistency': avg_consistency,
            'overall_delay_detection': np.mean(all_delay_diffs)
        }

        # Scoring criteria
        cross_consistency_pass = results['cross_scenario_consistency'] > 0.7
        avg_consistency_pass = results['average_consistency'] > 0.8
        delay_detection_pass = results['overall_delay_detection'] > 0.05  # Should detect delays

        total_score = (cross_consistency_pass + avg_consistency_pass + delay_detection_pass) / 3.0

        print(f"    Cross-Scenario Consistency: {results['cross_scenario_consistency']:.4f} {'✅' if cross_consistency_pass else '❌'}")
        print(f"    Average Consistency: {results['average_consistency']:.4f} {'✅' if avg_consistency_pass else '❌'}")
        print(f"    Delay Detection: {results['overall_delay_detection']:.4f} {'✅' if delay_detection_pass else '❌'}")
        print(f"    Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("    🎉 EXCEPTIONAL: Consistent delays across all scenarios!")
        elif total_score > 0.6:
            print("    👍 GOOD: Generally consistent delays")
        elif total_score > 0.3:
            print("    ⚠️  WEAK: Some consistency issues")
        else:
            print("    💀 FAILED: Inconsistent temporal behavior")

        self.test_results['test_1'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.6
        }

        return total_score, results

    def test_2_long_sequence_memory_stress(self):
        """
        TEST 2: Can temporal system maintain consistency over very long sequences?
        """
        print("\n🎯 TEST 2: Long Sequence Memory Stress")
        print("=" * 60)

        self.setup_integrator(enable_validation=True)

        # Very long sequence with repeated patterns
        seq_len = 500
        causal_states, weather_sequence = self.generate_structured_weather_sequence(seq_len)

        # Process entire sequence
        delay_tracking = {
            'weather_differences': [],
            'buffer_states': [],
            'consistency_over_time': []
        }

        chunk_size = 50
        chunk_consistencies = []

        for t, causal_state in enumerate(causal_states):
            test_action = np.array([1.0, 0.5])
            modified_action, temporal_info = self.integrator.apply_temporal_effects(test_action, causal_state)

            if t >= 5:  # After initial buffer fill
                delay_diff = temporal_info['delay_comparison']['weather_difference']
                delay_tracking['weather_differences'].append(delay_diff)

                # Check buffer health
                buffer_metrics = temporal_info['integration_info']['buffer_metrics']
                delay_tracking['buffer_states'].append(buffer_metrics.get('buffer_filled', 0))

                # Compute chunk-wise consistency
                if t % chunk_size == 0 and len(delay_tracking['weather_differences']) >= chunk_size:
                    recent_delays = delay_tracking['weather_differences'][-chunk_size:]
                    chunk_variance = np.var(recent_delays)
                    chunk_consistency = 1.0 / (1.0 + chunk_variance)
                    chunk_consistencies.append(chunk_consistency)

        # Analyze long-term consistency
        all_delays = delay_tracking['weather_differences']
        overall_variance = np.var(all_delays) if all_delays else 1.0
        overall_consistency = 1.0 / (1.0 + overall_variance)

        # Check for consistency degradation over time
        if len(chunk_consistencies) >= 4:
            early_consistency = np.mean(chunk_consistencies[:2])
            late_consistency = np.mean(chunk_consistencies[-2:])
            consistency_degradation = early_consistency - late_consistency
        else:
            early_consistency = late_consistency = overall_consistency
            consistency_degradation = 0

        # Buffer stability
        buffer_states = delay_tracking['buffer_states']
        buffer_stability = 1.0 - np.var(buffer_states) if buffer_states else 0

        results = {
            'sequence_length': seq_len,
            'overall_consistency': overall_consistency,
            'early_consistency': early_consistency,
            'late_consistency': late_consistency,
            'consistency_degradation': consistency_degradation,
            'buffer_stability': buffer_stability,
            'chunk_consistencies': chunk_consistencies,
            'total_delay_measurements': len(all_delays)
        }

        # Scoring criteria (very harsh for long sequences)
        overall_consistency_pass = results['overall_consistency'] > 0.7
        no_degradation_pass = results['consistency_degradation'] < 0.2
        buffer_stability_pass = results['buffer_stability'] > 0.8
        late_consistency_pass = results['late_consistency'] > 0.6

        total_score = (overall_consistency_pass + no_degradation_pass + buffer_stability_pass + late_consistency_pass) / 4.0

        print(f"    Sequence Length: {seq_len} timesteps")
        print(f"    Overall Consistency: {results['overall_consistency']:.4f} {'✅' if overall_consistency_pass else '❌'}")
        print(f"    Consistency Degradation: {results['consistency_degradation']:.4f} {'✅' if no_degradation_pass else '❌'}")
        print(f"    Buffer Stability: {results['buffer_stability']:.4f} {'✅' if buffer_stability_pass else '❌'}")
        print(f"    Late Consistency: {results['late_consistency']:.4f} {'✅' if late_consistency_pass else '❌'}")
        print(f"    Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("    🎉 EXCEPTIONAL: Maintains consistency over long sequences!")
        elif total_score > 0.6:
            print("    👍 GOOD: Good long-term consistency")
        elif total_score > 0.3:
            print("    ⚠️  WEAK: Some long-term degradation")
        else:
            print("    💀 FAILED: Cannot maintain long-term consistency")

        self.test_results['test_2'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.6
        }

        return total_score, results

    def test_3_conflicting_temporal_demands(self):
        """
        TEST 3: Can system handle conflicting temporal requirements?
        """
        print("\n🧠 TEST 3: Conflicting Temporal Demands")
        print("=" * 60)

        self.setup_integrator(enable_validation=True)

        # Create scenario with conflicting temporal patterns
        seq_len = 150
        causal_states = []

        # Conflicting patterns:
        # - Weather changes every 5 steps (should have 2-step delay)
        # - Crowd changes every 3 steps (should have 1-step delay)
        # - Time changes continuously (should be immediate)

        for t in range(seq_len):
            # Weather: slow changes (every 5 steps)
            weather_idx = (t // 5) % 4
            weather = [WeatherType.SUNNY, WeatherType.RAIN, WeatherType.SNOW, WeatherType.FOG][weather_idx]

            # Crowd: faster changes (every 3 steps)
            crowd_pattern = [0.2, 0.5, 0.8]
            crowd = crowd_pattern[t % len(crowd_pattern)]

            # Time: continuous change
            time_hour = (8 + t // 6) % 24

            # Event: very slow changes (every 20 steps)
            event_idx = (t // 20) % 3
            event = [EventType.NORMAL, EventType.GAMEDAY, EventType.EXAM][event_idx]

            causal_state = CausalState(
                time_hour=time_hour,
                day_week=t % 7,
                weather=weather,
                event=event,
                crowd_density=crowd
            )
            causal_states.append(causal_state)

        # Process sequence and analyze different delay behaviors
        weather_delays = []
        crowd_delays = []
        time_delays = []
        event_delays = []

        for t, causal_state in enumerate(causal_states):
            test_action = np.array([1.0, 0.5])
            modified_action, temporal_info = self.integrator.apply_temporal_effects(test_action, causal_state)

            if t >= 5:  # After buffer fills
                delay_comparison = temporal_info['delay_comparison']
                weather_delays.append(delay_comparison['weather_difference'])
                crowd_delays.append(delay_comparison['crowd_difference'])
                time_delays.append(delay_comparison['time_difference'])
                event_delays.append(delay_comparison['event_difference'])

        # Analyze delay behaviors
        def analyze_delay_pattern(delays, expected_delay, variable_name):
            avg_delay = np.mean(delays) if delays else 0
            delay_variance = np.var(delays) if delays else 0
            consistency = 1.0 / (1.0 + delay_variance)

            # Check if delay matches expectation
            if expected_delay == 'high':  # Weather should have high delay
                delay_appropriate = avg_delay > 0.1
            elif expected_delay == 'medium':  # Crowd should have medium delay
                delay_appropriate = 0.05 < avg_delay < 0.2
            elif expected_delay == 'low':  # Time should have low delay
                delay_appropriate = avg_delay < 0.1
            else:
                delay_appropriate = True

            return {
                'average_delay': avg_delay,
                'consistency': consistency,
                'delay_appropriate': delay_appropriate,
                'variance': delay_variance
            }

        weather_analysis = analyze_delay_pattern(weather_delays, 'high', 'weather')
        crowd_analysis = analyze_delay_pattern(crowd_delays, 'medium', 'crowd')
        time_analysis = analyze_delay_pattern(time_delays, 'low', 'time')
        event_analysis = analyze_delay_pattern(event_delays, 'low', 'event')

        results = {
            'weather_analysis': weather_analysis,
            'crowd_analysis': crowd_analysis,
            'time_analysis': time_analysis,
            'event_analysis': event_analysis,
            'sequence_length': seq_len
        }

        # Scoring criteria
        weather_pass = weather_analysis['delay_appropriate'] and weather_analysis['consistency'] > 0.6
        crowd_pass = crowd_analysis['delay_appropriate'] and crowd_analysis['consistency'] > 0.6
        time_pass = time_analysis['delay_appropriate'] and time_analysis['consistency'] > 0.7
        event_pass = event_analysis['delay_appropriate'] and event_analysis['consistency'] > 0.7

        total_score = (weather_pass + crowd_pass + time_pass + event_pass) / 4.0

        print(f"    Weather Delays (should be high): {weather_analysis['average_delay']:.4f} {'✅' if weather_pass else '❌'}")
        print(f"    Crowd Delays (should be medium): {crowd_analysis['average_delay']:.4f} {'✅' if crowd_pass else '❌'}")
        print(f"    Time Delays (should be low): {time_analysis['average_delay']:.4f} {'✅' if time_pass else '❌'}")
        print(f"    Event Delays (should be low): {event_analysis['average_delay']:.4f} {'✅' if event_pass else '❌'}")
        print(f"    Overall Score: {total_score:.3f}")

        if total_score > 0.8:
            print("    🎉 EXCEPTIONAL: Handles conflicting temporal demands perfectly!")
        elif total_score > 0.6:
            print("    👍 GOOD: Generally handles conflicting demands")
        elif total_score > 0.3:
            print("    ⚠️  WEAK: Some issues with conflicting demands")
        else:
            print("    💀 FAILED: Cannot handle conflicting temporal demands")

        self.test_results['test_3'] = {
            'score': total_score,
            'details': results,
            'passed': total_score > 0.6
        }

        return total_score, results

    def run_all_tests(self):
        """Run all temporal consistency stress tests"""
        print("🔥 EXTREME TEMPORAL CONSISTENCY STRESS TEST")
        print("=" * 80)
        print("Testing temporal integration under extreme stress conditions")
        print()

        # Run all tests
        score_1, _ = self.test_1_delay_consistency_across_scenarios()
        score_2, _ = self.test_2_long_sequence_memory_stress()
        score_3, _ = self.test_3_conflicting_temporal_demands()

        # Overall assessment
        overall_score = (score_1 + score_2 + score_3) / 3.0

        print("\n" + "=" * 80)
        print("📊 TEMPORAL CONSISTENCY STRESS TEST RESULTS")
        print("=" * 80)
        print(f"Test 1 - Scenario Consistency: {score_1:.3f}")
        print(f"Test 2 - Long Sequence Memory: {score_2:.3f}")
        print(f"Test 3 - Conflicting Demands: {score_3:.3f}")
        print(f"Overall Score: {overall_score:.3f}")

        if overall_score > 0.8:
            grade = "A+"
            status = "🔥 EXCEPTIONAL - Rock-solid temporal consistency!"
        elif overall_score > 0.7:
            grade = "A"
            status = "🎉 EXCELLENT - Strong temporal integration"
        elif overall_score > 0.6:
            grade = "B"
            status = "👍 GOOD - Decent temporal consistency"
        elif overall_score > 0.4:
            grade = "C"
            status = "⚠️ WEAK - Limited temporal consistency"
        else:
            grade = "F"
            status = "💀 FAILED - No reliable temporal integration"

        print(f"Grade: {grade}")
        print(f"Status: {status}")

        # Save results
        results_summary = {
            'overall_score': overall_score,
            'grade': grade,
            'status': status,
            'individual_tests': self.test_results,
            'timestamp': time.time()
        }

        return results_summary


def main():
    """Run the extreme temporal consistency stress test"""
    test = ExtremeTemporalConsistencyStressTest()
    results = test.run_all_tests()

    # Save results
    with open('extreme_temporal_consistency_stress_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        json.dump(convert_numpy(results), f, indent=2)

    print(f"\n📁 Results saved to: extreme_temporal_consistency_stress_results.json")

    return results


if __name__ == "__main__":
    results = main()