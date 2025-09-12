#!/usr/bin/env python3
"""
Causal Intervention Testing Framework
Systematic evaluation of causal reasoning capabilities in the world model

Tests various "what-if" scenarios to validate that the model:
1. Understands causal relationships
2. Can predict intervention effects
3. Generalizes to novel causal combinations
4. Shows reasonable uncertainty estimates
"""

import os
import sys
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.append('..')
sys.path.append('../controller')
sys.path.append('../causal_envs')

from causal_envs.campus_env import SimpleCampusEnv
from controller.causal_world_model_controller import CausalWorldModelController, ControllerConfig

@dataclass
class InterventionTest:
    """Defines a single causal intervention test"""
    name: str
    description: str
    base_state: Dict
    intervention: Dict
    expected_effect: str
    test_actions: List[int]
    success_criteria: Dict

class CausalInterventionTester:
    """Main class for running causal intervention tests"""
    
    def __init__(self, 
                 controller: CausalWorldModelController,
                 output_dir: str = './evaluation_results/'):
        
        self.controller = controller
        self.output_dir = output_dir
        self.env = SimpleCampusEnv()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define test suite
        self.test_suite = self._create_intervention_test_suite()
        
        print(f"✅ CausalInterventionTester initialized")
        print(f"   Test suite: {len(self.test_suite)} tests")
        print(f"   Output directory: {output_dir}")
    
    def _create_intervention_test_suite(self) -> List[InterventionTest]:
        """Create comprehensive suite of causal intervention tests"""
        
        tests = []
        
        # Test 1: Weather intervention - Rain effect
        tests.append(InterventionTest(
            name="weather_rain_intervention",
            description="Test if model predicts slower movement in rain",
            base_state={
                'time_hour': 14, 'day_week': 2, 'weather': 'sunny', 
                'event': 'normal', 'crowd_density': 2
            },
            intervention={'weather': 'rain'},
            expected_effect="slower_movement",
            test_actions=[2, 2, 2, 2, 2],  # Move East 5 times
            success_criteria={
                'min_latent_difference': 0.03,
                'expected_direction': 'movement_penalty'
            }
        ))
        
        # Test 2: Weather intervention - Snow effect  
        tests.append(InterventionTest(
            name="weather_snow_intervention",
            description="Test if model predicts significant changes in snow",
            base_state={
                'time_hour': 10, 'day_week': 1, 'weather': 'sunny',
                'event': 'normal', 'crowd_density': 1
            },
            intervention={'weather': 'snow'},
            expected_effect="major_change",
            test_actions=[1, 1, 1, 1, 1],  # Move South 5 times
            success_criteria={
                'min_latent_difference': 0.05,
                'expected_direction': 'major_penalty'
            }
        ))
        
        # Test 3: Event intervention - Gameday crowds
        tests.append(InterventionTest(
            name="event_gameday_intervention", 
            description="Test if model predicts crowd effects during gameday",
            base_state={
                'time_hour': 16, 'day_week': 5, 'weather': 'sunny',
                'event': 'normal', 'crowd_density': 2
            },
            intervention={'event': 'gameday'},
            expected_effect="crowd_avoidance",
            test_actions=[0, 2, 1, 3, 0],  # Mixed movements
            success_criteria={
                'min_latent_difference': 0.08,
                'expected_direction': 'crowd_penalty'
            }
        ))
        
        # Test 4: Time intervention - Night vs Day
        tests.append(InterventionTest(
            name="time_night_intervention",
            description="Test if model predicts different behavior at night",
            base_state={
                'time_hour': 14, 'day_week': 3, 'weather': 'sunny',
                'event': 'normal', 'crowd_density': 3
            },
            intervention={'time_hour': 22},
            expected_effect="night_effect",
            test_actions=[2, 1, 2, 1, 4],  # Movement + stay
            success_criteria={
                'min_latent_difference': 0.02,
                'expected_direction': 'visibility_penalty'
            }
        ))
        
        # Test 5: Crowd density intervention
        tests.append(InterventionTest(
            name="crowd_density_intervention",
            description="Test if model predicts effects of high crowd density",
            base_state={
                'time_hour': 12, 'day_week': 1, 'weather': 'sunny',
                'event': 'normal', 'crowd_density': 1
            },
            intervention={'crowd_density': 5},
            expected_effect="crowd_penalty", 
            test_actions=[2, 2, 1, 1, 2],
            success_criteria={
                'min_latent_difference': 0.04,
                'expected_direction': 'movement_difficulty'
            }
        ))
        
        # Test 6: Construction event intervention
        tests.append(InterventionTest(
            name="construction_intervention",
            description="Test if model predicts path blocking during construction",
            base_state={
                'time_hour': 10, 'day_week': 2, 'weather': 'sunny',
                'event': 'normal', 'crowd_density': 2
            },
            intervention={'event': 'construction'},
            expected_effect="path_blocking",
            test_actions=[2, 2, 1, 1, 3],
            success_criteria={
                'min_latent_difference': 0.06,
                'expected_direction': 'obstacle_penalty'
            }
        ))
        
        # Test 7: Multi-factor intervention - Perfect storm
        tests.append(InterventionTest(
            name="multi_factor_storm",
            description="Test complex scenario: rain + gameday + high crowds",
            base_state={
                'time_hour': 14, 'day_week': 5, 'weather': 'sunny',
                'event': 'normal', 'crowd_density': 2
            },
            intervention={'weather': 'rain', 'event': 'gameday', 'crowd_density': 5},
            expected_effect="compound_penalty",
            test_actions=[1, 2, 1, 2, 1],
            success_criteria={
                'min_latent_difference': 0.10,
                'expected_direction': 'severe_penalty'
            }
        ))
        
        # Test 8: Counter-intervention - Favorable conditions
        tests.append(InterventionTest(
            name="favorable_conditions",
            description="Test if model predicts easier navigation in good conditions",
            base_state={
                'time_hour': 22, 'day_week': 0, 'weather': 'rain',
                'event': 'construction', 'crowd_density': 5
            },
            intervention={'time_hour': 14, 'weather': 'sunny', 'event': 'normal', 'crowd_density': 1},
            expected_effect="improvement",
            test_actions=[2, 2, 2, 2, 2],
            success_criteria={
                'min_latent_difference': 0.08,
                'expected_direction': 'improvement'
            }
        ))
        
        return tests
    
    def run_intervention_test(self, test: InterventionTest) -> Dict:
        """Run a single causal intervention test"""
        
        print(f"\n🧪 Running Test: {test.name}")
        print(f"   {test.description}")
        print(f"   Base state: {test.base_state}")
        print(f"   Intervention: {test.intervention}")
        
        # Create test environment
        obs, info = self.env.reset(options={
            'causal_state': test.base_state,
            'goal_type': 'library'  # Standard goal
        })
        
        current_obs = obs
        test_actions = np.array(test.test_actions)
        
        try:
            # Perform causal intervention test
            intervention_result = self.controller.causal_intervention(
                current_obs=current_obs,
                base_causal_state=test.base_state,
                intervention=test.intervention,
                action_sequence=test_actions,
                horizon=len(test_actions)
            )
            
            # Evaluate test results
            test_result = self._evaluate_test_result(test, intervention_result)
            
            print(f"   Result: {'✅ PASS' if test_result['passed'] else '❌ FAIL'}")
            print(f"   Latent difference: {test_result['latent_difference']:.4f}")
            print(f"   Analysis: {test_result['analysis']}")
            
            return test_result
            
        except Exception as e:
            print(f"   ❌ TEST ERROR: {e}")
            return {
                'test_name': test.name,
                'passed': False,
                'error': str(e),
                'latent_difference': 0.0,
                'analysis': f"Test failed with error: {e}"
            }
    
    def _evaluate_test_result(self, test: InterventionTest, intervention_result: Dict) -> Dict:
        """Evaluate whether an intervention test passed"""
        
        latent_diff = intervention_result['latent_space_difference']
        significant_change = intervention_result['significant_change']
        
        # Check if latent difference meets minimum threshold
        min_diff_met = latent_diff >= test.success_criteria['min_latent_difference']
        
        # Determine if test passed
        passed = min_diff_met and significant_change
        
        # Generate analysis
        analysis = []
        
        if min_diff_met:
            analysis.append(f"Minimum difference threshold met ({latent_diff:.4f} >= {test.success_criteria['min_latent_difference']:.4f})")
        else:
            analysis.append(f"Minimum difference threshold NOT met ({latent_diff:.4f} < {test.success_criteria['min_latent_difference']:.4f})")
        
        if significant_change:
            analysis.append("Model detected significant change from intervention")
        else:
            analysis.append("Model did NOT detect significant change")
        
        # Add effect-specific analysis
        expected_direction = test.success_criteria.get('expected_direction', 'unknown')
        if expected_direction in intervention_result.get('analysis', ''):
            analysis.append(f"Expected effect direction observed: {expected_direction}")
        
        return {
            'test_name': test.name,
            'test_description': test.description,
            'base_state': test.base_state,
            'intervention': test.intervention,
            'expected_effect': test.expected_effect,
            'latent_difference': latent_diff,
            'significant_change': significant_change,
            'min_diff_threshold': test.success_criteria['min_latent_difference'],
            'passed': passed,
            'analysis': ' | '.join(analysis),
            'intervention_result': intervention_result
        }
    
    def run_full_test_suite(self) -> Dict:
        """Run all causal intervention tests"""
        
        print(f"\n🧪 Running Causal Intervention Test Suite")
        print(f"=" * 70)
        print(f"Total tests: {len(self.test_suite)}")
        
        start_time = time.time()
        results = []
        
        for test in self.test_suite:
            result = self.run_intervention_test(test)
            results.append(result)
            
            # Brief pause between tests
            time.sleep(0.5)
        
        # Generate summary
        total_time = time.time() - start_time
        summary = self._generate_test_summary(results, total_time)
        
        # Save results
        self._save_test_results(summary, results)
        
        return summary
    
    def _generate_test_summary(self, results: List[Dict], total_time: float) -> Dict:
        """Generate summary of all test results"""
        
        passed_tests = [r for r in results if r.get('passed', False)]
        failed_tests = [r for r in results if not r.get('passed', False)]
        
        # Calculate statistics
        pass_rate = len(passed_tests) / len(results) if results else 0
        avg_latent_diff = np.mean([r.get('latent_difference', 0) for r in results])
        
        # Categorize by intervention type
        intervention_categories = {}
        for result in results:
            category = self._categorize_intervention(result.get('intervention', {}))
            if category not in intervention_categories:
                intervention_categories[category] = {'passed': 0, 'total': 0}
            intervention_categories[category]['total'] += 1
            if result.get('passed', False):
                intervention_categories[category]['passed'] += 1
        
        summary = {
            'test_suite_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(results),
            'passed_tests': len(passed_tests),
            'failed_tests': len(failed_tests),
            'pass_rate': pass_rate,
            'total_time_seconds': total_time,
            'average_latent_difference': avg_latent_diff,
            'intervention_categories': intervention_categories,
            'model_capabilities': self._assess_model_capabilities(results),
            'recommendations': self._generate_recommendations(results)
        }
        
        # Print summary
        print(f"\n" + "="*70)
        print(f"🎯 CAUSAL INTERVENTION TEST RESULTS")
        print(f"="*70)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']} ({pass_rate*100:.1f}%)")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Average Latent Difference: {avg_latent_diff:.4f}")
        print(f"Total Time: {total_time:.1f} seconds")
        
        if passed_tests:
            print(f"\n✅ Passed Tests:")
            for test in passed_tests:
                print(f"   - {test['test_name']}: {test['latent_difference']:.4f}")
        
        if failed_tests:
            print(f"\n❌ Failed Tests:")
            for test in failed_tests:
                print(f"   - {test['test_name']}: {test.get('latent_difference', 0):.4f}")
        
        print(f"\n📊 Model Capabilities: {summary['model_capabilities']}")
        
        return summary
    
    def _categorize_intervention(self, intervention: Dict) -> str:
        """Categorize intervention by type"""
        
        if 'weather' in intervention:
            return 'weather'
        elif 'event' in intervention:
            return 'event'
        elif 'time_hour' in intervention:
            return 'temporal'
        elif 'crowd_density' in intervention:
            return 'crowd'
        elif len(intervention) > 1:
            return 'multi_factor'
        else:
            return 'other'
    
    def _assess_model_capabilities(self, results: List[Dict]) -> Dict:
        """Assess overall model causal reasoning capabilities"""
        
        capabilities = {
            'weather_sensitivity': 'unknown',
            'event_awareness': 'unknown', 
            'temporal_effects': 'unknown',
            'crowd_modeling': 'unknown',
            'multi_factor_reasoning': 'unknown',
            'overall_rating': 'unknown'
        }
        
        # Weather tests
        weather_tests = [r for r in results if 'weather' in r.get('intervention', {})]
        if weather_tests:
            weather_pass_rate = sum(r.get('passed', False) for r in weather_tests) / len(weather_tests)
            capabilities['weather_sensitivity'] = 'good' if weather_pass_rate > 0.7 else 'poor'
        
        # Event tests
        event_tests = [r for r in results if 'event' in r.get('intervention', {})]
        if event_tests:
            event_pass_rate = sum(r.get('passed', False) for r in event_tests) / len(event_tests)
            capabilities['event_awareness'] = 'good' if event_pass_rate > 0.7 else 'poor'
        
        # Multi-factor tests
        multi_tests = [r for r in results if len(r.get('intervention', {})) > 1]
        if multi_tests:
            multi_pass_rate = sum(r.get('passed', False) for r in multi_tests) / len(multi_tests)
            capabilities['multi_factor_reasoning'] = 'good' if multi_pass_rate > 0.5 else 'poor'
        
        # Overall rating
        pass_rate = sum(r.get('passed', False) for r in results) / len(results) if results else 0
        if pass_rate > 0.8:
            capabilities['overall_rating'] = 'excellent'
        elif pass_rate > 0.6:
            capabilities['overall_rating'] = 'good'
        elif pass_rate > 0.4:
            capabilities['overall_rating'] = 'fair'
        else:
            capabilities['overall_rating'] = 'poor'
        
        return capabilities
    
    def _generate_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Overall performance
        pass_rate = sum(r.get('passed', False) for r in results) / len(results) if results else 0
        
        if pass_rate < 0.5:
            recommendations.append("Low pass rate suggests fundamental issues with causal modeling")
            recommendations.append("Consider retraining causal RNN with more diverse data")
        
        # Specific failure patterns
        failed_tests = [r for r in results if not r.get('passed', False)]
        
        if len([r for r in failed_tests if 'weather' in r.get('intervention', {})]) > 1:
            recommendations.append("Weather interventions consistently failing - check weather effect implementation")
        
        if len([r for r in failed_tests if len(r.get('intervention', {})) > 1]) > 0:
            recommendations.append("Multi-factor interventions failing - model may not capture interaction effects")
        
        # Low sensitivity
        avg_latent_diff = np.mean([r.get('latent_difference', 0) for r in results])
        if avg_latent_diff < 0.03:
            recommendations.append("Low latent differences suggest model is not sensitive to causal changes")
            recommendations.append("Consider increasing causal factor weights in training")
        
        if pass_rate > 0.8:
            recommendations.append("Excellent causal reasoning performance - ready for deployment")
        
        return recommendations
    
    def _save_test_results(self, summary: Dict, detailed_results: List[Dict]):
        """Save test results to files"""
        
        timestamp = int(time.time())
        
        # Save summary
        summary_file = os.path.join(self.output_dir, f'causal_test_summary_{timestamp}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        results_file = os.path.join(self.output_dir, f'causal_test_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n📁 Test results saved:")
        print(f"   Summary: {summary_file}")
        print(f"   Details: {results_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Causal Intervention Testing')
    parser.add_argument('--phase1_models', type=str, default='./data/models/phase1/',
                       help='Phase 1 VAE models directory')
    parser.add_argument('--phase2a_models', type=str, default='./data/models/phase2a/',
                       help='Phase 2A causal RNN models directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results/',
                       help='Output directory for test results')
    parser.add_argument('--test_name', type=str, default=None,
                       help='Run specific test by name (default: run all)')
    
    args = parser.parse_args()
    
    print("🧪 Causal Intervention Testing Framework")
    print("=" * 60)
    
    # Create controller
    config = ControllerConfig(
        vae_model_name="best",
        causal_rnn_model_name="full_causal"
    )
    
    controller = CausalWorldModelController(
        config,
        phase1_models_dir=args.phase1_models,
        phase2a_models_dir=args.phase2a_models
    )
    
    # Create tester
    tester = CausalInterventionTester(controller, args.output_dir)
    
    if args.test_name:
        # Run specific test
        test = next((t for t in tester.test_suite if t.name == args.test_name), None)
        if test:
            result = tester.run_intervention_test(test)
            print(f"\n📊 Single test result: {result}")
        else:
            print(f"❌ Test '{args.test_name}' not found")
    else:
        # Run full test suite
        summary = tester.run_full_test_suite()
        
        if summary['pass_rate'] > 0.8:
            print("\n🎉 Excellent causal reasoning performance!")
        elif summary['pass_rate'] > 0.6:
            print("\n👍 Good causal reasoning performance")
        else:
            print("\n⚠️  Causal reasoning needs improvement")

if __name__ == "__main__":
    main()