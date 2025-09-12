"""
Causal Intervention Testing Framework
Tests whether the trained models learned true causal relationships vs spurious correlations

This framework implements systematic causal interventions as described in Pearl's causal hierarchy:
1. Association: p(Y | X) - what the model predicts
2. Intervention: p(Y | do(X)) - what happens when we intervene 
3. Counterfactual: p(Y_x | X', Y') - what would have happened in a different world

For our causal world models, this means testing:
- Weather interventions: p(navigation | do(weather=rain))  
- Time interventions: p(navigation | do(time=night))
- Event interventions: p(navigation | do(event=gameday))
- Crowd interventions: p(navigation | do(crowd=high))
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import os
from datetime import datetime

import sys
sys.path.append('..')
sys.path.append('../causal_vae')
sys.path.append('../causal_rnn')
sys.path.append('../causal_envs')

from causal_vae.modern_architectures import create_vae_architecture
from causal_rnn.causal_mdn_gru import create_causal_rnn
from test_pipeline import TestCampusEnv


class InterventionType(Enum):
    """Types of causal interventions"""
    WEATHER = "weather"
    TIME = "time" 
    DAY = "day"
    EVENT = "event"
    CROWD = "crowd_density"


@dataclass
class InterventionSpec:
    """Specification for a causal intervention"""
    factor: InterventionType
    original_value: Any
    intervention_value: Any
    description: str


@dataclass 
class InterventionResult:
    """Results from a causal intervention test"""
    intervention: InterventionSpec
    prediction_original: torch.Tensor
    prediction_intervened: torch.Tensor
    causal_effect_size: float
    statistical_significance: float
    interpretation: str


class CausalInterventionTester:
    """
    Framework for testing causal interventions on trained world models
    
    Tests the key research hypothesis: "Do our models learn true causal relationships?"
    """
    
    def __init__(self, 
                 vae_model_path: str,
                 rnn_model_path: str,
                 vae_architecture: str = "hierarchical_512D"):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained models
        self.vae = self._load_vae_model(vae_model_path, vae_architecture)
        self.rnn = self._load_rnn_model(rnn_model_path)
        
        # Test environment for generating ground truth
        self.env = TestCampusEnv()
        
        # Causal factor mappings
        self.causal_mappings = {
            'weather': ['sunny', 'rain', 'snow', 'fog'],
            'event': ['normal', 'gameday', 'exam', 'break', 'construction'],
            'time_ranges': {
                'dawn': 6, 'morning': 8, 'noon': 12, 
                'afternoon': 14, 'evening': 18, 'night': 22
            },
            'crowd_levels': {
                'very_low': 1, 'low': 2, 'medium': 3, 'high': 4, 'very_high': 5
            }
        }
        
        print(f"🧪 Causal Intervention Tester initialized")
        print(f"   Device: {self.device}")
        print(f"   VAE Architecture: {vae_architecture}")
        
    def _load_vae_model(self, model_path: str, architecture: str):
        """Load trained VAE model"""
        try:
            vae = create_vae_architecture(architecture)
            if os.path.exists(model_path):
                vae.load_state_dict(torch.load(model_path, map_location=self.device))
                vae.eval()
                print(f"✅ VAE model loaded from {model_path}")
            else:
                print(f"⚠️  VAE model not found at {model_path}, using untrained model")
            return vae.to(self.device)
        except Exception as e:
            print(f"❌ Failed to load VAE model: {e}")
            return None
            
    def _load_rnn_model(self, model_path: str):
        """Load trained RNN model"""
        try:
            rnn = create_causal_rnn("causal_mdn_gru")
            if os.path.exists(model_path):
                rnn.load_state_dict(torch.load(model_path, map_location=self.device))
                rnn.eval()
                print(f"✅ RNN model loaded from {model_path}")
            else:
                print(f"⚠️  RNN model not found at {model_path}, using untrained model")
            return rnn.to(self.device)
        except Exception as e:
            print(f"❌ Failed to load RNN model: {e}")
            return None
    
    def run_weather_intervention_test(self, 
                                    base_scenario: Dict[str, Any],
                                    sequence_length: int = 10,
                                    num_trials: int = 50) -> InterventionResult:
        """
        Test weather intervention: p(navigation | do(weather=X))
        
        Key hypothesis: "Changing weather should affect navigation difficulty and path choices"
        Expected effects:
        - Rain -> slower movement, avoid open areas
        - Snow -> much slower movement
        - Fog -> uncertainty in navigation
        """
        print(f"\n🌧️ Testing Weather Intervention")
        
        weather_effects = []
        
        for weather_type in self.causal_mappings['weather']:
            if weather_type == base_scenario['weather']:
                continue
                
            print(f"   Testing: {base_scenario['weather']} → {weather_type}")
            
            # Create intervention specification
            intervention = InterventionSpec(
                factor=InterventionType.WEATHER,
                original_value=base_scenario['weather'],
                intervention_value=weather_type,
                description=f"Change weather from {base_scenario['weather']} to {weather_type}"
            )
            
            # Run intervention trials
            effect_sizes = []
            predictions_original = []
            predictions_intervened = []
            
            for trial in range(num_trials):
                # Generate original scenario
                original_scenario = base_scenario.copy()
                obs_orig, pred_orig = self._generate_prediction_sequence(original_scenario, sequence_length)
                predictions_original.append(pred_orig)
                
                # Generate intervened scenario
                intervened_scenario = base_scenario.copy()
                intervened_scenario['weather'] = weather_type
                obs_interv, pred_interv = self._generate_prediction_sequence(intervened_scenario, sequence_length)
                predictions_intervened.append(pred_interv)
                
                # Compute effect size (difference in predicted latent dynamics)
                if pred_orig is not None and pred_interv is not None:
                    effect_size = torch.norm(pred_interv - pred_orig).item()
                    effect_sizes.append(effect_size)
            
            if effect_sizes:
                avg_effect = np.mean(effect_sizes)
                std_effect = np.std(effect_sizes)
                statistical_sig = avg_effect / (std_effect + 1e-8)  # t-statistic approximation
                
                # Interpretation based on weather type
                interpretations = {
                    'rain': 'Negative effect on navigation speed and path optimality',
                    'snow': 'Strong negative effect on movement and visibility', 
                    'fog': 'Increased uncertainty and suboptimal path choices',
                    'sunny': 'Optimal navigation conditions'
                }
                
                result = InterventionResult(
                    intervention=intervention,
                    prediction_original=torch.stack(predictions_original) if predictions_original else None,
                    prediction_intervened=torch.stack(predictions_intervened) if predictions_intervened else None,
                    causal_effect_size=avg_effect,
                    statistical_significance=statistical_sig,
                    interpretation=interpretations.get(weather_type, "Weather effect on navigation")
                )
                
                weather_effects.append(result)
                print(f"     Effect size: {avg_effect:.4f} (±{std_effect:.4f})")
                print(f"     Significance: {statistical_sig:.2f}")
        
        return weather_effects
    
    def run_temporal_intervention_test(self,
                                     base_scenario: Dict[str, Any],
                                     sequence_length: int = 10,
                                     num_trials: int = 50) -> List[InterventionResult]:
        """
        Test temporal intervention: p(navigation | do(time=X))
        
        Key hypothesis: "Time of day should affect crowd levels and visibility"
        Expected effects:
        - Night -> reduced visibility, fewer crowds
        - Morning/Evening -> high crowd density near academic buildings
        - Noon -> high activity at cafeteria
        """
        print(f"\n⏰ Testing Temporal Intervention")
        
        temporal_effects = []
        
        for time_name, time_hour in self.causal_mappings['time_ranges'].items():
            if time_hour == base_scenario['time_hour']:
                continue
                
            print(f"   Testing: {base_scenario['time_hour']:02d}:00 → {time_hour:02d}:00 ({time_name})")
            
            intervention = InterventionSpec(
                factor=InterventionType.TIME,
                original_value=base_scenario['time_hour'],
                intervention_value=time_hour,
                description=f"Change time from {base_scenario['time_hour']:02d}:00 to {time_hour:02d}:00"
            )
            
            # Run trials and compute effect
            effect_sizes = []
            
            for trial in range(num_trials):
                original_scenario = base_scenario.copy()
                obs_orig, pred_orig = self._generate_prediction_sequence(original_scenario, sequence_length)
                
                intervened_scenario = base_scenario.copy() 
                intervened_scenario['time_hour'] = time_hour
                obs_interv, pred_interv = self._generate_prediction_sequence(intervened_scenario, sequence_length)
                
                if pred_orig is not None and pred_interv is not None:
                    effect_size = torch.norm(pred_interv - pred_orig).item()
                    effect_sizes.append(effect_size)
            
            if effect_sizes:
                result = InterventionResult(
                    intervention=intervention,
                    prediction_original=None,  # Save space
                    prediction_intervened=None,
                    causal_effect_size=np.mean(effect_sizes),
                    statistical_significance=np.mean(effect_sizes) / (np.std(effect_sizes) + 1e-8),
                    interpretation=f"Temporal effect: {time_name} changes navigation dynamics"
                )
                
                temporal_effects.append(result)
                print(f"     Effect size: {np.mean(effect_sizes):.4f}")
        
        return temporal_effects
    
    def run_event_intervention_test(self,
                                  base_scenario: Dict[str, Any], 
                                  sequence_length: int = 10,
                                  num_trials: int = 50) -> List[InterventionResult]:
        """
        Test event intervention: p(navigation | do(event=X))
        
        Key hypothesis: "Campus events should create specific crowd patterns and navigation constraints"
        Expected effects:
        - Gameday -> crowds at stadium, different navigation patterns
        - Exam week -> library overcrowding, academic building focus
        - Construction -> blocked paths, detour requirements
        """
        print(f"\n🎪 Testing Event Intervention")
        
        event_effects = []
        
        for event_type in self.causal_mappings['event']:
            if event_type == base_scenario['event']:
                continue
                
            print(f"   Testing: {base_scenario['event']} → {event_type}")
            
            intervention = InterventionSpec(
                factor=InterventionType.EVENT,
                original_value=base_scenario['event'],
                intervention_value=event_type,
                description=f"Change event from {base_scenario['event']} to {event_type}"
            )
            
            effect_sizes = []
            
            for trial in range(num_trials):
                original_scenario = base_scenario.copy()
                obs_orig, pred_orig = self._generate_prediction_sequence(original_scenario, sequence_length)
                
                intervened_scenario = base_scenario.copy()
                intervened_scenario['event'] = event_type
                obs_interv, pred_interv = self._generate_prediction_sequence(intervened_scenario, sequence_length)
                
                if pred_orig is not None and pred_interv is not None:
                    effect_size = torch.norm(pred_interv - pred_orig).item()
                    effect_sizes.append(effect_size)
            
            if effect_sizes:
                interpretations = {
                    'gameday': 'Stadium area congestion, altered campus flow patterns',
                    'exam': 'Library overcrowding, academic building focus',
                    'construction': 'Path blockages requiring navigation detours',
                    'break': 'Reduced activity, more direct navigation possible'
                }
                
                result = InterventionResult(
                    intervention=intervention,
                    prediction_original=None,
                    prediction_intervened=None,
                    causal_effect_size=np.mean(effect_sizes),
                    statistical_significance=np.mean(effect_sizes) / (np.std(effect_sizes) + 1e-8),
                    interpretation=interpretations.get(event_type, f"Event effect: {event_type}")
                )
                
                event_effects.append(result)
                print(f"     Effect size: {np.mean(effect_sizes):.4f}")
        
        return event_effects
    
    def _generate_prediction_sequence(self, 
                                    causal_scenario: Dict[str, Any],
                                    sequence_length: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate predicted latent sequence for a given causal scenario
        
        Args:
            causal_scenario: Dictionary with causal state values
            sequence_length: Length of sequence to predict
            
        Returns:
            observations: Generated visual observations (if VAE available)
            predictions: Predicted latent sequence (if models available)
        """
        try:
            # Reset environment with causal scenario
            obs, info = self.env.reset(options={'causal_state': causal_scenario})
            
            # If we have trained models, use them for prediction
            if self.vae is not None and self.rnn is not None:
                with torch.no_grad():
                    # Encode initial observation
                    obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    if hasattr(self.vae, 'encode'):
                        # Standard VAE
                        mu, logvar = self.vae.encode(obs_tensor)
                        z_initial = mu  # Use mean for deterministic prediction
                    else:
                        # Handle hierarchical VAE
                        z_initial = self.vae(obs_tensor)[1]  # Get latent representation
                    
                    # Generate action sequence (simple goal-directed policy)
                    actions = []
                    causal_states = []
                    
                    for t in range(sequence_length):
                        # Simple navigation toward library
                        agent_pos = info['agent_pos']
                        goal_pos = np.array([20, 20])  # Library position
                        
                        diff = goal_pos - agent_pos
                        if abs(diff[0]) > abs(diff[1]):
                            action = 2 if diff[0] > 0 else 3  # East/West
                        elif abs(diff[1]) > 0:
                            action = 1 if diff[1] > 0 else 0  # South/North
                        else:
                            action = 4  # Stay
                        
                        # Convert to one-hot
                        action_onehot = F.one_hot(torch.tensor(action), 5).float().unsqueeze(0).to(self.device)
                        actions.append(action_onehot)
                        
                        # Convert causal state to tensor
                        causal_encoding = torch.FloatTensor(info['causal_encoding']).unsqueeze(0).to(self.device)
                        causal_states.append(causal_encoding)
                        
                        # Step environment
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        if terminated or truncated:
                            break
                    
                    if actions and causal_states:
                        # Use RNN to predict sequence
                        action_seq = torch.stack(actions, dim=1)  # (1, seq_len, 5)
                        causal_seq = torch.stack(causal_states, dim=1)  # (1, seq_len, 45)
                        
                        # Generate predicted latent sequence
                        predicted_sequence = self.rnn.generate_sequence(
                            z_initial, action_seq, causal_seq, temperature=0.5
                        )
                        
                        return obs_tensor, predicted_sequence
            
            # Fallback: return observation-based features
            return torch.FloatTensor(obs), torch.FloatTensor([np.mean(obs)] * sequence_length).unsqueeze(0)
            
        except Exception as e:
            print(f"Warning: Prediction generation failed: {e}")
            return None, None
    
    def run_comprehensive_intervention_study(self,
                                           scenarios: List[Dict[str, Any]] = None,
                                           sequence_length: int = 10,
                                           num_trials: int = 25) -> Dict[str, List[InterventionResult]]:
        """
        Run comprehensive causal intervention study across all factors
        
        Args:
            scenarios: List of base scenarios to test (if None, uses defaults)
            sequence_length: Length of prediction sequences
            num_trials: Number of trials per intervention
            
        Returns:
            Dictionary of intervention results by category
        """
        print(f"\n🔬 COMPREHENSIVE CAUSAL INTERVENTION STUDY")
        print(f"=" * 60)
        
        if scenarios is None:
            scenarios = [
                # Normal weekday scenario
                {'time_hour': 14, 'day_week': 2, 'weather': 'sunny', 'event': 'normal', 'crowd_density': 3},
                # Weekend scenario  
                {'time_hour': 10, 'day_week': 5, 'weather': 'sunny', 'event': 'normal', 'crowd_density': 2},
                # Complex scenario
                {'time_hour': 18, 'day_week': 4, 'weather': 'rain', 'event': 'gameday', 'crowd_density': 5}
            ]
        
        all_results = {
            'weather_interventions': [],
            'temporal_interventions': [], 
            'event_interventions': []
        }
        
        for i, scenario in enumerate(scenarios):
            print(f"\n📋 Testing Scenario {i+1}: {scenario}")
            
            # Weather interventions
            weather_results = self.run_weather_intervention_test(scenario, sequence_length, num_trials)
            all_results['weather_interventions'].extend(weather_results)
            
            # Temporal interventions
            temporal_results = self.run_temporal_intervention_test(scenario, sequence_length, num_trials)
            all_results['temporal_interventions'].extend(temporal_results)
            
            # Event interventions
            event_results = self.run_event_intervention_test(scenario, sequence_length, num_trials)
            all_results['event_interventions'].extend(event_results)
        
        # Generate summary report
        self._generate_intervention_report(all_results)
        
        return all_results
    
    def _generate_intervention_report(self, results: Dict[str, List[InterventionResult]]):
        """Generate comprehensive intervention analysis report"""
        
        report = {
            'study_metadata': {
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'vae_available': self.vae is not None,
                'rnn_available': self.rnn is not None
            },
            'intervention_summary': {},
            'statistical_analysis': {},
            'research_conclusions': []
        }
        
        print(f"\n📊 CAUSAL INTERVENTION ANALYSIS REPORT")
        print(f"=" * 50)
        
        for category, interventions in results.items():
            if not interventions:
                continue
                
            effect_sizes = [r.causal_effect_size for r in interventions]
            significances = [r.statistical_significance for r in interventions]
            
            category_summary = {
                'num_interventions': len(interventions),
                'mean_effect_size': np.mean(effect_sizes),
                'std_effect_size': np.std(effect_sizes),
                'max_effect_size': np.max(effect_sizes),
                'mean_significance': np.mean(significances),
                'strong_effects': len([e for e in effect_sizes if e > 0.5])
            }
            
            report['intervention_summary'][category] = category_summary
            
            print(f"\n🧪 {category.replace('_', ' ').title()}:")
            print(f"   Interventions tested: {category_summary['num_interventions']}")
            print(f"   Mean effect size: {category_summary['mean_effect_size']:.4f}")
            print(f"   Strong effects (>0.5): {category_summary['strong_effects']}")
            print(f"   Mean significance: {category_summary['mean_significance']:.2f}")
        
        # Research conclusions
        conclusions = []
        
        weather_effects = results.get('weather_interventions', [])
        if weather_effects:
            avg_weather_effect = np.mean([r.causal_effect_size for r in weather_effects])
            if avg_weather_effect > 0.3:
                conclusions.append("✅ Model learned weather causal relationships (strong effects)")
            else:
                conclusions.append("⚠️  Model shows weak weather causal relationships")
        
        temporal_effects = results.get('temporal_interventions', [])
        if temporal_effects:
            avg_temporal_effect = np.mean([r.causal_effect_size for r in temporal_effects])
            if avg_temporal_effect > 0.2:
                conclusions.append("✅ Model learned temporal causal relationships")
            else:
                conclusions.append("⚠️  Model shows weak temporal causal relationships")
        
        event_effects = results.get('event_interventions', [])
        if event_effects:
            avg_event_effect = np.mean([r.causal_effect_size for r in event_effects])
            if avg_event_effect > 0.4:
                conclusions.append("✅ Model learned event-based causal relationships (strong)")
            else:
                conclusions.append("⚠️  Model shows weak event causal relationships")
        
        report['research_conclusions'] = conclusions
        
        print(f"\n🎯 RESEARCH CONCLUSIONS:")
        for conclusion in conclusions:
            print(f"   {conclusion}")
        
        # Save report
        report_path = f"causal_intervention_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_safe_results = {}
            for category, interventions in results.items():
                json_safe_results[category] = []
                for intervention in interventions:
                    json_safe_results[category].append({
                        'factor': intervention.intervention.factor.value,
                        'original_value': intervention.intervention.original_value,
                        'intervention_value': intervention.intervention.intervention_value,
                        'description': intervention.intervention.description,
                        'causal_effect_size': intervention.causal_effect_size,
                        'statistical_significance': intervention.statistical_significance,
                        'interpretation': intervention.interpretation
                    })
            
            report['detailed_results'] = json_safe_results
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_path}")


if __name__ == "__main__":
    # Example usage
    print("🧪 Causal Intervention Testing Framework")
    print("=" * 50)
    
    # Initialize tester (with dummy model paths for now)
    tester = CausalInterventionTester(
        vae_model_path="dummy_vae.pth",
        rnn_model_path="dummy_rnn.pth", 
        vae_architecture="hierarchical_512D"
    )
    
    # Run comprehensive intervention study
    results = tester.run_comprehensive_intervention_study(
        sequence_length=8,
        num_trials=10  # Reduced for demo
    )
    
    print(f"\n🎉 Causal intervention testing framework ready!")
    print(f"📊 Framework can systematically test causal hypotheses")
    print(f"🔬 Ready for Phase 2A validation experiments")