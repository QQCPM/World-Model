#!/usr/bin/env python3
"""
Causal Data Generation Script
Generates structured exploration data with systematic causal state coverage
Based on World Models 01_generate_data.py but with causal-aware exploration

Usage:
python3 01_generate_causal_data.py --total_episodes 2000 --time_steps 200 --parallel_envs 16
"""

import numpy as np
import random
import os
import sys
import argparse
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
import time

# Add local modules
sys.path.append('causal_envs')
sys.path.append('utils')

from test_pipeline import TestCampusEnv, TestExploration  # Use simplified versions for now
# from campus_env import SimpleCampusEnv  # When dependencies available
# from exploration import CausalAwareExploration


class CausalDataGenerator:
    """Generates structured causal data with systematic coverage"""
    
    def __init__(self, output_dir='./data/causal_episodes/', num_parallel_envs=16):
        self.output_dir = output_dir
        self.num_parallel_envs = num_parallel_envs
        self.causal_scheduler = CausalStateScheduler()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"CausalDataGenerator initialized:")
        print(f"  Output directory: {output_dir}")
        print(f"  Parallel environments: {num_parallel_envs}")
    
    def generate_structured_episodes(self, total_episodes, time_steps, render=False):
        """Generate episodes with systematic causal coverage"""
        
        print(f"\n🎯 Generating {total_episodes} episodes with structured causal exploration")
        print(f"   Episode length: {time_steps} steps")
        
        # Get causal combinations to cover
        causal_combinations = self.causal_scheduler.get_structured_combinations()
        print(f"   Causal combinations to cover: {len(causal_combinations)}")
        
        # Distribute episodes across combinations
        episodes_per_combination = max(1, total_episodes // len(causal_combinations))
        print(f"   Episodes per combination: {episodes_per_combination}")
        
        # Generate episode assignments
        episode_assignments = []
        episode_id = 0
        
        for combination_name, causal_state, difficulty in causal_combinations:
            for _ in range(episodes_per_combination):
                if episode_id >= total_episodes:
                    break
                    
                episode_assignments.append({
                    'episode_id': episode_id,
                    'causal_state': causal_state,
                    'combination_name': combination_name,
                    'difficulty': difficulty,
                    'time_steps': time_steps,
                    'render': render
                })
                episode_id += 1
        
        # Fill remaining episodes with random states
        while len(episode_assignments) < total_episodes:
            random_state = self._sample_random_causal_state()
            episode_assignments.append({
                'episode_id': episode_id,
                'causal_state': random_state,
                'combination_name': 'random',
                'difficulty': 'varied',
                'time_steps': time_steps,
                'render': render
            })
            episode_id += 1
        
        # Shuffle for parallel processing
        random.shuffle(episode_assignments)
        
        # Process in parallel
        start_time = time.time()
        
        if self.num_parallel_envs > 1:
            # Parallel processing
            with mp.Pool(self.num_parallel_envs) as pool:
                results = pool.map(self._generate_single_episode, episode_assignments)
        else:
            # Sequential processing for debugging
            results = [self._generate_single_episode(assignment) for assignment in episode_assignments]
        
        # Analyze results
        successful_episodes = [r for r in results if r['success']]
        failed_episodes = [r for r in results if not r['success']]
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n📊 Data generation completed in {duration:.1f} seconds:")
        print(f"   ✅ Successful episodes: {len(successful_episodes)}/{total_episodes}")
        print(f"   ❌ Failed episodes: {len(failed_episodes)}")
        if successful_episodes:
            avg_reward = np.mean([r['total_reward'] for r in successful_episodes])
            avg_steps = np.mean([r['actual_steps'] for r in successful_episodes])
            print(f"   📈 Average reward: {avg_reward:.2f}")
            print(f"   📏 Average episode length: {avg_steps:.1f} steps")
        
        # Save generation summary
        self._save_generation_summary(successful_episodes, failed_episodes, causal_combinations)
        
        return successful_episodes
    
    def _generate_single_episode(self, assignment):
        """Generate a single episode with specified causal state"""
        try:
            episode_id = assignment['episode_id']
            causal_state = assignment['causal_state']
            time_steps = assignment['time_steps']
            
            # Create environment and explorer
            env = TestCampusEnv()
            explorer = TestExploration()
            
            # Reset with specified causal state and random goal
            goal_types = ['library', 'gym', 'cafeteria', 'academic', 'random']
            goal_type = random.choice(goal_types)
            
            observation, info = env.reset(options={
                'causal_state': causal_state,
                'goal_type': goal_type
            })
            
            # Episode data collection
            obs_sequence = []
            action_sequence = []
            reward_sequence = []
            done_sequence = []
            causal_sequence = []
            info_sequence = []
            
            total_reward = 0
            t = 0
            
            while t < time_steps:
                # Store current state
                obs_sequence.append(observation.copy())
                causal_sequence.append(info['causal_encoding'].copy())
                info_sequence.append({
                    'agent_pos': info['agent_pos'].copy(),
                    'goal_pos': info['goal_pos'].copy(),
                    'goal_type': info.get('goal_type', goal_type),
                    'episode_step': t
                })
                
                # Select action using structured exploration
                action = explorer.select_action(
                    info['agent_pos'],
                    info['goal_pos'],
                    causal_state,
                    t
                )
                
                # Execute action
                observation, reward, terminated, truncated, info = env.step(action)
                
                # Store results
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(terminated or truncated)
                
                total_reward += reward
                t += 1
                
                if terminated or truncated:
                    break
            
            # Save episode data
            filename = os.path.join(self.output_dir, f"episode_{episode_id:06d}.npz")
            np.savez_compressed(
                filename,
                obs=np.array(obs_sequence),
                action=np.array(action_sequence),
                reward=np.array(reward_sequence),
                done=np.array(done_sequence),
                causal=np.array(causal_sequence),
                info=info_sequence,
                metadata={
                    'episode_id': episode_id,
                    'causal_state': causal_state,
                    'combination_name': assignment['combination_name'],
                    'difficulty': assignment['difficulty'],
                    'goal_type': goal_type,
                    'total_reward': total_reward,
                    'actual_steps': t,
                    'terminated': terminated if 't' in locals() else False,
                    'truncated': truncated if 't' in locals() else False
                }
            )
            
            return {
                'success': True,
                'episode_id': episode_id,
                'total_reward': total_reward,
                'actual_steps': t,
                'combination_name': assignment['combination_name'],
                'filename': filename
            }
            
        except Exception as e:
            print(f"❌ Episode {assignment['episode_id']} failed: {e}")
            return {
                'success': False,
                'episode_id': assignment['episode_id'],
                'error': str(e)
            }
    
    def _sample_random_causal_state(self):
        """Sample a random causal state for variety"""
        return {
            'time_hour': random.randint(0, 23),
            'day_week': random.randint(0, 6),
            'weather': random.choice(['sunny', 'rain', 'snow', 'fog']),
            'event': random.choice(['normal', 'gameday', 'exam', 'break', 'construction']),
            'crowd_density': random.randint(1, 5)
        }
    
    def _save_generation_summary(self, successful_episodes, failed_episodes, causal_combinations):
        """Save summary of data generation process"""
        summary = {
            'generation_time': time.time(),
            'total_episodes': len(successful_episodes) + len(failed_episodes),
            'successful_episodes': len(successful_episodes),
            'failed_episodes': len(failed_episodes),
            'causal_combinations_used': len(causal_combinations),
            'combination_coverage': {}
        }
        
        # Analyze combination coverage
        for combo_name, _, difficulty in causal_combinations:
            combo_episodes = [e for e in successful_episodes if e['combination_name'] == combo_name]
            summary['combination_coverage'][combo_name] = {
                'count': len(combo_episodes),
                'difficulty': difficulty,
                'avg_reward': np.mean([e['total_reward'] for e in combo_episodes]) if combo_episodes else 0
            }
        
        # Save to file
        summary_file = os.path.join(self.output_dir, 'generation_summary.npz')
        np.savez_compressed(summary_file, **summary)
        print(f"📋 Generation summary saved to: {summary_file}")


class CausalStateScheduler:
    """Manages systematic coverage of causal state combinations"""
    
    def __init__(self):
        self.curriculum_phases = ['easy', 'medium', 'hard']
        
    def get_structured_combinations(self):
        """Get systematic causal combinations for structured exploration"""
        combinations = []
        
        # Phase 1: Single factor variations (Easy)
        base_state = {
            'time_hour': 14,
            'day_week': 2,  # Tuesday
            'weather': 'sunny',
            'event': 'normal',
            'crowd_density': 2
        }
        
        # Weather variations
        for weather in ['sunny', 'rain', 'snow', 'fog']:
            state = base_state.copy()
            state['weather'] = weather
            combinations.append((f'weather_{weather}', state, 'easy'))
        
        # Time variations
        for time_hour, time_name in [(6, 'dawn'), (8, 'morning'), (14, 'afternoon'), (18, 'evening'), (22, 'night')]:
            state = base_state.copy()
            state['time_hour'] = time_hour
            combinations.append((f'time_{time_name}', state, 'easy'))
        
        # Day variations
        for day, day_name in [(0, 'monday'), (2, 'wednesday'), (4, 'friday'), (6, 'sunday')]:
            state = base_state.copy()
            state['day_week'] = day
            combinations.append((f'day_{day_name}', state, 'easy'))
        
        # Crowd variations
        for crowd, crowd_name in [(1, 'low'), (3, 'medium'), (5, 'high')]:
            state = base_state.copy()
            state['crowd_density'] = crowd
            combinations.append((f'crowd_{crowd_name}', state, 'easy'))
        
        # Event variations
        for event in ['normal', 'gameday', 'exam', 'break', 'construction']:
            state = base_state.copy()
            state['event'] = event
            combinations.append((f'event_{event}', state, 'easy'))
        
        # Phase 2: Two-factor interactions (Medium)
        interaction_combinations = [
            ('rain_high_crowd', {
                'time_hour': 8, 'day_week': 1, 'weather': 'rain', 
                'event': 'normal', 'crowd_density': 4
            }),
            ('night_low_crowd', {
                'time_hour': 22, 'day_week': 5, 'weather': 'sunny', 
                'event': 'normal', 'crowd_density': 1
            }),
            ('gameday_afternoon', {
                'time_hour': 14, 'day_week': 5, 'weather': 'sunny', 
                'event': 'gameday', 'crowd_density': 5
            }),
            ('exam_week_morning', {
                'time_hour': 8, 'day_week': 1, 'weather': 'sunny', 
                'event': 'exam', 'crowd_density': 4
            }),
            ('construction_snow', {
                'time_hour': 10, 'day_week': 3, 'weather': 'snow', 
                'event': 'construction', 'crowd_density': 2
            }),
            ('weekend_break', {
                'time_hour': 14, 'day_week': 6, 'weather': 'sunny', 
                'event': 'break', 'crowd_density': 1
            })
        ]
        
        for name, state in interaction_combinations:
            combinations.append((name, state, 'medium'))
        
        # Phase 3: Complex multi-factor scenarios (Hard)
        complex_combinations = [
            ('perfect_storm', {
                'time_hour': 20, 'day_week': 4, 'weather': 'rain', 
                'event': 'gameday', 'crowd_density': 5
            }),
            ('quiet_night', {
                'time_hour': 2, 'day_week': 6, 'weather': 'fog', 
                'event': 'break', 'crowd_density': 1
            }),
            ('busy_morning', {
                'time_hour': 8, 'day_week': 1, 'weather': 'snow', 
                'event': 'exam', 'crowd_density': 5
            }),
            ('construction_chaos', {
                'time_hour': 12, 'day_week': 2, 'weather': 'rain', 
                'event': 'construction', 'crowd_density': 4
            }),
            ('gameday_snow', {
                'time_hour': 16, 'day_week': 5, 'weather': 'snow', 
                'event': 'gameday', 'crowd_density': 5
            })
        ]
        
        for name, state in complex_combinations:
            combinations.append((name, state, 'hard'))
        
        print(f"Generated {len(combinations)} structured causal combinations:")
        for phase in ['easy', 'medium', 'hard']:
            phase_count = len([c for c in combinations if c[2] == phase])
            print(f"  {phase.capitalize()}: {phase_count} combinations")
        
        return combinations


def main():
    parser = argparse.ArgumentParser(description='Generate causal world models training data')
    parser.add_argument('--total_episodes', type=int, default=1000,
                       help='Total number of episodes to generate')
    parser.add_argument('--time_steps', type=int, default=200,
                       help='Maximum timesteps per episode')
    parser.add_argument('--parallel_envs', type=int, default=16,
                       help='Number of parallel environments')
    parser.add_argument('--render', action='store_true',
                       help='Render environments during generation')
    parser.add_argument('--output_dir', type=str, default='./data/causal_episodes/',
                       help='Output directory for generated data')
    
    args = parser.parse_args()
    
    print("🚀 Causal World Models Data Generation")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Episodes: {args.total_episodes}")
    print(f"  Max steps per episode: {args.time_steps}")
    print(f"  Parallel environments: {args.parallel_envs}")
    print(f"  Render: {args.render}")
    print(f"  Output directory: {args.output_dir}")
    
    # Create data generator
    generator = CausalDataGenerator(
        output_dir=args.output_dir,
        num_parallel_envs=args.parallel_envs
    )
    
    # Generate structured episodes
    results = generator.generate_structured_episodes(
        total_episodes=args.total_episodes,
        time_steps=args.time_steps,
        render=args.render
    )
    
    print(f"\n🎉 Data generation completed successfully!")
    print(f"📁 Data saved to: {args.output_dir}")
    print(f"📊 Episodes generated: {len(results)}")
    
    # Quick data verification
    if results:
        sample_file = results[0]['filename']
        sample_data = np.load(sample_file, allow_pickle=True)
        print(f"\n🔍 Sample episode verification:")
        print(f"   Observation shape: {sample_data['obs'].shape}")
        print(f"   Action shape: {sample_data['action'].shape}")
        print(f"   Causal shape: {sample_data['causal'].shape}")
        print(f"   Reward sum: {sample_data['reward'].sum():.2f}")


if __name__ == "__main__":
    main()