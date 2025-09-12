#!/usr/bin/env python3
"""
Quick visualization runner for Causal World Models
Demonstrates different causal scenarios with visual output
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add CausalWorldModels to path
sys.path.append('CausalWorldModels')
sys.path.append('CausalWorldModels/causal_envs')

# Try importing the full environment, fall back to test version
try:
    from CausalWorldModels.causal_envs.campus_env import SimpleCampusEnv
    print("✅ Using full SimpleCampusEnv")
    use_full_env = True
except ImportError:
    try:
        from CausalWorldModels.test_pipeline import TestCampusEnv as SimpleCampusEnv
        print("⚠️  Using TestCampusEnv (simplified version)")
        use_full_env = False
    except ImportError:
        print("❌ Could not import environment. Make sure you're in the right directory.")
        sys.exit(1)

def visualize_causal_scenarios():
    """Show different causal scenarios side by side"""
    
    scenarios = [
        {
            'name': 'Sunny Morning',
            'state': {'time_hour': 8, 'day_week': 1, 'weather': 'sunny', 'event': 'normal', 'crowd_density': 2}
        },
        {
            'name': 'Rainy Gameday',
            'state': {'time_hour': 14, 'day_week': 4, 'weather': 'rain', 'event': 'gameday', 'crowd_density': 5}
        },
        {
            'name': 'Snowy Night',
            'state': {'time_hour': 22, 'day_week': 6, 'weather': 'snow', 'event': 'construction', 'crowd_density': 1}
        },
        {
            'name': 'Exam Week',
            'state': {'time_hour': 10, 'day_week': 2, 'weather': 'fog', 'event': 'exam', 'crowd_density': 4}
        }
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    env = SimpleCampusEnv()
    
    for i, scenario in enumerate(scenarios):
        print(f"\n🎬 Rendering scenario: {scenario['name']}")
        
        # Reset with specific causal state
        obs, info = env.reset(options={'causal_state': scenario['state']})
        
        # Display the observation
        axes[i].imshow(obs)
        axes[i].set_title(f"{scenario['name']}\n{scenario['state']['weather'].title()} | {scenario['state']['event'].title()}")
        axes[i].axis('off')
        
        # Print scenario details
        print(f"  Time: {scenario['state']['time_hour']}:00")
        print(f"  Weather: {scenario['state']['weather']}")
        print(f"  Event: {scenario['state']['event']}")
        print(f"  Crowd: {scenario['state']['crowd_density']}/5")
        print(f"  Agent: {info['agent_pos']}, Goal: {info['goal_pos']}")
        
    plt.suptitle('Causal Campus Environment - Different Scenarios', fontsize=16)
    plt.tight_layout()
    plt.savefig('causal_scenarios.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization to: causal_scenarios.png")
    plt.show()

def simulate_episode(scenario_name="rainy_gameday", steps=20):
    """Simulate and visualize an episode"""
    
    causal_states = {
        'sunny_normal': {'time_hour': 14, 'day_week': 2, 'weather': 'sunny', 'event': 'normal', 'crowd_density': 2},
        'rainy_gameday': {'time_hour': 16, 'day_week': 4, 'weather': 'rain', 'event': 'gameday', 'crowd_density': 5},
        'night_quiet': {'time_hour': 22, 'day_week': 6, 'weather': 'fog', 'event': 'break', 'crowd_density': 1},
        'busy_morning': {'time_hour': 8, 'day_week': 1, 'weather': 'snow', 'event': 'exam', 'crowd_density': 4}
    }
    
    if scenario_name not in causal_states:
        scenario_name = 'rainy_gameday'
        
    causal_state = causal_states[scenario_name]
    
    print(f"\n🎮 Simulating episode: {scenario_name}")
    print(f"   Scenario: {causal_state}")
    
    env = SimpleCampusEnv()
    obs, info = env.reset(options={'causal_state': causal_state, 'goal_type': 'library'})
    
    # Simple goal-directed policy
    trajectory = []
    total_reward = 0
    
    for step in range(steps):
        # Store current state
        trajectory.append({
            'step': step,
            'obs': obs.copy(),
            'agent_pos': info['agent_pos'].copy(),
            'reward': 0 if step == 0 else trajectory[-1]['reward']
        })
        
        # Simple navigation toward goal
        diff = info['goal_pos'] - info['agent_pos']
        if abs(diff[0]) > abs(diff[1]):
            action = 2 if diff[0] > 0 else 3  # East/West
        elif abs(diff[1]) > 0:
            action = 1 if diff[1] > 0 else 0  # South/North
        else:
            action = 4  # Stay
            
        # Add some randomness
        if np.random.random() < 0.1:
            action = np.random.randint(0, 5)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        trajectory[-1]['reward'] = reward
        
        action_names = ['North', 'South', 'East', 'West', 'Stay']
        distance = np.linalg.norm(info['agent_pos'] - info['goal_pos'])
        
        print(f"  Step {step:2d}: {action_names[action]:5s} -> pos={info['agent_pos']}, "
              f"reward={reward:+.2f}, distance={distance:.1f}")
        
        if terminated or truncated:
            print(f"  🎯 Episode ended: {'Goal reached!' if terminated else 'Max steps reached'}")
            break
    
    print(f"\n📊 Episode Summary:")
    print(f"   Total steps: {len(trajectory)}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Final distance to goal: {np.linalg.norm(info['agent_pos'] - info['goal_pos']):.1f}")
    
    # Visualize trajectory
    if len(trajectory) > 1:
        visualize_trajectory(trajectory, scenario_name)
    
    return trajectory

def visualize_trajectory(trajectory, scenario_name):
    """Visualize the agent's trajectory"""
    
    # Show first, middle, and last frames
    frames_to_show = [0, len(trajectory)//2, -1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, frame_idx in enumerate(frames_to_show):
        frame = trajectory[frame_idx]
        axes[i].imshow(frame['obs'])
        axes[i].set_title(f"Step {frame['step']}\nPos: {frame['agent_pos']}, Reward: {frame['reward']:+.2f}")
        axes[i].axis('off')
    
    plt.suptitle(f'Episode Trajectory: {scenario_name.replace("_", " ").title()}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'trajectory_{scenario_name}.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved trajectory to: trajectory_{scenario_name}.png")
    plt.show()

def main():
    print("🏛️  Causal Campus Navigation - Visualization Demo")
    print("=" * 60)
    
    # Test 1: Show different causal scenarios
    print("\n1️⃣  Visualizing Different Causal Scenarios...")
    try:
        visualize_causal_scenarios()
    except Exception as e:
        print(f"❌ Scenario visualization failed: {e}")
    
    # Test 2: Simulate an episode
    print("\n2️⃣  Simulating Navigation Episode...")
    try:
        trajectory = simulate_episode("rainy_gameday", steps=15)
        print(f"✅ Episode simulation completed with {len(trajectory)} steps")
    except Exception as e:
        print(f"❌ Episode simulation failed: {e}")
    
    # Test 3: Compare scenarios
    print("\n3️⃣  Comparing Multiple Scenarios...")
    scenarios = ['sunny_normal', 'rainy_gameday', 'night_quiet', 'busy_morning']
    
    for scenario in scenarios:
        try:
            print(f"\n🔄 Testing {scenario}...")
            traj = simulate_episode(scenario, steps=10)
            total_reward = sum(frame['reward'] for frame in traj)
            print(f"   Final reward: {total_reward:.2f}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n🎉 Visualization demo completed!")
    print(f"📁 Check current directory for saved images:")
    print(f"   - causal_scenarios.png")
    print(f"   - trajectory_*.png")

if __name__ == "__main__":
    main()