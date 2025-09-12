#!/usr/bin/env python3
"""
Interactive Campus Explorer
Use keyboard to control agent and see causal effects in real-time
"""

import sys
import os
import numpy as np

# Add CausalWorldModels to path
sys.path.append('CausalWorldModels')
sys.path.append('CausalWorldModels/causal_envs')

try:
    from campus_env import SimpleCampusEnv
    print("✅ Using full SimpleCampusEnv")
except ImportError:
    from test_pipeline import TestCampusEnv as SimpleCampusEnv
    print("⚠️  Using TestCampusEnv (simplified)")

def print_observation(obs, info, step, total_reward):
    """Print ASCII representation of the observation"""
    print("\n" + "="*70)
    print(f"Step {step} | Total Reward: {total_reward:.2f} | Agent: {info['agent_pos']} | Goal: {info['goal_pos']}")
    
    # Simple ASCII representation
    print("\nCampus Map (simplified view):")
    print("🟫 = Buildings, 🛤️ = Paths, 🎯 = Goal, 🚶 = Agent")
    
    # Create simple ASCII grid
    grid = [['⬛' for _ in range(16)] for _ in range(16)]  # 16x16 simplified view
    
    # Mark buildings (scaled down)
    buildings = {
        'library': (4, 4, 6, 6),
        'gym': (10, 3, 13, 5),
        'cafeteria': (3, 10, 5, 13),
        'academic': (9, 9, 13, 13),
        'stadium': (1, 1, 4, 4),
        'dorms': (13, 11, 15, 14)
    }
    
    for name, (x1, y1, x2, y2) in buildings.items():
        for y in range(max(0,y1), min(16,y2)):
            for x in range(max(0,x1), min(16,x2)):
                grid[y][x] = '🟫'
    
    # Mark paths
    for y in range(7, 9):  # Horizontal path
        for x in range(16):
            if grid[y][x] == '⬛':
                grid[y][x] = '🛤️'
    for x in range(7, 9):  # Vertical path
        for y in range(16):
            if grid[y][x] == '⬛':
                grid[y][x] = '🛤️'
    
    # Mark goal and agent (scaled positions)
    goal_x, goal_y = int(info['goal_pos'][0] // 4), int(info['goal_pos'][1] // 4)
    agent_x, agent_y = int(info['agent_pos'][0] // 4), int(info['agent_pos'][1] // 4)
    
    if 0 <= goal_y < 16 and 0 <= goal_x < 16:
        grid[goal_y][goal_x] = '🎯'
    if 0 <= agent_y < 16 and 0 <= agent_x < 16:
        grid[agent_y][agent_x] = '🚶'
    
    # Print grid
    for row in grid:
        print(''.join(row))

def print_causal_state(causal_state):
    """Print current causal state in readable format"""
    time_names = ['12AM','1AM','2AM','3AM','4AM','5AM','6AM','7AM','8AM','9AM','10AM','11AM',
                  '12PM','1PM','2PM','3PM','4PM','5PM','6PM','7PM','8PM','9PM','10PM','11PM']
    day_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    
    print(f"\n🌍 Current Causal State:")
    print(f"  ⏰ Time: {time_names[causal_state['time_hour']]} ({causal_state['time_hour']}:00)")
    print(f"  📅 Day: {day_names[causal_state['day_week']]}")
    print(f"  🌤️  Weather: {causal_state['weather'].title()}")
    print(f"  🎪 Event: {causal_state['event'].title()}")
    print(f"  👥 Crowd: {causal_state['crowd_density']}/5 {'🟥'*causal_state['crowd_density']+'⬜'*(5-causal_state['crowd_density'])}")

def interactive_exploration():
    """Interactive campus exploration with keyboard controls"""
    
    print("🏛️  Interactive Campus Explorer")
    print("="*50)
    print("\nControls:")
    print("  WASD or Arrow Keys: Move (W/↑=North, S/↓=South, A/←=West, D/→=East)")
    print("  SPACE: Stay in place")
    print("  C: Change causal state")
    print("  R: Reset episode")
    print("  Q: Quit")
    
    # Initialize environment
    env = SimpleCampusEnv()
    
    # Start with interesting causal state
    causal_state = {
        'time_hour': 14,
        'day_week': 4,  # Friday
        'weather': 'rain',
        'event': 'gameday',
        'crowd_density': 4
    }
    
    obs, info = env.reset(options={'causal_state': causal_state, 'goal_type': 'library'})
    
    step = 0
    total_reward = 0
    
    while True:
        # Display current state
        print_observation(obs, info, step, total_reward)
        print_causal_state(causal_state)
        
        # Get user input
        print(f"\nNext move? (WASD/Arrows to move, SPACE=stay, C=change causal, R=reset, Q=quit)")
        
        try:
            user_input = input(">>> ").lower().strip()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        
        if user_input in ['q', 'quit', 'exit']:
            print("\n👋 Goodbye!")
            break
        
        elif user_input in ['r', 'reset']:
            print("\n🔄 Resetting episode...")
            obs, info = env.reset(options={'causal_state': causal_state})
            step = 0
            total_reward = 0
            continue
            
        elif user_input in ['c', 'change', 'causal']:
            print("\n🌍 Changing causal state...")
            causal_state = change_causal_state(causal_state)
            obs, info = env.reset(options={'causal_state': causal_state})
            step = 0
            total_reward = 0
            continue
        
        # Map input to actions
        action = None
        if user_input in ['w', 'up', '↑']:
            action = 0  # North
        elif user_input in ['s', 'down', '↓']:
            action = 1  # South
        elif user_input in ['d', 'right', '→']:
            action = 2  # East
        elif user_input in ['a', 'left', '←']:
            action = 3  # West
        elif user_input in [' ', 'space', 'stay']:
            action = 4  # Stay
        else:
            print(f"❓ Unknown command: {user_input}")
            continue
        
        # Execute action
        prev_pos = info['agent_pos'].copy()
        obs, reward, terminated, truncated, info = env.step(action)
        
        step += 1
        total_reward += reward
        
        # Show action result
        action_names = ['North', 'South', 'East', 'West', 'Stay']
        distance = np.linalg.norm(info['agent_pos'] - info['goal_pos'])
        
        print(f"\n📍 Action: {action_names[action]}")
        print(f"   {prev_pos} → {info['agent_pos']}")
        print(f"   Reward: {reward:+.3f}")
        print(f"   Distance to goal: {distance:.1f}")
        
        if hasattr(info, 'reward_components'):
            components = info['reward_components']
            print(f"   Reward breakdown: Progress={components.get('progress',0):.2f}, "
                  f"Causal={components.get('causal',0):.2f}, Goal={components.get('goal',0):.2f}")
        
        # Check episode end
        if terminated:
            print(f"\n🎯 SUCCESS! You reached the goal in {step} steps!")
            print(f"   Final reward: {total_reward:.2f}")
            print("\nPress R to reset or Q to quit")
        elif truncated:
            print(f"\n⏰ Episode ended (max steps reached)")
            print(f"   Final reward: {total_reward:.2f}")
            print("\nPress R to reset or Q to quit")

def change_causal_state(current_state):
    """Interactive causal state changer"""
    new_state = current_state.copy()
    
    print("\nWhat would you like to change?")
    print("1. Time of day")
    print("2. Weather")
    print("3. Campus event")
    print("4. Crowd density")
    print("5. Random scenario")
    print("6. Keep current")
    
    choice = input("Choose (1-6): ").strip()
    
    if choice == '1':
        print("\nTime options:")
        times = [(6, "6AM - Dawn"), (8, "8AM - Morning Rush"), (12, "12PM - Lunch"),
                (16, "4PM - Afternoon"), (20, "8PM - Evening"), (22, "10PM - Night")]
        for i, (hour, desc) in enumerate(times):
            print(f"{i+1}. {desc}")
        
        try:
            time_choice = int(input("Pick time (1-6): ")) - 1
            if 0 <= time_choice < len(times):
                new_state['time_hour'] = times[time_choice][0]
                print(f"✅ Set time to {times[time_choice][1]}")
        except:
            pass
    
    elif choice == '2':
        weathers = ['sunny', 'rain', 'snow', 'fog']
        print(f"\nWeather options: {', '.join(w.title() for w in weathers)}")
        weather = input("Choose weather: ").lower().strip()
        if weather in weathers:
            new_state['weather'] = weather
            print(f"✅ Set weather to {weather}")
    
    elif choice == '3':
        events = ['normal', 'gameday', 'exam', 'break', 'construction']
        print(f"\nEvent options: {', '.join(e.title() for e in events)}")
        event = input("Choose event: ").lower().strip()
        if event in events:
            new_state['event'] = event
            print(f"✅ Set event to {event}")
    
    elif choice == '4':
        print("\nCrowd density (1=very low, 5=very high):")
        try:
            crowd = int(input("Choose 1-5: "))
            if 1 <= crowd <= 5:
                new_state['crowd_density'] = crowd
                print(f"✅ Set crowd density to {crowd}")
        except:
            pass
    
    elif choice == '5':
        scenarios = {
            'Perfect Storm': {'time_hour': 20, 'weather': 'rain', 'event': 'gameday', 'crowd_density': 5},
            'Quiet Night': {'time_hour': 2, 'weather': 'fog', 'event': 'break', 'crowd_density': 1},
            'Busy Morning': {'time_hour': 8, 'weather': 'snow', 'event': 'exam', 'crowd_density': 5},
            'Nice Day': {'time_hour': 14, 'weather': 'sunny', 'event': 'normal', 'crowd_density': 2}
        }
        
        print(f"\nRandom scenarios:")
        for i, name in enumerate(scenarios.keys()):
            print(f"{i+1}. {name}")
        
        try:
            scenario_choice = int(input("Pick scenario (1-4): ")) - 1
            scenario_names = list(scenarios.keys())
            if 0 <= scenario_choice < len(scenario_names):
                scenario_name = scenario_names[scenario_choice]
                new_state.update(scenarios[scenario_name])
                print(f"✅ Applied '{scenario_name}' scenario")
        except:
            pass
    
    return new_state

if __name__ == "__main__":
    interactive_exploration()