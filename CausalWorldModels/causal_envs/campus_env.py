"""
Simple Campus Environment with Causal State Rendering
Based on the Complete Project Plan - implements 64x64 grid world with buildings, 
causal effects (weather, crowds, events), and structured reward function.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Union
try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    print("⚠️  OpenAI Gym not available - using basic implementation")
    GYM_AVAILABLE = False
    
    # Create minimal gym-like interface
    class spaces:
        class Discrete:
            def __init__(self, n):
                self.n = n
            def sample(self):
                import random
                return random.randint(0, self.n-1)
        
        class Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype
    
    class gym:
        class Env:
            def __init__(self):
                pass
            def reset(self, seed=None, options=None):
                pass
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("⚠️  OpenCV not available - using basic implementation")
    CV2_AVAILABLE = False


class SimpleCampusEnv(gym.Env):
    """
    64x64 grid world representation of campus with causal state effects.
    
    Causal Variables (45 dimensions total):
    - time_hour: 24 (0-23 one-hot)
    - day_week: 7 (Monday-Sunday one-hot)  
    - weather: 4 (sunny/rain/snow/fog one-hot)
    - event: 5 (normal/gameday/exam/break/construction one-hot)
    - crowd_density: 5 (very_low to very_high one-hot)
    """
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Environment dimensions
        self.grid_size = 64
        self.action_space = spaces.Discrete(5)  # N, S, E, W, Stay
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(64, 64, 3), 
            dtype=np.uint8
        )
        
        # Campus layout - building coordinates (x1, y1, x2, y2)
        self.buildings = {
            'library': (15, 15, 25, 25),      # Academic building
            'gym': (40, 10, 50, 20),          # Recreation center  
            'cafeteria': (10, 40, 20, 50),    # Dining facility
            'academic': (35, 35, 50, 50),     # Classroom building
            'stadium': (5, 5, 15, 15),        # Stadium (gameday crowds)
            'dorms': (50, 45, 60, 55)         # Dormitories
        }
        
        # Path network (simplified)
        self.main_paths = [
            # Horizontal paths
            ((0, 30), (64, 30)),
            ((0, 35), (64, 35)),
            # Vertical paths
            ((30, 0), (30, 64)),
            ((35, 0), (35, 64)),
        ]
        
        # Spawn points
        self.spawn_points = [(5, 55), (30, 5), (55, 30), (25, 58)]
        
        # Goal locations
        self.goal_locations = {
            'library': (20, 20),
            'gym': (45, 15),
            'cafeteria': (15, 45),
            'academic': (42, 42),
            'random': None  # Will be set randomly
        }
        
        # Causal state dimensions
        self.causal_dims = {
            'time_hour': 24,
            'day_week': 7,
            'weather': 4,
            'event': 5,
            'crowd_density': 5
        }
        self.total_causal_dim = sum(self.causal_dims.values())  # 45
        
        # Current state
        self.agent_pos = None
        self.goal_pos = None
        self.goal_type = None
        self.causal_state = None
        self.episode_step = 0
        self.max_episode_steps = 200
        
        # Reward function components
        self.reward_function = CausalRewardFunction()
        
        # Rendering
        self.render_mode = render_mode
        self._last_rendered = None
    
    def reset(self, seed=None, options=None):
        """Reset environment with random spawn, goal, and causal state"""
        super().reset(seed=seed)
        
        # Random spawn position
        self.agent_pos = np.array(self.spawn_points[np.random.randint(len(self.spawn_points))])
        
        # Random goal (but not same as spawn)
        goal_types = list(self.goal_locations.keys())
        if options and 'goal_type' in options:
            self.goal_type = options['goal_type']
        else:
            self.goal_type = np.random.choice(goal_types)
            
        if self.goal_type == 'random':
            # Random location not on buildings
            while True:
                self.goal_pos = np.random.randint(2, self.grid_size-2, size=2)
                if not self._is_building(self.goal_pos):
                    break
        else:
            self.goal_pos = np.array(self.goal_locations[self.goal_type])
        
        # Random causal state (or from options)
        if options and 'causal_state' in options:
            self.causal_state = options['causal_state'].copy()
        else:
            self.causal_state = self._sample_random_causal_state()
            
        self.episode_step = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Execute action and return next state"""
        prev_pos = self.agent_pos.copy()
        
        # Execute action
        if action == 0:  # North
            new_pos = self.agent_pos + np.array([0, -1])
        elif action == 1:  # South
            new_pos = self.agent_pos + np.array([0, 1])
        elif action == 2:  # East
            new_pos = self.agent_pos + np.array([1, 0])
        elif action == 3:  # West
            new_pos = self.agent_pos + np.array([-1, 0])
        else:  # Stay
            new_pos = self.agent_pos.copy()
            
        # Check bounds and collisions
        new_pos = np.clip(new_pos, 0, self.grid_size-1)
        
        if not self._is_collision(new_pos):
            self.agent_pos = new_pos
        
        # Compute reward
        reward, reward_components = self.reward_function.compute_reward(
            prev_state={'agent_pos': prev_pos},
            action=action,
            next_state={'agent_pos': self.agent_pos},
            causal_state=self.causal_state,
            goal_pos=self.goal_pos
        )
        
        # Check termination
        self.episode_step += 1
        
        # Goal reached
        goal_distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        terminated = goal_distance < 2.0
        
        # Max steps reached
        truncated = self.episode_step >= self.max_episode_steps
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info['reward_components'] = reward_components
        info['goal_distance'] = goal_distance
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Render current state with causal effects"""
        return self.render_with_causals(self.causal_state)
    
    def _get_info(self):
        """Return environment info"""
        return {
            'agent_pos': self.agent_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
            'goal_type': self.goal_type,
            'causal_state': self.causal_state.copy(),
            'episode_step': self.episode_step,
            'causal_encoding': self._encode_causal_state(self.causal_state)
        }
    
    def render_with_causals(self, causal_state):
        """Render 64x64 RGB image with causal effects applied"""
        canvas = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Base background (paths are lighter)
        canvas.fill(50)  # Dark gray background
        
        # Draw main paths
        for (x1, y1), (x2, y2) in self.main_paths:
            if x1 == x2:  # Vertical path
                canvas[y1:y2, max(0,x1-1):min(64,x1+2)] = [100, 100, 100]
            else:  # Horizontal path  
                canvas[max(0,y1-1):min(64,y1+2), x1:x2] = [100, 100, 100]
        
        # Draw buildings
        for building, (x1, y1, x2, y2) in self.buildings.items():
            if building == 'library':
                color = [120, 80, 40]  # Brown
            elif building == 'gym':
                color = [40, 120, 40]  # Green
            elif building == 'cafeteria':
                color = [120, 120, 40]  # Yellow
            elif building == 'stadium':
                color = [120, 40, 40]  # Red
            else:
                color = [80, 80, 120]  # Blue
                
            canvas[y1:y2, x1:x2] = color
        
        # Apply causal effects
        canvas = self._apply_causal_effects(canvas, causal_state)
        
        # Draw goal (bright marker)
        gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
        canvas[max(0,gy-1):min(64,gy+2), max(0,gx-1):min(64,gx+2)] = [255, 255, 0]  # Yellow goal
        
        # Draw agent (bright marker)
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        canvas[max(0,ay-1):min(64,ay+2), max(0,ax-1):min(64,ax+2)] = [255, 0, 255]  # Magenta agent
        
        return canvas
    
    def _apply_causal_effects(self, canvas, causal_state):
        """Apply visual causal effects to the canvas"""
        result = canvas.copy()
        
        # Weather effects
        if causal_state['weather'] == 'rain':
            # Darker overall + puddle effects
            result = (result * 0.7).astype(np.uint8)
            self._add_puddles(result)
            
        elif causal_state['weather'] == 'snow':
            # Much darker + white specks
            result = (result * 0.5).astype(np.uint8) 
            self._add_snow(result)
            
        elif causal_state['weather'] == 'fog':
            # Reduced contrast
            result = (result * 0.8 + 50).astype(np.uint8)
        
        # Time effects
        if causal_state['time_hour'] < 6 or causal_state['time_hour'] > 20:
            # Night - much darker
            result = (result * 0.4).astype(np.uint8)
            
        elif causal_state['time_hour'] < 8 or causal_state['time_hour'] > 18:
            # Dawn/dusk - somewhat darker
            result = (result * 0.7).astype(np.uint8)
        
        # Crowd effects
        if causal_state['crowd_density'] >= 3:  # High or very high
            self._add_crowd_pixels(result, causal_state)
            
        # Event effects
        if causal_state['event'] == 'gameday':
            # Heavy crowds around stadium
            self._add_gameday_crowds(result)
            
        elif causal_state['event'] == 'construction':
            # Orange barriers on some paths
            self._add_construction_barriers(result)
            
        return result
    
    def _add_puddles(self, canvas):
        """Add blue puddle pixels for rain effect"""
        np.random.seed(42)  # Deterministic puddles
        num_puddles = 10
        for _ in range(num_puddles):
            x, y = np.random.randint(0, 64, 2)
            size = np.random.randint(1, 3)
            canvas[max(0,y-size):min(64,y+size+1), max(0,x-size):min(64,x+size+1)] = [0, 50, 150]
    
    def _add_snow(self, canvas):
        """Add white snow pixels"""
        np.random.seed(43)
        num_flakes = 50
        for _ in range(num_flakes):
            x, y = np.random.randint(0, 64, 2)
            canvas[y, x] = [200, 200, 255]
    
    def _add_crowd_pixels(self, canvas, causal_state):
        """Add red pixels representing people in crowded areas"""
        crowd_level = causal_state['crowd_density']
        num_people = crowd_level * 20  # 20-100 people
        
        # Crowds concentrate near buildings during certain hours
        for _ in range(num_people):
            if causal_state['time_hour'] in [8, 12, 17]:  # Peak hours
                # Near buildings
                building = np.random.choice(list(self.buildings.keys()))
                x1, y1, x2, y2 = self.buildings[building]
                x = np.random.randint(max(0, x1-5), min(64, x2+5))
                y = np.random.randint(max(0, y1-5), min(64, y2+5))
            else:
                # Random locations
                x, y = np.random.randint(0, 64, 2)
                
            canvas[y, x] = [200, 0, 0]  # Red person pixel
    
    def _add_gameday_crowds(self, canvas):
        """Heavy crowds around stadium during gameday"""
        x1, y1, x2, y2 = self.buildings['stadium']
        
        # Dense crowd around stadium
        for _ in range(200):  # Very crowded
            x = np.random.randint(max(0, x1-10), min(64, x2+10))
            y = np.random.randint(max(0, y1-10), min(64, y2+10))
            canvas[y, x] = [255, 0, 0]  # Bright red for gameday crowds
    
    def _add_construction_barriers(self, canvas):
        """Add orange construction barriers"""
        # Block some path segments
        barrier_coords = [
            (25, 30, 35, 32),  # Block part of horizontal path
            (30, 20, 32, 30),  # Block part of vertical path
        ]
        
        for x1, y1, x2, y2 in barrier_coords:
            canvas[y1:y2, x1:x2] = [255, 150, 0]  # Orange barriers
    
    def _is_building(self, pos):
        """Check if position is inside a building"""
        x, y = pos
        for (x1, y1, x2, y2) in self.buildings.values():
            if x1 <= x < x2 and y1 <= y < y2:
                return True
        return False
    
    def _is_collision(self, pos):
        """Check if position is blocked (building or barrier)"""
        if self._is_building(pos):
            return True
            
        # Check construction barriers during construction events
        if hasattr(self, 'causal_state') and self.causal_state['event'] == 'construction':
            x, y = pos
            barrier_coords = [
                (25, 30, 35, 32),
                (30, 20, 32, 30),
            ]
            for x1, y1, x2, y2 in barrier_coords:
                if x1 <= x < x2 and y1 <= y < y2:
                    return True
                    
        return False
    
    def _sample_random_causal_state(self):
        """Sample random causal state"""
        return {
            'time_hour': np.random.randint(0, 24),
            'day_week': np.random.randint(0, 7),  # 0=Monday
            'weather': np.random.choice(['sunny', 'rain', 'snow', 'fog']),
            'event': np.random.choice(['normal', 'gameday', 'exam', 'break', 'construction']),
            'crowd_density': np.random.randint(1, 6)  # 1=very_low, 5=very_high
        }
    
    def _encode_causal_state(self, causal_state):
        """Convert causal state to 45-dimensional one-hot encoding"""
        encoding = np.zeros(self.total_causal_dim)
        offset = 0
        
        # Time hour (24 dims)
        encoding[offset + causal_state['time_hour']] = 1
        offset += 24
        
        # Day of week (7 dims)
        encoding[offset + causal_state['day_week']] = 1
        offset += 7
        
        # Weather (4 dims)
        weather_map = {'sunny': 0, 'rain': 1, 'snow': 2, 'fog': 3}
        encoding[offset + weather_map[causal_state['weather']]] = 1
        offset += 4
        
        # Event (5 dims)
        event_map = {'normal': 0, 'gameday': 1, 'exam': 2, 'break': 3, 'construction': 4}
        encoding[offset + event_map[causal_state['event']]] = 1
        offset += 5
        
        # Crowd density (5 dims)
        encoding[offset + causal_state['crowd_density'] - 1] = 1  # -1 because 1-indexed
        
        return encoding
    
    def render(self, mode='rgb_array'):
        """Render current state"""
        if mode == 'rgb_array':
            return self._get_observation()
        elif mode == 'human':
            img = self._get_observation()
            plt.imshow(img)
            plt.title(f"Step {self.episode_step}, Goal: {self.goal_type}")
            plt.show()


class CausalRewardFunction:
    """Multi-objective reward function with causal modifiers"""
    
    def __init__(self):
        self.goal_reward_scale = 10.0
        self.efficiency_scale = 1.0
        self.causal_penalty_scale = 0.5
        
    def compute_reward(self, prev_state, action, next_state, causal_state, goal_pos):
        """Compute reward with causal modifiers"""
        
        # Base Navigation Reward
        dist_to_goal = np.linalg.norm(next_state['agent_pos'] - goal_pos)
        prev_dist = np.linalg.norm(prev_state['agent_pos'] - goal_pos)
        
        # Progress reward (positive for moving toward goal)
        progress_reward = (prev_dist - dist_to_goal) * self.goal_reward_scale
        
        # Efficiency penalty (energy cost of movement)
        action_cost = -0.01 if action != 4 else 0  # 4 = Stay action
        
        # Causal-based modifiers
        causal_modifiers = 0
        
        # Weather penalties
        if causal_state['weather'] == 'rain':
            causal_modifiers -= 0.1  # Harder to navigate in rain
            if action != 4 and self._is_uncovered_path(next_state['agent_pos']):
                causal_modifiers -= 0.2  # Extra penalty for uncovered paths in rain
                
        elif causal_state['weather'] == 'snow':
            causal_modifiers -= 0.15  # Even harder in snow
            
        # Crowd penalties
        crowd_level = causal_state['crowd_density']
        if crowd_level >= 3:  # High or very high
            if self._is_crowded_area(next_state['agent_pos'], causal_state['time_hour']):
                causal_modifiers -= 0.3 * (crowd_level / 5.0)
                
        # Time-based modifiers
        if causal_state['time_hour'] < 6 or causal_state['time_hour'] > 22:
            causal_modifiers -= 0.05  # Slight penalty for night navigation
            
        # Event-based modifiers
        if causal_state['event'] == 'gameday':
            if self._near_stadium(next_state['agent_pos']):
                causal_modifiers -= 1.0  # Heavy penalty near stadium on gameday
                
        elif causal_state['event'] == 'construction':
            # Penalty for trying to go through construction areas
            if action != 4:  # If moving
                causal_modifiers -= 0.3
                
        # Goal achievement bonus
        if dist_to_goal < 2:
            goal_bonus = 100.0
        else:
            goal_bonus = 0
            
        total_reward = progress_reward + action_cost + (causal_modifiers * self.causal_penalty_scale) + goal_bonus
        
        return total_reward, {
            'progress': progress_reward,
            'causal': causal_modifiers,
            'goal': goal_bonus,
            'action_cost': action_cost
        }
    
    def _is_uncovered_path(self, pos):
        """Check if position is on an uncovered path (affected by rain)"""
        # Simplified - assume most paths are uncovered
        return True  # For now, all paths affected by rain
    
    def _is_crowded_area(self, pos, time_hour):
        """Check if position is in a typically crowded area at this time"""
        # Areas near buildings during peak hours
        if time_hour in [8, 12, 17]:  # Peak hours
            # Check if near any building
            for (x1, y1, x2, y2) in [(15, 15, 25, 25), (40, 10, 50, 20), 
                                      (10, 40, 20, 50), (35, 35, 50, 50)]:
                if (x1-5 <= pos[0] <= x2+5) and (y1-5 <= pos[1] <= y2+5):
                    return True
        return False
    
    def _near_stadium(self, pos):
        """Check if position is near stadium"""
        x1, y1, x2, y2 = (5, 5, 15, 15)  # Stadium coords
        return (x1-10 <= pos[0] <= x2+10) and (y1-10 <= pos[1] <= y2+10)


if __name__ == "__main__":
    # Test the environment
    env = SimpleCampusEnv()
    
    # Test with different causal states
    causal_scenarios = [
        {'time_hour': 8, 'day_week': 0, 'weather': 'sunny', 'event': 'normal', 'crowd_density': 2},
        {'time_hour': 14, 'day_week': 4, 'weather': 'rain', 'event': 'gameday', 'crowd_density': 5},
        {'time_hour': 22, 'day_week': 6, 'weather': 'snow', 'event': 'construction', 'crowd_density': 1},
    ]
    
    for i, causal_state in enumerate(causal_scenarios):
        print(f"\n=== Scenario {i+1}: {causal_state} ===")
        
        obs, info = env.reset(options={'causal_state': causal_state, 'goal_type': 'library'})
        print(f"Initial position: {info['agent_pos']}, Goal: {info['goal_pos']}")
        print(f"Causal encoding shape: {info['causal_encoding'].shape}")
        print(f"Observation shape: {obs.shape}")
        
        # Take a few random actions
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step {step}: action={action}, reward={reward:.3f}, "
                  f"reward_components={info['reward_components']}, "
                  f"distance={info['goal_distance']:.1f}")
            
            if terminated or truncated:
                break
                
        print(f"Total reward: {total_reward:.3f}")