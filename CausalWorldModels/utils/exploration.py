"""
Causal-Aware Exploration Strategies
Implements structured exploration that systematically discovers causal effects
instead of random action sampling.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import heapq
from collections import deque


class CausalAwareExploration:
    """Smart exploration that systematically discovers causal effects"""
    
    def __init__(self, env_grid_size=64):
        self.grid_size = env_grid_size
        self.exploration_modes = ['goal_directed', 'causal_discovery', 'edge_case', 'random']
        self.mode_weights = [0.4, 0.3, 0.2, 0.1]
        
        # Buildings for reference (should match campus_env.py)
        self.buildings = {
            'library': (15, 15, 25, 25),
            'gym': (40, 10, 50, 20),
            'cafeteria': (10, 40, 20, 50),
            'academic': (35, 35, 50, 50),
            'stadium': (5, 5, 15, 15),
            'dorms': (50, 45, 60, 55)
        }
        
        # Path planner for goal-directed exploration
        self.path_planner = AStarPlanner(self.grid_size, self.buildings)
        
        # Causal discovery strategies
        self.causal_explorer = CausalDiscoveryExplorer(self.buildings, self.grid_size)
        
        # Edge case scenarios
        self.edge_case_explorer = EdgeCaseExplorer(self.buildings, self.grid_size)
    
    def select_action(self, agent_pos, goal_pos, causal_state, episode_step, mode=None):
        """Select action based on exploration strategy"""
        
        if mode is None:
            # Adaptive mode selection based on episode progress
            if episode_step < 50:  # Early episode - more exploration
                weights = [0.3, 0.4, 0.2, 0.1]
            else:  # Later episode - more goal-directed
                weights = [0.6, 0.2, 0.1, 0.1]
            mode = np.random.choice(self.exploration_modes, p=weights)
        
        if mode == 'goal_directed':
            return self._goal_directed_action(agent_pos, goal_pos)
            
        elif mode == 'causal_discovery':
            return self._causal_discovery_action(agent_pos, causal_state, goal_pos)
            
        elif mode == 'edge_case':
            return self._edge_case_action(agent_pos, causal_state, goal_pos)
            
        else:  # random
            return self._random_action()
    
    def _goal_directed_action(self, agent_pos, goal_pos):
        """A* pathfinding toward goal"""
        path = self.path_planner.find_path(agent_pos, goal_pos)
        
        if len(path) > 1:
            next_pos = path[1]  # path[0] is current position
            return self._pos_to_action(agent_pos, next_pos)
        else:
            return 4  # Stay if already at goal
    
    def _causal_discovery_action(self, agent_pos, causal_state, goal_pos):
        """Explore areas affected by current causal state"""
        return self.causal_explorer.get_exploration_action(agent_pos, causal_state, goal_pos)
    
    def _edge_case_action(self, agent_pos, causal_state, goal_pos):
        """Create challenging scenarios for learning"""
        return self.edge_case_explorer.get_edge_case_action(agent_pos, causal_state, goal_pos)
    
    def _random_action(self):
        """Random action selection"""
        return np.random.randint(0, 5)  # N, S, E, W, Stay
    
    def _pos_to_action(self, current_pos, next_pos):
        """Convert position difference to action"""
        diff = next_pos - current_pos
        
        if diff[1] < 0:  # North
            return 0
        elif diff[1] > 0:  # South
            return 1
        elif diff[0] > 0:  # East
            return 2
        elif diff[0] < 0:  # West
            return 3
        else:  # Same position
            return 4
    
    def generate_causal_curriculum(self):
        """Progressive curriculum for causal discovery"""
        curriculum = []
        
        # Phase 1: Single factors (easy)
        base_states = [
            {'time_hour': 14, 'day_week': 2, 'weather': 'sunny', 'event': 'normal', 'crowd_density': 2},
        ]
        
        # Vary weather
        for weather in ['sunny', 'rain', 'snow', 'fog']:
            state = base_states[0].copy()
            state['weather'] = weather
            curriculum.append(('weather_' + weather, state, 'easy'))
        
        # Vary time
        for time_hour in [8, 14, 20, 2]:  # Morning, afternoon, evening, night
            state = base_states[0].copy() 
            state['time_hour'] = time_hour
            curriculum.append((f'time_{time_hour}', state, 'easy'))
        
        # Vary crowds
        for crowd in [1, 3, 5]:  # Low, high, very high
            state = base_states[0].copy()
            state['crowd_density'] = crowd
            curriculum.append((f'crowd_{crowd}', state, 'easy'))
        
        # Phase 2: Interactions (medium)
        interactions = [
            ('rain_high_crowd', {'time_hour': 8, 'day_week': 1, 'weather': 'rain', 'event': 'normal', 'crowd_density': 4}),
            ('night_low_crowd', {'time_hour': 2, 'day_week': 5, 'weather': 'sunny', 'event': 'normal', 'crowd_density': 1}),
            ('exam_week_library', {'time_hour': 14, 'day_week': 2, 'weather': 'sunny', 'event': 'exam', 'crowd_density': 5}),
        ]
        
        for name, state in interactions:
            curriculum.append((name, state, 'medium'))
        
        # Phase 3: Complex scenarios (hard)
        complex_scenarios = [
            ('gameday_rain_night', {'time_hour': 20, 'day_week': 5, 'weather': 'rain', 'event': 'gameday', 'crowd_density': 5}),
            ('construction_snow_morning', {'time_hour': 8, 'day_week': 1, 'weather': 'snow', 'event': 'construction', 'crowd_density': 3}),
            ('exam_gameday_conflict', {'time_hour': 14, 'day_week': 4, 'weather': 'sunny', 'event': 'gameday', 'crowd_density': 5}),
        ]
        
        for name, state in complex_scenarios:
            curriculum.append((name, state, 'hard'))
        
        return curriculum


class AStarPlanner:
    """A* pathfinding for goal-directed exploration"""
    
    def __init__(self, grid_size, buildings):
        self.grid_size = grid_size
        self.buildings = buildings
    
    def find_path(self, start, goal, avoid_buildings=True):
        """Find shortest path from start to goal"""
        start = tuple(start.astype(int))
        goal = tuple(goal.astype(int))
        
        if start == goal:
            return [np.array(start)]
        
        # A* algorithm
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(np.array(current))
                    current = came_from[current]
                path.append(np.array(start))
                return path[::-1]
            
            # Explore neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:  # W, E, N, S
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue
                
                # Check obstacles
                if avoid_buildings and self._is_building(neighbor):
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found - return direct line
        return [np.array(start), np.array(goal)]
    
    def _heuristic(self, pos1, pos2):
        """Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _is_building(self, pos):
        """Check if position is inside a building"""
        x, y = pos
        for (x1, y1, x2, y2) in self.buildings.values():
            if x1 <= x < x2 and y1 <= y < y2:
                return True
        return False


class CausalDiscoveryExplorer:
    """Explores areas to systematically discover causal effects"""
    
    def __init__(self, buildings, grid_size):
        self.buildings = buildings
        self.grid_size = grid_size
        
        # Areas of interest for different causal factors
        self.weather_test_areas = [
            (20, 20, 30, 30),  # Open area for weather effects
            (0, 30, 64, 35),   # Main horizontal path
        ]
        
        self.crowd_test_areas = [
            (15, 15, 25, 25),  # Library area
            (40, 10, 50, 20),  # Gym area
            (10, 40, 20, 50),  # Cafeteria area
        ]
        
        self.event_test_areas = {
            'gameday': (5, 5, 15, 15),      # Stadium area
            'construction': (25, 30, 35, 35), # Construction zone
            'exam': (15, 15, 25, 25),       # Library during exams
        }
    
    def get_exploration_action(self, agent_pos, causal_state, goal_pos):
        """Get action that explores causal effects"""
        
        # Prioritize based on current causal state
        if causal_state['weather'] in ['rain', 'snow']:
            target_area = self._select_area_center(self.weather_test_areas)
            return self._move_toward(agent_pos, target_area)
            
        elif causal_state['crowd_density'] >= 3:
            target_area = self._select_area_center(self.crowd_test_areas)
            return self._move_toward(agent_pos, target_area)
            
        elif causal_state['event'] in self.event_test_areas:
            event_area = self.event_test_areas[causal_state['event']]
            target = np.array([(event_area[0] + event_area[2])//2, (event_area[1] + event_area[3])//2])
            return self._move_toward(agent_pos, target)
            
        else:
            # Default: explore toward goal with some randomness
            if np.random.random() < 0.7:
                return self._move_toward(agent_pos, goal_pos)
            else:
                return np.random.randint(0, 5)
    
    def _select_area_center(self, areas):
        """Select center of a random area"""
        area = areas[np.random.randint(len(areas))]
        center_x = (area[0] + area[2]) // 2
        center_y = (area[1] + area[3]) // 2
        return np.array([center_x, center_y])
    
    def _move_toward(self, current_pos, target_pos, randomness=0.2):
        """Move toward target with some randomness"""
        if np.random.random() < randomness:
            return np.random.randint(0, 5)
        
        diff = target_pos - current_pos
        
        # Choose dimension with larger difference
        if abs(diff[0]) > abs(diff[1]):
            # Move horizontally
            return 2 if diff[0] > 0 else 3  # East or West
        elif abs(diff[1]) > 0:
            # Move vertically  
            return 1 if diff[1] > 0 else 0  # South or North
        else:
            return 4  # Stay


class EdgeCaseExplorer:
    """Creates challenging edge case scenarios"""
    
    def __init__(self, buildings, grid_size):
        self.buildings = buildings
        self.grid_size = grid_size
        
        # Challenging navigation scenarios
        self.edge_scenarios = {
            'stadium_gameday': (5, 5, 15, 15),    # Navigate through gameday crowds
            'night_long_path': (0, 0, 60, 60),    # Long distance at night
            'rain_uncovered': (20, 20, 40, 40),   # Cross open areas in rain
            'construction_detour': (25, 30, 35, 35), # Navigate around barriers
        }
    
    def get_edge_case_action(self, agent_pos, causal_state, goal_pos):
        """Generate action for edge case exploration"""
        
        # Identify current edge case scenario
        scenario = self._identify_scenario(causal_state)
        
        if scenario == 'stadium_gameday':
            # Deliberately navigate through crowded stadium area
            stadium_center = np.array([10, 10])
            return self._move_toward_challenge(agent_pos, stadium_center)
            
        elif scenario == 'night_navigation':
            # Take longer paths during night (more challenging)
            detour_point = np.array([agent_pos[0] + 10, agent_pos[1] + 10])
            detour_point = np.clip(detour_point, 0, self.grid_size-1)
            return self._move_toward_challenge(agent_pos, detour_point)
            
        elif scenario == 'rain_crossing':
            # Cross open areas in rain (no shelter)
            open_center = np.array([32, 32])
            return self._move_toward_challenge(agent_pos, open_center)
            
        else:
            # Default challenging behavior - take suboptimal paths
            return self._suboptimal_navigation(agent_pos, goal_pos)
    
    def _identify_scenario(self, causal_state):
        """Identify current edge case scenario"""
        if causal_state['event'] == 'gameday':
            return 'stadium_gameday'
        elif causal_state['time_hour'] < 6 or causal_state['time_hour'] > 20:
            return 'night_navigation'
        elif causal_state['weather'] == 'rain':
            return 'rain_crossing'
        else:
            return 'general_challenge'
    
    def _move_toward_challenge(self, current_pos, target_pos):
        """Move toward target but with some inefficiency"""
        diff = target_pos - current_pos
        
        # Sometimes take orthogonal moves (inefficient)
        if np.random.random() < 0.3:
            # Take perpendicular direction
            if abs(diff[0]) > abs(diff[1]):
                return 1 if diff[1] > 0 else 0  # Move in Y direction instead
            else:
                return 2 if diff[0] > 0 else 3  # Move in X direction instead
        
        # Normal movement toward target
        if abs(diff[0]) > abs(diff[1]):
            return 2 if diff[0] > 0 else 3
        elif abs(diff[1]) > 0:
            return 1 if diff[1] > 0 else 0
        else:
            return 4
    
    def _suboptimal_navigation(self, agent_pos, goal_pos):
        """Navigate toward goal but with deliberate inefficiencies"""
        # 30% chance of taking a random detour
        if np.random.random() < 0.3:
            return np.random.randint(0, 4)  # Exclude stay action
        
        # Otherwise move toward goal
        diff = goal_pos - agent_pos
        if abs(diff[0]) > abs(diff[1]):
            return 2 if diff[0] > 0 else 3
        elif abs(diff[1]) > 0:
            return 1 if diff[1] > 0 else 0
        else:
            return 4


# Test the exploration strategies
if __name__ == "__main__":
    explorer = CausalAwareExploration()
    
    # Test causal curriculum
    curriculum = explorer.generate_causal_curriculum()
    print(f"Generated {len(curriculum)} curriculum scenarios:")
    for i, (name, state, difficulty) in enumerate(curriculum[:10]):
        print(f"{i+1:2d}. {name:20s} ({difficulty:6s}): {state}")
    
    # Test action selection
    agent_pos = np.array([10, 10])
    goal_pos = np.array([45, 15])  # Gym
    causal_state = {
        'time_hour': 8,
        'day_week': 1,
        'weather': 'rain', 
        'event': 'normal',
        'crowd_density': 4
    }
    
    print(f"\nTest action selection:")
    print(f"Agent: {agent_pos}, Goal: {goal_pos}")
    print(f"Causal state: {causal_state}")
    
    for mode in explorer.exploration_modes:
        action = explorer.select_action(agent_pos, goal_pos, causal_state, 25, mode=mode)
        action_names = ['North', 'South', 'East', 'West', 'Stay']
        print(f"{mode:15s}: {action} ({action_names[action]})")