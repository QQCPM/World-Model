"""
Continuous Campus Environment for World Models
Implements true continuous control with physics-based dynamics using PyMunk.

This environment addresses the fundamental paradigm error identified in CRITICAL_FLAWS_ANALYSIS.md:
- Continuous 2D action space (velocity control)
- Physics-based dynamics with momentum and friction
- Meaningful visual complexity for VAE learning
- Causal factors affect physics parameters, not just visuals

Design Principles:
1. True continuous control problem (World Models appropriate)
2. Physics simulation creates rich temporal dependencies
3. Causal factors modify movement dynamics and constraints
4. Visual complexity correlates with navigation complexity
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pymunk
import pymunk.pygame_util
import pygame
import math
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from enum import Enum

# Import temporal delay system (research-validated)
try:
    from .temporal_integration import TemporalCausalIntegrator, TemporalIntegrationConfig
except ImportError:
    # For direct execution or testing
    from temporal_integration import TemporalCausalIntegrator, TemporalIntegrationConfig


class WeatherType(Enum):
    """Weather conditions affecting movement physics"""
    SUNNY = "sunny"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"


class EventType(Enum):
    """Campus events affecting navigation"""
    NORMAL = "normal"
    GAMEDAY = "gameday"
    EXAM = "exam"
    BREAK = "break"
    CONSTRUCTION = "construction"


@dataclass
class CausalState:
    """Causal state affecting environment dynamics"""
    time_hour: int          # 0-23 (affects visibility and crowd patterns)
    day_week: int          # 0-6 (Monday=0, affects crowd patterns)
    weather: WeatherType   # Affects movement physics
    event: EventType       # Affects crowd density and obstacles
    crowd_density: float   # 0.0-1.0 (affects local movement costs)

    def to_vector(self) -> np.ndarray:
        """Convert causal state to numeric vector for model input"""
        # Normalize all factors to [0, 1] range
        weather_norm = list(WeatherType).index(self.weather) / (len(WeatherType) - 1)
        event_norm = list(EventType).index(self.event) / (len(EventType) - 1)
        time_norm = self.time_hour / 23.0
        day_norm = self.day_week / 6.0
        crowd_norm = self.crowd_density  # Already 0-1

        return np.array([weather_norm, event_norm, crowd_norm, time_norm, day_norm], dtype=np.float32)


@dataclass
class Building:
    """Building obstacle with physics properties"""
    name: str
    center: Tuple[float, float]
    width: float
    height: float
    friction: float = 0.8


class ContinuousCampusEnv(gym.Env):
    """
    Continuous 2D campus navigation environment with physics-based dynamics.

    Action Space: Box(2,) - [velocity_x, velocity_y] in range [-1.0, 1.0]
    Observation Space: Box(12,) - [agent_x, agent_y, goal_x, goal_y, vel_x, vel_y,
                                   goal_dist, obstacle_dist, weather, crowd, time, event]

    Physics Engine: PyMunk for realistic collision detection and momentum
    Causal Integration: Environmental factors modify physics parameters
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(self, render_mode: Optional[str] = None, enable_temporal_delays: bool = False, **kwargs):
        super().__init__()

        # Environment parameters
        self.world_width = 100.0
        self.world_height = 100.0
        self.max_speed = 5.0          # Maximum velocity magnitude
        self.agent_radius = 1.0       # Agent collision radius
        self.goal_radius = 2.0        # Goal achievement radius
        self.max_episode_steps = 1000  # Longer episodes for continuous control

        # Action space: 2D velocity control
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Observation space: 12D continuous state
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Campus buildings (continuous coordinates)
        self.buildings = [
            Building("library", (20.0, 20.0), 10.0, 10.0),
            Building("gym", (70.0, 15.0), 12.0, 8.0),
            Building("cafeteria", (15.0, 70.0), 8.0, 12.0),
            Building("academic", (80.0, 80.0), 15.0, 15.0),
            Building("stadium", (10.0, 10.0), 8.0, 8.0),
            Building("dorms", (90.0, 50.0), 8.0, 10.0)
        ]

        # Rendering setup
        self.render_mode = render_mode
        self.screen_width = 800
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.draw_options = None

        # Physics space (will be initialized in reset())
        self.space: Optional[pymunk.Space] = None
        self.agent_body: Optional[pymunk.Body] = None
        self.agent_shape: Optional[pymunk.Shape] = None
        self.building_bodies: List[pymunk.Body] = []

        # Environment state
        self.agent_position = np.array([0.0, 0.0])
        self.goal_position = np.array([0.0, 0.0])
        self.agent_velocity = np.array([0.0, 0.0])
        self.causal_state: Optional[CausalState] = None
        self.episode_step = 0

        # Temporal delay system (research-validated: 2-timestep weather delays)
        self.temporal_delays_enabled = enable_temporal_delays
        if self.temporal_delays_enabled:
            config = TemporalIntegrationConfig(
                enable_delays=True,
                enable_logging=False,
                validation_mode=False
            )
            self.temporal_integrator = TemporalCausalIntegrator(config)
        else:
            self.temporal_integrator = None

        # Initialize physics if we have a render mode
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Continuous Campus Environment")
            self.clock = pygame.time.Clock()

    def _create_physics_space(self) -> None:
        """Initialize PyMunk physics space with buildings and agent"""
        # Create physics space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # No gravity for 2D navigation

        # Create static building bodies
        self.building_bodies = []
        for building in self.buildings:
            # Create static body for building
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = building.center

            # Create box shape for building
            shape = pymunk.Poly.create_box(body, (building.width, building.height))
            shape.friction = building.friction

            self.space.add(body, shape)
            self.building_bodies.append(body)

        # Create agent physics body
        self._create_agent_body()

        # Set up collision handling
        self._setup_collision_handlers()

    def _create_agent_body(self) -> None:
        """Create physics body for the agent"""
        # Create dynamic body for agent
        mass = 1.0
        moment = pymunk.moment_for_circle(mass, 0, self.agent_radius)
        self.agent_body = pymunk.Body(mass, moment)
        self.agent_body.position = tuple(self.agent_position)

        # Create circular shape for agent
        self.agent_shape = pymunk.Circle(self.agent_body, self.agent_radius)
        self.agent_shape.friction = 0.7

        self.space.add(self.agent_body, self.agent_shape)

    def _setup_collision_handlers(self) -> None:
        """Set up collision detection between agent and buildings"""
        # For now, we'll handle collisions in the step function
        # This simplifies the PyMunk integration
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Initialize physics space
        self._create_physics_space()

        # Sample random spawn position (away from buildings)
        self.agent_position = self._sample_valid_position()
        self.agent_body.position = tuple(self.agent_position)

        # Sample random goal position (away from buildings and agent)
        self.goal_position = self._sample_goal_position()

        # Reset velocity
        self.agent_velocity = np.array([0.0, 0.0])
        self.agent_body.velocity = (0.0, 0.0)

        # Sample random causal state
        if options and 'causal_state' in options:
            self.causal_state = options['causal_state']
        else:
            self.causal_state = self._sample_causal_state()

        # Reset episode step counter
        self.episode_step = 0

        # Reset temporal integrator for new episode
        if self.temporal_integrator is not None:
            self.temporal_integrator.reset()

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step"""
        # Validate action
        action = np.clip(action, -1.0, 1.0)

        # Apply causal effects to movement dynamics (with optional temporal delays)
        if self.temporal_integrator is not None:
            # Use research-validated temporal delay system
            modified_action, temporal_info = self.temporal_integrator.apply_temporal_effects(action, self.causal_state)
        else:
            # Use original immediate effects (backward compatibility)
            modified_action = self._apply_causal_effects(action)
            temporal_info = None

        # Convert action to world velocity
        target_velocity = modified_action * self.max_speed

        # Apply velocity to agent body (with physics constraints)
        self.agent_body.velocity = tuple(target_velocity)

        # Step physics simulation
        dt = 1.0 / 10.0  # 10 FPS physics for larger steps
        self.space.step(dt)

        # Update agent state from physics
        self.agent_position = np.array(self.agent_body.position)
        self.agent_velocity = np.array(self.agent_body.velocity) / self.max_speed  # Normalize

        # Ensure agent stays within world bounds
        self._enforce_world_bounds()

        # Compute reward
        reward = self._compute_reward(action)

        # Check termination conditions
        goal_distance = np.linalg.norm(self.agent_position - self.goal_position)
        terminated = bool(goal_distance < self.goal_radius)

        # Check truncation (max steps)
        self.episode_step += 1
        truncated = bool(self.episode_step >= self.max_episode_steps)

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        info['goal_distance'] = goal_distance

        # Add temporal delay information if enabled
        if temporal_info is not None:
            info['temporal_delays'] = temporal_info

        return observation, reward, terminated, truncated, info

    def _sample_valid_position(self) -> np.ndarray:
        """Sample a position that doesn't collide with buildings"""
        max_attempts = 100
        for _ in range(max_attempts):
            pos = np.random.uniform([5.0, 5.0], [95.0, 95.0])
            if self._is_position_valid(pos):
                return pos

        # Fallback: return a safe corner position
        return np.array([5.0, 5.0])

    def _sample_goal_position(self) -> np.ndarray:
        """Sample a goal position away from buildings and agent"""
        max_attempts = 100
        for _ in range(max_attempts):
            pos = np.random.uniform([5.0, 5.0], [95.0, 95.0])
            if (self._is_position_valid(pos) and
                np.linalg.norm(pos - self.agent_position) > 10.0):
                return pos

        # Fallback: return opposite corner from agent
        if self.agent_position[0] < 50:
            return np.array([95.0, 95.0])
        else:
            return np.array([5.0, 5.0])

    def _is_position_valid(self, position: np.ndarray) -> bool:
        """Check if position collides with any building"""
        for building in self.buildings:
            dx = abs(position[0] - building.center[0])
            dy = abs(position[1] - building.center[1])

            if (dx < building.width/2 + self.agent_radius and
                dy < building.height/2 + self.agent_radius):
                return False
        return True

    def _sample_causal_state(self) -> CausalState:
        """Sample random causal state"""
        return CausalState(
            time_hour=np.random.randint(0, 24),
            day_week=np.random.randint(0, 7),
            weather=np.random.choice(list(WeatherType)),
            event=np.random.choice(list(EventType)),
            crowd_density=np.random.uniform(0.0, 1.0)
        )

    def _apply_causal_effects(self, action: np.ndarray) -> np.ndarray:
        """Apply causal state effects to movement dynamics"""
        modified_action = action.copy()

        # Weather effects on movement
        if self.causal_state.weather == WeatherType.RAIN:
            modified_action *= 0.8  # Reduced speed in rain
        elif self.causal_state.weather == WeatherType.SNOW:
            modified_action *= 0.6  # Even slower in snow
            # Add random noise for slipping
            noise = np.random.normal(0, 0.1, 2)
            modified_action += noise
        elif self.causal_state.weather == WeatherType.FOG:
            modified_action *= 0.9  # Slightly reduced speed

        # Time effects (night movement)
        if self.causal_state.time_hour < 6 or self.causal_state.time_hour > 22:
            modified_action *= 0.7  # Slower at night

        # Crowd effects (local slowdown)
        crowd_effect = self._compute_crowd_effect()
        modified_action *= (1.0 - crowd_effect * 0.5)

        # Event effects
        if self.causal_state.event == EventType.CONSTRUCTION:
            # Construction creates more obstacles and slower movement
            modified_action *= 0.8

        return np.clip(modified_action, -1.0, 1.0)

    def _compute_crowd_effect(self) -> float:
        """Compute crowd density effect at agent's current position"""
        # Simplified crowd effect: higher near buildings during peak hours
        is_peak_hour = self.causal_state.time_hour in [8, 12, 17]
        base_crowd = self.causal_state.crowd_density

        if is_peak_hour:
            # Check proximity to buildings
            min_building_distance = float('inf')
            for building in self.buildings:
                dist = np.linalg.norm(self.agent_position - np.array(building.center))
                min_building_distance = min(min_building_distance, dist)

            # Crowd effect increases near buildings
            if min_building_distance < 15.0:
                proximity_factor = (15.0 - min_building_distance) / 15.0
                return base_crowd * (1.0 + proximity_factor)

        return base_crowd * 0.5  # Lower crowd during non-peak hours

    def _enforce_world_bounds(self) -> None:
        """Keep agent within world boundaries"""
        pos = np.array(self.agent_body.position)

        # Clamp position to world bounds
        pos[0] = np.clip(pos[0], self.agent_radius, self.world_width - self.agent_radius)
        pos[1] = np.clip(pos[1], self.agent_radius, self.world_height - self.agent_radius)

        # Update physics body position
        self.agent_body.position = tuple(pos)
        self.agent_position = pos

        # Dampen velocity if hitting boundary
        vel = np.array(self.agent_body.velocity)
        if pos[0] <= self.agent_radius or pos[0] >= self.world_width - self.agent_radius:
            vel[0] *= -0.5  # Bounce with damping
        if pos[1] <= self.agent_radius or pos[1] >= self.world_height - self.agent_radius:
            vel[1] *= -0.5  # Bounce with damping

        self.agent_body.velocity = tuple(vel)

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward for current state and action"""
        # Distance-based progress reward
        goal_distance = np.linalg.norm(self.agent_position - self.goal_position)
        max_distance = np.sqrt(self.world_width**2 + self.world_height**2)
        progress_reward = (max_distance - goal_distance) / max_distance

        # Velocity penalty to encourage efficient movement
        velocity_magnitude = np.linalg.norm(self.agent_velocity)
        velocity_penalty = -0.01 * velocity_magnitude

        # Goal achievement bonus
        if goal_distance < self.goal_radius:
            goal_bonus = 100.0
        else:
            goal_bonus = 0.0

        # Time penalty (encourage faster completion)
        time_penalty = -0.1

        # Causal-based penalties
        causal_penalty = self._compute_causal_penalty()

        total_reward = progress_reward + velocity_penalty + goal_bonus + time_penalty + causal_penalty

        return total_reward

    def _compute_causal_penalty(self) -> float:
        """Compute penalty based on causal factors"""
        penalty = 0.0

        # Weather penalties
        if self.causal_state.weather == WeatherType.RAIN:
            penalty -= 0.1
        elif self.causal_state.weather == WeatherType.SNOW:
            penalty -= 0.2
        elif self.causal_state.weather == WeatherType.FOG:
            penalty -= 0.05

        # Crowd penalties
        crowd_effect = self._compute_crowd_effect()
        penalty -= crowd_effect * 0.3

        # Night navigation penalty
        if self.causal_state.time_hour < 6 or self.causal_state.time_hour > 22:
            penalty -= 0.05

        return penalty

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        # Find distance to nearest obstacle
        min_obstacle_distance = self._compute_nearest_obstacle_distance()

        # Normalize causal factors to [0, 1] range
        weather_norm = list(WeatherType).index(self.causal_state.weather) / (len(WeatherType) - 1)
        crowd_norm = self.causal_state.crowd_density
        time_norm = self.causal_state.time_hour / 23.0
        event_norm = list(EventType).index(self.causal_state.event) / (len(EventType) - 1)

        # Goal distance
        goal_distance = np.linalg.norm(self.agent_position - self.goal_position)

        observation = np.array([
            self.agent_position[0],      # agent_x
            self.agent_position[1],      # agent_y
            self.goal_position[0],       # goal_x
            self.goal_position[1],       # goal_y
            self.agent_velocity[0],      # velocity_x (normalized)
            self.agent_velocity[1],      # velocity_y (normalized)
            goal_distance,               # distance to goal
            min_obstacle_distance,       # distance to nearest obstacle
            weather_norm,                # weather effect intensity
            crowd_norm,                  # crowd density
            time_norm,                   # time of day effect
            event_norm                   # event effect intensity
        ], dtype=np.float32)

        return observation

    def _compute_nearest_obstacle_distance(self) -> float:
        """Compute distance to nearest building obstacle"""
        min_distance = float('inf')

        for building in self.buildings:
            # Distance to building center
            center_dist = np.linalg.norm(self.agent_position - np.array(building.center))
            # Approximate distance to building edge
            edge_dist = max(0.0, center_dist - max(building.width, building.height) / 2)
            min_distance = min(min_distance, edge_dist)

        # Normalize by maximum possible distance
        max_distance = np.sqrt(self.world_width**2 + self.world_height**2)
        return min(min_distance / max_distance, 1.0)

    def _get_info(self) -> dict:
        """Get environment info dictionary"""
        return {
            'agent_position': self.agent_position.copy(),
            'goal_position': self.goal_position.copy(),
            'agent_velocity': self.agent_velocity.copy(),
            'causal_state': {
                'time_hour': self.causal_state.time_hour,
                'day_week': self.causal_state.day_week,
                'weather': self.causal_state.weather.value,
                'event': self.causal_state.event.value,
                'crowd_density': self.causal_state.crowd_density
            },
            'episode_step': self.episode_step
        }

    def get_temporal_validation_report(self) -> Optional[Dict[str, Any]]:
        """
        Get temporal delay validation report for research verification

        Returns:
            validation_report: Dict with temporal delay metrics, or None if delays disabled
        """
        if self.temporal_integrator is None:
            return None

        return self.temporal_integrator.get_validation_report()

    def enable_temporal_delays(self, enable_logging: bool = False):
        """
        Enable temporal delays during runtime

        Args:
            enable_logging: Enable detailed logging for validation
        """
        if self.temporal_integrator is None:
            config = TemporalIntegrationConfig(
                enable_delays=True,
                enable_logging=enable_logging,
                validation_mode=enable_logging
            )
            self.temporal_integrator = TemporalCausalIntegrator(config)
        else:
            self.temporal_integrator.enable_delays()
            if enable_logging:
                self.temporal_integrator.enable_validation_mode()

        self.temporal_delays_enabled = True

    def disable_temporal_delays(self):
        """Disable temporal delays during runtime for comparison"""
        if self.temporal_integrator is not None:
            self.temporal_integrator.disable_delays()
        self.temporal_delays_enabled = False

    def render(self):
        """Render the environment (implementation in next step)"""
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "human":
            return self._render_human()

    def _render_rgb_array(self) -> np.ndarray:
        """Render environment as RGB array (placeholder)"""
        # Will implement detailed rendering in next step
        return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

    def _render_human(self) -> None:
        """Render environment for human viewing (placeholder)"""
        # Will implement interactive rendering in next step
        pass

    def close(self):
        """Clean up environment resources"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None