"""
Warehouse navigation environment following Reinforcement Learning standards.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np

from environment.maps import get_map, get_default_positions


class WarehouseEnv:
    """
    2D Grid Warehouse Environment.
    Grid Cell Encoding:
        0: Free cell
        1: Obstacle / Storage rack
        2: Start position
        3: Goal position
        4: Robot position (during rendering)
    Actions:
        0: UP    (-1,  0)
        1: DOWN  (+1,  0)
        2: LEFT  ( 0, -1)
        3: RIGHT ( 0, +1)
    """

    CELL_FREE = 0
    CELL_OBSTACLE = 1
    CELL_START = 2
    CELL_GOAL = 3
    CELL_ROBOT = 4

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    ACTION_DELTAS = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }

    ACTION_NAMES = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
    }

    def __init__(
        self,
        map_name: str = "moderate_map",
        max_steps: int = 120,
        reward_goal: float = 60.0,
        reward_step: float = -0.2,
        reward_collision: float = -5.0,
        reward_timeout: float = -10.0,
        reward_distance_shaping: bool = True,
        shaping_weight: float = 0.5,
    ):
        self.map_name = map_name
        self.obstacles = get_map(map_name)
        self.height, self.width = self.obstacles.shape
        self.max_steps = max_steps

        # Reward parameters
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_collision = reward_collision
        self.reward_timeout = reward_timeout
        self.reward_distance_shaping = reward_distance_shaping
        self.shaping_weight = shaping_weight

        # Default start and goal
        default_start, default_goal = get_default_positions(map_name)
        self.default_start = default_start
        self.default_goal = default_goal

        self.start_pos = default_start
        self.goal_pos = default_goal
        self.robot_pos = default_start

        # State tracking
        self.step_count = 0
        self.path_history: List[Tuple[int, int]] = []
        self.collision_count = 0
        self.prev_manhattan_dist = 0.0

        # Maximum possible distance in grid
        self.max_dist = (self.height - 1) + (self.width - 1)

    def reset(
        self,
        seed: Optional[int] = None,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.start_pos = start_pos if start_pos is not None else self.default_start
        self.goal_pos = goal_pos if goal_pos is not None else self.default_goal

        # Validate positions
        if self.obstacles[self.start_pos[0], self.start_pos[1]] == self.CELL_OBSTACLE:
            raise ValueError(f"Start position {self.start_pos} is an obstacle!")
        if self.obstacles[self.goal_pos[0], self.goal_pos[1]] == self.CELL_OBSTACLE:
            raise ValueError(f"Goal position {self.goal_pos} is an obstacle!")

        self.robot_pos = self.start_pos
        self.step_count = 0
        self.collision_count = 0
        self.path_history = [self.robot_pos]
        self.prev_manhattan_dist = self._manhattan_distance(self.robot_pos, self.goal_pos)

        obs = self._get_observation()
        info = {
            "start_pos": self.start_pos,
            "goal_pos": self.goal_pos,
            "robot_pos": self.robot_pos,
            "step_count": self.step_count,
            "collision_count": self.collision_count,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one action step.
        Returns:
            observation (np.ndarray): 19-dim normalized observation vector
            reward (float): scalar reward
            done (bool): whether episode is terminated
            info (dict): auxiliary metrics
        """
        if action not in self.ACTION_DELTAS:
            raise ValueError(f"Invalid action {action}. Must be in {list(self.ACTION_DELTAS.keys())}")

        self.step_count += 1
        delta_r, delta_c = self.ACTION_DELTAS[action]
        next_r = self.robot_pos[0] + delta_r
        next_c = self.robot_pos[1] + delta_c

        collision = False
        reached_goal = False
        timeout = False
        done = False
        reward = self.reward_step

        # Boundary and obstacle check
        is_out_of_bounds = not (0 <= next_r < self.height and 0 <= next_c < self.width)
        is_obstacle = False if is_out_of_bounds else (self.obstacles[next_r, next_c] == self.CELL_OBSTACLE)

        if is_out_of_bounds or is_obstacle:
            # Collision occurred: robot stays in place
            collision = True
            self.collision_count += 1
            reward += self.reward_collision
        else:
            # Valid movement
            self.robot_pos = (next_r, next_c)

        self.path_history.append(self.robot_pos)

        # Distance shaping calculation
        curr_dist = self._manhattan_distance(self.robot_pos, self.goal_pos)
        if self.reward_distance_shaping and not collision:
            # Positive reward for getting closer, penalty for getting farther
            dist_diff = (self.prev_manhattan_dist - curr_dist)
            reward += self.shaping_weight * dist_diff
        self.prev_manhattan_dist = curr_dist

        # Check goal reached
        if self.robot_pos == self.goal_pos:
            reached_goal = True
            done = True
            reward += self.reward_goal
        elif self.step_count >= self.max_steps:
            timeout = True
            done = True
            reward += self.reward_timeout

        obs = self._get_observation()
        info = {
            "robot_pos": self.robot_pos,
            "goal_pos": self.goal_pos,
            "collision": collision,
            "collision_count": self.collision_count,
            "reached_goal": reached_goal,
            "timeout": timeout,
            "step_count": self.step_count,
            "path_length": len(self.path_history) - 1,
        }
        return obs, reward, done, info

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return float(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    def _get_observation(self) -> np.ndarray:
        """
        Construct 19-dimensional normalized observation vector:
        - 2: Normalized robot position (r/(H-1), c/(W-1))
        - 2: Normalized goal position (gr/(H-1), gc/(W-1))
        - 2: Relative vector to goal ((gr-r)/(H-1), (gc-c)/(W-1))
        - 1: Normalized Manhattan distance to goal
        - 8: 3x3 surrounding cells (1 if blocked/wall, 0 if free, excluding center)
        - 4: Ray-cast distance to obstacle/wall in 4 directions [UP, DOWN, LEFT, RIGHT]
        Total: 2 + 2 + 2 + 1 + 8 + 4 = 19 dimensions
        """
        r, c = self.robot_pos
        gr, gc = self.goal_pos
        H, W = self.height, self.width

        # 1. Normalized robot pos
        norm_r = r / max(H - 1, 1)
        norm_c = c / max(W - 1, 1)

        # 2. Normalized goal pos
        norm_gr = gr / max(H - 1, 1)
        norm_gc = gc / max(W - 1, 1)

        # 3. Relative displacement
        rel_r = (gr - r) / max(H - 1, 1)
        rel_c = (gc - c) / max(W - 1, 1)

        # 4. Normalized Manhattan distance
        norm_dist = (abs(gr - r) + abs(gc - c)) / max(self.max_dist, 1)

        # 5. Local 3x3 surrounding (8 neighbors: N, NE, E, SE, S, SW, W, NW)
        surroundings = []
        neighbor_offsets = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        for dr, dc in neighbor_offsets:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                surroundings.append(1.0)  # Wall is blocked
            elif self.obstacles[nr, nc] == self.CELL_OBSTACLE:
                surroundings.append(1.0)  # Obstacle is blocked
            else:
                surroundings.append(0.0)  # Free

        # 6. Ray distances in 4 cardinal directions [UP, DOWN, LEFT, RIGHT]
        ray_distances = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            dist = 0
            curr_r, curr_c = r + dr, c + dc
            while 0 <= curr_r < H and 0 <= curr_c < W and self.obstacles[curr_r, curr_c] != self.CELL_OBSTACLE:
                dist += 1
                curr_r += dr
                curr_c += dc
            # Normalize by max dimension
            max_dim = max(H, W)
            ray_distances.append(dist / max(max_dim, 1))

        obs = np.array(
            [norm_r, norm_c, norm_gr, norm_gc, rel_r, rel_c, norm_dist]
            + surroundings
            + ray_distances,
            dtype=np.float32,
        )
        return obs

    def render_grid_array(self) -> np.ndarray:
        """
        Return 2D grid array representing current state:
        0: Free, 1: Obstacle, 2: Start, 3: Goal, 4: Robot
        """
        grid = self.obstacles.copy()
        grid[self.start_pos[0], self.start_pos[1]] = self.CELL_START
        grid[self.goal_pos[0], self.goal_pos[1]] = self.CELL_GOAL
        grid[self.robot_pos[0], self.robot_pos[1]] = self.CELL_ROBOT
        return grid
