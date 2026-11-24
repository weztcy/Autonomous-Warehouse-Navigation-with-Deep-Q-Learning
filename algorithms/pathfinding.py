"""
Classical pathfinding algorithms: BFS and A* Search.
Used as baselines and for environment path solvability validation.
"""

from typing import List, Tuple, Optional, Dict, Any, Set
import heapq
import time
from collections import deque
import numpy as np


class PathfindingResult:
    def __init__(
        self,
        success: bool,
        path: List[Tuple[int, int]],
        actions: List[int],
        nodes_explored: int,
        execution_time: float,
        algorithm: str,
    ):
        self.success = success
        self.path = path
        self.actions = actions
        self.length = len(path) - 1 if path else 0
        self.nodes_explored = nodes_explored
        self.execution_time = execution_time
        self.algorithm = algorithm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "success": self.success,
            "path_length": self.length,
            "nodes_explored": self.nodes_explored,
            "execution_time_ms": round(self.execution_time * 1000, 3),
        }


def _get_action_between(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Return action index (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT) from pos1 to pos2."""
    dr = pos2[0] - pos1[0]
    dc = pos2[1] - pos1[1]
    if dr == -1 and dc == 0:
        return 0  # UP
    if dr == 1 and dc == 0:
        return 1  # DOWN
    if dr == 0 and dc == -1:
        return 2  # LEFT
    if dr == 0 and dc == 1:
        return 3  # RIGHT
    raise ValueError(f"Invalid transition from {pos1} to {pos2}")


def _reconstruct_path(
    parent_map: Dict[Tuple[int, int], Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Reconstruct path coordinates and action sequence from parent map."""
    path = [goal]
    curr = goal
    while curr != start:
        curr = parent_map[curr]
        path.append(curr)
    path.reverse()

    actions = []
    for i in range(len(path) - 1):
        actions.append(_get_action_between(path[i], path[i + 1]))
    return path, actions


def bfs_search(
    obstacles: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> PathfindingResult:
    """
    Breadth-First Search (BFS) for grid pathfinding.
    Guarantees shortest path in unweighted uniform cost grid.
    """
    t0 = time.perf_counter()
    H, W = obstacles.shape

    if start == goal:
        t1 = time.perf_counter()
        return PathfindingResult(True, [start], [], 0, t1 - t0, "BFS")

    if obstacles[start[0], start[1]] == 1 or obstacles[goal[0], goal[1]] == 1:
        t1 = time.perf_counter()
        return PathfindingResult(False, [], [], 0, t1 - t0, "BFS")

    queue = deque([start])
    visited: Set[Tuple[int, int]] = {start}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    nodes_explored = 0

    # 4-directional moves: UP, DOWN, LEFT, RIGHT
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        curr = queue.popleft()
        nodes_explored += 1

        if curr == goal:
            path, actions = _reconstruct_path(parent, start, goal)
            t1 = time.perf_counter()
            return PathfindingResult(True, path, actions, nodes_explored, t1 - t0, "BFS")

        for dr, dc in directions:
            nr, nc = curr[0] + dr, curr[1] + dc
            neighbor = (nr, nc)

            if 0 <= nr < H and 0 <= nc < W:
                if obstacles[nr, nc] == 0 and neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)

    t1 = time.perf_counter()
    return PathfindingResult(False, [], [], nodes_explored, t1 - t0, "BFS")


def astar_search(
    obstacles: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> PathfindingResult:
    """
    A* Search using Manhattan distance heuristic.
    Guarantees shortest path and expands fewer nodes than BFS.
    """
    t0 = time.perf_counter()
    H, W = obstacles.shape

    if start == goal:
        t1 = time.perf_counter()
        return PathfindingResult(True, [start], [], 0, t1 - t0, "A*")

    if obstacles[start[0], start[1]] == 1 or obstacles[goal[0], goal[1]] == 1:
        t1 = time.perf_counter()
        return PathfindingResult(False, [], [], 0, t1 - t0, "A*")

    def heuristic(pos: Tuple[int, int]) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # Priority queue entries: (f_score, tie_breaker_counter, pos)
    counter = 0
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start), 0, counter, start))

    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], int] = {start: 0}
    visited: Set[Tuple[int, int]] = set()
    nodes_explored = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_heap:
        f, g, _, curr = heapq.heappop(open_heap)

        if curr in visited:
            continue
        visited.add(curr)
        nodes_explored += 1

        if curr == goal:
            path, actions = _reconstruct_path(parent, start, goal)
            t1 = time.perf_counter()
            return PathfindingResult(True, path, actions, nodes_explored, t1 - t0, "A*")

        for dr, dc in directions:
            nr, nc = curr[0] + dr, curr[1] + dc
            neighbor = (nr, nc)

            if 0 <= nr < H and 0 <= nc < W and obstacles[nr, nc] == 0:
                tentative_g = g + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = curr
                    h = heuristic(neighbor)
                    f_new = tentative_g + h
                    counter += 1
                    heapq.heappush(open_heap, (f_new, tentative_g, counter, neighbor))

    t1 = time.perf_counter()
    return PathfindingResult(False, [], [], nodes_explored, t1 - t0, "A*")


def validate_map_solvability(
    obstacles: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> bool:
    """Check whether a valid path exists from start to goal."""
    res = bfs_search(obstacles, start, goal)
    return res.success
