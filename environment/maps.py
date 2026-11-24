"""
Warehouse map layouts and scenario generators.
Defines grids where 0 = Free space, 1 = Obstacle/Rack.
"""

from typing import Dict, Tuple, List
import numpy as np

# Map definitions: 0 = Free, 1 = Rack/Obstacle

# 1. Simple Map: 10x10 open warehouse layout with a few storage stacks
SIMPLE_MAP = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# 2. Moderate Map: 12x12 realistic warehouse layout with vertical storage racks and cross aisles
MODERATE_MAP = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Middle cross-aisle
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# 3. Complex Map: 12x12 layout with asymmetric racks, bottlenecks, and dead ends
COMPLEX_MAP = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# 4. Generalization Map: 12x12 warehouse with rearranged shelving units and alternative goal
GENERALIZATION_MAP = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

DEFAULT_POSITIONS: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    "simple_map": ((0, 0), (9, 9)),
    "moderate_map": ((0, 0), (11, 11)),
    "complex_map": ((0, 0), (10, 10)),
    "generalization_map": ((0, 0), (11, 11)),
}

MAP_REGISTRY: Dict[str, np.ndarray] = {
    "simple_map": np.array(SIMPLE_MAP, dtype=np.int32),
    "moderate_map": np.array(MODERATE_MAP, dtype=np.int32),
    "complex_map": np.array(COMPLEX_MAP, dtype=np.int32),
    "generalization_map": np.array(GENERALIZATION_MAP, dtype=np.int32),
}


def get_map(map_name: str) -> np.ndarray:
    """Retrieve copy of the grid for a named map."""
    if map_name not in MAP_REGISTRY:
        raise ValueError(
            f"Unknown map '{map_name}'. Available maps: {list_maps()}"
        )
    return MAP_REGISTRY[map_name].copy()


def get_default_positions(
    map_name: str,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Retrieve default (start_pos, goal_pos) for a map."""
    if map_name not in DEFAULT_POSITIONS:
        raise ValueError(f"Unknown map '{map_name}'.")
    return DEFAULT_POSITIONS[map_name]


def list_maps() -> List[str]:
    """List all available map identifiers."""
    return list(MAP_REGISTRY.keys())
