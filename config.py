"""
Central configuration for Autonomous Warehouse Navigation with Deep Q-Learning.
Centralizes hyperparameters, environment settings, file paths, and seed controls.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
VIS_DIR = BASE_DIR / "visualizations"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

for d in [MODELS_DIR, RESULTS_DIR, VIS_DIR, SCREENSHOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class EnvConfig:
    map_name: str = "moderate_map"
    max_steps_per_episode: int = 120
    # Reward parameters
    reward_goal: float = 60.0
    reward_step: float = -0.2
    reward_collision: float = -5.0
    reward_timeout: float = -10.0
    reward_distance_shaping: bool = True
    shaping_weight: float = 0.5


@dataclass
class DQNConfig:
    state_dim: int = 19
    action_dim: int = 4
    hidden_dims: Tuple[int, ...] = (128, 128, 64)
    learning_rate: float = 5e-4
    gamma: float = 0.99
    replay_capacity: int = 40000
    batch_size: int = 64
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.994
    target_update_tau: float = 0.005  # Soft update parameter
    target_update_interval: int = 10  # Frequency for soft updates
    grad_norm_clip: float = 5.0


@dataclass
class TrainingConfig:
    num_episodes: int = 600
    seed: int = 42
    eval_frequency: int = 50
    eval_episodes: int = 15
    save_best_model: bool = True
    best_model_path: Path = MODELS_DIR / "best_model.pth"
    final_model_path: Path = MODELS_DIR / "final_model.pth"
    history_csv_path: Path = RESULTS_DIR / "training_history.csv"
    config_json_path: Path = RESULTS_DIR / "experiment_config.json"


@dataclass
class EvalConfig:
    seed: int = 100
    eval_episodes_per_map: int = 30
    eval_results_csv_path: Path = RESULTS_DIR / "evaluation_results.csv"
    maps_to_evaluate: List[str] = field(
        default_factory=lambda: [
            "simple_map",
            "moderate_map",
            "complex_map",
            "generalization_map",
        ]
    )


# Combined configuration object
@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


# Default singleton config
default_config = Config()
