"""
Main entry point for Autonomous Warehouse Navigation with Deep Q-Learning.
Supports CLI arguments:
    --mode {train, evaluate, visualize, all}
    --map {moderate_map, simple_map, complex_map, generalization_map}
    --episodes <int>
    --seed <int>
"""

import argparse
import sys
from pathlib import Path

from config import default_config, Config, EnvConfig, DQNConfig, TrainingConfig, EvalConfig
from training.train import train_dqn
from training.evaluate import evaluate_all, evaluate_agent_on_map
from algorithms.pathfinding import bfs_search, astar_search
from environment.maps import get_default_positions
from environment.warehouse_env import WarehouseEnv
from agents.dqn_agent import DQNAgent
from visualization.render_map import render_warehouse_map
from visualization.plot_training import plot_all_training_metrics
from visualization.plot_evaluation import plot_evaluation_comparison
from visualization.animate_path import plot_trajectories, generate_navigation_gif


def run_visualizations(cfg: Config = default_config) -> None:
    """Generate all required static and animated visualizations."""
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Render warehouse layout map
    render_warehouse_map("moderate_map")

    # 2. Render training metric curves
    if cfg.training.history_csv_path.exists():
        plot_all_training_metrics(cfg.training.history_csv_path)
    else:
        print("Warning: training_history.csv not found, skipping training curves plot.")

    # 3. Render benchmark evaluation comparisons
    if cfg.evaluation.eval_results_csv_path.exists():
        plot_evaluation_comparison(cfg.evaluation.eval_results_csv_path)
    else:
        print("Warning: evaluation_results.csv not found, skipping evaluation plots.")

    # 4. Generate trajectory overlay and animated GIF
    model_path = cfg.training.best_model_path if cfg.training.best_model_path.exists() else cfg.training.final_model_path
    if model_path.exists():
        print(f"Loading model from {model_path} for trajectory animation...")
        agent = DQNAgent(cfg.dqn, seed=cfg.evaluation.seed)
        agent.load(model_path)

        # Run DQN on moderate_map
        map_name = "moderate_map"
        env = WarehouseEnv(map_name=map_name)
        obs, _ = env.reset(seed=cfg.evaluation.seed)
        done = False
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, _, done, _ = env.step(action)

        dqn_path = list(env.path_history)
        start_pos, goal_pos = get_default_positions(map_name)
        bfs_res = bfs_search(env.obstacles, start_pos, goal_pos)
        astar_res = astar_search(env.obstacles, start_pos, goal_pos)

        # Save trajectory comparison image
        plot_trajectories(
            map_name=map_name,
            dqn_path=dqn_path,
            bfs_path=bfs_res.path,
            astar_path=astar_res.path,
        )

        # Generate animated navigation GIF
        generate_navigation_gif(
            map_name=map_name,
            path_history=dqn_path,
        )
    else:
        print("Warning: No model checkpoint found, skipping trajectory visual.")

    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Warehouse Navigation with Deep Q-Learning"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "evaluate", "visualize", "all"],
        help="Execution mode: train, evaluate, visualize, or all (default: all)",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="moderate_map",
        help="Warehouse map to train or evaluate on (default: moderate_map)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=600,
        help="Number of training episodes (default: 600)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    cfg = default_config
    cfg.env.map_name = args.map
    cfg.training.num_episodes = args.episodes
    cfg.training.seed = args.seed

    if args.mode in ["train", "all"]:
        train_dqn(cfg)

    if args.mode in ["evaluate", "all"]:
        evaluate_all(cfg=cfg)

    if args.mode in ["visualize", "all"]:
        run_visualizations(cfg)


if __name__ == "__main__":
    main()
