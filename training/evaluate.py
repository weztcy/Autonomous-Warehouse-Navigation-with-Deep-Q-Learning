"""
Evaluation and benchmark suite.
Compares trained DQN agent against classical BFS and A* baselines across all map scenarios.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import numpy as np
import pandas as pd
import torch

from config import Config, default_config
from environment.warehouse_env import WarehouseEnv
from environment.maps import get_default_positions
from agents.dqn_agent import DQNAgent
from algorithms.pathfinding import bfs_search, astar_search


def evaluate_agent_on_map(
    agent: DQNAgent,
    map_name: str,
    num_episodes: int = 30,
    seed: int = 100,
    max_steps: int = 120,
) -> Dict[str, Any]:
    """Run deterministic evaluation episodes for DQN on a specific map."""
    env = WarehouseEnv(map_name=map_name, max_steps=max_steps)

    successes = []
    collisions_list = []
    path_lengths = []
    steps_list = []
    rewards_list = []
    latencies_ms = []
    sample_trajectory = []

    np.random.seed(seed)

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        step_times = []

        while not done:
            t0 = time.perf_counter()
            action = agent.select_action(obs, evaluate=True)
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000.0)

            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            ep_reward += reward

        successes.append(1.0 if info["reached_goal"] else 0.0)
        collisions_list.append(info["collision_count"])
        path_lengths.append(info["path_length"])
        steps_list.append(info["step_count"])
        rewards_list.append(ep_reward)
        latencies_ms.append(float(np.mean(step_times)))

        if ep == 0:
            sample_trajectory = list(env.path_history)

    total_steps = sum(steps_list)
    total_collisions = sum(collisions_list)
    collision_step_ratio = total_collisions / max(total_steps, 1)

    return {
        "map_name": map_name,
        "method": "DQN",
        "num_episodes": num_episodes,
        "success_rate": round(float(np.mean(successes)), 4),
        "mean_reward": round(float(np.mean(rewards_list)), 2),
        "std_reward": round(float(np.std(rewards_list)), 2),
        "mean_steps": round(float(np.mean(steps_list)), 2),
        "mean_path_length": round(float(np.mean(path_lengths)), 2),
        "mean_collisions": round(float(np.mean(collisions_list)), 2),
        "collision_step_ratio": round(float(collision_step_ratio), 4),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 4),
        "sample_trajectory": sample_trajectory,
    }


def evaluate_all(
    model_path: Optional[Path] = None,
    cfg: Config = default_config,
) -> pd.DataFrame:
    """
    Run full benchmark across all map scenarios comparing DQN, BFS, and A*.
    Saves results to results/evaluation_results.csv.
    """
    print("=" * 70)
    print("STARTING FULL BENCHMARK EVALUATION (DQN vs BFS vs A*)")
    print("=" * 70)

    # Determine model path
    if model_path is None:
        if cfg.training.best_model_path.exists():
            model_path = cfg.training.best_model_path
        else:
            model_path = cfg.training.final_model_path

    agent = DQNAgent(cfg.dqn, seed=cfg.evaluation.seed)
    agent.load(model_path)
    print(f"Loaded trained DQN model from: {model_path}")

    all_results: List[Dict[str, Any]] = []

    for map_name in cfg.evaluation.maps_to_evaluate:
        print(f"\n--- Evaluating on Scenario: {map_name} ---")
        env = WarehouseEnv(map_name=map_name)
        start_pos, goal_pos = get_default_positions(map_name)

        # 1. Evaluate BFS
        bfs_res = bfs_search(env.obstacles, start_pos, goal_pos)
        bfs_dict = {
            "map_name": map_name,
            "method": "BFS",
            "num_episodes": 1,
            "success_rate": 1.0 if bfs_res.success else 0.0,
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "mean_steps": bfs_res.length,
            "mean_path_length": bfs_res.length,
            "mean_collisions": 0.0,
            "collision_step_ratio": 0.0,
            "mean_latency_ms": round(bfs_res.execution_time * 1000.0, 4),
            "sample_trajectory": bfs_res.path,
        }
        all_results.append(bfs_dict)
        print(
            f"BFS: Success={bfs_dict['success_rate']*100:.1f}%, "
            f"Path={bfs_dict['mean_path_length']}, "
            f"Latency={bfs_dict['mean_latency_ms']:.3f}ms"
        )

        # 2. Evaluate A*
        astar_res = astar_search(env.obstacles, start_pos, goal_pos)
        astar_dict = {
            "map_name": map_name,
            "method": "A*",
            "num_episodes": 1,
            "success_rate": 1.0 if astar_res.success else 0.0,
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "mean_steps": astar_res.length,
            "mean_path_length": astar_res.length,
            "mean_collisions": 0.0,
            "collision_step_ratio": 0.0,
            "mean_latency_ms": round(astar_res.execution_time * 1000.0, 4),
            "sample_trajectory": astar_res.path,
        }
        all_results.append(astar_dict)
        print(
            f"A* : Success={astar_dict['success_rate']*100:.1f}%, "
            f"Path={astar_dict['mean_path_length']}, "
            f"Latency={astar_dict['mean_latency_ms']:.3f}ms"
        )

        # 3. Evaluate DQN
        dqn_dict = evaluate_agent_on_map(
            agent=agent,
            map_name=map_name,
            num_episodes=cfg.evaluation.eval_episodes_per_map,
            seed=cfg.evaluation.seed,
            max_steps=cfg.env.max_steps_per_episode,
        )
        all_results.append(dqn_dict)
        print(
            f"DQN: Success={dqn_dict['success_rate']*100:.1f}%, "
            f"Path={dqn_dict['mean_path_length']}, "
            f"Steps={dqn_dict['mean_steps']}, "
            f"Collisions={dqn_dict['mean_collisions']}, "
            f"Latency={dqn_dict['mean_latency_ms']:.3f}ms"
        )

    # Save summary dataframe (without long trajectory arrays)
    records_to_save = []
    for r in all_results:
        rec = {k: v for k, v in r.items() if k != "sample_trajectory"}
        records_to_save.append(rec)

    df_eval = pd.DataFrame(records_to_save)
    cfg.evaluation.eval_results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_eval.to_csv(cfg.evaluation.eval_results_csv_path, index=False)
    print(f"\nEvaluation summary saved to: {cfg.evaluation.eval_results_csv_path}")
    print("=" * 70)

    return df_eval
