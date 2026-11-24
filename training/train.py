"""
Training script for Deep Q-Network on Warehouse Navigation.
Logs metrics, saves checkpoints, and outputs experiment records.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import time
import numpy as np
import pandas as pd
import torch

from config import Config, default_config
from environment.warehouse_env import WarehouseEnv
from agents.dqn_agent import DQNAgent


def train_dqn(cfg: Config = default_config) -> pd.DataFrame:
    """Train DQN agent according to configuration."""
    print("=" * 70)
    print("STARTING AUTONOMOUS WAREHOUSE NAVIGATION DQN TRAINING")
    print(f"Map: {cfg.env.map_name} | Episodes: {cfg.training.num_episodes}")
    print(f"Seed: {cfg.training.seed} | Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)

    # Set seeds
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    env = WarehouseEnv(
        map_name=cfg.env.map_name,
        max_steps=cfg.env.max_steps_per_episode,
        reward_goal=cfg.env.reward_goal,
        reward_step=cfg.env.reward_step,
        reward_collision=cfg.env.reward_collision,
        reward_timeout=cfg.env.reward_timeout,
        reward_distance_shaping=cfg.env.reward_distance_shaping,
        shaping_weight=cfg.env.shaping_weight,
    )

    agent = DQNAgent(cfg.dqn, seed=cfg.training.seed)

    history: List[Dict[str, Any]] = []
    best_eval_success = -1.0
    best_moving_reward = -float("inf")

    start_time = time.time()

    for episode in range(1, cfg.training.num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_losses: List[float] = []
        done = False

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, done, info = env.step(action)

            loss = agent.step(obs, action, reward, next_obs, done)
            if loss is not None:
                episode_losses.append(loss)

            obs = next_obs
            episode_reward += reward

        # Decay epsilon at the end of episode
        curr_eps = agent.decay_epsilon()
        mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        success = 1.0 if info["reached_goal"] else 0.0
        collisions = info["collision_count"]
        steps = info["step_count"]

        # Record metrics
        ep_record = {
            "episode": episode,
            "reward": round(episode_reward, 2),
            "steps": steps,
            "collisions": collisions,
            "success": success,
            "epsilon": round(curr_eps, 4),
            "loss": round(mean_loss, 4),
        }
        history.append(ep_record)

        # Compute rolling window metrics
        recent_episodes = history[-50:]
        moving_reward = np.mean([h["reward"] for h in recent_episodes])
        moving_success = np.mean([h["success"] for h in recent_episodes])
        ep_record["moving_reward_50"] = round(moving_reward, 2)
        ep_record["moving_success_50"] = round(moving_success, 4)

        # Periodic logging and evaluation
        if episode % 25 == 0 or episode == 1:
            elapsed = time.time() - start_time
            print(
                f"Ep {episode:4d}/{cfg.training.num_episodes} | "
                f"Reward: {episode_reward:6.1f} | "
                f"MovAvg(50): {moving_reward:6.1f} | "
                f"Succ(50): {moving_success*100:5.1f}% | "
                f"Steps: {steps:3d} | "
                f"Colls: {collisions:2d} | "
                f"Loss: {mean_loss:6.4f} | "
                f"Eps: {curr_eps:5.3f} | "
                f"Time: {elapsed:5.1f}s"
            )

        # Checkpoint: Save model if moving average reward or success improves significantly
        if episode >= 100:
            if moving_success > best_eval_success or (
                moving_success == best_eval_success and moving_reward > best_moving_reward
            ):
                best_eval_success = moving_success
                best_moving_reward = moving_reward
                agent.save(cfg.training.best_model_path)

    # Save final model
    agent.save(cfg.training.final_model_path)
    print(f"Final model saved to: {cfg.training.final_model_path}")
    if cfg.training.best_model_path.exists():
        print(f"Best model checkpoint saved to: {cfg.training.best_model_path}")

    # Export training history to CSV
    df_history = pd.DataFrame(history)
    cfg.training.history_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_history.to_csv(cfg.training.history_csv_path, index=False)
    print(f"Training history saved to: {cfg.training.history_csv_path}")

    # Save experiment config JSON
    exp_config = {
        "env": {
            "map_name": cfg.env.map_name,
            "max_steps": cfg.env.max_steps_per_episode,
            "reward_goal": cfg.env.reward_goal,
            "reward_step": cfg.env.reward_step,
            "reward_collision": cfg.env.reward_collision,
            "reward_timeout": cfg.env.reward_timeout,
            "reward_distance_shaping": cfg.env.reward_distance_shaping,
            "shaping_weight": cfg.env.shaping_weight,
        },
        "dqn": {
            "state_dim": cfg.dqn.state_dim,
            "action_dim": cfg.dqn.action_dim,
            "hidden_dims": list(cfg.dqn.hidden_dims),
            "learning_rate": cfg.dqn.learning_rate,
            "gamma": cfg.dqn.gamma,
            "replay_capacity": cfg.dqn.replay_capacity,
            "batch_size": cfg.dqn.batch_size,
            "epsilon_start": cfg.dqn.epsilon_start,
            "epsilon_min": cfg.dqn.epsilon_min,
            "epsilon_decay": cfg.dqn.epsilon_decay,
            "target_update_tau": cfg.dqn.target_update_tau,
        },
        "training": {
            "num_episodes": cfg.training.num_episodes,
            "seed": cfg.training.seed,
            "total_time_seconds": round(time.time() - start_time, 2),
            "best_moving_success_50": round(float(best_eval_success), 4),
            "best_moving_reward_50": round(float(best_moving_reward), 2),
        },
    }
    with open(cfg.training.config_json_path, "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=4)
    print(f"Experiment config saved to: {cfg.training.config_json_path}")
    print("TRAINING COMPLETE!")
    print("=" * 70)

    return df_history
