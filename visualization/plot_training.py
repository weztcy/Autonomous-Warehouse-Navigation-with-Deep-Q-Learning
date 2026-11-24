"""
Training metrics visualization.
Plots rewards, moving average, training loss, success rate, and epsilon decay curves.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import VIS_DIR, RESULTS_DIR


def plot_all_training_metrics(
    csv_path: Optional[Path] = None,
    save_dir: Optional[Path] = None,
    dpi: int = 200,
) -> None:
    """Generate individual and composite training plots."""
    if csv_path is None:
        csv_path = RESULTS_DIR / "training_history.csv"
    if save_dir is None:
        save_dir = VIS_DIR

    csv_path = Path(csv_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Training history CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    episodes = df["episode"]

    # Style setting for dark theme
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    # 1. Plot Reward & Moving Average
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#1a1a24")
    ax.set_facecolor("#222330")
    ax.plot(episodes, df["reward"], color="#4cc9f0", alpha=0.35, label="Raw Episode Reward", lw=1.2)
    if "moving_reward_50" in df.columns:
        ax.plot(episodes, df["moving_reward_50"], color="#f72585", lw=2.5, label="50-Ep Moving Average Reward")
    ax.set_title("Training Reward Progression", color="#ffffff", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Episode", color="#ffffff", fontsize=10)
    ax.set_ylabel("Total Reward", color="#ffffff", fontsize=10)
    ax.tick_params(colors="#e0e0e0")
    ax.grid(color="#3d405b", linestyle="--", alpha=0.6)
    ax.legend(facecolor="#1a1a24", edgecolor="#3d405b", labelcolor="#ffffff")
    plt.tight_layout()
    reward_path = save_dir / "training_reward.png"
    plt.savefig(reward_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {reward_path}")

    # 2. Plot Training Loss
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#1a1a24")
    ax.set_facecolor("#222330")
    ax.plot(episodes, df["loss"], color="#7209b7", lw=1.8, label="Huber Loss (Mean per Episode)")
    if len(df) >= 20:
        loss_ma = df["loss"].rolling(window=20, min_periods=1).mean()
        ax.plot(episodes, loss_ma, color="#4895ef", lw=2.2, label="20-Ep Moving Average Loss")
    ax.set_title("DQN Policy Network Huber Loss", color="#ffffff", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Episode", color="#ffffff", fontsize=10)
    ax.set_ylabel("Loss", color="#ffffff", fontsize=10)
    ax.tick_params(colors="#e0e0e0")
    ax.grid(color="#3d405b", linestyle="--", alpha=0.6)
    ax.legend(facecolor="#1a1a24", edgecolor="#3d405b", labelcolor="#ffffff")
    plt.tight_layout()
    loss_path = save_dir / "training_loss.png"
    plt.savefig(loss_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {loss_path}")

    # 3. Plot Success Rate
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#1a1a24")
    ax.set_facecolor("#222330")
    if "moving_success_50" in df.columns:
        ax.plot(episodes, df["moving_success_50"] * 100, color="#4ade80", lw=2.5, label="50-Ep Success Rate (%)")
    else:
        succ_ma = df["success"].rolling(window=50, min_periods=1).mean() * 100
        ax.plot(episodes, succ_ma, color="#4ade80", lw=2.5, label="50-Ep Success Rate (%)")
    ax.axhline(100, color="#6c757d", linestyle=":", alpha=0.7)
    ax.set_title("Training Navigation Success Rate (Goal Reached)", color="#ffffff", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Episode", color="#ffffff", fontsize=10)
    ax.set_ylabel("Success Rate (%)", color="#ffffff", fontsize=10)
    ax.set_ylim(-5, 105)
    ax.tick_params(colors="#e0e0e0")
    ax.grid(color="#3d405b", linestyle="--", alpha=0.6)
    ax.legend(facecolor="#1a1a24", edgecolor="#3d405b", labelcolor="#ffffff")
    plt.tight_layout()
    succ_path = save_dir / "success_rate.png"
    plt.savefig(succ_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {succ_path}")

    # 4. Plot Epsilon Decay
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#1a1a24")
    ax.set_facecolor("#222330")
    ax.plot(episodes, df["epsilon"], color="#fca311", lw=2.5, label="Exploration Rate (Epsilon)")
    ax.set_title("Epsilon-Greedy Exploration Schedule", color="#ffffff", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Episode", color="#ffffff", fontsize=10)
    ax.set_ylabel(r"Epsilon ($\epsilon$)", color="#ffffff", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(colors="#e0e0e0")
    ax.grid(color="#3d405b", linestyle="--", alpha=0.6)
    ax.legend(facecolor="#1a1a24", edgecolor="#3d405b", labelcolor="#ffffff")
    plt.tight_layout()
    eps_path = save_dir / "epsilon_decay.png"
    plt.savefig(eps_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {eps_path}")

    # 5. Composite 4-Panel Training Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), facecolor="#1a1a24")

    # [0, 0] Reward
    axs[0, 0].set_facecolor("#222330")
    axs[0, 0].plot(episodes, df["reward"], color="#4cc9f0", alpha=0.35, label="Episode Reward")
    if "moving_reward_50" in df.columns:
        axs[0, 0].plot(episodes, df["moving_reward_50"], color="#f72585", lw=2.2, label="Moving Avg (50)")
    axs[0, 0].set_title("(A) Cumulative Reward Progression", color="#ffffff", fontsize=11, weight="bold")
    axs[0, 0].set_xlabel("Episode", color="#ffffff", fontsize=9)
    axs[0, 0].set_ylabel("Reward", color="#ffffff", fontsize=9)
    axs[0, 0].tick_params(colors="#e0e0e0")
    axs[0, 0].grid(color="#3d405b", linestyle="--", alpha=0.6)
    axs[0, 0].legend(facecolor="#1a1a24", edgecolor="#3d405b", labelcolor="#ffffff", fontsize=8)

    # [0, 1] Loss
    axs[0, 1].set_facecolor("#222330")
    axs[0, 1].plot(episodes, df["loss"], color="#7209b7", lw=1.5, alpha=0.7, label="Huber Loss")
    if len(df) >= 20:
        loss_ma = df["loss"].rolling(window=20, min_periods=1).mean()
        axs[0, 1].plot(episodes, loss_ma, color="#4895ef", lw=2.0, label="Moving Avg (20)")
    axs[0, 1].set_title("(B) Bellman Huber Loss Convergence", color="#ffffff", fontsize=11, weight="bold")
    axs[0, 1].set_xlabel("Episode", color="#ffffff", fontsize=9)
    axs[0, 1].set_ylabel("Loss", color="#ffffff", fontsize=9)
    axs[0, 1].tick_params(colors="#e0e0e0")
    axs[0, 1].grid(color="#3d405b", linestyle="--", alpha=0.6)
    axs[0, 1].legend(facecolor="#1a1a24", edgecolor="#3d405b", labelcolor="#ffffff", fontsize=8)

    # [1, 0] Success Rate
    axs[1, 0].set_facecolor("#222330")
    if "moving_success_50" in df.columns:
        axs[1, 0].plot(episodes, df["moving_success_50"] * 100, color="#4ade80", lw=2.2, label="Success Rate (%)")
    axs[1, 0].set_title("(C) Goal Navigation Success Rate (%)", color="#ffffff", fontsize=11, weight="bold")
    axs[1, 0].set_xlabel("Episode", color="#ffffff", fontsize=9)
    axs[1, 0].set_ylabel("Success Rate (%)", color="#ffffff", fontsize=9)
    axs[1, 0].set_ylim(-5, 105)
    axs[1, 0].tick_params(colors="#e0e0e0")
    axs[1, 0].grid(color="#3d405b", linestyle="--", alpha=0.6)
    axs[1, 0].legend(facecolor="#1a1a24", edgecolor="#3d405b", labelcolor="#ffffff", fontsize=8)

    # [1, 1] Epsilon & Steps
    axs[1, 1].set_facecolor("#222330")
    color_eps = "#fca311"
    axs[1, 1].plot(episodes, df["epsilon"], color=color_eps, lw=2.0, label="Epsilon (Exploration)")
    axs[1, 1].set_title("(D) Exploration Rate & Episode Steps", color="#ffffff", fontsize=11, weight="bold")
    axs[1, 1].set_xlabel("Episode", color="#ffffff", fontsize=9)
    axs[1, 1].set_ylabel("Epsilon", color=color_eps, fontsize=9)
    axs[1, 1].tick_params(axis="y", labelcolor=color_eps)
    axs[1, 1].tick_params(axis="x", colors="#e0e0e0")
    axs[1, 1].set_ylim(0.0, 1.05)
    axs[1, 1].grid(color="#3d405b", linestyle="--", alpha=0.6)

    # Twin axis for steps
    ax2 = axs[1, 1].twinx()
    steps_ma = df["steps"].rolling(window=20, min_periods=1).mean()
    color_steps = "#a8dadc"
    ax2.plot(episodes, steps_ma, color=color_steps, lw=1.8, linestyle="--", label="Steps (20-MA)")
    ax2.set_ylabel("Steps to Complete", color=color_steps, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=color_steps)
    ax2.grid(False)

    plt.tight_layout()
    composite_path = save_dir / "training_summary.png"
    plt.savefig(composite_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved composite summary: {composite_path}")
