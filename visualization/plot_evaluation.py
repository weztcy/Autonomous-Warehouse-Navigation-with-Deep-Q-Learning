"""
Evaluation comparison plotting module.
Visualizes DQN vs BFS vs A* metrics across maps: success rate, path length, collisions, and latency.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import VIS_DIR, RESULTS_DIR


def plot_evaluation_comparison(
    csv_path: Optional[Path] = None,
    save_dir: Optional[Path] = None,
    dpi: int = 200,
) -> None:
    """Generate comparative visualization between DQN, BFS, and A*."""
    if csv_path is None:
        csv_path = RESULTS_DIR / "evaluation_results.csv"
    if save_dir is None:
        save_dir = VIS_DIR

    csv_path = Path(csv_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation results CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Bar Chart: Path Length Comparison
    maps = df["map_name"].unique()
    methods = ["DQN", "BFS", "A*"]
    x = np.arange(len(maps))
    width = 0.25

    colors = {"DQN": "#4cc9f0", "BFS": "#f72585", "A*": "#4ade80"}

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#1a1a24")
    ax.set_facecolor("#222330")

    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("map_name")
        lengths = [sub.loc[m]["mean_path_length"] if m in sub.index else 0 for m in maps]
        rects = ax.bar(x + (i - 1) * width, lengths, width, label=method, color=colors[method], edgecolor="#ffffff", linewidth=0.8)
        # Value label on top of bars
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", color="#ffffff", fontsize=8, weight="bold"
                )

    clean_map_names = [m.replace("_", " ").title() for m in maps]
    ax.set_xticks(x)
    ax.set_xticklabels(clean_map_names, color="#ffffff", fontsize=10, weight="bold")
    ax.set_title("Navigation Path Length Comparison: DQN vs BFS vs A*", color="#ffffff", fontsize=13, weight="bold", pad=14)
    ax.set_ylabel("Mean Steps / Path Length", color="#ffffff", fontsize=10)
    ax.tick_params(colors="#e0e0e0")
    ax.grid(color="#3d405b", linestyle="--", alpha=0.5, axis="y")
    ax.legend(facecolor="#1a1a24", edgecolor="#3d405b", labelcolor="#ffffff")
    plt.tight_layout()
    comparison_path = save_dir / "baseline_comparison.png"
    plt.savefig(comparison_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {comparison_path}")

    # 2. Multi-metric Comparison Grid (Success Rate, Path Length, Latency, Collisions)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), facecolor="#1a1a24")

    # (A) Success Rate
    axs[0, 0].set_facecolor("#222330")
    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("map_name")
        succs = [sub.loc[m]["success_rate"] * 100 if m in sub.index else 0 for m in maps]
        axs[0, 0].bar(x + (i - 1) * width, succs, width, label=method, color=colors[method], edgecolor="#ffffff", linewidth=0.5)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(clean_map_names, color="#ffffff", fontsize=8)
    axs[0, 0].set_title("(A) Success Rate (%)", color="#ffffff", fontsize=11, weight="bold")
    axs[0, 0].set_ylabel("Success (%)", color="#ffffff", fontsize=9)
    axs[0, 0].set_ylim(0, 115)
    axs[0, 0].tick_params(colors="#e0e0e0")
    axs[0, 0].grid(color="#3d405b", linestyle="--", alpha=0.5, axis="y")
    axs[0, 0].legend(facecolor="#1a1a24", edgecolor="#3d405b", labelcolor="#ffffff", fontsize=8)

    # (B) Path Length
    axs[0, 1].set_facecolor("#222330")
    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("map_name")
        lengths = [sub.loc[m]["mean_path_length"] if m in sub.index else 0 for m in maps]
        axs[0, 1].bar(x + (i - 1) * width, lengths, width, label=method, color=colors[method], edgecolor="#ffffff", linewidth=0.5)
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(clean_map_names, color="#ffffff", fontsize=8)
    axs[0, 1].set_title("(B) Path Length (Steps)", color="#ffffff", fontsize=11, weight="bold")
    axs[0, 1].set_ylabel("Steps", color="#ffffff", fontsize=9)
    axs[0, 1].tick_params(colors="#e0e0e0")
    axs[0, 1].grid(color="#3d405b", linestyle="--", alpha=0.5, axis="y")
    axs[0, 1].legend(facecolor="#1a1a24", edgecolor="#3d405b", labelcolor="#ffffff", fontsize=8)

    # (C) Latency / Execution Time (Log Scale)
    axs[1, 0].set_facecolor("#222330")
    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("map_name")
        lats = [sub.loc[m]["mean_latency_ms"] if m in sub.index else 0 for m in maps]
        axs[1, 0].bar(x + (i - 1) * width, lats, width, label=method, color=colors[method], edgecolor="#ffffff", linewidth=0.5)
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(clean_map_names, color="#ffffff", fontsize=8)
    axs[1, 0].set_title("(C) Execution Latency (ms)", color="#ffffff", fontsize=11, weight="bold")
    axs[1, 0].set_ylabel("Latency (ms)", color="#ffffff", fontsize=9)
    axs[1, 0].tick_params(colors="#e0e0e0")
    axs[1, 0].grid(color="#3d405b", linestyle="--", alpha=0.5, axis="y")
    axs[1, 0].legend(facecolor="#1a1a24", edgecolor="#3d405b", labelcolor="#ffffff", fontsize=8)

    # (D) Collision Count
    axs[1, 1].set_facecolor("#222330")
    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("map_name")
        colls = [sub.loc[m]["mean_collisions"] if m in sub.index else 0 for m in maps]
        axs[1, 1].bar(x + (i - 1) * width, colls, width, label=method, color=colors[method], edgecolor="#ffffff", linewidth=0.5)
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(clean_map_names, color="#ffffff", fontsize=8)
    axs[1, 1].set_title("(D) Average Collisions per Episode", color="#ffffff", fontsize=11, weight="bold")
    axs[1, 1].set_ylabel("Collisions", color="#ffffff", fontsize=9)
    axs[1, 1].tick_params(colors="#e0e0e0")
    axs[1, 1].grid(color="#3d405b", linestyle="--", alpha=0.5, axis="y")
    axs[1, 1].legend(facecolor="#1a1a24", edgecolor="#3d405b", labelcolor="#ffffff", fontsize=8)

    plt.tight_layout()
    metrics_path = save_dir / "evaluation_metrics.png"
    plt.savefig(metrics_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {metrics_path}")
