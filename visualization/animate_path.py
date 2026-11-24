"""
Path visualization and navigation animation module.
Generates trajectory overlays (dqn_path.png) and high-quality GIF animations (robot_navigation.gif).
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from config import VIS_DIR
from environment.maps import get_map, get_default_positions
from environment.warehouse_env import WarehouseEnv
from agents.dqn_agent import DQNAgent


def plot_trajectories(
    map_name: str,
    dqn_path: List[Tuple[int, int]],
    bfs_path: Optional[List[Tuple[int, int]]] = None,
    astar_path: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[Path] = None,
    dpi: int = 200,
) -> Path:
    """
    Overlay navigation trajectories of DQN vs BFS / A* on warehouse grid.
    """
    grid = get_map(map_name)
    start_pos, goal_pos = get_default_positions(map_name)
    H, W = grid.shape

    if save_path is None:
        save_path = VIS_DIR / "dqn_path.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#1e1e24")
    ax.set_facecolor("#2b2d42")

    # Draw grid floor and obstacles
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 1:
                rect = patches.Rectangle(
                    (c, H - 1 - r), 1, 1,
                    facecolor="#4a4e69", edgecolor="#22223b", linewidth=1.2
                )
                ax.add_patch(rect)
                ax.plot([c + 0.2, c + 0.8], [H - 1 - r + 0.2, H - 1 - r + 0.8], color="#6c757d", lw=0.8)
                ax.plot([c + 0.2, c + 0.8], [H - 1 - r + 0.8, H - 1 - r + 0.2], color="#6c757d", lw=0.8)
            else:
                rect = patches.Rectangle(
                    (c, H - 1 - r), 1, 1,
                    facecolor="#edf2f4", edgecolor="#8d99ae", linewidth=0.5
                )
                ax.add_patch(rect)

    # Start & Goal
    start_r, start_c = start_pos
    rect_start = patches.Rectangle(
        (start_c + 0.1, H - 1 - start_r + 0.1), 0.8, 0.8,
        facecolor="#2ec4b6", edgecolor="#ffffff", linewidth=2.0
    )
    ax.add_patch(rect_start)
    ax.text(start_c + 0.5, H - 1 - start_r + 0.5, "START", color="white", weight="bold", fontsize=8, ha="center", va="center")

    goal_r, goal_c = goal_pos
    rect_goal = patches.Rectangle(
        (goal_c + 0.1, H - 1 - goal_r + 0.1), 0.8, 0.8,
        facecolor="#e71d36", edgecolor="#ffffff", linewidth=2.0
    )
    ax.add_patch(rect_goal)
    ax.text(goal_c + 0.5, H - 1 - goal_r + 0.5, "GOAL", color="white", weight="bold", fontsize=8, ha="center", va="center")

    # Plot BFS path
    if bfs_path and len(bfs_path) > 1:
        bfs_cols = [p[1] + 0.5 for p in bfs_path]
        bfs_rows = [H - 1 - p[0] + 0.5 for p in bfs_path]
        ax.plot(bfs_cols, bfs_rows, color="#f72585", lw=3.0, linestyle="--", alpha=0.85, label=f"BFS Baseline ({len(bfs_path)-1} steps)")

    # Plot A* path
    if astar_path and len(astar_path) > 1:
        astar_cols = [p[1] + 0.5 for p in astar_path]
        astar_rows = [H - 1 - p[0] + 0.5 for p in astar_path]
        ax.plot(astar_cols, astar_rows, color="#7209b7", lw=2.0, linestyle=":", alpha=0.9, label=f"A* Baseline ({len(astar_path)-1} steps)")

    # Plot DQN path
    if dqn_path and len(dqn_path) > 1:
        dqn_cols = [p[1] + 0.5 for p in dqn_path]
        dqn_rows = [H - 1 - p[0] + 0.5 for p in dqn_path]
        ax.plot(dqn_cols, dqn_rows, color="#4cc9f0", lw=3.5, alpha=0.95, label=f"DQN Learned Path ({len(dqn_path)-1} steps)")
        # Plot markers at intermediate steps
        ax.scatter(dqn_cols[1:-1], dqn_rows[1:-1], color="#4cc9f0", edgecolors="#ffffff", s=40, zorder=5)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks(np.arange(0.5, W, 1))
    ax.set_yticks(np.arange(0.5, H, 1))
    ax.set_xticklabels(range(W), color="#f8f9fa", fontsize=8)
    ax.set_yticklabels([str(H - 1 - i) for i in range(H)], color="#f8f9fa", fontsize=8)
    ax.tick_params(axis="both", which="both", length=0)

    ax.set_title(f"Warehouse Trajectory Comparison ({map_name})", color="#ffffff", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=True, facecolor="#2b2d42", edgecolor="#8d99ae", labelcolor="#ffffff", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Trajectory plot saved to: {save_path}")
    return save_path


def generate_navigation_gif(
    map_name: str,
    path_history: List[Tuple[int, int]],
    save_path: Optional[Path] = None,
    fps: int = 4,
    dpi: int = 120,
) -> Path:
    """
    Render step-by-step animated GIF of robot navigating the warehouse grid.
    """
    grid = get_map(map_name)
    start_pos, goal_pos = get_default_positions(map_name)
    H, W = grid.shape

    if save_path is None:
        save_path = VIS_DIR / "robot_navigation.gif"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    frames: List[Image.Image] = []

    for step_idx, current_pos in enumerate(path_history):
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#1e1e24")
        ax.set_facecolor("#2b2d42")

        # Draw grid cells
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 1:
                    rect = patches.Rectangle(
                        (c, H - 1 - r), 1, 1,
                        facecolor="#4a4e69", edgecolor="#22223b", linewidth=1.0
                    )
                    ax.add_patch(rect)
                else:
                    rect = patches.Rectangle(
                        (c, H - 1 - r), 1, 1,
                        facecolor="#edf2f4", edgecolor="#8d99ae", linewidth=0.5
                    )
                    ax.add_patch(rect)

        # Start & Goal
        start_r, start_c = start_pos
        rect_start = patches.Rectangle(
            (start_c + 0.1, H - 1 - start_r + 0.1), 0.8, 0.8,
            facecolor="#2ec4b6", edgecolor="#ffffff", linewidth=1.5
        )
        ax.add_patch(rect_start)
        ax.text(start_c + 0.5, H - 1 - start_r + 0.5, "S", color="white", weight="bold", fontsize=10, ha="center", va="center")

        goal_r, goal_c = goal_pos
        rect_goal = patches.Rectangle(
            (goal_c + 0.1, H - 1 - goal_r + 0.1), 0.8, 0.8,
            facecolor="#e71d36", edgecolor="#ffffff", linewidth=1.5
        )
        ax.add_patch(rect_goal)
        ax.text(goal_c + 0.5, H - 1 - goal_r + 0.5, "G", color="white", weight="bold", fontsize=10, ha="center", va="center")

        # Breadcrumbs of past path
        past_path = path_history[: step_idx + 1]
        if len(past_path) > 1:
            p_cols = [p[1] + 0.5 for p in past_path]
            p_rows = [H - 1 - p[0] + 0.5 for p in past_path]
            ax.plot(p_cols, p_rows, color="#4cc9f0", lw=2.5, alpha=0.7)

        # Current Robot Position
        rob_r, rob_c = current_pos
        robot_circle = patches.Circle(
            (rob_c + 0.5, H - 1 - rob_r + 0.5), 0.35,
            facecolor="#ffd166", edgecolor="#ffffff", linewidth=2.0, zorder=6
        )
        ax.add_patch(robot_circle)
        ax.text(rob_c + 0.5, H - 1 - rob_r + 0.5, "AGV", color="#1e1e24", weight="bold", fontsize=7, ha="center", va="center", zorder=7)

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_xticks(np.arange(0.5, W, 1))
        ax.set_yticks(np.arange(0.5, H, 1))
        ax.set_xticklabels(range(W), color="#f8f9fa", fontsize=7)
        ax.set_yticklabels([str(H - 1 - i) for i in range(H)], color="#f8f9fa", fontsize=7)
        ax.tick_params(axis="both", which="both", length=0)

        status_text = "STATUS: REACHED GOAL!" if current_pos == goal_pos else "STATUS: NAVIGATING"
        ax.set_title(
            f"Step: {step_idx}/{len(path_history)-1} | Pos: {current_pos} | {status_text}",
            color="#ffffff", fontsize=10, weight="bold", pad=10
        )

        plt.tight_layout()
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        frame_img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), rgba_buffer, "raw", "RGBA", 0, 1)
        frames.append(frame_img.convert("RGB"))
        plt.close()

    # Save animated GIF using PIL
    if frames:
        # Pause at the end for 1.5 seconds
        final_frame = frames[-1]
        for _ in range(5):
            frames.append(final_frame)

        duration_ms = int(1000 / fps)
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        print(f"Animated GIF saved to: {save_path} ({len(frames)} frames)")

    return save_path
