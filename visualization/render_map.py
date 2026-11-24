"""
Warehouse map rendering module.
Generates publication-quality 2D grid visual representations of warehouse scenarios.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import VIS_DIR
from environment.maps import get_map, get_default_positions


def render_warehouse_map(
    map_name: str = "moderate_map",
    save_path: Optional[Path] = None,
    custom_title: Optional[str] = None,
    dpi: int = 200,
) -> Path:
    """
    Render 2D warehouse grid with racks, aisles, start, and goal positions.
    """
    grid = get_map(map_name)
    start_pos, goal_pos = get_default_positions(map_name)
    H, W = grid.shape

    if save_path is None:
        save_path = VIS_DIR / "warehouse_map.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#1e1e24")
    ax.set_facecolor("#2b2d42")

    # Draw cells
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 1:
                # Obstacle / Warehouse Rack
                rect = patches.Rectangle(
                    (c, H - 1 - r), 1, 1,
                    facecolor="#4a4e69", edgecolor="#22223b", linewidth=1.5
                )
                ax.add_patch(rect)
                # Rack cross pattern
                ax.plot([c + 0.2, c + 0.8], [H - 1 - r + 0.2, H - 1 - r + 0.8], color="#6c757d", lw=1.0)
                ax.plot([c + 0.2, c + 0.8], [H - 1 - r + 0.8, H - 1 - r + 0.2], color="#6c757d", lw=1.0)
            else:
                # Free floor
                rect = patches.Rectangle(
                    (c, H - 1 - r), 1, 1,
                    facecolor="#edf2f4", edgecolor="#8d99ae", linewidth=0.5
                )
                ax.add_patch(rect)

    # Highlight Start (Green)
    start_r, start_c = start_pos
    rect_start = patches.Rectangle(
        (start_c + 0.1, H - 1 - start_r + 0.1), 0.8, 0.8,
        facecolor="#2ec4b6", edgecolor="#ffffff", linewidth=2.0
    )
    ax.add_patch(rect_start)
    ax.text(
        start_c + 0.5, H - 1 - start_r + 0.5, "START",
        color="white", weight="bold", fontsize=9, ha="center", va="center"
    )

    # Highlight Goal (Red / Gold)
    goal_r, goal_c = goal_pos
    rect_goal = patches.Rectangle(
        (goal_c + 0.1, H - 1 - goal_r + 0.1), 0.8, 0.8,
        facecolor="#e71d36", edgecolor="#ffffff", linewidth=2.0
    )
    ax.add_patch(rect_goal)
    ax.text(
        goal_c + 0.5, H - 1 - goal_r + 0.5, "GOAL",
        color="white", weight="bold", fontsize=9, ha="center", va="center"
    )

    # Grid axes and coordinates
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks(np.arange(0.5, W, 1))
    ax.set_yticks(np.arange(0.5, H, 1))
    ax.set_xticklabels(range(W), color="#f8f9fa", fontsize=9)
    ax.set_yticklabels([str(H - 1 - i) for i in range(H)], color="#f8f9fa", fontsize=9)
    ax.tick_params(axis="both", which="both", length=0)

    title = custom_title or f"Autonomous Warehouse Navigation Grid ({map_name} - {H}x{W})"
    ax.set_title(title, color="#ffffff", fontsize=13, weight="bold", pad=15)

    # Legend elements
    legend_elements = [
        patches.Patch(facecolor="#2ec4b6", edgecolor="#ffffff", label="Start Location"),
        patches.Patch(facecolor="#e71d36", edgecolor="#ffffff", label="Goal / Drop-off Bay"),
        patches.Patch(facecolor="#4a4e69", edgecolor="#22223b", label="Storage Racks (Obstacle)"),
        patches.Patch(facecolor="#edf2f4", edgecolor="#8d99ae", label="Aisle / Free Corridor"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.06),
        ncol=2, frameon=True, facecolor="#2b2d42", edgecolor="#8d99ae",
        labelcolor="#ffffff", fontsize=9
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Warehouse map saved to: {save_path}")
    return save_path
