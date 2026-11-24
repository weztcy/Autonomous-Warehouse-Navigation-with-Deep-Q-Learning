"""
Q-Network architecture in PyTorch.
Multi-layer Perceptron (MLP) mapping observation vector to Q-values for each action.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Deep Q-Network for Warehouse Navigation.
    Maps observation vector (19-dim) to discrete Q-values (4 actions).
    """

    def __init__(
        self,
        state_dim: int = 19,
        action_dim: int = 4,
        hidden_dims: Tuple[int, ...] = (128, 128, 64),
    ):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            state (torch.Tensor): Shape (batch_size, state_dim) or (state_dim,)
        Returns:
            torch.Tensor: Q-values of shape (batch_size, action_dim)
        """
        return self.network(state)
