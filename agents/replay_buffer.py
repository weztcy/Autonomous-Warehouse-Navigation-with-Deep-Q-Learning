"""
Experience Replay Buffer for DQN.
Stores transitions (state, action, reward, next_state, done) and samples mini-batches.
"""

from typing import Tuple
from collections import deque
import random
import numpy as np
import torch


class ReplayBuffer:
    """Experience Replay Buffer supporting uniform mini-batch sampling."""

    def __init__(self, capacity: int = 40000, seed: int = 42):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)
        np.random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random mini-batch of transitions.
        Returns:
            states, actions, rewards, next_states, dones as torch tensors on device.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)
