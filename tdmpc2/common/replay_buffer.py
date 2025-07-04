import torch
import numpy as np


class ReplayBuffer:
        """Simple replay buffer for off-policy algorithms."""

        def __init__(self, obs_dim, action_dim, capacity, device='cpu'):
                self.capacity = capacity
                self.device = torch.device(device)
                self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.device)
                self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.device)
                self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.device)
                self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
                self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
                self.idx = 0
                self.full = False

        def add(self, obs, action, reward, next_obs, done):
                self.obs[self.idx] = obs
                self.actions[self.idx] = action
                self.rewards[self.idx] = reward
                self.next_obs[self.idx] = next_obs
                self.dones[self.idx] = done
                self.idx = (self.idx + 1) % self.capacity
                self.full = self.full or self.idx == 0

        def sample(self, batch_size):
                max_idx = self.capacity if self.full else self.idx
                idx = torch.randint(0, max_idx, (batch_size,), device=self.device)
                return (
                        self.obs[idx],
                        self.actions[idx],
                        self.rewards[idx],
                        self.next_obs[idx],
                        self.dones[idx],
                )

        def __len__(self):
                return self.capacity if self.full else self.idx
