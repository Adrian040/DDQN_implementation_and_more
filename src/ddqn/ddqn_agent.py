"""
Implementación de Double DQN (DDQN) en PyTorch.

Características principales:
- Replay buffer
- Red online y red target
- Target DDQN:
    a* = argmax_a Q_online(s', a)
    y  = r + gamma * Q_target(s', a*)
- Pérdida de Huber
- Optimizador Adam
- Gradient clipping
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    """
    Buffer de repetición de experiencia con almacenamiento en arreglos.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device, obs_dtype=np.uint8):
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape

        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size, self.capacity - 1) + 1

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)

        obs = torch.as_tensor(self.obs[idx], device=self.device)
        next_obs = torch.as_tensor(self.next_obs[idx], device=self.device)
        actions = torch.as_tensor(self.actions[idx], device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(self.rewards[idx], device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(self.dones[idx], device=self.device, dtype=torch.float32)

        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return self.size


class QNetwork(nn.Module):
    """
    Red Q adaptable a observaciones de:
    - imágenes pequeñas (MinAtar)
    - imágenes Atari 84x84
    - vectores (si hiciera falta)
    """

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int):
        super().__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        if len(obs_shape) == 1:
            self.feature_extractor = nn.Sequential(
                nn.Linear(obs_shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            feature_dim = 256

        elif len(obs_shape) == 3:
            c, h, w = obs_shape

            # Arquitectura pequeña para MinAtar
            if h <= 16 and w <= 16:
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(c, 16, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
            else:
                # Arquitectura tipo Nature DQN para Atari
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(c, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )

            with torch.no_grad():
                dummy = torch.zeros(1, *obs_shape)
                feature_dim = self.feature_extractor(dummy).shape[1]
        else:
            raise ValueError(f"Forma de observación no soportada: {obs_shape}")

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512 if len(obs_shape) == 3 and obs_shape[-1] >= 32 else 128),
            nn.ReLU(),
            nn.Linear(512 if len(obs_shape) == 3 and obs_shape[-1] >= 32 else 128, n_actions),
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        if len(self.obs_shape) == 1:
            feats = self.feature_extractor(x)
        else:
            feats = self.feature_extractor(x)
        return self.head(feats)


@dataclass
class DDQNConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    min_buffer_size: int = 10_000
    target_update_freq: int = 1_000
    train_freq: int = 4
    grad_clip_norm: float = 10.0


class DDQNAgent:
    def __init__(self, obs_shape: tuple[int, ...], n_actions: int, device: torch.device, cfg: DDQNConfig):
        self.device = device
        self.cfg = cfg
        self.n_actions = n_actions

        self.online_net = QNetwork(obs_shape, n_actions).to(device)
        self.target_net = QNetwork(obs_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.buffer_size, obs_shape, device)

        self.num_updates = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)

        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        q_values = self.online_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item())

    @torch.no_grad()
    def act_greedy(self, obs: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        q_values = self.online_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item())

    def store(self, obs, action, reward, next_obs, done):
        self.replay.add(obs, action, reward, next_obs, done)

    def can_update(self) -> bool:
        return len(self.replay) >= self.cfg.min_buffer_size

    def update(self):
        obs, actions, rewards, next_obs, dones = self.replay.sample(self.cfg.batch_size)

        # Q(s, a) de la red online
        q_values = self.online_net(obs)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # DDQN:
            # 1) seleccionar acción con la red online
            next_actions = self.online_net(next_obs).argmax(dim=1, keepdim=True)

            # 2) evaluar acción con la red target
            next_q_target = self.target_net(next_obs).gather(1, next_actions).squeeze(1)

            targets = rewards + self.cfg.gamma * (1.0 - dones) * next_q_target

        loss = F.huber_loss(q_sa, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()

        self.num_updates += 1
        return float(loss.item())

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str):
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])