"""
Implementación de PPO para acciones discretas.

Componentes:
- red actor-crítico
- almacenamiento de rollouts
- GAE(lambda)
- objetivo clipped
- pérdida de valor y bonificación de entropía
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ActorCritic(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], n_actions: int):
        super().__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        if len(obs_shape) == 1:
            self.feature_extractor = nn.Sequential(
                nn.Linear(obs_shape[0], 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
            )
            feature_dim = 256

        elif len(obs_shape) == 3:
            c, h, w = obs_shape

            if h <= 16 and w <= 16:
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(c, 16, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
            else:
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

        hidden_dim = 512 if len(obs_shape) == 3 and obs_shape[-1] >= 32 else 128

        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        return x

    def forward(self, x: torch.Tensor):
        x = self.preprocess(x)
        feats = self.feature_extractor(x)
        logits = self.policy_head(feats)
        value = self.value_head(feats).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


@dataclass
class PPOConfig:
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 2048
    normalize_advantages: bool = True


class RolloutBuffer:
    def __init__(self, rollout_steps: int, obs_shape: tuple[int, ...], device: torch.device):
        self.rollout_steps = rollout_steps
        self.obs_shape = obs_shape
        self.device = device
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(np.array(obs, copy=True))
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def get_tensors(self, advantages, returns):
        obs = torch.as_tensor(np.array(self.obs), device=self.device)
        actions = torch.as_tensor(np.array(self.actions), device=self.device, dtype=torch.long)
        old_log_probs = torch.as_tensor(np.array(self.log_probs), device=self.device, dtype=torch.float32)
        advantages = torch.as_tensor(advantages, device=self.device, dtype=torch.float32)
        returns = torch.as_tensor(returns, device=self.device, dtype=torch.float32)
        values = torch.as_tensor(np.array(self.values), device=self.device, dtype=torch.float32)
        return obs, actions, old_log_probs, advantages, returns, values


class PPOAgent:
    def __init__(self, obs_shape: tuple[int, ...], n_actions: int, device: torch.device, cfg: PPOConfig):
        self.device = device
        self.cfg = cfg
        self.model = ActorCritic(obs_shape, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        action, log_prob, _, value = self.model.get_action_and_value(obs_t)
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def act_greedy(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        logits, _ = self.model(obs_t)
        action = torch.argmax(logits, dim=1)
        return int(action.item())

    @torch.no_grad()
    def get_value(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        _, value = self.model(obs_t)
        return float(value.item())

    def update(self, rollout_buffer: RolloutBuffer, last_value: float):
        advantages, returns = rollout_buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        obs, actions, old_log_probs, advantages, returns, old_values = rollout_buffer.get_tensors(
            advantages, returns
        )

        if self.cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = obs.shape[0]
        indices = np.arange(n_samples)

        losses_info = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }
        n_updates = 0

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_idx = indices[start:end]

                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                    obs[mb_idx], actions[mb_idx]
                )

                log_ratio = new_log_probs - old_log_probs[mb_idx]
                ratio = torch.exp(log_ratio)

                pg_loss1 = -advantages[mb_idx] * ratio
                pg_loss2 = -advantages[mb_idx] * torch.clamp(
                    ratio,
                    1.0 - self.cfg.clip_coef,
                    1.0 + self.cfg.clip_coef,
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = F.mse_loss(new_values, returns[mb_idx])
                entropy_loss = entropy.mean()

                total_loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                losses_info["policy_loss"] += float(policy_loss.item())
                losses_info["value_loss"] += float(value_loss.item())
                losses_info["entropy"] += float(entropy_loss.item())
                losses_info["total_loss"] += float(total_loss.item())
                n_updates += 1

        for k in losses_info:
            losses_info[k] /= max(n_updates, 1)

        return losses_info

    def save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])