"""
Implementación de política neuronal y utilidades para:
- Natural Evolution Strategies (NES) simplificado
- Genetic Algorithm (GA)

La política produce logits para acciones discretas.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PolicyNetwork(nn.Module):
    """
    Política para observaciones vectoriales o tipo imagen (C,H,W).
    Devuelve logits sobre acciones discretas.
    """

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int):
        super().__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        if len(obs_shape) == 1:
            self.feature_extractor = nn.Sequential(
                nn.Linear(obs_shape[0], 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
            )
            feature_dim = 128

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
                # versión más ligera que la de DDQN/PPO para no encarecer tanto NES/GA
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(c, 16, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Flatten(),
                )

            with torch.no_grad():
                dummy = torch.zeros(1, *obs_shape)
                feature_dim = self.feature_extractor(dummy).shape[1]
        else:
            raise ValueError(f"Forma de observación no soportada: {obs_shape}")

        hidden_dim = 128
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        feats = self.feature_extractor(x)
        logits = self.policy_head(feats)
        return logits

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False, device: torch.device | str = "cpu") -> int:
        obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
        logits = self.forward(obs_t)
        if deterministic:
            return int(torch.argmax(logits, dim=1).item())
        dist = Categorical(logits=logits)
        return int(dist.sample().item())


def flatten_params(model: nn.Module) -> np.ndarray:
    """
    Convierte todos los parámetros del modelo en un solo vector numpy.
    """
    return np.concatenate([p.data.detach().cpu().numpy().ravel() for p in model.parameters()])


def set_params_from_vector(model: nn.Module, vector: np.ndarray) -> None:
    """
    Sobrescribe los parámetros del modelo a partir de un vector plano.
    """
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        block = vector[pointer:pointer + numel]
        block = torch.as_tensor(block, dtype=p.data.dtype, device=p.data.device).view_as(p.data)
        p.data.copy_(block)
        pointer += numel


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


@dataclass
class NESConfig:
    population_size: int = 30
    sigma: float = 0.05
    learning_rate: float = 0.02
    use_fitness_shaping: bool = True


@dataclass
class GAConfig:
    population_size: int = 30
    elite_frac: float = 0.2
    mutation_std: float = 0.02
    mutation_rate: float = 0.1
    crossover: bool = False


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """
    Transformación típica de fitness shaping.
    """
    ranks = np.argsort(np.argsort(x)).astype(np.float32)
    ranks /= (len(x) - 1)
    ranks -= 0.5
    return ranks


class NESAgent:
    def __init__(self, model: nn.Module, cfg: NESConfig):
        self.model = model
        self.cfg = cfg
        self.theta = flatten_params(model)

    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Genera perturbaciones antitéticas:
        theta + sigma * eps
        theta - sigma * eps
        """
        half = self.cfg.population_size // 2
        eps = np.random.randn(half, self.theta.size).astype(np.float32)
        eps_full = np.concatenate([eps, -eps], axis=0)

        if eps_full.shape[0] < self.cfg.population_size:
            extra = np.random.randn(1, self.theta.size).astype(np.float32)
            eps_full = np.concatenate([eps_full, extra], axis=0)

        population = self.theta[None, :] + self.cfg.sigma * eps_full
        return population, eps_full

    def tell(self, rewards: np.ndarray, eps_full: np.ndarray):
        if self.cfg.use_fitness_shaping:
            rewards_used = centered_ranks(rewards)
        else:
            rewards_used = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        grad_estimate = (eps_full.T @ rewards_used) / (len(rewards_used) * self.cfg.sigma)
        self.theta = self.theta + self.cfg.learning_rate * grad_estimate
        set_params_from_vector(self.model, self.theta)


class GAAgent:
    def __init__(self, base_model: nn.Module, cfg: GAConfig):
        self.base_model = base_model
        self.cfg = cfg
        self.population = [flatten_params(clone_model(base_model)) for _ in range(cfg.population_size)]

        # inicialización ligera alrededor del modelo base
        for i in range(cfg.population_size):
            self.population[i] = self.population[i] + 0.01 * np.random.randn(*self.population[i].shape).astype(np.float32)

        self.best_params = self.population[0].copy()
        self.best_reward = -np.inf

    def ask(self) -> List[np.ndarray]:
        return self.population

    def tell(self, rewards: np.ndarray):
        idx_sorted = np.argsort(rewards)[::-1]
        self.population = [self.population[i] for i in idx_sorted]
        rewards_sorted = rewards[idx_sorted]

        if rewards_sorted[0] > self.best_reward:
            self.best_reward = float(rewards_sorted[0])
            self.best_params = self.population[0].copy()

        n_elite = max(1, int(self.cfg.elite_frac * self.cfg.population_size))
        elites = self.population[:n_elite]

        new_population = [elite.copy() for elite in elites]

        while len(new_population) < self.cfg.population_size:
            parent = elites[np.random.randint(n_elite)].copy()

            if self.cfg.crossover and n_elite >= 2:
                parent2 = elites[np.random.randint(n_elite)].copy()
                mask = np.random.rand(parent.size) < 0.5
                child = np.where(mask, parent, parent2)
            else:
                child = parent

            mutation_mask = (np.random.rand(child.size) < self.cfg.mutation_rate).astype(np.float32)
            noise = self.cfg.mutation_std * np.random.randn(child.size).astype(np.float32)
            child = child + mutation_mask * noise
            new_population.append(child)

        self.population = new_population[:self.cfg.population_size]

    def get_best(self) -> np.ndarray:
        return self.best_params.copy()