"""
Entrena el baseline oficial DQN de Stable-Baselines3 para compararlo
contra la implementación propia de DDQN.

Comparación recomendada:
- DDQN propio vs DQN oficial de SB3
- mismo entorno
- mismas semillas
- mismos timesteps
- misma frecuencia de evaluación
- mismo número de episodios de evaluación

Ejemplos:
python train_sb3_dqn.py --env minatar --seed 0 --total_steps 200000
python train_sb3_dqn.py --env atari --seed 0 --total_steps 500000
"""

from __future__ import annotations

import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed

from src.common.envs import make_minatar_breakout, make_atari_breakout


def make_env(env_name: str, seed: int, monitor_path: str | None = None):
    def _thunk():
        if env_name == "minatar":
            env = make_minatar_breakout(seed)
        elif env_name == "atari":
            env = make_atari_breakout(seed)
        else:
            raise ValueError(f"Entorno no soportado: {env_name}")

        env = Monitor(env, filename=monitor_path)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _thunk


class SmallOrNatureCNN(BaseFeaturesExtractor):
    """
    Extractor compatible con:
    - MinAtar: observaciones pequeñas tipo (C, H, W)
    - Atari: observaciones 4x84x84

    Usa una CNN pequeña para MinAtar y una tipo Nature para Atari.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        obs_shape = observation_space.shape
        if len(obs_shape) != 3:
            raise ValueError(
                f"Este extractor espera observaciones 3D (C,H,W). Recibido: {obs_shape}"
            )

        c, h, w = obs_shape

        if h <= 16 and w <= 16:
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(self._preprocess(sample)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        return x

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(observations)
        return self.linear(self.cnn(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["minatar", "atari"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--eval_every", type=int, default=20_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--run_name", type=str, default=None)

    # Hiperparámetros alineados con la implementación propia
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--learning_starts", type=int, default=10_000)
    parser.add_argument("--target_update_interval", type=int, default=1000)
    parser.add_argument("--train_freq", type=int, default=4)

    # Exploración epsilon-greedy
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay_steps", type=int, default=100_000)

    args = parser.parse_args()

    set_random_seed(args.seed)

    run_name = args.run_name or f"sb3_dqn_{args.env}_seed{args.seed}_{int(time.time())}"
    out_dir = os.path.join("results", run_name)
    os.makedirs(out_dir, exist_ok=True)

    train_monitor_path = os.path.join(out_dir, "train_monitor.csv")
    eval_log_path = out_dir
    model_path = os.path.join(out_dir, "model_final")

    train_env = DummyVecEnv([
        make_env(args.env, args.seed, monitor_path=train_monitor_path)
    ])

    eval_env = DummyVecEnv([
        make_env(args.env, args.seed + 10_000)
    ])

    if args.env == "minatar":
        features_dim = 128
    else:
        features_dim = 512

    exploration_fraction = min(1.0, args.eps_decay_steps / float(args.total_steps))

    policy_kwargs = dict(
        features_extractor_class=SmallOrNatureCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        normalize_images=False,
    )

    model = DQN(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=1.0,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=1,
        target_update_interval=args.target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=args.eps_start,
        exploration_final_eps=args.eps_end,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=eval_log_path,
        eval_freq=args.eval_every,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    config = vars(args).copy()
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    model.learn(
        total_timesteps=args.total_steps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(model_path)
    train_env.close()
    eval_env.close()

    print(f"Entrenamiento terminado. Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()