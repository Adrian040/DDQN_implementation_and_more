"""
Entrena el baseline oficial PPO de Stable-Baselines3 para compararlo
contra la implementación propia de PPO.

Ejemplos:
python train_sb3_ppo.py --env minatar --seed 0 --total_steps 200000
python train_sb3_ppo.py --env atari --seed 0 --total_steps 500000
"""

from __future__ import annotations

import os
import json
import time
import argparse

import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3 import PPO
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
    Extractor compatible con MinAtar y Atari.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        obs_shape = observation_space.shape
        if len(obs_shape) != 3:
            raise ValueError(f"Se esperaba observación (C,H,W). Recibido: {obs_shape}")

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

    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=4)

    args = parser.parse_args()

    set_random_seed(args.seed)

    run_name = args.run_name or f"sb3_ppo_{args.env}_seed{args.seed}_{int(time.time())}"
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

    features_dim = 128 if args.env == "minatar" else 512

    policy_kwargs = dict(
        features_extractor_class=SmallOrNatureCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        normalize_images=False,
    )

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
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