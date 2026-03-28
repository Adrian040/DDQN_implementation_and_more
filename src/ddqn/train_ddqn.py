"""
Entrenamiento de DDQN para MinAtar Breakout o Atari Breakout.

Ejemplos de uso:
python train_ddqn.py --env minatar --seed 0 --total_steps 200000
python train_ddqn.py --env atari   --seed 0 --total_steps 500000

Se guardan:
- modelo final
- train_log.csv
- eval_log.csv
"""

from __future__ import annotations

import os
import csv
import time
import argparse
from collections import deque

import numpy as np
import torch

from src.common.envs import make_minatar_breakout, make_atari_breakout
from src.ddqn.ddqn_agent import DDQNAgent, DDQNConfig, set_global_seed


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    if step >= duration:
        return end
    frac = step / float(duration)
    return start + frac * (end - start)


def evaluate(agent: DDQNAgent, env_fn, n_episodes: int, seed: int) -> tuple[float, float]:
    returns = []

    for ep in range(n_episodes):
        env = env_fn(seed + 10_000 + ep)
        obs, _ = env.reset(seed=seed + 10_000 + ep)
        done = False
        ep_return = 0.0

        while not done:
            action = agent.act_greedy(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
            obs = next_obs

        returns.append(ep_return)
        env.close()

    return float(np.mean(returns)), float(np.std(returns))


def make_env(env_name: str, seed: int):
    if env_name == "minatar":
        return make_minatar_breakout(seed)
    elif env_name == "atari":
        return make_atari_breakout(seed)
    raise ValueError(f"Entorno no soportado: {env_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["minatar", "atari"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--eval_every", type=int, default=20_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--run_name", type=str, default=None)

    # Hiperparámetros principales
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--min_buffer_size", type=int, default=10_000)
    parser.add_argument("--target_update_freq", type=int, default=1000)
    parser.add_argument("--train_freq", type=int, default=4)

    # Epsilon-greedy
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay_steps", type=int, default=100_000)

    args = parser.parse_args()

    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.env, args.seed)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    cfg = DDQNConfig(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer_size,
        target_update_freq=args.target_update_freq,
        train_freq=args.train_freq,
    )

    agent = DDQNAgent(obs_shape, n_actions, device, cfg)

    run_name = args.run_name or f"ddqn_{args.env}_seed{args.seed}_{int(time.time())}"
    out_dir = os.path.join("results", run_name)
    os.makedirs(out_dir, exist_ok=True)

    train_csv = os.path.join(out_dir, "train_log.csv")
    eval_csv = os.path.join(out_dir, "eval_log.csv")
    model_path = os.path.join(out_dir, "model_final.pt")

    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["global_step", "episode", "episode_return", "episode_length", "epsilon", "loss"])

    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["global_step", "eval_mean_return", "eval_std_return"])

    obs, _ = env.reset(seed=args.seed)
    episode_idx = 0
    recent_returns = deque(maxlen=20)
    last_loss = np.nan

    for global_step in range(1, args.total_steps + 1):
        epsilon = linear_schedule(
            global_step,
            args.eps_start,
            args.eps_end,
            args.eps_decay_steps,
        )

        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, done)
        obs = next_obs

        if agent.can_update() and global_step % cfg.train_freq == 0:
            last_loss = agent.update()

        if agent.can_update() and global_step % cfg.target_update_freq == 0:
            agent.sync_target()

        if done:
            episode_idx += 1

            ep_info = info.get("episode", None)
            ep_return = ep_info["r"] if ep_info is not None else np.nan
            ep_length = ep_info["l"] if ep_info is not None else np.nan

            recent_returns.append(ep_return)

            with open(train_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([global_step, episode_idx, ep_return, ep_length, epsilon, last_loss])

            if episode_idx % 10 == 0:
                avg20 = np.mean(recent_returns) if len(recent_returns) > 0 else np.nan
                print(
                    f"[{args.env}] step={global_step:>7d} | ep={episode_idx:>4d} "
                    f"| retorno={ep_return:>7.2f} | avg20={avg20:>7.2f} "
                    f"| eps={epsilon:>5.3f} | loss={last_loss:>8.4f}"
                )

            obs, _ = env.reset()

        if global_step % args.eval_every == 0:
            mean_ret, std_ret = evaluate(
                agent,
                lambda s: make_env(args.env, s),
                args.eval_episodes,
                args.seed,
            )

            with open(eval_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([global_step, mean_ret, std_ret])

            print(
                f"[EVAL] env={args.env} | step={global_step} "
                f"| mean_return={mean_ret:.3f} | std_return={std_ret:.3f}"
            )

    agent.save(model_path)
    env.close()
    print(f"Entrenamiento terminado. Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()