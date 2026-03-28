"""
Entrenamiento de PPO para MinAtar Breakout o Atari Breakout.

Ejemplos:
python train_ppo.py --env minatar --seed 0 --total_steps 200000
python train_ppo.py --env atari --seed 0 --total_steps 500000
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
from src.ppo.ppo_agent import PPOAgent, PPOConfig, RolloutBuffer, set_global_seed


def make_env(env_name: str, seed: int):
    if env_name == "minatar":
        return make_minatar_breakout(seed)
    elif env_name == "atari":
        return make_atari_breakout(seed)
    raise ValueError(f"Entorno no soportado: {env_name}")


def evaluate(agent: PPOAgent, env_fn, n_episodes: int, seed: int):
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
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--rollout_steps", type=int, default=2048)

    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(args.env, args.seed)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    cfg = PPOConfig(
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        rollout_steps=args.rollout_steps,
    )

    agent = PPOAgent(obs_shape, n_actions, device, cfg)

    run_name = args.run_name or f"ppo_{args.env}_seed{args.seed}_{int(time.time())}"
    out_dir = os.path.join("results", run_name)
    os.makedirs(out_dir, exist_ok=True)

    train_csv = os.path.join(out_dir, "train_log.csv")
    eval_csv = os.path.join(out_dir, "eval_log.csv")
    model_path = os.path.join(out_dir, "model_final.pt")

    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "global_step", "update", "mean_train_return",
            "policy_loss", "value_loss", "entropy", "total_loss"
        ])

    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["global_step", "eval_mean_return", "eval_std_return"])

    obs, _ = env.reset(seed=args.seed)
    global_step = 0
    update_idx = 0
    episode_returns = deque(maxlen=20)

    while global_step < args.total_steps:
        rollout = RolloutBuffer(cfg.rollout_steps, obs_shape, device)

        for _ in range(cfg.rollout_steps):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rollout.add(obs, action, log_prob, reward, done, value)
            obs = next_obs
            global_step += 1

            if done:
                ep_info = info.get("episode", None)
                if ep_info is not None:
                    episode_returns.append(ep_info["r"])
                obs, _ = env.reset()

            if global_step >= args.total_steps:
                break

        last_value = 0.0 if done else agent.get_value(obs)
        losses = agent.update(rollout, last_value)
        update_idx += 1

        mean_train_return = float(np.mean(episode_returns)) if len(episode_returns) > 0 else np.nan

        with open(train_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                global_step,
                update_idx,
                mean_train_return,
                losses["policy_loss"],
                losses["value_loss"],
                losses["entropy"],
                losses["total_loss"],
            ])

        print(
            f"[{args.env}] step={global_step:>7d} | upd={update_idx:>4d} "
            f"| avg20={mean_train_return:>7.2f} | "
            f"pi_loss={losses['policy_loss']:.4f} | "
            f"v_loss={losses['value_loss']:.4f}"
        )

        if global_step % args.eval_every < cfg.rollout_steps:
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