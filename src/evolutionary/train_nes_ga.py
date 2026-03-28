"""
Entrenamiento de NES o GA para:
- MinAtar Breakout
- Atari Breakout

Ejemplos:
python train_nes_ga.py --method nes --env minatar --seed 0 --generations 200
python train_nes_ga.py --method ga  --env minatar --seed 0 --generations 200
python train_nes_ga.py --method nes --env atari   --seed 0 --generations 100
"""

from __future__ import annotations

import os
import csv
import time
import argparse

import numpy as np
import torch

from src.common.envs import make_minatar_breakout, make_atari_breakout
from src.evolutionary.nes_ga_agent import (
    set_global_seed,
    PolicyNetwork,
    set_params_from_vector,
    clone_model,
    NESAgent,
    GAAgent,
    NESConfig,
    GAConfig,
)

def make_env(env_name: str, seed: int):
    if env_name == "minatar":
        return make_minatar_breakout(seed)
    elif env_name == "atari":
        return make_atari_breakout(seed)
    raise ValueError(f"Entorno no soportado: {env_name}")


def evaluate_policy_vector(
    param_vector: np.ndarray,
    base_model: PolicyNetwork,
    env_name: str,
    seed: int,
    n_episodes: int,
    deterministic: bool,
    device: torch.device,
) -> float:
    model = clone_model(base_model).to(device)
    set_params_from_vector(model, param_vector)
    model.eval()

    returns = []

    for ep in range(n_episodes):
        env = make_env(env_name, seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            action = model.act(obs, deterministic=deterministic, device=device)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)
        env.close()

    return float(np.mean(returns))


def evaluate_current_model(
    model: PolicyNetwork,
    env_name: str,
    seed: int,
    n_episodes: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    returns = []

    for ep in range(n_episodes):
        env = make_env(env_name, seed + 10_000 + ep)
        obs, _ = env.reset(seed=seed + 10_000 + ep)
        done = False
        ep_return = 0.0

        while not done:
            action = model.act(obs, deterministic=True, device=device)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)
        env.close()

    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["nes", "ga"], required=True)
    parser.add_argument("--env", type=str, choices=["minatar", "atari"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--episodes_per_candidate", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--run_name", type=str, default=None)

    # NES
    parser.add_argument("--population_size", type=int, default=30)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=0.02)

    # GA
    parser.add_argument("--elite_frac", type=float, default=0.2)
    parser.add_argument("--mutation_std", type=float, default=0.02)
    parser.add_argument("--mutation_rate", type=float, default=0.1)
    parser.add_argument("--crossover", action="store_true")

    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(args.env, args.seed)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    env.close()

    base_model = PolicyNetwork(obs_shape, n_actions).to(device)

    if args.method == "nes":
        agent = NESAgent(
            base_model,
            NESConfig(
                population_size=args.population_size,
                sigma=args.sigma,
                learning_rate=args.learning_rate,
                use_fitness_shaping=True,
            ),
        )
    else:
        agent = GAAgent(
            base_model,
            GAConfig(
                population_size=args.population_size,
                elite_frac=args.elite_frac,
                mutation_std=args.mutation_std,
                mutation_rate=args.mutation_rate,
                crossover=args.crossover,
            ),
        )

    run_name = args.run_name or f"{args.method}_{args.env}_seed{args.seed}_{int(time.time())}"
    out_dir = os.path.join("results", run_name)
    os.makedirs(out_dir, exist_ok=True)

    train_csv = os.path.join(out_dir, "train_log.csv")
    eval_csv = os.path.join(out_dir, "eval_log.csv")
    model_path = os.path.join(out_dir, "best_model.pt")

    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation", "mean_population_return", "std_population_return",
            "best_population_return", "global_interactions_est"
        ])

    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "eval_mean_return", "eval_std_return"])

    best_global_return = -np.inf
    best_global_params = None
    global_interactions_est = 0

    for gen in range(1, args.generations + 1):
        if args.method == "nes":
            population, eps_full = agent.ask()
        else:
            population = agent.ask()

        rewards = []

        for i, params in enumerate(population):
            reward = evaluate_policy_vector(
                param_vector=params,
                base_model=base_model,
                env_name=args.env,
                seed=args.seed + gen * 1000 + i * 10,
                n_episodes=args.episodes_per_candidate,
                deterministic=False,
                device=device,
            )
            rewards.append(reward)

        rewards = np.array(rewards, dtype=np.float32)

        if args.method == "nes":
            agent.tell(rewards, eps_full)
            current_params = agent.theta.copy()
        else:
            agent.tell(rewards)
            current_params = agent.get_best()

        if rewards.max() > best_global_return:
            best_global_return = float(rewards.max())
            best_global_params = population[int(np.argmax(rewards))].copy()

        # estimación simple del número de interacciones
        global_interactions_est += len(population) * args.episodes_per_candidate

        with open(train_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                gen,
                float(rewards.mean()),
                float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
                float(rewards.max()),
                global_interactions_est,
            ])

        print(
            f"[{args.method.upper()}][{args.env}] gen={gen:>4d} "
            f"| mean_fit={rewards.mean():>7.3f} "
            f"| best_fit={rewards.max():>7.3f}"
        )

        if gen % args.eval_every == 0:
            set_params_from_vector(base_model, current_params)
            mean_ret, std_ret = evaluate_current_model(
                base_model,
                args.env,
                args.seed,
                args.eval_episodes,
                device,
            )

            with open(eval_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([gen, mean_ret, std_ret])

            print(
                f"[EVAL] method={args.method} | env={args.env} | gen={gen} "
                f"| mean_return={mean_ret:.3f} | std_return={std_ret:.3f}"
            )

    if best_global_params is not None:
        set_params_from_vector(base_model, best_global_params)

    torch.save(
        {"model_state_dict": base_model.state_dict()},
        model_path,
    )

    print(f"Entrenamiento terminado. Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()