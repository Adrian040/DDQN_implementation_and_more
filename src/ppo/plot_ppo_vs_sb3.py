"""
Grafica la comparación:
- PPO propio (eval_log.csv)
vs
- PPO oficial de SB3 (evaluations.npz)

Ejemplo:
python plot_ppo_vs_sb3.py \
  --ours_pattern "results/ppo_minatar_seed*/eval_log.csv" \
  --sb3_pattern  "results/sb3_ppo_minatar_seed*/evaluations.npz" \
  --title "PPO propio vs PPO oficial SB3 en MinAtar Breakout" \
  --save_path "compare_minatar_ppo_vs_sb3.png"
"""

from __future__ import annotations

import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_ours(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos para ours_pattern: {pattern}")

    frames = []
    for run_idx, path in enumerate(files):
        df = pd.read_csv(path)
        df = df[["global_step", "eval_mean_return"]].copy()
        df["run_idx"] = run_idx
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_sb3(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos para sb3_pattern: {pattern}")

    frames = []
    for run_idx, path in enumerate(files):
        data = np.load(path)
        timesteps = np.array(data["timesteps"]).reshape(-1)
        results = np.array(data["results"])
        means = results.mean(axis=1)

        df = pd.DataFrame({
            "global_step": timesteps,
            "eval_mean_return": means,
            "run_idx": run_idx,
        })
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame):
    grouped = df.groupby("global_step")["eval_mean_return"]
    mean = grouped.mean()
    std = grouped.std().fillna(0.0)
    return mean.index.values, mean.values, std.values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours_pattern", type=str, required=True)
    parser.add_argument("--sb3_pattern", type=str, required=True)
    parser.add_argument("--title", type=str, default="Comparación PPO vs SB3 PPO")
    parser.add_argument("--save_path", type=str, default="ppo_vs_sb3.png")
    args = parser.parse_args()

    ours = load_ours(args.ours_pattern)
    sb3 = load_sb3(args.sb3_pattern)

    x1, y1, s1 = summarize(ours)
    x2, y2, s2 = summarize(sb3)

    plt.figure(figsize=(8, 5))

    plt.plot(x1, y1, label="PPO propio")
    plt.fill_between(x1, y1 - s1, y1 + s1, alpha=0.25)

    plt.plot(x2, y2, label="PPO oficial SB3")
    plt.fill_between(x2, y2 - s2, y2 + s2, alpha=0.25)

    plt.xlabel("Paso de entrenamiento")
    plt.ylabel("Retorno promedio en evaluación")
    plt.title(args.title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=200)
    print(f"Gráfica guardada en: {args.save_path}")


if __name__ == "__main__":
    main()