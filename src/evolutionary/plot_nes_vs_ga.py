"""
Grafica la comparación:
- NES (eval_log.csv)
vs
- GA  (eval_log.csv)

Ejemplo:
python -m src.evolutionary.plot_nes_vs_ga \
  --nes_pattern "results/nes_minatar_seed*/eval_log.csv" \
  --ga_pattern  "results/ga_minatar_seed*/eval_log.csv" \
  --title "NES vs GA en MinAtar Breakout" \
  --save_path "compare_minatar_nes_vs_ga.png"
"""

from __future__ import annotations

import glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def load_runs(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos con el patrón: {pattern}")

    dfs = []
    for i, path in enumerate(files):
        df = pd.read_csv(path)

        if "generation" not in df.columns or "eval_mean_return" not in df.columns:
            raise ValueError(
                f"El archivo {path} no contiene las columnas esperadas "
                f"'generation' y 'eval_mean_return'."
            )

        df = df[["generation", "eval_mean_return"]].copy()
        df["run_idx"] = i
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def summarize(df: pd.DataFrame):
    grouped = df.groupby("generation")["eval_mean_return"]
    mean = grouped.mean()
    std = grouped.std().fillna(0.0)
    return mean.index.values, mean.values, std.values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nes_pattern", type=str, required=True)
    parser.add_argument("--ga_pattern", type=str, required=True)
    parser.add_argument("--title", type=str, default="Comparación NES vs GA")
    parser.add_argument("--save_path", type=str, default="nes_vs_ga.png")
    args = parser.parse_args()

    nes = load_runs(args.nes_pattern)
    ga = load_runs(args.ga_pattern)

    x1, y1, s1 = summarize(nes)
    x2, y2, s2 = summarize(ga)

    plt.figure(figsize=(8, 5))

    plt.plot(x1, y1, label="NES")
    plt.fill_between(x1, y1 - s1, y1 + s1, alpha=0.25)

    plt.plot(x2, y2, label="GA")
    plt.fill_between(x2, y2 - s2, y2 + s2, alpha=0.25)

    plt.xlabel("Generación")
    plt.ylabel("Retorno promedio en evaluación")
    plt.title(args.title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=200)
    print(f"Gráfica guardada en: {args.save_path}")


if __name__ == "__main__":
    main()