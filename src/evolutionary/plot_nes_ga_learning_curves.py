from __future__ import annotations

import glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--title", type=str, default="NES/GA - Curva de convergencia")
    parser.add_argument("--save_path", type=str, default="nes_ga_curve.png")
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos con el patrón: {args.pattern}")

    dfs = []
    for i, path in enumerate(files):
        df = pd.read_csv(path)
        df["seed_idx"] = i
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    grouped = data.groupby("generation")["eval_mean_return"]
    mean = grouped.mean()
    std = grouped.std().fillna(0.0)

    x = mean.index.values
    y = mean.values
    s = std.values

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="Media")
    plt.fill_between(x, y - s, y + s, alpha=0.25, label="±1 desviación estándar")
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