"""
Estadística descriptiva y prueba de Wilcoxon rank-sum
para comparar PPO propio vs PPO oficial de SB3.

Compara el rendimiento final de cada corrida.
"""

from __future__ import annotations

import glob
import argparse

import numpy as np
import pandas as pd
from scipy.stats import ranksums


def load_ours_final(pattern: str) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos para ours_pattern: {pattern}")

    finals = []
    for path in files:
        df = pd.read_csv(path)
        finals.append(float(df["eval_mean_return"].iloc[-1]))
    return np.array(finals, dtype=np.float64)


def load_sb3_final(pattern: str) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos para sb3_pattern: {pattern}")

    finals = []
    for path in files:
        data = np.load(path)
        results = np.array(data["results"])
        finals.append(float(results[-1].mean()))
    return np.array(finals, dtype=np.float64)


def describe(x: np.ndarray, name: str) -> dict:
    return {
        "método": name,
        "n": len(x),
        "media": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "mediana": float(np.median(x)),
        "mín": float(np.min(x)),
        "máx": float(np.max(x)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours_pattern", type=str, required=True)
    parser.add_argument("--sb3_pattern", type=str, required=True)
    args = parser.parse_args()

    ours = load_ours_final(args.ours_pattern)
    sb3 = load_sb3_final(args.sb3_pattern)

    desc = pd.DataFrame([
        describe(ours, "PPO propio"),
        describe(sb3, "PPO oficial SB3"),
    ])

    stat, pvalue = ranksums(ours, sb3)

    print("\n=== Estadística descriptiva ===")
    print(desc.to_string(index=False))

    print("\n=== Wilcoxon rank-sum ===")
    print(f"estadístico = {stat:.6f}")
    print(f"p-valor     = {pvalue:.6f}")

    if pvalue < 0.05:
        print("Conclusión: existe diferencia estadísticamente significativa (alpha = 0.05).")
    else:
        print("Conclusión: no se detectó diferencia estadísticamente significativa (alpha = 0.05).")


if __name__ == "__main__":
    main()