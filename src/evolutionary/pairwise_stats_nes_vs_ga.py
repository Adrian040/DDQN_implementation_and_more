"""
Estadística descriptiva y prueba de Wilcoxon rank-sum
para comparar NES vs GA.

Compara el rendimiento final de cada corrida:
- NES: último eval_mean_return de cada eval_log.csv
- GA:  último eval_mean_return de cada eval_log.csv

Ejemplo:
python -m src.evolutionary.pairwise_stats_nes_vs_ga \
  --nes_pattern "results/nes_minatar_seed*/eval_log.csv" \
  --ga_pattern  "results/ga_minatar_seed*/eval_log.csv"
"""

from __future__ import annotations

import glob
import argparse

import numpy as np
import pandas as pd
from scipy.stats import ranksums


def load_final(pattern: str) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos con el patrón: {pattern}")

    finals = []
    for path in files:
        df = pd.read_csv(path)

        if "eval_mean_return" not in df.columns:
            raise ValueError(f"El archivo {path} no contiene la columna 'eval_mean_return'.")

        if len(df) == 0:
            raise ValueError(f"El archivo {path} está vacío.")

        finals.append(float(df["eval_mean_return"].iloc[-1]))

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
    parser.add_argument("--nes_pattern", type=str, required=True)
    parser.add_argument("--ga_pattern", type=str, required=True)
    args = parser.parse_args()

    nes = load_final(args.nes_pattern)
    ga = load_final(args.ga_pattern)

    desc = pd.DataFrame([
        describe(nes, "NES"),
        describe(ga, "GA"),
    ])

    stat, pvalue = ranksums(nes, ga)

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