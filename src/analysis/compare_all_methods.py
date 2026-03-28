"""
Comparación global de métodos para la tarea de RL.

Métodos esperados:
- ddqn
- sb3_dqn
- ppo
- sb3_ppo
- nes
- ga

Entornos esperados:
- minatar
- atari

Este script genera:
1) tabla consolidada de resultados finales
2) estadística descriptiva por método y entorno
3) comparaciones por pares con Wilcoxon rank-sum
4) ranking promedio
5) diagrama de diferencias críticas aproximado
6) gráfica conjunta de convergencia

Uso típico:
python compare_all_methods.py --results_dir results --output_dir analysis_outputs
"""

from __future__ import annotations

import os
import glob
import argparse
import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ranksums

# scikit-posthocs se usará si está disponible
try:
    import scikit_posthocs as sp
    HAS_SCPH = True
except ImportError:
    HAS_SCPH = False


# ============================================================
# Utilidades de carga
# ============================================================

def parse_run_name(run_dir_name: str) -> Tuple[str, str]:
    """
    Extrae método y entorno a partir del nombre de carpeta.

    Convenciones esperadas:
    - ddqn_minatar_seed0_...
    - sb3_dqn_minatar_seed0_...
    - ppo_atari_seed1_...
    - sb3_ppo_atari_seed2_...
    - nes_minatar_seed3_...
    - ga_atari_seed4_...

    Regresa: (method, env)
    """
    prefixes = [
        "sb3_dqn",
        "sb3_ppo",
        "ddqn",
        "ppo",
        "nes",
        "ga",
    ]

    for prefix in prefixes:
        if run_dir_name.startswith(prefix + "_"):
            rest = run_dir_name[len(prefix) + 1:]
            if rest.startswith("minatar_"):
                return prefix, "minatar"
            elif rest.startswith("atari_"):
                return prefix, "atari"

    raise ValueError(f"No se pudo inferir método/entorno desde: {run_dir_name}")


def extract_seed(run_dir_name: str) -> int:
    """
    Busca el patrón 'seedX' en el nombre de carpeta.
    """
    parts = run_dir_name.split("_")
    for p in parts:
        if p.startswith("seed"):
            return int(p.replace("seed", ""))
    raise ValueError(f"No se encontró la semilla en: {run_dir_name}")


def load_final_score_from_run(run_dir: str) -> float:
    """
    Carga el score final desde una corrida:
    - eval_log.csv para implementaciones propias y NES/GA
    - evaluations.npz para SB3
    """
    eval_csv = os.path.join(run_dir, "eval_log.csv")
    eval_npz = os.path.join(run_dir, "evaluations.npz")

    if os.path.isfile(eval_csv):
        df = pd.read_csv(eval_csv)
        if len(df) == 0:
            raise ValueError(f"Archivo vacío: {eval_csv}")
        return float(df["eval_mean_return"].iloc[-1])

    if os.path.isfile(eval_npz):
        data = np.load(eval_npz)
        results = np.array(data["results"])
        return float(results[-1].mean())

    raise FileNotFoundError(f"No se encontró eval_log.csv ni evaluations.npz en: {run_dir}")


def load_curve_from_run(run_dir: str) -> pd.DataFrame:
    """
    Regresa un DataFrame con columnas:
    - x
    - y

    Para DDQN/PPO:
      x = global_step, y = eval_mean_return
    Para NES/GA:
      x = generation, y = eval_mean_return
    Para SB3:
      x = timesteps, y = mean(results)
    """
    eval_csv = os.path.join(run_dir, "eval_log.csv")
    eval_npz = os.path.join(run_dir, "evaluations.npz")

    if os.path.isfile(eval_csv):
        df = pd.read_csv(eval_csv)

        if "global_step" in df.columns:
            return pd.DataFrame({
                "x": df["global_step"].values,
                "y": df["eval_mean_return"].values,
            })

        if "generation" in df.columns:
            return pd.DataFrame({
                "x": df["generation"].values,
                "y": df["eval_mean_return"].values,
            })

        raise ValueError(f"Formato no reconocido en {eval_csv}")

    if os.path.isfile(eval_npz):
        data = np.load(eval_npz)
        timesteps = np.array(data["timesteps"]).reshape(-1)
        results = np.array(data["results"])
        means = results.mean(axis=1)
        return pd.DataFrame({"x": timesteps, "y": means})

    raise FileNotFoundError(f"No se encontró curva en: {run_dir}")


def collect_all_runs(results_dir: str) -> pd.DataFrame:
    """
    Recorre todas las carpetas de resultados y construye un DataFrame
    consolidado de scores finales.
    """
    run_dirs = [
        os.path.join(results_dir, d)
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    rows = []
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)

        try:
            method, env = parse_run_name(run_name)
            seed = extract_seed(run_name)
            final_score = load_final_score_from_run(run_dir)
        except Exception:
            continue

        rows.append({
            "run_name": run_name,
            "run_dir": run_dir,
            "method": method,
            "env": env,
            "seed": seed,
            "final_score": final_score,
        })

    if not rows:
        raise RuntimeError("No se encontraron corridas válidas en results_dir.")

    return pd.DataFrame(rows)


# ============================================================
# Estadística descriptiva y Wilcoxon
# ============================================================

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["env", "method"])["final_score"]

    rows = []
    for (env, method), x in grouped:
        x = np.array(x, dtype=np.float64)
        rows.append({
            "env": env,
            "method": method,
            "n": len(x),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "median": float(np.median(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        })

    return pd.DataFrame(rows).sort_values(["env", "mean"], ascending=[True, False])


def pairwise_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wilcoxon rank-sum por pares dentro de cada entorno.
    """
    rows = []

    for env in sorted(df["env"].unique()):
        df_env = df[df["env"] == env]
        methods = sorted(df_env["method"].unique())

        for m1, m2 in itertools.combinations(methods, 2):
            x1 = df_env[df_env["method"] == m1]["final_score"].values
            x2 = df_env[df_env["method"] == m2]["final_score"].values

            if len(x1) == 0 or len(x2) == 0:
                continue

            stat, pvalue = ranksums(x1, x2)

            rows.append({
                "env": env,
                "method_1": m1,
                "method_2": m2,
                "n1": len(x1),
                "n2": len(x2),
                "statistic": float(stat),
                "pvalue": float(pvalue),
                "significant_0.05": bool(pvalue < 0.05),
            })

    return pd.DataFrame(rows)


# ============================================================
# Ranking y diferencias críticas
# ============================================================

def compute_problem_seed_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula rangos por combinación (env, seed).
    Mayor final_score = mejor = rango 1.
    """
    rows = []

    for (env, seed), group in df.groupby(["env", "seed"]):
        group = group.copy()
        group["rank"] = group["final_score"].rank(ascending=False, method="average")

        for _, row in group.iterrows():
            rows.append({
                "env": env,
                "seed": seed,
                "method": row["method"],
                "rank": float(row["rank"]),
                "final_score": float(row["final_score"]),
            })

    return pd.DataFrame(rows)


def average_ranks(rank_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        rank_df.groupby("method")["rank"]
        .mean()
        .reset_index()
        .rename(columns={"rank": "avg_rank"})
        .sort_values("avg_rank", ascending=True)
    )
    return out


def nemenyi_cd(k: int, N: int, q_alpha: float = 2.85) -> float:
    """
    Aproximación de diferencia crítica para Nemenyi.
    q_alpha ~ 2.85 es una aproximación razonable para alpha=0.05 y varios k.
    Para una tarea, esta aproximación suele ser suficiente si se documenta.

    CD = q_alpha * sqrt(k(k+1)/(6N))
    """
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))


def plot_critical_difference(avg_rank_df: pd.DataFrame, cd: float, save_path: str):
    """
    Dibuja un diagrama simple de diferencias críticas.
    """
    methods = avg_rank_df["method"].tolist()
    ranks = avg_rank_df["avg_rank"].tolist()

    k = len(methods)
    min_rank = 1.0
    max_rank = float(max(k, np.ceil(max(ranks) + 0.5)))

    plt.figure(figsize=(10, 2.8))
    ax = plt.gca()

    # Línea base
    ax.hlines(1.0, min_rank, max_rank, linewidth=1.5)

    # Ticks de rango
    for r in range(1, int(max_rank) + 1):
        ax.vlines(r, 0.95, 1.05, linewidth=1.0)
        ax.text(r, 1.08, str(r), ha="center", va="bottom", fontsize=10)

    # Métodos
    y0 = 0.75
    dy = 0.12
    for i, (m, r) in enumerate(zip(methods, ranks)):
        y = y0 - i * dy
        ax.plot([r, r], [1.0, y], linewidth=1.0)
        ax.scatter([r], [1.0], s=35)
        ax.text(r + 0.03, y, f"{m} ({r:.2f})", va="center", fontsize=10)

    # Barra de CD
    cd_start = min_rank
    cd_end = min_rank + cd
    ax.plot([cd_start, cd_end], [1.28, 1.28], linewidth=2.0)
    ax.vlines([cd_start, cd_end], 1.24, 1.32, linewidth=2.0)
    ax.text((cd_start + cd_end) / 2, 1.34, f"CD = {cd:.3f}", ha="center", fontsize=10)

    ax.set_xlim(min_rank - 0.2, max_rank + 0.8)
    ax.set_ylim(0.0, 1.5)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    print(f"Diagrama de diferencias críticas guardado en: {save_path}")


# ============================================================
# Gráficas conjuntas
# ============================================================

PRETTY_NAMES = {
    "ddqn": "DDQN propio",
    "sb3_dqn": "DQN oficial SB3",
    "ppo": "PPO propio",
    "sb3_ppo": "PPO oficial SB3",
    "nes": "NES",
    "ga": "GA",
}


def plot_combined_curves(df_runs: pd.DataFrame, env: str, save_path: str):
    """
    Grafica curvas promedio por método para un entorno dado.
    """
    methods = sorted(df_runs[df_runs["env"] == env]["method"].unique())

    plt.figure(figsize=(9, 5.5))

    for method in methods:
        run_dirs = df_runs[(df_runs["env"] == env) & (df_runs["method"] == method)]["run_dir"].tolist()
        if not run_dirs:
            continue

        curves = []
        for run_dir in run_dirs:
            curve = load_curve_from_run(run_dir)
            curve["run_id"] = os.path.basename(run_dir)
            curves.append(curve)

        data = pd.concat(curves, ignore_index=True)
        grouped = data.groupby("x")["y"]
        mean = grouped.mean()
        std = grouped.std().fillna(0.0)

        x = mean.index.values
        y = mean.values
        s = std.values

        label = PRETTY_NAMES.get(method, method)
        plt.plot(x, y, label=label)
        plt.fill_between(x, y - s, y + s, alpha=0.18)

    xlabel = "Paso de entrenamiento"
    if env == "minatar":
        xlabel = "Paso / generación"
    elif env == "atari":
        xlabel = "Paso / generación"

    plt.xlabel(xlabel)
    plt.ylabel("Retorno promedio en evaluación")
    plt.title(f"Comparación global de métodos en {env}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    print(f"Gráfica conjunta guardada en: {save_path}")


# ============================================================
# Posthoc opcional con Friedman/Nemenyi
# ============================================================

def run_friedman_nemenyi_if_possible(rank_df: pd.DataFrame, output_dir: str):
    """
    Si scikit-posthocs está disponible, construye una matriz problema x método
    para posthoc de Friedman/Nemenyi.
    """
    if not HAS_SCPH:
        print("scikit-posthocs no está instalado; se omite posthoc Friedman/Nemenyi.")
        return

    # Cada problema será (env, seed)
    pivot = rank_df.pivot_table(
        index=["env", "seed"],
        columns="method",
        values="final_score",
        aggfunc="mean",
    )

    pivot = pivot.dropna(axis=0, how="any")
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        print("No hay suficientes datos completos para Friedman/Nemenyi.")
        return

    # Nemenyi posthoc
    nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
    nemenyi.index = pivot.columns
    nemenyi.columns = pivot.columns

    out_path = os.path.join(output_dir, "nemenyi_pvalues.csv")
    nemenyi.to_csv(out_path, encoding="utf-8")
    print(f"Matriz de p-valores Nemenyi guardada en: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="analysis_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Carga consolidada
    df = collect_all_runs(args.results_dir)
    df.to_csv(os.path.join(args.output_dir, "all_final_scores.csv"), index=False, encoding="utf-8")
    print("Tabla consolidada guardada.")

    # 2) Estadística descriptiva
    desc = descriptive_stats(df)
    desc.to_csv(os.path.join(args.output_dir, "descriptive_stats.csv"), index=False, encoding="utf-8")
    print("Estadística descriptiva guardada.")

    # 3) Wilcoxon rank-sum por pares
    pairwise = pairwise_wilcoxon(df)
    pairwise.to_csv(os.path.join(args.output_dir, "pairwise_wilcoxon.csv"), index=False, encoding="utf-8")
    print("Comparaciones por pares guardadas.")

    # 4) Ranking promedio
    rank_df = compute_problem_seed_ranks(df)
    rank_df.to_csv(os.path.join(args.output_dir, "problem_seed_ranks.csv"), index=False, encoding="utf-8")

    avg_rank_df = average_ranks(rank_df)
    avg_rank_df.to_csv(os.path.join(args.output_dir, "average_ranks.csv"), index=False, encoding="utf-8")
    print("Ranking promedio guardado.")

    # 5) Diferencias críticas
    k = avg_rank_df.shape[0]
    N = rank_df.groupby(["env", "seed"]).ngroups
    cd = nemenyi_cd(k=k, N=N, q_alpha=2.85)
    plot_critical_difference(
        avg_rank_df,
        cd=cd,
        save_path=os.path.join(args.output_dir, "critical_difference.png"),
    )

    # 6) Curvas conjuntas por entorno
    for env in sorted(df["env"].unique()):
        plot_combined_curves(
            df_runs=df,
            env=env,
            save_path=os.path.join(args.output_dir, f"combined_curves_{env}.png"),
        )

    # 7) Posthoc opcional
    run_friedman_nemenyi_if_possible(rank_df, args.output_dir)

    print("\nResumen rápido:")
    print(desc.to_string(index=False))
    print("\nRanking promedio:")
    print(avg_rank_df.to_string(index=False))


if __name__ == "__main__":
    main()