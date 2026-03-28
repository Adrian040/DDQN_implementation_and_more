"""
Lanza varias semillas de DDQN.

Ejemplos:
python run_ddqn_experiments.py --env minatar --seeds 0 1 2 3 4 --total_steps 200000
python run_ddqn_experiments.py --env atari   --seeds 0 1 2 3 4 --total_steps 500000
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["minatar", "atari"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--total_steps", type=int, required=True)
    args = parser.parse_args()

    for seed in args.seeds:
        cmd = [
            sys.executable,
            "train_ddqn.py",
            "--env", args.env,
            "--seed", str(seed),
            "--total_steps", str(args.total_steps),
        ]
        print("Ejecutando:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()