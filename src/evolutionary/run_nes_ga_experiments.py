from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["nes", "ga"], required=True)
    parser.add_argument("--env", type=str, choices=["minatar", "atari"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--generations", type=int, required=True)
    args = parser.parse_args()

    for seed in args.seeds:
        cmd = [
            sys.executable,
            "train_nes_ga.py",
            "--method", args.method,
            "--env", args.env,
            "--seed", str(seed),
            "--generations", str(args.generations),
        ]
        print("Ejecutando:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()