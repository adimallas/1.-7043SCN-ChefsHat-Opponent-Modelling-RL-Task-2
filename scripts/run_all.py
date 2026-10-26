#!/usr/bin/env python3
"""Finite current-task experiment pipeline, not an unattended service."""
import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def call(command):
    subprocess.run([sys.executable, *command], cwd=ROOT, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text())
    conditions = ["dqn_random", "dqn_mixed", "dqn_aware"]
    jobs = [["scripts/train.py", "--config", args.config, "--condition", c, "--seed", str(s)] + (["--resume"] if args.resume else []) for c in conditions for s in config["training_seeds"]]
    if not args.evaluation_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(call, jobs))
    jobs = [["scripts/evaluate.py", "--config", args.config, "--agent", a] for a in ["random", "heuristic"]]
    jobs += [["scripts/evaluate.py", "--config", args.config, "--agent", c, "--seed", str(s)] for c in conditions for s in config["training_seeds"]]
    jobs += [["scripts/evaluate.py", "--config", args.config, "--agent", "dqn_aware", "--seed", str(s), "--features-off"] for s in config["training_seeds"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(call, jobs))
    print("All requested training, evaluation, and inference ablations completed.", flush=True)
