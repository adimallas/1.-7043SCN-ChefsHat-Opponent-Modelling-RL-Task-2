#!/usr/bin/env python3
import argparse
from pathlib import Path
from sampath_rl.experiment import train, load_config, CONDITIONS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train genuine Double DQN agents in the official Chef's Hat engine")
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.condition, args.seed, load_config(args.config), resume=args.resume)
