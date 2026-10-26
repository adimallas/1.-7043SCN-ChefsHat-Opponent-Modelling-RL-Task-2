#!/usr/bin/env python3
import argparse
from sampath_rl.experiment import evaluate, load_config, AGENTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate frozen policies over held-out seeds and all seats")
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument("--agent", choices=AGENTS, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--features-off", action="store_true", help="Same aware checkpoint, public-behaviour inputs zeroed")
    args = parser.parse_args()
    if args.agent.startswith("dqn") and args.seed is None:
        parser.error("DQN evaluation requires --seed for the checkpoint")
    evaluate(args.agent, load_config(args.config), training_seed=args.seed, aware_override=False if args.features_off else None)
