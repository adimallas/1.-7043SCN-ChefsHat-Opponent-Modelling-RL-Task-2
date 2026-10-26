"""Training and evaluation protocol; all scores originate from official games."""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import yaml
from .agents import DQNAgent, RandomAgent, HeuristicAgent
from .core import SingleAgentEnv
from .policy import progress_mask, ProgressPolicy

CONDITIONS = ("dqn_random", "dqn_mixed", "dqn_aware")
AGENTS = ("random", "heuristic", *CONDITIONS)
OPPONENTS = ("random", "aggressive", "conservative", "mixed", "switching")
ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_opponents(kind, seed):
    styles = ["random"] * 3 if kind == "random" else (["random", "aggressive", "conservative"] if kind == "mixed" else [kind] * 3)
    # Rotate mixed style positions deterministically; not conditioned on learner.
    styles = styles[seed % 3:] + styles[:seed % 3]
    return [ProgressPolicy(RandomAgent(seed=seed + 101 * (i + 1)) if style == "random" else HeuristicAgent(seed=seed + 101 * (i + 1), strategy=style)) for i, style in enumerate(styles)]


def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def row_metrics(info):
    keys = ["position", "score", "performance_score", "actions", "learner_actions", "passes", "invalid_actions", "rounds", "duration_seconds"]
    metrics = {k: info.get(k) for k in keys}
    if isinstance(metrics["actions"], (list, tuple, np.ndarray)):
        metrics["actions"] = int(sum(metrics["actions"]))
    return metrics


def output_dirs(config):
    if config["preset"] == "quick":
        return ROOT / "results" / "quick", ROOT / "checkpoints" / "quick"
    return ROOT / "results", ROOT / "checkpoints"


def train(condition, seed, config, resume=False):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    out, checkpoints = output_dirs(config)
    checkpoints.mkdir(parents=True, exist_ok=True)
    run = f"{condition}_seed{seed}"
    checkpoint = checkpoints / f"{run}.pt"
    train_path = out / "training" / f"{run}.csv"
    loss_path = out / "training" / f"{run}_loss.csv"
    meta_path = out / "training" / f"{run}_metadata.json"
    if train_path.exists() and not resume:
        raise FileExistsError(f"Run already exists: {run}; use --resume or a fresh output directory")
    agent = DQNAgent.load(checkpoint, load_replay=True) if resume and checkpoint.exists() else DQNAgent(seed=seed, config=config["network"])
    previous = json.loads(meta_path.read_text()) if resume and meta_path.exists() else {}
    start_episode = int(previous.get("completed_episodes", 0))
    if start_episode and not checkpoint.exists():
        raise RuntimeError("Cannot resume without checkpoint")
    if resume and previous.get("config") != config:
        raise ValueError("Resume must use the same recorded configuration")
    if start_episode == 0 and resume and train_path.exists():
        raise RuntimeError("Results exist without a resumable checkpoint")
    # Metadata and CSV must refer to the exact same checkpoint boundary.
    if start_episode:
        import pandas as pd
        for path, column, bound in [(train_path, "episode", start_episode), (loss_path, "step", agent.steps)]:
            if path.exists():
                frame = pd.read_csv(path)
                frame = frame[frame[column] <= bound]
                frame.to_csv(path, index=False)
    start = time.perf_counter()
    rewards = []
    total = config["train_episodes"]
    metadata = {"condition": condition, "training_seed": seed, "config": config, "started_utc": previous.get("started_utc", datetime.now(timezone.utc).isoformat()), "completed_episodes": start_episode, "duration_seconds": previous.get("duration_seconds", 0), "status": "running"}
    for episode in range(start_episode, total):
        env_seed = seed * 100000 + episode
        kind = "random" if condition == "dqn_random" else "mixed"
        env = SingleAgentEnv(make_opponents(kind, env_seed), seat=episode % 4, aware=condition == "dqn_aware", max_actions=config["max_actions"], shaping=config["shaping"])
        state, mask = env.reset(seed=env_seed)
        mask = progress_mask(state, mask)
        decay = max(1, total * config["epsilon_decay_fraction"])
        epsilon = config["epsilon_end"] + (config["epsilon_start"] - config["epsilon_end"]) * max(0, 1 - episode / decay)
        episode_reward = 0.0
        while True:
            action = agent.act(state, mask, epsilon=epsilon)
            next_state, reward, terminal, truncated, info = env.step(action)
            if truncated:
                raise RuntimeError(f"Training game exceeded diagnostic bound: {run}, episode {episode}, {info}")
            info["action_mask"] = progress_mask(next_state, info["action_mask"])
            loss = agent.observe(state, action, reward, next_state, next_mask=info["action_mask"], done=terminal, mask=mask)
            if loss is not None:
                append_row(loss_path, {"episode": episode + 1, "step": agent.steps, "loss": loss})
            episode_reward += reward
            state, mask = next_state, info["action_mask"]
            if terminal:
                break
        rewards.append(episode_reward)
        append_row(train_path, {"condition": condition, "training_seed": seed, "episode": episode + 1, "environment_seed": env_seed, "seat": episode % 4, "epsilon": epsilon, "reward": episode_reward, "step": agent.steps, **row_metrics(info)})
        if (episode + 1) % 25 == 0 or episode + 1 == total:
            elapsed = time.perf_counter() - start
            metadata.update(completed_episodes=episode + 1, duration_seconds=previous.get("duration_seconds", 0) + elapsed, steps=agent.steps, status="completed" if episode + 1 == total else "running")
            agent.save(checkpoint, metadata=metadata, include_replay=True)
            meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"{run}: {episode+1}/{total} matches, decisions={agent.steps}, mean last25 reward={np.mean(rewards[-25:]):.3f}, elapsed={elapsed:.1f}s", flush=True)
    return checkpoint


def evaluate(agent_name, config, training_seed=None, aware_override=None, destination=None):
    out, checkpoints = output_dirs(config)
    if agent_name not in AGENTS:
        raise ValueError(agent_name)
    run_seed = training_seed if training_seed is not None else -1
    suffix = "_features_off" if aware_override is False else ""
    dest = Path(destination) if destination else out / "evaluation" / f"{agent_name}{suffix}_seed{run_seed}.csv"
    if dest.exists():
        raise FileExistsError(f"Evaluation exists: {dest}")
    agent = None
    if agent_name in CONDITIONS:
        agent = DQNAgent.load(checkpoints / f"{agent_name}_seed{training_seed}.pt")
    aware = agent_name == "dqn_aware" if aware_override is None else aware_override
    started = time.perf_counter()
    for kind in OPPONENTS:
        for block in range(config["evaluation_seeds"]):
            env_seed = config["evaluation_seed_start"] + block
            for seat in config["seats"]:
                if agent_name == "random":
                    policy = RandomAgent(seed=env_seed + 700 + seat)
                elif agent_name == "heuristic":
                    policy = HeuristicAgent(seed=env_seed + 700 + seat, strategy="heuristic")
                else:
                    policy = agent
                env = SingleAgentEnv(make_opponents(kind, env_seed), seat=seat, aware=aware, max_actions=config["max_actions"], shaping=0.0)
                state, mask = env.reset(seed=env_seed)
                mask = progress_mask(state, mask)
                while True:
                    action = policy.act(state, mask, epsilon=0.0)
                    state, reward, terminal, truncated, info = env.step(action)
                    mask = progress_mask(state, info["action_mask"])
                    if terminal or truncated:
                        break
                append_row(dest, {"agent": agent_name + suffix, "training_seed": run_seed, "opponent": kind, "environment_seed": env_seed, "seat": seat, "completed": terminal and not truncated, "win": int(info.get("position") == 1) if not truncated else None, **row_metrics(info)})
                if truncated:
                    raise RuntimeError(f"Evaluation game truncated; row saved, not interpreted as a result: {info}")
        print(f"Eval {agent_name}{suffix} seed{run_seed} / {kind}: {config['evaluation_seeds']*len(config['seats'])} held-out matches; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return dest
