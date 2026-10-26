#!/usr/bin/env python3
"""Play one full official match with a frozen trained agent; save public evidence."""
import argparse
import json
import time
from pathlib import Path
from sampath_rl.agents import DQNAgent
from sampath_rl.core import SingleAgentEnv, ACTIONS
from sampath_rl.experiment import make_opponents, ROOT
from sampath_rl.policy import progress_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/dqn_aware_seed17.pt")
    parser.add_argument("--seed", type=int, default=777888)
    parser.add_argument("--seat", type=int, choices=range(4), default=0)
    parser.add_argument("--delay", type=float, default=0.0, help="Optional pacing for your screen-recorded viva")
    parser.add_argument("--output", default="viva/gameplay")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / args.checkpoint
    agent = DQNAgent.load(checkpoint)
    (output / "official_match_trace.jsonl").unlink(missing_ok=True)
    env = SingleAgentEnv(make_opponents("mixed", args.seed), seat=args.seat, aware=True, max_actions=5000, shaping=0.0, trace_path=output / "official_match_trace.jsonl")
    state, mask = env.reset(seed=args.seed)
    lines = [f"PATCHED OFFICIAL CHEF'S HAT — FROZEN TRAINED-AGENT DEMONSTRATION", f"Checkpoint: {args.checkpoint}", f"Seed: {args.seed}; learner seat: {args.seat}; opponents: mixed", "Backend: pinned official Game plus documented one-line joker fix; shared progress policy.", "Public trace excludes opponent private card identities; legal count is the policy-choice subset."]
    for line in lines:
        print(line, flush=True)
    decisions = 0
    while True:
        mask = progress_mask(state, mask)
        action = agent.act(state, mask, epsilon=0.0)
        own_hand = [rank + 1 for rank, normalized_count in enumerate(state[:12]) for _ in range(int(round(float(normalized_count) * 11)))]
        line = f"Decision {decisions+1:03d}: own_hand={own_hand}; legal={int(mask.sum())}; selected={ACTIONS[action]}"
        lines.append(line)
        print(line, flush=True)
        state, reward, terminal, truncated, info = env.step(action)
        decisions += 1
        mask = info["action_mask"]
        if args.delay:
            time.sleep(args.delay)
        if terminal or truncated:
            break
    result = {k: v for k, v in info.items() if k != "action_mask"}
    result.update(checkpoint=args.checkpoint, demonstration_seed=args.seed, learner_decisions=decisions, completed=terminal and not truncated)
    if truncated:
        raise RuntimeError("Demonstration truncated; do not present as a completed match")
    def encode(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)
    (output / "demo_summary.json").write_text(json.dumps(result, indent=2, default=encode) + "\n")
    line = "COMPLETE OFFICIAL MATCH: " + json.dumps(result, default=encode)
    print(line, flush=True)
    lines.append(line)
    (output / "gameplay_transcript.txt").write_text("\n".join(lines) + "\n")
