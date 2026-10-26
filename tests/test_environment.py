"""Integration tests for the synchronous official Chef's Hat adapter.

The policies are intentionally tiny local policies: all rule validation remains
inside the upstream Game that ``SingleAgentEnv`` drives.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from sampath_rl.core import ACTIONS, SingleAgentEnv


class LocalRandomPolicy:
    """Independent seeded policy supporting the project's epsilon API."""

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)
        self.calls = 0
        self.last_epsilon = None

    def act(self, state: np.ndarray, mask: np.ndarray, epsilon: float = 0) -> int:
        assert state.shape == (64,)
        assert state.dtype == np.float32
        assert mask.shape == (200,)
        assert mask.dtype == np.bool_
        self.calls += 1
        self.last_epsilon = epsilon
        legal = np.flatnonzero(mask)
        assert legal.size > 0
        return int(legal[self._random.randrange(legal.size)])


class TwoArgumentFirstLegalPolicy:
    """Exercise the documented minimal ``act(state, mask)`` opponent protocol."""

    def act(self, state: np.ndarray, mask: np.ndarray) -> int:
        assert state.shape == (64,)
        return int(np.flatnonzero(mask)[0])


def _play_to_end(env: SingleAgentEnv, learner: LocalRandomPolicy, seed: int):
    state, mask = env.reset(seed=seed)
    assert state.shape == (64,)
    assert mask.shape == (200,)
    assert mask.dtype == np.bool_
    assert len(ACTIONS) == 200

    for _ in range(5000):
        action = learner.act(state, mask, epsilon=0)
        state, reward, terminated, truncated, info = env.step(action)
        next_mask = info["action_mask"]
        assert next_mask.shape == (200,)
        assert next_mask.dtype == np.bool_
        if terminated or truncated:
            return state, reward, terminated, truncated, info
        state, mask = state, next_mask
    pytest.fail("the official game did not finish under the environment action guard")


def test_full_four_player_random_game_contract_and_mask() -> None:
    """Complete one actual official match before any expensive RL workflow."""

    learner = LocalRandomPolicy(100)
    opponents = [LocalRandomPolicy(101), LocalRandomPolicy(102), LocalRandomPolicy(103)]
    env = SingleAgentEnv(opponents, seat=0, aware=False, shaping=0.01)

    initial_state, initial_mask = env.reset(seed=410)
    assert initial_state.shape == (64,)
    assert np.all(initial_state[47:62] == 0.0)  # exact architecture ablation
    assert np.all(initial_state[62:64] == 0.0)
    assert initial_mask.any()

    state, mask = initial_state, initial_mask
    for _ in range(5000):
        state, reward, terminated, truncated, info = env.step(learner.act(state, mask, epsilon=0))
        mask = info["action_mask"]
        if terminated or truncated:
            break
    else:
        pytest.fail("the official match exceeded the diagnostic action cap")

    assert terminated is True
    assert truncated is False
    assert state.shape == (64,)
    assert not mask.any()  # terminal next mask is always all false
    for key in (
        "position",
        "score",
        "performance_score",
        "actions",
        "learner_actions",
        "passes",
        "invalid_actions",
        "rounds",
        "duration_seconds",
        "finishing_order",
    ):
        assert key in info
    assert info["position"] in {1, 2, 3, 4}
    assert info["score"] in {0, 1, 2, 3}
    assert info["environment_score"] == info["score"]
    assert sorted(info["finishing_order"]) == [0, 1, 2, 3]
    assert len(info["actions"]) == 4
    assert info["learner_actions"] == info["actions"][0]
    assert info["invalid_actions"] == 0
    assert info["rounds"] >= 1
    assert info["duration_seconds"] >= 0.0
    assert all(policy.last_epsilon == 0 for policy in opponents if policy.calls)
    # Nonzero shaping is raw card count coefficient; terminal also includes
    # the independently labelled official final-match score reward.
    assert reward >= info["score"] / 3.0


def test_two_match_roles_exchange_trace_and_minimal_opponent_api(tmp_path: Path) -> None:
    """Run two matches, forcing official roles/exchange before match two."""

    trace = tmp_path / "learner_trace.jsonl"
    learner = LocalRandomPolicy(200)
    env = SingleAgentEnv(
        [TwoArgumentFirstLegalPolicy(), LocalRandomPolicy(202), LocalRandomPolicy(203)],
        seat=2,
        aware=True,
        max_matches=2,
        trace_path=trace,
    )

    state, reward, terminated, truncated, info = _play_to_end(env, learner, seed=901)

    assert terminated is True
    assert truncated is False
    assert info["matches"] == 2
    assert info["rounds"] >= 2
    assert set(info["roles"].values()) == {"chef", "souschef", "waiter", "dishwasher"}
    assert sum(info["game_scores"].values()) == 12  # 3 + 2 + 1 points per official match
    assert info["action_mask"].dtype == np.bool_
    assert not info["action_mask"].any()
    assert np.all(state[62:64] == 0.0)
    # aware=True uses the same 64 slots, with public history becoming populated
    assert np.all(np.isfinite(state[47:62]))

    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == info["learner_actions"]
    assert rows and all(row["event"] == "learner_action" for row in rows)
    assert all("learner_hand" in row for row in rows)
    assert all("opponent_hand" not in row for row in rows)
    assert all("remaining_cards" in row["public"] for row in rows)


def test_invalid_action_is_rejected_without_random_fallback() -> None:
    """An illegal learner id fails explicitly and does not advance the engine."""

    env = SingleAgentEnv([TwoArgumentFirstLegalPolicy()] * 3)
    _, mask = env.reset(seed=11)
    illegal = int(np.flatnonzero(~mask)[0])
    with pytest.raises(ValueError, match="no fallback"):
        env.step(illegal)
    assert env._total_actions == sum(env._actions_by_seat)
