"""Focused behavioral tests for the policy and Double-DQN agents."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sampath_rl.agents import ACTION_DIM, DQNAgent, HeuristicAgent, RandomAgent, STATE_DIM


def _state(hand_total: float = 0.0) -> np.ndarray:
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[0] = hand_total
    return state


def _mask(*legal: int) -> np.ndarray:
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[list(legal)] = True
    return mask


def _set_constant_qs(agent: DQNAgent, values: np.ndarray, *, target: bool = False) -> None:
    """Make a network's output independent of state for exact target tests."""

    network = agent.target if target else agent.online
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.layers[-1].bias.copy_(torch.as_tensor(values, dtype=torch.float32))


def test_random_and_heuristic_agents_always_choose_legal_actions() -> None:
    state = _state(9)
    legal = _mask(2, 17, 91)
    random_a = RandomAgent(seed=9)
    random_b = RandomAgent(seed=9)
    sequence_a = [random_a.act(state, legal) for _ in range(8)]
    sequence_b = [random_b.act(state, legal) for _ in range(8)]
    assert sequence_a == sequence_b
    assert set(sequence_a).issubset({2, 17, 91})

    for strategy in ("heuristic", "aggressive", "conservative", "switching"):
        agent = HeuristicAgent(seed=3, strategy=strategy)
        assert agent.act(state, legal) in {2, 17, 91}


def test_switching_policy_changes_after_observed_opponent_decisions() -> None:
    agent = HeuristicAgent(seed=1, strategy="switching", switch_interval=2)
    assert agent.active_strategy == "aggressive"
    agent.observe_opponent_decision()
    assert agent.active_strategy == "aggressive"
    agent.observe_opponent_action()
    assert agent.active_strategy == "conservative"
    agent.reset_match()
    assert agent.active_strategy == "aggressive"


def test_agents_reject_empty_action_masks() -> None:
    state = _state()
    empty = np.zeros(ACTION_DIM, dtype=bool)
    with pytest.raises(ValueError, match="no legal"):
        RandomAgent(0).act(state, empty)
    with pytest.raises(ValueError, match="no legal"):
        HeuristicAgent(0).act(state, empty)
    with pytest.raises(ValueError, match="no legal"):
        DQNAgent(0).act(state, empty)


def test_dqn_greedy_action_applies_legal_mask() -> None:
    agent = DQNAgent(seed=2)
    values = np.arange(ACTION_DIM, dtype=np.float32)
    _set_constant_qs(agent, values)
    # Highest global Q is illegal; the best legal action must be selected.
    legal = _mask(7, 41, 123)
    assert agent.act(_state(), legal, epsilon=0.0) == 123


def test_double_dqn_targets_use_gamma_and_do_not_bootstrap_terminal() -> None:
    agent = DQNAgent(seed=4, config={"gamma": 0.5})
    online_values = np.zeros(ACTION_DIM, dtype=np.float32)
    online_values[12] = 10.0
    target_values = np.full(ACTION_DIM, 4.0, dtype=np.float32)
    _set_constant_qs(agent, online_values)
    _set_constant_qs(agent, target_values, target=True)

    rewards = torch.tensor([2.0, 3.0])
    next_states = torch.zeros((2, STATE_DIM))
    next_masks = torch.zeros((2, ACTION_DIM), dtype=torch.bool)
    next_masks[0, 12] = True
    # The terminal row deliberately has an all-zero mask.  It must never be
    # passed through a masked argmax and should equal its immediate reward.
    dones = torch.tensor([False, True])
    targets = agent._compute_targets(rewards, next_states, next_masks, dones)
    torch.testing.assert_close(targets, torch.tensor([4.0, 3.0]))


def test_observe_accepts_terminal_zero_next_mask_and_returns_scheduled_loss() -> None:
    agent = DQNAgent(
        seed=5,
        config={"batch_size": 1, "learn_start": 1, "update_every": 1, "target_interval": 1},
    )
    loss = agent.observe(_state(), 0, 1.0, _state(), np.zeros(ACTION_DIM, dtype=bool), True)
    assert isinstance(loss, float)
    assert np.isfinite(loss)
    assert agent.steps == 1
    assert agent.learn_steps == 1


def test_checkpoint_has_equal_inference_and_resumes_rng_and_replay_ring(tmp_path: Path) -> None:
    config = {
        "capacity": 3,
        "batch_size": 1,
        "learn_start": 100,
        "update_every": 1,
        "target_interval": 100,
    }
    agent = DQNAgent(seed=12, config=config)
    legal = _mask(0, 4, 9)
    for transition in range(5):
        state = _state(float(transition))
        agent.observe(
            state,
            action=0,
            reward=float(transition),
            next_state=_state(float(transition + 1)),
            next_mask=legal,
            done=False,
            mask=legal,
        )

    checkpoint = tmp_path / "agent.pt"
    agent.save(checkpoint, metadata={"experiment": "test"}, include_replay=True)
    with pytest.warns(UserWarning, match="trusted"):
        restored = DQNAgent.load(checkpoint, load_replay=True)

    probe = _state(1.5)
    with torch.no_grad():
        expected = agent.online(torch.as_tensor(probe).unsqueeze(0))
        actual = restored.online(torch.as_tensor(probe).unsqueeze(0))
    torch.testing.assert_close(expected, actual)
    assert restored.act(probe, legal) == agent.act(probe, legal)

    # Saving preserves the next random draw.  The replay ring is full and has
    # wrapped, so both write position and physical contents are relevant.
    assert restored.replay.size == 3
    assert restored.replay.position == agent.replay.position
    np.testing.assert_array_equal(restored.replay.states, agent.replay.states)
    np.testing.assert_array_equal(restored.replay.next_masks, agent.replay.next_masks)
    original_rollout = [agent.act(probe, legal, epsilon=1.0) for _ in range(12)]
    restored_rollout = [restored.act(probe, legal, epsilon=1.0) for _ in range(12)]
    assert restored_rollout == original_rollout


def test_nonterminal_zero_next_mask_is_rejected() -> None:
    agent = DQNAgent(seed=8)
    with pytest.raises(ValueError, match="no legal"):
        agent.observe(
            _state(),
            action=0,
            reward=0.0,
            next_state=_state(),
            next_mask=np.zeros(ACTION_DIM, dtype=bool),
            done=False,
        )
