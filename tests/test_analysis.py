"""Algorithmic unit tests for analysis helpers; no experiment result files are used."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sampath_rl.analysis import (
    OPPONENTS,
    _paired_block_ci,
    _paired_environment_seed_blocks,
    _paired_macro_ci,
    paired_bootstrap_ci,
    summarize_evaluation_frame,
    wilson_interval,
)


def test_wilson_interval_boundary_cases() -> None:
    """Wilson intervals remain bounded and non-degenerate at 0/1 observations."""

    assert wilson_interval(0, 0) == (0.0, 1.0)
    zero_low, zero_high = wilson_interval(0, 1)
    one_low, one_high = wilson_interval(1, 1)
    assert zero_low == pytest.approx(0.0)
    assert zero_high == pytest.approx(0.7934506856, abs=1e-9)
    assert one_low == pytest.approx(0.2065493144, abs=1e-9)
    assert one_high == pytest.approx(1.0)


def test_paired_bootstrap_equal_blocks_is_exactly_zero() -> None:
    """Shared resampling preserves an identically zero candidate-reference gap."""

    candidate = np.array([0.0, 0.25, 0.75, 1.0])
    observed, low, high = paired_bootstrap_ci(candidate, candidate.copy(), reps=300, seed=991)
    assert observed == 0.0
    assert low == 0.0
    assert high == 0.0


def test_summary_metrics_and_deal_block_count_are_correct() -> None:
    """A tiny in-memory fixture checks score, position, action, and pass formulas."""

    # This deliberately small synthetic fixture is confined to a unit test.  It
    # is not written to results/ and is not an experiment output dataset.
    fixture = pd.DataFrame(
        {
            "agent": ["dqn_aware"] * 4,
            "training_seed": [17] * 4,
            "opponent": ["random"] * 4,
            "environment_seed": [9000000, 9000000, 9000001, 9000001],
            "seat": [0, 1, 0, 1],
            "win": [1, 0, 0, 0],
            "position": [1, 2, 3, 4],
            "score": [3, 2, 1, 0],
            "performance_score": [3.0, 2.0, 1.0, 0.0],
            "actions": [10, 20, 30, 40],
            "learner_actions": [8, 16, 24, 32],
            "passes": [2, 4, 6, 8],
            "rounds": [2, 3, 4, 5],
            "duration_seconds": [1.0, 2.0, 3.0, 4.0],
        }
    )
    summary = summarize_evaluation_frame(fixture, aggregation="pooled", reps=200, seed=13)
    assert summary["n_matches"] == 4
    assert summary["n_environment_seeds"] == 2
    assert summary["n_deal_seed_blocks"] == 2
    assert summary["win_rate"] == pytest.approx(0.25)
    assert summary["mean_score"] == pytest.approx(1.5)
    assert summary["mean_position"] == pytest.approx(2.5)
    assert summary["mean_actions"] == pytest.approx(25.0)
    assert summary["pass_frequency"] == pytest.approx(20.0 / 80.0)


def _perfectly_correlated_opponent_fixture() -> pd.DataFrame:
    """Two seed blocks whose outcomes are identical across all opponents."""

    rows: list[dict[str, int | float | str]] = []
    for environment_seed, win in ((9100000, 1), (9100001, 0)):
        for opponent in OPPONENTS:
            position = 1 if win else 4
            rows.append(
                {
                    "agent": "dqn_aware",
                    "training_seed": 17,
                    "opponent": opponent,
                    "environment_seed": environment_seed,
                    "seat": 0,
                    "win": win,
                    "position": position,
                    "score": 4 - position,
                    "performance_score": float(4 - position),
                    "actions": 1,
                    "learner_actions": 1,
                    "passes": 0,
                    "rounds": 1,
                    "duration_seconds": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_all_opponent_bootstrap_keeps_shared_seed_correlation() -> None:
    """Aggregate draws retain all opponent outcomes for a selected seed together."""

    fixture = _perfectly_correlated_opponent_fixture()
    pooled = summarize_evaluation_frame(fixture, aggregation="pooled", reps=4000, seed=71)
    macro = summarize_evaluation_frame(
        fixture, aggregation="macro_equal_opponents", reps=4000, seed=71
    )

    # There are two environment-seed resampling units, not 2 × 5 opponent/seed
    # units.  A draw can therefore select either correlated outcome twice.
    for summary in (pooled, macro):
        assert summary["n_deal_seed_blocks"] == 2
        assert summary["win_rate"] == pytest.approx(0.5)
        assert summary["win_rate_ci95_bootstrap_low"] == pytest.approx(0.0)
        assert summary["win_rate_ci95_bootstrap_high"] == pytest.approx(1.0)


def test_paired_all_opponent_bootstrap_keeps_shared_seed_correlation() -> None:
    """Paired pooled and macro draws use one shared set of seed indices."""

    rows: list[dict[str, int | str]] = []
    for environment_seed, candidate_win, reference_win in ((9200000, 1, 0), (9200001, 0, 1)):
        for opponent in OPPONENTS:
            rows.append(
                {
                    "opponent": opponent,
                    "environment_seed": environment_seed,
                    "candidate_wins": candidate_win,
                    "reference_wins": reference_win,
                    "n": 1,
                }
            )
    blocks = pd.DataFrame(rows)

    pooled_blocks = _paired_environment_seed_blocks(blocks)
    pooled_observed, pooled_low, pooled_high = _paired_block_ci(
        pooled_blocks, reps=4000, seed=73
    )
    macro_observed, macro_low, macro_high = _paired_macro_ci(blocks, reps=4000, seed=73)

    assert len(pooled_blocks) == 2
    for observed, low, high in (
        (pooled_observed, pooled_low, pooled_high),
        (macro_observed, macro_low, macro_high),
    ):
        assert observed == pytest.approx(0.0)
        assert low == pytest.approx(-1.0)
        assert high == pytest.approx(1.0)
