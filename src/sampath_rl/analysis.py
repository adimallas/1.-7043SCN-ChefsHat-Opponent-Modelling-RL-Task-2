"""Reproducible analysis of completed Chef's Hat opponent-modelling experiments.

This module deliberately reads only experiment artefacts written by
:mod:`sampath_rl.experiment`.  It never simulates matches, fills in missing
rows, or creates placeholder result data.  ``generate_all`` validates the
complete standard protocol before it writes summaries, figures, or RESULTS.md.

Run after the standard experiment has completed::

    python -m sampath_rl.analysis --root /path/to/repository

The primary uncertainty estimates resample held-out environment-seed blocks.
Each per-opponent block contains every seat and (for trained policies) every
training checkpoint for an environment seed.  For all-opponent aggregates, a
block additionally contains all five opponent conditions evaluated with that
same environment seed, preserving the shared-deal dependence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

# A non-interactive backend keeps command-line analysis reliable on CI and HPC
# nodes while retaining vector SVG output.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]

CONDITIONS: tuple[str, ...] = ("dqn_random", "dqn_mixed", "dqn_aware")
AGENTS: tuple[str, ...] = (
    "random",
    "heuristic",
    "dqn_random",
    "dqn_mixed",
    "dqn_aware",
    "dqn_aware_features_off",
)
OPPONENTS: tuple[str, ...] = (
    "random",
    "aggressive",
    "conservative",
    "mixed",
    "switching",
)
TRAINING_SEEDS: tuple[int, ...] = (17, 29, 43)
BASELINE_SEED = -1
BOOTSTRAP_REPS = 3000

EVALUATION_COLUMNS: tuple[str, ...] = (
    "agent",
    "training_seed",
    "opponent",
    "environment_seed",
    "seat",
    "completed",
    "win",
    "position",
    "score",
    "performance_score",
    "actions",
    "learner_actions",
    "passes",
    "invalid_actions",
    "rounds",
    "duration_seconds",
)
TRAINING_COLUMNS: tuple[str, ...] = (
    "condition",
    "training_seed",
    "episode",
    "environment_seed",
    "seat",
    "epsilon",
    "reward",
    "step",
    "position",
    "score",
    "performance_score",
    "actions",
    "learner_actions",
    "passes",
    "invalid_actions",
    "rounds",
    "duration_seconds",
)
LOSS_COLUMNS: tuple[str, ...] = ("episode", "step", "loss")

# The first five hues are deliberately the requested scholarly palette.  Slate
# distinguishes the random baseline without competing with the DQN conditions.
PALETTE: Mapping[str, str] = {
    "random": "#52616B",       # slate baseline
    "heuristic": "#6C4A86",    # purple
    "dqn_random": "#173F5F",   # navy
    "dqn_mixed": "#008C95",    # teal
    "dqn_aware": "#E28E1C",    # orange
    "dqn_aware_features_off": "#C43C39",  # red
}
DISPLAY_NAMES: Mapping[str, str] = {
    "random": "Random",
    "heuristic": "Heuristic",
    "dqn_random": "DQN random",
    "dqn_mixed": "DQN mixed",
    "dqn_aware": "DQN aware",
    "dqn_aware_features_off": "Aware, features off",
}


class ResultDataError(RuntimeError):
    """Raised when saved artefacts do not constitute the complete protocol."""


@dataclass(frozen=True)
class ExperimentData:
    """Validated source artefacts used to produce every analysis output."""

    root: Path
    config: Mapping[str, Any]
    evaluation: pd.DataFrame
    training: pd.DataFrame
    losses: pd.DataFrame
    metadata: Mapping[str, Mapping[str, Any]]
    evaluation_files: tuple[Path, ...]
    training_files: tuple[Path, ...]
    loss_files: tuple[Path, ...]
    metadata_files: tuple[Path, ...]
    checkpoint_files: tuple[Path, ...]


def _agent_seeds(agent: str) -> tuple[int, ...]:
    if agent in {"random", "heuristic"}:
        return (BASELINE_SEED,)
    if agent in AGENTS:
        return TRAINING_SEEDS
    raise ValueError(f"unknown analysis agent {agent!r}")


def _evaluation_filename(agent: str, seed: int) -> str:
    return f"{agent}_seed{seed}.csv"


def _stable_seed(label: str, base_seed: int = 20261025) -> int:
    """Derive a platform-independent deterministic NumPy seed from a label."""

    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + int(base_seed)) % (2**32)


def _as_root(root: Path | str) -> Path:
    return Path(root).expanduser().resolve()


def _read_standard_config(root: Path) -> Mapping[str, Any]:
    path = root / "configs" / "standard.yaml"
    if not path.is_file():
        raise ResultDataError(f"missing standard configuration: {path}")
    try:
        config = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ResultDataError(f"could not read standard configuration {path}: {exc}") from exc
    if not isinstance(config, Mapping):
        raise ResultDataError(f"standard configuration is not a mapping: {path}")
    required = {
        "preset",
        "train_episodes",
        "training_seeds",
        "evaluation_seed_start",
        "evaluation_seeds",
        "seats",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ResultDataError(f"standard configuration misses required keys: {missing}")
    if config["preset"] != "standard":
        raise ResultDataError("configs/standard.yaml must declare preset: standard")
    if int(config["train_episodes"]) != 600:
        raise ResultDataError(
            f"standard protocol requires 600 training episodes, found {config['train_episodes']}"
        )
    if tuple(int(value) for value in config["training_seeds"]) != TRAINING_SEEDS:
        raise ResultDataError(
            "standard protocol requires training_seeds [17, 29, 43], found "
            f"{config['training_seeds']}"
        )
    if int(config["evaluation_seeds"]) != 32:
        raise ResultDataError(
            f"standard protocol requires 32 evaluation deal seeds, found {config['evaluation_seeds']}"
        )
    if tuple(int(value) for value in config["seats"]) != (0, 1, 2, 3):
        raise ResultDataError(
            f"standard protocol requires seats [0, 1, 2, 3], found {config['seats']}"
        )
    return config


def _require_columns(frame: pd.DataFrame, required: Sequence[str], path: Path) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ResultDataError(f"{path} is missing required columns: {missing}")


def _read_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise ResultDataError(f"missing required result file: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pandas exposes several parsing exception types
        raise ResultDataError(f"could not parse CSV {path}: {exc}") from exc
    _require_columns(frame, columns, path)
    if frame.empty:
        raise ResultDataError(f"required CSV is empty: {path}")
    return frame


def _numeric(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    """Coerce numeric columns in place and reject missing/non-finite entries."""

    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise ResultDataError(f"{path}: {column} contains missing or non-finite values")
        frame[column] = converted


def _integer_columns(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.allclose(values, np.round(values), rtol=0.0, atol=1e-10):
            raise ResultDataError(f"{path}: {column} must contain integer values")
        frame[column] = np.round(values).astype(np.int64)


def _normalise_completed(series: pd.Series, path: Path) -> pd.Series:
    def convert(value: Any) -> bool | None:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
        return None

    converted = series.map(convert)
    if converted.isna().any():
        bad = series[converted.isna()].iloc[0]
        raise ResultDataError(f"{path}: completed contains an invalid boolean value {bad!r}")
    return converted.astype(bool)


def _check_metric_invariants(frame: pd.DataFrame, path: Path) -> None:
    """Validate official-game consistency shared by training and evaluation."""

    if not frame["position"].between(1, 4).all():
        raise ResultDataError(f"{path}: position must be in [1, 4]")
    expected_score = 4 - frame["position"].to_numpy(dtype=float)
    if not np.allclose(frame["score"].to_numpy(dtype=float), expected_score, atol=1e-10):
        raise ResultDataError(f"{path}: score must equal 4 - position for every match")
    for column in ("actions", "learner_actions", "passes", "invalid_actions", "rounds"):
        if (frame[column] < 0).any():
            raise ResultDataError(f"{path}: {column} cannot be negative")
    if (frame["invalid_actions"] != 0).any():
        raise ResultDataError(f"{path}: invalid_actions must be zero in completed official games")
    if (frame["passes"] > frame["learner_actions"]).any():
        raise ResultDataError(f"{path}: passes cannot exceed learner_actions")
    if (frame["duration_seconds"] < 0).any():
        raise ResultDataError(f"{path}: duration_seconds cannot be negative")


def _validate_evaluation_file(
    frame: pd.DataFrame,
    path: Path,
    *,
    agent: str,
    training_seed: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    frame = frame.loc[:, list(EVALUATION_COLUMNS)].copy()
    _numeric(
        frame,
        (
            "training_seed",
            "environment_seed",
            "seat",
            "win",
            "position",
            "score",
            "performance_score",
            "actions",
            "learner_actions",
            "passes",
            "invalid_actions",
            "rounds",
            "duration_seconds",
        ),
        path,
    )
    _integer_columns(
        frame,
        (
            "training_seed",
            "environment_seed",
            "seat",
            "win",
            "position",
            "actions",
            "learner_actions",
            "passes",
            "invalid_actions",
            "rounds",
        ),
        path,
    )
    frame["completed"] = _normalise_completed(frame["completed"], path)
    if set(frame["agent"].astype(str)) != {agent}:
        raise ResultDataError(f"{path}: agent column must contain only {agent!r}")
    if set(frame["training_seed"].astype(int)) != {training_seed}:
        raise ResultDataError(f"{path}: training_seed column must contain only {training_seed}")
    if not frame["completed"].all():
        raise ResultDataError(f"{path}: every held-out evaluation must be completed")
    if not set(frame["win"].unique()).issubset({0, 1}):
        raise ResultDataError(f"{path}: win must be binary")
    expected_win = (frame["position"] == 1).astype(int).to_numpy()
    if not np.array_equal(frame["win"].to_numpy(dtype=int), expected_win):
        raise ResultDataError(f"{path}: win must equal (position == 1)")
    _check_metric_invariants(frame, path)

    expected_opponents = set(OPPONENTS)
    actual_opponents = set(frame["opponent"].astype(str))
    if actual_opponents != expected_opponents:
        raise ResultDataError(
            f"{path}: opponents must be {sorted(expected_opponents)}, found {sorted(actual_opponents)}"
        )
    env_start = int(config["evaluation_seed_start"])
    expected_envs = set(range(env_start, env_start + int(config["evaluation_seeds"])))
    expected_seats = set(int(value) for value in config["seats"])
    expected_pairs = {(env, seat) for env in expected_envs for seat in expected_seats}
    expected_per_opponent = len(expected_pairs)
    for opponent in OPPONENTS:
        subset = frame.loc[frame["opponent"] == opponent]
        actual_pairs = set(
            zip(subset["environment_seed"].astype(int), subset["seat"].astype(int), strict=True)
        )
        if len(subset) != expected_per_opponent or actual_pairs != expected_pairs:
            raise ResultDataError(
                f"{path}: {opponent} must contain each of {len(expected_envs)} deal seeds "
                f"at all {len(expected_seats)} seats exactly once ({expected_per_opponent} rows)"
            )
    expected_rows = expected_per_opponent * len(OPPONENTS)
    if len(frame) != expected_rows:
        raise ResultDataError(f"{path}: expected {expected_rows} rows, found {len(frame)}")
    frame["_source_file"] = str(path)
    return frame


def _validate_training_file(
    frame: pd.DataFrame,
    path: Path,
    *,
    condition: str,
    training_seed: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    frame = frame.loc[:, list(TRAINING_COLUMNS)].copy()
    _numeric(
        frame,
        (
            "training_seed",
            "episode",
            "environment_seed",
            "seat",
            "epsilon",
            "reward",
            "step",
            "position",
            "score",
            "performance_score",
            "actions",
            "learner_actions",
            "passes",
            "invalid_actions",
            "rounds",
            "duration_seconds",
        ),
        path,
    )
    _integer_columns(
        frame,
        (
            "training_seed",
            "episode",
            "environment_seed",
            "seat",
            "step",
            "position",
            "actions",
            "learner_actions",
            "passes",
            "invalid_actions",
            "rounds",
        ),
        path,
    )
    if set(frame["condition"].astype(str)) != {condition}:
        raise ResultDataError(f"{path}: condition column must contain only {condition!r}")
    if set(frame["training_seed"].astype(int)) != {training_seed}:
        raise ResultDataError(f"{path}: training_seed column must contain only {training_seed}")
    episodes = int(config["train_episodes"])
    if len(frame) != episodes or set(frame["episode"].astype(int)) != set(range(1, episodes + 1)):
        raise ResultDataError(f"{path}: must contain exactly episodes 1..{episodes} once")
    expected_env = training_seed * 100000 + frame["episode"].to_numpy(dtype=int) - 1
    if not np.array_equal(frame["environment_seed"].to_numpy(dtype=int), expected_env):
        raise ResultDataError(f"{path}: environment_seed does not match the experiment training schedule")
    expected_seat = (frame["episode"].to_numpy(dtype=int) - 1) % 4
    if not np.array_equal(frame["seat"].to_numpy(dtype=int), expected_seat):
        raise ResultDataError(f"{path}: seat does not match the experiment training schedule")
    if not frame["epsilon"].between(0.0, 1.0).all():
        raise ResultDataError(f"{path}: epsilon must lie in [0, 1]")
    _check_metric_invariants(frame, path)
    frame["_source_file"] = str(path)
    return frame


def _validate_loss_file(frame: pd.DataFrame, path: Path, episodes: int) -> pd.DataFrame:
    frame = frame.loc[:, list(LOSS_COLUMNS)].copy()
    _numeric(frame, LOSS_COLUMNS, path)
    _integer_columns(frame, ("episode", "step"), path)
    if not frame["episode"].between(1, episodes).all():
        raise ResultDataError(f"{path}: loss episode must lie in [1, {episodes}]")
    if (frame["step"] <= 0).any() or (frame["loss"] < 0).any():
        raise ResultDataError(f"{path}: step must be positive and loss non-negative")
    frame["_source_file"] = str(path)
    return frame


def _read_metadata(path: Path, condition: str, seed: int, episodes: int) -> Mapping[str, Any]:
    if not path.is_file():
        raise ResultDataError(f"missing required metadata file: {path}")
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultDataError(f"could not parse metadata {path}: {exc}") from exc
    if not isinstance(metadata, Mapping):
        raise ResultDataError(f"metadata is not a JSON object: {path}")
    try:
        metadata_seed = int(metadata.get("training_seed", -999999))
        completed_episodes = int(metadata.get("completed_episodes", -1))
    except (TypeError, ValueError) as exc:
        raise ResultDataError(f"{path}: training_seed and completed_episodes must be integers") from exc
    if metadata.get("condition") != condition or metadata_seed != seed:
        raise ResultDataError(f"{path}: condition/training_seed does not match its filename")
    if completed_episodes != episodes or metadata.get("status") != "completed":
        raise ResultDataError(f"{path}: metadata must record a completed {episodes}-episode run")
    return metadata


def load_experiment_data(root: Path | str = ROOT) -> ExperimentData:
    """Read and validate every required standard-protocol source artefact.

    This is intentionally strict.  Missing, incomplete, malformed, or
    incompatible files result in :class:`ResultDataError`; no downstream output
    is written by this function or by :func:`generate_all` in that case.
    """

    root_path = _as_root(root)
    config = _read_standard_config(root_path)
    results = root_path / "results"
    evaluation_dir = results / "evaluation"
    training_dir = results / "training"
    checkpoints = root_path / "checkpoints"

    evaluation_frames: list[pd.DataFrame] = []
    evaluation_files: list[Path] = []
    for agent in AGENTS:
        for seed in _agent_seeds(agent):
            path = evaluation_dir / _evaluation_filename(agent, seed)
            frame = _read_csv(path, EVALUATION_COLUMNS)
            evaluation_frames.append(
                _validate_evaluation_file(
                    frame, path, agent=agent, training_seed=seed, config=config
                )
            )
            evaluation_files.append(path)

    training_frames: list[pd.DataFrame] = []
    loss_frames: list[pd.DataFrame] = []
    training_files: list[Path] = []
    loss_files: list[Path] = []
    metadata_files: list[Path] = []
    checkpoint_files: list[Path] = []
    metadata: dict[str, Mapping[str, Any]] = {}
    for condition in CONDITIONS:
        for seed in TRAINING_SEEDS:
            run = f"{condition}_seed{seed}"
            training_path = training_dir / f"{run}.csv"
            loss_path = training_dir / f"{run}_loss.csv"
            metadata_path = training_dir / f"{run}_metadata.json"
            checkpoint_path = checkpoints / f"{run}.pt"
            training_frames.append(
                _validate_training_file(
                    _read_csv(training_path, TRAINING_COLUMNS),
                    training_path,
                    condition=condition,
                    training_seed=seed,
                    config=config,
                )
            )
            loss_frames.append(
                _validate_loss_file(
                    _read_csv(loss_path, LOSS_COLUMNS),
                    loss_path,
                    int(config["train_episodes"]),
                )
            )
            metadata[run] = _read_metadata(
                metadata_path, condition, seed, int(config["train_episodes"])
            )
            if not checkpoint_path.is_file():
                raise ResultDataError(f"missing required frozen checkpoint: {checkpoint_path}")
            training_files.append(training_path)
            loss_files.append(loss_path)
            metadata_files.append(metadata_path)
            checkpoint_files.append(checkpoint_path)

    evaluation = pd.concat(evaluation_frames, ignore_index=True)
    training = pd.concat(training_frames, ignore_index=True)
    losses = pd.concat(loss_frames, ignore_index=True)

    # Held-out evaluation deal seeds must be disjoint from every actual training
    # game; this also catches accidentally pointing analysis at quick artefacts.
    train_deals = set(training["environment_seed"].astype(int))
    evaluation_deals = set(evaluation["environment_seed"].astype(int))
    overlap = sorted(train_deals.intersection(evaluation_deals))
    if overlap:
        raise ResultDataError(
            "training and held-out evaluation environment seeds overlap: "
            f"{overlap[:10]}{'...' if len(overlap) > 10 else ''}"
        )

    return ExperimentData(
        root=root_path,
        config=config,
        evaluation=evaluation,
        training=training,
        losses=losses,
        metadata=metadata,
        evaluation_files=tuple(evaluation_files),
        training_files=tuple(training_files),
        loss_files=tuple(loss_files),
        metadata_files=tuple(metadata_files),
        checkpoint_files=tuple(checkpoint_files),
    )


def wilson_interval(successes: int | float, trials: int | float, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the two-sided Wilson interval, including well-defined boundaries."""

    successes = float(successes)
    trials = float(trials)
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes must lie between 0 and trials")
    if trials == 0:
        return (0.0, 1.0)
    proportion = successes / trials
    denominator = 1.0 + (z * z) / trials
    centre = (proportion + (z * z) / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        (proportion * (1.0 - proportion) + (z * z) / (4.0 * trials)) / trials
    ) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def paired_bootstrap_ci(
    candidate_blocks: Sequence[float] | np.ndarray,
    reference_blocks: Sequence[float] | np.ndarray,
    *,
    reps: int = BOOTSTRAP_REPS,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap a mean paired difference while resampling common block indices.

    This small public helper is useful for unit tests and for readers auditing
    the pairing rule.  Production comparisons use weighted block totals below,
    because a deal block contains multiple match rows.
    """

    candidate = np.asarray(candidate_blocks, dtype=float).reshape(-1)
    reference = np.asarray(reference_blocks, dtype=float).reshape(-1)
    if candidate.size == 0 or candidate.shape != reference.shape:
        raise ValueError("candidate and reference blocks must be non-empty and equally shaped")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("block values must be finite")
    if int(reps) <= 0:
        raise ValueError("reps must be positive")
    differences = candidate - reference
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, differences.size, size=(int(reps), differences.size))
    draws = differences[choices].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(differences.mean()), float(low), float(high)


# The public table deliberately reports all requested behavioural metrics.  The
# same bootstrap block resamples produce every primary confidence interval.
METRIC_COLUMNS: tuple[str, ...] = (
    "win_rate",
    "mean_score",
    "mean_position",
    "mean_actions",
    "mean_learner_actions",
    "pass_frequency",
    "mean_rounds",
    "mean_duration_seconds",
    "mean_performance_score",
)


def _metric_values_from_totals(totals: Mapping[str, float]) -> dict[str, float]:
    n = float(totals["n"])
    learner_actions = float(totals["learner_actions"])
    if n <= 0 or learner_actions <= 0:
        raise ResultDataError("cannot calculate metrics from an empty or action-free result set")
    return {
        "win_rate": float(totals["wins"]) / n,
        "mean_score": float(totals["score"]) / n,
        "mean_position": float(totals["position"]) / n,
        "mean_actions": float(totals["actions"]) / n,
        "mean_learner_actions": learner_actions / n,
        "pass_frequency": float(totals["passes"]) / learner_actions,
        "mean_rounds": float(totals["rounds"]) / n,
        "mean_duration_seconds": float(totals["duration_seconds"]) / n,
        "mean_performance_score": float(totals["performance_score"]) / n,
    }


def _totals_from_frame(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "n": float(len(frame)),
        "wins": float(frame["win"].sum()),
        "score": float(frame["score"].sum()),
        "position": float(frame["position"].sum()),
        "actions": float(frame["actions"].sum()),
        "learner_actions": float(frame["learner_actions"].sum()),
        "passes": float(frame["passes"].sum()),
        "rounds": float(frame["rounds"].sum()),
        "duration_seconds": float(frame["duration_seconds"].sum()),
        "performance_score": float(frame["performance_score"].sum()),
    }


def _deal_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse seats/checkpoints into per-opponent environment-seed blocks."""

    sum_columns = [
        "win",
        "score",
        "position",
        "actions",
        "learner_actions",
        "passes",
        "rounds",
        "duration_seconds",
        "performance_score",
    ]
    grouped = (
        frame.groupby(["opponent", "environment_seed"], sort=True, as_index=False)[sum_columns]
        .sum()
        .rename(columns={"win": "wins"})
    )
    counts = frame.groupby(["opponent", "environment_seed"], sort=True).size().rename("n")
    grouped = grouped.merge(counts.reset_index(), on=["opponent", "environment_seed"], validate="one_to_one")
    return grouped


_BLOCK_TOTAL_COLUMNS: tuple[str, ...] = (
    "n",
    "wins",
    "score",
    "position",
    "actions",
    "learner_actions",
    "passes",
    "rounds",
    "duration_seconds",
    "performance_score",
)


def _environment_seed_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse all conditions, seats, and checkpoints by environment seed."""

    sum_columns = [column for column in _BLOCK_TOTAL_COLUMNS if column not in {"n", "wins"}]
    grouped = (
        frame.groupby("environment_seed", sort=True, as_index=False)[["win", *sum_columns]]
        .sum()
        .rename(columns={"win": "wins"})
    )
    counts = frame.groupby("environment_seed", sort=True).size().rename("n")
    return grouped.merge(counts.reset_index(), on="environment_seed", validate="one_to_one")


def _opponent_environment_block_cube(frame: pd.DataFrame, *, context: str) -> np.ndarray:
    """Return aligned opponent totals indexed by shared environment seed.

    The first axis is the resampling unit.  It is deliberately shared by every
    opponent type, so aggregate bootstrap draws cannot break the dependence
    caused by reusing each environment seed across all five conditions.
    """

    blocks = _deal_blocks(frame)
    opponents = set(blocks["opponent"].astype(str))
    if opponents != set(OPPONENTS):
        raise ResultDataError(
            f"{context} requires blocks for all opponents {list(OPPONENTS)}, "
            f"found {sorted(opponents)}"
        )
    environment_seeds = np.sort(blocks["environment_seed"].unique())
    expected_index = pd.MultiIndex.from_product(
        [environment_seeds, OPPONENTS], names=["environment_seed", "opponent"]
    )
    indexed = blocks.set_index(["environment_seed", "opponent"])
    aligned = indexed.reindex(expected_index)
    if aligned.loc[:, list(_BLOCK_TOTAL_COLUMNS)].isna().any().any():
        raise ResultDataError(
            f"{context} requires every environment seed to include all opponent conditions"
        )
    values = aligned.loc[:, list(_BLOCK_TOTAL_COLUMNS)].to_numpy(dtype=float)
    return values.reshape(len(environment_seeds), len(OPPONENTS), len(_BLOCK_TOTAL_COLUMNS))


def _metrics_from_block_draws(draw_totals: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = draw_totals["n"]
    learner_actions = draw_totals["learner_actions"]
    if np.any(n <= 0) or np.any(learner_actions <= 0):
        raise ResultDataError("bootstrap encountered an empty or action-free deal block draw")
    return {
        "win_rate": draw_totals["wins"] / n,
        "mean_score": draw_totals["score"] / n,
        "mean_position": draw_totals["position"] / n,
        "mean_actions": draw_totals["actions"] / n,
        "mean_learner_actions": learner_actions / n,
        "pass_frequency": draw_totals["passes"] / learner_actions,
        "mean_rounds": draw_totals["rounds"] / n,
        "mean_duration_seconds": draw_totals["duration_seconds"] / n,
        "mean_performance_score": draw_totals["performance_score"] / n,
    }


def _bootstrap_from_blocks(blocks: pd.DataFrame, reps: int, seed: int) -> dict[str, tuple[float, float]]:
    if blocks.empty:
        raise ResultDataError("cannot bootstrap an empty set of deal blocks")
    if int(reps) <= 0:
        raise ValueError("bootstrap reps must be positive")
    values = blocks.loc[:, list(_BLOCK_TOTAL_COLUMNS)].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(blocks), size=(int(reps), len(blocks)))
    sampled = values[choices].sum(axis=1)
    draw_totals = {column: sampled[:, i] for i, column in enumerate(_BLOCK_TOTAL_COLUMNS)}
    draw_metrics = _metrics_from_block_draws(draw_totals)
    return {
        metric: tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))
        for metric, draws in draw_metrics.items()
    }


def _macro_bootstrap(frame: pd.DataFrame, reps: int, seed: int) -> dict[str, tuple[float, float]]:
    """Bootstrap an equal-weight mean using shared environment-seed draws."""

    if int(reps) <= 0:
        raise ValueError("bootstrap reps must be positive")
    values = _opponent_environment_block_cube(frame, context="macro bootstrap")
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(values), size=(int(reps), len(values)))
    sampled = values[choices].sum(axis=1)
    per_opponent_metrics = _metrics_from_block_draws(
        {column: sampled[:, :, i] for i, column in enumerate(_BLOCK_TOTAL_COLUMNS)}
    )
    draws_by_metric = {
        metric: draws.mean(axis=1) for metric, draws in per_opponent_metrics.items()
    }
    return {
        metric: tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))
        for metric, draws in draws_by_metric.items()
    }


def _macro_metric_values(frame: pd.DataFrame) -> dict[str, float]:
    values = []
    for opponent in OPPONENTS:
        subset = frame.loc[frame["opponent"] == opponent]
        if subset.empty:
            raise ResultDataError(f"macro summary requires results for opponent {opponent!r}")
        values.append(_metric_values_from_totals(_totals_from_frame(subset)))
    return {metric: float(np.mean([value[metric] for value in values])) for metric in METRIC_COLUMNS}


def summarize_evaluation_frame(
    frame: pd.DataFrame,
    *,
    aggregation: str = "pooled",
    reps: int = BOOTSTRAP_REPS,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarise validated evaluation rows with environment-seed-block bootstrap CIs.

    ``aggregation='pooled'`` weights every saved match equally.  ``'macro_equal_opponents'``
    gives each of the five opponent types equal weight and is the primary
    all-opponent presentation.  Per-opponent rows resample environment seeds;
    all-opponent rows resample the same environment-seed indices jointly across
    every opponent type.
    """

    if frame.empty:
        raise ResultDataError("cannot summarise an empty evaluation frame")
    totals = _totals_from_frame(frame)
    if aggregation == "pooled":
        metrics = _metric_values_from_totals(totals)
        intervals = _bootstrap_from_blocks(_environment_seed_blocks(frame), reps, seed)
    elif aggregation == "macro_equal_opponents":
        metrics = _macro_metric_values(frame)
        intervals = _macro_bootstrap(frame, reps, seed)
    else:
        raise ValueError(f"unsupported aggregation {aggregation!r}")
    wilson_low, wilson_high = wilson_interval(totals["wins"], totals["n"])
    row: dict[str, Any] = {
        "n_matches": int(totals["n"]),
        "wins": int(totals["wins"]),
        "n_environment_seeds": int(frame["environment_seed"].nunique()),
        "n_training_seeds": int(frame["training_seed"].nunique()),
        "n_deal_seed_blocks": int(len(_environment_seed_blocks(frame))),
        "ci95_method_primary": "bootstrap_environment_seed_block_percentile",
        "bootstrap_replicates": int(reps),
        "win_rate_ci95_wilson_low": wilson_low,
        "win_rate_ci95_wilson_high": wilson_high,
    }
    row.update(metrics)
    for metric, (low, high) in intervals.items():
        row[f"{metric}_ci95_bootstrap_low"] = low
        row[f"{metric}_ci95_bootstrap_high"] = high
    return row


def _ordered_summary_frame(rows: list[dict[str, Any]], id_columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    metric_columns: list[str] = []
    for metric in METRIC_COLUMNS:
        metric_columns.append(metric)
        if metric == "win_rate":
            metric_columns.extend(["win_rate_ci95_wilson_low", "win_rate_ci95_wilson_high"])
        metric_columns.extend(
            [f"{metric}_ci95_bootstrap_low", f"{metric}_ci95_bootstrap_high"]
        )
    count_columns = [
        "n_matches",
        "wins",
        "n_environment_seeds",
        "n_training_seeds",
        "n_deal_seed_blocks",
        "ci95_method_primary",
        "bootstrap_replicates",
    ]
    columns = [*id_columns, *count_columns, *metric_columns]
    return frame.loc[:, columns]


def _paired_block_totals(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    keys = ["opponent", "environment_seed", "seat", "training_seed"]
    candidate_rows = candidate.loc[:, [*keys, "win"]].rename(columns={"win": "candidate_win"})
    reference_rows = reference.loc[:, [*keys, "win"]].rename(columns={"win": "reference_win"})
    try:
        merged = candidate_rows.merge(
            reference_rows, on=keys, how="outer", validate="one_to_one", indicator=True
        )
    except Exception as exc:
        raise ResultDataError(f"could not align paired evaluation rows: {exc}") from exc
    if not (merged["_merge"] == "both").all():
        mismatch = merged.loc[merged["_merge"] != "both", keys + ["_merge"]].head().to_dict("records")
        raise ResultDataError(f"paired evaluations do not share all rows: {mismatch}")
    blocks = (
        merged.groupby(["opponent", "environment_seed"], sort=True, as_index=False)[
            ["candidate_win", "reference_win"]
        ]
        .sum()
        .rename(columns={"candidate_win": "candidate_wins", "reference_win": "reference_wins"})
    )
    counts = merged.groupby(["opponent", "environment_seed"], sort=True).size().rename("n")
    return blocks.merge(counts.reset_index(), on=["opponent", "environment_seed"], validate="one_to_one")


def _paired_values(blocks: pd.DataFrame) -> tuple[float, float, float]:
    n = float(blocks["n"].sum())
    candidate = float(blocks["candidate_wins"].sum()) / n
    reference = float(blocks["reference_wins"].sum()) / n
    return candidate, reference, candidate - reference


def _paired_environment_seed_blocks(blocks: pd.DataFrame) -> pd.DataFrame:
    """Collapse paired opponent/deal totals into all-opponent seed blocks."""

    return blocks.groupby("environment_seed", sort=True, as_index=False)[
        ["candidate_wins", "reference_wins", "n"]
    ].sum()


def _paired_opponent_environment_cube(blocks: pd.DataFrame, *, context: str) -> np.ndarray:
    """Align paired totals so one draw selects every opponent at one seed."""

    opponents = set(blocks["opponent"].astype(str))
    if opponents != set(OPPONENTS):
        raise ResultDataError(
            f"{context} requires blocks for all opponents {list(OPPONENTS)}, "
            f"found {sorted(opponents)}"
        )
    environment_seeds = np.sort(blocks["environment_seed"].unique())
    expected_index = pd.MultiIndex.from_product(
        [environment_seeds, OPPONENTS], names=["environment_seed", "opponent"]
    )
    columns = ["candidate_wins", "reference_wins", "n"]
    aligned = blocks.set_index(["environment_seed", "opponent"]).reindex(expected_index)
    if aligned.loc[:, columns].isna().any().any():
        raise ResultDataError(
            f"{context} requires every environment seed to include all opponent conditions"
        )
    values = aligned.loc[:, columns].to_numpy(dtype=float)
    return values.reshape(len(environment_seeds), len(OPPONENTS), len(columns))


def _paired_block_ci(blocks: pd.DataFrame, reps: int, seed: int) -> tuple[float, float, float]:
    if blocks.empty:
        raise ResultDataError("cannot bootstrap an empty paired block set")
    values = blocks.loc[:, ["candidate_wins", "reference_wins", "n"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(blocks), size=(int(reps), len(blocks)))
    sample = values[choices].sum(axis=1)
    draws = sample[:, 0] / sample[:, 2] - sample[:, 1] / sample[:, 2]
    observed = _paired_values(blocks)[2]
    low, high = np.quantile(draws, (0.025, 0.975))
    return observed, float(low), float(high)


def _paired_macro_ci(blocks: pd.DataFrame, reps: int, seed: int) -> tuple[float, float, float]:
    if int(reps) <= 0:
        raise ValueError("bootstrap reps must be positive")
    values = _paired_opponent_environment_cube(blocks, context="paired macro comparison")
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(values), size=(int(reps), len(values)))
    sampled = values[choices].sum(axis=1)
    draws = (
        sampled[:, :, 0] / sampled[:, :, 2] - sampled[:, :, 1] / sampled[:, :, 2]
    ).mean(axis=1)
    observed_differences = [
        _paired_values(blocks.loc[blocks["opponent"] == opponent])[2] for opponent in OPPONENTS
    ]
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(np.mean(observed_differences)), float(low), float(high)


def _paired_summary(
    evaluation: pd.DataFrame, *, reps: int, seed: int
) -> pd.DataFrame:
    comparisons = (
        ("dqn_aware - dqn_mixed", "dqn_aware", "dqn_mixed"),
        ("dqn_aware - dqn_random", "dqn_aware", "dqn_random"),
        ("dqn_aware - dqn_aware_features_off", "dqn_aware", "dqn_aware_features_off"),
    )
    rows: list[dict[str, Any]] = []
    for comparison, candidate_name, reference_name in comparisons:
        candidate = evaluation.loc[evaluation["agent"] == candidate_name]
        reference = evaluation.loc[evaluation["agent"] == reference_name]
        blocks = _paired_block_totals(candidate, reference)
        for opponent in OPPONENTS:
            subset = blocks.loc[blocks["opponent"] == opponent]
            candidate_rate, reference_rate, difference = _paired_values(subset)
            _, low, high = _paired_block_ci(
                subset, reps, _stable_seed(f"paired:{comparison}:{opponent}", seed)
            )
            rows.append(
                {
                    "comparison": comparison,
                    "candidate": candidate_name,
                    "reference": reference_name,
                    "opponent": opponent,
                    "aggregation": "per_opponent",
                    "n_matches": int(subset["n"].sum()),
                    "n_deal_seed_blocks": int(len(subset)),
                    "candidate_win_rate": candidate_rate,
                    "reference_win_rate": reference_rate,
                    "win_rate_difference": difference,
                    "win_rate_difference_ci95_bootstrap_low": low,
                    "win_rate_difference_ci95_bootstrap_high": high,
                    "ci95_method_primary": "paired_bootstrap_shared_deal_seed_blocks_percentile",
                    "bootstrap_replicates": int(reps),
                }
            )
        candidate_rate, reference_rate, difference = _paired_values(blocks)
        overall_blocks = _paired_environment_seed_blocks(blocks)
        _, low, high = _paired_block_ci(
            overall_blocks, reps, _stable_seed(f"paired:{comparison}:pooled", seed)
        )
        rows.append(
            {
                "comparison": comparison,
                "candidate": candidate_name,
                "reference": reference_name,
                "opponent": "all",
                "aggregation": "pooled",
                "n_matches": int(blocks["n"].sum()),
                "n_deal_seed_blocks": int(len(overall_blocks)),
                "candidate_win_rate": candidate_rate,
                "reference_win_rate": reference_rate,
                "win_rate_difference": difference,
                "win_rate_difference_ci95_bootstrap_low": low,
                "win_rate_difference_ci95_bootstrap_high": high,
                "ci95_method_primary": "paired_bootstrap_shared_environment_seed_blocks_percentile",
                "bootstrap_replicates": int(reps),
            }
        )
        difference, low, high = _paired_macro_ci(
            blocks, reps, _stable_seed(f"paired:{comparison}:macro", seed)
        )
        # Equal n is enforced by the validation protocol, but calculate the two
        # displayed rates independently for clarity when readers inspect a row.
        candidate_macro = float(
            np.mean([_paired_values(blocks.loc[blocks["opponent"] == opponent])[0] for opponent in OPPONENTS])
        )
        reference_macro = float(
            np.mean([_paired_values(blocks.loc[blocks["opponent"] == opponent])[1] for opponent in OPPONENTS])
        )
        rows.append(
            {
                "comparison": comparison,
                "candidate": candidate_name,
                "reference": reference_name,
                "opponent": "all",
                "aggregation": "macro_equal_opponents",
                "n_matches": int(blocks["n"].sum()),
                "n_deal_seed_blocks": int(len(overall_blocks)),
                "candidate_win_rate": candidate_macro,
                "reference_win_rate": reference_macro,
                "win_rate_difference": difference,
                "win_rate_difference_ci95_bootstrap_low": low,
                "win_rate_difference_ci95_bootstrap_high": high,
                "ci95_method_primary": "paired_bootstrap_shared_environment_seed_blocks_percentile",
                "bootstrap_replicates": int(reps),
            }
        )
    return pd.DataFrame(rows)


def calculate_summaries(
    evaluation: pd.DataFrame,
    *,
    reps: int = BOOTSTRAP_REPS,
    bootstrap_seed: int = 20261025,
) -> dict[str, pd.DataFrame]:
    """Calculate all CSV summary tables from already validated evaluation rows."""

    if int(reps) <= 0:
        raise ValueError("bootstrap reps must be positive")
    by_opponent_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    by_seed_rows: list[dict[str, Any]] = []
    for agent in AGENTS:
        agent_frame = evaluation.loc[evaluation["agent"] == agent]
        for opponent in OPPONENTS:
            row = summarize_evaluation_frame(
                agent_frame.loc[agent_frame["opponent"] == opponent],
                aggregation="pooled",
                reps=reps,
                seed=_stable_seed(f"summary:{agent}:{opponent}", bootstrap_seed),
            )
            row.update(agent=agent, opponent=opponent, aggregation="per_opponent")
            by_opponent_rows.append(row)
        for aggregation in ("pooled", "macro_equal_opponents"):
            row = summarize_evaluation_frame(
                agent_frame,
                aggregation=aggregation,
                reps=reps,
                seed=_stable_seed(f"summary:{agent}:{aggregation}", bootstrap_seed),
            )
            row.update(agent=agent, opponent="all", aggregation=aggregation)
            overall_rows.append(row)
        for training_seed in _agent_seeds(agent):
            seed_frame = agent_frame.loc[agent_frame["training_seed"] == training_seed]
            for aggregation in ("pooled", "macro_equal_opponents"):
                row = summarize_evaluation_frame(
                    seed_frame,
                    aggregation=aggregation,
                    reps=reps,
                    seed=_stable_seed(
                        f"summary:{agent}:seed{training_seed}:{aggregation}", bootstrap_seed
                    ),
                )
                row.update(
                    agent=agent,
                    training_seed=int(training_seed),
                    opponent="all",
                    aggregation=aggregation,
                )
                by_seed_rows.append(row)
    by_opponent = _ordered_summary_frame(
        by_opponent_rows, ("agent", "opponent", "aggregation")
    ).sort_values(["agent", "opponent"], kind="stable")
    overall = _ordered_summary_frame(
        overall_rows, ("agent", "opponent", "aggregation")
    ).sort_values(["agent", "aggregation"], kind="stable")
    by_training_seed = _ordered_summary_frame(
        by_seed_rows, ("agent", "training_seed", "opponent", "aggregation")
    ).sort_values(["agent", "training_seed", "aggregation"], kind="stable")
    paired = _paired_summary(evaluation, reps=reps, seed=bootstrap_seed).sort_values(
        ["comparison", "aggregation", "opponent"], kind="stable"
    )
    return {
        "summary_by_opponent": by_opponent.reset_index(drop=True),
        "summary_overall": overall.reset_index(drop=True),
        "by_training_seed": by_training_seed.reset_index(drop=True),
        "paired_differences": paired.reset_index(drop=True),
    }


def _source_relative(path: Path | str, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4A5560",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "svg.fonttype": "none",
        }
    )


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _error_lengths(value: float, low: float, high: float) -> list[list[float]]:
    """Return non-negative matplotlib error lengths without changing saved CIs."""

    return [[max(0.0, value - low)], [max(0.0, high - value)]]


def _summary_row(frame: pd.DataFrame, agent: str) -> pd.Series:
    rows = frame.loc[
        (frame["agent"] == agent)
        & (frame["opponent"] == "all")
        & (frame["aggregation"] == "macro_equal_opponents")
    ]
    if len(rows) != 1:
        raise ResultDataError(f"expected one macro overall summary row for {agent}")
    return rows.iloc[0]


def _plot_figures(
    data: ExperimentData,
    tables: Mapping[str, pd.DataFrame],
    figures_dir: Path,
) -> dict[str, Any]:
    """Render requested plots and return an auditable numeric-data manifest."""

    _apply_plot_style()
    figure_data: dict[str, Any] = {}
    overall = tables["summary_overall"]
    by_opponent = tables["summary_by_opponent"]
    by_seed = tables["by_training_seed"]
    paired = tables["paired_differences"]

    # 1. Learning curves: rolling-50 reward for each seed faintly, then the
    # mean of those saved curves by condition.
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    learning_series: list[dict[str, Any]] = []
    learning_sources = sorted({_source_relative(path, data.root) for path in data.training_files})
    for condition in CONDITIONS:
        seed_curves: list[pd.DataFrame] = []
        for seed in TRAINING_SEEDS:
            subset = data.training.loc[
                (data.training["condition"] == condition)
                & (data.training["training_seed"] == seed)
            ].sort_values("episode")
            curve = subset.loc[:, ["episode", "reward"]].copy()
            curve["rolling_reward"] = curve["reward"].rolling(50, min_periods=1).mean()
            ax.plot(curve["episode"], curve["rolling_reward"], color=PALETTE[condition], alpha=0.22, linewidth=1)
            learning_series.append(
                {
                    "series": f"{condition}_seed{seed}_rolling50",
                    "x": curve["episode"].astype(int).tolist(),
                    "y": curve["rolling_reward"].astype(float).tolist(),
                }
            )
            seed_curves.append(curve.loc[:, ["episode", "rolling_reward"]])
        mean_curve = pd.concat(seed_curves).groupby("episode", as_index=False)["rolling_reward"].mean()
        ax.plot(mean_curve["episode"], mean_curve["rolling_reward"], color=PALETTE[condition], linewidth=2.4, label=DISPLAY_NAMES[condition])
        learning_series.append(
            {
                "series": f"{condition}_mean_rolling50",
                "x": mean_curve["episode"].astype(int).tolist(),
                "y": mean_curve["rolling_reward"].astype(float).tolist(),
            }
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Reward (rolling 50)")
    ax.legend(frameon=False, ncol=3, loc="best")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "learning_curves")
    learning_manifest = {"sources": learning_sources, "plotted_data": learning_series}

    # 2. Loss curves: per-episode means then rolling-50, entirely from loss CSV.
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    loss_series: list[dict[str, Any]] = []
    loss_sources = sorted({_source_relative(path, data.root) for path in data.loss_files})
    for condition in CONDITIONS:
        seed_curves = []
        for seed in TRAINING_SEEDS:
            subset = data.losses.loc[
                data.losses["_source_file"].str.endswith(f"{condition}_seed{seed}_loss.csv")
            ].sort_values("episode")
            episode_curve = subset.groupby("episode", as_index=False)["loss"].mean()
            episode_curve["rolling_loss"] = episode_curve["loss"].rolling(50, min_periods=1).mean()
            ax.plot(episode_curve["episode"], episode_curve["rolling_loss"], color=PALETTE[condition], alpha=0.22, linewidth=1)
            loss_series.append(
                {
                    "series": f"{condition}_seed{seed}_episode_mean_rolling50",
                    "x": episode_curve["episode"].astype(int).tolist(),
                    "y": episode_curve["rolling_loss"].astype(float).tolist(),
                }
            )
            seed_curves.append(episode_curve.loc[:, ["episode", "rolling_loss"]])
        mean_curve = pd.concat(seed_curves).groupby("episode", as_index=False)["rolling_loss"].mean()
        ax.plot(mean_curve["episode"], mean_curve["rolling_loss"], color=PALETTE[condition], linewidth=2.4, label=DISPLAY_NAMES[condition])
        loss_series.append(
            {
                "series": f"{condition}_mean_episode_rolling50",
                "x": mean_curve["episode"].astype(int).tolist(),
                "y": mean_curve["rolling_loss"].astype(float).tolist(),
            }
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Loss (episode mean, rolling 50)")
    ax.legend(frameon=False, ncol=3, loc="best")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "loss_curves")
    loss_manifest = {"sources": loss_sources, "plotted_data": loss_series}

    # 3. Overall win rates with primary block-bootstrap confidence intervals.
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    x = np.arange(len(AGENTS))
    plotted_win: list[dict[str, Any]] = []
    for index, agent in enumerate(AGENTS):
        row = _summary_row(overall, agent)
        rate = float(row["win_rate"])
        low = float(row["win_rate_ci95_bootstrap_low"])
        high = float(row["win_rate_ci95_bootstrap_high"])
        ax.bar(index, rate, color=PALETTE[agent], width=0.68)
        ax.errorbar(index, rate, yerr=_error_lengths(rate, low, high), color="#222222", capsize=3, linewidth=1.2)
        plotted_win.append({"agent": agent, "x": int(index), "y": rate, "ci95_low": low, "ci95_high": high})
    ax.set_xticks(x, [DISPLAY_NAMES[agent] for agent in AGENTS], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win rate")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "win_rates")
    win_manifest = {"sources": ["results/summaries/summary_overall.csv"], "plotted_data": plotted_win}

    # 4. Agent-by-opponent heatmap of saved win-rate summaries.
    matrix = np.empty((len(AGENTS), len(OPPONENTS)), dtype=float)
    for agent_index, agent in enumerate(AGENTS):
        for opponent_index, opponent in enumerate(OPPONENTS):
            row = by_opponent.loc[(by_opponent["agent"] == agent) & (by_opponent["opponent"] == opponent)]
            if len(row) != 1:
                raise ResultDataError(f"missing summary heatmap cell for {agent}/{opponent}")
            matrix[agent_index, opponent_index] = float(row.iloc[0]["win_rate"])
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(OPPONENTS)), [name.title() for name in OPPONENTS])
    ax.set_yticks(np.arange(len(AGENTS)), [DISPLAY_NAMES[agent] for agent in AGENTS])
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            text_colour = "white" if matrix[row_index, col_index] > 0.58 else "#202020"
            ax.text(col_index, row_index, f"{matrix[row_index, col_index]:.2f}", ha="center", va="center", color=text_colour, fontsize=9)
    colourbar = fig.colorbar(image, ax=ax, pad=0.02)
    colourbar.set_label("Win rate")
    fig.tight_layout()
    _save_figure(fig, figures_dir, "opponent_heatmap")
    heatmap_manifest = {
        "sources": ["results/summaries/summary_by_opponent.csv"],
        "plotted_data": {"matrix": matrix.astype(float).tolist(), "rows": list(AGENTS), "columns": list(OPPONENTS)},
    }

    # 5. Mean score with primary deal-block uncertainty.
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    plotted_score: list[dict[str, Any]] = []
    for index, agent in enumerate(AGENTS):
        row = _summary_row(overall, agent)
        value = float(row["mean_score"])
        low = float(row["mean_score_ci95_bootstrap_low"])
        high = float(row["mean_score_ci95_bootstrap_high"])
        ax.bar(index, value, color=PALETTE[agent], width=0.68)
        ax.errorbar(index, value, yerr=_error_lengths(value, low, high), color="#222222", capsize=3, linewidth=1.2)
        plotted_score.append({"agent": agent, "x": int(index), "y": value, "ci95_low": low, "ci95_high": high})
    ax.set_xticks(x, [DISPLAY_NAMES[agent] for agent in AGENTS], rotation=20, ha="right")
    ax.set_ylabel("Average score")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "average_score")
    score_manifest = {"sources": ["results/summaries/summary_overall.csv"], "plotted_data": plotted_score}

    # 6. Input-ablation comparison by opponent.
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ablation_agents = ("dqn_aware", "dqn_aware_features_off")
    positions = np.arange(len(OPPONENTS))
    width = 0.36
    ablation_data: list[dict[str, Any]] = []
    for offset, agent in enumerate(ablation_agents):
        values = []
        for opponent in OPPONENTS:
            row = by_opponent.loc[(by_opponent["agent"] == agent) & (by_opponent["opponent"] == opponent)].iloc[0]
            values.append(float(row["win_rate"]))
            ablation_data.append({"agent": agent, "opponent": opponent, "x": float(positions[len(values)-1] + (offset - 0.5) * width), "y": float(row["win_rate"]), "ci95_low": float(row["win_rate_ci95_bootstrap_low"]), "ci95_high": float(row["win_rate_ci95_bootstrap_high"])})
        ax.bar(positions + (offset - 0.5) * width, values, width=width, color=PALETTE[agent], label=DISPLAY_NAMES[agent])
    ax.set_xticks(positions, [name.title() for name in OPPONENTS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win rate")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "ablation")
    ablation_manifest = {"sources": ["results/summaries/summary_by_opponent.csv"], "plotted_data": ablation_data}

    # 7. Robustness against the non-stationary switching opponent.
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    switching_data: list[dict[str, Any]] = []
    for index, agent in enumerate(AGENTS):
        row = by_opponent.loc[(by_opponent["agent"] == agent) & (by_opponent["opponent"] == "switching")].iloc[0]
        value = float(row["win_rate"])
        low = float(row["win_rate_ci95_bootstrap_low"])
        high = float(row["win_rate_ci95_bootstrap_high"])
        ax.bar(index, value, color=PALETTE[agent], width=0.68)
        ax.errorbar(index, value, yerr=_error_lengths(value, low, high), color="#222222", capsize=3, linewidth=1.2)
        switching_data.append({"agent": agent, "x": int(index), "y": value, "ci95_low": low, "ci95_high": high})
    ax.set_xticks(x, [DISPLAY_NAMES[agent] for agent in AGENTS], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win rate vs switching")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "switching")
    switching_manifest = {"sources": ["results/summaries/summary_by_opponent.csv"], "plotted_data": switching_data}

    # 8. Frozen-policy seed stability; each point remains a single checkpoint.
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    stability_data: list[dict[str, Any]] = []
    stability_agents = ("dqn_random", "dqn_mixed", "dqn_aware", "dqn_aware_features_off")
    for agent in stability_agents:
        subset = by_seed.loc[
            (by_seed["agent"] == agent) & (by_seed["aggregation"] == "macro_equal_opponents")
        ].sort_values("training_seed")
        values = subset["win_rate"].to_numpy(dtype=float)
        low = subset["win_rate_ci95_bootstrap_low"].to_numpy(dtype=float)
        high = subset["win_rate_ci95_bootstrap_high"].to_numpy(dtype=float)
        seeds = subset["training_seed"].to_numpy(dtype=int)
        ax.errorbar(seeds, values, yerr=np.vstack((values - low, high - values)), marker="o", markersize=5, capsize=3, linewidth=1.8, color=PALETTE[agent], label=DISPLAY_NAMES[agent])
        stability_data.append({"agent": agent, "x": seeds.astype(int).tolist(), "y": values.astype(float).tolist(), "ci95_low": low.astype(float).tolist(), "ci95_high": high.astype(float).tolist()})
    ax.set_xticks(TRAINING_SEEDS)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Macro win rate")
    ax.legend(frameon=False, ncol=2, loc="best")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    fig.tight_layout()
    _save_figure(fig, figures_dir, "training_seed_stability")
    stability_manifest = {"sources": ["results/summaries/by_training_seed.csv"], "plotted_data": stability_data}

    # Figures are traceable to saved CSVs.  The paired table is not itself a
    # plot source above, so retain it in the manifest for auditing comparisons.
    paired_manifest = {
        "sources": ["results/summaries/paired_differences.csv"],
        "paired_summaries": paired.to_dict(orient="records"),
    }
    manifests = {
        "learning_curves": learning_manifest,
        "loss_curves": loss_manifest,
        "win_rates": win_manifest,
        "opponent_heatmap": heatmap_manifest,
        "average_score": score_manifest,
        "ablation": ablation_manifest,
        "switching": switching_manifest,
        "training_seed_stability": stability_manifest,
    }
    for stem, manifest in manifests.items():
        figure_data[f"{stem}.png"] = manifest
        figure_data[f"{stem}.svg"] = manifest
    figure_data["paired_differences.csv"] = paired_manifest
    figure_data["bootstrap_clustering"] = {
        "per_opponent": "Each percentile-bootstrap draw resamples environment_seed blocks, retaining all seats and all training checkpoints for that opponent.",
        "all_opponent": "Each pooled or macro all-opponent draw resamples shared environment_seed indices, retaining all five opponent conditions, seats, and training checkpoints together.",
        "paired_all_opponent": "Paired pooled and macro contrasts resample the same environment_seed indices jointly for candidate/reference rows across all five opponent conditions.",
    }
    return figure_data


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")


def _counts_by_agent(evaluation: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent in AGENTS:
        subset = evaluation.loc[evaluation["agent"] == agent]
        rows.append(
            {
                "agent": agent,
                "evaluation_files": len(_agent_seeds(agent)),
                "rows": int(len(subset)),
                "rows_per_opponent": int(len(subset) // len(OPPONENTS)),
                "deal_seed_blocks": int(len(_environment_seed_blocks(subset))),
                "training_seeds": list(_agent_seeds(agent)),
            }
        )
    return rows


def _analysis_metadata(data: ExperimentData, tables: Mapping[str, pd.DataFrame], reps: int) -> dict[str, Any]:
    expected_per_checkpoint_opponent = int(data.config["evaluation_seeds"]) * len(data.config["seats"])
    expected_per_checkpoint = expected_per_checkpoint_opponent * len(OPPONENTS)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "preset": "standard",
            "training_episodes_per_run": int(data.config["train_episodes"]),
            "training_seeds": list(TRAINING_SEEDS),
            "evaluation_seed_start": int(data.config["evaluation_seed_start"]),
            "evaluation_deal_seeds": int(data.config["evaluation_seeds"]),
            "seats": [int(value) for value in data.config["seats"]],
            "opponents": list(OPPONENTS),
            "agents": list(AGENTS),
        },
        "expected_counts": {
            "evaluation_rows_per_opponent_per_checkpoint": expected_per_checkpoint_opponent,
            "evaluation_rows_per_checkpoint": expected_per_checkpoint,
            "baseline_rows_per_opponent": expected_per_checkpoint_opponent,
            "trained_agent_rows_per_opponent_across_three_checkpoints": expected_per_checkpoint_opponent * len(TRAINING_SEEDS),
            "total_evaluation_rows": int(len(data.evaluation)),
            "total_training_rows": int(len(data.training)),
            "total_loss_rows": int(len(data.losses)),
        },
        "observed_counts_by_agent": _counts_by_agent(data.evaluation),
        "uncertainty": {
            "wilson": "Descriptive 95% Wilson interval for the saved match-level win proportion.",
            "primary": "95% percentile bootstrap, 3,000 deterministic replicates by default, resampling environment_seed blocks. Each per-opponent block retains all 4 seats and all 3 training checkpoints for that agent. For pooled and macro all-opponent rows, each draw retains all five opponent conditions at the selected environment seed together.",
            "paired": "Aware-versus-comparator win differences use shared candidate/reference environment-seed block resampling. Pooled and macro all-opponent rows jointly resample all five opponent conditions at each selected seed; no p-values or unsupported significance claims are reported.",
            "all_opponent_aggregation": "Both pooled and macro_equal_opponents rows are saved. Macro point estimates weight all five opponent types equally, while its bootstrap draws use the same shared environment-seed indices across those types.",
        },
        "summary_rows": {name: int(len(frame)) for name, frame in tables.items()},
        "bootstrap_replicates": int(reps),
    }


def _results_markdown(data: ExperimentData, tables: Mapping[str, pd.DataFrame], reps: int) -> str:
    counts = _counts_by_agent(data.evaluation)
    expected_per_opponent = int(data.config["evaluation_seeds"]) * len(data.config["seats"])
    expected_per_file = expected_per_opponent * len(OPPONENTS)
    lines = [
        "# Chef's Hat opponent-modelling results",
        "",
        "This report was generated only after all saved standard-protocol artefacts from the final backend (the patched official `Game` implementation with its common progress guard) passed schema, completion, legal-action, and consistency checks. It does not create or impute game results.",
        "",
        "## Exact protocol counts",
        "",
        f"- Training: **{int(data.config['train_episodes'])} episodes** for each of 3 conditions × 3 seeds = **{len(data.training):,} saved training matches**.",
        f"- Evaluation: **{int(data.config['evaluation_seeds'])} held-out deal seeds** × **{len(data.config['seats'])} seats** = **{expected_per_opponent} matches per opponent per checkpoint**.",
        f"- Each baseline configuration has **{expected_per_file} matches** across 5 opponents; each trained condition/ablation checkpoint also has **{expected_per_file} matches**.",
        f"- Each trained agent aggregate has **{expected_per_opponent * len(TRAINING_SEEDS)} matches per opponent** (all 4 seats × 3 frozen training seeds); baselines retain their single seed and are not double-counted.",
        f"- Total analysed evaluation matches: **{len(data.evaluation):,}**.",
        "",
        "| Agent | Evaluation files | Observed matches/opponent | Observed total matches | Environment-seed blocks |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in counts:
        lines.append(
            f"| {DISPLAY_NAMES[row['agent']]} | {row['evaluation_files']} | {row['rows_per_opponent']} | {row['rows']} | {row['deal_seed_blocks']} |"
        )
    lines.extend(
        [
            "",
            "## Metrics and uncertainty",
            "",
            "The CSV tables report win rate, mean score, mean position, mean actions, learner actions, pass frequency (total passes / total learner actions), rounds, duration, and performance score. `score = 4 - position` and `win = (position == 1)` were checked row by row.",
            "",
            "`win_rate_ci95_wilson_*` is a descriptive 95% Wilson interval based on saved match rows. The primary intervals (`*_ci95_bootstrap_*`) are percentile intervals from **"
            f"{reps:,} deterministic bootstrap replicates** of held-out environment-seed blocks. A per-opponent block preserves all four seats and all three frozen training seeds. For pooled and macro all-opponent rows, each selected environment-seed block also contains all five opponent conditions, so shared-deal correlations are never treated as independently resampled observations.",
            "",
            "`summary_overall.csv` and `by_training_seed.csv` include both `pooled` and `macro_equal_opponents` rows. The macro rows give random, aggressive, conservative, mixed, and switching opponents equal weight. Baseline rows use only training seed `-1`; they are not replicated to mimic learned-policy seed counts.",
            "",
            "`paired_differences.csv` compares DQN-aware with DQN-mixed, DQN-random, and the awareness-features-off ablation. Its per-opponent rows resample the same candidate/reference environment-seed blocks together; pooled and equal-opponent macro rows additionally resample the same environment-seed indices jointly across all five opponent types. The table intentionally reports confidence intervals only, not p-values or categorical significance claims.",
            "",
            "## Outputs",
            "",
            "- `results/summaries/summary_by_opponent.csv`: one row per agent/opponent.",
            "- `results/summaries/summary_overall.csv`: pooled and equal-opponent macro summaries.",
            "- `results/summaries/by_training_seed.csv`: checkpoint-level pooled and macro summaries.",
            "- `results/summaries/paired_differences.csv`: paired DQN-aware contrasts.",
            "- `results/summaries/analysis.json`: machine-readable protocol/count/uncertainty metadata.",
            "- `results/figures/`: PNG and SVG figures; `figure_data.json` records each plotted numeric series and source CSV.",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_all(
    root: Path | str = ROOT,
    *,
    output_root: Path | str | None = None,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> dict[str, Path]:
    """Validate completed experiments and generate every requested analysis artefact.

    Parameters
    ----------
    root:
        Repository root containing ``configs/``, ``results/``, and
        ``checkpoints/``.  Defaults to the project containing this module.
    output_root:
        Optional destination for generated ``summaries`` and ``figures``.  It
        exists principally for verifier regeneration; source data always comes
        from ``root/results``.  The normal destination is ``root/results``.
    bootstrap_reps:
        Number of deterministic percentile bootstrap replicates (default 3000).

    Returns
    -------
    dict[str, pathlib.Path]
        Paths of generated reports and directories.
    """

    root_path = _as_root(root)
    destination = _as_root(output_root) if output_root is not None else root_path / "results"
    if int(bootstrap_reps) <= 0:
        raise ValueError("bootstrap_reps must be positive")

    # Validation happens before output directories are created.  Thus invoking
    # this on an in-progress run leaves no fake or partial analysis dataset.
    data = load_experiment_data(root_path)
    tables = calculate_summaries(data.evaluation, reps=int(bootstrap_reps))

    summaries_dir = destination / "summaries"
    figures_dir = destination / "figures"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        _write_csv(frame, summaries_dir / f"{name}.csv")
    metadata = _analysis_metadata(data, tables, int(bootstrap_reps))
    (summaries_dir / "analysis.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    figure_data = _plot_figures(data, tables, figures_dir)
    (figures_dir / "figure_data.json").write_text(
        json.dumps(figure_data, indent=2, sort_keys=True, default=_json_default) + "\n"
    )

    # Keep the readable report at the repository root under ordinary use.  A
    # temporary destination used by the verifier gets its own sibling report.
    report_root = root_path if output_root is None else destination.parent
    report_path = report_root / "RESULTS.md"
    report_path.write_text(_results_markdown(data, tables, int(bootstrap_reps)))
    return {
        "results_markdown": report_path,
        "summaries": summaries_dir,
        "figures": figures_dir,
        "analysis_json": summaries_dir / "analysis.json",
        "figure_data": figures_dir / "figure_data.json",
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate analysis only from complete saved standard-protocol experiment CSVs."
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (default: module project root)")
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=BOOTSTRAP_REPS,
        help="deterministic deal-block bootstrap replicates (default: 3000)",
    )
    args = parser.parse_args(argv)
    try:
        outputs = generate_all(args.root, bootstrap_reps=args.bootstrap_reps)
    except (ResultDataError, ValueError) as exc:
        parser.error(str(exc))
        return 2
    print(f"Analysis written to {outputs['summaries']} and {outputs['figures']}")
    print(f"Readable report: {outputs['results_markdown']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(_main())
