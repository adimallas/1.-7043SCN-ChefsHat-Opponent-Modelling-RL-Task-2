#!/usr/bin/env python3
"""Verify completed Chef's Hat experiment artefacts and generated analysis.

The script is intentionally a verifier, not a repair tool: it does not invent
rows, edit CSVs, or regenerate missing training/evaluation data.  It validates
the standard protocol through :mod:`sampath_rl.analysis`, loads every trusted
frozen DQN checkpoint once, and recomputes summary tables for comparison with
saved outputs.

Run from the repository root or provide an explicit root::

    python scripts/verify_results.py --root /path/to/repository
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sampath_rl import analysis


SUMMARY_FILES = (
    "summary_by_opponent.csv",
    "summary_overall.csv",
    "by_training_seed.csv",
    "paired_differences.csv",
)
FIGURE_STEMS = (
    "learning_curves",
    "loss_curves",
    "win_rates",
    "opponent_heatmap",
    "average_score",
    "ablation",
    "switching",
    "training_seed_stability",
)


def _compare_summary_frames(existing: pd.DataFrame, regenerated: pd.DataFrame, path: Path) -> list[str]:
    """Compare non-floats exactly and floats with a narrow CSV round-trip tolerance."""

    issues: list[str] = []
    if list(existing.columns) != list(regenerated.columns):
        issues.append(f"{path}: summary columns differ from regenerated output")
        return issues
    if len(existing) != len(regenerated):
        issues.append(f"{path}: row count {len(existing)} differs from regenerated {len(regenerated)}")
        return issues
    columns = list(existing.columns)
    # Summary writers sort deterministically, but sorting makes comparison robust
    # to a harmless manual row-order change while retaining exact field checks.
    sort_columns = [column for column in columns if not pd.api.types.is_float_dtype(regenerated[column])]
    if sort_columns:
        existing = existing.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        regenerated = regenerated.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    for column in columns:
        left = existing[column]
        right = regenerated[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if pd.api.types.is_float_dtype(left) or pd.api.types.is_float_dtype(right):
                left_values = left.to_numpy(dtype=float)
                right_values = right.to_numpy(dtype=float)
                if not np.allclose(left_values, right_values, rtol=1e-10, atol=1e-12, equal_nan=True):
                    max_error = float(np.nanmax(np.abs(left_values - right_values)))
                    issues.append(f"{path}: float column {column!r} differs from regeneration (max abs error {max_error:.3g})")
            elif not np.array_equal(left.to_numpy(), right.to_numpy()):
                issues.append(f"{path}: integer column {column!r} differs from regeneration")
        else:
            left_values = left.fillna("<NA>").astype(str).to_numpy()
            right_values = right.fillna("<NA>").astype(str).to_numpy()
            if not np.array_equal(left_values, right_values):
                issues.append(f"{path}: non-numeric column {column!r} differs from regeneration")
    return issues


def _check_expected_input_files(data: analysis.ExperimentData) -> list[str]:
    """Reject stale or unused evaluation CSVs that are outside the protocol."""

    issues: list[str] = []
    evaluation_dir = data.root / "results" / "evaluation"
    expected = {path.name for path in data.evaluation_files}
    actual = {path.name for path in evaluation_dir.glob("*.csv")}
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extra:
        issues.append(f"{evaluation_dir}: unexpected evaluation CSVs not used by the standard analysis: {extra}")
    if missing:
        issues.append(f"{evaluation_dir}: expected evaluation CSVs missing: {missing}")

    training_dir = data.root / "results" / "training"
    expected_training = {
        *(path.name for path in data.training_files),
        *(path.name for path in data.loss_files),
        *(path.name for path in data.metadata_files),
    }
    actual_training = {path.name for path in training_dir.glob("*.csv")} | {
        path.name for path in training_dir.glob("*.json")
    }
    extra_training = sorted(actual_training - expected_training)
    if extra_training:
        issues.append(f"{training_dir}: unexpected training CSV/JSON artefacts not used by the standard analysis: {extra_training}")
    return issues


def _check_checkpoints(data: analysis.ExperimentData) -> list[str]:
    """Load each trusted checkpoint and confirm ``act`` returns a legal action."""

    issues: list[str] = []
    try:
        from sampath_rl.agents import ACTION_DIM, STATE_DIM, DQNAgent
    except Exception as exc:
        return [f"could not import DQNAgent for checkpoint verification: {exc}"]

    rng = np.random.default_rng(9274)
    state = rng.normal(size=STATE_DIM).astype(np.float32)
    mask = rng.random(ACTION_DIM) > 0.70
    mask[0] = True  # Guarantee at least one legal action regardless of RNG draw.
    for checkpoint in data.checkpoint_files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                agent = DQNAgent.load(checkpoint)
            action = int(agent.act(state, mask, epsilon=0.0))
            if action < 0 or action >= ACTION_DIM or not bool(mask[action]):
                issues.append(f"{checkpoint}: loaded agent selected illegal action {action}")
        except Exception as exc:
            issues.append(f"{checkpoint}: could not load/select a legal action: {exc}")
    return issues


def _check_generated_outputs(
    data: analysis.ExperimentData,
    regenerated: dict[str, pd.DataFrame],
) -> list[str]:
    """Check analysis artefacts, their source manifest, and regenerated tables."""

    issues: list[str] = []
    summaries = data.root / "results" / "summaries"
    figures = data.root / "results" / "figures"
    for filename in SUMMARY_FILES:
        path = summaries / filename
        if not path.is_file():
            issues.append(f"missing generated summary: {path}")
            continue
        try:
            existing = pd.read_csv(path)
        except Exception as exc:
            issues.append(f"could not parse generated summary {path}: {exc}")
            continue
        issues.extend(_compare_summary_frames(existing, regenerated[path.stem], path))
    analysis_json = summaries / "analysis.json"
    if not analysis_json.is_file():
        issues.append(f"missing analysis metadata: {analysis_json}")
    else:
        try:
            metadata = json.loads(analysis_json.read_text())
            if int(metadata.get("bootstrap_replicates", -1)) != analysis.BOOTSTRAP_REPS:
                issues.append(f"{analysis_json}: bootstrap_replicates must be {analysis.BOOTSTRAP_REPS}")
        except Exception as exc:
            issues.append(f"could not parse analysis metadata {analysis_json}: {exc}")
    report = data.root / "RESULTS.md"
    if not report.is_file():
        issues.append(f"missing readable report: {report}")

    figure_manifest_path = figures / "figure_data.json"
    manifest: dict[str, object] = {}
    if not figure_manifest_path.is_file():
        issues.append(f"missing figure-data manifest: {figure_manifest_path}")
    else:
        try:
            manifest = json.loads(figure_manifest_path.read_text())
        except Exception as exc:
            issues.append(f"could not parse figure-data manifest {figure_manifest_path}: {exc}")
    for stem in FIGURE_STEMS:
        for suffix in (".png", ".svg"):
            artifact = figures / f"{stem}{suffix}"
            if not artifact.is_file() or artifact.stat().st_size == 0:
                issues.append(f"missing or empty figure: {artifact}")
            if manifest and f"{stem}{suffix}" not in manifest:
                issues.append(f"{figure_manifest_path}: missing numeric/source mapping for {stem}{suffix}")
    if manifest:
        paired_entry = manifest.get("paired_differences.csv")
        if not isinstance(paired_entry, dict) or "paired_summaries" not in paired_entry:
            issues.append(f"{figure_manifest_path}: paired comparison source/data mapping is absent")
        for name, entry in manifest.items():
            if not isinstance(entry, dict):
                issues.append(f"{figure_manifest_path}: {name} is not a mapping")
                continue
            for source in entry.get("sources", []):
                source_path = data.root / str(source)
                if not source_path.is_file():
                    issues.append(f"{figure_manifest_path}: listed source does not exist: {source}")
    return issues


def verify_results(root: Path | str = ROOT, *, check_checkpoints: bool = True) -> list[str]:
    """Return all detected protocol and generated-output verification errors."""

    root_path = Path(root).expanduser().resolve()
    try:
        data = analysis.load_experiment_data(root_path)
    except analysis.ResultDataError as exc:
        return [str(exc)]

    issues = _check_expected_input_files(data)
    if check_checkpoints:
        issues.extend(_check_checkpoints(data))

    # This regenerates summary values solely in memory from the already saved
    # source CSVs.  Non-floats must match exactly; float differences are limited
    # to the explicit CSV serialisation tolerance in _compare_summary_frames.
    try:
        regenerated = analysis.calculate_summaries(
            data.evaluation,
            reps=analysis.BOOTSTRAP_REPS,
        )
    except Exception as exc:
        issues.append(f"could not regenerate analysis summaries: {exc}")
        return issues
    issues.extend(_check_generated_outputs(data, regenerated))
    return issues


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify standard experiment data and analysis outputs.")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    parser.add_argument(
        "--skip-checkpoint-load",
        action="store_true",
        help="skip trusted DQN checkpoint load/legal-action checks (not recommended for final verification)",
    )
    args = parser.parse_args(argv)
    issues = verify_results(args.root, check_checkpoints=not args.skip_checkpoint_load)
    if issues:
        print("VERIFY FAILED")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("VERIFY OK: standard CSVs, metadata, checkpoints, summaries, figures, and source mappings agree.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())
