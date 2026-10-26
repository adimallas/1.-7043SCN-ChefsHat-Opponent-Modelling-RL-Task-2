# Chef's Hat opponent-modelling results

The saved standard-protocol artefacts from the final backend (the patched official `Game` implementation with its common progress guard) passed schema, completion, legal-action and consistency checks. Reported performance values are computed from the retained match-level records.

## Exact protocol counts

- Training: **600 episodes** for each of 3 conditions × 3 seeds = **5,400 saved training matches**.
- Evaluation: **32 held-out deal seeds** × **4 seats** = **128 matches per opponent per checkpoint**.
- Each baseline configuration has **640 matches** across 5 opponents; each trained condition/ablation checkpoint also has **640 matches**.
- Each trained agent aggregate has **384 matches per opponent** (all 4 seats × 3 frozen training seeds); baselines retain their single seed and are not double-counted.
- Total analysed evaluation matches: **8,960**.

| Agent | Evaluation files | Observed matches/opponent | Observed total matches | Environment-seed blocks |
|---|---:|---:|---:|---:|
| Random | 1 | 128 | 640 | 32 |
| Heuristic | 1 | 128 | 640 | 32 |
| DQN random | 3 | 384 | 1920 | 32 |
| DQN mixed | 3 | 384 | 1920 | 32 |
| DQN aware | 3 | 384 | 1920 | 32 |
| Aware, features off | 3 | 384 | 1920 | 32 |

## Metrics and uncertainty

The CSV tables report win rate, mean score, mean position, mean actions, learner actions, pass frequency (total passes / total learner actions), rounds, duration, and performance score. `score = 4 - position` and `win = (position == 1)` were checked row by row.

`win_rate_ci95_wilson_*` is a descriptive 95% Wilson interval based on saved match rows. The primary intervals (`*_ci95_bootstrap_*`) are percentile intervals from **3,000 deterministic bootstrap replicates** of held-out environment-seed blocks. A per-opponent block preserves all four seats and all three frozen training seeds. For pooled and macro all-opponent rows, each selected environment-seed block also contains all five opponent conditions, so shared-deal correlations are never treated as independently resampled observations.

`summary_overall.csv` and `by_training_seed.csv` include both `pooled` and `macro_equal_opponents` rows. The macro rows give random, aggressive, conservative, mixed, and switching opponents equal weight. Baseline rows use only training seed `-1`; they are not replicated to mimic learned-policy seed counts.

`paired_differences.csv` compares DQN-aware with DQN-mixed, DQN-random, and the awareness-features-off ablation. Its per-opponent rows resample the same candidate/reference environment-seed blocks together; pooled and equal-opponent macro rows additionally resample the same environment-seed indices jointly across all five opponent types. The table intentionally reports confidence intervals only, not p-values or categorical significance claims.

## Outputs

- `results/summaries/summary_by_opponent.csv`: one row per agent/opponent.
- `results/summaries/summary_overall.csv`: pooled and equal-opponent macro summaries.
- `results/summaries/by_training_seed.csv`: checkpoint-level pooled and macro summaries.
- `results/summaries/paired_differences.csv`: paired DQN-aware contrasts.
- `results/summaries/analysis.json`: machine-readable protocol/count/uncertainty metadata.
- `results/figures/`: PNG and SVG figures; `figure_data.json` records each plotted numeric series and source CSV.
