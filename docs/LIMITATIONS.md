# Limitations and responsible interpretation

These are genuine, bounded CPU experiments in a patched official Chef’s Hat engine, not a claim of state-of-the-art or a guaranteed marking outcome. All nine final training runs completed 600 matches each across three conditions and three independent seeds. Completion of this finite budget does not establish convergence.

## Environment and policy class

A source-only bootstrap is used because upstream Flit metadata fails in the pinned revision. The final backend is the official `Game` at `bf882e55eccd537d189f8b99fef1ad104339c92b` plus one documented patch: the joker-only condition changes from `jokers == 1` to `jokers >= 1`, allowing existing action `C0;Q0;J1` when two jokers are held. The current V3 API differs from older online documentation. Separately, every policy shares a disclosed progress constraint that excludes passing on an empty board when a discard is legal; otherwise an early greedy policy can restart the same two-player round indefinitely. The random baseline is uniform over the constrained legal choices, not every upstream-allowed action. Results cannot be directly compared with unrestricted random-play experiments without accounting for either the patched backend or this different policy class.

Main experiments are single-match games. Role assignment/card exchange is validated separately in a two-match integration test, but long multi-match cumulative-role competition is not evaluated. Joker special declarations are not implemented in the adapter. Apart from the disclosed two-joker compatibility patch, upstream card/round semantics are retained; no other semantics are silently fixed.

The true scoring metric is official finishing points, 3 to 0. `performance_score` is an explicitly labelled derivation of the older documentation’s speed-normalised formula, not an environment-returned V3 performance field.

## Learning and features

Behavioural features are simple running public-action summaries, not a trained opponent classifier, recurrent memory, or access to private cards. All public history resets per game. Non-aware agents keep the same network dimension with these fifteen inputs zeroed. This controls nominal parameter count but not effective feature capacity.

The fixed shaping reward encourages discarding and can change the learned policy. No shaping sensitivity experiment is claimed. Better training reward or lower Huber loss is not automatically better held-out play. Training step count can differ under equal match budgets because policies generate different numbers of learner decisions.

Switching opponents alternate heuristic tendencies within a match. This is controlled non-stationarity among hand-designed opponents, not adaptive self-play or general human behaviour. Aggressive and conservative strategies are operational definitions in code, not validated psychological models.

## Statistical scope

All four seats are rotated and common held-out deal seeds are used. Deal-seed block bootstraps retain seat/checkpoint dependence, while Wilson intervals are descriptive and assume independent Bernoulli trials. Only three independently trained checkpoints per condition limit inference about training-seed uncertainty. Comparisons across many opponents and ablations are exploratory, without a multiple-testing correction. Feature-off inference can introduce distribution shift, so it is not the same as independently retraining without features.

## Reproducibility and integrity

Wall times depend on the actual CPU architecture and competing load. Exact bitwise training reproducibility across software/architectures is not guaranteed. All partial pre-patch training is archived in `results/pre_joker_patch/` and `checkpoints/pre_joker_patch/`; every final condition restarted from fresh weights and its original seed. Final results are traceable to CSVs and the completed final-budget checkpoints; quick runs are preserved separately as `results/quick_failed/`, `results/quick_pre_joker_patch/`, and `results/quick/`. The original misconfigured switching evaluation is separately archived, not aggregated. No fabricated scores, earlier development dates, or synthetic student recording are included.

Before submission, inspect and understand the implementation, verify results against saved evidence, record the viva, publish the repository through an authorised account and confirm assessor access. The supplied restricted assignment document is not redistributed. Upstream source/assets are retrieved directly under their own terms rather than bundled.
