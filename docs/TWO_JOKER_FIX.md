# Two-joker compatibility fix and experiment restart

Final-budget training under the untouched source exposed a second non-terminating edge case. A reproduction of seed 1,700,047, learner seat 3, found two remaining players: one holding `[12, 12]` (two jokers) and one holding a rank-11 card. On an empty board, the original `get_possible_actions` allows a joker-only discard only when `jokers == 1`. With two jokers and no ordinary card, only passing is available. The round repeatedly returns to that player and the 5,000-action diagnostic bound is reached. No finishing points can legitimately be assigned.

The minimal patch changes `if jokers == 1:` to `if jokers >= 1:` at the joker-only empty-board action condition. This allows the existing action `C0;Q0;J1` to remove one joker while another is held. No new action ID is introduced: the catalogue remains 200. Other legality rules, scoring, dealing, round progression and exchanges are unchanged. The patch is saved as `upstream_joker_fix.patch` and applied explicitly by the installation script.

This is a transparent modified official environment, not a claim that the original upstream code already included the fix. Every final experiment uses the same pinned base commit plus this exact patch. The common empty-board progress policy remains separately documented. Final findings are conditional on these choices.

All pre-patch training outputs, including the completed random-opponent runs and interrupted mixed/aware runs, were moved to `results/pre_joker_patch/` and `checkpoints/pre_joker_patch/`. The quick validations are also separated by backend version. Every condition is restarted from its original training seed and fresh weights after the patch; pre-patch results are not combined into final summaries or described as completed final experiments.

Regression tests check that the joker-only action is available with two jokers on an empty board and that the 200-action mapping is unchanged. Environment provenance records the base commit, patch SHA-256, actual source diff and software versions.
