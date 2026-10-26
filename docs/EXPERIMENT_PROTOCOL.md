# Fixed experiment protocol

## Research question

Does exposure to diverse opponents and explicit public-behaviour features improve held-out performance in Chef’s Hat? The assigned opponent-modelling variant follows `17522555 % 7 = 1`.

## Comparisons

Random and rule-based heuristic agents establish untrained baselines. DQN-Random learns against three random opponents. DQN-Mixed learns against one random, one aggressive and one conservative opponent, with their seat order rotated deterministically. DQN-Opponent-Aware uses the same mixed schedule and the same 64→128→128→200 network, but its 15 public-behaviour inputs are active. Those inputs are zeroed in the two other DQN variants. All DQN variants share training budgets, initialisation seeds, optimisation parameters, reward shaping, and seat rotation.

The standard preset completed 600 single-match games per condition for independent training seeds 17, 29 and 43, giving 5,400 final training matches. No checkpoint was selected using held-out outcomes: only the final-budget checkpoints are evaluated. Epsilon decays linearly from 1.0 to 0.05 over the first 80% of matches. The quick preset is pipeline validation only and is kept separate from final results.

## Held-out evaluation

Every baseline and every trained checkpoint faces random, aggressive, conservative, mixed and within-match switching opponents. Each opponent condition uses 32 deal seeds beginning at 9,000,000 and all four learner seats, giving 128 complete matches per checkpoint per condition. Training and evaluation seeds do not overlap. Evaluation disables exploration, gradient updates and shaping. Opponents never observe other players’ private cards. Public behaviour resets at each game; there is no claim of cross-game memory or learned recurrent opponent inference.

The additional feature-off ablation evaluates the exact same opponent-aware checkpoints after zeroing the public-behaviour inputs. This tests input reliance but also induces a distribution shift; it does not replace the controlled independently trained DQN-Mixed comparison.

## Metrics and uncertainty

First-place win rate, finishing points from the official engine, mean finishing position, complete-match action counts, passing frequency, invalid actions, and the documentation-derived speed-normalised performance statistic are saved at match level. The upstream V3 engine does not supply the older documented performance-score field; the derived statistic is labelled as such. Training rewards, per-update Huber losses, decisions and wall time are recorded separately.

Win-rate Wilson intervals are descriptive; games sharing a deal seed and trained checkpoint are correlated. Primary comparative intervals use paired bootstrap resampling of 32 evaluation deal-seed blocks, retaining all seats, training seeds and opponent conditions sharing each deal. Between-training-seed variability is reported separately. Three training seeds and a finite held-out set do not establish general superiority across opponents or human play. Raw data and analysis scripts must support every numerical claim.

## Game bounds and failures

Each game has a 5,000-action diagnostic bound. Exceeding it is a failure/truncation, not a completed game. No fabricated finishing order or substituted score is used. Progress appears every 25 training games, and checkpoints include replay, optimiser, target network, counters and RNG state. Interrupted results are trimmed back to the last checkpoint boundary before authentic resumption. The final backend is the official `Game` at `bf882e55eccd537d189f8b99fef1ad104339c92b` plus the explicitly documented one-line two-joker patch; it is not represented as untouched upstream rules.

## Experimental provenance

Saved results correspond to completed final-budget runs. Experimental conditions, evaluation seeds, software versions and environment modifications are recorded alongside the analysis.

## Quick-validation corrections and final restart

The original quick test exposed an empty-board passing loop; its failed results are archived in `results/quick_failed/`. Final experiments therefore share a separate policy-level progress mask across **all** learners, baselines and opponents: if an empty board has at least one officially legal discard, pass is not selected. This applies to replay/bootstrap masks as well as acting and does not alter the engine legality mask. Separately, the final backend applies the one-line joker patch (`jokers == 1` to `jokers >= 1`) so existing `C0;Q0;J1` is available with two jokers. Successful pre-patch quick output is in `results/quick_pre_joker_patch/`, while current patched quick output is in `results/quick/`. All partial pre-patch final training is archived in `results/pre_joker_patch/` and `checkpoints/pre_joker_patch/`; every final condition restarted from fresh weights and its original seed. Details are in `PASS_LOOP_DIAGNOSIS.md` and `TWO_JOKER_FIX.md`. Any final results are conditional on both the patched backend and the shared progress guard, not untouched-upstream compliance.
