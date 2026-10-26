# Switching-opponent event-hook validation

The first complete held-out evaluation produced exactly identical aggressive and switching outcomes. Inspection found that the switching heuristic’s `observe_opponent_decision` counter existed, but the environment adapter had not notified it of public actions. The initial switching configuration therefore behaved as aggressive throughout a match and cannot support a non-stationarity claim.

The adapter now notifies each opponent policy once for every publicly observed action by another player. The progress wrapper forwards this count without passing any private information. The switching policy alternates between aggressive and conservative every 20 such opponent decisions. The counter resets per match. An integration test verifies that both modes actually occur during a complete official game.

Every held-out evaluation file was rerun from the same frozen final-budget checkpoints, common seeds and seats. The original misconfigured evaluation remains in `results/evaluation_pre_switching_fix/`, outside final summaries. Training is unaffected: the three training conditions use random or fixed mixed opponents, not switching opponents. No performance score is changed by hand.

Aggregate bootstrap intervals also preserve all opponent conditions that share a deal seed in the same resampled block. This retains dependency across seats, training checkpoints and opponent conditions. The corrected summaries and figures are regenerated solely from final CSVs.
