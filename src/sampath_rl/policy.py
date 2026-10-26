"""Shared policy-level progress constraint, separate from official legality.

The pinned engine permits passing on an empty board. In a two-player round,
that can restart the round with the same leader indefinitely. We therefore
restrict *policy choice*, for every learner/baseline/opponent equally: if the
board is empty and at least one discard is officially legal, do not pass.
The Game, its rules and the adapter's official legal mask remain unmodified.
"""
import numpy as np
from .core import ACTIONS

PASS_ID = ACTIONS.index("pass")

def progress_mask(state, official_mask):
    mask = np.asarray(official_mask, dtype=bool).copy()
    if float(state[24]) > 0 and mask.sum() > int(mask[PASS_ID]):
        mask[PASS_ID] = False
    return mask

class ProgressPolicy:
    def __init__(self, policy):
        self.policy = policy

    def act(self, state, mask, epsilon=0.0):
        return self.policy.act(state, progress_mask(state, mask), epsilon=epsilon)

    def observe_opponent_decision(self, count=1):
        if hasattr(self.policy, "observe_opponent_decision"):
            self.policy.observe_opponent_decision(count)

    def reset_match(self):
        if hasattr(self.policy, "reset_match"):
            self.policy.reset_match()
