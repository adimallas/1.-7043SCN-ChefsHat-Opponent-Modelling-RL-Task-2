import numpy as np
from sampath_rl.core import SingleAgentEnv
from sampath_rl.agents import RandomAgent

def test_learner_encoding_does_not_depend_on_opponent_card_identities():
    env = SingleAgentEnv([RandomAgent(10+i) for i in range(3)], seat=0, aware=True)
    env.reset(seed=20261025)
    original = env._state_for_actor(0).copy()
    # Change hidden card identities, preserving every publicly visible hand size.
    # No game action is executed with this intentionally altered test fixture.
    for player in env.game.players[1:]:
        player.cards = [1] * len(player.cards)
    changed = env._state_for_actor(0)
    np.testing.assert_array_equal(original, changed)
