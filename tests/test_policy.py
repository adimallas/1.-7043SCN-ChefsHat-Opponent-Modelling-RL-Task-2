import numpy as np
from sampath_rl.core import ACTIONS, SingleAgentEnv
from sampath_rl.policy import progress_mask, ProgressPolicy, PASS_ID
from sampath_rl.agents import RandomAgent

def test_progress_constraint_preserves_official_mask():
    state = np.zeros(64, dtype=np.float32)
    state[24] = 1 / 13
    legal = np.zeros(200, dtype=bool)
    legal[0] = legal[PASS_ID] = True
    choice = progress_mask(state, legal)
    assert legal[PASS_ID] and not choice[PASS_ID] and choice[0]
    state[24] = 0
    assert progress_mask(state, legal)[PASS_ID]
    legal[0] = False
    state[24] = 1 / 13
    assert progress_mask(state, legal)[PASS_ID]

def test_always_pass_when_possible_policy_completes_with_shared_constraint():
    class PreferPass:
        def act(self, state, mask, epsilon=0):
            return PASS_ID if mask[PASS_ID] else int(np.flatnonzero(mask)[-1])
    env = SingleAgentEnv([ProgressPolicy(RandomAgent(i+20)) for i in range(3)], max_actions=5000)
    policy = ProgressPolicy(PreferPass())
    state, mask = env.reset(seed=8000000)
    while True:
        state, reward, terminal, truncated, info = env.step(policy.act(state, mask))
        mask = info['action_mask']
        if terminal or truncated:
            break
    assert terminal and not truncated
    assert info['invalid_actions'] == 0
