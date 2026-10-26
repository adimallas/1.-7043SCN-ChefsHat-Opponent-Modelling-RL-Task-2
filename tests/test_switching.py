import numpy as np
from sampath_rl.agents import HeuristicAgent, RandomAgent
from sampath_rl.core import SingleAgentEnv
from sampath_rl.policy import ProgressPolicy

def test_official_public_action_notifications_drive_switching():
    class RecordingSwitch(HeuristicAgent):
        def __init__(self):
            super().__init__(1, strategy='switching', switch_interval=5)
            self.seen = set()
        def act(self, state, mask, epsilon=0):
            self.seen.add(self.active_strategy)
            return super().act(state, mask, epsilon)
    switch = RecordingSwitch()
    env = SingleAgentEnv([ProgressPolicy(switch),ProgressPolicy(RandomAgent(2)),ProgressPolicy(RandomAgent(3))])
    policy=ProgressPolicy(RandomAgent(4))
    state,mask=env.reset(20261025)
    while True:
        state,reward,terminal,truncated,info=env.step(policy.act(state,mask))
        mask=info['action_mask']
        if terminal or truncated:break
    assert terminal and not truncated
    assert switch.seen == {'aggressive','conservative'}
    assert switch.opponent_decisions > 5
