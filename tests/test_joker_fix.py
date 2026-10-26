from sampath_rl.core import ACTIONS
from core.utils.rules import get_possible_actions

def test_two_jokers_can_discard_one_on_empty_board():
    actions = get_possible_actions([12,12], [13], is_first_round=False, is_first_turn=True)
    assert 'C0;Q0;J1' in actions
    assert 'C0;Q0;J1' in ACTIONS
    assert len(ACTIONS) == 200
