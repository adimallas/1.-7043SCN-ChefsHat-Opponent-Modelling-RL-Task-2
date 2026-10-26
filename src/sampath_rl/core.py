"""Synchronous single-learner adapter for the official Chef's Hat V3 engine.

This module deliberately delegates every rule decision to
``core.game_env.game.Game`` from the pinned official source.  It does not
reimplement legal-move, discard, pizza, finishing, score, role, or card
exchange rules.  The only local work is arranging one learner's decisions,
calling opponent policies between them, and producing a fixed public-state
representation suitable for RL.

The upstream source is intentionally not vendored.  At import time this module
looks first at ``CHEFSHAT_SRC``, then at the repository's external
``.deps/ChefsHatGYM/src`` checkout created by ``scripts/install_environment.py``.
A checkout at ``/home/ubuntu/work/ChefsHatGYM/src`` is accepted only as a
convenient development fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
from numbers import Integral
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np


class Opponent(Protocol):
    """Minimal protocol expected from each non-learner player."""

    def act(self, state64: np.ndarray, mask200: np.ndarray) -> int:
        """Return one legal integer action id for the supplied observation."""


class _NoOpEngineLogger:
    """The official engine only needs an object exposing ``engine_log``."""

    @staticmethod
    def engine_log(_: str) -> None:
        return None


def _candidate_source_dirs() -> Iterable[Path]:
    """Yield possible official-source roots in the documented precedence order."""

    env_source = os.environ.get("CHEFSHAT_SRC")
    if env_source:
        yield Path(env_source).expanduser()

    repository_root = Path(__file__).resolve().parents[2]
    # ``.deps`` is the normal installation path.  The vendor candidates are
    # discovery-only compatibility paths; this package does not create or ship
    # an upstream copy there.
    yield repository_root / ".deps" / "ChefsHatGYM" / "src"
    yield repository_root / "vendor" / "chefshat_core" / "src"
    yield repository_root / "vendor" / "chefshat_core"
    yield Path("/home/ubuntu/work/ChefsHatGYM/src")


def _locate_upstream_source() -> Path:
    """Locate a source directory containing the actual official ``Game`` class."""

    checked: List[str] = []
    for supplied in _candidate_source_dirs():
        # An environment variable may name either the upstream checkout or its
        # ``src`` directory.  Do not assume a particular user-facing form.
        for candidate in (supplied, supplied / "src"):
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            checked.append(str(resolved))
            if (resolved / "core" / "game_env" / "game.py").is_file():
                return resolved

    joined = "\n  - ".join(checked)
    raise ImportError(
        "The official Chef's Hat source could not be found. Run "
        "`python scripts/install_environment.py` from the project root, or "
        "set CHEFSHAT_SRC to the upstream checkout's src directory. Checked:\n"
        f"  - {joined}"
    )


UPSTREAM_SOURCE = _locate_upstream_source()
if str(UPSTREAM_SOURCE) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SOURCE))

try:
    # Imports are deliberately from upstream's public module layout.  Keeping
    # them here, rather than copying helpers, makes all game logic authoritative.
    Game = importlib.import_module("core.game_env.game").Game
    get_high_level_actions = importlib.import_module("core.utils.rules").get_high_level_actions
except Exception as exc:  # pragma: no cover - message is valuable on bad installs
    raise ImportError(
        f"Unable to import the official Chef's Hat source at {UPSTREAM_SOURCE}. "
        "Ensure its runtime dependencies (numpy and pandas) are installed."
    ) from exc


ACTIONS: Tuple[str, ...] = tuple(get_high_level_actions())
"""The exact 200-action ordering returned by the official source."""

if len(ACTIONS) != 200:  # Defensive: a different upstream revision is not compatible.
    raise RuntimeError(
        f"Expected 200 official high-level actions, received {len(ACTIONS)} from {UPSTREAM_SOURCE}."
    )

_ACTION_TO_ID: Mapping[str, int] = {action: index for index, action in enumerate(ACTIONS)}
_ROLE_VALUE = {"chef": 0.0, "souschef": 1.0, "waiter": 2.0, "dishwasher": 3.0}


@dataclass
class _Behaviour:
    """Publicly inferable action history for one seat."""

    actions: int = 0
    passes: int = 0
    discard_actions: int = 0
    discarded_cards: int = 0
    discarded_rank_total: float = 0.0
    last_pass: bool = False
    last_discard_size: int = 0

    def observe(self, action: str, discarded: Sequence[int]) -> None:
        self.actions += 1
        if action == "pass":
            self.passes += 1
            self.last_pass = True
            self.last_discard_size = 0
            return

        self.discard_actions += 1
        self.discarded_cards += len(discarded)
        self.discarded_rank_total += float(sum(discarded))
        self.last_pass = False
        self.last_discard_size = len(discarded)

    def features(self) -> Tuple[float, float, float, float, float]:
        """Return the documented five public opponent-behaviour features."""

        pass_fraction = self.passes / self.actions if self.actions else 0.0
        mean_discard_count = (
            (self.discarded_cards / self.discard_actions) / 13.0
            if self.discard_actions
            else 0.0
        )
        mean_rank = (
            (self.discarded_rank_total / self.discarded_cards) / 12.0
            if self.discarded_cards
            else 0.0
        )
        return (
            float(pass_fraction),
            float(mean_discard_count),
            float(mean_rank),
            float(self.last_pass),
            float(self.last_discard_size / 13.0),
        )


class SingleAgentEnv:
    """Compact, synchronous RL adapter around the official four-player engine.

    Parameters
    ----------
    opponents:
        Three objects with ``act(state64, mask200) -> int``.  They map to the
        three non-learner seats in cyclic order after ``seat``: ``seat + 1``,
        ``seat + 2``, and ``seat + 3`` modulo four.
    seat:
        The learner's fixed player index in ``[0, 3]``.
    aware:
        If false (the default), state entries 47:62 are exactly zero, creating
        the opponent-modelling ablation without changing the 64-dimensional
        architecture.  If true, they contain five public history features for
        each other seat in cyclic order from the current actor.
    max_actions:
        A hard diagnostic cap on *all* official actions (learner and opponents)
        in one episode.  Reaching it returns ``truncated=True`` and never
        invents a finishing order.
    max_matches:
        Number of official matches in the game.  For match two and beyond this
        adapter calls upstream ``assign_roles``, ``get_exchange_requests`` and
        ``process_card_exchange`` before creating the next upstream match.
    trace_path:
        Optional JSONL output.  It records only learner decisions, their own
        hand, and public information; no opponent hand is written.
    shaping:
        Optional dense reward coefficient.  On every learner discard the
        reward adds ``shaping * discarded_cards``.  Thus a caller selecting
        ``shaping=0.01`` receives the training protocol's 0.01 per discarded
        card.  At its requested default zero,
        reward is exactly zero until natural game termination, when it is the
        official final-match points divided by three.  The shaping formula is
        policy-independent and is identical for aware and ablation variants.

    Notes
    -----
    ``performance_score`` in terminal ``info`` is a labelled compatibility
    calculation using the older upstream documentation formula
    ``((match_points * 10) / total_rounds) / completed_matches``.  Current
    upstream ``Game`` does not calculate or return that metric.  The real
    official score is the separate integer ``score`` (0--3) produced by
    ``Game.update_scores``.
    """

    ACTIONS = ACTIONS
    STATE_SIZE = 64
    ACTION_SIZE = len(ACTIONS)
    PLAYER_COUNT = 4
    CARDS_PER_PLAYER = 17

    def __init__(
        self,
        opponents: Sequence[Opponent],
        seat: int = 0,
        aware: bool = False,
        max_actions: int = 5000,
        max_matches: int = 1,
        trace_path: Optional[os.PathLike[str] | str] = None,
        shaping: float = 0.0,
    ) -> None:
        if len(opponents) != 3:
            raise ValueError("SingleAgentEnv requires exactly three opponent objects.")
        if isinstance(seat, bool) or not isinstance(seat, Integral) or int(seat) not in range(4):
            raise ValueError("seat must be an integer from 0 through 3.")
        if isinstance(max_actions, bool) or not isinstance(max_actions, Integral) or int(max_actions) <= 0:
            raise ValueError("max_actions must be a positive integer.")
        if isinstance(max_matches, bool) or not isinstance(max_matches, Integral) or int(max_matches) <= 0:
            raise ValueError("max_matches must be a positive integer.")

        self.opponents: Tuple[Opponent, ...] = tuple(opponents)
        self.seat = int(seat)
        self.aware = bool(aware)
        self.max_actions = int(max_actions)
        self.max_matches = int(max_matches)
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.shaping = float(shaping)

        self.game: Any = None
        self._name_by_seat = tuple(f"P{index}" for index in range(self.PLAYER_COUNT))
        self._seat_by_name = {name: index for index, name in enumerate(self._name_by_seat)}
        self._opponent_for_seat: Dict[int, Opponent] = {}
        self._seed: Optional[int] = None
        self._started_at: float = 0.0
        self._last_request: Optional[Dict[str, Any]] = None
        self._last_actor_seat: Optional[int] = None
        self._terminated = False
        self._truncated = False
        self._terminal_delivered = False
        self._total_actions = 0
        self._actions_by_seat = [0] * self.PLAYER_COUNT
        self._learner_actions = 0
        self._learner_passes = 0
        self._invalid_actions = 0
        self._behaviour = [_Behaviour() for _ in range(self.PLAYER_COUNT)]
        self._completed_rounds = 0
        self._completed_matches = 0
        self._last_match_rounds = 0
        self._last_finishing_order: List[str] = []
        self._last_match_score = 0

    # ---- Gym-like public API -------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Start a fresh official game and advance opponents to learner's turn.

        Supplying a seed seeds Python's ``random`` and NumPy's legacy global
        generator before upstream deals or exchanges cards, matching the random
        sources used by the pinned engine.
        """

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise ValueError("seed must be an integer or None.")
            self._seed = int(seed)
            random.seed(self._seed)
            np.random.seed(self._seed)
        else:
            self._seed = None

        self._started_at = time.monotonic()
        self._last_request = None
        self._last_actor_seat = None
        self._terminated = False
        self._truncated = False
        self._terminal_delivered = False
        self._total_actions = 0
        self._actions_by_seat = [0] * self.PLAYER_COUNT
        self._learner_actions = 0
        self._learner_passes = 0
        self._invalid_actions = 0
        self._behaviour = [_Behaviour() for _ in range(self.PLAYER_COUNT)]
        self._completed_rounds = 0
        self._completed_matches = 0
        self._last_match_rounds = 0
        self._last_finishing_order = []
        self._last_match_score = 0

        cyclic_other_seats = [
            (self.seat + offset) % self.PLAYER_COUNT for offset in range(1, self.PLAYER_COUNT)
        ]
        self._opponent_for_seat = dict(zip(cyclic_other_seats, self.opponents))

        # The only game implementation instantiated here is the official one.
        self.game = Game(
            player_names=list(self._name_by_seat),
            max_matches=self.max_matches,
            logger=_NoOpEngineLogger(),
            save_dataset=False,
        )
        self.game.start()
        self._begin_next_match()
        self._drive_until_learner_or_done()

        # A tiny cap can technically be exhausted while opponents precede the
        # learner on reset.  Return a learner-private terminal-style snapshot,
        # never an opponent hand, in that exceptional diagnostic situation.
        if self._truncated:
            return self._state_for_actor(self.seat), np.zeros(self.ACTION_SIZE, dtype=np.bool_)
        if self._terminated:
            return self._state_for_actor(self.seat), np.zeros(self.ACTION_SIZE, dtype=np.bool_)
        return self._state_for_actor(self.seat), self._mask_from_pending_request()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply one learner action and synchronously play opponents in between.

        An out-of-range or currently illegal action raises ``ValueError``.  It
        is deliberately not replaced with a random legal action, unlike some
        room-level convenience behavior in upstream examples.
        """

        if self.game is None:
            raise RuntimeError("Call reset() before step().")
        if self._terminated or self._truncated:
            raise RuntimeError("Episode is already finished; call reset() before step().")
        if self._last_actor_seat != self.seat or self._last_request is None:
            raise RuntimeError("Internal turn synchronization failed: learner is not the current actor.")

        action_id = self._validate_action_id(action, self._mask_from_pending_request(), actor_seat=self.seat)
        hand_before = list(self._player(self.seat).cards)
        result, discarded = self._submit_pending_action(self.seat, action_id)
        shaped_reward = self.shaping * len(discarded)
        self._write_trace(action_id, hand_before, result)

        if result.get("match_over"):
            self._finish_match_and_maybe_start_next()

        # The learner may have just emptied their hand.  This loop deliberately
        # continues through all remaining official actions in that match (and,
        # for multi-match games, until the learner's next decision or final
        # game end) rather than making up a terminal position.
        if not self._terminated and not self._truncated:
            self._drive_until_learner_or_done()

        if self._terminated:
            info = self._terminal_info()
            return (
                self._state_for_actor(self.seat),
                float(shaped_reward + self._last_match_score / 3.0),
                True,
                False,
                info,
            )
        if self._truncated:
            return (
                self._state_for_actor(self.seat),
                float(shaped_reward),
                False,
                True,
                self._truncation_info(),
            )

        return (
            self._state_for_actor(self.seat),
            float(shaped_reward),
            False,
            False,
            self._step_info(),
        )

    def run_full_game(
        self,
        learner_policy: Callable[[np.ndarray, np.ndarray], int],
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Convenience rollout through all configured official matches.

        It is intentionally small and does not alter game rules.  ``learner_policy``
        receives the same state and bool mask as ordinary callers of ``step``.
        The returned tuple is the last tuple produced by ``step``.
        """

        state, mask = self.reset(seed=seed)
        while True:
            if self._truncated:
                return state, 0.0, False, True, self._truncation_info()
            action = learner_policy(state.copy(), mask.copy())
            state, reward, terminated, truncated, info = self.step(action)
            if terminated or truncated:
                return state, reward, terminated, truncated, info
            mask = self._mask_from_pending_request()

    # ---- Official-engine lifecycle ------------------------------------------

    def _begin_next_match(self) -> None:
        """Use the upstream room lifecycle for deal, role, exchange, and match."""

        for opponent in self._opponent_for_seat.values():
            if hasattr(opponent, "reset_match"):
                opponent.reset_match()
        self.game.deal_cards()
        if self.game.current_match_count > 0:
            # This exactly mirrors the relevant room lifecycle, while using the
            # engine's own random validation fallback for unmodelled exchange
            # choices.  The exchange changes no public hand counts.
            self.game.assign_roles()
            self.game.get_exchange_requests()
            if not self.game.skip_card_exchange:
                self.game.process_card_exchange({})

        self.game.create_new_match()
        self.game.start_match()

    def _drive_until_learner_or_done(self) -> None:
        """Call only legal opponent actions until the learner or a real ending."""

        while not self._terminated and not self._truncated:
            actor_seat, _ = self._request_current_action()
            if actor_seat == self.seat:
                return

            if self._total_actions >= self.max_actions:
                self._truncated = True
                return

            state = self._state_for_actor(actor_seat)
            mask = self._mask_from_pending_request()
            opponent = self._opponent_for_seat[actor_seat]
            if not hasattr(opponent, "act"):
                raise TypeError(f"Opponent for seat {actor_seat} has no act(state64, mask200) method.")
            action = self._call_opponent(opponent, state, mask)
            action_id = self._validate_action_id(action, mask, actor_seat=actor_seat)
            result, _ = self._submit_pending_action(actor_seat, action_id)

            if result.get("match_over"):
                self._finish_match_and_maybe_start_next()

    def _request_current_action(self) -> Tuple[int, Dict[str, Any]]:
        """Ask the official game who acts next without mutating game state."""

        payload = self.game.step()
        if not payload or not payload.get("request_action"):
            raise RuntimeError("Official Game did not return an action request while the episode was active.")
        player_name = payload.get("player")
        if player_name not in self._seat_by_name:
            raise RuntimeError(f"Official Game requested unknown player {player_name!r}.")
        actor_seat = self._seat_by_name[player_name]
        self._last_request = payload
        self._last_actor_seat = actor_seat
        return actor_seat, payload

    def _submit_pending_action(self, actor_seat: int, action_id: int) -> Tuple[Dict[str, Any], List[int]]:
        """Submit a checked id to official ``Game.step`` and collect public stats."""

        if self._total_actions >= self.max_actions:
            self._truncated = True
            return {}, []
        if self._last_actor_seat != actor_seat:
            raise RuntimeError("Attempted to submit an action for a non-current player.")

        action = ACTIONS[action_id]
        hand_before = list(self._player(actor_seat).cards)
        result = self.game.step(action)
        if not result or not result.get("valid_action", False):
            # A mask from the engine should make this impossible.  Do not hide
            # the discrepancy by trying a different action.
            self._invalid_actions += 1
            raise RuntimeError(f"Official engine rejected prevalidated action {action_id}: {action!r}.")

        hand_after = list(self._player(actor_seat).cards)
        discarded = self._multiset_difference(hand_before, hand_after)
        self._total_actions += 1
        self._actions_by_seat[actor_seat] += 1
        if actor_seat == self.seat:
            self._learner_actions += 1
            if action == "pass":
                self._learner_passes += 1
        self._behaviour[actor_seat].observe(action, discarded)
        for observer_seat, opponent in self._opponent_for_seat.items():
            if observer_seat != actor_seat and hasattr(opponent, "observe_opponent_decision"):
                opponent.observe_opponent_decision()
        self._last_request = None
        self._last_actor_seat = None

        if not result.get("match_over") and self._total_actions >= self.max_actions:
            self._truncated = True
        return result, discarded

    def _finish_match_and_maybe_start_next(self) -> None:
        """Capture actual upstream results, then let upstream lifecycle continue."""

        order = list(self.game.finishing_order_last_game)
        self._last_finishing_order = order
        self._last_match_rounds = int(self.game.current_match.rounds_played)
        self._completed_rounds += self._last_match_rounds
        self._completed_matches += 1
        try:
            learner_position = order.index(self._name_by_seat[self.seat]) + 1
        except ValueError as exc:
            raise RuntimeError("Official game ended a match without learner in its finishing order.") from exc
        self._last_match_score = self._points_for_position(learner_position)

        if self.game.finished:
            self._terminated = True
            return
        self._begin_next_match()

    # ---- Validation, state, and public info ---------------------------------

    def _validate_action_id(self, action: Any, mask: np.ndarray, actor_seat: int) -> int:
        """Require a legal int id; never silently replace selected actions."""

        if isinstance(action, bool) or not isinstance(action, Integral):
            self._invalid_actions += 1
            raise ValueError(f"Seat {actor_seat} action must be an integer id in [0, {self.ACTION_SIZE - 1}].")
        action_id = int(action)
        if action_id < 0 or action_id >= self.ACTION_SIZE:
            self._invalid_actions += 1
            raise ValueError(f"Seat {actor_seat} selected out-of-range action id {action_id}.")
        if not bool(mask[action_id]):
            self._invalid_actions += 1
            raise ValueError(
                f"Seat {actor_seat} selected illegal action id {action_id} ({ACTIONS[action_id]!r}); "
                "no fallback action is used."
            )
        return action_id

    @staticmethod
    def _call_opponent(opponent: Opponent, state: np.ndarray, mask: np.ndarray) -> int:
        """Call simple policies and the baseline ``epsilon=0`` policy interface."""

        method = opponent.act
        try:
            parameters = inspect.signature(method).parameters.values()
            accepts_epsilon = any(
                parameter.name == "epsilon" or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            # Extension callables sometimes hide their signature; the project
            # baseline contract has the epsilon keyword.
            accepts_epsilon = True
        if accepts_epsilon:
            return method(state.copy(), mask.copy(), epsilon=0)
        return method(state.copy(), mask.copy())

    @staticmethod
    def _multiset_difference(before: Sequence[int], after: Sequence[int]) -> List[int]:
        """Return cards removed from a hand, preserving multiplicity."""

        remaining = list(after)
        removed: List[int] = []
        for card in before:
            if card in remaining:
                remaining.remove(card)
            else:
                removed.append(card)
        return removed

    def _mask_from_pending_request(self) -> np.ndarray:
        if self._last_request is None:
            raise RuntimeError("No official action request is available for a mask.")
        possible_actions = self._last_request.get("observation", {}).get("possible_actions", ())
        possible = set(possible_actions)
        unknown = possible.difference(_ACTION_TO_ID)
        if unknown:
            raise RuntimeError(f"Official engine returned actions absent from ACTIONS: {sorted(unknown)!r}")
        return np.fromiter((action in possible for action in ACTIONS), dtype=np.bool_, count=self.ACTION_SIZE)

    def _player(self, seat: int) -> Any:
        return self.game.players[seat]

    def _current_round(self) -> Any:
        if self.game is None or self.game.current_match is None or not hasattr(self.game.current_match, "round"):
            raise RuntimeError("No official round is active.")
        return self.game.current_match.round

    @staticmethod
    def _rank_counts(cards: Sequence[int], maximum_rank: int) -> np.ndarray:
        counts = np.zeros(maximum_rank, dtype=np.float32)
        for card in cards:
            if 1 <= int(card) <= maximum_rank:
                counts[int(card) - 1] += 1.0
        return counts

    def _state_for_actor(self, actor_seat: int) -> np.ndarray:
        """Build the fixed public state, with only the current actor's hand."""

        if self.game is None:
            raise RuntimeError("Call reset() before requesting an observation.")
        current_round = self._current_round()
        state = np.zeros(self.STATE_SIZE, dtype=np.float32)

        # 0:12 -- private only to the actor receiving this state.
        state[0:12] = self._rank_counts(self._player(actor_seat).cards, 12) / 11.0
        # 12:25 -- the board is public, including the engine's empty-board rank 13.
        state[12:25] = self._rank_counts(current_round.board, 13) / 13.0
        # 25:29 -- actor id.
        state[25 + actor_seat] = 1.0
        # 29:33 -- only public remaining-card counts for every seat.
        state[29:33] = np.asarray(
            [len(self._player(index).cards) / float(self.CARDS_PER_PLAYER) for index in range(4)],
            dtype=np.float32,
        )

        finished_names = set(self.game.current_match.finishing_order)
        finished_names.update(current_round.finishing_players)
        passed_names = set(current_round.passing_players)
        for index, name in enumerate(self._name_by_seat):
            state[33 + index] = float(name in finished_names)
            state[37 + index] = float(name in passed_names)
            role = self.game.roles.get(name)
            role_value = _ROLE_VALUE.get(str(role).lower(), 0.0) if role is not None else 0.0
            state[41 + index] = role_value / 3.0

        state[45] = min(float(self.game.current_match.rounds_played), 50.0) / 50.0
        state[46] = float(current_round.is_first_turn)

        # 47:62 -- actor-relative cyclic order, all information inferred from
        # visible action history.  ``aware=False`` leaves the exact same slots
        # zero for a clean architectural ablation.
        if self.aware:
            offset = 47
            for other_seat in ((actor_seat + 1) % 4, (actor_seat + 2) % 4, (actor_seat + 3) % 4):
                state[offset : offset + 5] = self._behaviour[other_seat].features()
                offset += 5
        # 62:64 intentionally remain zero-reserved fields.
        return state

    def _step_info(self) -> Dict[str, Any]:
        return {
            "seat": self.seat,
            "current_actor": self.seat,
            "match": int(self.game.current_match_count),
            "round": int(self.game.current_match.rounds_played),
            "actions": list(self._actions_by_seat),
            "learner_actions": int(self._learner_actions),
            "passes": int(self._learner_passes),
            "invalid_actions": int(self._invalid_actions),
            "action_mask": self._mask_from_pending_request(),
        }

    def _terminal_info(self) -> Dict[str, Any]:
        learner_name = self._name_by_seat[self.seat]
        position = self._last_finishing_order.index(learner_name) + 1
        completed_matches = max(1, self._completed_matches)
        completed_rounds = max(1, self._completed_rounds)
        legacy_performance = ((self._last_match_score * 10.0) / completed_rounds) / completed_matches
        return {
            "position": int(position),
            # This is the real environment/game score for the final official
            # match, exactly corresponding to Game.update_scores: 3, 2, 1, 0.
            "score": int(self._last_match_score),
            "environment_score": int(self._last_match_score),
            "score_source": "official Game.update_scores final-match finishing points",
            # Upstream V3 Game does not implement this; retain a clearly marked
            # legacy documentation compatibility calculation instead of calling
            # it an environment-provided score.
            "performance_score": float(legacy_performance),
            "performance_score_source": (
                "legacy documentation formula ((match_points * 10) / total_rounds) / completed_matches; "
                "not emitted by current upstream Game"
            ),
            "actions": list(self._actions_by_seat),
            "learner_actions": int(self._learner_actions),
            "passes": int(self._learner_passes),
            "invalid_actions": int(self._invalid_actions),
            "rounds": int(self._completed_rounds),
            "match_rounds": int(self._last_match_rounds),
            "matches": int(self._completed_matches),
            "seed": self._seed,
            "seat": self.seat,
            "duration_seconds": float(time.monotonic() - self._started_at),
            # Seat ids are public identities, not cards.  Names are included
            # separately for interoperability with the upstream room API.
            "finishing_order": [self._seat_by_name[name] for name in self._last_finishing_order],
            "finishing_order_names": list(self._last_finishing_order),
            "game_scores": dict(self.game.scores),
            "roles": dict(self.game.roles),
            "action_mask": np.zeros(self.ACTION_SIZE, dtype=np.bool_),
        }

    def _truncation_info(self) -> Dict[str, Any]:
        return {
            "truncation_reason": "max_actions",
            "max_actions": int(self.max_actions),
            "actions": list(self._actions_by_seat),
            "learner_actions": int(self._learner_actions),
            "passes": int(self._learner_passes),
            "invalid_actions": int(self._invalid_actions),
            "rounds": int(self._completed_rounds),
            "seed": self._seed,
            "seat": self.seat,
            "duration_seconds": float(time.monotonic() - self._started_at),
            # Explicitly no made-up result for a non-finished official match.
            "finishing_order": None,
            "action_mask": np.zeros(self.ACTION_SIZE, dtype=np.bool_),
        }

    @staticmethod
    def _points_for_position(position: int) -> int:
        return {1: 3, 2: 2, 3: 1, 4: 0}.get(position, 0)

    # ---- Optional learner-only trace ----------------------------------------

    def _write_trace(self, action_id: int, learner_hand: Sequence[int], result: Mapping[str, Any]) -> None:
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        current_round = self._current_round()
        record = {
            "event": "learner_action",
            "seed": self._seed,
            "seat": self.seat,
            "match": int(self.game.current_match_count),
            "round": int(result.get("this_round_number", current_round.this_round_number)),
            "action_id": int(action_id),
            "action": ACTIONS[action_id],
            # The learner's own private cards are allowed.  No other hand is
            # serialized, only counts publicly observable after each action.
            "learner_hand": list(learner_hand),
            "public": {
                "board": list(current_round.board),
                "remaining_cards": [len(self._player(index).cards) for index in range(4)],
                "finished": [
                    self._name_by_seat[index] in set(self.game.current_match.finishing_order)
                    or self._name_by_seat[index] in set(current_round.finishing_players)
                    for index in range(4)
                ],
                "passed": [self._name_by_seat[index] in set(current_round.passing_players) for index in range(4)],
                "roles": dict(self.game.roles),
                "match_over": bool(result.get("match_over", False)),
            },
        }
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def run_full_game(
    env: SingleAgentEnv,
    learner_policy: Callable[[np.ndarray, np.ndarray], int],
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    """Module-level convenience alias for :meth:`SingleAgentEnv.run_full_game`."""

    return env.run_full_game(learner_policy=learner_policy, seed=seed)


__all__ = ["ACTIONS", "UPSTREAM_SOURCE", "Opponent", "SingleAgentEnv", "run_full_game"]
