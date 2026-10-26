"""Action-selection policies and a CPU Double-DQN learner for Sampath RL.

The high-level action ordering is deliberately imported from :mod:`.core`, so
all policies use the same 200 action indices as the environment.  States are
64-dimensional; the first twelve entries are a player's hand-count features
and entries 12:25 are board counts.  The heuristic policies use the hand
counts only for resource-conservation decisions--the action mask remains the
source of truth for game legality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import re
from typing import Any, Mapping, Optional
import warnings

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .core import ACTIONS

# Training small fully-connected models is markedly slower (and unnecessarily
# noisy in multi-agent experiments) when PyTorch starts one worker per host
# core.  Agents are CPU-only by design, so keep this process setting bounded.
torch.set_num_threads(1)

STATE_DIM = 64
ACTION_DIM = 200

if len(ACTIONS) != ACTION_DIM:
    raise ValueError(
        f"core.ACTIONS must contain exactly {ACTION_DIM} actions; got {len(ACTIONS)}"
    )

_ACTION_RE = re.compile(r"^C(?P<rank>\d+);Q(?P<quantity>\d+);J(?P<jokers>\d+)$")


@dataclass(frozen=True)
class _ActionInfo:
    """Decoded, action-index independent description of an official action."""

    index: int
    rank: Optional[int]
    quantity: int
    jokers: int
    is_pass: bool

    @property
    def cards_played(self) -> int:
        return self.quantity + self.jokers


def _decode_actions() -> tuple[_ActionInfo, ...]:
    """Decode the official action tuple once, retaining indices exactly."""

    decoded: list[_ActionInfo] = []
    for index, raw_action in enumerate(ACTIONS):
        action = str(raw_action)
        if action.lower() == "pass":
            decoded.append(_ActionInfo(index, None, 0, 0, True))
            continue
        match = _ACTION_RE.fullmatch(action)
        if match is None:
            raise ValueError(
                "Unsupported action format in core.ACTIONS at index "
                f"{index}: {raw_action!r}"
            )
        decoded.append(
            _ActionInfo(
                index=index,
                rank=int(match.group("rank")),
                quantity=int(match.group("quantity")),
                jokers=int(match.group("jokers")),
                is_pass=False,
            )
        )
    return tuple(decoded)


_ACTION_INFO = _decode_actions()


def _as_state(state: Any) -> np.ndarray:
    """Return a finite, flattened float32 state with the fixed contract size."""

    array = np.asarray(state, dtype=np.float32).reshape(-1)
    if array.size != STATE_DIM:
        raise ValueError(f"state must have {STATE_DIM} values; got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError("state must contain only finite numeric values")
    return array


def _as_legal_mask(mask: Any, *, require_legal: bool = True) -> np.ndarray:
    """Validate and return a one-dimensional boolean action mask.

    Terminal transitions may explicitly use an all-false next-state mask; all
    action-selection calls and non-terminal transitions must include at least
    one legal action.
    """

    array = np.asarray(mask, dtype=bool).reshape(-1)
    if array.size != ACTION_DIM:
        raise ValueError(f"action mask must have {ACTION_DIM} values; got {array.size}")
    if require_legal and not bool(array.any()):
        raise ValueError("action mask contains no legal actions")
    return array


def _validate_epsilon(epsilon: float) -> float:
    epsilon = float(epsilon)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")
    return epsilon


class RandomAgent:
    """Uniformly sample from the legal actions with an agent-local NumPy RNG."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def act(self, state: Any, mask: Any, epsilon: float = 0.0) -> int:
        """Return a legal action index.

        ``epsilon`` is accepted for interchangeable policy-call signatures. A
        random policy is already fully exploratory, so its value does not
        change the uniform legal-action distribution.
        """

        _as_state(state)
        legal_mask = _as_legal_mask(mask)
        _validate_epsilon(epsilon)
        legal_actions = np.flatnonzero(legal_mask)
        return int(self.rng.choice(legal_actions))


class HeuristicAgent:
    """A legal card-management baseline with deterministic tie-breaking.

    Strategies infer card quality from Chef's Hat's official rank convention:
    a lower numbered natural card is stronger, because it can beat a higher
    card.  Thus high ranks are inexpensive/weak cards to expend first.

    ``"aggressive"`` maximizes cards discarded, prefers weak high ranks, and
    spends jokers as part of large dumps.  ``"conservative"`` preserves jokers
    and low (strong) ranks, preferring inexpensive high-rank non-joker plays;
    when pass is legal it passes rather than spend a strong resource while the
    hand is still large.  ``"heuristic"`` combines maximum discard with the
    same weak-rank and joker preservation tie-breaks.

    A ``"switching"`` policy begins aggressive and alternates aggressive and
    conservative every ``switch_interval`` observed *opponent* decisions.
    Call :meth:`observe_opponent_decision` once for each observed opponent
    action (and :meth:`reset_match` at a new match).  This makes the policy
    intentionally non-stationary within a match without conflating the
    agent's own decisions with opponent observations.
    """

    _VALID_STRATEGIES = frozenset(
        {"heuristic", "aggressive", "conservative", "switching"}
    )

    def __init__(
        self,
        seed: int,
        strategy: str = "heuristic",
        *,
        switch_interval: int = 20,
    ) -> None:
        if strategy not in self._VALID_STRATEGIES:
            options = ", ".join(sorted(self._VALID_STRATEGIES))
            raise ValueError(f"unknown strategy {strategy!r}; choose one of {options}")
        if int(switch_interval) <= 0:
            raise ValueError("switch_interval must be positive")
        self.seed = int(seed)
        self.strategy = strategy
        self.switch_interval = int(switch_interval)
        self.rng = np.random.default_rng(self.seed)
        self.opponent_decisions = 0

    @property
    def active_strategy(self) -> str:
        """The currently active concrete strategy for this policy."""

        if self.strategy != "switching":
            return self.strategy
        phase = self.opponent_decisions // self.switch_interval
        return "aggressive" if phase % 2 == 0 else "conservative"

    def observe_opponent_decision(self, count: int = 1) -> None:
        """Record one or more opponent actions for switching-policy cadence."""

        count = int(count)
        if count < 0:
            raise ValueError("observed opponent decision count cannot be negative")
        self.opponent_decisions += count

    # An explicit synonym is convenient for environments which name events by
    # their action rather than their decision point.
    observe_opponent_action = observe_opponent_decision

    def reset_match(self) -> None:
        """Reset the observed-opponent counter at a new match boundary."""

        self.opponent_decisions = 0

    @staticmethod
    def _rank_value(info: _ActionInfo) -> int:
        # Joker-only action C0;Q0;J1 has no natural weak card to rank.  It is
        # treated as resource-expensive rather than a weak rank.
        return -1 if info.rank is None else info.rank

    def _choose_heuristic_action(
        self, legal_indices: np.ndarray, hand_counts: np.ndarray
    ) -> int:
        infos = [_ACTION_INFO[int(index)] for index in legal_indices]
        pass_info = next((info for info in infos if info.is_pass), None)
        plays = [info for info in infos if not info.is_pass]
        if not plays:
            # The legal mask is nonempty, so this is necessarily pass.
            assert pass_info is not None
            return pass_info.index

        strategy = self.active_strategy
        if strategy == "aggressive":
            # Dump the most cards; then use weak high ranks and joker-supported
            # dumps before preserving any strong, low-rank card.
            best = max(
                plays,
                key=lambda info: (
                    info.cards_played,
                    self._rank_value(info),
                    info.jokers,
                    info.quantity,
                    -info.index,
                ),
            )
        elif strategy == "conservative":
            # Do not spend a joker or strong low rank unless the mask leaves no
            # cheaper play.  Small discards conserve the rest of the hand.
            best = max(
                plays,
                key=lambda info: (
                    -info.jokers,
                    self._rank_value(info),
                    -info.cards_played,
                    -info.quantity,
                    -info.index,
                ),
            )
            hand_total = float(np.clip(hand_counts, 0.0, None).sum())
            spends_strong_resource = best.jokers > 0 or (
                best.rank is not None and best.rank <= 3
            )
            if pass_info is not None and hand_total >= 8 and spends_strong_resource:
                return pass_info.index
        else:  # "heuristic": discard many cards while preserving resources.
            best = max(
                plays,
                key=lambda info: (
                    info.cards_played,
                    self._rank_value(info),
                    -info.jokers,
                    info.quantity,
                    -info.index,
                ),
            )
        return best.index

    def act(self, state: Any, mask: Any, epsilon: float = 0.0) -> int:
        """Select a legal action, using epsilon-greedy local exploration."""

        array_state = _as_state(state)
        legal_mask = _as_legal_mask(mask)
        epsilon = _validate_epsilon(epsilon)
        legal_indices = np.flatnonzero(legal_mask)
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.choice(legal_indices))
        return int(self._choose_heuristic_action(legal_indices, array_state[:12]))


class QNetwork(nn.Module):
    """Fixed architecture shared by all DQN uses (64 -> 128 -> 128 -> 200)."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.layers(state)


class ReplayBuffer:
    """Preallocated NumPy ring buffer used by :class:`DQNAgent`."""

    def __init__(self, capacity: int) -> None:
        if int(capacity) <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.states = np.zeros((self.capacity, STATE_DIM), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, STATE_DIM), dtype=np.float32)
        self.masks = np.zeros((self.capacity, ACTION_DIM), dtype=np.bool_)
        self.next_masks = np.zeros((self.capacity, ACTION_DIM), dtype=np.bool_)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)
        self.size = 0
        self.position = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        mask: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        """Insert one transition and advance the circular write position."""

        idx = self.position
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.masks[idx] = mask
        self.next_masks[idx] = next_mask
        self.dones[idx] = done
        self.position = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw a without-replacement mini-batch from populated ring entries."""

        if batch_size > self.size:
            raise ValueError(
                f"cannot sample batch_size={batch_size} from replay size={self.size}"
            )
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return {
            "states": self.states[indices].copy(),
            "actions": self.actions[indices].copy(),
            "rewards": self.rewards[indices].copy(),
            "next_states": self.next_states[indices].copy(),
            "masks": self.masks[indices].copy(),
            "next_masks": self.next_masks[indices].copy(),
            "dones": self.dones[indices].copy(),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return an independent serialization-safe snapshot of the ring."""

        return {
            "capacity": self.capacity,
            "states": self.states.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "next_states": self.next_states.copy(),
            "masks": self.masks.copy(),
            "next_masks": self.next_masks.copy(),
            "dones": self.dones.copy(),
            "size": self.size,
            "position": self.position,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a checkpointed ring while validating its fixed dimensions."""

        capacity = int(state["capacity"])
        if capacity != self.capacity:
            raise ValueError(
                "replay capacity mismatch between checkpoint "
                f"({capacity}) and configuration ({self.capacity})"
            )
        expected = {
            "states": (capacity, STATE_DIM),
            "actions": (capacity,),
            "rewards": (capacity,),
            "next_states": (capacity, STATE_DIM),
            "masks": (capacity, ACTION_DIM),
            "next_masks": (capacity, ACTION_DIM),
            "dones": (capacity,),
        }
        for name, shape in expected.items():
            value = np.asarray(state[name])
            if value.shape != shape:
                raise ValueError(
                    f"invalid replay {name} shape {value.shape}; expected {shape}"
                )
            getattr(self, name)[:] = value
        size = int(state["size"])
        position = int(state["position"])
        if not 0 <= size <= capacity or not 0 <= position < capacity:
            raise ValueError("invalid replay size or position in checkpoint")
        self.size = size
        self.position = position


class DQNAgent:
    """CPU Double DQN with legal-action masking and NumPy ring replay.

    Epsilon scheduling belongs to the caller: :meth:`act` never mutates an
    epsilon value or trains.  :meth:`observe` records exactly one learner
    transition, advances the transition counter, and performs scheduled
    training.  Checkpoints use PyTorch's pickle-based serialization and must
    therefore only be loaded from trusted, self-created files.
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "gamma": 0.99,
        "capacity": 20_000,
        "learn_start": 256,
        "update_every": 4,
        "target_interval": 250,
        "grad_clip": 10.0,
    }

    def __init__(self, seed: int, config: Optional[Mapping[str, Any]] = None) -> None:
        self.seed = int(seed)
        supplied = {} if config is None else dict(config)
        unknown = set(supplied).difference(self.DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown DQN config keys: {sorted(unknown)}")
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.config.update(supplied)
        self._validate_config()

        self.device = torch.device("cpu")
        self.rng = np.random.default_rng(self.seed)
        self.torch_rng = torch.Generator(device="cpu")
        self.torch_rng.manual_seed(self.seed)

        # Initialising under a fork gives equal networks for equal agent seeds
        # without perturbing an application's process-global torch RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.online = QNetwork().to(self.device)
            self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=float(self.config["learning_rate"])
        )
        self.replay = ReplayBuffer(int(self.config["capacity"]))
        self.steps = 0
        self.learn_steps = 0
        self.last_loss: Optional[float] = None

    def _validate_config(self) -> None:
        positive_ints = (
            "batch_size",
            "capacity",
            "learn_start",
            "update_every",
            "target_interval",
        )
        for name in positive_ints:
            value = self.config[name]
            if int(value) != value or int(value) <= 0:
                raise ValueError(f"config {name!r} must be a positive integer")
            self.config[name] = int(value)
        for name in ("learning_rate", "gamma", "grad_clip"):
            value = float(self.config[name])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"config {name!r} must be finite and positive")
            self.config[name] = value
        if self.config["gamma"] > 1.0:
            raise ValueError("config 'gamma' must not exceed 1")

    @property
    def online_net(self) -> QNetwork:
        """Compatibility-friendly explicit name for the online Q-network."""

        return self.online

    @property
    def target_net(self) -> QNetwork:
        """Compatibility-friendly explicit name for the target Q-network."""

        return self.target

    def act(self, state: Any, mask: Any, epsilon: float = 0.0) -> int:
        """Choose an epsilon-greedy legal action without changing train state."""

        array_state = _as_state(state)
        legal_mask = _as_legal_mask(mask)
        epsilon = _validate_epsilon(epsilon)
        legal_actions = np.flatnonzero(legal_mask)
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.choice(legal_actions))

        self.online.eval()
        with torch.no_grad():
            values = self.online(
                torch.as_tensor(array_state, device=self.device).unsqueeze(0)
            ).squeeze(0)
            legal_tensor = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device)
            # ``finfo.min`` rather than multiplication avoids selecting illegal
            # positive Q values and never changes valid negative Q values.
            masked_values = values.masked_fill(~legal_tensor, torch.finfo(values.dtype).min)
            return int(torch.argmax(masked_values).item())

    def observe(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        next_mask: Any,
        done: bool,
        mask: Any = None,
    ) -> Optional[float]:
        """Store a transition and run one scheduled Double-DQN update.

        ``mask`` is accepted as an optional final argument for callers that
        retain the current-state legality mask.  When omitted, the action is
        still range-checked and a permissive current mask is stored; this
        retains the requested six-argument learner API while allowing replay
        checkpoints to hold masks for both states.
        """

        array_state = _as_state(state)
        array_next_state = _as_state(next_state)
        done = bool(done)
        if not np.isfinite(float(reward)):
            raise ValueError("reward must be finite")
        action = int(action)
        if not 0 <= action < ACTION_DIM:
            raise ValueError(f"action must be in [0, {ACTION_DIM}); got {action}")

        # The public requested signature has no current mask.  An all-true
        # placeholder is valid for old callers; new callers can pass ``mask``
        # to preserve it in replay and validate that the executed action was
        # legal under that mask.
        if mask is None:
            current_mask = np.ones(ACTION_DIM, dtype=np.bool_)
        else:
            current_mask = _as_legal_mask(mask)
            if not current_mask[action]:
                raise ValueError("observed action is not legal under mask")
        future_mask = _as_legal_mask(next_mask, require_legal=not done)

        self.replay.add(
            array_state,
            action,
            float(reward),
            array_next_state,
            current_mask,
            future_mask,
            done,
        )
        self.steps += 1

        loss: Optional[float] = None
        if (
            self.replay.size >= int(self.config["learn_start"])
            and self.steps % int(self.config["update_every"]) == 0
        ):
            loss = self.learn()
        if self.steps % int(self.config["target_interval"]) == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss

    def _compute_targets(
        self,
        rewards: Tensor,
        next_states: Tensor,
        next_masks: Tensor,
        dones: Tensor,
    ) -> Tensor:
        """Compute masked Double-DQN targets without touching terminal masks."""

        rewards = rewards.to(self.device, dtype=torch.float32).reshape(-1)
        dones = dones.to(self.device, dtype=torch.bool).reshape(-1)
        if next_states.ndim != 2 or next_states.shape[1] != STATE_DIM:
            raise ValueError("next_states must have shape (batch, 64)")
        if next_masks.shape != (next_states.shape[0], ACTION_DIM):
            raise ValueError("next_masks must have shape (batch, 200)")
        if rewards.shape[0] != next_states.shape[0] or dones.shape[0] != next_states.shape[0]:
            raise ValueError("target tensors must share their batch dimension")

        bootstrap = torch.zeros_like(rewards)
        nonterminal = ~dones
        if bool(nonterminal.any()):
            nonterminal_masks = next_masks[nonterminal].to(
                self.device, dtype=torch.bool
            )
            if not bool(nonterminal_masks.any(dim=1).all()):
                raise ValueError("non-terminal transition has no legal next action")
            with torch.no_grad():
                next_online = self.online(next_states[nonterminal].to(self.device))
                next_online = next_online.masked_fill(
                    ~nonterminal_masks, torch.finfo(next_online.dtype).min
                )
                next_actions = torch.argmax(next_online, dim=1, keepdim=True)
                next_target = self.target(next_states[nonterminal].to(self.device))
                bootstrap[nonterminal] = next_target.gather(1, next_actions).squeeze(1)
        return rewards + float(self.config["gamma"]) * bootstrap

    def learn(self) -> float:
        """Run one Huber-loss Double-DQN mini-batch update and return its loss."""

        batch_size = int(self.config["batch_size"])
        batch = self.replay.sample(batch_size, self.rng)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            batch["next_states"], dtype=torch.float32, device=self.device
        )
        next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.bool, device=self.device)

        self.online.train()
        predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        target = self._compute_targets(rewards, next_states, next_masks, dones)
        loss = F.smooth_l1_loss(predicted, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), max_norm=float(self.config["grad_clip"])
        )
        self.optimizer.step()
        self.learn_steps += 1
        self.last_loss = float(loss.detach().cpu().item())
        return self.last_loss

    def save(
        self,
        path: str | Path,
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        include_replay: bool = False,
    ) -> None:
        """Save a trusted-resume checkpoint, optionally including full replay.

        The resulting ``.pt`` file is a PyTorch pickle checkpoint.  Do not
        distribute it to untrusted parties and never load a checkpoint whose
        provenance is unknown.
        """

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "format_version": 1,
            "seed": self.seed,
            "config": copy.deepcopy(self.config),
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps": self.steps,
            "learn_steps": self.learn_steps,
            "last_loss": self.last_loss,
            "metadata": {} if metadata is None else dict(metadata),
            "numpy_rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "torch_rng_state": torch.get_rng_state().clone(),
            "agent_torch_rng_state": self.torch_rng.get_state().clone(),
            "includes_replay": bool(include_replay),
        }
        if include_replay:
            checkpoint["replay"] = self.replay.state_dict()
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path, load_replay: bool = False) -> "DQNAgent":
        """Load a self-created trusted checkpoint onto CPU.

        PyTorch checkpoints use Python pickle.  Loading a malicious or
        untrusted file can execute arbitrary code, so this method intentionally
        uses ``weights_only=False`` only for files created by :meth:`save`.
        """

        warnings.warn(
            "DQNAgent.load uses pickle-based PyTorch checkpoints; only load "
            "trusted, self-created files.",
            UserWarning,
            stacklevel=2,
        )
        # ``weights_only=False`` is explicit: this checkpoint intentionally
        # holds optimizer, NumPy-RNG, metadata, and optionally replay state.
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("invalid DQN checkpoint: expected a mapping")
        required = {
            "seed",
            "config",
            "online_state_dict",
            "target_state_dict",
            "optimizer_state_dict",
            "steps",
            "learn_steps",
            "numpy_rng_state",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"invalid DQN checkpoint; missing keys: {sorted(missing)}")

        agent = cls(seed=int(checkpoint["seed"]), config=checkpoint["config"])
        agent.online.load_state_dict(checkpoint["online_state_dict"])
        agent.target.load_state_dict(checkpoint["target_state_dict"])
        agent.target.eval()
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Optimizer state loaded with map_location is already CPU, but explicitly
        # move tensors in case an older checkpoint carried a device annotation.
        for optimizer_state in agent.optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, Tensor):
                    optimizer_state[key] = value.to(agent.device)
        agent.steps = int(checkpoint["steps"])
        agent.learn_steps = int(checkpoint["learn_steps"])
        agent.last_loss = checkpoint.get("last_loss")
        agent.rng.bit_generator.state = checkpoint["numpy_rng_state"]
        if "agent_torch_rng_state" in checkpoint:
            agent.torch_rng.set_state(checkpoint["agent_torch_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])

        if load_replay:
            if not checkpoint.get("includes_replay", False) or "replay" not in checkpoint:
                raise ValueError("checkpoint does not include replay; save with include_replay=True")
            agent.replay.load_state_dict(checkpoint["replay"])
        return agent


__all__ = [
    "ACTION_DIM",
    "STATE_DIM",
    "RandomAgent",
    "HeuristicAgent",
    "QNetwork",
    "ReplayBuffer",
    "DQNAgent",
]
