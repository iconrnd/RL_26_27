"""Agents. Every agent has act(state) -> (move, policy), where policy is the
probability vector of length state.num_actions from which the move was drawn.
"""

import numpy as np

from .go import PASS, move_to_str, parse_move
from .mcts import search    # mcts.py must not import this module (circular import)


class RandomAgent:
    """Stage 1. Plays uniformly among state.playout_moves(); the policy is
    uniform over those moves."""

    def __init__(self, rng):
        raise NotImplementedError

    def act(self, state) -> tuple[int, np.ndarray]:
        raise NotImplementedError


class HeuristicAgent:
    """Stage 1, Go only. Plays, in order of priority: a placement that
    captures, preferring the largest capture (ties at random); otherwise a
    uniformly random playout move that does not leave the group of the placed
    stone with a single liberty; otherwise a uniformly random playout move.
    The policy is one-hot at the chosen move."""

    def __init__(self, rng):
        raise NotImplementedError

    def act(self, state) -> tuple[int, np.ndarray]:
        raise NotImplementedError


class HumanAgent:
    """Stage 1, Go only. Prints the board and reads a move such as "C3" or
    "pass" with input(), asking again after invalid or illegal input. The
    policy is one-hot at the chosen move."""

    def __init__(self):
        raise NotImplementedError

    def act(self, state) -> tuple[int, np.ndarray]:
        raise NotImplementedError


class MCTSAgent:
    """Stage 2, extended in Stage 3.

    act() runs search(state, evaluator, n_simulations, rule, cfg, rng,
    root_noise=self_play) and returns the normalized root visit counts
    N(s0, a) / N(s0) as the policy. The move is the most visited root move,
    ties broken at random. rule is "uct" (Stage 2) or "puct" (Stage 3).

    Stage 3, self_play=True: the search adds root noise, and while
    state.move_number < cfg.temp_moves the move is drawn with probability
    N(s0, a) / N(s0) instead (temperature 1). The returned policy is the
    normalized visit counts in every case.
    """

    def __init__(self, evaluator, n_simulations, rule, cfg, rng, self_play=False):
        raise NotImplementedError

    def act(self, state) -> tuple[int, np.ndarray]:
        raise NotImplementedError


class PolicyAgent:
    """Stage 3, Go only. Plays the legal move with the highest prior of the
    network, without search; the policy is the prior of NetEvaluator."""

    def __init__(self, net):
        raise NotImplementedError

    def act(self, state) -> tuple[int, np.ndarray]:
        raise NotImplementedError
