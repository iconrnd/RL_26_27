"""Helpers of the public tests. Provided; not to be edited."""

import numpy as np

from az5.go import PASS, GoState, parse_move
from az5.tictactoe import TicTacToeState


def go_from(moves):
    """The GoState after `moves` (strings such as "C3" or "pass") from the
    empty board."""
    s = GoState()
    for m in moves:
        s = s.play(parse_move(m))
    return s


def ttt_from(moves):
    """The TicTacToeState after `moves` (indices 0..8) from the empty board."""
    s = TicTacToeState()
    for m in moves:
        s = s.play(m)
    return s


def one_hot(n, m):
    p = np.zeros(n, dtype=np.float32)
    p[m] = 1.0
    return p


class ScriptedAgent:
    """Plays the given moves (strings such as "C3" or "pass") in order."""

    def __init__(self, moves):
        self.moves = [parse_move(m) for m in moves]

    def act(self, state):
        m = self.moves.pop(0)
        return m, one_hot(state.num_actions, m)


class FirstMoveAgent:
    """Deterministic: plays the first of state.playout_moves()."""

    def act(self, state):
        m = state.playout_moves()[0]
        return m, one_hot(state.num_actions, m)


class PassAgent:
    """Always passes (Go only)."""

    def act(self, state):
        return PASS, one_hot(state.num_actions, PASS)
