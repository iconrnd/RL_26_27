"""Games and intervals (Stage 1); matches and the gate (Stage 2)."""

import time
from dataclasses import dataclass

import numpy as np

from .data import GameRecord
from .go import BLACK, WHITE, GoState


def play_game(black, white, state=None) -> GameRecord:
    """Stage 1. Plays from `state` (default: GoState()) to the end of the game
    and returns its record. `black` plays the first player (to_play == +1),
    `white` the second; the record holds the moves played from `state` on."""
    raise NotImplementedError


def win_rate_ci(wins, n) -> tuple[float, float, float]:
    """Stage 1. (w / n, low, high): the share of w successes in n games and
    its 95% Wilson score interval (specification, Section 5.1)."""
    raise NotImplementedError


@dataclass
class MatchResult:
    wins_a: int                  # games won by agent A
    games: int                   # games played
    wins_a_black: int            # games won by A as Black
    wins_a_white: int            # games won by A as White
    distinct_games: int          # games whose opening and moves differ from all others
    seconds_per_move: tuple      # mean time per move: (agent A, agent B)
    records: list                # GameRecord of every game


def play_match(agent_a, agent_b, n_pairs, cfg, rng) -> MatchResult:
    """Stage 2. Plays n_pairs pairs of Go games. Both games of a pair start
    from the same opening of cfg.opening_plies moves, each drawn uniformly
    (with rng) from the playout moves; agent A plays Black in the first game
    of the pair and White in the second. The records hold the moves played
    after the opening."""
    raise NotImplementedError


def gate(candidate, incumbent, cfg, rng) -> tuple[bool, float]:
    """Stage 2. Plays cfg.gate_pairs pairs (candidate as agent A) and returns
    (win rate >= cfg.gate_threshold, win rate of the candidate)."""
    raise NotImplementedError
