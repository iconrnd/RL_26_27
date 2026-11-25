"""Stage 2 public tests: matches and the gate. Provided; not to be edited."""

import numpy as np

from az5.arena import gate, play_match
from az5.config import Config
from helpers import FirstMoveAgent, PassAgent

# FirstMoveAgent fills the board while PassAgent passes, so FirstMoveAgent
# wins every game with either colour.


def test_play_match_counts_wins_by_colour():
    r = play_match(FirstMoveAgent(), PassAgent(), 3, Config(), np.random.default_rng(0))
    assert r.games == 6 and len(r.records) == 6
    assert (r.wins_a, r.wins_a_black, r.wins_a_white) == (6, 3, 3)
    assert len(r.seconds_per_move) == 2 and min(r.seconds_per_move) >= 0


def test_play_match_with_agents_exchanged():
    r = play_match(PassAgent(), FirstMoveAgent(), 3, Config(), np.random.default_rng(0))
    assert (r.wins_a, r.wins_a_black, r.wins_a_white) == (0, 0, 0)


def test_both_games_of_a_pair_share_the_opening():
    # One deterministic agent on both sides: the two games of a pair are
    # identical, and different pairs have different random openings.
    r = play_match(FirstMoveAgent(), FirstMoveAgent(), 5, Config(), np.random.default_rng(0))
    assert r.games == 10
    assert 1 < r.distinct_games <= 5


def test_gate():
    cfg = Config(gate_pairs=2)
    rng = np.random.default_rng(0)
    assert gate(FirstMoveAgent(), PassAgent(), cfg, rng) == (True, 1.0)
    assert gate(PassAgent(), FirstMoveAgent(), cfg, rng) == (False, 0.0)
