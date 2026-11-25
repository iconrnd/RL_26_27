"""Stage 1 public tests: random, heuristic, and human agents; games;
intervals. Provided; not to be edited."""

import numpy as np
import pytest

from az5.agents import HeuristicAgent, HumanAgent, RandomAgent
from az5.arena import play_game, win_rate_ci
from az5.go import BLACK, PASS, GoState, parse_move
from az5.tictactoe import TicTacToeState
from helpers import FirstMoveAgent, ScriptedAgent, go_from, ttt_from


def test_random_agent_on_the_empty_board():
    m, pi = RandomAgent(np.random.default_rng(0)).act(GoState())
    assert m in range(25)
    assert len(pi) == 26 and pi.sum() == pytest.approx(1.0)
    assert np.allclose(pi[:25], 1 / 25) and pi[PASS] == 0


def test_random_agent_plays_playout_moves_only():
    s = go_from(["B5", "E1", "A4", "E2"])        # A5 is an own eye of Black
    agent = RandomAgent(np.random.default_rng(0))
    moves = set()
    for _ in range(300):
        m, pi = agent.act(s)
        moves.add(m)
        assert pi[parse_move("A5")] == 0
    assert moves == set(s.playout_moves())


def test_random_agent_on_tictactoe():
    s = ttt_from([4])
    m, pi = RandomAgent(np.random.default_rng(0)).act(s)
    assert m in s.legal_moves()
    assert len(pi) == 9 and pi[4] == 0 and pi.sum() == pytest.approx(1.0)


def test_heuristic_agent_prefers_the_largest_capture():
    # 5 O O . . .      Black can capture A5-B5 at C5 (two stones) or E1 at
    # 4 X X . . .      D1 (one stone).
    # 2 . . . . X
    # 1 . . . . O
    s = go_from(["A4", "A5", "B4", "B5", "E2", "E1"])
    agent = HeuristicAgent(np.random.default_rng(0))
    for _ in range(5):
        m, pi = agent.act(s)
        assert m == parse_move("C5")
        assert pi[m] == 1 and pi.sum() == pytest.approx(1.0)


def test_human_agent_asks_again_after_bad_input(monkeypatch):
    answers = ["Z9", "C3", "pass"]

    def fake_input(prompt=""):
        return answers.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    m, pi = HumanAgent().act(go_from(["C3"]))       # C3 is occupied
    assert m == PASS and pi[PASS] == 1


def test_play_game_with_scripted_agents():
    record = play_game(ScriptedAgent(["C3", "pass"]), ScriptedAgent(["pass"]))
    assert record.moves == [parse_move("C3"), PASS, PASS]
    assert len(record.policies) == 3
    assert record.winner == BLACK


def test_play_game_from_a_given_state():
    record = play_game(ScriptedAgent(["pass"]), ScriptedAgent(["pass"]), go_from(["C3"]))
    assert record.moves == [PASS, PASS]                 # White moves first
    assert record.winner == BLACK


def test_play_game_on_tictactoe():
    record = play_game(FirstMoveAgent(), FirstMoveAgent(), TicTacToeState())
    assert record.moves == [0, 1, 2, 3, 4, 5, 6]
    assert record.winner == 1


def test_win_rate_ci():
    for (w, n), expected in {(5, 10): (0.5, 0.2366, 0.7634),
                             (0, 10): (0.0, 0.0, 0.2775),
                             (10, 10): (1.0, 0.7225, 1.0),
                             (70, 100): (0.7, 0.6042, 0.7811)}.items():
        assert win_rate_ci(w, n) == pytest.approx(expected, abs=1e-3)
