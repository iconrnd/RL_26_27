"""Stage 1 public tests: the rules engine. Provided; not to be edited.

Board diagrams in the comments show rows 5..1 from top to bottom and columns
A..E from left to right; X is Black, O is White.
"""

import json
import pathlib

import pytest

from az5.go import BLACK, EMPTY, PASS, WHITE, GoState, parse_move
from helpers import go_from


def stones(state, colour):
    return int((state.board == colour).sum())


def test_initial_state():
    s = GoState()
    assert s.board.shape == (5, 5)
    assert (s.board == EMPTY).all()
    assert s.to_play == BLACK
    assert s.move_number == 0
    assert s.num_actions == 26
    assert s.legal_moves() == list(range(26))
    assert not s.is_terminal()


def test_play_returns_a_new_state():
    s = GoState()
    t = s.play(parse_move("C3"))
    assert s.board[2, 2] == EMPTY and s.to_play == BLACK and s.move_number == 0
    assert t.board[2, 2] == BLACK and t.to_play == WHITE and t.move_number == 1
    assert parse_move("C3") not in t.legal_moves()
    assert t.play(PASS).to_play == BLACK


def test_single_capture():
    # White C3 loses its last liberty when Black plays C2.
    s = go_from(["B3", "C3", "C4", "A1", "D3", "A2", "C2"])
    assert s.board[2, 2] == EMPTY
    assert stones(s, WHITE) == 2 and stones(s, BLACK) == 4


def test_multi_stone_capture():
    # 5 O O X . .      White A5-B5 is captured by Black C5.
    # 4 X X . . .
    s = go_from(["A4", "A5", "B4", "B5", "C5"])
    assert s.board[0, 0] == EMPTY and s.board[0, 1] == EMPTY
    assert stones(s, WHITE) == 0 and stones(s, BLACK) == 3


def test_suicide_is_illegal():
    # 5 . O . . .      Black A5 would have no liberties and captures nothing.
    # 4 O . . . .
    s = go_from(["E1", "B5", "E2", "A4"])
    assert s.to_play == BLACK
    assert parse_move("A5") not in s.legal_moves()
    assert parse_move("A5") not in s.playout_moves()


def test_capture_makes_a_placement_legal():
    # 5 . O X . .      Black A5 has no liberties before the captures, but it
    # 4 O X . . .      captures White B5 and A4, so it is legal.
    # 3 X . . . .
    s = go_from(["C5", "B5", "B4", "A4", "A3", "pass"])
    assert parse_move("A5") in s.legal_moves()
    t = s.play(parse_move("A5"))
    assert t.board[0, 1] == EMPTY and t.board[1, 0] == EMPTY
    assert stones(t, WHITE) == 0


# 5 . X O . .      Black C4 captures White B4 (ko). An immediate recapture
# 4 X O X O .      at B4 would recreate the earlier arrangement.
# 3 . X O . .
KO = ["B5", "C5", "A4", "B4", "B3", "D4", "pass", "C3", "C4"]


def test_ko_recapture_is_forbidden_by_superko():
    s = go_from(KO)
    assert s.board[1, 1] == EMPTY and s.board[1, 2] == BLACK
    assert s.to_play == WHITE
    assert parse_move("B4") not in s.legal_moves()


def test_ko_recapture_is_legal_after_an_exchange():
    s = go_from(KO + ["E1", "A1"])
    assert parse_move("B4") in s.legal_moves()
    t = s.play(parse_move("B4"))
    assert t.board[1, 2] == EMPTY


def test_pass_is_always_legal():
    # Black fills every point except A5 and E1; White can only pass.
    moves = []
    for p in range(1, 24):
        moves.append(p)
        if p < 23:
            moves.append(PASS)
    s = GoState()
    for m in moves:
        s = s.play(m)
    assert s.to_play == WHITE
    assert s.legal_moves() == [PASS]
    t = s.play(PASS)
    assert t.legal_moves() == [parse_move("A5"), parse_move("E1"), PASS]
    assert t.playout_moves() == [PASS]


def test_playout_moves_exclude_own_eyes():
    # 5 . X . . .      A5 is an own eye of Black: legal, but not a playout move.
    # 4 X . . . .
    s = go_from(["B5", "E1", "A4", "E2"])
    assert s.to_play == BLACK
    assert parse_move("A5") in s.legal_moves()
    assert parse_move("A5") not in s.playout_moves()
    assert PASS not in s.playout_moves()


def test_two_consecutive_passes_end_the_game():
    assert not go_from(["pass"]).is_terminal()
    assert go_from(["pass", "pass"]).is_terminal()
    assert not go_from(["pass", "C3", "pass"]).is_terminal()
    assert go_from(["pass", "C3", "pass", "pass"]).is_terminal()


def test_score_of_the_empty_board():
    s = go_from(["pass", "pass"])
    assert s.score() == pytest.approx(-0.5)
    assert s.winner() == WHITE


def test_score_of_a_single_stone():
    s = go_from(["C3", "pass", "pass"])
    assert s.score() == pytest.approx(24.5)
    assert s.winner() == BLACK


def test_score_with_neutral_points():
    # 5 . X . O .      Column A is Black's, column E is White's; column C
    # 4 . X . O .      borders both colours and is neutral.
    s = go_from(["B5", "D5", "B4", "D4", "B3", "D3", "B2", "D2", "B1", "D1",
                 "pass", "pass"])
    assert s.is_terminal()
    assert s.score() == pytest.approx(-0.5)
    assert s.winner() == WHITE


def test_str():
    text = str(go_from(["C3"]))
    assert isinstance(text, str) and len(text.splitlines()) >= 5


RECORDED = json.loads((pathlib.Path(__file__).parent / "recorded_games.json").read_text())


@pytest.mark.parametrize("game", RECORDED, ids=[f"{g['kind']}-{i}" for i, g in enumerate(RECORDED)])
def test_recorded_game(game):
    """Replays a recorded game (with captures, ko fights, and eye fills),
    comparing the number of legal moves in every position, the end of the
    game, the final board, and the score."""
    s = GoState()
    for t, m in enumerate(game["moves"]):
        assert not s.is_terminal(), f"position {t}: the game ended too early"
        n = len(s.legal_moves())
        assert n == game["legal_counts"][t], (
            f"position {t} (after the moves {game['moves'][:t]}): "
            f"{n} legal moves, expected {game['legal_counts'][t]}")
        s = s.play(m)
    assert s.is_terminal()
    assert s.board.tolist() == game["board"]
    assert s.score() == pytest.approx(game["score"])
