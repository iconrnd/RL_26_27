"""5x5 Go rules engine (Stage 1).

The rules are those of the specification, Section 4: area scoring, suicide
forbidden, positional superko; the game ends after two consecutive passes or
after MAX_MOVES moves.

Moves are integers. The point in row r and column c (0 <= r, c <= 4, row 0 at
the top) has the index 5*r + c; the pass has the index PASS = 25. Printed
coordinates: columns A-E from left to right, rows 5-1 from top to bottom, so
"C3" is the centre point (index 12).
"""

import numpy as np

BOARD = 5                          # points per side
PASS = BOARD * BOARD               # index of the pass
NUM_ACTIONS = BOARD * BOARD + 1    # 25 points + pass
KOMI = 0.5
MAX_MOVES = 60                     # passes included
BLACK, WHITE, EMPTY = 1, -1, 0
COLUMNS = "ABCDE"


def parse_move(text: str) -> int:
    """Provided. "C3" -> 12, "pass" -> PASS (case-insensitive).

    Raises ValueError for text that is neither a point of the board nor "pass".
    """
    t = text.strip().upper()
    if t == "PASS":
        return PASS
    if len(t) != 2 or t[0] not in COLUMNS or t[1] not in "12345":
        raise ValueError(f"not a move: {text!r}")
    return BOARD * (BOARD - int(t[1])) + COLUMNS.index(t[0])


def move_to_str(move: int) -> str:
    """Provided. 12 -> "C3", PASS -> "pass"."""
    if move == PASS:
        return "pass"
    r, c = divmod(move, BOARD)
    return f"{COLUMNS[c]}{BOARD - r}"


class GoState:
    """A position of 5x5 Go, with the game interface of the specification
    (Section 3.3). A state is immutable: play() returns a new state and leaves
    the old one unchanged.

    Public attributes:
        board        np.ndarray of shape (5, 5), dtype int8, holding BLACK,
                     WHITE, or EMPTY at every point (board[r, c])
        to_play      BLACK or WHITE, the player to move
        move_number  number of moves played so far, passes included
        num_actions  NUM_ACTIONS

    Further attributes (the number of consecutive passes, the earlier
    arrangements needed for superko, ...) belong to the implementation.
    """

    num_actions = NUM_ACTIONS

    def __init__(self):
        """The empty board, Black to move."""
        raise NotImplementedError

    def legal_moves(self) -> list[int]:
        """The legal moves in increasing order; PASS is always legal.

        A placement on an empty point is legal unless it is suicide (the group
        of the placed stone has no liberties once the captured opponent groups
        are removed) or recreates an arrangement of stones that occurred
        earlier in the game (positional superko).
        """
        raise NotImplementedError

    def playout_moves(self) -> list[int]:
        """The moves a random play-out may choose: the legal placements that
        do not fill an own eye, or [PASS] if there are none. An own eye is an
        empty point whose horizontal and vertical neighbours are all stones of
        the player to move."""
        raise NotImplementedError

    def play(self, move: int) -> "GoState":
        """The state after `move`, which must be legal. Opponent groups left
        without liberties are removed from the board."""
        raise NotImplementedError

    def is_terminal(self) -> bool:
        """True after two consecutive passes, or once MAX_MOVES moves have
        been played."""
        raise NotImplementedError

    def score(self) -> float:
        """area(Black) - area(White) - KOMI. A player's area is the number of
        the player's stones plus the number of empty points in empty regions
        (maximal connected sets of empty points) that border only that
        player's stones."""
        raise NotImplementedError

    def winner(self) -> int:
        """BLACK if score() > 0, otherwise WHITE. Called on terminal states."""
        raise NotImplementedError

    def __str__(self) -> str:
        """The board with coordinates, for example X for Black, O for White,
        and . for an empty point."""
        raise NotImplementedError
