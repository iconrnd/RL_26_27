"""Tic-tac-toe with the game interface of the specification (Section 3.3).

Provided and complete; not to be edited. It serves to test the search on a
game whose exact values are known: with perfect play every game is a draw.
Moves are 0..8, row by row from the top left. The first player (X) is +1.
"""

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8),
         (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6))


class TicTacToeState:
    num_actions = 9

    def __init__(self, board=(0,) * 9, to_play=1):
        self.board = board              # tuple of 9 entries: +1, -1, or 0
        self.to_play = to_play
        self.move_number = sum(1 for v in board if v != 0)
        self._winner = 0
        for a, b, c in LINES:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                self._winner = board[a]

    def legal_moves(self) -> list[int]:
        if self.is_terminal():
            return []
        return [i for i, v in enumerate(self.board) if v == 0]

    def playout_moves(self) -> list[int]:
        return self.legal_moves()

    def play(self, move: int) -> "TicTacToeState":
        board = list(self.board)
        board[move] = self.to_play
        return TicTacToeState(tuple(board), -self.to_play)

    def is_terminal(self) -> bool:
        return self._winner != 0 or all(v != 0 for v in self.board)

    def winner(self) -> int:
        """+1 or -1 for a win, 0 for a draw."""
        return self._winner

    def __str__(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        rows = [" ".join(symbols[v] for v in self.board[i:i + 3]) for i in (0, 3, 6)]
        return "\n".join(rows)
