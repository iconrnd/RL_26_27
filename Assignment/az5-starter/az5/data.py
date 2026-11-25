"""State encoding, symmetries, game records, and the replay buffer (Stage 1)."""

from dataclasses import dataclass

import numpy as np

from .go import BLACK, GoState


def encode(state) -> np.ndarray:
    """The network input for a GoState: an array of shape (3, 5, 5), float32,
    from the perspective of the player to move.

    Plane 0: stones of the player to move (1 where present, else 0).
    Plane 1: stones of the opponent.
    Plane 2: all ones if Black is to move, all zeros otherwise.
    """
    raise NotImplementedError


def symmetries(planes, pi) -> list[tuple[np.ndarray, np.ndarray]]:
    """The eight symmetries of the square (four rotations, each with and
    without a reflection), applied to `planes` (shape (C, 5, 5)) and,
    consistently, to the 25 point entries of `pi` (length 26, point 5*r + c
    at board position (r, c)). The pass entry pi[25] is unchanged.

    Returns 8 pairs (planes', pi'); the identity comes first.
    """
    raise NotImplementedError


@dataclass
class GameRecord:
    """A finished game.

    moves[t]     the move played in the t-th position of the game
    policies[t]  the probability vector (length num_actions) the move was
                 drawn from
    winner       +1 (Black, first player), -1, or 0 (draw, tic-tac-toe only)
    """

    moves: list
    policies: list
    winner: int

    def to_sgf(self) -> str:
        """The game in the Smart Game Format (FF[4]), for a Go game started
        from the empty board: a header such as
        (;GM[1]FF[4]SZ[5]KM[0.5]RU[Tromp-Taylor]RE[B+] followed by one node per
        move and a closing parenthesis. A node is ;B[cc] for Black at C3
        (first letter: column a-e from the left, second letter: row a-e from
        the top) and ;W[] for a pass by White."""
        raise NotImplementedError


def make_samples(record) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """One training sample (encode(s_t), policies[t], z_t) for every position
    s_t of a Go game started from the empty board, in the order of the game;
    z_t = +1 if the player to move in s_t won the game, z_t = -1 otherwise."""
    raise NotImplementedError


class ReplayBuffer:
    """The most recent `capacity` samples, first in, first out.

    With augment=True, sample() transforms every drawn sample by one of the
    eight symmetries, chosen uniformly at random (Config.augment).
    """

    def __init__(self, capacity: int, augment: bool = True):
        raise NotImplementedError

    def __len__(self) -> int:
        """The number of stored samples."""
        raise NotImplementedError

    def add(self, samples) -> None:
        """Appends samples (planes, pi, z), dropping the oldest beyond
        capacity."""
        raise NotImplementedError

    def sample(self, batch_size, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A minibatch of samples drawn uniformly with replacement:
        X (batch_size, 3, 5, 5), P (batch_size, 26), Z (batch_size,), all
        float32."""
        raise NotImplementedError

    def save(self, path) -> None:
        """Writes the samples to `path`, a file name ending in .npz."""
        raise NotImplementedError

    def load(self, path) -> None:
        """Replaces the content by the samples saved at `path`."""
        raise NotImplementedError
