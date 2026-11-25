"""Monte Carlo tree search: UCT (Stage 2) and PUCT (Stage 3).

All edge statistics of a node are stated from the perspective of the player
to move in its state. Q(s, a) = W(s, a) / N(s, a), and Q(s, a) = 0 where
N(s, a) = 0; N(s) is the sum of N(s, a) over all moves.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    """A node of the search tree. The arrays are set by expand()."""

    state: object
    N: np.ndarray = None          # visit counts N(s, a), length num_actions
    W: np.ndarray = None          # total values W(s, a)
    P: np.ndarray = None          # priors P(s, a)
    legal: np.ndarray = None      # boolean mask of the legal moves
    children: dict = field(default_factory=dict)   # move -> Node, created on first visit
    expanded: bool = False


class RolloutEvaluator:
    """Stage 2. evaluate(state) -> (priors, value) for a non-terminal state:
    uniform priors over the legal moves (zero elsewhere), and the result of one
    random play-out from `state` (moves drawn uniformly from playout_moves()
    for both players until the game ends), from the perspective of the player
    to move in `state`: +1 win, -1 loss, 0 draw. The play-out draws the moves
    itself: agents.py imports this module, so importing RandomAgent here would
    be a circular import."""

    def __init__(self, rng):
        raise NotImplementedError

    def evaluate(self, state) -> tuple[np.ndarray, float]:
        raise NotImplementedError


def uct_scores(node, c) -> np.ndarray:
    """Stage 2. For every move: Q(s, a) + c * sqrt(ln N(s) / N(s, a)) for
    legal moves with N(s, a) > 0, +inf for legal moves with N(s, a) = 0, and
    -inf for illegal moves."""
    raise NotImplementedError


def puct_scores(node, c) -> np.ndarray:
    """Stage 3. For every move: Q(s, a) + c * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
    for legal moves, with Q(s, a) = 0 where N(s, a) = 0, and -inf for illegal
    moves."""
    raise NotImplementedError


def select_leaf(root, rule, cfg, rng) -> tuple[list, Node]:
    """Stage 2. From the root, follows the edge with the highest score
    (uct_scores with cfg.c_uct for rule "uct", puct_scores with cfg.c_puct for
    rule "puct"; ties broken uniformly at random) until it reaches a node that
    is not expanded or whose state is terminal. A child node is created when
    its edge is followed for the first time.

    Returns the path, a list of (node, move) pairs from the root, and the
    reached node.
    """
    raise NotImplementedError


def expand(node, priors) -> None:
    """Stage 2. Sets node.legal from node.state.legal_moves(), node.P to a
    copy of `priors`, node.N and node.W to zeros, and node.expanded to True."""
    raise NotImplementedError


def backup(path, value) -> None:
    """Stage 2. `value` is the value of the reached node from the perspective
    of the player to move in it. Every (node, move) on the path receives
    N += 1 and W += the value from the perspective of the player to move in
    `node`: -value for the last pair of the path, +value for the pair before
    it, and so on, alternating up to the root."""
    raise NotImplementedError


def add_dirichlet_noise(node, alpha, eps, rng) -> None:
    """Stage 3. On the legal moves of an expanded node:
    P(s, a) <- (1 - eps) * P(s, a) + eps * eta_a with
    eta ~ Dir(alpha, ..., alpha) over the legal moves."""
    raise NotImplementedError


def search(state, evaluator, n_simulations, rule, cfg, rng, root_noise=False) -> Node:
    """Stage 2. Builds a tree from the non-terminal `state` and returns its root.

    The root is expanded with the evaluator's priors first (Stage 3: then, if
    root_noise, add_dirichlet_noise(root, cfg.dirichlet_alpha,
    cfg.dirichlet_eps, rng) is applied once). Each of the n_simulations
    simulations then runs select_leaf; evaluates the reached node (a terminal
    state by its game result from the perspective of its player to move,
    otherwise by evaluator.evaluate, after which the node is expanded with the
    returned priors); and calls backup. Hence root.N.sum() == n_simulations.
    """
    raise NotImplementedError
