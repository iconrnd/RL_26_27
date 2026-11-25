"""Stage 2 public tests: the search with UCT, mostly on tic-tac-toe, where
exact values are known. Provided; not to be edited.

Tic-tac-toe squares:  0 1 2
                      3 4 5
                      6 7 8
"""

import math

import numpy as np
import pytest

from az5.agents import MCTSAgent
from az5.arena import play_game
from az5.config import Config
from az5.mcts import Node, RolloutEvaluator, backup, expand, search, select_leaf, uct_scores
from az5.tictactoe import TicTacToeState
from helpers import ttt_from

CFG = Config()


def _node(N, W, legal, P=None):
    n = len(N)
    return Node(state=None, N=np.array(N, float), W=np.array(W, float),
                P=np.full(n, 1 / n) if P is None else np.array(P, float),
                legal=np.array(legal, bool), expanded=True)


def test_uct_scores():
    node = _node(N=[1, 3, 0, 0], W=[1, -1, 0, 0], legal=[True, True, True, False])
    s = uct_scores(node, 1.0)
    assert s[0] == pytest.approx(1 + math.sqrt(math.log(4) / 1))
    assert s[1] == pytest.approx(-1 / 3 + math.sqrt(math.log(4) / 3))
    assert s[2] == math.inf
    assert s[3] == -math.inf


def test_expand():
    node = Node(ttt_from([4]))
    priors = np.full(9, 1 / 9)
    expand(node, priors)
    priors[0] = 5.0                              # node.P must be a copy
    assert node.expanded
    assert node.legal.tolist() == [True] * 4 + [False] + [True] * 4
    assert np.allclose(node.P, 1 / 9)
    assert (node.N == 0).all() and (node.W == 0).all()
    assert len(node.N) == len(node.W) == 9


def test_backup_alternates_signs():
    nodes = [_node([0, 0], [0, 0], [True, True]) for _ in range(3)]
    path = [(nodes[0], 1), (nodes[1], 0), (nodes[2], 1)]
    backup(path, 1.0)                            # the reached node's player wins
    assert nodes[2].W[1] == -1 and nodes[1].W[0] == 1 and nodes[0].W[1] == -1
    assert nodes[2].N[1] == nodes[1].N[0] == nodes[0].N[1] == 1
    assert nodes[0].N[0] == 0


def test_select_leaf_creates_the_child():
    root = Node(TicTacToeState())
    expand(root, np.full(9, 1 / 9))
    path, leaf = select_leaf(root, "uct", CFG, np.random.default_rng(0))
    assert len(path) == 1 and path[0][0] is root
    move = path[0][1]
    assert root.children[move] is leaf
    assert leaf.state.board == root.state.play(move).board
    assert not leaf.expanded


def test_select_leaf_follows_the_highest_score():
    root = Node(TicTacToeState())
    expand(root, np.full(9, 1 / 9))
    root.N[:] = 1
    root.W[:] = -1
    root.W[4] = 1                                # equal bonus, best mean value
    path, _ = select_leaf(root, "uct", CFG, np.random.default_rng(0))
    assert path[0][1] == 4


def test_rollout_evaluator_priors_and_values():
    ev = RolloutEvaluator(np.random.default_rng(0))
    priors, value = ev.evaluate(ttt_from([4]))
    assert priors[4] == 0 and np.allclose(np.delete(priors, 4), 1 / 8)
    assert value in (-1, 0, 1)


def test_rollout_evaluator_perspective():
    ev = RolloutEvaluator(np.random.default_rng(0))
    # X to move can win at 2: random play-outs favour X (mean about +0.10).
    assert np.mean([ev.evaluate(ttt_from([0, 3, 1, 4]))[1] for _ in range(4000)]) > 0.03
    # O to move must block at 2: random play-outs favour X (mean about -0.19).
    assert np.mean([ev.evaluate(ttt_from([0, 4, 1]))[1] for _ in range(4000)]) < -0.08


def test_search_visit_counts():
    rng = np.random.default_rng(0)
    root = search(ttt_from([4]), RolloutEvaluator(rng), 50, "uct", CFG, rng)
    assert root.expanded
    assert root.N.sum() == 50
    assert root.N[4] == 0


def test_search_takes_an_immediate_win():
    rng = np.random.default_rng(0)
    root = search(ttt_from([0, 3, 1, 4]), RolloutEvaluator(rng), 200, "uct", CFG, rng)
    assert np.argmax(root.N) == 2


def test_search_blocks_an_immediate_loss():
    rng = np.random.default_rng(0)
    root = search(ttt_from([0, 4, 1]), RolloutEvaluator(rng), 300, "uct", CFG, rng)
    assert np.argmax(root.N) == 2


def test_mcts_agent():
    rng = np.random.default_rng(0)
    agent = MCTSAgent(RolloutEvaluator(rng), 200, "uct", CFG, rng)
    m, pi = agent.act(ttt_from([0, 3, 1, 4]))
    assert m == 2
    assert pi.sum() == pytest.approx(1.0) and np.argmax(pi) == 2
    assert pi[0] == pi[1] == pi[3] == pi[4] == 0


def test_mcts_self_play_on_tictactoe_draws():
    results = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        agent = MCTSAgent(RolloutEvaluator(rng), 2000, "uct", CFG, rng)
        results.append(play_game(agent, agent, TicTacToeState()).winner)
    assert results.count(0) >= 2, f"results {results}: perfect play draws"
