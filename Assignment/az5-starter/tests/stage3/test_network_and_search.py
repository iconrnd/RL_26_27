"""Stage 3 public tests: network, evaluator, PUCT, root noise, self-play.
Provided; not to be edited."""

import math

import numpy as np
import pytest
import torch

from az5.agents import MCTSAgent, PolicyAgent
from az5.config import Config
from az5.go import PASS, GoState, parse_move
from az5.mcts import Node, add_dirichlet_noise, puct_scores, search
from az5.net import NetEvaluator, PolicyValueNet
from az5.selfplay import self_play_game
from helpers import go_from

SMALL = Config(channels=8, res_blocks=1)
KO = ["B5", "C5", "A4", "B4", "B3", "D4", "pass", "C3", "C4"]   # White may not retake at B4


def _net(cfg=SMALL):
    torch.manual_seed(0)
    return PolicyValueNet(cfg)


def _node(N, W, P, legal):
    return Node(state=None, N=np.array(N, float), W=np.array(W, float),
                P=np.array(P, float), legal=np.array(legal, bool), expanded=True)


def test_network_output_shapes():
    logits, v = _net()(torch.zeros(4, 3, 5, 5))
    assert logits.shape == (4, 26) and v.shape == (4,)
    assert (v.abs() <= 1).all()


def test_default_network_size():
    n = sum(p.numel() for p in PolicyValueNet(Config()).parameters())
    assert 5e4 < n < 5e5


def test_net_evaluator_masks_illegal_moves():
    s = go_from(KO)
    priors, value = NetEvaluator(_net()).evaluate(s)
    assert isinstance(priors, np.ndarray) and priors.shape == (26,)
    assert priors.sum() == pytest.approx(1.0)
    legal = set(s.legal_moves())
    assert all((priors[m] > 0) == (m in legal) for m in range(26))
    assert priors[parse_move("B4")] == 0          # forbidden by superko
    assert isinstance(value, float) and -1 <= value <= 1


def test_policy_agent_plays_the_highest_prior():
    net = _net()
    s = go_from(KO)
    m, pi = PolicyAgent(net).act(s)
    priors, _ = NetEvaluator(net).evaluate(s)
    assert m == int(np.argmax(priors)) and m in s.legal_moves()
    assert np.allclose(pi, priors, atol=1e-6)


def test_puct_scores():
    node = _node(N=[2, 0, 0], W=[1, 0, 0], P=[0.5, 0.3, 0.2], legal=[True, True, False])
    s = puct_scores(node, 1.5)
    assert s[0] == pytest.approx(0.5 + 1.5 * 0.5 * math.sqrt(2) / 3)
    assert s[1] == pytest.approx(1.5 * 0.3 * math.sqrt(2) / 1)
    assert s[2] == -math.inf


def test_dirichlet_noise():
    rng = np.random.default_rng(0)
    node = _node(N=[0] * 4, W=[0] * 4, P=[0.5, 0.3, 0.2, 0.0], legal=[True, True, True, False])
    add_dirichlet_noise(node, 0.5, 0.0, rng)
    assert np.allclose(node.P, [0.5, 0.3, 0.2, 0.0])
    add_dirichlet_noise(node, 0.5, 0.25, rng)
    assert node.P[3] == 0
    assert node.P.sum() == pytest.approx(1.0)
    assert not np.allclose(node.P, [0.5, 0.3, 0.2, 0.0])
    assert (node.P[:3] >= 0.75 * np.array([0.5, 0.3, 0.2]) - 1e-9).all()


def test_search_with_the_network():
    net = _net()
    rng = np.random.default_rng(0)
    root = search(GoState(), NetEvaluator(net), 20, "puct", SMALL, rng)
    assert root.N.sum() == 20
    priors, _ = NetEvaluator(net).evaluate(GoState())
    assert np.allclose(root.P, priors)
    noisy = search(GoState(), NetEvaluator(net), 20, "puct", SMALL, rng, root_noise=True)
    assert noisy.P.sum() == pytest.approx(1.0)
    assert not np.allclose(noisy.P, priors)


def test_self_play_agent_samples_early_moves():
    cfg = Config(channels=8, res_blocks=1, temp_moves=100)
    rng = np.random.default_rng(0)
    agent = MCTSAgent(NetEvaluator(_net(cfg)), 16, "puct", cfg, rng, self_play=True)
    moves = set()
    for _ in range(30):
        m, pi = agent.act(GoState())
        assert pi.sum() == pytest.approx(1.0) and pi[m] > 0
        moves.add(m)
    assert len(moves) >= 2


def test_self_play_game():
    cfg = Config(channels=8, res_blocks=1, selfplay_simulations=4)
    record = self_play_game(_net(cfg), cfg, np.random.default_rng(0))
    assert len(record.moves) == len(record.policies) > 0
    assert record.winner in (1, -1)
    s = GoState()
    for m, pi in zip(record.moves, record.policies):
        assert m in s.legal_moves()
        assert len(pi) == 26 and pi.sum() == pytest.approx(1.0)
        s = s.play(m)
    assert s.is_terminal() and s.winner() == record.winner
