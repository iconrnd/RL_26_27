"""Self-play games (Stage 3)."""

from .agents import MCTSAgent
from .arena import play_game
from .data import GameRecord
from .net import NetEvaluator


def self_play_game(net, cfg, rng) -> GameRecord:
    """One game from the empty board in which both sides are
    MCTSAgent(NetEvaluator(net), cfg.selfplay_simulations, "puct", cfg, rng,
    self_play=True). The record holds the policy vector of every move."""
    raise NotImplementedError
