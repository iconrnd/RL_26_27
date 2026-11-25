"""Policy-value network and network evaluator (Stage 3)."""

import numpy as np
import torch
from torch import nn

from .data import encode


class PolicyValueNet(nn.Module):
    """The AlphaGo Zero architecture at reduced size (specification,
    Section 7.1): a 3x3 convolution stem with cfg.channels filters,
    cfg.res_blocks residual blocks, a policy head, and a value head.

    forward(x): x of shape (B, 3, 5, 5), float32 ->
    (logits of shape (B, 26), values of shape (B,) in [-1, 1]).
    """

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class NetEvaluator:
    """evaluate(state) -> (priors, value) for a GoState: encodes the state,
    runs the network in evaluation mode without gradients (on the device of
    its parameters), sets the logits of illegal moves to -inf, and returns the
    soft-max of the logits (a NumPy array of length 26) and the value (a float,
    from the perspective of the player to move)."""

    def __init__(self, net):
        raise NotImplementedError

    def evaluate(self, state) -> tuple[np.ndarray, float]:
        raise NotImplementedError
