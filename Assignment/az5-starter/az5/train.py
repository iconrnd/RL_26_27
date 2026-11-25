"""Loss, training steps, and the training loop (Stage 3)."""

import numpy as np
import torch

# Columns of log.csv, in this order. accepted is 1 or 0; gate_win_rate is
# empty without a gate (Config.use_gate = False); anchor_win_rate is empty in
# the iterations without an anchor match.
LOG_COLUMNS = ("iteration", "buffer_size", "value_loss", "policy_loss",
               "gate_win_rate", "accepted", "anchor_win_rate", "seconds")


def loss_fn(net, X, P, Z) -> tuple[torch.Tensor, float, float]:
    """The loss of the specification (Section 7.3) for a minibatch of tensors
    X (B, 3, 5, 5), P (B, 26), Z (B,):

        value part   mean over the batch of (z - v)^2
        policy part  mean over the batch of -sum_a pi(a) log p(a | s),
                     with log p the log-soft-max over all 26 logits

    Returns (value part + policy part as a tensor, value part, policy part);
    the L2 term is applied through the optimizer's weight_decay.
    """
    raise NotImplementedError


def train_steps(net, optimizer, buffer, cfg, rng) -> dict:
    """cfg.train_steps optimizer steps on minibatches of cfg.batch_size
    samples from `buffer`, with the network in training mode. Returns
    {"value_loss": ..., "policy_loss": ...}, the means over the steps."""
    raise NotImplementedError


def run(cfg, workdir) -> None:
    """The training loop of the specification (Section 7.3) for
    cfg.iterations iterations, numbered from 1.

    Writes to the directory `workdir` (created if missing):
        checkpoint.pt   the latest state of the run: best and training
                        networks, optimizer, and iteration (overwritten
                        after every iteration)
        buffer.npz      the replay buffer (overwritten after every iteration)
        best_XXX.pt     state_dict of the best network after iteration XXX
                        (three digits, e.g. best_007.pt)
        log.csv         a header with LOG_COLUMNS and one row per iteration

    If workdir/checkpoint.pt exists, the run resumes after the iteration
    stored in it; completed iterations are not repeated. A later call with a
    larger cfg.iterations continues the run. Iteration i draws its random
    numbers from np.random.default_rng([cfg.seed, i]), so that a resumed run
    continues as an uninterrupted run would.
    """
    raise NotImplementedError
