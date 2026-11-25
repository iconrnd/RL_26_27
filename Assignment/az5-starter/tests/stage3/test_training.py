"""Stage 3 public tests: loss, training steps, the training loop and its
resumption. Provided; not to be edited."""

import csv
from dataclasses import replace

import numpy as np
import pytest
import torch

from az5.config import Config
from az5.data import ReplayBuffer
from az5.net import PolicyValueNet
from az5.train import LOG_COLUMNS, loss_fn, run, train_steps

SMALL = Config(channels=8, res_blocks=1)


def _batch(n=8, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 3, 5, 5)) < 0.3).astype(np.float32)
    P = rng.random((n, 26)).astype(np.float32)
    P /= P.sum(axis=1, keepdims=True)
    Z = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    return X, P, Z


def test_loss_fn():
    torch.manual_seed(0)
    net = PolicyValueNet(SMALL).eval()
    X, P, Z = (torch.from_numpy(a) for a in _batch())
    total, value, policy = loss_fn(net, X, P, Z)
    with torch.no_grad():
        logits, v = net(X)
        expected_value = ((Z - v) ** 2).mean().item()
        expected_policy = -(P * torch.log_softmax(logits, dim=1)).sum(dim=1).mean().item()
    assert value == pytest.approx(expected_value, rel=1e-5)
    assert policy == pytest.approx(expected_policy, rel=1e-5)
    assert total.item() == pytest.approx(value + policy, rel=1e-5)
    assert total.requires_grad


def test_train_steps_fit_a_small_buffer():
    torch.manual_seed(0)
    net = PolicyValueNet(SMALL)
    buffer = ReplayBuffer(100, augment=False)
    X, _, Z = _batch(16)
    P = np.eye(26, dtype=np.float32)[np.random.default_rng(1).integers(25, size=16)]
    buffer.add(list(zip(X, P, Z.tolist())))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    rng = np.random.default_rng(0)
    first = train_steps(net, optimizer, buffer, replace(SMALL, train_steps=1, batch_size=16), rng)
    assert set(first) == {"value_loss", "policy_loss"}
    train_steps(net, optimizer, buffer, replace(SMALL, train_steps=300, batch_size=16), rng)
    last = train_steps(net, optimizer, buffer, replace(SMALL, train_steps=1, batch_size=16), rng)
    assert last["value_loss"] < 0.5 * first["value_loss"]
    assert last["policy_loss"] < 0.5 * first["policy_loss"]


TINY = Config(channels=8, res_blocks=1, iterations=2, games_per_iter=2,
              selfplay_simulations=4, eval_simulations=4, gate_pairs=1,
              train_steps=2, batch_size=8, buffer_capacity=1000,
              anchor_every=2, anchor_pairs=1, baseline_simulations=4)


def _rows(workdir):
    with open(workdir / "log.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_run_writes_log_and_checkpoints(tmp_path):
    run(TINY, tmp_path)
    rows = _rows(tmp_path)
    assert tuple(rows[0].keys()) == LOG_COLUMNS
    assert [int(r["iteration"]) for r in rows] == [1, 2]
    assert rows[0]["anchor_win_rate"] == "" and rows[1]["anchor_win_rate"] != ""
    assert all(r["accepted"] in ("0", "1") for r in rows)
    assert int(rows[1]["buffer_size"]) > 0
    for name in ("checkpoint.pt", "buffer.npz", "best_001.pt", "best_002.pt"):
        assert (tmp_path / name).exists(), name
    net = PolicyValueNet(TINY)
    net.load_state_dict(torch.load(tmp_path / "best_002.pt"))


def test_run_resumes(tmp_path):
    run(TINY, tmp_path)
    before = _rows(tmp_path)
    # Resuming with a changed configuration: a restarted run would rewrite the
    # rows of iterations 1 and 2 (with smaller buffers).
    run(replace(TINY, iterations=3, games_per_iter=1), tmp_path)
    after = _rows(tmp_path)
    assert [int(r["iteration"]) for r in after] == [1, 2, 3]
    assert after[:2] == before
    assert (tmp_path / "best_003.pt").exists()
    stamp = (tmp_path / "best_003.pt").stat().st_mtime_ns
    run(replace(TINY, iterations=3), tmp_path)        # nothing left to do
    assert len(_rows(tmp_path)) == 3
    assert (tmp_path / "best_003.pt").stat().st_mtime_ns == stamp


def test_run_without_gate(tmp_path):
    run(replace(TINY, iterations=1, use_gate=False), tmp_path)
    rows = _rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["accepted"] == "1" and row["gate_win_rate"] == ""
