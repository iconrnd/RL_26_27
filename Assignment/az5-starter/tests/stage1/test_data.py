"""Stage 1 public tests: encoding, symmetries, records, replay buffer.
Provided; not to be edited."""

import numpy as np
import pytest

from az5.data import GameRecord, ReplayBuffer, encode, make_samples, symmetries
from az5.go import BLACK, PASS, GoState, parse_move
from helpers import go_from, one_hot


def test_encode_empty_board():
    x = encode(GoState())
    assert x.shape == (3, 5, 5) and x.dtype == np.float32
    assert x[0].sum() == 0 and x[1].sum() == 0
    assert (x[2] == 1).all()


def test_encode_is_relative_to_the_player_to_move():
    x = encode(go_from(["C3"]))                  # White to move
    assert x[0].sum() == 0
    assert x[1][2, 2] == 1 and x[1].sum() == 1
    assert (x[2] == 0).all()
    y = encode(go_from(["C3", "B2"]))            # Black to move
    assert y[0][2, 2] == 1 and y[0].sum() == 1
    assert y[1][3, 1] == 1 and y[1].sum() == 1
    assert (y[2] == 1).all()


def _all_transforms(a):
    out = []
    for b in (a, a[:, :, ::-1]):
        for k in range(4):
            out.append(np.rot90(b, k, axes=(1, 2)))
    return out


def test_symmetries_form_the_symmetry_group_of_the_square():
    rng = np.random.default_rng(0)
    planes = rng.random((3, 5, 5)).astype(np.float32)
    pi = rng.random(26).astype(np.float32)
    pi /= pi.sum()
    result = symmetries(planes, pi)
    assert len(result) == 8
    assert np.array_equal(result[0][0], planes) and np.allclose(result[0][1], pi)
    expected = _all_transforms(planes)          # eight different arrays
    for p, _ in result:
        assert sum(np.array_equal(p, e) for e in expected) == 1
    for e in expected:
        assert any(np.array_equal(p, e) for p, _ in result)


def test_symmetries_move_the_policy_with_the_stones():
    planes = np.zeros((3, 5, 5), dtype=np.float32)
    planes[0, 0, 1] = 1                          # a stone at B5 (index 1)
    planes[1, 3, 0] = 1                          # an opponent stone at A2
    pi = np.zeros(26, dtype=np.float32)
    pi[1], pi[15], pi[PASS] = 0.5, 0.2, 0.3      # B5, A2, pass
    for p, q in symmetries(planes, pi):
        assert p[0].sum() == 1 and p[1].sum() == 1
        r, c = np.argwhere(p[0] == 1)[0]
        r2, c2 = np.argwhere(p[1] == 1)[0]
        assert q[5 * r + c] == pytest.approx(0.5)
        assert q[5 * r2 + c2] == pytest.approx(0.2)
        assert q[PASS] == pytest.approx(0.3)
        assert q.sum() == pytest.approx(1.0)


def _record():
    moves = [parse_move("C3"), PASS, PASS]
    return GameRecord(moves, [one_hot(26, m) for m in moves], BLACK)


def test_game_record_to_sgf():
    moves = [parse_move("C3"), parse_move("B2"), PASS]
    sgf = "".join(GameRecord(moves, [one_hot(26, m) for m in moves], BLACK).to_sgf().split())
    assert sgf.startswith("(;") and sgf.endswith(")")
    assert "SZ[5]" in sgf
    assert ";B[cc];W[bd];B[]" in sgf


def test_make_samples():
    record = _record()
    samples = make_samples(record)
    assert len(samples) == 3
    assert [z for _, _, z in samples] == [1, -1, 1]
    assert np.array_equal(samples[0][0], encode(GoState()))
    assert np.array_equal(samples[1][0], encode(go_from(["C3"])))
    for (_, pi, _), p in zip(samples, record.policies):
        assert np.allclose(pi, p)


def _sample(z, point=1):
    planes = np.zeros((3, 5, 5), dtype=np.float32)
    planes[0].flat[point] = 1
    return planes, one_hot(26, point), float(z)


def test_replay_buffer_shapes():
    buf = ReplayBuffer(100)
    buf.add([_sample(1), _sample(-1)])
    assert len(buf) == 2
    X, P, Z = buf.sample(16, np.random.default_rng(0))
    assert X.shape == (16, 3, 5, 5) and P.shape == (16, 26) and Z.shape == (16,)
    assert X.dtype == P.dtype == Z.dtype == np.float32


def test_replay_buffer_is_first_in_first_out():
    buf = ReplayBuffer(5, augment=False)
    buf.add([_sample(z) for z in range(8)])
    assert len(buf) == 5
    _, _, Z = buf.sample(500, np.random.default_rng(0))
    assert set(Z.tolist()) == {3.0, 4.0, 5.0, 6.0, 7.0}


def test_replay_buffer_augmentation():
    rng = np.random.default_rng(0)
    plain = ReplayBuffer(10, augment=False)
    plain.add([_sample(1)])
    X, P, _ = plain.sample(50, rng)
    assert (X == X[0]).all() and X[0, 0, 0, 1] == 1

    aug = ReplayBuffer(10, augment=True)
    aug.add([_sample(1)])
    X, P, _ = aug.sample(400, rng)
    distinct = []
    for x in X:
        if not any(np.array_equal(x, d) for d in distinct):
            distinct.append(x)
    assert len(distinct) == 8                    # B5 has 8 distinct images
    for x, p in zip(X, P):
        assert x[0].sum() == 1
        r, c = np.argwhere(x[0] == 1)[0]
        assert p[5 * r + c] == 1


def test_replay_buffer_save_and_load(tmp_path):
    buf = ReplayBuffer(100, augment=False)
    buf.add([_sample(z, point=z) for z in range(10)])
    path = tmp_path / "buffer.npz"
    buf.save(path)
    other = ReplayBuffer(100, augment=False)
    other.load(path)
    assert len(other) == 10
    a = buf.sample(32, np.random.default_rng(1))
    b = other.sample(32, np.random.default_rng(1))
    for u, v in zip(a, b):
        assert np.array_equal(u, v)
