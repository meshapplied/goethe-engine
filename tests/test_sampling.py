import torch

from goethe_engine.sampling import SamplingParams as P
from goethe_engine.sampling import sample, uniform


def lg(r=6, v=40, seed=0):
    return torch.randn(r, v, generator=torch.Generator().manual_seed(seed))


def test_greedy_is_argmax_and_ignores_seed():
    x = lg()
    assert sample(x, [P()] * 6, [1] * 6, [0] * 6, [None] * 6) == x.argmax(1).tolist()


def test_seeded_determinism_and_token_index_changes_stream():
    x = lg()
    p = [P(temperature=1.0)] * 6
    a = sample(x, p, [5] * 6, list(range(6)), [None] * 6)
    assert a == sample(x, p, [5] * 6, list(range(6)), [None] * 6)
    many = [sample(x[:1], p[:1], [5], [i], [None])[0] for i in range(60)]
    assert len(set(many)) > 3 and uniform(5, 0) != uniform(5, 1) and uniform(5, 0) != uniform(6, 0)


def test_batch_invariance_rows_equal_singletons_in_any_order():
    x = lg(8)
    ps = [P(temperature=0.7 + i / 10, top_k=i % 4 * 5, top_p=1 - 0.1 * (i % 3)) if i % 3 else P() for i in range(8)]
    seeds, idx = list(range(100, 108)), [i * 3 for i in range(8)]
    batch = sample(x, ps, seeds, idx, [None] * 8)
    solo = [sample(x[i:i + 1], ps[i:i + 1], seeds[i:i + 1], idx[i:i + 1], [None])[0] for i in range(8)]
    assert batch == solo
    perm = [5, 2, 7, 0, 1, 6, 3, 4]
    shuf = sample(x[perm], [ps[i] for i in perm], [seeds[i] for i in perm], [idx[i] for i in perm], [None] * 8)
    assert shuf == [batch[i] for i in perm]


def test_top_k_top_p_restrict_support():
    x = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    seen_k = {sample(x, [P(temperature=5.0, top_k=2)], [s], [0], [None])[0] for s in range(200)}
    assert seen_k == {0, 1}
    seen_p = {sample(x, [P(temperature=1.0, top_p=0.5)], [s], [0], [None])[0] for s in range(200)}
    assert seen_p == {0}  # p(0) = 0.64 alone covers 0.5
    seen_1 = {sample(x, [P(temperature=1.0, top_k=1)], [s], [0], [None])[0] for s in range(50)}
    assert seen_1 == {0}


def test_distribution_follows_softmax():
    x = torch.tensor([[1.0, 0.0, -1.0]])
    n = 4000
    c = torch.bincount(torch.tensor([sample(x, [P(temperature=1.0)], [7], [i], [None])[0] for i in range(n)]), minlength=3)
    want = torch.softmax(x[0], 0) * n
    assert ((c - want).abs() < 4 * (want * (1 - want / n)).sqrt()).all()


def test_mask_and_failure_rows():
    x = lg(3, 10)
    m = torch.zeros(10, dtype=torch.bool)
    m[3] = True
    nan = x.clone()
    nan[2, 4] = float("nan")
    out = sample(nan, [P(), P(temperature=1.0), P()], [0, 1, 2], [0, 0, 0], [m, m, None])
    assert out == [3, 3, -1]
    assert sample(x[:1], [P()], [0], [0], [torch.zeros(10, dtype=torch.bool)]) == [-1]


def test_validate():
    import pytest
    for bad in (P(temperature=-1), P(top_p=0), P(top_k=-1), P(max_tokens=0), P(n=2), P(stop_sequences=((),))):
        with pytest.raises(ValueError):
            bad.validate()
