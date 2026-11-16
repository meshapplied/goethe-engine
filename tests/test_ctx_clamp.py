"""A requested max_model_len above the opened handle's limit (a stream bucket keeps rows of its own: 4092 of a 4096 context) is clamped
to that limit with one note, so a one-sequence open at the pack's context serves instead of refusing."""
from goethe_engine import api
from conftest import fake_greedy
from toypack import toy_pack


def test_max_model_len_above_the_handle_is_clamped(tmp_path, key, mocklib, capsys):
    pack, w, _ = toy_pack(str(tmp_path), key, mocklib)
    with api.open(pack, w, B=2, sku="mock", trusted=(key.kid,)) as m0:
        lim = m0.backend.max_len
    with api.open(pack, w, B=2, sku="mock", trusted=(key.kid,), config=dict(max_num_seqs=2, max_model_len=lim + 4)) as m:
        assert m.cfg.max_model_len == lim and m.generate([[1, 2, 3]], 5) == [fake_greedy([1, 2, 3], 5)]
    err = capsys.readouterr().err
    assert err.count(f"max_model_len {lim + 4} is above this handle's {lim}; using {lim}") == 1


def test_max_model_len_inside_the_handle_is_kept(tmp_path, key, mocklib, capsys):
    pack, w, _ = toy_pack(str(tmp_path), key, mocklib)
    with api.open(pack, w, B=2, sku="mock", trusted=(key.kid,), config=dict(max_num_seqs=2, max_model_len=64)) as m:
        assert m.cfg.max_model_len == 64
    assert "max_model_len" not in capsys.readouterr().err
