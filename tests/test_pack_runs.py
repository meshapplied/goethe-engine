"""`goethe-engine pack runs FILE`: does the visible device run one of the pack's SKUs (signature and manifest only, no weights)."""
import json

import pytest

from goethe_engine import cli, pack as P, sign
from toypack import toy_pack


def dev(sms):
    return lambda i=None: dict(platform="cuda", name="card", cc="12.0", sms=sms, smem_block=1, cuda="12.8", arch_ok=P.cuda_arch_ok)


def run(argv, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    return e.value.code, capsys.readouterr()


def test_runs_true_names_the_sku(tmp_path, key, mocklib, monkeypatch, capsys):
    monkeypatch.setattr(sign, "TRUSTED_KEY_IDS", (key.kid,))
    monkeypatch.setattr(P, "device", dev(170))
    pack, _, _ = toy_pack(str(tmp_path), key, mocklib)
    code, out = run(["pack", "runs", pack], capsys)
    assert code == 0 and json.loads(out.out) == dict(runs=True, sku="mock", device="card", sms=170, why=None)


def test_runs_false_says_why_and_exits_3(tmp_path, key, mocklib, monkeypatch, capsys):
    monkeypatch.setattr(sign, "TRUSTED_KEY_IDS", (key.kid,))
    monkeypatch.setattr(P, "device", dev(100))
    pack, _, _ = toy_pack(str(tmp_path), key, mocklib)
    code, out = run(["pack", "runs", pack], capsys)
    j = json.loads(out.out)
    assert code == 3 and j["runs"] is False and j["sku"] is None and j["sms"] == 100
    assert "the plan's grid needs 170 SMs co-resident, the device has 100" in j["why"]


def test_runs_refuses_an_untrusted_pack_with_exit_1(tmp_path, key, mocklib, monkeypatch, capsys):
    monkeypatch.setattr(P, "device", dev(170))
    pack, _, _ = toy_pack(str(tmp_path), key, mocklib)
    code, out = run(["pack", "runs", pack], capsys)
    assert code not in (0, 3) and "REFUSED" in str(code) and out.out == ""
