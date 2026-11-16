"""Refusals, errors and help text name no private tool, host or path: every string the package can show, and the messages it raises."""
import ast
import glob
import os
import re

import pytest

from goethe_engine import cli, pack as P, params
from goethe_engine.gc_c import GcBackend
from toypack import toy_pack

PKG = os.path.dirname(P.__file__)
# split so that this file does not match the repo's leak guard either
HINTS = re.compile("|".join([
    "goethe" + "-compiler", r"\bgc" + r" pack\b", "gc" + "_pack", "green" + "goblin", "little" + "boy", "/scra" + "tch/", "/ho" + "me/[a-z]",
    r"\bcompiler/", r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.(?!1\.)\d+\.\d+",
]))


def strings():
    for f in sorted(glob.glob(os.path.join(PKG, "*.py"))):
        for n in ast.walk(ast.parse(open(f, encoding="utf-8").read())):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                yield os.path.basename(f), n.value
    for f in sorted(glob.glob(os.path.join(params.SPEC, "**", "*.json"), recursive=True)):
        yield os.path.relpath(f, PKG), open(f, encoding="utf-8").read()


def test_no_string_in_the_package_hints_at_a_private_tool():
    bad = [(f, m.group(0)) for f, s in strings() for m in [HINTS.search(s)] if m]
    assert bad == []


def test_help_text_has_no_hints(capsys):
    for argv in ([], ["serve"], ["worker"], ["index", "verify"], ["pack"]):
        try:
            cli.main(argv + ["--help"])
        except SystemExit:
            pass
    assert HINTS.search(capsys.readouterr().out) is None


def test_refusal_messages_have_no_hints(tmp_path, key, mocklib, capsys):
    def mut(m):
        m["variants"]["q4"]["desc"] = dict(format="gguf", quant="q4_0", registry_key="toy")
    pack, w, dw = toy_pack(str(tmp_path / "a"), key, mocklib, mutate=mut)
    msgs = []
    for call in (lambda: GcBackend(pack, dw, sku="mock", trusted=(key.kid,)),      # weights no variant fits
                 lambda: GcBackend(pack, w, sku="mock", trusted=()),               # key not pinned
                 lambda: GcBackend(str(tmp_path / "missing.gcpack"), w, sku="mock"),
                 lambda: GcBackend(pack, w, sku="mock", trusted=(key.kid,), ctx=10**9)):
        with pytest.raises((Exception,)) as e:
            call()
        msgs.append(str(e.value))
    for argv in (["pack", "verify", pack], ["pack", "runs", pack], ["pack", "show", str(tmp_path / "nope")], ["pack", "need", pack],
                 ["index", "verify", "--index", str(tmp_path / "i"), "--sig", str(tmp_path / "s")]):
        with pytest.raises(SystemExit) as e:
            code = cli.main(argv)
            raise SystemExit(code)
        msgs.append(str(e.value.code) + capsys.readouterr().err)
    msgs = [m.replace(str(tmp_path), "<tmp>") for m in msgs]           # the caller's own paths are not the package's text
    assert all(msgs) and [m for m in msgs if HINTS.search(m)] == []
