"""The README's CLI reference matches the real parsers: every command and flag documented, none invented, every --help works."""
import os
import re

import pytest

from goethe_engine import cli, index, numerics_kw, pack, serve

README = open(os.path.join(os.path.dirname(__file__), "..", "README.md"), encoding="utf-8").read()
FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z-]*)")


def flags(ap):
    return {s for act in ap._actions for s in act.option_strings if s.startswith("--") and s != "--help"}


def choices(ap, dest):
    return next(a.choices for a in ap._actions if a.dest == dest)


IDX, PK = index.parser(), pack.parser()
# command -> (flag group, parser): the commands of one parser share its flag set
GROUPS = {
    "serve": ("serve", serve.parser()),
    "worker": ("worker", cli.worker_parser()),
    "index verify": ("index", IDX),
    "pack match": ("index", IDX),
    **{f"pack {c}": ("pack", PK) for c in choices(PK, "cmd")},
}


def sections():
    """{command: text} of the '### `command`' blocks of the CLI reference."""
    ref = README.split("\n## CLI reference\n", 1)[1].split("\n## ", 1)[0]
    parts = re.split(r"^### ", ref, flags=re.M)[1:]
    return {re.match(r"`([^`]+)`", p).group(1): p for p in parts if p.startswith("`")}


def test_every_real_command_is_documented_and_no_other():
    assert set(sections()) == set(GROUPS)


def test_documented_flags_exist_and_every_flag_is_documented():
    docs = sections()
    seen = {}
    for cmd, text in docs.items():
        group, ap = GROUPS[cmd]
        got = set(FLAG.findall(text))
        assert got <= flags(ap), f"{cmd}: documented flags not in the parser: {sorted(got - flags(ap))}"
        seen.setdefault(group, set()).update(got)
    for group, ap in {g: p for g, p in GROUPS.values()}.items():
        assert seen[group] == flags(ap), f"{group}: parser flags missing from the README: {sorted(flags(ap) - seen[group])}"
    # index.parser() carries match-only flags; both index commands together must cover it
    assert seen["index"] == flags(index.parser())


def test_commands_table_lists_every_command_with_its_exit_codes():
    table = README.split("| Command |", 1)[1].split("\n\n", 1)[0]
    rows = {m.group(1): m.group(2) for m in re.finditer(r"^\| `([^`]+)` \|.*\| ([0-9, ]+) \|$", table, flags=re.M)}
    assert set(rows) == set(GROUPS)
    assert "3" in rows["pack runs"].split(", ") and "3" in rows["pack match"].split(", ")
    assert not any("3" in v.split(", ") for k, v in rows.items() if k not in ("pack runs", "pack match"))


def test_pack_runs_documents_exit_3():
    assert "Exit 0 runs, 3 no SKU runs, 1 pack refused" in sections()["pack runs"]


def test_every_numerics_tier_is_documented_and_each_flag_value_is_real():
    tiers = README.split("### Numerics tiers", 1)[1]
    for name in numerics_kw.CHOICES:
        assert f"`{name}`" in tiers, name
    assert numerics_kw.DEFAULT == "invariant" and "default `invariant`" in tiers
    assert set(choices(serve.parser(), "numerics")) == set(numerics_kw.CHOICES)
    for dest, vals in (("act", "auto|a8|a16"), ("kv", "auto|f16|fp8"), ("ref_type", "decoder|embedding|encoder-decoder")):
        assert "|".join(choices(serve.parser(), dest)) == vals and vals in sections()["serve"], dest


def test_defaults_documented_match_the_parsers():
    ap = serve.parser()
    assert f"{ap.get_default('host')}:{ap.get_default('port')}" in sections()["serve"]
    assert f"default {worker_default()}" in sections()["worker"]


def worker_default():
    return cli.worker_parser().get_default("device")


def help_exit(argv):
    try:
        return cli.main(argv + ["--help"])
    except SystemExit as e:
        return e.code


@pytest.mark.parametrize("cmd", [""] + sorted(GROUPS))
def test_help_exits_0_with_text(cmd, capsys):
    assert help_exit(cmd.split()) == 0
    assert capsys.readouterr().out.strip()


def test_commands_of_the_choices_are_the_ones_the_dispatcher_routes():
    assert choices(IDX, "cmd") == ("verify", "match") and choices(PK, "cmd") == ("verify", "show", "need", "runs", "plan", "build")
