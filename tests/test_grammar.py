import json
import re

import pytest

pytest.importorskip("xgrammar")
from conftest import FakeBackend, V, toks, run
from goethe_engine import Config, Engine, SamplingParams as P
from goethe_engine.grammar import XGrammarCompiler
from goethe_engine.spec import NgramDrafter

CH = list(' {}":,-0123456789abcdefghijklmnopqrstuvwxyz')
VOCAB = CH + ["true", "false", "null", '{"', '":', "ab", "12", "x1", "--", "ok"][:V - 1 - len(CH)]
VOCAB += ["<pad%d>" % i for i in range(V - 1 - len(VOCAB))] + ["</s>"]
EOS = V - 1
SCHEMA = json.dumps({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}},
                     "required": ["a", "b"], "additionalProperties": False})


def text(ids):
    return "".join(VOCAB[i] for i in ids if i != EOS)


def eng(**kw):
    be = FakeBackend(kw.pop("blocks", 64), 4)
    return Engine(be, Config(**{"max_num_seqs": 8, **kw}), grammar=XGrammarCompiler(VOCAB, [EOS], V), eos_ids=[EOS])


@pytest.mark.parametrize("seed", range(4))
def test_json_schema_output_is_valid(seed):
    e = eng()
    p = P(temperature=1.0, seed=seed, max_tokens=60, grammar=("json_schema", SCHEMA))
    e.add_request("a", [1, 2, 3], p)
    out = run(e)
    o = [x for x in out if x.finished][0]
    t = toks(out)[("a", 0)]
    assert o.finish_reason == "stop" and t[-1] == EOS
    d = json.loads(text(t))
    assert set(d) == {"a", "b"} and isinstance(d["a"], int) and isinstance(d["b"], bool)


def test_regex_greedy_and_batched_equals_alone_with_preemption_and_spec():
    p = P(max_tokens=30, grammar=("regex", "[0-9]{3}-[a-z]{2}"))
    reqs = [([i + 1, i + 2, 5], p if i % 2 else P(temperature=0.8, seed=i, max_tokens=30, grammar=p.grammar)) for i in range(6)]
    e = eng(blocks=14, max_num_batched_tokens=6, spec_k=3)
    e.sched.drafter = NgramDrafter()
    for i, (pr, sp) in enumerate(reqs):
        e.add_request(i, pr, sp)
    got = toks(run(e))
    for i, (pr, sp) in enumerate(reqs):
        assert re.fullmatch(r"[0-9]{3}-[a-z]{2}", text(got[(i, 0)])), got[(i, 0)]
    ref_e = eng(blocks=256, prefix_cache=False, max_num_seqs=1, max_num_batched_tokens=4096)
    for i, (pr, sp) in enumerate(reqs):
        ref_e.add_request(("r", i), pr, sp)
        assert toks(run(ref_e, check=False))[(("r", i), 0)] == got[(i, 0)]


def test_bad_grammar_is_a_value_error_and_xgrammar_is_swappable():
    e = eng()
    with pytest.raises(ValueError):
        e.add_request("a", [1], P(grammar=("regex", "(")))
    with pytest.raises(ValueError):
        Engine(FakeBackend(), Config()).add_request("a", [1], P(grammar=("regex", "a")))
