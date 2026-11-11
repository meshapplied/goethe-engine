"""Structured output behind the engine's own interface: a compiler makes a Matcher per sequence.

Matcher: mask() -> bool tensor [vocab] of allowed next tokens; accept(tok) -> bool; done() -> bool.
Compiler: compile(kind, spec) -> object with new_matcher(). kind is "json_schema", "regex", "ebnf" or "json"."""
import torch


class XGrammarCompiler:
    def __init__(self, vocab, eos_ids=(), vocab_size=None):
        import xgrammar as xgr
        self.xgr, self.vocab_size = xgr, vocab_size or len(vocab)
        info = xgr.TokenizerInfo(vocab, vocab_size=self.vocab_size, stop_token_ids=list(eos_ids))
        self.c = xgr.GrammarCompiler(info)
        self.cache = {}

    def compile(self, kind, spec):
        key = (kind, spec)
        if key not in self.cache:
            c = self.c
            try:
                g = {"json_schema": lambda: c.compile_json_schema(spec), "regex": lambda: c.compile_regex(spec),
                     "ebnf": lambda: c.compile_grammar(spec), "json": c.compile_builtin_json_grammar}[kind]()
            except Exception as e:
                raise ValueError(f"grammar does not compile: {e}") from e
            self.cache[key] = _Compiled(self.xgr, g, self.vocab_size)
        return self.cache[key]


class _Compiled:
    def __init__(self, xgr, g, v):
        self.xgr, self.g, self.v = xgr, g, v

    def new_matcher(self):
        return _Matcher(self.xgr, self.xgr.GrammarMatcher(self.g), self.v)


class _Matcher:
    def __init__(self, xgr, m, v):
        self.xgr, self.m, self.v = xgr, m, v
        self.bm = xgr.allocate_token_bitmask(1, v)

    def mask(self):
        self.m.fill_next_token_bitmask(self.bm, 0)
        bits = (self.bm[0][:, None] >> torch.arange(32, dtype=torch.int32)) & 1
        return bits.reshape(-1)[:self.v].bool()

    def accept(self, tok):
        return self.m.accept_token(tok)

    def done(self):
        return self.m.is_terminated()
