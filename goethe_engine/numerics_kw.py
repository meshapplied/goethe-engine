"""Numerics contract keywords of every command line (mirror of goethe-compiler compiler/numerics/keywords.py).

invariant     batch-invariant (BI), the shipped default                    -> exact | deterministic | fast axis: deterministic
invariant-hp  batch-invariant with 16-bit activations (BIH)                -> deterministic, act a16 unless one is asked
reference     bitwise llama.cpp validation tier (E0), only on request      -> exact
tolerance     the later tolerance contract (fast: deterministic until a tolerance bucket exists)
Old spellings exact | deterministic print a deprecation warning."""
import sys

KEYWORDS = {"invariant": "deterministic", "invariant-hp": "deterministic", "reference": "exact", "tolerance": "fast", "fast": "fast"}
OLD = {"exact": "reference", "deterministic": "invariant"}
DEFAULT = "invariant"


def resolve(numerics, act="auto", warn=True):
    """(keyword or old name, act) -> (internal numerics, act): exact | deterministic | fast."""
    n = numerics.strip().lower()
    if n in KEYWORDS:
        return KEYWORDS[n], ("a16" if n == "invariant-hp" and act == "auto" else act)
    if n in OLD:
        if warn:
            print(f"warning: numerics {n} is deprecated, use {OLD[n]}", file=sys.stderr)
        return n, act
    raise ValueError(f"numerics {numerics!r}: one of {' '.join(KEYWORDS)} (old names: {' '.join(OLD)})")


CHOICES = tuple(KEYWORDS) + tuple(OLD)
