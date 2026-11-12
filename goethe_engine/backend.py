from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class KVGeometry:
    block_size: int
    num_blocks: int
    layers: int
    kv_heads: int
    head_dim: int
    dtype: str = "f16"  # "f16" | "fp8"

    @property
    def bytes_per_block(self):
        return 2 * self.layers * self.block_size * self.kv_heads * self.head_dim * (1 if self.dtype == "fp8" else 2)


@dataclass
class Work:
    sid: int  # sequence id, stable while the sequence lives
    tokens: list  # tokens to run; token i writes KV row pos + i
    pos: int
    blocks: list  # block table covering rows 0 .. pos + len - 1
    want: int  # logits rows wanted: those of the last `want` tokens
    greedy: bool = False  # every wanted row is argmax with no mask: the backend may return tokens instead of logits
    spec: bool = False  # a decode the backend may advance by a whole speculative round (needs spec_width)
    parents: list | None = None  # tree verify: parent index of each token (-1: the cached prefix); None: a chain
    feats: bool = False  # also return the tap features of every token row (caps taps)


class Backend(Protocol):
    geometry: KVGeometry
    vocab_size: int
    max_seqs: int

    def step(self, work: list):
        """float32 logits [sum(want), vocab] in work order; rows must not depend on the rest of the batch. Token i runs
        at position pos + depth(i) (spec.depths) and attends to rows < pos and to its ancestors' rows.
        If every item is greedy the backend may instead return one token list per item: want tokens (the argmax of
        each wanted row), or for a spec item 1 .. spec_width tokens (a round: the engine takes them as accepted drafts).
        If any item has feats: returns (rows, feats), feats [sum(len(tokens) of feats items), sum(tap widths)]."""

    def copy_blocks(self, pairs: list) -> None:
        """KV rows of block src into block dst, for each (src, dst)."""

    def release(self, sid: int) -> None:
        """The sequence left the batch (finished, aborted or preempted)."""

# Optional: paged (False: a block is a whole sequence page, block_size = max_len: KV rows, window ring and recurrent state),
# state_pages (a page backend: spare pages num_blocks ... num_blocks + state_pages - 1 for copy_blocks only, its prefix cache
# keeps snapshots there; 0: no prefix cache), max_len (longest sequence the backend runs),
# spec_width (> 0: the backend drafts and verifies on its own, see Work.spec), caps (spec.CAPS keys: verify width,
# tree, taps, recurrent, logits, drafters), commit(sid, base, path): after a verify, the rows base + path[j] become
# rows base + j and the recurrent state is the one after token path[-1] (path[0] = 0; called for every item that
# carried drafts).
