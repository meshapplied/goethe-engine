"""Paged KV bookkeeping. BlockPool follows vLLM v1 block_pool (ref counts), RadixCache follows SGLang radix_cache
(prefix tree, LRU leaf eviction) at block granularity, KVManager follows vLLM v1 kv_cache_manager (allocate, CoW, free).
A page pack's block is a whole sequence page (KV rows, window ring, recurrent state): the same allocator, with StateCache
as its prefix cache."""
import heapq


class BlockPool:
    def __init__(self, n):
        self.ref = [0] * n
        self.free = list(range(n - 1, -1, -1))

    def take(self, n):
        out = [self.free.pop() for _ in range(n)]
        for b in out:
            self.ref[b] = 1
        return out

    def incref(self, b):
        assert self.ref[b] > 0, f"incref of free block {b}"
        self.ref[b] += 1

    def decref(self, b):
        assert self.ref[b] > 0, f"double free of block {b}"
        self.ref[b] -= 1
        if self.ref[b] == 0:
            self.free.append(b)


class Node:
    __slots__ = ("pages", "blocks", "children", "parent", "last")

    def __init__(self, pages, blocks, parent):
        self.pages, self.blocks, self.parent, self.children, self.last = pages, blocks, parent, {}, 0


class RadixCache:
    """Radix tree over token pages. The tree holds one ref per block it indexes; a block with another ref is in use."""

    def __init__(self, pool, bs):
        self.pool, self.bs, self.tick = pool, bs, 0
        self.root = Node([], [], None)

    def _page(self, tokens, i):
        return tuple(tokens[i * self.bs:(i + 1) * self.bs])

    def match(self, tokens, max_blocks, touch=True):
        """Blocks of the longest cached prefix of tokens, at most max_blocks."""
        self.tick += touch
        out, cur = [], self.root
        while len(out) < max_blocks:
            child = cur.children.get(self._page(tokens, len(out)))
            if child is None:
                break
            k, i = 0, len(out)
            while k < len(child.pages) and i + k < max_blocks and child.pages[k] == self._page(tokens, i + k):
                k += 1
            if touch:
                child.last = self.tick
            out += child.blocks[:k]
            if k < len(child.pages):
                break
            cur = child
        return out

    def _split(self, node, k):
        low = Node(node.pages[k:], node.blocks[k:], node)
        low.children, low.last = node.children, node.last
        for c in low.children.values():
            c.parent = low
        node.pages, node.blocks, node.children = node.pages[:k], node.blocks[:k], {low.pages[0]: low}

    def insert(self, tokens, blocks, start=0):
        """Index blocks for tokens[:len(blocks) * bs]; pages below start are known to be in the tree already.

        Returns the tree's block for every page from start on (a block already indexed wins over the caller's)."""
        n, cur, i, canon = len(blocks), self.root, 0, []
        self.tick += 1
        while i < n:
            child = cur.children.get(self._page(tokens, i))
            if child is None:
                new = Node([self._page(tokens, j) for j in range(i, n)], list(blocks[i:]), cur)
                for b in new.blocks:
                    self.pool.incref(b)
                new.last = self.tick
                cur.children[new.pages[0]] = new
                canon += blocks[max(i, start):]
                break
            k = 0
            while k < len(child.pages) and i + k < n and (i + k < start or child.pages[k] == self._page(tokens, i + k)):
                k += 1
            if k < len(child.pages):
                self._split(child, k)
            canon += child.blocks[max(0, start - i):]
            child.last = self.tick
            cur, i = child, i + k
        return canon

    def evict(self, n):
        """Free up to n blocks from least recently used leaves, trimming a leaf from its tail past unused blocks."""
        heap, stack, freed = [], [self.root], 0
        while stack:
            nd = stack.pop()
            stack += nd.children.values()
            if not nd.children and nd is not self.root:
                heap.append((nd.last, id(nd), nd))
        heapq.heapify(heap)
        while freed < n and heap:
            _, _, nd = heapq.heappop(heap)
            key = nd.pages[0]
            while freed < n and nd.blocks and self.pool.ref[nd.blocks[-1]] == 1:
                nd.pages.pop()
                self.pool.decref(nd.blocks.pop())
                freed += 1
            if not nd.blocks:
                p = nd.parent
                del p.children[key]
                if not p.children and p is not self.root:
                    heapq.heappush(heap, (p.last, id(p), p))
        return freed

    def nodes(self):
        stack, out = [self.root], []
        while stack:
            nd = stack.pop()
            out.append(nd)
            stack += nd.children.values()
        return out


class StateCache:
    """Prefix cache of a page pack. A page's recurrent state and window ring are valid at one position only, so a snapshot
    is a whole-page copy into a spare page, keyed by the exact tokens it followed, and a hit copies it back: no partial
    sharing. Spare pages are ids first .. first + n - 1; the least recently used snapshot is evicted for a new one."""

    def __init__(self, first, n):
        self.first, self.n, self.free, self.keys, self.tick = first, n, list(range(first + n - 1, first - 1, -1)), {}, 0

    def match(self, tokens, touch=True):
        """(page, pos) of the longest snapshot that is a prefix of tokens[:-1] (its step gives the next token), else (None, 0)."""
        self.tick += touch
        for pos in sorted({len(k) for k in self.keys}, reverse=True):
            e = self.keys.get(tuple(tokens[:pos])) if pos < len(tokens) else None
            if e:
                if touch:
                    e[1] = self.tick
                return e[0], pos
        return None, 0

    def put(self, tokens):
        """The spare page to copy the page that followed tokens into, or None if cached already or there is no spare page."""
        key = tuple(tokens)
        self.tick += 1
        if key in self.keys:
            self.keys[key][1] = self.tick
            return None
        if not self.free and self.keys:
            self.free.append(self.keys.pop(min(self.keys, key=lambda k: self.keys[k][1]))[0])
        if not self.free:
            return None
        self.keys[key] = [self.free.pop(), self.tick]
        return self.keys[key][0]

    def check(self):
        used = [e[0] for e in self.keys.values()]
        assert sorted(self.free + used) == list(range(self.first, self.first + self.n)), "snapshot page leaked or twice owned"


class KVManager:
    def __init__(self, num_blocks, bs, prefix_cache=True, state_pages=0):
        self.bs, self.pool = bs, BlockPool(num_blocks)
        self.cache = RadixCache(self.pool, bs) if prefix_cache and not state_pages else None
        self.snaps = StateCache(num_blocks, state_pages) if prefix_cache and state_pages else None

    def match(self, tokens, touch=False):
        """Cached blocks for a prompt; the last token is never cached, its step gives the next token."""
        return self.cache.match(tokens, (len(tokens) - 1) // self.bs, touch) if self.cache else []

    def hit(self, tokens):
        """Prompt tokens the prefix cache would serve."""
        return self.snaps.match(tokens, False)[1] if self.snaps else len(self.match(tokens)) * self.bs

    def attach(self, seq):
        if self.snaps:
            seq.restore, seq.computed = self.snaps.match(seq.tokens)
            seq.blocks, seq.tpages = [], 0
            return
        seq.blocks = self.match(seq.tokens, touch=True)
        for b in seq.blocks:
            self.pool.incref(b)
        seq.tpages = len(seq.blocks)
        seq.computed = seq.tpages * self.bs

    def reserve(self, seq, upto):
        """Make the blocks for rows [seq.computed, upto) writable. Returns the CoW copies (src, dst) or None if the pool is
        short, in which case nothing changed."""
        bs, pool = self.bs, self.pool
        need = -(-upto // bs)
        cow = [i for i in range(seq.computed // bs, min(len(seq.blocks), need)) if pool.ref[seq.blocks[i]] > 1]
        fresh = max(0, need - len(seq.blocks))
        short = len(cow) + fresh - len(pool.free)
        if short > 0 and self.cache:
            self.cache.evict(short)
        if len(cow) + fresh > len(pool.free):
            return None
        pairs = []
        for i in cow:
            dst, = pool.take(1)
            pairs.append((seq.blocks[i], dst))
            pool.decref(seq.blocks[i])
            seq.blocks[i] = dst
        seq.blocks += pool.take(fresh)
        if seq.restore is not None:
            pairs.append((seq.restore, seq.blocks[0]))
            seq.restore = None
        return pairs

    def snapshot(self, seq):
        """Copies (page, spare) that save the sequence's page at its computed position, if that is worth a snapshot."""
        spare = self.snaps.put(seq.tokens[:seq.computed]) if self.snaps and seq.blocks and seq.computed else None
        return [] if spare is None else [(seq.blocks[0], spare)]

    def commit(self, seq):
        """Index the full blocks of the computed rows; adopt the tree's block where another sequence indexed it first.
        A page pack snapshots at the last prompt token (every prompt step ends there) and returns the copies to run."""
        if self.snaps:
            return self.snapshot(seq) if seq.computed == seq.nprompt - 1 else []
        nf = seq.computed // self.bs
        if not self.cache or nf <= seq.tpages:
            return []
        canon = self.cache.insert(seq.tokens, seq.blocks[:nf], seq.tpages)
        for j, c in enumerate(canon, seq.tpages):
            b = seq.blocks[j]
            if c != b:
                self.pool.incref(c)
                self.pool.decref(b)
                seq.blocks[j] = c
        seq.tpages = nf
        return []

    def release(self, seq):
        for b in seq.blocks:
            self.pool.decref(b)
        seq.blocks, seq.computed, seq.tpages, seq.restore = [], 0, 0, None

    def fork(self, seq, child):
        child.blocks, child.computed, child.tpages = list(seq.blocks), seq.computed, seq.tpages
        for b in child.blocks:
            self.pool.incref(b)

    def check(self, seqs):
        """Every ref is explained by a sequence table or a tree node, and the free list is exactly the zero refs."""
        want = [0] * len(self.pool.ref)
        for s in seqs:
            assert len(set(s.blocks)) == len(s.blocks), "block twice in one table"
            for b in s.blocks:
                want[b] += 1
        for nd in self.cache.nodes() if self.cache else []:
            assert len(nd.pages) == len(nd.blocks)
            for b in nd.blocks:
                want[b] += 1
        assert want == self.pool.ref, "ref counts do not match owners"
        assert sorted(self.pool.free) == [b for b, r in enumerate(want) if r == 0], "free list mismatch"
        if self.snaps:
            self.snaps.check()
