"""One engine API over every model type: open(pack, weights) -> Model, whose task kind comes from the pack manifest.

  generate    model.type decoder: prompts -> tokens (batch ladder, prefill chain or seq form, sampling; a drafter's weights open the
              pack's speculative pair, its rounds run on the device)
  embed       model.type embedding: token lists -> pooled float32 rows
  transcribe  model.type media (media rows into a decoder) or encoder-decoder (cross-attention): audio -> tokens

capabilities(pack) reads the manifest only (no blob, no device). A new model is a pack: nothing here names one."""
import io
import json
import os

from . import pack as P, params as P_, registry as R
from .errors import UnsupportedParameter

KIND = {"decoder": "generate", "embedding": "embed", "encoder-decoder": "transcribe", "media": "transcribe"}


def manifest(pack):
    with io.open(pack, "rb") as f:     # io.open: this module defines open()
        return json.loads(P.read_container(f, os.path.getsize(pack))[2])


def capabilities(pack):
    """{type, kind, bucket kinds, media kinds, contracts, ladder columns, prefill forms {W: form}, spec policy kinds, drafter roles} of a
    pack file (of the base file of [base, overlays...])."""
    m = manifest(pack[0] if isinstance(pack, (list, tuple)) else pack)
    t = R.model_type(m)
    bs = m["buckets"]
    return dict(type=t, kind=KIND[t], kinds=sorted({P.axes(b)[1] for b in bs}),
                media=sorted({(b.get("io") or {}).get("media") for b in bs} - {None}), contracts=m["contracts"],
                ladder=sorted({b["ncol"] for b in bs if b.get("batch")}),
                prefill={b["prefill"]: b.get("form", "chain") for b in bs if b.get("prefill")},
                spec=sorted({s["policy"].get("kind") for s in m.get("spec", [])}), skus=sorted({b["sku"] for b in bs}))


def files(weights):
    """Weight file paths: a directory stands for its *.safetensors shards (an HF checkpoint), sorted."""
    out = []
    for w in [weights] if isinstance(weights, str) else weights:
        out += sorted(os.path.join(w, f) for f in os.listdir(w) if f.endswith(".safetensors")) if os.path.isdir(w) else [w]
    return out


class Model:
    """A pack (a file, or [base, overlays...]) opened on one device for its task kind. B, ctx, plen, numerics, act, kv, drafter: decoder
    handles only. config: scheduler Config fields over the defaults the backend implies (its max_len, prefix cache, max_seqs cap)."""

    def __init__(self, pack, weights, *, drafter=None, B=1, ctx=None, plen=None, numerics="invariant", act="auto", kv="auto", device=-1,
                 sku=None, trusted=None, store=None, config=None, eos_ids=(), **kw):
        self.type = R.pack_type(pack)
        self.kind = KIND[self.type]
        if self.type in ("decoder", "encoder-decoder"):
            kw.update(B=B, ctx=ctx, plen=plen, act=act, kv=kv, drafter_weights=drafter and files(drafter))
        self.backend = R.open_backend(pack, files(weights), mesh=R.Mesh.single(device), sku=sku, trusted=trusted, store=store,
                                      numerics=numerics, **kw)
        self.cfg = None
        if self.type in ("decoder", "encoder-decoder"):    # the media path stays torch free: a pinned body binds its own cuBLAS,
            from dataclasses import replace                 # and torch preloads the wheels' copies
            from .scheduler import Config
            be, c = self.backend, dict(config or {})
            c["max_num_seqs"] = min(c.get("max_num_seqs") or be.max_seqs, be.max_seqs)
            self.cfg = replace(Config(max_model_len=getattr(be, "max_len", None) or 4096, prefix_cache=getattr(be, "prefix_cache", None)),
                               **{k: v for k, v in c.items() if v is not None})
        self.runner = R.runner(self.backend, self.cfg, **({"eos_ids": eos_ids} if self.type == "decoder" else {}))

    def _need(self, kind):
        P.need(self.kind == kind, f"this pack is model.type {self.type} (task {self.kind}), not {kind}")

    def check(self, **req):
        """The request parameters validated against the task schema and this pack's io.params (params.check)."""
        return P_.check(self.kind, req, getattr(self.backend, "bucket", None))

    def sampling(self, **req):
        """SamplingParams of a generate request (checked)."""
        from .sampling import SamplingParams
        r = self.check(**req)
        for k in ("logprobs",):
            if k in r:
                raise UnsupportedParameter(f"{k!r}: declared by the pack, not implemented by this engine")
        if "structured" in r:
            r["grammar"] = (r["structured"]["kind"], r.pop("structured")["spec"])
        if "stop" in r:
            r["stop_sequences"] = tuple(r.pop("stop"))
        if "stop_token_ids" in r:
            r["stop_token_ids"] = tuple(r["stop_token_ids"])
        return SamplingParams(**r)

    def generate(self, prompts, max_tokens=64, **params):
        """Per prompt (a token id list) the generated ids; greedy unless params say otherwise (spec/params/generate.schema.json)."""
        self._need("generate")
        sp, r = self.sampling(max_tokens=max_tokens, **params), self.runner
        out = {}
        for i, p in enumerate(prompts):
            r.add_request(i, list(p), sp)
        while r.has_work():
            for o in r.step():
                if o.finish_reason == "error":         # a failed step (device error, no valid token) is raised, never an empty result
                    raise RuntimeError(f"request {o.rid}: {o.error or 'error'}")
                out.setdefault((o.rid, o.index), []).extend(o.token_ids)
        return [out.get((i, 0), []) for i in range(len(prompts))]

    def embed(self, items, normalize=True, **params):
        """float32 rows of token lists (spec/params/embed.schema.json): dimensions keeps the leading components (then normalizes),
        truncation right cuts an input to the longest the pack embeds."""
        self._need("embed")
        r = self.check(normalize=normalize, **params)
        if r.get("truncation") == "right":
            items = [list(t)[:min(self.backend.max_len, self.backend.max_cols)] for t in items]
        if "dimensions" not in r:
            return self.runner.embed(items, r["normalize"])
        from .embed import normalize as l2
        x = self.runner.embed(items, False)[:, :r["dimensions"]]
        return l2(x) if r["normalize"] else x

    def transcribe(self, media, prompt, max_new_tokens=2048, **params):
        """media: PCM arrays (one per marker of prompt, a chat-templated string) for a media pack; frames [T, F] and decoder prompt ids
        for an encoder-decoder pack (spec/params/transcribe.schema.json)."""
        self._need("transcribe")
        r = self.check(max_new_tokens=max_new_tokens, **params)
        extra = sorted(set(r) - {"max_new_tokens"})
        if extra:
            raise UnsupportedParameter(f"{extra}: declared by the pack, not implemented by this engine's {self.type} runner")
        if self.type == "media":
            return self.runner.transcribe(media, prompt, r["max_new_tokens"])
        from .sampling import SamplingParams
        self.runner.add_request(0, prompt, SamplingParams(max_tokens=r["max_new_tokens"]), frames=media)
        out = []
        while self.runner.has_work():
            for o in self.runner.step():
                out.extend(o.token_ids)
        return out

    def close(self):
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def open(pack, weights, **kw):
    return Model(pack, weights, **kw)
