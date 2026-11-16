"""One engine API over every model type: open(pack, weights) -> Model, whose task kind comes from the pack manifest.

  generate    model.type decoder: prompts -> tokens (batch ladder, prefill chain or seq form, sampling; a drafter's weights open the
              pack's speculative pair, its rounds run on the device)
  embed       model.type embedding: token lists -> pooled float32 rows
  transcribe  model.type media (media rows into a decoder) or encoder-decoder (cross-attention): audio -> tokens

capabilities(pack) reads the manifest only (no blob, no device). A new model is a pack: nothing here names one."""
import io
import json
import os

from . import pack as P, registry as R

KIND = {"decoder": "generate", "embedding": "embed", "encoder-decoder": "transcribe", "media": "transcribe"}


def manifest(pack):
    with io.open(pack, "rb") as f:     # io.open: this module defines open()
        return json.loads(P.read_container(f, os.path.getsize(pack))[2])


def capabilities(pack):
    """{type, kind, bucket kinds, media kinds, contracts, ladder columns, prefill forms {W: form}, spec policy kinds, drafter roles}."""
    m = manifest(pack)
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
    """A pack opened on one device for its task kind. B, ctx, plen, numerics, drafter: decoder handles only."""

    def __init__(self, pack, weights, *, drafter=None, B=1, ctx=None, plen=None, numerics="invariant", device=-1, sku=None,
                 trusted=None, store=None, cfg=None, eos_ids=(), **kw):
        self.type = R.pack_type(pack)
        self.kind = KIND[self.type]
        if self.type in ("decoder", "encoder-decoder"):
            kw.update(B=B, ctx=ctx, plen=plen, numerics=numerics, drafter_weights=drafter and files(drafter))
        self.backend = R.open_backend(pack, files(weights), mesh=R.Mesh.single(device), sku=sku, trusted=trusted, store=store, **kw)
        self.cfg = cfg
        if self.type in ("decoder", "encoder-decoder") and cfg is None:   # the media path stays torch free: a pinned body binds its own
            from .scheduler import Config                                  # cuBLAS, and torch preloads the wheels' copies
            self.cfg = Config(max_model_len=getattr(self.backend, "max_len", None) or 4096)
        self.runner = R.runner(self.backend, self.cfg, **({"eos_ids": eos_ids} if self.type == "decoder" else {}))

    def _need(self, kind):
        P.need(self.kind == kind, f"this pack is model.type {self.type} (task {self.kind}), not {kind}")

    def generate(self, prompts, max_tokens=64, params=None):
        """Per prompt (a token id list) the generated ids; greedy unless params (SamplingParams) say otherwise."""
        from .sampling import SamplingParams
        self._need("generate")
        sp = params or SamplingParams(max_tokens=max_tokens)
        return [o[0] for o in self.runner.generate([(list(p), sp) for p in prompts])]

    def embed(self, items, norm=True):
        self._need("embed")
        return self.runner.embed(items, norm)

    def transcribe(self, media, prompt, max_tokens=2048):
        """media: PCM arrays (one per marker of prompt, a chat-templated string) for a media pack; frames [T, F] and decoder prompt ids
        for an encoder-decoder pack."""
        self._need("transcribe")
        if self.type == "media":
            return self.runner.transcribe(media, prompt, max_tokens)
        from .sampling import SamplingParams
        self.runner.add_request(0, prompt, SamplingParams(max_tokens=max_tokens), frames=media)
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
