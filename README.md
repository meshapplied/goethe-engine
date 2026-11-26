# goethe-engine

goethe-engine is the serving core that [Goethe](https://github.com/meshapplied/goethe) runs. It serves text models,
embedding models and speech-to-text models through an OpenAI-compatible API.

A model runs from a signed kernel pack: one `.gcpack` file per model and GPU architecture, plus the model's own
weights (GGUF or safetensors). Kernel packs are built and signed separately. A pack fits every model of the same
structure, finetunes included.

## Install

If you use Goethe, you don't need to do anything: Goethe installs goethe-engine and keeps it up to date.

To install it on its own, take the wheel from the [latest release](https://github.com/meshapplied/goethe-engine/releases):

    pip install goethe_engine-<version>-py3-none-any.whl

Add `[serve]` for tokenizers and chat templates, `[ref]` for the CPU reference model, `[worker]` for the worker protocol.

## Run

    goethe-engine serve --pack model.gcpack --weights model.gguf

The server listens on 127.0.0.1:30000 and speaks `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
`/v1/audio/transcriptions` and `/v1/models`. To try the API without a GPU, run `goethe-engine serve --ref`.
The other commands (checking a pack or a pack index, the worker) are in the CLI reference below.

## Safe by default

goethe-engine works offline and fails closed. It checks the pack's signature, every piece of the pack it reads, the
ABI and the GPU before it loads anything, and refuses the pack if any check fails.

## CLI reference

`goethe-engine <command> [flags]`. The `index` and `pack` commands work offline, print one JSON document (text lines for
`pack show`) on stdout and the reason on stderr. Exit 0 ok, 1 refused or failed, 2 usage error, 3 "no match" (`pack match`, `pack runs` only).
`goethe-engine --help` and `<command> --help` print the full flag text; `goethe-engine --version` prints the version.

| Command | Purpose | Exit |
|---|---|---|
| `serve` | OpenAI-compatible server for a pack and its weights | 0, 1, 2 |
| `worker` | worker protocol v1 over a unix socket | 0, 2 |
| `index verify` | check a signed pack index | 0, 1, 2 |
| `index build` | write and sign a pack index from signed packs | 0, 1, 2 |
| `pack match` | pick the pack for a model directory from an index | 0, 1, 2, 3 |
| `pack verify` | check a pack's signature and blobs | 0, 1, 2 |
| `pack show` | one line per bucket of a pack | 0, 1, 2 |
| `pack need` | byte ranges to download for a model directory | 0, 1, 2 |
| `pack runs` | does the visible device run this pack | 0, 1, 2, 3 |
| `pack plan` | what each variant downloads, from a pack's first bytes | 0, 1, 2 |
| `pack build` | build a pack from a declarative pack spec | 0, 1, 2 |
| `pack sign` | sign a pack's manifest (writes `PACK.sig`) | 0, 1, 2 |

### `serve`

Serves a pack with its weights, or `--ref` for the CPU reference model. Listens on 127.0.0.1:30000. Needs `[serve]`.
Exit 1: pack refused, or `--pack`/`--weights` missing without `--ref`.

- `--pack FILE`, `--weights PATH...` (target, then a drafter; a media model: projector, then text model), `--overlay FILE` (repeatable), `--store DIR`
  (pack 4 blob store, default `$GOETHE_BLOB_STORE`), `--tokenizer DIR`, `--instruction TEXT` (media models), `--device N`
- `--host H`, `--port P`, `--api-key K`, `--served-model-name S`, `--context-length N`, `--max-running-requests N`
- `--numerics TIER` (below), `--act auto|a8|a16`, `--kv auto|f16|fp8`
- `--ref`, `--ref-type decoder|embedding|encoder-decoder`

    goethe-engine serve --pack model.gcpack --weights model.gguf --numerics invariant

### `worker`

Worker protocol v1 (`goethe_engine/spec/proto/worker.proto`). Needs `[worker]`.

- `--socket PATH` (required), `--device N` (default 0)

    goethe-engine worker --socket /tmp/ge.sock

### `index verify`

Checks the index signature, the pinned key, the schema, expiry and rollback. Prints `{"serial", "expires", "keyid", "entries"}`.
Exit 1: bad signature, key not pinned, schema, `serial < --last-serial`, expired.

- `--index F`, `--sig F` (both required), `--last-serial N` (default 0), `--now RFC3339` (tests)

    goethe-engine index verify --index index.json --sig index.sig --last-serial 7


### `index build`

Writes `index.json` and `index.sig` into `--out` for the signed packs given with `--pack` (each with `PACK.sig` beside it), signed with
`--key`: one entry per (arch, target variant). An entry's `model` is that variant's `source` and files (name, bytes, sha256), its
`structure` that variant's fingerprint, and `model.with` the drafters its speculative pairs name, each with its own source and files.
With `--index F --sig F` (the current index) its entries for other pack files are kept, after it verifies, and `--serial` must be above
its own. The index is verified before anything is written. Exit 1: a pack older than abi minor 2, weights without sha256, files of two
formats in one variant, a current index that does not verify, a serial not above it.

- `--pack F` (repeatable), `--serial N`, `--expires YYYY-MM-DDTHH:MM:SSZ`, `--key PEM`, `--out DIR` (all required), `--index F --sig F`
  (the current index), `--tier T` (default base), `--min-engine V` (default this engine's version)

    goethe-engine index build --pack model.gcpack --serial 9 --expires 2027-03-01T00:00:00Z --key index_key.pem --out site/
### `pack match`

Prints the best index entry whose structure the files of `--model-dir` have, for `--arch`, ABI major 3, a runtime this engine runs and
`min_engine <= --engine-version`; then tier, then `manifest_sha256`. Exit 3: valid index, no entry fits. Exit 1 as `index verify`.

- `--index F`, `--sig F`, `--model-dir DIR`, `--arch sm_XX` (all required), `--last-serial N`, `--now RFC3339`, `--tier T`, `--engine-version V`

    goethe-engine pack match --index index.json --sig index.sig --model-dir ./model --arch sm_120a

### `pack verify`

Without `--partial`, extracts and hash-checks every blob. With `--partial`, checks the manifest signature and only the blobs the branch
chosen by `--model-dir` needs, and adds `"weights"` (the model files to pass to `serve` as weights) and, when the folder holds the
pack's drafter (an MTP file beside the checkpoint, for example), `"drafter"`. Prints
`{"ok", "manifest_sha256", "keyid", "abi", "arch", "runtime", "buckets", "variants"}`. Exit 1: signature missing or bad, key not pinned,
container malformed, blob missing or not hashing to its name, ABI major not 3 or 4, runtime not run here, `--arch` differs, no variant has the model's structure.

- `FILE`, `--sig F` (default `FILE.sig`), `--overlay F` (repeatable), `--store DIR`, `--arch sm_XX`
- `--partial`, `--model-dir DIR`, `--sku S`, `--numerics TIER`, `--act A`, `--kv K` (branch selection for `--partial`)

    goethe-engine pack verify model.gcpack --partial --model-dir ./model

### `pack show`

One line per bucket: id, variant, contract, ncol, maxctx, file count. Takes `FILE`, `--sig F`, `--overlay F`, `--store DIR`, `--arch sm_XX`.

    goethe-engine pack show model.gcpack

### `pack need`

Reads a PREFIX file (the first bytes of a pack: 16 bytes, header, manifest) and prints
`{"manifest_sha256", "prefix": {"off", "len"}, "ranges": [{"off", "len", "sha256"}]}`: the byte ranges of the blobs the branch needs
(pack 4: also `"store"`, the hashes to fetch by name). Compare `manifest_sha256` with the index entry's; `pack verify` checks the signature
once the ranges are fetched. Exit 1: prefix malformed, no variant has the model's structure, blob missing from the header.

- `PREFIX`, `--model-dir DIR` (required), `--arch sm_XX`, `--sku S`, `--numerics TIER`, `--act A`, `--kv K`

    goethe-engine pack need model.prefix --model-dir ./model --arch sm_120a

### `pack runs`

Checks only the signature and manifest (no blob, no weights, no model load) and whether the visible device runs one of the pack's SKUs.
Prints `{"runs": true|false, "sku": name or null, "device", "sms", "why": reason or null}`.
Exit 0 runs, 3 no SKU runs, 1 pack refused.

- `FILE`, `--sig F`, `--overlay F`, `--store DIR`, `--arch sm_XX`, `--sku S`

    goethe-engine pack runs model.gcpack; echo $?


### `pack plan`

Reads a PREFIX file (the first bytes of a pack: 16 bytes, header, manifest) and prints `{"manifest_sha256", "variants": [{"id", "role",
"quant", "source", "files": [{"name", "bytes"}], "max_ctx", "runs", "why"}]}`: per variant the model files a download fetches and from where
(`source` {hf_repo, revision}), its longest context, and whether the visible device runs one of its SKUs (`runs` null without a visible
device). No weights are needed; compare `manifest_sha256` with the index entry's. From pack ABI minor 2 every variant carries `desc.quant` and
`source`. Exit 1: prefix malformed, `--arch` differs.

- `PREFIX`, `--arch sm_XX`

    goethe-engine pack plan model.prefix --arch sm_120a

### `pack build`

Builds a pack from a pack SPEC (JSON, data only; the format is in `goethe_engine/build.py`): SKUs with their built library and
`requires`, variants with `quant`, `source` and their model files, plan blobs, bucket templates (expanded per SKU), speculative pairs and
measured bench rows. A template is named `contract.suffix` (its bucket id is `role.sku.contract.suffix`); `spec` and `measured` name
templates that way, and a name used twice is refused. The builder reads what a build can read: structure from the model files, weight
sizes and sha256, `cuda` from `nvcc`; `step_us` and the hw record hash only from `measured`. It writes through `pack.write` and prints `{"manifest_sha256", "out", "variants", "buckets"}`.
Exit 1: a library not built, a manifest `pack.write` refuses, no `nvcc`.

- `SPEC`, `--out F` (required), `--key PEM` (sign), `--store DIR` (pack 4)

    goethe-engine pack build pack.json --out model.gcpack --key pack_key.pem

### `pack sign`

Writes `FILE.sig`: the signature over the pack's manifest (its magic fixes the domain), with `--key`; the pack itself is unchanged, so a
pack built and signed with one key is re-signed with another without a rebuild. Prints `{"sig", "keyid"}`.

- `FILE`, `--key PEM` (required)

    goethe-engine pack sign model.gcpack --key pack_key.pem
### Numerics tiers

`--numerics` on `serve`, `pack verify` and `pack need` (default `invariant`). A tier picks the buckets whose measured contract matches.

| Tier | Contract |
|---|---|
| `invariant` | batch-invariant: a request's output does not depend on the batch it runs in |
| `invariant-hp` | batch-invariant with 16-bit activations |
| `reference` | bitwise llama.cpp validation tier, one stream, only on request |
| `tolerance` | KL-bounded: tolerance buckets if the pack has them, else `invariant` |
| `fast` | same as `tolerance` |
| `exact`, `deterministic` | deprecated names of `reference`, `invariant`; warn on stderr |

## Docs

- `goethe_engine/spec/`: the pack manifest schema, request parameters, error codes and the worker protocol
- `abi/gc.h`: the C interface a pack's runtime implements
- `goethe_engine.pack.write`: the one way to make a pack: it validates the manifest, takes exactly the blobs its buckets name, pins their sizes and frames pack 3 or 4 (sign with `pack.sign_pack`)
- `SPEC.md`: how speculative decoding works

Tests: `pip install -e '.[test]' && python -m pytest -q` (CPU only; needs a C compiler and openssl).

## License

Apache-2.0.
