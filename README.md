# goethe-engine

Token-level LLM serving core in Python. Continuous batching, paged KV blocks with a prefix cache, seeded sampling,
speculative decoding (see SPEC.md). Device work sits behind a small `Backend` protocol; a pure torch reference model
ships for tests.

Kernel packs (one signed `.gcpack` file per model and GPU architecture, pack ABI 3) run through the C ABI in `abi/gc.h`.
Loading is offline and fails closed: pinned signing key over the manifest, blob hashes, ABI, architecture, runtime. A pack names the
structure of the models it was compiled for (tensor table and metadata keys, never weight values), so it serves any weights of that
structure, finetunes included. Only the blobs of the branch the weights choose are read, each hash-checked, so a partly downloaded file
is enough. A blob may be stored xz compressed; its raw size is pinned in the signed manifest (`blob_raw`: {sha256: bytes}) and the decode is bounded by it (an xz blob without a pin is refused).

## Install

    pip install goethe_engine-<version>-py3-none-any.whl          # from a GitHub release; torch is a dependency
    pip install 'goethe_engine-<version>-py3-none-any.whl[serve]' # adds transformers for tokenizers and chat templates

## CLI

    goethe-engine serve --pack FILE.gcpack --weights W [DRAFTER_W] [--tokenizer DIR] [--host H] [--port P]
                        [--context-length N] [--max-running-requests N] [--served-model-name S] [--api-key K]
                        [--numerics exact|deterministic|fast] [--act auto|a8|a16] [--kv auto|f16|fp8]
    goethe-engine serve --ref                               # CPU reference model, to try the API
    goethe-engine index verify --index F --sig F            # signed pack index: serial, expiry, entries
    goethe-engine pack match --index F --sig F --model-dir DIR --arch sm_XX   # entry for the model files in DIR, exit 3 if none
    goethe-engine pack need PREFIX --model-dir DIR [--sku S --numerics ..]    # byte ranges of the blobs the model's branch needs
    goethe-engine pack verify FILE.gcpack [--sig F] [--partial --model-dir DIR]   # signature defaults to FILE.gcpack.sig
    goethe-engine pack show FILE.gcpack

`serve` speaks the OpenAI API: `/v1/chat/completions` and `/v1/completions` (JSON or SSE), `/v1/models`, `/health`. It
listens on 127.0.0.1 by default.

`GOETHE_ENGINE_ALLOW_UNSIGNED=1` turns off signature checks. Development only; it warns on every use.

## Tests

    pip install pytest && python -m pytest -q

CPU only; needs a C compiler and openssl.

## License

Apache-2.0.
