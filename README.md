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
Run `goethe-engine` with no arguments to list the other commands: checking a pack or a pack index, and the worker.

## Safe by default

goethe-engine works offline and fails closed. It checks the pack's signature, every piece of the pack it reads, the
ABI and the GPU before it loads anything, and refuses the pack if any check fails.

## Docs

- `goethe_engine/spec/`: the pack manifest schema, request parameters, error codes and the worker protocol
- `abi/gc.h`: the C interface a pack's runtime implements
- `SPEC.md`: how speculative decoding works

Tests: `pip install pytest && python -m pytest -q` (CPU only; needs a C compiler and openssl).

## License

Apache-2.0.
