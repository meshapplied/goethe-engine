#!/bin/sh
# A CPU-only minimal host: the package installed with its one dependency (numpy), no torch, no transformers. The pack and index tooling
# must import and run: --help, --version, and `pack verify` on a toy pack (refused when its key is not pinned, ok when signatures are off).
set -eu
root=$(git rev-parse --show-toplevel)
work=$(mktemp -d)
if command -v uv >/dev/null; then
    uv venv -q "$work/venv"
    uv pip install -q --python "$work/venv/bin/python" "$root"
else
    python -m venv "$work/venv"
    "$work/venv/bin/pip" install -q "$root"
fi
py="$work/venv/bin/python"
ge="$work/venv/bin/goethe-engine"
! "$py" -c "import torch" 2>/dev/null || { echo "torch is installed: not a minimal host"; exit 1; }
"$py" -c "
import goethe_engine, goethe_engine.cli, goethe_engine.pack, goethe_engine.index, goethe_engine.sign, goethe_engine.structure
import sys; assert 'torch' not in sys.modules and 'transformers' not in sys.modules, 'tooling imported torch'"
"$ge" --version
"$ge" --help >/dev/null
for c in "index verify" "pack match" "pack verify" "pack runs"; do "$ge" $c --help >/dev/null; done
cc -std=c99 -O1 -shared -fPIC -o "$work/libgck.so" "$root/tests/mock/gck_mock.c"
WORK="$work" PYTHONPATH="$root/tests" "$py" -c "
import os
from toypack import Key, toy_pack
w = os.environ['WORK']
toy_pack(w + '/pack', Key(w), w + '/libgck.so')"
pack="$work/pack/toy.gcpack"
set +e
"$ge" pack verify "$pack" 2>"$work/err"; rc=$?
set -e
[ "$rc" = 1 ] && grep -q REFUSED "$work/err" && ! grep -q Traceback "$work/err" || { echo "unpinned key not refused cleanly (exit $rc)"; cat "$work/err"; exit 1; }
GOETHE_ENGINE_ALLOW_UNSIGNED=1 "$ge" pack verify "$pack" 2>/dev/null | grep -q '"ok": true'
GOETHE_ENGINE_ALLOW_UNSIGNED=1 "$ge" pack show "$pack" >/dev/null 2>&1
echo "import-clean: ok"
rm -rf "$work"
