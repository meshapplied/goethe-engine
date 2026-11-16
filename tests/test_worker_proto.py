"""spec/proto/worker.desc is protoc's compile of worker.proto (no protobuf runtime needed)."""
import os
import subprocess

import pytest

SPEC = os.path.join(os.path.dirname(__file__), "..", "goethe_engine", "spec", "proto")


def test_descriptor_set_is_the_compiled_proto(tmp_path):
    proto = os.path.join(SPEC, "worker.proto")
    try:
        subprocess.run(["protoc", f"--proto_path={SPEC}", f"--descriptor_set_out={tmp_path / 'w.desc'}", proto], check=True)
    except FileNotFoundError:
        pytest.skip("no protoc")
    assert open(tmp_path / "w.desc", "rb").read() == open(os.path.join(SPEC, "worker.desc"), "rb").read()
