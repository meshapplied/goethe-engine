"""Bucket field "needs": host libraries the bucket's program opens at bind (["cublas"]); the runtime opens the pack without a bucket
whose needs are not met, so the loader only validates the field."""
import json
import os

from goethe_engine import pack as P, params
from toypack import toy_pack

SCHEMA = json.load(open(os.path.join(os.path.dirname(P.__file__), "spec", "manifest.schema.json")))


def manifest(tmp_path, key, mocklib, needs):
    def mut(m):
        m["buckets"][0]["needs"] = needs
    pack, _, _ = toy_pack(str(tmp_path), key, mocklib, mutate=mut)
    with open(pack, "rb") as f:
        return json.loads(P.read_container(f, os.path.getsize(pack))[2])


def test_needs_cublas_validates(tmp_path, key, mocklib):
    m = manifest(tmp_path, key, mocklib, ["cublas"])
    P.validate(m)
    assert params.errors(SCHEMA, m, "m") == []


def test_needs_unknown_or_repeated_is_a_schema_error(tmp_path, key, mocklib):
    assert params.errors(SCHEMA, manifest(tmp_path / "a", key, mocklib, ["cudnn"]), "m") == ["m.buckets[0].needs[0]: one of ['cublas']"]
    assert params.errors(SCHEMA, manifest(tmp_path / "b", key, mocklib, ["cublas", "cublas"]), "m") == ["m.buckets[0].needs: items must be unique"]
