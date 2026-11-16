"""spec/*.json are valid JSON Schema, and every manifest the test fixtures build validates against manifest.schema.json."""
import glob
import json
import os

import jsonschema
import pytest

from goethe_engine import pack as P, params
from test_pack4 import v4
from toypack import toy_pack

SPEC = params.SPEC
SCHEMAS = sorted(glob.glob(os.path.join(SPEC, "**", "*.schema.json"), recursive=True))
MANIFEST = json.load(open(os.path.join(SPEC, "manifest.schema.json")))


@pytest.mark.parametrize("f", SCHEMAS, ids=lambda f: os.path.relpath(f, SPEC))
def test_schema_file_is_a_valid_json_schema(f):
    jsonschema.Draft202012Validator.check_schema(json.load(open(f)))


def test_spec_json_files_parse():
    for f in glob.glob(os.path.join(SPEC, "**", "*.json"), recursive=True):
        json.load(open(f))


def manifest_of(pack):
    with open(pack, "rb") as f:
        return json.loads(P.read_container_v(f, os.path.getsize(pack))[3])


def fixtures(tmp_path, key, lib):
    yield "decoder", toy_pack(str(tmp_path / "a"), key, lib)[0]
    yield "spec pair", toy_pack(str(tmp_path / "b"), key, lib, spec=True)[0]
    yield "two variants", toy_pack(str(tmp_path / "c"), key, lib, alt=True)[0]
    yield "pack 4", v4(str(tmp_path / "d"), key, lib)[0]


def test_every_fixture_manifest_validates_against_the_schema_file(tmp_path, key, mocklib):
    val = jsonschema.Draft202012Validator(MANIFEST)
    for name, pack in fixtures(tmp_path, key, mocklib):
        m = manifest_of(pack)
        assert [e.message for e in val.iter_errors(m)] == [], name
        assert params.errors(MANIFEST, m, "m") == [], name


def test_the_schema_rejects_a_broken_manifest(tmp_path, key, mocklib):
    m = manifest_of(toy_pack(str(tmp_path), key, mocklib)[0])
    val = jsonschema.Draft202012Validator(MANIFEST)
    for k in MANIFEST["required"]:
        assert list(val.iter_errors({x: v for x, v in m.items() if x != k})), k
