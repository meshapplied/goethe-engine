"""Request parameters of a task, one mechanism for the API, the HTTP front door and the CLI.

spec/params/<task>.schema.json defines every parameter the engine knows for generate | embed | transcribe (JSON Schema subset);
a pack declares in its bucket io.params the names it supports (absent: the schema's default_declared). check() refuses an unknown
or undeclared name (unsupported_parameter) and a value outside its schema (invalid_parameter), and returns the request filled
with nothing else: defaults stay the caller's."""
import json
import os

from .errors import InvalidParameter, UnsupportedParameter

SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec")
_CACHE = {}


def schema(task):
    if task not in _CACHE:
        with open(os.path.join(SPEC, "params", f"{task}.schema.json")) as f:
            _CACHE[task] = json.load(f)
    return _CACHE[task]


def declared(task, bucket=None):
    """The parameter names a bucket serves: io.params (list), else the task's default set."""
    d = ((bucket or {}).get("io") or {}).get("params")
    return list(d) if d is not None else list(schema(task)["default_declared"])


def errors(sch, v, path):
    """Violations of a JSON Schema subset (type, enum, minimum, exclusiveMinimum, maximum, minLength, maxItems, items, properties,
    required, additionalProperties false) as text."""
    out, t = [], sch.get("type")
    kinds = {"number": (int, float), "integer": (int,), "string": (str,), "boolean": (bool,), "array": (list,), "object": (dict,)}
    if t and (not isinstance(v, kinds[t]) or (t in ("number", "integer") and isinstance(v, bool))):
        return [f"{path}: {t} expected, got {type(v).__name__}"]
    if "enum" in sch and v not in sch["enum"]:
        out.append(f"{path}: one of {sch['enum']}")
    if "minimum" in sch and v < sch["minimum"]:
        out.append(f"{path}: >= {sch['minimum']}")
    if "exclusiveMinimum" in sch and v <= sch["exclusiveMinimum"]:
        out.append(f"{path}: > {sch['exclusiveMinimum']}")
    if "maximum" in sch and v > sch["maximum"]:
        out.append(f"{path}: <= {sch['maximum']}")
    if "minLength" in sch and len(v) < sch["minLength"]:
        out.append(f"{path}: at least {sch['minLength']} characters")
    if isinstance(v, list):
        if "maxItems" in sch and len(v) > sch["maxItems"]:
            out.append(f"{path}: at most {sch['maxItems']} items")
        for i, x in enumerate(v):
            out += errors(sch.get("items", {}), x, f"{path}[{i}]")
    if isinstance(v, dict):
        props = sch.get("properties", {})
        out += [f"{path}: {k} is required" for k in sch.get("required", []) if k not in v]
        if sch.get("additionalProperties") is False:
            out += [f"{path}: unknown field {k}" for k in v if k not in props]
        for k, x in v.items():
            if k in props:
                out += errors(props[k], x, f"{path}.{k}")
    return out


def check(task, req, bucket=None):
    """req (dict) validated against the task schema and the bucket's declared names; returns a copy without None values."""
    sch, have = schema(task), declared(task, bucket)
    req = {k: v for k, v in (req or {}).items() if v is not None}
    for k in req:
        if k not in sch["properties"]:
            raise UnsupportedParameter(f"{task} has no parameter {k!r} (parameters: {sorted(sch['properties'])})")
        if k not in have:
            raise UnsupportedParameter(f"{k!r}: this pack does not support it for {task} (io.params: {sorted(have)})")
    bad = errors({k: v for k, v in sch.items() if k != "default_declared"}, req, task)
    if bad:
        raise InvalidParameter("; ".join(bad))
    return req
