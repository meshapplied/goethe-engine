"""Request parameters of a task, one mechanism for the API, the HTTP front door and the CLI.

spec/params/<task>.schema.json defines every parameter the engine knows for generate | embed | transcribe (JSON Schema subset);
a pack declares in its bucket io.params the names it supports (absent: the schema's default_declared). check() refuses an unknown
or undeclared name (unsupported_parameter) and a value outside its schema (invalid_parameter), and returns the request filled
with nothing else: defaults stay the caller's."""
import json
import os
import re

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
    """Violations of a JSON Schema subset (type, enum, minimum, exclusiveMinimum, maximum, minLength, maxItems, pattern, anyOf, items,
    properties, required, additionalProperties false or a schema) as text."""
    out, t = [], sch.get("type")
    kinds = {"number": (int, float), "integer": (int,), "string": (str,), "boolean": (bool,), "array": (list,), "object": (dict,),
             "null": (type(None),)}
    ts = [t] if isinstance(t, str) else t or []
    if ts and not any(isinstance(v, kinds[x]) and not (x in ("number", "integer") and isinstance(v, bool)) for x in ts):
        return [f"{path}: {' or '.join(ts)} expected, got {type(v).__name__}"]
    if v is None:
        return out
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
    if "pattern" in sch and isinstance(v, str) and not re.fullmatch(sch["pattern"], v):
        out.append(f"{path}: must match {sch['pattern']}")
    if "anyOf" in sch and all(errors(x, v, path) for x in sch["anyOf"]):
        out.append(f"{path}: matches none of {len(sch['anyOf'])} forms")
    if isinstance(v, dict):
        props, extra = sch.get("properties", {}), sch.get("additionalProperties", True)
        out += [f"{path}: {k} is required" for k in sch.get("required", []) if k not in v]
        for k, x in v.items():
            if k in props:
                out += errors(props[k], x, f"{path}.{k}")
            elif extra is False:
                out.append(f"{path}: unknown field {k}")
            elif isinstance(extra, dict):
                out += errors(extra, x, f"{path}.{k}")
    return out


def check(task, req, bucket=None):
    """req (dict) validated against the task schema and the bucket's declared names; returns a copy without None values."""
    sch, have = schema(task), declared(task, bucket)
    req = {k: v for k, v in (req or {}).items() if v is not None}
    for k in req:
        if k not in sch["properties"]:
            raise UnsupportedParameter(f"{task} has no parameter {k!r} (parameters: {sorted(sch['properties'])})", k)
        if k not in have:
            raise UnsupportedParameter(f"{k!r}: this pack does not support it for {task} (io.params: {sorted(have)})", k)
    for k, v in req.items():
        bad = errors(sch["properties"][k], v, k)
        if bad:
            raise InvalidParameter("; ".join(bad), k)
    return req
