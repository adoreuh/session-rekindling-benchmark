"""Strict JSON, canonical encoding, and hashing.

Three separate concerns deliberately kept apart:

``check_json``       rejects anything not safely round-trippable
``json_copy``        validate then deep copy (no shared references)
``canonical_json``   byte-stable encoding, used for every hash in the system

The whole audit story depends on ``canonical_json`` being stable across runs
and platforms, so it is defined once, here, and never re-implemented.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .errors import ValidationError

SCHEMA_VERSION = 1


def check_json(value: Any, path: str = "value") -> None:
    """Reject non-JSON types and non-finite numbers before any copying."""
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{path}: non-finite number is not JSON")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{path}: dict keys must be strings")
            check_json(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            check_json(item, f"{path}[{index}]")
        return
    raise ValidationError(f"{path}: unsupported type {type(value).__name__}")


def json_copy(value: Any, what: str = "value") -> Any:
    """Validate as strict JSON, then return an isolated deep copy."""
    check_json(value, what)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def canonical_json(obj: Any) -> str:
    """Canonical encoding: sorted keys, compact separators, UTF-8 preserved."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(obj: Any) -> str:
    """Digest of any JSON value, via its canonical encoding."""
    return sha256_hex(canonical_json(obj))


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
