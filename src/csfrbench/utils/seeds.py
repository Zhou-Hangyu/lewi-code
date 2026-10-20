"""Deterministic, collision-resistant seed derivation."""
from __future__ import annotations

import hashlib
import struct

_MAX_SEED = 2**63 - 1


def derive(*parts: object) -> int:
    """Derive a deterministic nonnegative seed from ordered typed keys.

    Components are encoded with both their type and byte length, avoiding
    delimiter collisions and common cross-type ambiguities such as ``1`` and
    ``"1"``. Pass only stable scalar/tuple keys, not arbitrary mutable objects.
    """
    hasher = hashlib.blake2b(digest_size=8, person=b"arityseed")
    for part in parts:
        type_name = type(part).__qualname__.encode("utf-8")
        value = repr(part).encode("utf-8")
        hasher.update(struct.pack(">I", len(type_name)))
        hasher.update(type_name)
        hasher.update(struct.pack(">Q", len(value)))
        hasher.update(value)
    return int.from_bytes(hasher.digest(), "big") % _MAX_SEED
