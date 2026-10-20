"""Deterministic seed derivation.

Every random stream of the benchmark has a seed that is derived from a list of
keys (for example the target fingerprint, the training-set size and the stream
name). The function is a keyed hash, so two different key lists give
independent seeds and the same key list always gives the same seed.
"""
from __future__ import annotations

import hashlib
import struct

_MAX_SEED = 2**63 - 1


def derive(*parts: object) -> int:
    """Return a non-negative seed below 2^63 - 1 from ordered keys.

    Each key is hashed with its type name and its ``repr``, both with their
    byte length, so ``1`` and ``"1"`` give different seeds and no two key lists
    collide through a delimiter. Use only ``int``, ``str``, ``float`` and tuples
    of these as keys.
    """
    hasher = hashlib.blake2b(digest_size=8, person=b"arityseed")
    for part in parts:
        if type(part) not in (int, str, float, tuple):          # the text of another type can change with a library
            raise TypeError(f"a seed key must be an int, a str, a float or a tuple; received {type(part).__qualname__}")
        type_name = type(part).__qualname__.encode("utf-8")
        value = repr(part).encode("utf-8")
        hasher.update(struct.pack(">I", len(type_name)))
        hasher.update(type_name)
        hasher.update(struct.pack(">Q", len(value)))
        hasher.update(value)
    return int.from_bytes(hasher.digest(), "big") % _MAX_SEED
