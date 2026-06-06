"""Shingling — turn a document into a set of hashed k-grams.

A document's *shingle set* is the unit MinHash works on: two documents are
"similar" to the degree their shingle sets overlap (Jaccard similarity). Word-level
k-shingles (default) capture phrasing; character-level k-shingles are better for
short/typo-noisy text. Each shingle is hashed to a 32-bit int with a *stable* hash
(``zlib.crc32``) so results are reproducible across processes and runs — unlike the
builtin ``hash()`` on strings, which is salted.
"""

from __future__ import annotations

import re
import zlib

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _h(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def shingles(text: str, k: int = 3, char: bool = False) -> set[int]:
    """Return the set of hashed k-shingles for ``text``.

    Word shingles by default; ``char=True`` for character k-grams. A document with
    fewer than ``k`` units collapses to a single shingle; empty text yields ``set()``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if char:
        s = text.lower()
        if not s:
            return set()
        if len(s) <= k:
            return {_h(s)}
        return {_h(s[i:i + k]) for i in range(len(s) - k + 1)}

    toks = tokenize(text)
    if not toks:
        return set()
    if len(toks) < k:
        return {_h(" ".join(toks))}
    return {_h(" ".join(toks[i:i + k])) for i in range(len(toks) - k + 1)}


__all__ = ["tokenize", "shingles"]
