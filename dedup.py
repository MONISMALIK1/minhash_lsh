"""High-level near-duplicate detection: add documents, get pairs / clusters.

Wires shingling + MinHash + LSH into one object. LSH proposes candidate pairs in
roughly linear time; each candidate is then *verified* by its estimated Jaccard so
the reported pairs actually clear the threshold. Clustering is union-find over the
surviving pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lsh import LSHIndex, choose_bands
from .minhash import MinHasher, jaccard
from .shingle import shingles


@dataclass
class DupPair:
    a: object
    b: object
    similarity: float


def _key(x) -> str:
    return str(x)


class MinHashDeduper:
    """Streaming near-duplicate detector.

    >>> d = MinHashDeduper(threshold=0.7)
    >>> d.add("a", "the quick brown fox jumps over the lazy dog")
    >>> d.add("b", "the quick brown fox jumps over the lazy cat")
    >>> [(p.a, p.b) for p in d.duplicate_pairs()]
    [('a', 'b')]
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, k: int = 3,
                 char: bool = False, seed: int = 1, bands: int | None = None):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self.k = k
        self.char = char
        self.hasher = MinHasher(num_perm=num_perm, seed=seed)
        self.bands = bands or choose_bands(num_perm, threshold)
        self.index = LSHIndex(num_perm=num_perm, bands=self.bands)
        self._sigs: dict = {}

    def add(self, key, text: str) -> None:
        sig = self.hasher.signature(shingles(text, k=self.k, char=self.char))
        self._sigs[key] = sig
        self.index.add(key, sig)

    def add_many(self, items) -> "MinHashDeduper":
        """Add an iterable of ``(key, text)`` pairs in bulk. Returns self."""
        for key, text in items:
            self.add(key, text)
        return self

    def query(self, text: str) -> list[tuple]:
        """Near-duplicates of an *unseen* document among the indexed ones.

        Computes the query's signature, pulls LSH candidates, and returns the
        verified matches as ``(key, similarity)`` pairs — **without** adding the
        document to the index (handy for screening incoming docs against a corpus).
        """
        sig = self.hasher.signature(shingles(text, k=self.k, char=self.char))
        out = []
        for cand in self.index.candidates(sig):
            sim = jaccard(sig, self._sigs[cand])
            if sim >= self.threshold:
                out.append((cand, round(sim, 4)))
        out.sort(key=lambda t: (-t[1], _key(t[0])))
        return out

    def __len__(self) -> int:
        return len(self._sigs)

    def duplicate_pairs(self) -> list[DupPair]:
        """All verified near-duplicate pairs (estimated Jaccard >= threshold)."""
        seen: set = set()
        pairs: list[DupPair] = []
        for key, sig in self._sigs.items():
            for cand in self.index.candidates(sig):
                if cand == key:
                    continue
                pair = tuple(sorted((key, cand), key=_key))
                if pair in seen:
                    continue
                seen.add(pair)
                sim = jaccard(sig, self._sigs[cand])
                if sim >= self.threshold:
                    pairs.append(DupPair(pair[0], pair[1], round(sim, 4)))
        pairs.sort(key=lambda p: (-p.similarity, _key(p.a), _key(p.b)))
        return pairs

    def clusters(self) -> list[list]:
        """Connected components of the near-duplicate graph (incl. singletons)."""
        parent = {k: k for k in self._sigs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for p in self.duplicate_pairs():
            ra, rb = find(p.a), find(p.b)
            if ra != rb:
                parent[ra] = rb

        groups: dict = {}
        for k in self._sigs:
            groups.setdefault(find(k), []).append(k)
        return sorted((sorted(g, key=_key) for g in groups.values()), key=lambda g: _key(g[0]))

    def unique_keys(self) -> list:
        """One representative per cluster (the min key) — i.e. the deduplicated set."""
        return sorted((min(g, key=_key) for g in self.clusters()), key=_key)


__all__ = ["MinHashDeduper", "DupPair"]
