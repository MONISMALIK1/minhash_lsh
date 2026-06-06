"""Locality-Sensitive Hashing — find candidate near-duplicates without O(n²).

Reference: Indyk & Motwani (1998); the banding scheme follows Leskovec, Rajaraman &
Ullman, "Mining of Massive Datasets," ch. 3.

Split each ``num_perm``-length signature into ``b`` bands of ``r`` rows
(``b·r = num_perm``). Two documents are *candidates* if they are identical in **any**
whole band. The probability two docs with Jaccard ``s`` become candidates is::

    P(candidate) = 1 - (1 - s^r)^b

an S-curve whose steep threshold sits near ``(1/b)^(1/r)``. Tuning ``b`` (and hence
``r``) moves that threshold, trading recall against the number of false candidates.
"""

from __future__ import annotations

from collections import defaultdict


def choose_bands(num_perm: int, threshold: float) -> int:
    """Pick a divisor ``b`` of ``num_perm`` whose LSH threshold best matches ``threshold``."""
    best, best_err = 1, float("inf")
    for b in range(1, num_perm + 1):
        if num_perm % b:
            continue
        r = num_perm // b
        thr = (1.0 / b) ** (1.0 / r)
        err = abs(thr - threshold)
        if err < best_err:
            best, best_err = b, err
    return best


class LSHIndex:
    """Banded LSH over MinHash signatures. ``add`` then ``candidates``."""

    def __init__(self, num_perm: int = 128, bands: int = 16):
        if bands < 1 or num_perm % bands:
            raise ValueError("bands must be >= 1 and divide num_perm")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self._buckets: dict = defaultdict(set)  # (band_index, band_tuple) -> {keys}
        self.keys: set = set()

    def _bands_of(self, signature):
        r = self.rows
        for bi in range(self.bands):
            yield bi, tuple(signature[bi * r:(bi + 1) * r])

    def add(self, key, signature) -> None:
        if len(signature) != self.num_perm:
            raise ValueError("signature length does not match num_perm")
        self.keys.add(key)
        for bi, band in self._bands_of(signature):
            self._buckets[(bi, band)].add(key)

    def candidates(self, signature) -> set:
        """Keys that share at least one whole band with ``signature``."""
        out: set = set()
        for bi, band in self._bands_of(signature):
            out |= self._buckets.get((bi, band), set())
        return out


__all__ = ["LSHIndex", "choose_bands"]
