"""MinHash — estimate Jaccard similarity from tiny fixed-size signatures.

Reference: Broder (1997), "On the resemblance and containment of documents."

The trick: for a random hash ``h``, the probability that two sets share the same
*minimum* hash value equals their Jaccard similarity:

    P[ min_x h(x in A) == min_x h(x in B) ] = |A ∩ B| / |A ∪ B|

So with ``num_perm`` independent hashes, the fraction of signature positions that
agree is an unbiased estimator of Jaccard — and the signature is the same small size
(``num_perm`` ints) no matter how large the document. We realize the "independent
hashes" with universal hashing ``h_i(x) = (a_i·x + b_i) mod p`` over a Mersenne
prime; the coefficients are seeded so signatures are fully reproducible.
"""

from __future__ import annotations

import random

_MERSENNE = (1 << 61) - 1  # prime, comfortably larger than 32-bit shingles


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        if num_perm < 1:
            raise ValueError("num_perm must be >= 1")
        self.num_perm = num_perm
        self.seed = seed
        rng = random.Random(seed)
        self._a = [rng.randrange(1, _MERSENNE) for _ in range(num_perm)]
        self._b = [rng.randrange(0, _MERSENNE) for _ in range(num_perm)]

    def signature(self, shingle_set) -> list[int]:
        """The MinHash signature (one min per permutation) of a shingle set."""
        p, n = _MERSENNE, self.num_perm
        sig = [p] * n
        if not shingle_set:
            return sig  # empty set -> sentinel signature (matches only other empties)
        a, b = self._a, self._b
        for x in shingle_set:
            for i in range(n):
                hv = (a[i] * x + b[i]) % p
                if hv < sig[i]:
                    sig[i] = hv
        return sig


def jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimated Jaccard similarity = fraction of agreeing signature positions."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must be the same length")
    if not sig_a:
        return 1.0
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)


__all__ = ["MinHasher", "jaccard"]
