"""minhash_lsh — near-duplicate detection from scratch, zero dependencies.

MinHash estimates Jaccard similarity from small fixed-size signatures (Broder, 1997);
LSH banding turns that into sublinear candidate search (Indyk & Motwani, 1998), so
you can deduplicate large corpora without comparing every pair.

    from minhash_lsh import MinHashDeduper

    d = MinHashDeduper(threshold=0.8)
    d.add("doc1", "the quick brown fox jumps over the lazy dog")
    d.add("doc2", "the quick brown fox jumps over the lazy cat")
    d.duplicate_pairs()      # -> [DupPair(a='doc1', b='doc2', similarity=...)]
"""

from __future__ import annotations

__version__ = "0.1.0"

from .dedup import DupPair, MinHashDeduper
from .lsh import LSHIndex, choose_bands
from .minhash import MinHasher, jaccard
from .shingle import shingles, tokenize

__all__ = [
    "__version__",
    "MinHashDeduper", "DupPair",
    "MinHasher", "jaccard",
    "LSHIndex", "choose_bands",
    "shingles", "tokenize",
]
