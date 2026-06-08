# minhash_lsh

[![tests](https://github.com/MONISMALIK1/minhash_lsh/actions/workflows/test.yml/badge.svg)](https://github.com/MONISMALIK1/minhash_lsh/actions/workflows/test.yml)

**Near-duplicate detection from scratch** — MinHash for similarity estimation, LSH
banding for sublinear candidate search. Deduplicate a corpus, dataset, or document
collection **without comparing every pair**. Pure Python standard library, zero
dependencies.

> Broder (1997), *On the resemblance and containment of documents* (MinHash) ·
> Indyk & Motwani (1998), *Approximate nearest neighbors* (LSH). Banding follows
> Leskovec, Rajaraman & Ullman, *Mining of Massive Datasets*, ch. 3.

## The real problem

You have a pile of documents — a training corpus, a RAG index, scraped pages, user
submissions — and you need the near-duplicates: re-posts, light edits, boilerplate.
Comparing every pair is **O(n²)**: a million documents is ~500 billion comparisons.
And exact equality misses anything that differs by a word.

MinHash + LSH solves both: it estimates *fuzzy* similarity (Jaccard) and only ever
compares documents that are *likely* similar.

## How it works

```
document ──► shingles (k-grams)         a set per document
                │
                ▼
            MinHash signature           128 ints, regardless of doc size
                │   P(sig positions agree) = Jaccard similarity
                ▼
            LSH banding                 split the signature into b bands;
                │                       docs identical in ANY band = candidates
                ▼
            verify candidates           keep pairs whose estimated Jaccard ≥ threshold
```

- **MinHash** (`minhash.py`): for a random hash `h`, `P[min h(A) == min h(B)] =
  Jaccard(A, B)`. With 128 hashes the fraction of agreeing signature slots estimates
  Jaccard — in a fixed 128-int signature no matter how big the document.
- **LSH** (`lsh.py`): banding gives an S-curve `1 - (1 - sᶜ)ᵇ` with a tunable
  threshold near `(1/b)^(1/r)`, so only plausibly-similar docs ever get compared.

## Use

```bash
# JSONL corpus: one object per line; pick the text field and an id field
python -m minhash_lsh corpus.jsonl --field text --id-field id            # pairs
python -m minhash_lsh corpus.jsonl --clusters                            # groups
python -m minhash_lsh corpus.jsonl --unique > keep.txt                   # dedup -> keys to keep

# a directory of .txt files (key = filename), or a line-per-doc text file
python -m minhash_lsh ./docs/  --threshold 0.7
python -m minhash_lsh notes.txt --char --k 5

# tune: --threshold 0.8  --num-perm 128  --k 3  --char  --seed 1
```

As a library:

```python
from minhash_lsh import MinHashDeduper

d = MinHashDeduper(threshold=0.8)
for doc_id, text in corpus:
    d.add(doc_id, text)

d.duplicate_pairs()   # [DupPair(a, b, similarity), ...]  (verified ≥ threshold)
d.clusters()          # connected components of the near-dup graph
d.unique_keys()       # one representative per cluster — the deduplicated set

d.add_many(corpus)            # bulk-load (key, text) pairs
d.query("an incoming doc")    # [(key, similarity), ...] near-dupes of an UNSEEN doc
                              #   (screens against the index without inserting it)
```

It pairs naturally with the RAG repos
([self_rag](https://github.com/MONISMALIK1/self_rag),
[hyde](https://github.com/MONISMALIK1/hyde)): dedup a corpus *before* you index it.

## Design

| Module | Responsibility |
| --- | --- |
| `shingle.py` | document → set of hashed k-grams (word or char), via a stable hash |
| `minhash.py` | seeded universal-hash MinHash signatures + Jaccard estimate |
| `lsh.py` | banded LSH index (`add` / `candidates`) + threshold→bands selection |
| `dedup.py` | `MinHashDeduper`: pairs, clusters (union-find), unique set |

Everything is seeded, so results are **reproducible** run-to-run, and every step is
unit-tested offline.

## Test

```bash
make test        # or: python -m unittest discover -s minhash_lsh/tests -t . -v
```

28 tests: shingling, MinHash accuracy against **known Jaccard values**, LSH band
collisions and threshold selection, and end-to-end dedup (it flags a one-word edit
as a near-duplicate, leaves an unrelated doc alone, clusters, and reduces the set).

## Accuracy & limitations

- **Approximate by design.** MinHash estimates Jaccard with error ~`1/√num_perm`
  (≈9% at 128, ≈4% at 512). Raise `--num-perm` for tighter estimates at more cost.
- **LSH trades recall for speed.** Banding can miss a true pair near the threshold;
  more bands = higher recall but more candidates to verify. `--threshold` drives the
  automatic band choice.
- **Jaccard is lexical.** It measures token/character overlap, not meaning — great
  for edits/boilerplate, not for paraphrases that share few words.
- **Signature cost** is `O(shingles × num_perm)` per document (pure Python, no
  vectorization); fine for large corpora, not tuned for billions of docs.

## Part of a series

From-scratch, dependency-free implementations of foundational algorithms & papers:
**probabilistic data structures** —
**minhash_lsh** (this repo, near-duplicate detection) ·
[hyperloglog](https://github.com/MONISMALIK1/hyperloglog) (distinct counting); and
**LLM-reasoning / RAG** —
[self_rag](https://github.com/MONISMALIK1/self_rag) ·
[corrective_rag](https://github.com/MONISMALIK1/corrective_rag) ·
[hyde](https://github.com/MONISMALIK1/hyde) ·
[chain_of_verification](https://github.com/MONISMALIK1/chain_of_verification).

## License

MIT
