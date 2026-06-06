"""CLI: find near-duplicates / deduplicate a corpus.

Usage:
    # JSONL: one object per line; --field picks the text, --id-field the key
    python -m minhash_lsh corpus.jsonl --field text --id-field id

    # a directory of .txt files (each file is one document; key = filename)
    python -m minhash_lsh ./docs/

    # a plain text file, one document per line
    python -m minhash_lsh notes.txt

Output modes: --pairs (default), --clusters, --unique (keys to keep).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .dedup import MinHashDeduper


def _load(path: str, field: str, id_field: str):
    """Yield (key, text) pairs from a dir of .txt, a .jsonl, or a line-per-doc file."""
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.glob("*.txt")):
            yield f.name, f.read_text(encoding="utf-8", errors="replace")
        return
    if not p.exists():
        raise SystemExit(f"path not found: {path}")

    if p.suffix == ".jsonl":
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{n}: invalid JSON ({exc})")
            if field not in obj:
                raise SystemExit(f"{path}:{n}: missing field '{field}'")
            key = obj.get(id_field, f"line{n}")
            yield key, str(obj[field])
        return

    # plain text: one document per non-empty line
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            yield f"line{n}", line


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="minhash_lsh",
        description="Near-duplicate detection with MinHash + LSH (Broder 1997; Indyk-Motwani 1998).",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("path", help="A .jsonl file, a directory of .txt files, or a line-per-doc file.")
    ap.add_argument("--field", default="text", help="JSONL field holding the text (default: text).")
    ap.add_argument("--id-field", default="id", help="JSONL field to use as the key (default: id).")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="Similarity threshold for a near-duplicate (default: 0.8).")
    ap.add_argument("--num-perm", type=int, default=128, help="MinHash permutations (default: 128).")
    ap.add_argument("--k", type=int, default=3, help="Shingle size (default: 3).")
    ap.add_argument("--char", action="store_true", help="Use character k-shingles instead of word.")
    ap.add_argument("--seed", type=int, default=1, help="Hash seed (reproducibility).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pairs", action="store_true", help="Print near-duplicate pairs (default).")
    g.add_argument("--clusters", action="store_true", help="Print near-duplicate clusters.")
    g.add_argument("--unique", action="store_true", help="Print the keys to keep (deduplicated).")
    args = ap.parse_args()

    deduper = MinHashDeduper(threshold=args.threshold, num_perm=args.num_perm,
                             k=args.k, char=args.char, seed=args.seed)
    for key, text in _load(args.path, args.field, args.id_field):
        deduper.add(key, text)

    print(f"loaded {len(deduper)} documents | threshold={args.threshold} "
          f"num_perm={args.num_perm} bands={deduper.bands} k={args.k}", file=sys.stderr)

    if args.clusters:
        groups = [g for g in deduper.clusters() if len(g) > 1]
        if not groups:
            print("(no near-duplicate clusters)")
        for i, g in enumerate(groups, 1):
            print(f"cluster {i} ({len(g)}): {', '.join(str(k) for k in g)}")
    elif args.unique:
        kept = deduper.unique_keys()
        print(f"# {len(kept)} of {len(deduper)} kept after dedup", file=sys.stderr)
        for k in kept:
            print(k)
    else:  # pairs
        pairs = deduper.duplicate_pairs()
        if not pairs:
            print("(no near-duplicate pairs)")
        for p in pairs:
            print(f"{p.similarity:.3f}\t{p.a}\t{p.b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
