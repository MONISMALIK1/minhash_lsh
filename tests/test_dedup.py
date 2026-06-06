"""End-to-end near-duplicate detection on small, known documents."""

import unittest

from minhash_lsh import MinHashDeduper


class DedupTests(unittest.TestCase):
    def _deduper(self):
        d = MinHashDeduper(threshold=0.5, num_perm=128, k=3, seed=1)
        # a/b differ by one word -> near-duplicate; c is unrelated
        d.add("a", "the quick brown fox jumps over the lazy dog")
        d.add("b", "the quick brown fox jumps over the lazy cat")
        d.add("c", "relational databases use indexes to speed up queries")
        return d

    def test_finds_the_near_duplicate_pair(self):
        pairs = self._deduper().duplicate_pairs()
        found = {tuple(sorted((p.a, p.b))) for p in pairs}
        self.assertIn(("a", "b"), found)

    def test_does_not_flag_unrelated_doc(self):
        pairs = self._deduper().duplicate_pairs()
        keys_in_pairs = {p.a for p in pairs} | {p.b for p in pairs}
        self.assertNotIn("c", keys_in_pairs)

    def test_similarity_is_reported(self):
        pairs = self._deduper().duplicate_pairs()
        ab = next(p for p in pairs if {p.a, p.b} == {"a", "b"})
        self.assertGreaterEqual(ab.similarity, 0.5)

    def test_clusters_group_duplicates(self):
        clusters = self._deduper().clusters()
        big = [c for c in clusters if len(c) > 1]
        self.assertEqual(big, [["a", "b"]])

    def test_unique_keys_drops_one_of_the_pair(self):
        kept = self._deduper().unique_keys()
        self.assertEqual(len(kept), 2)        # a/b collapse to one; c stays
        self.assertIn("c", kept)

    def test_exact_duplicate_is_max_similarity(self):
        d = MinHashDeduper(threshold=0.8, num_perm=128, seed=1)
        d.add("x", "identical content here for testing purposes")
        d.add("y", "identical content here for testing purposes")
        pairs = d.duplicate_pairs()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].similarity, 1.0)

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            MinHashDeduper(threshold=0)


if __name__ == "__main__":
    unittest.main()
