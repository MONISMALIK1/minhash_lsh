"""MinHash signatures estimate Jaccard similarity (deterministic, seeded)."""

import unittest

from minhash_lsh.minhash import MinHasher, jaccard


class MinHashTests(unittest.TestCase):
    def setUp(self):
        self.mh = MinHasher(num_perm=256, seed=1)

    def test_identical_sets_estimate_one(self):
        s = self.mh.signature(set(range(100)))
        self.assertEqual(jaccard(s, s), 1.0)

    def test_disjoint_sets_estimate_near_zero(self):
        a = self.mh.signature(set(range(0, 100)))
        b = self.mh.signature(set(range(1000, 1100)))
        self.assertLess(jaccard(a, b), 0.1)

    def test_known_jaccard_is_approximated(self):
        # |A∩B| = 50, |A∪B| = 150 -> true Jaccard = 1/3
        a = self.mh.signature(set(range(0, 100)))
        b = self.mh.signature(set(range(50, 150)))
        est = jaccard(a, b)
        self.assertAlmostEqual(est, 1 / 3, delta=0.12)

    def test_empty_sets_match(self):
        a = self.mh.signature(set())
        b = self.mh.signature(set())
        self.assertEqual(jaccard(a, b), 1.0)

    def test_signature_length(self):
        self.assertEqual(len(self.mh.signature({1, 2, 3})), 256)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            jaccard([1, 2, 3], [1, 2])

    def test_seed_is_reproducible(self):
        other = MinHasher(num_perm=256, seed=1)
        self.assertEqual(self.mh.signature({1, 2, 3, 4}), other.signature({1, 2, 3, 4}))


if __name__ == "__main__":
    unittest.main()
