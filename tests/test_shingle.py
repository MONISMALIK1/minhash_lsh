"""Shingling into hashed k-grams."""

import unittest

from minhash_lsh.shingle import shingles, tokenize


class TokenizeTests(unittest.TestCase):
    def test_lowercase_words(self):
        self.assertEqual(tokenize("The Quick, brown FOX!"), ["the", "quick", "brown", "fox"])


class ShingleTests(unittest.TestCase):
    def test_word_shingle_count(self):
        # 5 tokens, k=3 -> 3 overlapping shingles
        s = shingles("a b c d e", k=3)
        self.assertEqual(len(s), 3)

    def test_short_doc_collapses_to_one(self):
        self.assertEqual(len(shingles("a b", k=3)), 1)

    def test_empty_is_empty(self):
        self.assertEqual(shingles("   ", k=3), set())

    def test_char_shingles(self):
        s = shingles("abcd", k=2, char=True)  # ab, bc, cd
        self.assertEqual(len(s), 3)

    def test_stable_hash_across_calls(self):
        self.assertEqual(shingles("the quick brown fox", k=2),
                         shingles("the quick brown fox", k=2))

    def test_similar_docs_overlap(self):
        a = shingles("the quick brown fox jumps", k=2)
        b = shingles("the quick brown fox leaps", k=2)
        self.assertTrue(a & b)  # they share shingles


if __name__ == "__main__":
    unittest.main()
