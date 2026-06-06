"""LSH banding: band selection and candidate collisions."""

import unittest

from minhash_lsh.lsh import LSHIndex, choose_bands


class ChooseBandsTests(unittest.TestCase):
    def test_returns_a_divisor(self):
        for thr in (0.5, 0.7, 0.8, 0.9):
            b = choose_bands(128, thr)
            self.assertEqual(128 % b, 0)

    def test_higher_threshold_fewer_bands(self):
        # fewer bands -> higher LSH threshold (need more agreement to collide)
        self.assertLessEqual(choose_bands(128, 0.9), choose_bands(128, 0.5))


class LSHIndexTests(unittest.TestCase):
    def test_rejects_non_dividing_bands(self):
        with self.assertRaises(ValueError):
            LSHIndex(num_perm=128, bands=7)

    def test_identical_signatures_are_candidates(self):
        idx = LSHIndex(num_perm=8, bands=4)
        sig = [1, 2, 3, 4, 5, 6, 7, 8]
        idx.add("a", sig)
        idx.add("b", list(sig))
        self.assertIn("b", idx.candidates(sig))
        self.assertIn("a", idx.candidates(sig))

    def test_one_shared_band_is_enough(self):
        idx = LSHIndex(num_perm=8, bands=4)  # rows = 2
        idx.add("a", [1, 2, 9, 9, 9, 9, 9, 9])
        # shares the FIRST band (1,2), differs elsewhere -> still a candidate
        self.assertIn("a", idx.candidates([1, 2, 0, 0, 0, 0, 0, 0]))

    def test_no_shared_band_no_candidate(self):
        idx = LSHIndex(num_perm=8, bands=4)
        idx.add("a", [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(idx.candidates([8, 7, 6, 5, 4, 3, 2, 1]), set())

    def test_bad_signature_length(self):
        idx = LSHIndex(num_perm=8, bands=4)
        with self.assertRaises(ValueError):
            idx.add("a", [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
