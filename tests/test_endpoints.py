from __future__ import annotations

import unittest

from analysis.data_io import endpoint_counts, load_study_data


class EndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.study = load_study_data()

    def test_primary_endpoint_counts(self) -> None:
        counts = endpoint_counts(self.study.meta, "primary")
        self.assertEqual(counts["n_contexts"], 576)
        self.assertEqual(counts["n_positive"], 64)
        self.assertEqual(counts["n_negative"], 512)

    def test_expanded_endpoint_counts(self) -> None:
        counts = endpoint_counts(self.study.meta, "expanded")
        self.assertEqual(counts["n_contexts"], 1248)
        self.assertEqual(counts["n_positive"], 640)
        self.assertEqual(counts["n_negative"], 608)

    def test_original_alignment(self) -> None:
        self.assertEqual(len(self.study.meta), 1248)
        self.assertEqual(len(self.study.original_paper), 1248)


if __name__ == "__main__":
    unittest.main()
