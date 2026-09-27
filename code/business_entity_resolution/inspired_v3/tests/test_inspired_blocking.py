from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er_core import (
    FEATURE_NAMES, PAIR_FEATURE_COLUMNS, blocking_keys, normalize_name,
    normalize_text, romanize_text, training_split_ranges,
)


class InspiredBlockingTests(unittest.TestCase):
    def test_full_training_split_uses_80_10_10_buckets(self):
        ranges = training_split_ranges(smoke_test=False)
        self.assertEqual(ranges, ((0, 80), (80, 90), (90, 100)))
        for bucket in range(100):
            owners = [start <= bucket < stop for start, stop in ranges]
            self.assertEqual(sum(owners), 1, f"bucket {bucket} must be assigned once")

    def test_smoke_split_remains_small_and_disjoint(self):
        ranges = training_split_ranges(smoke_test=True)
        self.assertEqual(ranges, ((0, 1), (1, 2), (2, 3)))

    def test_composite_blocks_cover_reordering_and_address_evidence(self):
        left = blocking_keys("Northwind Trading LLC", "15 Emerald Avenue Boston MA 02110")
        reordered = blocking_keys("Trading Northwind", "15 Emerald Ave Boston MA 02110")
        self.assertIn(("compact_name", "northwindtrading"), left)
        self.assertIn(("name_pair", "northwind|trading"), left)
        self.assertIn(("name_pair", "northwind|trading"), reordered)
        self.assertIn(("name_house", "northwind|15"), left)
        self.assertIn(("house_address", "15|emerald"), left)
        self.assertIn(("postal_house", "02110|15"), left)

    def test_candidate_feature_schema_has_no_duplicates(self):
        self.assertEqual(len(PAIR_FEATURE_COLUMNS), len(set(PAIR_FEATURE_COLUMNS)))
        for feature in ("composite_key_hits", "name_composite_hits",
                        "address_composite_hits", "house_number_equal",
                        "candidate_rank_reciprocal"):
            self.assertIn(feature, FEATURE_NAMES)

    def test_native_script_is_preserved_for_blocking(self):
        self.assertEqual(normalize_text("भारत"), "भारत")
        native_name = normalize_name("भारत ट्रेडर्स")
        self.assertIn(("exact_name", native_name), blocking_keys("भारत ट्रेडर्स", "दिल्ली"))

    def test_transliteration_view_adds_parallel_keys_when_available(self):
        try:
            import anyascii  # noqa: F401
        except ImportError:
            self.skipTest("anyascii is installed by the AWS bootstrap, not this local test env")
        source_name = "भारत ट्रेडर्स"
        roman_name = normalize_name(romanize_text(source_name))
        native_name = normalize_name(source_name)
        self.assertNotEqual(roman_name, native_name)
        self.assertIn(("roman_exact_name", roman_name), blocking_keys(source_name, "दिल्ली"))

    def test_key_fanout_is_bounded(self):
        keys = blocking_keys(
            "Distinctive International Wholesale Distribution Logistics Company 123",
            "12345 Extremely Long Industrial Boulevard Building 987 Suite 543 Unit 12",
        )
        self.assertLessEqual(len(keys), 64)


if __name__ == "__main__":
    unittest.main()
