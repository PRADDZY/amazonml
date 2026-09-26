import unittest

from src.er_core import (
    PAIR_FEATURE_COLUMNS,
    blocking_keys,
    macro_f0_5,
    normalize_address,
    normalize_name,
    select_operating_point,
    select_top_candidates,
    spark_s3_uri,
)


class NormalizationTests(unittest.TestCase):
    def test_spark_paths_use_s3a_outside_emr(self):
        self.assertEqual(spark_s3_uri("s3://bucket/prefix"), "s3a://bucket/prefix")
        self.assertEqual(spark_s3_uri("s3a://bucket/prefix"), "s3a://bucket/prefix")
        self.assertEqual(spark_s3_uri("/tmp/input.tsv"), "/tmp/input.tsv")

    def test_glue_spark_paths_keep_the_native_s3_scheme(self):
        self.assertEqual(spark_s3_uri("s3://bucket/prefix", scheme="s3"), "s3://bucket/prefix")

    def test_pair_feature_output_schema_has_no_duplicate_columns(self):
        self.assertEqual(len(PAIR_FEATURE_COLUMNS), len(set(PAIR_FEATURE_COLUMNS)))
        self.assertEqual(PAIR_FEATURE_COLUMNS.count("retrieval_score"), 1)

    def test_normalizes_accents_punctuation_and_repeated_legal_suffixes(self):
        self.assertEqual(normalize_name("Café North, Inc. Limited"), "cafe north")

    def test_normalizes_address_abbreviations_and_accents(self):
        self.assertEqual(normalize_address("12 Élm Street, Suite 5"), "12 elm st ste 5")

    def test_blocking_keys_include_exact_and_rare_feature_keys(self):
        keys = blocking_keys("Café North Inc", "12 Elm Street, Boston MA 02110")
        self.assertIn(("exact_name", "cafe north"), keys)
        self.assertIn(("exact_address", "12 elm st boston ma 02110"), keys)
        self.assertIn(("postal_house", "02110|12"), keys)
        self.assertIn(("name_token", "cafe"), keys)
        self.assertIn(("name_gram", "caf"), keys)

    def test_blocking_keys_have_a_fixed_per_record_fanout_bound(self):
        name = "Distinctive International Wholesale Distribution Logistics Company 123"
        address = "12345 Extremely Long Industrial Boulevard Building 987 Suite 543 Unit 12"
        self.assertLessEqual(len(blocking_keys(name, address)), 18)


class CandidateTests(unittest.TestCase):
    def test_top_candidates_are_unique_and_capped_per_target_source(self):
        rows = [
            {"source1_id": "a", "target_id": "s2-1", "target_source": "S2", "rank": 0.9},
            {"source1_id": "a", "target_id": "s2-1", "target_source": "S2", "rank": 0.8},
            {"source1_id": "a", "target_id": "s2-2", "target_source": "S2", "rank": 0.7},
            {"source1_id": "a", "target_id": "s3-1", "target_source": "S3", "rank": 0.6},
        ]
        selected = select_top_candidates(rows, cap_per_source=1)
        self.assertEqual([(r["target_source"], r["target_id"]) for r in selected],
                         [("S2", "s2-1"), ("S3", "s3-1")])

    def test_macro_f0_5_gives_singleton_credit_only_for_empty_prediction(self):
        self.assertAlmostEqual(macro_f0_5([set(), {"x"}], [set(), set()]), 0.5)

    def test_operating_point_prefers_smallest_near_best_cap_with_recall_gate(self):
        metrics = [
            {"cap_per_source": 4, "candidate_recall": 0.95, "macro_f0_5": 0.91, "threshold": 0.5},
            {"cap_per_source": 8, "candidate_recall": 0.99, "macro_f0_5": 0.94, "threshold": 0.6},
            {"cap_per_source": 16, "candidate_recall": 1.0, "macro_f0_5": 0.941, "threshold": 0.7},
        ]
        self.assertEqual(select_operating_point(metrics), metrics[1])


if __name__ == "__main__":
    unittest.main()
