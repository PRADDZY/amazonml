import tempfile
import unittest
from pathlib import Path

from aws.package_submission import render_documentation, validate_candidate_file


class DocumentationRenderingTests(unittest.TestCase):
    def test_renders_runtime_without_requesting_cost_data(self):
        metrics = {
            "selected_cap_per_source": 1,
            "selected_candidate_recall": 1.0,
            "candidate_pairs": 2,
            "candidate_reduction_ratio": 0.9,
            "selected_threshold": 0.5,
            "validation_macro_f0_5": 0.8,
            "validation_pair_decisions": {
                "true_positives": 2,
                "false_positives": 0,
                "false_negatives": 0,
                "missed_by_blocking": 0,
            },
            "runtime_seconds": 5400,
            "candidate_count_mean": 1.0,
            "candidate_count_median": 1,
            "candidate_count_p95": 2,
            "candidate_count_p99": 2,
            "candidate_count_max": 2,
            "per_country": {"US": {"candidate_pairs": 2, "source1_rows": 2}},
        }

        rendered = render_documentation(metrics)

        self.assertIn("**AWS EC2 runtime:** 1.50 hours.", rendered)
        self.assertNotIn("cost", rendered.lower())


class CandidateFileValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.test_source1 = root / "test_source1.tsv"
        self.test_source1.write_text(
            "entity_id\tbusiness_name\nS1-1\tOne\nS1-2\tTwo\n", encoding="utf-8"
        )
        self.candidates = root / "candidate_pairs.tsv"

    def tearDown(self):
        self.temp.cleanup()

    def test_checks_source1_coverage_prefixes_and_per_source_cap(self):
        self.candidates.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-1\tS2-1,S3-1\n"
            "S1-2\t\n",
            encoding="utf-8",
        )
        self.assertEqual(validate_candidate_file(self.candidates, self.test_source1, 1), (2, 1))

    def test_rejects_missing_source1_rows(self):
        self.candidates.write_text(
            "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "missing 1 Source 1 rows"):
            validate_candidate_file(self.candidates, self.test_source1, 1)

    def test_rejects_duplicate_ids_and_cap_overflow(self):
        self.candidates.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-1\tS2-1,S2-1\n"
            "S1-2\t\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Repeated candidate ID"):
            validate_candidate_file(self.candidates, self.test_source1, 1)

        self.candidates.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-1\tS2-1,S2-2\n"
            "S1-2\t\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Candidate cap exceeded"):
            validate_candidate_file(self.candidates, self.test_source1, 1)


if __name__ == "__main__":
    unittest.main()
