import tempfile
import unittest
from pathlib import Path

from aws.package_submission import validate_candidate_file


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
