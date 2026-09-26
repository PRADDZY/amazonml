import tempfile
import unittest
from pathlib import Path

from src.materialize_output import merge_tsv_parts


class MaterializeOutputTests(unittest.TestCase):
    def test_merges_spark_parts_with_one_header_and_preserves_empty_lists(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parts = root / "parts"
            parts.mkdir()
            (parts / "part-00000.csv").write_text(
                "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1\n", encoding="utf-8"
            )
            (parts / "part-00001.csv").write_text(
                "source1_entity_id\tcandidate_entity_ids\nS1-2\t\n", encoding="utf-8"
            )
            (parts / "_SUCCESS").write_text("", encoding="utf-8")
            output = root / "candidate_pairs.tsv"
            merge_tsv_parts(parts, output, ["source1_entity_id", "candidate_entity_ids"])
            self.assertEqual(output.read_text(encoding="utf-8"),
                             "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1\nS1-2\t\n")

    def test_rejects_missing_or_inconsistent_headers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parts = root / "parts"
            parts.mkdir()
            (parts / "part-00000.csv").write_text("wrong\theader\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                merge_tsv_parts(parts, root / "result.tsv", ["expected", "columns"])


if __name__ == "__main__":
    unittest.main()
