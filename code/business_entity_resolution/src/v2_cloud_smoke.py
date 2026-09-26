"""Small dependency and index integration check, executed on AWS before real data."""
import csv
import tempfile
from pathlib import Path

from v2_retrieval import Retriever, build_index
from v2_text import Record


def main():
    examples = [
        ("S1-1", "Acme Industrial LLC", "12 North Road, Boston MA", "US"),
        ("S1-2", "Acme Financial Inc", "75 South Avenue, Austin TX", "US"),
        ("S1-3", "राम मार्केटिंग प्राइवेट लिमिटेड", "570 New Delhi", "India"),
        ("S1-4", "Société Étoile SARL", "12 Rue de la Paix, Paris", "France"),
    ]
    columns = ("entity_id", "business_name", "business_address", "country")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.tsv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(columns)
            writer.writerows(examples)
        build_index(source, root / "index", threads=1)
        retriever = Retriever(root / "index")
        for target_id, name, address, country, expected in [
            ("S2-1", "Acme Industrials", "12 N Rd Boston MA", "US", 0),
            ("S2-2", "Ram Marketing", "570 New Delhi", "India", 2),
            ("S3-3", "Societe Etoile", "12 rue de la paix Paris", "France", 3),
        ]:
            record = Record.from_row(dict(zip(columns, (target_id, name, address, country))))
            hits = retriever.retrieve(record, 3)
            assert hits[expected]["joint_rank"] == 1, (record, hits)
        empty = Record("S2-4", "US", "", "", "", "")
        assert retriever.retrieve(empty) == {}
    import lightgbm, rapidfuzz, tantivy
    print("AWS integration smoke passed", lightgbm.__version__, rapidfuzz.__version__, tantivy.__version__, flush=True)


if __name__ == "__main__":
    main()
