"""Small dependency and index integration check, executed on AWS before real data."""
import csv
import tempfile
from pathlib import Path

import numpy
import pyarrow
import pyarrow.parquet as pq
from anyascii import anyascii
from v2_retrieval import Retriever, build_index
import v2_pipeline
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
        sharding = root / "sharding"
        sharding.mkdir()
        shard_columns = ("entity_id", "matched_entity_ids")
        source_ids = [f"S2-{index}" for index in range(30)] + [f"S3-{index}" for index in range(30)]
        for source in (2, 3):
            path = sharding / f"train_source{source}.tsv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(shard_columns)
                writer.writerows((entity_id, "") for entity_id in source_ids if entity_id.startswith(f"S{source}-"))
            test_path = sharding / f"test_source{source}.tsv"
            with test_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(shard_columns)
                writer.writerows((entity_id, "") for entity_id in source_ids if entity_id.startswith(f"S{source}-"))
        owners = {entity_id: index % 4 for index, entity_id in enumerate(source_ids[:12])}
        v2_pipeline._REFERENCE_FOLDS = numpy.asarray([0, 1, 2, 3], dtype=numpy.int8)
        training_shards = [
            list(v2_pipeline._training_tasks(sharding, owners, shard, 3)) for shard in range(3)
        ]
        training_ids = [task[1]["entity_id"] for shard in training_shards for task in shard]
        assert len(training_ids) == len(source_ids) and set(training_ids) == set(source_ids)
        test_shards = [list(v2_pipeline._test_tasks(sharding, shard, 3)) for shard in range(3)]
        test_ids = [task[1]["entity_id"] for shard in test_shards for task in shard]
        assert len(test_ids) == len(source_ids) and set(test_ids) == set(source_ids)
        parquet_shards = []
        for shard, values in enumerate(([1, 3], [0, 2])):
            path = root / f"targets-{shard}.parquet"
            pq.write_table(
                pyarrow.table({"seq": values, "owner": [value + 10 for value in values]}), path,
            )
            parquet_shards.append(path)
        merged = root / "targets-merged.parquet"
        v2_pipeline._merge_parquet_shards(parquet_shards, merged)
        targets = pq.read_table(merged)
        order = numpy.argsort(targets["seq"].to_numpy())
        assert targets["seq"].to_numpy()[order].tolist() == [0, 1, 2, 3]
        assert targets["owner"].to_numpy()[order].tolist() == [10, 11, 12, 13]

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
    assert anyascii("東京")
    print("AWS integration smoke passed", numpy.__version__, pyarrow.__version__,
          lightgbm.__version__, rapidfuzz.__version__, tantivy.__version__, flush=True)


if __name__ == "__main__":
    main()
