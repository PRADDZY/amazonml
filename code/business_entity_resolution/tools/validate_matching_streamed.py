"""Run organizer format checks in batches and exact ID checks in disk partitions."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import runpy
import tempfile
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path


PARTITIONS = 128
BATCH_SIZE = 5000
HEADER = "source1_entity_id\tmatched_entity_ids\n"


def partition(entity_id: str) -> int:
    return zlib.crc32(entity_id.encode("utf-8")) % PARTITIONS


@contextlib.contextmanager
def writers(root: Path, prefix: str):
    with contextlib.ExitStack() as stack:
        streams = [
            stack.enter_context(
                (root / f"{prefix}-{i:03}.txt").open("w", encoding="utf-8", newline="\n")
            )
            for i in range(PARTITIONS)
        ]
        yield streams


def validate(matching: Path, test_dir: Path, validator: Path, report_dir: Path) -> dict:
    started = time.monotonic()
    official = runpy.run_path(str(validator))
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = pairs = empties = 0
    target_counts = {}
    temporary = tempfile.TemporaryDirectory(prefix="validation_", dir=report_dir)
    scratch = Path(temporary.name).resolve()
    # TemporaryDirectory recursively cleans only this newly created, owned directory.
    if scratch.parent != report_dir.resolve():
        raise RuntimeError("Validation scratch directory escaped its intended parent")

    with temporary, (report_dir / "validation.log").open("w", encoding="utf-8") as log:
        batch_file = scratch / "batch.tsv"
        with (
            matching.open(encoding="utf-8", newline="") as predictions,
            (test_dir / "test_source1.tsv").open(encoding="utf-8") as source,
            writers(scratch, "queries") as query_files,
            writers(scratch, "requested") as requested_files,
        ):
            if predictions.readline().rstrip("\r\n") != HEADER.rstrip("\n"):
                raise ValueError("Matching header does not have the exact required columns")
            if source.readline().split("\t", 1)[0] != "entity_id":
                raise ValueError("Unexpected test Source 1 header")
            while lines := list(itertools.islice(predictions, BATCH_SIZE)):
                required = set()
                for line in lines:
                    expected = source.readline()
                    if not expected:
                        raise ValueError("Prediction file has more rows than test Source 1")
                    expected_id = expected.split("\t", 1)[0]
                    actual_id = line.split("\t", 1)[0]
                    if actual_id != expected_id:
                        raise ValueError(
                            f"Source 1 order/coverage mismatch: {actual_id!r} vs {expected_id!r}"
                        )
                    required.add(expected_id)
                    query_files[partition(actual_id)].write(actual_id + "\n")
                with batch_file.open("w", encoding="utf-8", newline="") as batch:
                    batch.write(HEADER)
                    batch.writelines(lines)
                errors = []
                with contextlib.redirect_stdout(log):
                    mapping = official["validate_id_list_file"](
                        str(batch_file), official["MATCHING_HEADER"],
                        "matched_entity_ids", required, None, errors,
                    )
                if errors or mapping is None:
                    raise ValueError("Organizer format checks failed: " + "; ".join(errors))
                for matched_ids in mapping.values():
                    empties += not matched_ids
                    pairs += len(matched_ids)
                    for target_id in matched_ids:
                        requested_files[partition(target_id)].write(target_id + "\n")
                rows += len(lines)
                del mapping
                if rows % 100000 == 0:
                    print(f"Organizer format checks: {rows:,} Source 1 rows", flush=True)
            if source.readline():
                raise ValueError("Prediction file is missing trailing Source 1 rows")

        print(f"Format and Source 1 alignment checked: {rows:,} rows", flush=True)
        with writers(scratch, "targets") as target_files:
            for filename in ("test_source2.tsv", "test_source3.tsv"):
                count = 0
                with (test_dir / filename).open(encoding="utf-8") as source:
                    if source.readline().split("\t", 1)[0] != "entity_id":
                        raise ValueError(f"Unexpected header in {filename}")
                    for line in source:
                        if not line.strip():
                            continue
                        entity_id = line.split("\t", 1)[0].strip()
                        target_files[partition(entity_id)].write(entity_id + "\n")
                        count += 1
                        if count % 1000000 == 0:
                            print(f"Indexing {filename}: {count:,} IDs", flush=True)
                target_counts[filename] = count

        checked_pairs = unique_queries = 0
        for i in range(PARTITIONS):
            with (scratch / f"targets-{i:03}.txt").open(encoding="utf-8") as target_file:
                valid_ids = {line.rstrip("\n") for line in target_file}
            with (scratch / f"requested-{i:03}.txt").open(encoding="utf-8") as requested:
                for line in requested:
                    entity_id = line.rstrip("\n")
                    if entity_id not in valid_ids:
                        raise ValueError(f"Matched ID is absent from test Source 2/3: {entity_id}")
                    checked_pairs += 1
            del valid_ids
            seen = set()
            with (scratch / f"queries-{i:03}.txt").open(encoding="utf-8") as queries:
                for line in queries:
                    entity_id = line.rstrip("\n")
                    if entity_id in seen:
                        raise ValueError(f"Duplicate Source 1 row across batches: {entity_id}")
                    seen.add(entity_id)
            unique_queries += len(seen)
            del seen
            if (i + 1) % 32 == 0:
                print(f"Exact target-ID and duplicate checks: {i + 1}/{PARTITIONS} partitions", flush=True)

        if checked_pairs != pairs or unique_queries != rows:
            raise ValueError("Partition accounting does not match the validated TSV")
        result = {
            "status": "PASS",
            "matching_file": str(matching.resolve()),
            "source1_rows": rows,
            "unique_source1_rows": unique_queries,
            "matched_pairs": pairs,
            "empty_rows": empties,
            "target_source_counts": target_counts,
            "organizer_format_checks": "PASS (unmodified function, batches of 5000 rows)",
            "complete_source1_alignment": True,
            "global_source1_duplicate_check": "PASS",
            "all_matched_ids_exist_in_test_sources": True,
            "candidate_subset_checked": False,
            "score_computed": False,
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        log.write(json.dumps(result, indent=2) + "\n")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    args = parser.parse_args()
    report_dir = args.matching.resolve().parent
    try:
        result = validate(args.matching, args.test_dir, args.validator, report_dir)
    except Exception as exc:
        result = {"status": "FAIL", "error": str(exc)}
    (report_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
