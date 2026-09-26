#!/usr/bin/env python3
"""Stream Spark output parts into the flat files required for submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def _part_files(directory: Path) -> list[Path]:
    files = sorted(path for path in directory.glob("part-*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No Spark part files found in {directory}")
    return files


def merge_tsv_parts(directory: Path, destination: Path, columns: Sequence[str]) -> int:
    """Merge Spark CSV parts in bounded memory, validating each repeated header."""
    expected_header = "\t".join(columns)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with destination.open("w", encoding="utf-8", newline="") as output:
        output.write(expected_header + "\n")
        for part in _part_files(directory):
            with part.open("r", encoding="utf-8-sig", newline="") as source:
                header = source.readline().rstrip("\r\n")
                if header != expected_header:
                    raise ValueError(
                        f"Unexpected TSV header in {part}: {header!r}; expected {expected_header!r}"
                    )
                for line in source:
                    output.write(line)
                    rows += 1
    return rows


def read_json_parts(directory: Path) -> dict[str, object]:
    for part in _part_files(directory):
        with part.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"Metrics JSON in {part} must be an object")
                    return value
    raise ValueError(f"No JSON metrics found in {directory}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-parts", type=Path, required=True)
    parser.add_argument("--matching-parts", type=Path, required=True)
    parser.add_argument("--metrics-parts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate_rows = merge_tsv_parts(
        args.candidate_parts, args.output_dir / "candidate_pairs.tsv",
        ["source1_entity_id", "candidate_entity_ids"],
    )
    matching_rows = merge_tsv_parts(
        args.matching_parts, args.output_dir / "matching_results.tsv",
        ["source1_entity_id", "matched_entity_ids"],
    )
    metrics = read_json_parts(args.metrics_parts)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Materialized {candidate_rows:,} candidate rows and {matching_rows:,} matching rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
