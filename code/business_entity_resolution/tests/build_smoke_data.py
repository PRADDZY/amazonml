#!/usr/bin/env python3
"""Create a small local fixture for the SageMaker Processing Spark run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ["entity_id", "business_name", "business_address", "country"]


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def record(entity_id: str, name: str, address: str, country: str) -> dict[str, str]:
    return {"entity_id": entity_id, "business_name": name,
            "business_address": address, "country": country}


def build(root: Path, count: int = 250) -> None:
    train_s1: list[dict[str, str]] = []
    train_s2: list[dict[str, str]] = []
    train_s3: list[dict[str, str]] = []
    truth: list[dict[str, str]] = []
    for index in range(count):
        country = "US" if index % 2 == 0 else "India"
        number = 100 + index
        postal = f"{10000 + index:05d}"
        s1_id = f"S1-TRAIN-{index:06d}"
        s2_id = f"S2-TRAIN-{index:06d}"
        s3_id = f"S3-TRAIN-{index:06d}"
        train_s1.append(record(s1_id, f"Cafe Shop {index} LLC",
                               f"{number} Elm Street Suite 3, Town {index} {postal}", country))
        train_s2.append(record(s2_id, f"Café Shop {index} Incorporated",
                               f"{number} Elm St Ste 3 Town {index} {postal}", country))
        train_s3.append(record(s3_id, f"Cafe Shop {index} Ltd",
                               f"{number} Elm Rd Ste 3 Town {index} {postal}", country))
        train_s2.append(record(f"S2-TRAIN-DECOY-{index:06d}", f"Cafe Shop {index} Market",
                               f"{number + 2} Elm Road Suite 3 Town {index} {postal}", country))
        train_s3.append(record(f"S3-TRAIN-DECOY-{index:06d}", f"Cafe Shop {index} Market",
                               f"{number + 2} Elm Road Suite 3 Town {index} {postal}", country))
        truth.append({"source1_entity_id": s1_id,
                      "matched_entity_ids": f"{s2_id},{s3_id}"})

    test_s1: list[dict[str, str]] = []
    test_s2: list[dict[str, str]] = []
    test_s3: list[dict[str, str]] = []
    for index in range(max(60, count // 2)):
        country = ("France" if index % 3 == 0 else
                   ("US" if index % 3 == 1 else "India"))
        number = 700 + index
        postal = f"{30000 + index:05d}"
        s1_id = f"S1-TEST-{index:06d}"
        s2_id = f"S2-TEST-{index:06d}"
        s3_id = f"S3-TEST-{index:06d}"
        address = (f"{number} Rue Exemple {index} {postal}" if country == "France"
                   else f"{number} Pine Street Suite 2, Town {index} {postal}")
        target_address = (f"{number} Rue Exemple {index} {postal}" if country == "France"
                          else f"{number} Pine St Ste 2 Town {index} {postal}")
        test_s1.append(record(s1_id, f"Bluebird Market {index} LLC", address, country))
        test_s2.append(record(s2_id, f"Bluebird Market {index} Inc", target_address, country))
        test_s3.append(record(s3_id, f"Bluebird Market {index} Limited", target_address, country))
        test_s2.append(record(f"S2-TEST-DECOY-{index:06d}", f"Bluebird Market {index} Cafe",
                              f"{number + 4} Pine Road Suite 2 Town {index} {postal}", country))
        test_s3.append(record(f"S3-TEST-DECOY-{index:06d}", f"Bluebird Market {index} Cafe",
                              f"{number + 4} Pine Road Suite 2 Town {index} {postal}", country))

    test_s1.append(record("S1-TEST-SINGLETON", "Isolated Garden Supply",
                          "987654 Remote Road Nowhere 99999", "France"))
    train_root = root / "train"
    test_root = root / "test"
    write_tsv(train_root / "train_source1.tsv", FIELDS, train_s1)
    write_tsv(train_root / "train_source2.tsv", FIELDS, train_s2)
    write_tsv(train_root / "train_source3.tsv", FIELDS, train_s3)
    write_tsv(train_root / "train_ground_truth.tsv",
              ["source1_entity_id", "matched_entity_ids"], truth)
    write_tsv(test_root / "test_source1.tsv", FIELDS, test_s1)
    write_tsv(test_root / "test_source2.tsv", FIELDS, test_s2)
    write_tsv(test_root / "test_source3.tsv", FIELDS, test_s3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=250)
    args = parser.parse_args()
    build(args.root, args.count)


if __name__ == "__main__":
    main()
