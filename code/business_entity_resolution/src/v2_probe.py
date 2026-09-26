"""AWS-only full-reference retrieval audit on a deterministic labeled query sample."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import logging
import multiprocessing as mp
from pathlib import Path
import time

from v2_retrieval import Retriever, build_index, rows
from v2_text import Record, entity_partition

LOG = logging.getLogger("v2.probe")
RETRIEVER = None


def init_worker(index_root):
    global RETRIEVER
    RETRIEVER = Retriever(Path(index_root))


def retrieve_one(task):
    row, true_rid = task
    record = Record.from_row(row)
    candidates = RETRIEVER.retrieve(record, 24)
    return asdict(record), true_rid, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-modulus", type=int, default=500)
    args = parser.parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source1 = args.train_dir / "train_source1.tsv"
    sample, all_rids = {}, {}
    countries = Counter()
    for rid, row in enumerate(rows(source1)):
        all_rids[row["entity_id"]] = rid
        countries[row["country"]] += 1
        if entity_partition(row["entity_id"], args.sample_modulus) == 0:
            sample[row["entity_id"]] = (rid, row)
    LOG.info("Reference counts %s; sampled %s Source 1 records", countries, len(sample))
    selected_targets, sizes = {}, Counter()
    for row in rows(args.train_dir / "train_ground_truth.tsv"):
        target_ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
        sizes[len(target_ids)] += 1
        if row["source1_entity_id"] in sample:
            for target_id in target_ids:
                rid = sample[row["source1_entity_id"]][0]
                if target_id in selected_targets and selected_targets[target_id] != rid:
                    raise RuntimeError("Ground truth contradicts target exclusivity")
                selected_targets[target_id] = rid
    LOG.info("Selected %s labeled target queries; singleton groups=%s", len(selected_targets), sizes[0])
    manifest = build_index(source1, args.index_dir)
    LOG.info("Index ready; collecting target query sample")
    tasks = []
    for source in (2, 3):
        for row in rows(args.train_dir / f"train_source{source}.tsv"):
            if row["entity_id"] in selected_targets:
                tasks.append((row, selected_targets[row["entity_id"]]))
    if len(tasks) != len(selected_targets):
        raise RuntimeError("Not all sampled truth targets exist in the source files")
    hits, totals, misses = Counter(), Counter(), []
    by_country = defaultdict(Counter)
    retrieval_started = time.monotonic()
    with (args.output_dir / "retrieved.jsonl").open("w", encoding="utf-8") as output:
        with mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(str(args.index_dir),)) as pool:
            for processed, (record, true_rid, candidates) in enumerate(pool.imap_unordered(retrieve_one, tasks, chunksize=24), 1):
                evidence = candidates.get(true_rid, {})
                country = record["country"]
                totals[country] += 1
                for k in (1, 2, 3, 4, 8, 12, 24):
                    for view in ("joint", "name", "address", "union"):
                        found = min((evidence.get(v + "_rank", 999) for v in ("joint", "name", "address"))) <= k if view == "union" else evidence.get(view + "_rank", 999) <= k
                        if found:
                            hits[f"{view}@{k}"] += 1
                            by_country[country][f"{view}@{k}"] += 1
                if not evidence and len(misses) < 100:
                    misses.append({"target": record, "truth_rid": true_rid, "candidates": candidates})
                output.write(json.dumps({"target": record, "truth_rid": true_rid, "candidates": candidates}, ensure_ascii=False) + "\n")
                if processed % 1000 == 0:
                    LOG.info("Retrieved %s/%s targets; union@24 recall %.5f; %.1f queries/sec", processed, len(tasks), hits["union@24"] / processed, processed / (time.monotonic() - retrieval_started))
    missed_rids = {example["truth_rid"] for example in misses}
    reference_examples = {rid: row for rid, row in enumerate(rows(source1)) if rid in missed_rids}
    for example in misses:
        example["true_reference"] = reference_examples[example["truth_rid"]]
    report = {
        "kind": "full_reference_sampled_positive_retrieval_audit",
        "scope": "Candidate recall only; this is not a macro-F0.5 model evaluation",
        "reference_counts": dict(countries), "sampled_source1": len(sample),
        "sampled_targets": len(tasks), "truth_group_sizes": dict(sizes),
        "recall": {key: value / len(tasks) for key, value in sorted(hits.items())},
        "by_country": {country: {key: value / totals[country] for key, value in counts.items()} for country, counts in by_country.items()},
        "retrieval_seconds": time.monotonic() - retrieval_started,
        "runtime_seconds": time.monotonic() - started, "index_manifest": manifest,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "misses.json").write_text(json.dumps(misses, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info("Audit complete: %s", json.dumps(report))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
