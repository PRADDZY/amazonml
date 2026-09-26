"""Cross-fitted retrieval, matching, and submission generation on AWS."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from v2_features import FEATURE_NAMES, pair_features
from v2_retrieval import Retriever, build_index, rows
from v2_text import Record, entity_partition

LOG = logging.getLogger("v2.pipeline")
FOLDS = 5
CANDIDATES_PER_TARGET = 8
THRESHOLDS = (0.01, 0.025) + tuple(i / 100 for i in range(5, 96, 5)) + (0.975, 0.99, 0.995, 0.999, 1.01)
CAPS = (1, 2, 3, 4, 6, 8)
RRF_OFFSET = 60

_RETRIEVER = None
_REFERENCES: list[Record] = []
_REFERENCE_FOLDS = np.asarray([], dtype=np.int8)
_MODEL = None
_OPERATING = None


def _read_truth(path: Path, source1_path: Path) -> tuple[dict[str, int], list[Record], Counter]:
    references = []
    source1_ids = {}
    for rid, row in enumerate(rows(source1_path)):
        entity_id = row["entity_id"]
        if entity_id in source1_ids:
            raise RuntimeError(f"Duplicate Source 1 entity ID: {entity_id}")
        source1_ids[entity_id] = rid
        references.append(Record.from_row(row))
    owners: dict[str, int] = {}
    group_sizes = Counter()
    for row in rows(path):
        source1_id = row["source1_entity_id"]
        if source1_id not in source1_ids:
            raise RuntimeError(f"Ground truth references unknown Source 1 ID: {source1_id}")
        rid = source1_ids[source1_id]
        target_ids = [value.strip() for value in row["matched_entity_ids"].split(",") if value.strip()]
        group_sizes[len(target_ids)] += 1
        for target_id in target_ids:
            previous = owners.setdefault(target_id, rid)
            if previous != rid:
                raise RuntimeError("Ground truth assigns one target to multiple Source 1 entities")
    return owners, references, group_sizes


def _init_retriever(index_root: str) -> None:
    global _RETRIEVER
    _RETRIEVER = Retriever(Path(index_root))


def _init_predictor(index_root: str, model_path: str, operating: dict) -> None:
    global _RETRIEVER, _MODEL, _OPERATING
    _RETRIEVER = Retriever(Path(index_root))
    _MODEL = lgb.Booster(model_file=model_path)
    _OPERATING = operating


def _ordered_candidates(target: Record, evidence: dict[int, dict], references: list[Record],
                        cap: int, target_source: str):
    ranked = sorted(
        evidence.items(),
        key=lambda item: (
            -sum(1.0 / (RRF_OFFSET + item[1].get(view + "_rank", 25))
                 for view in ("joint", "name", "address")),
            item[1].get("joint_rank", 25),
            item[1].get("name_rank", 25),
            item[1].get("address_rank", 25),
            item[0],
        ),
    )
    selected = ranked[:cap]
    return [
        (rid, rank, pair_features(target, references[rid], ev, target_source))
        for rank, (rid, ev) in enumerate(selected, 1)
    ]


def _training_task(task):
    seq, row, owner_rid, fold, source_code = task
    target = Record.from_row(row)
    evidence = _RETRIEVER.retrieve(target, 24)
    pairs = _ordered_candidates(target, evidence, _REFERENCES, CANDIDATES_PER_TARGET, source_code)
    return seq, owner_rid, fold, target.country, pairs


def _training_tasks(train_dir: Path, owners: dict[str, int], shard_index: int = 0,
                    shard_count: int = 1):
    seq = 0
    for source, source_code in ((2, "S2"), (3, "S3")):
        for row in rows(train_dir / f"train_source{source}.tsv"):
            current_seq = seq
            seq += 1
            if entity_partition(row["entity_id"], shard_count) != shard_index:
                continue
            owner = owners.get(row["entity_id"], -1)
            fold = int(_REFERENCE_FOLDS[owner]) if owner >= 0 else entity_partition(row["entity_id"], FOLDS)
            yield current_seq, row, owner, fold, source_code


def _write_training_features(train_dir: Path, index_root: Path, output_dir: Path,
                             owners: dict[str, int], references: list[Record], workers: int,
                             shard_index: int = 0, shard_count: int = 1):
    global _REFERENCES, _REFERENCE_FOLDS
    _REFERENCES = references
    _REFERENCE_FOLDS = np.asarray(
        [entity_partition(reference.entity_id, FOLDS) for reference in references], dtype=np.int8,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-shard-{shard_index:03d}-of-{shard_count:03d}" if shard_count > 1 else ""
    feature_path = output_dir / f"train_features{suffix}.parquet"
    target_path = output_dir / f"train_targets{suffix}.parquet"
    feature_schema = pa.schema([
        ("seq", pa.int64()), ("owner", pa.int32()), ("fold", pa.int8()),
        ("rid", pa.int32()), ("rid_fold", pa.int8()), ("rank", pa.int8()), ("label", pa.int8()),
        *[(name, pa.float32()) for name in FEATURE_NAMES],
    ])
    target_schema = pa.schema([
        ("seq", pa.int64()), ("owner", pa.int32()), ("fold", pa.int8()),
        ("country", pa.string()),
    ])
    feature_writer = pq.ParquetWriter(feature_path, feature_schema, compression="zstd")
    target_writer = pq.ParquetWriter(target_path, target_schema, compression="zstd")
    feature_rows, target_rows = [], []
    query_count = candidate_count = positive_seen = 0
    recall_hits = Counter()
    started = time.monotonic()

    def flush():
        nonlocal feature_rows, target_rows
        if feature_rows:
            feature_writer.write_table(pa.Table.from_pylist(feature_rows, schema=feature_schema))
            feature_rows = []
        if target_rows:
            target_writer.write_table(pa.Table.from_pylist(target_rows, schema=target_schema))
            target_rows = []

    tasks = _training_tasks(train_dir, owners, shard_index, shard_count)
    with mp.get_context("fork").Pool(workers, initializer=_init_retriever, initargs=(str(index_root),)) as pool:
        for seq, owner, fold, country, pairs in pool.imap(_training_task, tasks, chunksize=32):
            query_count += 1
            candidate_count += len(pairs)
            if owner >= 0:
                positive_seen += 1
            target_rows.append({"seq": seq, "owner": owner, "fold": fold, "country": country})
            for rid, rank, values in pairs:
                label = int(rid == owner)
                if label:
                    for cap in CAPS:
                        if rank <= cap:
                            recall_hits[cap] += 1
                feature_rows.append({
                    "seq": seq, "owner": owner, "fold": fold, "rid": rid,
                    "rid_fold": int(_REFERENCE_FOLDS[rid]),
                    "rank": rank, "label": label,
                    **dict(zip(FEATURE_NAMES, values)),
                })
            if query_count % 10000 == 0:
                flush()
                LOG.info("Built features for %s targets; %s candidate pairs; %.1f targets/sec",
                         query_count, candidate_count, query_count / max(time.monotonic() - started, 1))
    flush()
    feature_writer.close()
    target_writer.close()
    expected_positives = sum(
        entity_partition(target_id, shard_count) == shard_index for target_id in owners
    )
    if positive_seen != expected_positives:
        raise RuntimeError(
            f"Shard {shard_index}/{shard_count} found {positive_seen} labeled targets; "
            f"expected {expected_positives}"
        )
    metrics = {
        "training_targets": query_count,
        "candidate_pairs": candidate_count,
        "positive_targets": positive_seen,
        "candidate_hits_by_cap": {str(cap): recall_hits[cap] for cap in CAPS},
        "candidate_recall_by_cap": {
            str(cap): recall_hits[cap] / positive_seen if positive_seen else 0.0 for cap in CAPS
        },
        "feature_seconds": time.monotonic() - started,
    }
    LOG.info("Training feature collection complete: %s", json.dumps(metrics))
    return feature_path, target_path, metrics


def _score_counts(probabilities: np.ndarray, candidate_rids: np.ndarray, owners: np.ndarray,
                  truth_counts: np.ndarray, source1_countries: list[str], reference_folds: np.ndarray):
    n_targets = candidate_rids.shape[0]
    n_refs = len(truth_counts)
    thresholds = np.asarray(THRESHOLDS, dtype=np.float32)
    predictions = np.zeros((len(CAPS), len(thresholds), n_refs), dtype=np.uint32)
    true_positives = np.zeros_like(predictions)
    country_index = np.asarray([{"US": 0, "India": 1, "France": 2}.get(c, 3) for c in source1_countries])
    for cap_i, cap in enumerate(CAPS):
        chosen_col = np.argmax(probabilities[:, :cap], axis=1)
        best_prob = probabilities[np.arange(n_targets), chosen_col]
        best_rid = candidate_rids[np.arange(n_targets), chosen_col]
        valid = (best_rid >= 0) & (best_prob >= thresholds[0])
        for threshold_i, threshold in enumerate(thresholds):
            predicted = valid & (best_prob >= threshold)
            refs = best_rid[predicted]
            predictions[cap_i, threshold_i] = np.bincount(refs, minlength=n_refs)
            correct = owners[predicted] == refs
            true_positives[cap_i, threshold_i] = np.bincount(refs[correct], minlength=n_refs)

    scores = np.ones_like(predictions, dtype=np.float32)
    for cap_i in range(len(CAPS)):
        for threshold_i in range(len(thresholds)):
            predicted = predictions[cap_i, threshold_i]
            tp = true_positives[cap_i, threshold_i]
            denominator = predicted + 0.25 * truth_counts
            nonempty = denominator > 0
            scores[cap_i, threshold_i, nonempty] = 1.25 * tp[nonempty] / denominator[nonempty]
    calibration_entities = reference_folds != FOLDS - 1
    audit_entities = reference_folds == FOLDS - 1
    mean_scores = scores[:, :, calibration_entities].mean(axis=2)
    best_score = float(mean_scores.max())
    eligible = np.argwhere(mean_scores >= best_score - 0.0002)
    best_cap_i, best_threshold_i = min(
        eligible,
        key=lambda pair: (CAPS[int(pair[0])], -float(mean_scores[tuple(pair)]), -float(thresholds[int(pair[1])])),
    )
    operating = {
        "cap_per_target": int(CAPS[int(best_cap_i)]),
        "threshold": float(thresholds[int(best_threshold_i)]),
        "macro_f0_5": float(mean_scores[best_cap_i, best_threshold_i]),
        "audit_macro_f0_5": float(scores[best_cap_i, best_threshold_i, audit_entities].mean()),
        "audit_source1_entities": int(audit_entities.sum()),
        "candidate_pairs": int(np.sum(candidate_rids[:, :CAPS[int(best_cap_i)]] >= 0)),
    }
    grid = []
    for cap_i, cap in enumerate(CAPS):
        row = {"cap_per_target": cap, "candidate_recall": 0.0}
        for threshold_i, threshold in enumerate(thresholds):
            row[f"macro_f0_5@{threshold:g}"] = float(mean_scores[cap_i, threshold_i])
        grid.append(row)
    best_tp = true_positives[best_cap_i, best_threshold_i, calibration_entities]
    best_pred = predictions[best_cap_i, best_threshold_i, calibration_entities]
    country_scores = {}
    for name, code in (("US", 0), ("India", 1), ("France", 2), ("Other", 3)):
        members = (country_index == code) & calibration_entities
        if members.any():
            country_scores[name] = float(scores[best_cap_i, best_threshold_i, members].mean())
    operating["per_country_macro_f0_5"] = country_scores
    operating["truth_entities"] = int(np.sum(truth_counts > 0))
    operating["matched_pairs"] = int(best_tp.sum())
    operating["predicted_pairs"] = int(best_pred.sum())
    return operating, grid, predictions, true_positives


def _fit_oof(feature_path: Path, target_path: Path, references: list[Record], owners: dict[str, int],
             output_dir: Path):
    started = time.monotonic()
    feature_table = pq.read_table(feature_path)
    targets = pq.read_table(target_path)
    n_targets = targets.num_rows
    seq = feature_table["seq"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    rid = feature_table["rid"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    rank = feature_table["rank"].to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    fold = feature_table["fold"].to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    rid_fold = feature_table["rid_fold"].to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    label = feature_table["label"].to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    X = np.empty((feature_table.num_rows, len(FEATURE_NAMES)), dtype=np.float32)
    for col, name in enumerate(FEATURE_NAMES):
        X[:, col] = feature_table[name].to_numpy(zero_copy_only=False)
    del feature_table
    target_seq = targets["seq"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    target_order = np.argsort(target_seq, kind="stable")
    if not np.array_equal(target_seq[target_order], np.arange(n_targets, dtype=np.int64)):
        raise RuntimeError("Training target shards do not cover each global sequence exactly once")
    target_owner = targets["owner"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)[target_order]
    source1_countries = [reference.country for reference in references]
    reference_folds = np.asarray(
        [entity_partition(reference.entity_id, FOLDS) for reference in references], dtype=np.int8,
    )
    truth_counts = np.bincount(target_owner[target_owner >= 0], minlength=len(references)).astype(np.int32)
    y = label.astype(np.float32, copy=False)
    oof = np.full((X.shape[0],), np.nan, dtype=np.float32)
    fold_iterations = []
    training_threads = min(32, os.cpu_count() or 8)
    params = {
        "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
        "num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 80,
        "feature_fraction": 0.9, "bagging_fraction": 0.85, "bagging_freq": 1,
        "lambda_l2": 2.0, "max_bin": 63, "verbosity": -1, "num_threads": training_threads,
        "seed": 20260926, "feature_fraction_seed": 20260927,
        "bagging_seed": 20260928, "deterministic": True,
    }
    for validation_fold in range(FOLDS):
        # Hold out every candidate for a target together. Scoring candidate rows
        # in separate folds gives one target probabilities from several models,
        # unlike inference where a single model ranks the entire candidate set.
        # The final reference fold stays out of every training split so its
        # per-entity audit remains untouched, including its negative examples.
        train_mask = (
            (fold != validation_fold)
            & (rid_fold != validation_fold)
            & (rid_fold != FOLDS - 1)
            & (rank <= max(CAPS))
        )
        valid_mask = fold == validation_fold
        if len(np.unique(label[train_mask])) < 2 or not valid_mask.any():
            raise RuntimeError(f"Fold {validation_fold} lacks both classes or validation rows")
        train_set = lgb.Dataset(X[train_mask], label=y[train_mask], feature_name=list(FEATURE_NAMES))
        valid_set = lgb.Dataset(X[valid_mask], label=y[valid_mask], reference=train_set)
        model = lgb.train(
            params, train_set, num_boost_round=900, valid_sets=[valid_set],
            valid_names=["heldout"], callbacks=[lgb.early_stopping(60, verbose=False)],
        )
        oof[valid_mask] = model.predict(X[valid_mask], num_threads=training_threads).astype(np.float32)
        fold_iterations.append(int(model.best_iteration or 900))
        LOG.info("Cross-fit fold %s finished at iteration %s", validation_fold, fold_iterations[-1])
        del train_set, valid_set, model

    if np.isnan(oof).any():
        raise RuntimeError("Some candidate pairs were not assigned an out-of-fold prediction")

    probability_matrix = np.zeros((n_targets, CANDIDATES_PER_TARGET), dtype=np.float32)
    candidate_matrix = np.full((n_targets, CANDIDATES_PER_TARGET), -1, dtype=np.int32)
    probability_matrix[seq, rank.astype(np.int64) - 1] = oof
    candidate_matrix[seq, rank.astype(np.int64) - 1] = rid
    candidate_recall = {}
    for cap in CAPS:
        found = np.any(candidate_matrix[:, :cap] == target_owner[:, None], axis=1)
        found &= target_owner >= 0
        candidate_recall[str(cap)] = float(found.sum() / max(1, len(owners)))
    operating, grid, _, _ = _score_counts(
        probability_matrix, candidate_matrix, target_owner, truth_counts,
        source1_countries, reference_folds,
    )
    for row in grid:
        row["candidate_recall"] = candidate_recall[str(row["cap_per_target"])]
    operating["candidate_recall_by_cap"] = candidate_recall
    operating["fold_iterations"] = fold_iterations
    operating["feature_rows"] = int(X.shape[0])
    operating["target_rows"] = int(n_targets)
    operating["fit_and_oof_seconds"] = time.monotonic() - started
    (output_dir / "oof_metrics.json").write_text(json.dumps({"operating_point": operating, "grid": grid}, indent=2) + "\n")
    LOG.info("Selected operating point: %s", json.dumps(operating))

    final_iterations = max(150, int(np.median(fold_iterations)))
    final_set = lgb.Dataset(X, label=y, feature_name=list(FEATURE_NAMES))
    final_model = lgb.train(params, final_set, num_boost_round=final_iterations)
    model_path = output_dir / "model.txt"
    final_model.save_model(str(model_path))
    del final_set, final_model, X, y, oof
    return operating, model_path


def _merge_parquet_shards(shard_paths: list[Path], destination: Path) -> None:
    if not shard_paths or any(not path.is_file() for path in shard_paths):
        raise RuntimeError("One or more feature shards are missing")
    schema = pq.read_schema(shard_paths[0])
    with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
        for path in shard_paths:
            parquet = pq.ParquetFile(path)
            if not schema.equals(parquet.schema_arrow, check_metadata=False):
                raise RuntimeError(f"Parquet shard schema mismatch: {path}")
            for batch in parquet.iter_batches(batch_size=65536):
                writer.write_table(pa.Table.from_batches([batch]))


def _fit_feature_shards(train_dir: Path, test_dir: Path, index_dir: Path,
                        output_dir: Path, workers: int, shard_count: int,
                        feature_dir: Path | None = None):
    owners, train_references, _ = _read_truth(
        train_dir / "train_ground_truth.tsv", train_dir / "train_source1.tsv",
    )
    shard_root = feature_dir or output_dir / "feature-shards"
    feature_paths = [
        shard_root / f"{index:03d}" / f"train_features-shard-{index:03d}-of-{shard_count:03d}.parquet"
        for index in range(shard_count)
    ]
    target_paths = [
        shard_root / f"{index:03d}" / f"train_targets-shard-{index:03d}-of-{shard_count:03d}.parquet"
        for index in range(shard_count)
    ]
    metric_paths = [shard_root / f"{index:03d}" / "feature-shard-metrics.json" for index in range(shard_count)]
    missing = [path for path in [*feature_paths, *target_paths, *metric_paths] if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} shard artifacts; first missing: {missing[0]}")

    feature_path = output_dir / "train_features.parquet"
    target_path = output_dir / "train_targets.parquet"
    _merge_parquet_shards(feature_paths, feature_path)
    _merge_parquet_shards(target_paths, target_path)
    shard_metrics = [json.loads(path.read_text()) for path in metric_paths]
    positive_targets = sum(item["positive_targets"] for item in shard_metrics)
    if positive_targets != len(owners):
        raise RuntimeError(f"Feature shards cover {positive_targets} positives; expected {len(owners)}")
    data_metrics = {
        "training_targets": sum(item["training_targets"] for item in shard_metrics),
        "candidate_pairs": sum(item["candidate_pairs"] for item in shard_metrics),
        "positive_targets": positive_targets,
        "candidate_recall_by_cap": {
            str(cap): sum(item["candidate_hits_by_cap"][str(cap)] for item in shard_metrics) / positive_targets
            for cap in CAPS
        },
        "feature_seconds_sum": sum(item["feature_seconds"] for item in shard_metrics),
        "feature_shards": shard_count,
    }
    operating, model_path = _fit_oof(feature_path, target_path, train_references, owners, output_dir)
    test_source1 = test_dir / "test_source1.tsv"
    test_index = output_dir / "test-index"
    test_manifest = build_index(test_source1, test_index, threads=min(16, workers))
    (output_dir / "fit_metrics.json").write_text(json.dumps({
        "training": data_metrics,
        "training_oof_operating_point": operating,
        "training_index": json.loads((index_dir / "manifest.json").read_text()),
        "test_index": test_manifest,
    }, indent=2) + "\n")
    return data_metrics, operating, model_path, test_index, test_manifest


def _test_task(task):
    seq, row, source_code = task
    target = Record.from_row(row)
    evidence = _RETRIEVER.retrieve(target, 24)
    selected = _ordered_candidates(
        target, evidence, _REFERENCES, int(_OPERATING["cap_per_target"]), source_code,
    )
    if not selected:
        return seq, row["entity_id"], target.country, [], None
    matrix = np.asarray([item[2] for item in selected], dtype=np.float32)
    probabilities = _MODEL.predict(matrix, num_threads=1)
    scored = [(item[0], item[1], float(probability))
              for item, probability in zip(selected, probabilities)]
    best = max(scored, key=lambda item: (item[2], -item[1]))
    matched = best if best[2] >= float(_OPERATING["threshold"]) else None
    return seq, row["entity_id"], target.country, scored, matched


def _test_tasks(test_dir: Path, shard_index: int = 0, shard_count: int = 1):
    seq = 0
    for source, source_code in ((2, "S2"), (3, "S3")):
        for row in rows(test_dir / f"test_source{source}.tsv"):
            current_seq = seq
            seq += 1
            if entity_partition(row["entity_id"], shard_count) == shard_index:
                yield current_seq, row, source_code


def _sort_pairs(source: Path, destination: Path, keys: list[str]):
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    subprocess.run(["sort", "-T", str(destination.parent), "-t", "\t", *keys,
                    "-o", str(destination), str(source)], check=True, env=env)


def _write_grouped_pairs(sorted_path: Path, output_path: Path, references: list[Record], kind: str):
    id_column = "candidate_entity_ids" if kind == "candidate" else "matched_entity_ids"
    with sorted_path.open(encoding="utf-8", newline="") as source, output_path.open("w", encoding="utf-8", newline="") as output:
        output.write(f"source1_entity_id\t{id_column}\n")
        counts = []
        current = source.readline().rstrip("\n")
        for rid, reference in enumerate(references):
            ids = []
            seen = set()
            while current:
                parts = current.split("\t")
                if int(parts[0]) != rid:
                    break
                target_id = parts[2]
                if target_id not in seen:
                    ids.append(target_id)
                    seen.add(target_id)
                current = source.readline().rstrip("\n")
            output.write(f"{reference.entity_id}\t{','.join(ids)}\n")
            counts.append(len(ids))
        if current:
            raise RuntimeError(f"Output contains a candidate for invalid Source 1 row: {current}")
    return np.asarray(counts, dtype=np.int32)


def _write_final_test_outputs(output_dir: Path, references: list[Record], candidate_pairs: Path,
                              match_pairs: Path, total: int, candidate_count: int, matches: int,
                              targets_by_country: Counter, operating: dict, runtime_seconds: float):
    sorted_candidates = output_dir / "candidate-pairs.sorted.tsv"
    sorted_matches = output_dir / "match-pairs.sorted.tsv"
    _sort_pairs(candidate_pairs, sorted_candidates, ["-k1,1n", "-k2,2n", "-k3,3"])
    _sort_pairs(match_pairs, sorted_matches, ["-k1,1n", "-k2,2gr", "-k3,3"])
    candidate_counts = _write_grouped_pairs(
        sorted_candidates, output_dir / "candidate_pairs.tsv", references, "candidate",
    )
    match_counts = _write_grouped_pairs(
        sorted_matches, output_dir / "matching_results.tsv", references, "matching",
    )
    if int(candidate_counts.sum()) != candidate_count or int(match_counts.sum()) != matches:
        raise RuntimeError("Grouped TSV rows do not match the pair totals")
    references_by_country = Counter(reference.country for reference in references)
    possible_pairs = sum(references_by_country[country] * count for country, count in targets_by_country.items())
    country_counts = {}
    for country in sorted(references_by_country):
        members = np.asarray([ref.country == country for ref in references], dtype=bool)
        values = candidate_counts[members]
        country_counts[country] = {
            "source1_rows": int(members.sum()), "candidate_pairs": int(values.sum()),
            "candidate_count_mean": float(values.mean()) if values.size else 0.0,
            "candidate_count_p95": float(np.percentile(values, 95)) if values.size else 0.0,
            "candidate_count_p99": float(np.percentile(values, 99)) if values.size else 0.0,
            "candidate_count_max": int(values.max()) if values.size else 0,
        }
    test_metrics = {
        "test_targets": total, "candidate_pairs": candidate_count, "predicted_matches": matches,
        "candidate_count_mean": candidate_count / max(1, len(references)),
        "candidate_count_median": float(np.median(candidate_counts)) if candidate_counts.size else 0.0,
        "candidate_count_p95": float(np.percentile(candidate_counts, 95)) if candidate_counts.size else 0.0,
        "candidate_count_p99": float(np.percentile(candidate_counts, 99)) if candidate_counts.size else 0.0,
        "candidate_count_max": int(candidate_counts.max()) if candidate_counts.size else 0,
        "matched_count_mean": float(match_counts.mean()) if match_counts.size else 0.0,
        "possible_same_country_pairs": int(possible_pairs),
        "candidate_reduction_ratio": 1.0 - candidate_count / max(1, possible_pairs),
        "per_country": country_counts,
        "runtime_seconds": runtime_seconds,
        "operating_point": operating,
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")
    return total, candidate_count, matches, possible_pairs


def _predict_test(test_dir: Path, index_root: Path, output_dir: Path, references: list[Record],
                  model_path: Path, operating: dict, workers: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_pairs = output_dir / "candidate-pairs.unsorted.tsv"
    match_pairs = output_dir / "match-pairs.unsorted.tsv"
    global _REFERENCES
    _REFERENCES = references
    total = matches = candidate_count = 0
    targets_by_country = Counter()
    started = time.monotonic()
    with candidate_pairs.open("w", encoding="utf-8", newline="") as candidates, \
            match_pairs.open("w", encoding="utf-8", newline="") as matched:
        initializer_args = (str(index_root), str(model_path), operating)
        with mp.get_context("fork").Pool(workers, initializer=_init_predictor, initargs=initializer_args) as pool:
            for total, (_, target_id, country, pairs, match) in enumerate(
                    pool.imap(_test_task, _test_tasks(test_dir), chunksize=32), 1):
                targets_by_country[country] += 1
                for rid, rank, probability in pairs:
                    candidates.write(f"{rid}\t{rank}\t{target_id}\n")
                    candidate_count += 1
                if match:
                    rid, _, probability = match
                    matched.write(f"{rid}\t{probability:.8f}\t{target_id}\n")
                    matches += 1
                if total % 10000 == 0:
                    LOG.info("Scored %s targets; %s candidates, %s matches; %.1f targets/sec",
                             total, candidate_count, matches, total / max(time.monotonic() - started, 1))
    result = _write_final_test_outputs(
        output_dir, references, candidate_pairs, match_pairs, total, candidate_count, matches,
        targets_by_country, operating, time.monotonic() - started,
    )
    for path in (candidate_pairs, match_pairs, output_dir / "candidate-pairs.sorted.tsv",
                 output_dir / "match-pairs.sorted.tsv"):
        path.unlink(missing_ok=True)
    return result


def _predict_test_shard(test_dir: Path, index_root: Path, output_dir: Path, references: list[Record],
                        model_path: Path, operating: dict, workers: int,
                        shard_index: int, shard_count: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_pairs = output_dir / "candidate-pairs.raw.tsv"
    match_pairs = output_dir / "match-pairs.raw.tsv"
    global _REFERENCES
    _REFERENCES = references
    total = matches = candidate_count = 0
    targets_by_country = Counter()
    started = time.monotonic()
    with candidate_pairs.open("w", encoding="utf-8", newline="") as candidates, \
            match_pairs.open("w", encoding="utf-8", newline="") as matched:
        initializer_args = (str(index_root), str(model_path), operating)
        with mp.get_context("fork").Pool(workers, initializer=_init_predictor, initargs=initializer_args) as pool:
            for total, (_, target_id, country, pairs, match) in enumerate(
                    pool.imap(_test_task, _test_tasks(test_dir, shard_index, shard_count), chunksize=32), 1):
                targets_by_country[country] += 1
                for rid, rank, probability in pairs:
                    candidates.write(f"{rid}\t{rank}\t{target_id}\n")
                    candidate_count += 1
                if match:
                    rid, _, probability = match
                    matched.write(f"{rid}\t{probability:.8f}\t{target_id}\n")
                    matches += 1
                if total % 10000 == 0:
                    LOG.info("Shard %s/%s scored %s targets; %s candidates; %.1f targets/sec",
                             shard_index, shard_count, total, candidate_count,
                             total / max(time.monotonic() - started, 1))
    metrics = {
        "shard_index": shard_index, "shard_count": shard_count, "test_targets": total,
        "candidate_pairs": candidate_count, "predicted_matches": matches,
        "targets_by_country": dict(targets_by_country),
        "runtime_seconds": time.monotonic() - started,
    }
    (output_dir / "test-shard-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def _finalize_test_shards(test_dir: Path, output_dir: Path, references: list[Record],
                          operating: dict, shard_count: int):
    shard_root = output_dir / "test-shards"
    candidate_pairs = output_dir / "candidate-pairs.unsorted.tsv"
    match_pairs = output_dir / "match-pairs.unsorted.tsv"
    total = candidate_count = matches = 0
    runtime_seconds = 0.0
    targets_by_country = Counter()
    with candidate_pairs.open("wb") as candidate_out, match_pairs.open("wb") as match_out:
        for shard_index in range(shard_count):
            shard_dir = shard_root / f"{shard_index:03d}"
            metric_path = shard_dir / "test-shard-metrics.json"
            candidate_path = shard_dir / "candidate-pairs.raw.tsv"
            match_path = shard_dir / "match-pairs.raw.tsv"
            if not all(path.is_file() for path in (metric_path, candidate_path, match_path)):
                raise RuntimeError(f"Test shard {shard_index}/{shard_count} is incomplete")
            metrics = json.loads(metric_path.read_text())
            if metrics.get("shard_index") != shard_index or metrics.get("shard_count") != shard_count:
                raise RuntimeError(f"Test shard manifest does not match shard {shard_index}")
            with candidate_path.open("rb") as shard_file:
                shutil.copyfileobj(shard_file, candidate_out)
            with match_path.open("rb") as shard_file:
                shutil.copyfileobj(shard_file, match_out)
            total += metrics["test_targets"]
            candidate_count += metrics["candidate_pairs"]
            matches += metrics["predicted_matches"]
            runtime_seconds = max(runtime_seconds, metrics["runtime_seconds"])
            targets_by_country.update(metrics["targets_by_country"])
    test_source_rows = sum(1 for _ in rows(test_dir / "test_source2.tsv")) + sum(
        1 for _ in rows(test_dir / "test_source3.tsv")
    )
    if total != test_source_rows:
        raise RuntimeError(f"Test shards cover {total} targets; expected {test_source_rows}")
    result = _write_final_test_outputs(
        output_dir, references, candidate_pairs, match_pairs, total, candidate_count, matches,
        targets_by_country, operating, runtime_seconds,
    )
    for path in (candidate_pairs, match_pairs, output_dir / "candidate-pairs.sorted.tsv",
                 output_dir / "match-pairs.sorted.tsv"):
        path.unlink(missing_ok=True)
    return result


def main():
    global _REFERENCES
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--phase", choices=("full", "features", "fit", "test", "finalize"), default="full")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must be in [0, shard-count)")
    if args.phase in ("features", "test") and args.shard_count < 2:
        parser.error("features and test phases require shard-count >= 2")
    if args.phase in ("test", "finalize") and args.fit_dir is None:
        parser.error("test and finalize phases require --fit-dir")
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "features":
        if not (args.index_dir / "manifest.json").is_file():
            raise RuntimeError("Training index cache is missing")
        owners, train_references, _ = _read_truth(
            args.train_dir / "train_ground_truth.tsv", args.train_dir / "train_source1.tsv",
        )
        _, _, data_metrics = _write_training_features(
            args.train_dir, args.index_dir, args.output_dir, owners, train_references,
            args.workers, args.shard_index, args.shard_count,
        )
        (args.output_dir / "feature-shard-metrics.json").write_text(
            json.dumps({"shard_index": args.shard_index, "shard_count": args.shard_count,
                        **data_metrics}, indent=2) + "\n"
        )
        LOG.info("Feature shard complete: %s", json.dumps(data_metrics))
        return

    if args.phase == "fit":
        _fit_feature_shards(
            args.train_dir, args.test_dir, args.index_dir, args.output_dir,
            args.workers, args.shard_count, args.feature_dir,
        )
        return

    if args.phase == "test":
        fit_dir = args.fit_dir
        fit_metrics = json.loads((fit_dir / "fit_metrics.json").read_text())
        operating = json.loads((fit_dir / "oof_metrics.json").read_text())["operating_point"]
        references = [Record.from_row(row) for row in rows(args.test_dir / "test_source1.tsv")]
        if not references:
            raise RuntimeError("Test Source 1 is empty")
        _predict_test_shard(
            args.test_dir, args.index_dir, args.output_dir, references,
            fit_dir / "model.txt", operating, args.workers,
            args.shard_index, args.shard_count,
        )
        return

    if args.phase == "finalize":
        fit_dir = args.fit_dir
        fit_metrics = json.loads((fit_dir / "fit_metrics.json").read_text())
        operating = json.loads((fit_dir / "oof_metrics.json").read_text())["operating_point"]
        references = [Record.from_row(row) for row in rows(args.test_dir / "test_source1.tsv")]
        if not references:
            raise RuntimeError("Test Source 1 is empty")
        test_targets, candidate_count, matches, possible_pairs = _finalize_test_shards(
            args.test_dir, args.output_dir, references, operating, args.shard_count,
        )
        finalize_seconds = time.monotonic() - started
        run_started = os.environ.get("DEVCORE_RUN_STARTED_UTC")
        if run_started:
            parsed_start = datetime.fromisoformat(run_started.replace("Z", "+00:00"))
            if parsed_start.tzinfo is None:
                raise RuntimeError("Workflow start timestamp must include a UTC offset")
            workflow_seconds = (datetime.now(timezone.utc) - parsed_start.astimezone(timezone.utc)).total_seconds()
        else:
            workflow_seconds = finalize_seconds
        metrics = {
            "kind": "unicode_bm25_lightgbm_oof_v2_sharded",
            "runtime_seconds": workflow_seconds,
            "finalize_runtime_seconds": finalize_seconds,
            "workflow_started_utc": run_started,
            "training": fit_metrics["training"],
            "training_oof_operating_point": operating,
            "training_index": fit_metrics["training_index"],
            "test_index": fit_metrics["test_index"],
            "test_targets": test_targets,
            "test_candidate_pairs": candidate_count,
            "test_predicted_matches": matches,
            "possible_same_country_pairs": possible_pairs,
            "candidate_reduction_ratio": 1.0 - candidate_count / max(1, possible_pairs),
            "feature_names": list(FEATURE_NAMES),
            "shard_count": args.shard_count,
        }
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return

    source1_train = args.train_dir / "train_source1.tsv"
    owners, train_references, truth_group_sizes = _read_truth(
        args.train_dir / "train_ground_truth.tsv", source1_train,
    )
    LOG.info("Loaded %s Source 1 references and %s labeled targets; truth group sizes=%s",
             len(train_references), len(owners), truth_group_sizes)
    manifest = build_index(source1_train, args.index_dir, threads=min(16, args.workers))
    feature_path, target_path, data_metrics = _write_training_features(
        args.train_dir, args.index_dir, args.output_dir, owners, train_references, args.workers,
    )
    operating, model_path = _fit_oof(
        feature_path, target_path, train_references, owners, args.output_dir,
    )
    del owners, train_references
    _REFERENCES = []
    gc.collect()
    # The training index and labels are not used for test rows. Replace references before scoring.
    test_source1 = args.test_dir / "test_source1.tsv"
    test_references = [Record.from_row(row) for row in rows(test_source1)]
    if not test_references:
        raise RuntimeError("Test Source 1 is empty")
    # Test Source 1 can use a different corpus; build an isolated index and never mix IDs across splits.
    test_index = args.index_dir.parent / "test-index"
    test_manifest = build_index(test_source1, test_index, threads=min(16, args.workers))
    test_candidates_total, candidate_count, matches, possible_pairs = _predict_test(
        args.test_dir, test_index, args.output_dir, test_references, model_path, operating, args.workers,
    )
    metrics = {
        "kind": "unicode_bm25_lightgbm_oof_v2",
        "runtime_seconds": time.monotonic() - started,
        "training": data_metrics,
        "training_oof_operating_point": operating,
        "training_index": manifest,
        "test_index": test_manifest,
        "test_targets": test_candidates_total,
        "test_candidate_pairs": candidate_count,
        "test_predicted_matches": matches,
        "possible_same_country_pairs": possible_pairs,
        "candidate_reduction_ratio": 1.0 - candidate_count / max(1, possible_pairs),
        "feature_names": list(FEATURE_NAMES),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    feature_path.unlink(missing_ok=True)
    target_path.unlink(missing_ok=True)
    LOG.info("Pipeline complete: %s", json.dumps(metrics))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
