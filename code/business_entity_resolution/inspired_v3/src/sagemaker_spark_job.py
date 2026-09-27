#!/usr/bin/env python3
"""Distributed entity-resolution job for AWS EC2 Spark 3.5."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from typing import Any

import lightgbm as lgb
import numpy as np

from er_core import (
    FEATURE_NAMES, PAIR_FEATURE_COLUMNS, blocking_keys, normalize_address,
    normalize_name, romanize_text, select_operating_point, spark_s3_uri,
)

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


LOG = logging.getLogger("business_entity_resolution")
CAPS = (4, 8, 12, 16, 24, 32)
THRESHOLDS = tuple(i / 100 for i in range(5, 100, 5)) + (0.975, 0.99, 0.995, 1.01)
MAX_CANDIDATES_PER_SOURCE = max(CAPS)

KEY_SCHEMA = ArrayType(StructType([
    StructField("kind", StringType(), False),
    StructField("value", StringType(), False),
]))
PREPARED_SCHEMA = StructType([
    StructField("norm_name", StringType(), False),
    StructField("norm_address", StringType(), False),
    StructField("roman_name", StringType(), False),
    StructField("roman_address", StringType(), False),
    StructField("name_tokens", ArrayType(StringType(), False), False),
    StructField("address_tokens", ArrayType(StringType(), False), False),
    StructField("roman_name_tokens", ArrayType(StringType(), False), False),
    StructField("roman_address_tokens", ArrayType(StringType(), False), False),
    StructField("numbers", ArrayType(StringType(), False), False),
    StructField("postal", StringType(), False),
    StructField("block_keys", KEY_SCHEMA, False),
])


def _prepare_fields(name: str | None, address: str | None) -> dict[str, Any]:
    normalized_name = normalize_name(name)
    normalized_address = normalize_address(address)
    roman_name = normalize_name(romanize_text(name))
    roman_address = normalize_address(romanize_text(address))
    address_tokens = normalized_address.split()
    numbers = list(dict.fromkeys(re.findall(r"\d+", normalized_address)))
    postal = next((token for token in reversed(numbers) if len(token) in (5, 6)), "")
    keys = sorted(blocking_keys(name, address))
    return {
        "norm_name": normalized_name,
        "norm_address": normalized_address,
        "roman_name": roman_name,
        "roman_address": roman_address,
        "name_tokens": normalized_name.split(),
        "address_tokens": address_tokens,
        "roman_name_tokens": roman_name.split(),
        "roman_address_tokens": roman_address.split(),
        "numbers": numbers,
        "postal": postal,
        "block_keys": [{"kind": kind, "value": value} for kind, value in keys],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-prefix", required=True)
    parser.add_argument("--test-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--smoke-test", choices=("true", "false"), default="false")
    parser.add_argument("--instance-type", default="unknown")
    parser.add_argument("--compute-service", default="Amazon EC2 Spark 3.5.4")
    parser.add_argument("--s3-scheme", choices=("s3", "s3a"), default="s3a")
    args, _ = parser.parse_known_args()
    args.smoke_test = args.smoke_test == "true"
    return args


def read_tsv(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read.option("header", "true")
        .option("sep", "\t")
        .option("encoding", "UTF-8")
        .option("mode", "PERMISSIVE")
        .csv(path)
    )


def prepare_records(records: DataFrame, source_column: str | None = None) -> DataFrame:
    prepare_udf = F.udf(_prepare_fields, PREPARED_SCHEMA)
    prepared = records.withColumn(
        "_prepared", prepare_udf(F.col("business_name"), F.col("business_address"))
    )
    selected = prepared.select(
        F.trim(F.coalesce(F.col("entity_id"), F.lit(""))).alias("entity_id"),
        F.trim(F.coalesce(F.col("country"), F.lit(""))).alias("country"),
        F.coalesce(F.col("business_name"), F.lit("")).alias("business_name"),
        F.coalesce(F.col("business_address"), F.lit("")).alias("business_address"),
        F.col("_prepared.norm_name").alias("norm_name"),
        F.col("_prepared.norm_address").alias("norm_address"),
        F.col("_prepared.roman_name").alias("roman_name"),
        F.col("_prepared.roman_address").alias("roman_address"),
        F.col("_prepared.name_tokens").alias("name_tokens"),
        F.col("_prepared.address_tokens").alias("address_tokens"),
        F.col("_prepared.roman_name_tokens").alias("roman_name_tokens"),
        F.col("_prepared.roman_address_tokens").alias("roman_address_tokens"),
        F.col("_prepared.numbers").alias("numbers"),
        F.col("_prepared.postal").alias("postal"),
        F.col("_prepared.block_keys").alias("block_keys"),
    ).filter(F.col("entity_id") != "")
    if source_column:
        selected = selected.withColumn("target_source", F.lit(source_column))
    return selected


def read_source(spark: SparkSession, path: str, source_column: str | None = None) -> DataFrame:
    return prepare_records(read_tsv(spark, path), source_column)


def _key_type_limit() -> F.Column:
    return (
        F.when(F.col("kind").isin(
            "exact_name", "exact_address", "roman_exact_name", "roman_exact_address",
        ), F.lit(64))
        .when(F.col("kind").isin("postal_house", "roman_postal_house"), F.lit(32))
        .when(F.col("kind").isin(
            "compact_name", "name_pair", "name_house",
            "roman_compact_name", "roman_name_pair", "roman_name_house",
        ), F.lit(220))
        .when(F.col("kind").isin("name_place", "roman_name_place"), F.lit(180))
        .when(F.col("kind").isin("address_pair", "roman_address_pair"), F.lit(160))
        .when(F.col("kind").isin("house_address", "roman_house_address"), F.lit(120))
        .when(F.col("kind").isin("name_anchor", "roman_name_anchor"), F.lit(48))
        .when(F.col("kind").isin(
            "name_token", "address_token", "roman_name_token", "roman_address_token",
        ), F.lit(32))
        .when(F.col("kind").isin(
            "name_gram", "address_gram", "roman_name_gram", "roman_address_gram",
        ), F.lit(16))
        .otherwise(F.lit(16))
    )


def _query_key_limit() -> F.Column:
    return (
        F.when(F.col("kind").isin(
        "name_pair", "address_pair", "roman_name_pair", "roman_address_pair",
        ), F.lit(2))
        .when(F.col("kind").isin(
            "exact_name", "exact_address", "postal_house", "compact_name", "name_anchor",
            "name_house", "name_place", "house_address",
            "roman_exact_name", "roman_exact_address", "roman_postal_house",
            "roman_compact_name", "roman_name_anchor", "roman_name_house",
            "roman_name_place", "roman_house_address",
        ), F.lit(1))
        .when(F.col("kind").isin(
            "name_token", "address_token", "roman_name_token", "roman_address_token",
        ), F.lit(2))
        .otherwise(F.lit(2))
    )


def generate_candidates(queries: DataFrame, targets: DataFrame) -> DataFrame:
    """Use frequency-pruned inverted blocks, then retain the top 32 per target source."""
    query_keys = (
        queries.select(
            "entity_id", "country", F.explode("block_keys").alias("block_key")
        )
        .select(
            "entity_id", "country", F.col("block_key.kind").alias("kind"),
            F.col("block_key.value").alias("value"),
        )
    )
    target_keys = (
        targets.select(
            "entity_id", "country", "target_source", F.explode("block_keys").alias("block_key")
        )
        .select(
            "entity_id", "country", "target_source",
            F.col("block_key.kind").alias("kind"), F.col("block_key.value").alias("value"),
        )
    )
    key_counts = target_keys.groupBy("country", "target_source", "kind", "value").agg(
        F.countDistinct("entity_id").alias("posting_df")
    ).withColumn("posting_limit", _key_type_limit())
    eligible_keys = key_counts.filter(F.col("posting_df") <= F.col("posting_limit"))

    query_sources = queries.select("entity_id").crossJoin(
        queries.sparkSession.createDataFrame([("S2",), ("S3",)], ["target_source"])
    )
    query_by_source = (
        query_keys.join(query_sources, "entity_id", "inner")
        .join(eligible_keys, ["country", "target_source", "kind", "value"], "inner")
        .withColumn("query_key_limit", _query_key_limit())
    )
    key_window = Window.partitionBy("entity_id", "target_source", "kind").orderBy(
        F.asc("posting_df"), F.asc("value")
    )
    query_by_source = (
        query_by_source.withColumn("key_rank", F.row_number().over(key_window))
        .filter(F.col("key_rank") <= F.col("query_key_limit"))
        .select("entity_id", "country", "target_source", "kind", "value", "posting_df")
    )
    target_posts = target_keys.join(
        eligible_keys.select("country", "target_source", "kind", "value", "posting_df"),
        ["country", "target_source", "kind", "value"], "inner",
    ).select(
        "country", "target_source", "kind", "value", "posting_df",
        F.col("entity_id").alias("target_id"),
    )
    hits = query_by_source.join(
        target_posts, ["country", "target_source", "kind", "value", "posting_df"], "inner"
    ).select(
        F.col("entity_id").alias("source1_id"), "country", "target_source", "target_id",
        "kind", "posting_df",
    )
    weighted = hits.withColumn(
        "rarity_weight",
        F.when(F.col("kind").isin(
            "exact_name", "exact_address", "roman_exact_name", "roman_exact_address",
        ), F.lit(8.0))
        .when(F.col("kind").isin("postal_house", "roman_postal_house"), F.lit(5.0))
        .when(F.col("kind").isin(
            "compact_name", "name_pair", "name_house",
            "roman_compact_name", "roman_name_pair", "roman_name_house",
        ), F.lit(4.0))
        .when(F.col("kind").isin(
            "name_place", "address_pair", "house_address",
            "roman_name_place", "roman_address_pair", "roman_house_address",
        ), F.lit(3.0))
        .when(F.col("kind").isin("name_anchor", "roman_name_anchor"), F.lit(1.7))
        .when(F.col("kind").isin(
            "name_token", "address_token", "roman_name_token", "roman_address_token",
        ), F.lit(1.0))
        .otherwise(F.lit(0.45))
        / F.log(F.col("posting_df").cast("double") + F.lit(1.0)),
    )
    evidence = weighted.groupBy("source1_id", "country", "target_source", "target_id").agg(
        F.max(F.when(F.col("kind") == "exact_name", 1).otherwise(0)).alias("exact_name_hit"),
        F.max(F.when(F.col("kind") == "exact_address", 1).otherwise(0)).alias("exact_address_hit"),
        F.max(F.when(F.col("kind") == "roman_exact_name", 1).otherwise(0)).alias("roman_exact_name_hit"),
        F.max(F.when(F.col("kind") == "roman_exact_address", 1).otherwise(0)).alias("roman_exact_address_hit"),
        F.max(F.when(F.col("kind").isin("postal_house", "roman_postal_house"), 1).otherwise(0))
        .alias("postal_house_hit"),
        F.sum(F.when(F.col("kind").isin("name_token", "roman_name_token"), 1).otherwise(0)).alias("name_token_hits"),
        F.sum(F.when(F.col("kind").isin("address_token", "roman_address_token"), 1).otherwise(0)).alias("address_token_hits"),
        F.sum(F.when(F.col("kind").isin("name_gram", "roman_name_gram"), 1).otherwise(0)).alias("name_gram_hits"),
        F.sum(F.when(F.col("kind").isin("address_gram", "roman_address_gram"), 1).otherwise(0)).alias("address_gram_hits"),
        F.sum(F.when(F.col("kind").isin(
            "compact_name", "name_anchor", "name_pair", "name_house", "name_place",
            "address_pair", "house_address", "roman_compact_name", "roman_name_anchor",
            "roman_name_pair", "roman_name_house", "roman_name_place", "roman_address_pair",
            "roman_house_address",
        ), 1).otherwise(0)).alias("composite_key_hits"),
        F.sum(F.when(F.col("kind").isin(
            "compact_name", "name_anchor", "name_pair", "name_house", "name_place",
            "roman_compact_name", "roman_name_anchor", "roman_name_pair", "roman_name_house",
            "roman_name_place",
        ), 1).otherwise(0)).alias("name_composite_hits"),
        F.sum(F.when(F.col("kind").isin(
            "address_pair", "house_address", "roman_address_pair", "roman_house_address",
        ), 1).otherwise(0))
        .alias("address_composite_hits"),
        F.sum("rarity_weight").alias("rarity_score"),
    ).withColumn(
        "retrieval_score",
        12.0 * F.col("exact_name_hit") + 12.0 * F.col("exact_address_hit")
        + 12.0 * F.col("roman_exact_name_hit") + 12.0 * F.col("roman_exact_address_hit")
        + 6.0 * F.col("postal_house_hit") + 0.75 * F.col("name_token_hits")
        + 0.55 * F.col("address_token_hits") + 0.20 * F.col("name_gram_hits")
        + 0.15 * F.col("address_gram_hits")
        + 1.25 * F.col("name_composite_hits")
        + 1.0 * F.col("address_composite_hits") + F.col("rarity_score"),
    )
    candidate_window = Window.partitionBy("source1_id", "target_source").orderBy(
        F.desc("retrieval_score"), F.desc("exact_name_hit"), F.desc("exact_address_hit"),
        F.asc("target_id"),
    )
    return (
        evidence.withColumn("candidate_rank", F.row_number().over(candidate_window))
        .filter(F.col("candidate_rank") <= MAX_CANDIDATES_PER_SOURCE)
    )


def build_pair_features(
    queries: DataFrame, targets: DataFrame, candidates: DataFrame,
) -> DataFrame:
    query_fields = queries.select(
        F.col("entity_id").alias("source1_id"), "country",
        F.col("norm_name").alias("query_name"),
        F.col("norm_address").alias("query_address"),
        F.col("roman_name").alias("query_roman_name"),
        F.col("roman_address").alias("query_roman_address"),
        F.col("name_tokens").alias("query_name_tokens"),
        F.col("address_tokens").alias("query_address_tokens"),
        F.col("roman_name_tokens").alias("query_roman_name_tokens"),
        F.col("roman_address_tokens").alias("query_roman_address_tokens"),
        F.col("numbers").alias("query_numbers"), F.col("postal").alias("query_postal"),
    )
    target_fields = targets.select(
        F.col("entity_id").alias("target_id"), "country", "target_source",
        F.col("norm_name").alias("target_name"),
        F.col("norm_address").alias("target_address"),
        F.col("roman_name").alias("target_roman_name"),
        F.col("roman_address").alias("target_roman_address"),
        F.col("name_tokens").alias("target_name_tokens"),
        F.col("address_tokens").alias("target_address_tokens"),
        F.col("roman_name_tokens").alias("target_roman_name_tokens"),
        F.col("roman_address_tokens").alias("target_roman_address_tokens"),
        F.col("numbers").alias("target_numbers"), F.col("postal").alias("target_postal"),
    )
    pairs = candidates.join(query_fields, ["source1_id", "country"], "inner").join(
        target_fields, ["country", "target_source", "target_id"], "inner"
    )
    name_max_len = F.greatest(F.length("query_name"), F.length("target_name"))
    address_max_len = F.greatest(F.length("query_address"), F.length("target_address"))
    roman_name_max_len = F.greatest(F.length("query_roman_name"), F.length("target_roman_name"))
    roman_address_max_len = F.greatest(
        F.length("query_roman_address"), F.length("target_roman_address")
    )
    name_intersection = F.size(F.array_intersect("query_name_tokens", "target_name_tokens"))
    name_union = F.size(F.array_union("query_name_tokens", "target_name_tokens"))
    address_intersection = F.size(F.array_intersect("query_address_tokens", "target_address_tokens"))
    address_union = F.size(F.array_union("query_address_tokens", "target_address_tokens"))
    number_intersection = F.size(F.array_intersect("query_numbers", "target_numbers"))
    number_union = F.size(F.array_union("query_numbers", "target_numbers"))
    roman_name_intersection = F.size(F.array_intersect(
        "query_roman_name_tokens", "target_roman_name_tokens",
    ))
    roman_name_union = F.size(F.array_union(
        "query_roman_name_tokens", "target_roman_name_tokens",
    ))
    roman_address_intersection = F.size(F.array_intersect(
        "query_roman_address_tokens", "target_roman_address_tokens",
    ))
    roman_address_union = F.size(F.array_union(
        "query_roman_address_tokens", "target_roman_address_tokens",
    ))
    features = pairs
    feature_expressions = {
        "name_edit_similarity": F.when(name_max_len == 0, 0.0).otherwise(
            1.0 - F.levenshtein("query_name", "target_name") / name_max_len
        ),
        "name_token_jaccard": name_intersection / F.greatest(name_union, F.lit(1)),
        "name_token_containment": name_intersection / F.greatest(
            F.least(F.size("query_name_tokens"), F.size("target_name_tokens")), F.lit(1)
        ),
        "address_edit_similarity": F.when(address_max_len == 0, 0.0).otherwise(
            1.0 - F.levenshtein("query_address", "target_address") / address_max_len
        ),
        "address_token_jaccard": address_intersection / F.greatest(address_union, F.lit(1)),
        "address_token_containment": address_intersection / F.greatest(
            F.least(F.size("query_address_tokens"), F.size("target_address_tokens")), F.lit(1)
        ),
        "number_jaccard": number_intersection / F.greatest(number_union, F.lit(1)),
        "postal_equal": F.when(
            (F.col("query_postal") != "") & (F.col("query_postal") == F.col("target_postal")), 1.0
        ).otherwise(0.0),
        "exact_name": F.when(
            (F.col("query_name") != "") & (F.col("query_name") == F.col("target_name")), 1.0
        ).otherwise(0.0),
        "exact_address": F.when(
            (F.col("query_address") != "") & (F.col("query_address") == F.col("target_address")), 1.0
        ).otherwise(0.0),
        "name_length_ratio": F.least(F.length("query_name"), F.length("target_name"))
        / F.greatest(name_max_len, F.lit(1)),
        "address_length_ratio": F.least(F.length("query_address"), F.length("target_address"))
        / F.greatest(address_max_len, F.lit(1)),
        "house_number_equal": F.when(
            (F.size("query_numbers") > 0) & (F.size("target_numbers") > 0)
            & (F.element_at("query_numbers", 1) == F.element_at("target_numbers", 1)), 1.0
        ).otherwise(0.0),
        "candidate_rank_reciprocal": 1.0 / F.greatest(
            F.col("candidate_rank").cast("double"), F.lit(1.0)
        ),
        "roman_name_edit_similarity": F.when(roman_name_max_len == 0, 0.0).otherwise(
            1.0 - F.levenshtein("query_roman_name", "target_roman_name") / roman_name_max_len
        ),
        "roman_name_token_jaccard": roman_name_intersection / F.greatest(
            roman_name_union, F.lit(1)
        ),
        "roman_address_edit_similarity": F.when(roman_address_max_len == 0, 0.0).otherwise(
            1.0 - F.levenshtein("query_roman_address", "target_roman_address")
            / roman_address_max_len
        ),
        "roman_address_token_jaccard": roman_address_intersection / F.greatest(
            roman_address_union, F.lit(1)
        ),
        "roman_exact_name": F.when(
            (F.col("query_roman_name") != "")
            & (F.col("query_roman_name") == F.col("target_roman_name")), 1.0
        ).otherwise(0.0),
        "roman_exact_address": F.when(
            (F.col("query_roman_address") != "")
            & (F.col("query_roman_address") == F.col("target_roman_address")), 1.0
        ).otherwise(0.0),
    }
    for name, expression in feature_expressions.items():
        features = features.withColumn(name, expression.cast("double"))
    features = (
        features.withColumn("target_is_s2", F.when(F.col("target_source") == "S2", 1.0).otherwise(0.0))
        .withColumn("query_address_missing", F.when(F.col("query_address") == "", 1.0).otherwise(0.0))
        .withColumn("target_address_missing", F.when(F.col("target_address") == "", 1.0).otherwise(0.0))
        .withColumn("composite_key_hits", F.col("composite_key_hits").cast("double"))
        .withColumn("name_composite_hits", F.col("name_composite_hits").cast("double"))
        .withColumn("address_composite_hits", F.col("address_composite_hits").cast("double"))
    )
    return features.select(*PAIR_FEATURE_COLUMNS)


def truth_pairs(spark: SparkSession, path: str) -> DataFrame:
    raw = read_tsv(spark, path)
    return (
        raw.select(
            F.trim(F.col("source1_entity_id")).alias("source1_id"),
            F.explode(F.split(F.coalesce(F.col("matched_entity_ids"), F.lit("")), ",")).alias("target_id"),
        )
        .select("source1_id", F.trim("target_id").alias("target_id"))
        .filter((F.col("source1_id") != "") & (F.col("target_id") != ""))
        .dropDuplicates(["source1_id", "target_id"])
    )


EVAL_SCHEMA = ArrayType(StructType([
    StructField("cap", IntegerType(), False),
    StructField("threshold", DoubleType(), False),
    StructField("score", DoubleType(), False),
    StructField("candidate_hits", LongType(), False),
    StructField("truth_count", LongType(), False),
]))
CALIBRATION_SCHEMA = ArrayType(StructType([
    StructField("threshold", DoubleType(), False),
    StructField("score", DoubleType(), False),
]))


def _evaluate_query(candidates: list[Any] | None, truth_count: int | None) -> list[tuple[Any, ...]]:
    truth_n = int(truth_count or 0)
    usable = [row for row in (candidates or []) if row and row["candidate_rank"] is not None]
    output: list[tuple[Any, ...]] = []
    for cap in CAPS:
        selected = [row for row in usable if int(row["candidate_rank"]) <= cap]
        hits = sum(int(row["label"]) for row in selected)
        for threshold in THRESHOLDS:
            predicted = [row for row in selected if float(row["probability"]) >= threshold]
            tp = sum(int(row["label"]) for row in predicted)
            if truth_n == 0 and not predicted:
                score = 1.0
            elif not predicted or truth_n == 0:
                score = 0.0
            else:
                precision = tp / len(predicted)
                recall = tp / truth_n
                denom = 0.25 * precision + recall
                score = 1.25 * precision * recall / denom if denom else 0.0
            output.append((cap, threshold, score, hits, truth_n))
    return output


def evaluate_validation(
    spark: SparkSession, validation_queries: DataFrame, scored: DataFrame,
    validation_truth: DataFrame,
) -> tuple[list[dict[str, Any]], float]:
    truth_counts = validation_truth.groupBy("source1_id").agg(
        F.countDistinct("target_id").alias("truth_count")
    )
    candidate_groups = scored.groupBy("source1_id").agg(
        F.collect_list(F.struct("candidate_rank", "probability", "label")).alias("candidate_rows")
    )
    per_query = (
        validation_queries.select(F.col("entity_id").alias("source1_id")).join(
            truth_counts, "source1_id", "left"
        ).join(candidate_groups, "source1_id", "left")
        .withColumn("truth_count", F.coalesce(F.col("truth_count"), F.lit(0)))
    )
    eval_udf = F.udf(_evaluate_query, EVAL_SCHEMA)
    expanded = per_query.withColumn(
        "evaluation", F.explode(eval_udf("candidate_rows", "truth_count"))
    ).select(
        F.col("evaluation.cap").alias("cap_per_source"),
        F.col("evaluation.threshold").alias("threshold"),
        F.col("evaluation.score").alias("macro_item_score"),
        F.col("evaluation.candidate_hits").alias("candidate_hits"),
        F.col("evaluation.truth_count").alias("truth_count"),
    )
    summary = expanded.groupBy("cap_per_source", "threshold").agg(
        F.avg("macro_item_score").alias("macro_f0_5"),
        F.sum("candidate_hits").alias("candidate_hits"),
        F.sum("truth_count").alias("truth_links"),
    ).withColumn(
        "candidate_recall",
        F.when(F.col("truth_links") > 0, F.col("candidate_hits") / F.col("truth_links"))
        .otherwise(F.lit(0.0)),
    )
    rows = [row.asDict() for row in summary.collect()]
    by_cap: dict[int, dict[str, Any]] = {}
    for cap in CAPS:
        options = [row for row in rows if int(row["cap_per_source"]) == cap]
        if not options:
            continue
        best = min(options, key=lambda row: (-float(row["macro_f0_5"]), float(row["threshold"])))
        by_cap[cap] = {
            "cap_per_source": cap,
            "candidate_recall": float(best["candidate_recall"]),
            "macro_f0_5": float(best["macro_f0_5"]),
            "threshold": float(best["threshold"]),
        }
    return list(by_cap.values()), float(max(THRESHOLDS))


def _calibrate_query_thresholds(
    candidates: list[Any] | None, truth_count: int | None, cap: int,
    fixed_s2: float, fixed_s3: float, varying_source: str,
) -> list[tuple[float, float]]:
    truth_n = int(truth_count or 0)
    usable = [row for row in (candidates or [])
              if row and row["candidate_rank"] is not None
              and int(row["candidate_rank"]) <= cap]
    output: list[tuple[float, float]] = []
    for threshold in THRESHOLDS:
        predicted = []
        for row in usable:
            source = str(row["target_source"])
            cutoff = threshold if source == varying_source else (fixed_s2 if source == "S2" else fixed_s3)
            if float(row["probability"]) >= cutoff:
                predicted.append(row)
        tp = sum(int(row["label"]) for row in predicted)
        if truth_n == 0 and not predicted:
            score = 1.0
        elif not predicted or truth_n == 0:
            score = 0.0
        else:
            precision = tp / len(predicted)
            recall = tp / truth_n
            denominator = 0.25 * precision + recall
            score = 1.25 * precision * recall / denominator if denominator else 0.0
        output.append((float(threshold), float(score)))
    return output


def calibrate_source_threshold(
    validation_queries: DataFrame, scored: DataFrame, validation_truth: DataFrame,
    cap: int, fixed_s2: float, fixed_s3: float, varying_source: str,
) -> tuple[float, float]:
    truth_counts = validation_truth.groupBy("source1_id").agg(
        F.countDistinct("target_id").alias("truth_count")
    )
    candidate_groups = scored.groupBy("source1_id").agg(
        F.collect_list(F.struct(
            "candidate_rank", "probability", "label", "target_source",
        )).alias("candidate_rows")
    )
    per_query = (
        validation_queries.select(F.col("entity_id").alias("source1_id"))
        .join(truth_counts, "source1_id", "left")
        .join(candidate_groups, "source1_id", "left")
        .withColumn("truth_count", F.coalesce(F.col("truth_count"), F.lit(0)))
    )
    evaluate = F.udf(_calibrate_query_thresholds, CALIBRATION_SCHEMA)
    expanded = per_query.withColumn(
        "evaluation", F.explode(evaluate(
            "candidate_rows", "truth_count", F.lit(cap), F.lit(fixed_s2),
            F.lit(fixed_s3), F.lit(varying_source),
        ))
    ).select(
        F.col("evaluation.threshold").alias("threshold"),
        F.col("evaluation.score").alias("macro_item_score"),
    )
    options = expanded.groupBy("threshold").agg(
        F.avg("macro_item_score").alias("macro_f0_5")
    ).collect()
    if not options:
        raise RuntimeError(f"Could not calibrate a validation threshold for {varying_source}.")
    best = max(options, key=lambda row: (float(row["macro_f0_5"]), float(row["threshold"])))
    return float(best["threshold"]), float(best["macro_f0_5"])


def calibrate_source_thresholds(
    validation_queries: DataFrame, scored: DataFrame, validation_truth: DataFrame,
    cap: int, initial_threshold: float,
) -> tuple[dict[str, float], float]:
    thresholds = {"S2": initial_threshold, "S3": initial_threshold}
    score = 0.0
    for _ in range(2):
        for source in ("S2", "S3"):
            thresholds[source], score = calibrate_source_threshold(
                validation_queries, scored, validation_truth, cap,
                thresholds["S2"], thresholds["S3"], source,
            )
    return thresholds, score


def score_source1_queries(
    queries: DataFrame, scored: DataFrame, truth: DataFrame, cap: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Report exact per-query macro F0.5 at fixed thresholds, without tuning here."""
    threshold_column = F.when(
        F.col("target_source") == "S2", F.lit(thresholds["S2"])
    ).otherwise(F.lit(thresholds["S3"]))
    selected = scored.filter(F.col("candidate_rank") <= cap).withColumn(
        "predicted", F.col("probability") >= threshold_column
    )
    candidate_hits = selected.groupBy("source1_id").agg(
        F.sum(F.when(F.col("label") == 1, 1).otherwise(0)).alias("candidate_hits")
    )
    target_window = Window.partitionBy("target_id").orderBy(
        F.desc("probability"), F.asc("candidate_rank"), F.asc("source1_id"),
    )
    exclusive_predictions = (
        selected.filter(F.col("predicted"))
        .withColumn("target_assignment_rank", F.row_number().over(target_window))
        .filter(F.col("target_assignment_rank") == 1)
    )
    pair_counts = exclusive_predictions.groupBy("source1_id").agg(
        F.sum(F.when(F.col("label") == 1, 1).otherwise(0)).alias("tp"),
        F.count(F.lit(1)).alias("predicted_count"),
    )
    truth_counts = truth.groupBy("source1_id").agg(
        F.countDistinct("target_id").alias("truth_count")
    )
    per_query = (
        queries.select(F.col("entity_id").alias("source1_id"))
        .join(truth_counts, "source1_id", "left")
        .join(candidate_hits, "source1_id", "left")
        .join(pair_counts, "source1_id", "left")
        .fillna({"truth_count": 0, "tp": 0, "predicted_count": 0, "candidate_hits": 0})
        .withColumn("truth_count", F.col("truth_count").cast("long"))
        .withColumn("tp", F.col("tp").cast("long"))
        .withColumn("predicted_count", F.col("predicted_count").cast("long"))
        .withColumn("candidate_hits", F.col("candidate_hits").cast("long"))
        .withColumn(
            "item_score",
            F.when((F.col("truth_count") == 0) & (F.col("predicted_count") == 0), 1.0)
            .when((F.col("truth_count") == 0) | (F.col("predicted_count") == 0), 0.0)
            .otherwise(1.25 * F.col("tp") /
                       (F.col("predicted_count") + 0.25 * F.col("truth_count"))),
        )
    )
    summary = per_query.agg(
        F.avg("item_score").alias("macro_f0_5"),
        F.sum("tp").alias("true_positives"),
        F.sum("predicted_count").alias("predicted_pairs"),
        F.sum("truth_count").alias("truth_links"),
        F.sum("candidate_hits").alias("candidate_hits"),
    ).first().asDict()
    truth_links = int(summary["truth_links"] or 0)
    tp = int(summary["true_positives"] or 0)
    predicted = int(summary["predicted_pairs"] or 0)
    return {
        "macro_f0_5": float(summary["macro_f0_5"] or 0.0),
        "candidate_recall": float(summary["candidate_hits"] or 0) / truth_links
        if truth_links else 0.0,
        "true_positives": tp,
        "false_positives": max(0, predicted - tp),
        "false_negatives": max(0, truth_links - tp),
        "missed_by_blocking": max(0, truth_links - int(summary["candidate_hits"] or 0)),
        "false_negatives_within_candidate_set": max(
            0, int(summary["candidate_hits"] or 0) - tp,
        ),
        "truth_links": truth_links,
        "predicted_pairs": predicted,
    }


def fit_pair_model(model_rows: DataFrame) -> lgb.Booster:
    """Fit the CPU pair classifier on AWS from bounded candidate-only features."""
    training_frame = model_rows.select(*FEATURE_NAMES, "label").toPandas()
    features = training_frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    labels = training_frame["label"].to_numpy(dtype=np.float32)
    LOG.info("Training LightGBM on %s pairs with %s features", len(labels), features.shape[1])
    del training_frame
    if len(np.unique(labels)) < 2:
        raise RuntimeError("LightGBM pair training requires both positive and negative examples")
    training_set = lgb.Dataset(features, label=labels, feature_name=list(FEATURE_NAMES))
    params = {
        "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
        "num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 50,
        "feature_fraction": 0.9, "bagging_fraction": 0.85, "bagging_freq": 1,
        "lambda_l2": 2.0, "max_bin": 63, "verbosity": -1,
        "num_threads": max(1, min(8, os.cpu_count() or 1)),
        "seed": 20260926, "feature_fraction_seed": 20260927,
        "bagging_seed": 20260928, "deterministic": True,
    }
    return lgb.train(params, training_set, num_boost_round=700)


def score_pair_rows(rows: DataFrame, model: lgb.Booster) -> DataFrame:
    """Score candidate rows in Arrow batches so the full pair table stays distributed."""
    result_schema = StructType(
        list(rows.schema.fields) + [StructField("probability", DoubleType(), False)]
    )
    feature_names = list(FEATURE_NAMES)

    def predict_batches(frames):
        for frame in frames:
            values = frame.loc[:, feature_names].to_numpy(dtype=np.float32)
            frame["probability"] = model.predict(values, num_threads=1).astype(np.float64)
            yield frame

    return rows.mapInPandas(predict_batches, schema=result_schema)


def write_tsv(data: DataFrame, path: str) -> None:
    (
        data.repartition(16)
        .write.mode("overwrite")
        .option("header", "true")
        .option("sep", "\t")
        .option("encoding", "UTF-8")
        .option("emptyValue", "")
        .csv(path)
    )


def write_json(spark: SparkSession, payload: dict[str, Any], path: str) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    spark.createDataFrame([(text,)], ["value"]).coalesce(1).write.mode("overwrite").text(path)


def assert_no_rows(data: DataFrame, label: str) -> None:
    if data.limit(1).count():
        raise RuntimeError(f"Output contract check failed: {label}")


def run(args: argparse.Namespace) -> None:
    started = time.time()
    spark = SparkSession.builder.appName("AmazonMLEntityResolution").getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", "200")
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

    train_prefix = spark_s3_uri(args.train_prefix.rstrip("/"), scheme=args.s3_scheme)
    test_prefix = spark_s3_uri(args.test_prefix.rstrip("/"), scheme=args.s3_scheme)
    output_prefix = spark_s3_uri(args.output_prefix.rstrip("/"), scheme=args.s3_scheme)
    training_s1 = read_source(spark, f"{train_prefix}/train_source1.tsv")
    training_s2 = read_source(spark, f"{train_prefix}/train_source2.tsv", "S2")
    training_s3 = read_source(spark, f"{train_prefix}/train_source3.tsv", "S3")
    training_targets = training_s2.unionByName(training_s3)
    ground_truth = truth_pairs(spark, f"{train_prefix}/train_ground_truth.tsv")

    split_modulus = 5 if args.smoke_test else 100
    train_with_split = training_s1.withColumn(
        "split_bucket", F.pmod(F.xxhash64("entity_id"), F.lit(split_modulus))
    )
    if args.smoke_test:
        training_queries = train_with_split.filter(F.col("split_bucket") == 0).drop("split_bucket")
        validation_queries = train_with_split.filter(F.col("split_bucket") == 1).drop("split_bucket")
        audit_queries = train_with_split.filter(F.col("split_bucket") == 2).drop("split_bucket")
    else:
        training_queries = train_with_split.filter(F.col("split_bucket") < 5).drop("split_bucket")
        validation_queries = train_with_split.filter(
            (F.col("split_bucket") >= 5) & (F.col("split_bucket") < 10)
        ).drop("split_bucket")
        audit_queries = train_with_split.filter(
            (F.col("split_bucket") >= 10) & (F.col("split_bucket") < 15)
        ).drop("split_bucket")
    fitting_and_validation = training_queries.unionByName(validation_queries)
    training_candidates = generate_candidates(fitting_and_validation, training_targets)
    training_features = build_pair_features(
        fitting_and_validation, training_targets, training_candidates,
    ).persist(StorageLevel.MEMORY_AND_DISK)
    labeled = training_features.join(ground_truth.withColumn("label", F.lit(1)),
                                    ["source1_id", "target_id"], "left").withColumn(
        "label", F.coalesce(F.col("label"), F.lit(0)).cast("double")
    )

    negative_window = Window.partitionBy("source1_id", "target_source").orderBy(
        F.desc("retrieval_score"), F.asc("target_id")
    )
    fitting_query_ids = training_queries.select(F.col("entity_id").alias("source1_id"))
    fitting_rows = labeled.join(fitting_query_ids, "source1_id", "inner")
    negatives = fitting_rows.filter(F.col("label") == 0).withColumn(
        "negative_rank", F.row_number().over(negative_window)
    ).filter(F.col("negative_rank") <= 8)
    model_rows = fitting_rows.filter(F.col("label") == 1).unionByName(
        negatives.drop("negative_rank"), allowMissingColumns=True
    )
    class_counts = {int(row["label"]): int(row["count"]) for row in
                    model_rows.groupBy("label").count().collect()}
    if class_counts.get(0, 0) == 0 or class_counts.get(1, 0) == 0:
        raise RuntimeError(f"Candidate training requires both classes; observed {class_counts}")

    model = fit_pair_model(model_rows)
    if not output_prefix.startswith(("s3://", "s3a://")):
        os.makedirs(output_prefix, exist_ok=True)
        model.save_model(os.path.join(output_prefix, "model.txt"))

    validation_ids = validation_queries.select(F.col("entity_id").alias("source1_id"))
    validation_truth = ground_truth.join(validation_ids, "source1_id", "inner")
    validation_features = training_features.join(validation_ids, "source1_id", "inner")
    validation_labeled = validation_features.join(ground_truth.withColumn("label", F.lit(1)),
                                                   ["source1_id", "target_id"], "left").withColumn(
        "label", F.coalesce(F.col("label"), F.lit(0)).cast("double")
    )
    validation_scored = score_pair_rows(validation_labeled, model).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    cap_metrics, _ = evaluate_validation(spark, validation_queries, validation_scored, validation_truth)
    training_features.unpersist(blocking=False)
    if len(cap_metrics) != len(CAPS):
        raise RuntimeError(f"Validation did not produce metrics for all caps: {cap_metrics}")
    operating = select_operating_point(cap_metrics)
    cap = int(operating["cap_per_source"])
    global_threshold = float(operating["threshold"])
    source_thresholds, _calibrated_validation_score = calibrate_source_thresholds(
        validation_queries, validation_scored, validation_truth, cap, global_threshold,
    )
    calibration_metrics = score_source1_queries(
        validation_queries, validation_scored, validation_truth, cap, source_thresholds,
    )
    audit_ids = audit_queries.select(F.col("entity_id").alias("source1_id"))
    audit_truth = ground_truth.join(audit_ids, "source1_id", "inner")
    audit_candidates = generate_candidates(audit_queries, training_targets)
    audit_features = build_pair_features(audit_queries, training_targets, audit_candidates)
    audit_labeled = audit_features.join(ground_truth.withColumn("label", F.lit(1)),
                                        ["source1_id", "target_id"], "left").withColumn(
        "label", F.coalesce(F.col("label"), F.lit(0)).cast("double")
    )
    audit_scored = score_pair_rows(audit_labeled, model).persist(StorageLevel.MEMORY_AND_DISK)
    audit_metrics = score_source1_queries(
        audit_queries, audit_scored, audit_truth, cap, source_thresholds,
    )
    audit_scored.unpersist(blocking=False)

    test_s1 = read_source(spark, f"{test_prefix}/test_source1.tsv")
    test_s2 = read_source(spark, f"{test_prefix}/test_source2.tsv", "S2")
    test_s3 = read_source(spark, f"{test_prefix}/test_source3.tsv", "S3")
    test_targets = test_s2.unionByName(test_s3)
    test_candidates = generate_candidates(test_s1, test_targets)
    test_features = build_pair_features(test_s1, test_targets, test_candidates)
    test_scored = score_pair_rows(test_features, model).persist(StorageLevel.MEMORY_AND_DISK)
    selected = test_scored.filter(F.col("candidate_rank") <= cap)
    threshold_column = F.when(
        F.col("target_source") == "S2", F.lit(source_thresholds["S2"])
    ).otherwise(F.lit(source_thresholds["S3"]))

    global_order = Window.partitionBy("source1_id").orderBy(
        F.desc("retrieval_score"), F.asc("target_source"), F.asc("target_id")
    )
    selected_ordered = selected.withColumn("output_order", F.row_number().over(global_order))
    candidate_lists = selected_ordered.groupBy("source1_id").agg(
        F.sort_array(F.collect_list(F.struct("output_order", "target_id"))).alias("items")
    ).select(
        "source1_id",
        F.expr("concat_ws(',', transform(items, x -> x.target_id))").alias("candidate_entity_ids"),
    )
    all_source1 = test_s1.select(F.col("entity_id").alias("source1_id"))
    candidate_output = all_source1.join(candidate_lists, "source1_id", "left").select(
        F.col("source1_id").alias("source1_entity_id"),
        F.coalesce(F.col("candidate_entity_ids"), F.lit("")).alias("candidate_entity_ids"),
    )

    target_assignment_window = Window.partitionBy("target_id").orderBy(
        F.desc("probability"), F.asc("candidate_rank"), F.asc("source1_id"),
    )
    matches = (
        selected.filter(F.col("probability") >= threshold_column)
        .withColumn("target_assignment_rank", F.row_number().over(target_assignment_window))
        .filter(F.col("target_assignment_rank") == 1)
        .withColumn("negative_probability", -F.col("probability"))
    )
    match_lists = matches.groupBy("source1_id").agg(
        F.sort_array(F.collect_list(F.struct("negative_probability", "target_id"))).alias("items")
    ).select(
        "source1_id",
        F.expr("concat_ws(',', transform(items, x -> x.target_id))").alias("matched_entity_ids"),
    )
    matching_output = all_source1.join(match_lists, "source1_id", "left").select(
        F.col("source1_id").alias("source1_entity_id"),
        F.coalesce(F.col("matched_entity_ids"), F.lit("")).alias("matched_entity_ids"),
    )

    match_pair_ids = matches.select("source1_id", "target_id").dropDuplicates()
    candidate_pair_ids = selected.select("source1_id", "target_id").dropDuplicates()
    assert_no_rows(match_pair_ids.join(candidate_pair_ids, ["source1_id", "target_id"], "left_anti"),
                   "predicted match missing from candidate set")
    assert_no_rows(candidate_output.groupBy("source1_entity_id").count().filter(F.col("count") != 1),
                   "duplicate candidate output row")
    assert_no_rows(matching_output.groupBy("source1_entity_id").count().filter(F.col("count") != 1),
                   "duplicate matching output row")
    assert_no_rows(
        selected.groupBy("source1_id", "target_source").count().filter(F.col("count") > cap),
        "selected candidates exceed the per-source cap",
    )
    assert_no_rows(
        matches.groupBy("target_id").count().filter(F.col("count") > 1),
        "a target entity is assigned to more than one Source 1 entity",
    )

    total_s1 = test_s1.count()
    if candidate_output.count() != total_s1 or matching_output.count() != total_s1:
        raise RuntimeError("Submission outputs do not contain exactly one row per test S1 entity")
    candidate_total = selected.count()
    per_query_counts = all_source1.join(
        selected.groupBy("source1_id").count().withColumnRenamed("count", "candidate_count"),
        "source1_id", "left",
    ).fillna({"candidate_count": 0})
    distribution = per_query_counts.agg(
        F.avg("candidate_count").alias("mean"),
        F.percentile_approx("candidate_count", 0.5, 10000).alias("median"),
        F.percentile_approx("candidate_count", 0.95, 10000).alias("p95"),
        F.percentile_approx("candidate_count", 0.99, 10000).alias("p99"),
        F.max("candidate_count").alias("max"),
    ).first().asDict()
    s1_country_counts = test_s1.groupBy("country").count().withColumnRenamed("count", "s1_count")
    target_country_counts = test_targets.groupBy("country").count().withColumnRenamed("count", "target_count")
    possible_pairs = s1_country_counts.join(target_country_counts, "country", "inner").agg(
        F.sum(F.col("s1_count").cast("double") * F.col("target_count").cast("double")).alias("n")
    ).first()["n"] or 0.0
    reduction_ratio = 1.0 - candidate_total / possible_pairs if possible_pairs else 0.0
    country_counts = (
        test_s1.select(F.col("entity_id").alias("source1_id"), "country")
        .join(selected.groupBy("source1_id").count().withColumnRenamed("count", "candidate_count"),
              "source1_id", "left")
        .fillna({"candidate_count": 0})
        .groupBy("country")
        .agg(
            F.count("source1_id").alias("source1_rows"),
            F.sum("candidate_count").alias("candidate_pairs"),
            F.avg("candidate_count").alias("candidate_count_mean"),
            F.percentile_approx("candidate_count", 0.95, 10000).alias("candidate_count_p95"),
        )
        .collect()
    )
    validation_scored.unpersist(blocking=False)

    output_root = f"{output_prefix}/submission"
    write_tsv(candidate_output, f"{output_root}/candidate_pairs")
    write_tsv(matching_output, f"{output_root}/matching_results")
    metrics = {
        "runtime_seconds": round(time.time() - started, 2),
        "smoke_test": bool(args.smoke_test),
        "compute_service": args.compute_service,
        "instance_type": args.instance_type,
        "spark_version": spark.version,
        "model": "LightGBM CPU binary classifier",
        "model_parameters": {
            "num_boost_round": 700, "num_leaves": 31, "min_data_in_leaf": 50,
            "learning_rate": 0.05, "lambda_l2": 2.0, "seed": 20260926,
        },
        "feature_names": list(FEATURE_NAMES),
        "training_pair_counts": class_counts,
        "selected_cap_per_source": cap,
        "selected_thresholds": source_thresholds,
        "initial_global_threshold": global_threshold,
        "selected_candidate_recall": float(operating["candidate_recall"]),
        "validation_macro_f0_5": float(calibration_metrics["macro_f0_5"]),
        "calibration_source1_rows": int(validation_queries.count()),
        "independent_audit_macro_f0_5": float(audit_metrics["macro_f0_5"]),
        "independent_audit_candidate_recall": float(audit_metrics["candidate_recall"]),
        "independent_audit_source1_rows": int(audit_queries.count()),
        "independent_audit_pair_decisions": audit_metrics,
        "global_threshold_validation_macro_f0_5": float(operating["macro_f0_5"]),
        "validation_pair_decisions": calibration_metrics,
        "validation_cap_sweep": cap_metrics,
        "test_source1_rows": int(total_s1),
        "candidate_pairs": int(candidate_total),
        "candidate_count_mean": float(distribution["mean"] or 0.0),
        "candidate_count_median": int(distribution["median"] or 0),
        "candidate_count_p95": int(distribution["p95"] or 0),
        "candidate_count_p99": int(distribution["p99"] or 0),
        "candidate_count_max": int(distribution["max"] or 0),
        "possible_same_country_pairs": int(possible_pairs),
        "candidate_reduction_ratio": float(reduction_ratio),
        "test_candidate_sources_max": {"S2": cap, "S3": cap},
        "per_country": {row["country"]: {
            "source1_rows": int(row["source1_rows"]),
            "candidate_pairs": int(row["candidate_pairs"] or 0),
            "candidate_count_mean": float(row["candidate_count_mean"] or 0.0),
            "candidate_count_p95": int(row["candidate_count_p95"] or 0),
        } for row in country_counts},
    }
    write_json(spark, metrics, f"{output_prefix}/metrics")
    LOG.info("Entity resolution complete: %s", json.dumps(metrics, sort_keys=True))
    test_scored.unpersist(blocking=False)
    spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
