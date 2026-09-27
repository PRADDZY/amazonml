#!/usr/bin/env python3
"""Generate matching_results.tsv with fixed rules; this job never trains or fits."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import tempfile

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel

import sagemaker_spark_job as er


LOG = logging.getLogger("no_fit_submission")


def fixed_rule_matches(features):
    """Apply conservative, fixed name/address evidence rules to retrieved pairs."""
    n = F.greatest("name_edit_similarity", "roman_name_edit_similarity")
    nj = F.greatest("name_token_jaccard", "roman_name_token_jaccard")
    a = F.greatest("address_edit_similarity", "roman_address_edit_similarity")
    aj = F.greatest("address_token_jaccard", "roman_address_token_jaccard")
    exact_n = F.greatest("exact_name", "roman_exact_name")
    exact_a = F.greatest("exact_address", "roman_exact_address")
    address_support = (
        (F.col("postal_equal") > 0)
        | (F.col("house_number_equal") > 0)
        | (F.col("number_jaccard") >= 0.5)
    )

    confidence = (
        0.37 * n + 0.12 * nj + 0.27 * a + 0.10 * aj
        + 0.06 * F.col("house_number_equal")
        + 0.05 * F.col("postal_equal")
        + 0.18 * exact_n + 0.08 * exact_a
        + 0.04 * F.col("candidate_rank_reciprocal")
    )
    accepted = (
        ((exact_n >= 1) & ((a >= 0.25) | (aj >= 0.30) | (exact_a >= 1) | address_support))
        | ((n >= 0.94) & (nj >= 0.55)
           & ((a >= 0.45) | (aj >= 0.35) | (exact_a >= 1) | address_support))
        | ((n >= 0.87) & (nj >= 0.50) & ((a >= 0.70) | (aj >= 0.55)))
        | (((a >= 0.93) | (aj >= 0.78) | (exact_a >= 1)) & (n >= 0.75) & (nj >= 0.35))
    )
    return features.withColumn("rule_score", confidence).filter(accepted)


def write_one_tsv(frame, destination: str) -> None:
    """Write one flat TSV atomically from a single Spark part file."""
    destination = os.path.abspath(destination)
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(destination):
        raise FileExistsError(f"Refusing to overwrite an existing submission: {destination}")
    with tempfile.TemporaryDirectory(prefix=".matching_stage_", dir=parent) as stage_parent:
        temp_dir = os.path.join(stage_parent, "parts")
        (
            frame.coalesce(1)
            .write.mode("errorifexists")
            .option("header", "true")
            .option("sep", "\t")
            .option("encoding", "UTF-8")
            .option("emptyValue", "")
            .csv(temp_dir)
        )
        parts = glob.glob(os.path.join(temp_dir, "part-*.csv"))
        if len(parts) != 1:
            raise RuntimeError(f"Expected one Spark output part, found {len(parts)}")
        shutil.copyfile(parts[0], destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-prefix", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--parallelism", type=int, default=48)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spark = (
        SparkSession.builder.appName("AmazonMLNoFitSubmission")
        .config("spark.sql.shuffle.partitions", str(args.parallelism))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .getOrCreate()
    )

    root = args.test_prefix.rstrip("/")
    source1 = er.read_source(spark, f"{root}/test_source1.tsv")
    source2 = er.read_source(spark, f"{root}/test_source2.tsv", "S2")
    source3 = er.read_source(spark, f"{root}/test_source3.tsv", "S3")
    targets = source2.unionByName(source3)
    source1_ids = source1.select(F.col("entity_id").alias("source1_id"))
    LOG.info("Loaded test inputs; generating bounded blocking candidates")

    candidates = er.generate_candidates(source1, targets).persist(StorageLevel.MEMORY_AND_DISK)
    candidate_count = candidates.count()
    LOG.info("Generated %s candidate pairs", f"{candidate_count:,}")

    pair_features = er.build_pair_features(source1, targets, candidates)
    accepted = fixed_rule_matches(pair_features).persist(StorageLevel.MEMORY_AND_DISK)
    accepted_count = accepted.count()
    LOG.info("Fixed rules retained %s pairs", f"{accepted_count:,}")

    # Each Source 2/3 record can belong to at most one Source 1 entity.
    target_winner = Window.partitionBy("target_id").orderBy(
        F.desc("rule_score"), F.asc("candidate_rank"), F.asc("source1_id")
    )
    winners = accepted.withColumn("owner_rank", F.row_number().over(target_winner)).filter(
        F.col("owner_rank") == 1
    )
    ordered = Window.partitionBy("source1_id").orderBy(
        F.desc("rule_score"), F.asc("target_source"), F.asc("target_id")
    )
    match_lists = (
        winners.withColumn("output_rank", F.row_number().over(ordered))
        .groupBy("source1_id")
        .agg(F.sort_array(F.collect_list(F.struct("output_rank", "target_id"))).alias("items"))
        .select(
            "source1_id",
            F.expr("concat_ws(',', transform(items, x -> x.target_id))").alias("matched_entity_ids"),
        )
    )
    output = source1_ids.join(match_lists, "source1_id", "left").select(
        F.col("source1_id").alias("source1_entity_id"),
        F.coalesce(F.col("matched_entity_ids"), F.lit("")).alias("matched_entity_ids"),
    )

    total_queries = source1_ids.count()
    output_count = output.count()
    unique_queries = output.select("source1_entity_id").distinct().count()
    if output_count != total_queries or unique_queries != total_queries:
        raise RuntimeError(
            f"Output row contract failed: rows={output_count}, unique={unique_queries}, "
            f"expected={total_queries}"
        )
    if winners.groupBy("target_id").count().filter(F.col("count") > 1).limit(1).count():
        raise RuntimeError("A target record was assigned to multiple Source 1 entities")

    write_one_tsv(output, args.output_file)
    LOG.info("Wrote %s rows to %s", f"{output_count:,}", args.output_file)
    accepted.unpersist(blocking=False)
    candidates.unpersist(blocking=False)
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
