#!/usr/bin/env python3
"""Validate challenge outputs and build the DevCore submission archive."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[3]
STUDENT_ROOT = WORKSPACE / "6ab10eb3b23ba_student_resource" / "student_resource"
PACKAGE_ROOT = WORKSPACE / "code" / "business_entity_resolution"
OUTPUT_ROOT = WORKSPACE / "output"
ARCHIVE = WORKSPACE / "DevCore_submission.zip"
DOCUMENTATION = WORKSPACE / "Documentation_template.md"
EXCLUDED_FILES = {
    "sagemaker-trust-policy.json",
    "run-aws.ps1",
    "run-glue.ps1",
    "package-submission.ps1",
}


def render_documentation(metrics: dict[str, Any], test_metrics: dict[str, Any] | None = None) -> str:
    """Fill reportable run metrics into the methodology document."""
    text = DOCUMENTATION.read_text(encoding="utf-8")
    if "[[SELECTED_CAP]]" in text:
        operating = metrics["training_oof_operating_point"]
        test_metrics = test_metrics or metrics
        replacements = {
            "[[SELECTED_CANDIDATES_PER_TARGET]]": str(operating["cap_per_target"]),
            "[[TEST_CANDIDATE_PAIRS]]": f"{test_metrics.get('candidate_pairs', metrics.get('test_candidate_pairs', 0)):,}",
            "[[CANDIDATE_DISTRIBUTION]]": "mean {mean:.3f}; median {median:.3f}; p95 {p95:.3f}; p99 {p99:.3f}; max {maximum}".format(
                mean=test_metrics.get("candidate_count_mean", 0),
                median=test_metrics.get("candidate_count_median", 0),
                p95=test_metrics.get("candidate_count_p95", 0),
                p99=test_metrics.get("candidate_count_p99", 0),
                maximum=test_metrics.get("candidate_count_max", 0),
            ),
            "[[CANDIDATE_REDUCTION]]": f"{test_metrics.get('candidate_reduction_ratio', metrics.get('candidate_reduction_ratio', 0)):.2%}",
            "[[SELECTED_CAP]]": str(operating["cap_per_target"]),
            "[[SELECTED_THRESHOLD]]": f"{operating['threshold']:.4f}".rstrip("0").rstrip("."),
            "[[CALIBRATION_F05]]": f"{operating['macro_f0_5']:.6f}",
            "[[AUDIT_F05]]": f"{operating['audit_macro_f0_5']:.6f}",
        }
        for token, value in replacements.items():
            text = text.replace(token, value)
        if "[[" in text:
            raise ValueError("V2 methodology still contains an unfilled placeholder")
        return text

    validation_decisions = metrics["validation_pair_decisions"]
    per_country = metrics["per_country"]
    country_summary = "; ".join(
        f"{country}: {values['candidate_pairs']:,} candidates across "
        f"{values['source1_rows']:,} Source 1 rows"
        for country, values in sorted(per_country.items())
    ) or "No country metrics reported"
    cap = metrics["selected_cap_per_source"]
    values = {
        "**Selected cap per target source:** To be filled from `output/metrics.json`.  ":
            f"**Selected cap per target source:** {cap} (maximum {2 * cap} total per Source 1).  ",
        "**Held-out candidate recall:** To be filled from `output/metrics.json`.  ":
            f"**Held-out candidate recall:** {metrics['selected_candidate_recall']:.4f}.  ",
        "**Test candidate pairs:** To be filled from `output/metrics.json`.  ":
            f"**Test candidate pairs:** {metrics['candidate_pairs']:,}.  ",
        "**Test candidate reduction ratio:** To be filled from `output/metrics.json`.":
            f"**Test candidate reduction ratio:** {metrics['candidate_reduction_ratio']:.4%}.  ",
        "**Selected threshold:** To be filled from `output/metrics.json`.  ":
            f"**Selected threshold:** {metrics['selected_threshold']:.6f}.  ",
        "**Validation macro F0.5:** To be filled from `output/metrics.json`.  ":
            f"**Validation macro F0.5:** {metrics['validation_macro_f0_5']:.4f}.  ",
        "**False-positive/false-negative review:** To be completed from held-out predictions.":
            "**Held-out pair decisions:** "
            f"{validation_decisions['true_positives']:,} true positives, "
            f"{validation_decisions['false_positives']:,} false positives, "
            f"{validation_decisions['false_negatives']:,} false negatives; "
            f"{validation_decisions['missed_by_blocking']:,} true links missed by blocking.",
        "- **AWS EC2 runtime:** To be filled from the completed job run.":
            f"- **AWS EC2 runtime:** {metrics['runtime_seconds'] / 3600:.2f} hours.",
        "**Candidate count mean / median / p95 / p99 / max:** To be filled from\n"
        "`output/metrics.json`.  ":
            "**Candidate count mean / median / p95 / p99 / max:** "
            f"{metrics['candidate_count_mean']:.2f} / "
            f"{metrics['candidate_count_median']:,} / "
            f"{metrics['candidate_count_p95']:,} / "
            f"{metrics['candidate_count_p99']:,} / "
            f"{metrics['candidate_count_max']:,}.  ",
        "**Per-country counts:** To be filled from `output/metrics.json`.":
            f"**Per-country candidate counts:** {country_summary}.",
    }
    for old, new in values.items():
        if old not in text:
            raise ValueError(f"Documentation template is missing expected placeholder: {old!r}")
        text = text.replace(old, new, 1)
    return text


def validate_candidate_file(path: Path, test_source1: Path, cap_per_source: int | None) -> tuple[int, int]:
    """Validate candidate rows in bounded memory without retaining candidate pairs."""
    if cap_per_source is not None and cap_per_source < 1:
        raise ValueError("Candidate cap must be a positive integer")

    required: set[str] = set()
    with test_source1.open("r", encoding="utf-8-sig", newline="") as source:
        next(source, None)
        for line in source:
            if line.strip():
                required.add(line.split("\t", 1)[0].strip())
    if not required:
        raise ValueError(f"No Source 1 entity IDs found in {test_source1}")

    rows = empty_rows = 0
    expected_header = "source1_entity_id\tcandidate_entity_ids"
    with path.open("r", encoding="utf-8-sig", newline="") as candidates:
        header = candidates.readline().rstrip("\r\n")
        if header != expected_header:
            raise ValueError(f"Unexpected candidate header: {header!r}")

        for line_number, line in enumerate(candidates, start=2):
            row = line.rstrip("\r\n")
            source1_id, separator, raw_ids = row.partition("\t")
            if not separator or not source1_id or source1_id.strip() != source1_id:
                raise ValueError(f"Malformed candidate row at line {line_number}")
            if source1_id not in required:
                raise ValueError(
                    f"Duplicate or unknown Source 1 ID in candidate file at line {line_number}: "
                    f"{source1_id}"
                )
            required.remove(source1_id)

            ids = raw_ids.split(",") if raw_ids else []
            if any(not entity_id for entity_id in ids):
                raise ValueError(f"Empty candidate ID at line {line_number}")
            if len(ids) != len(set(ids)):
                raise ValueError(f"Repeated candidate ID at line {line_number}")
            s2_count = sum(entity_id.startswith("S2-") for entity_id in ids)
            s3_count = sum(entity_id.startswith("S3-") for entity_id in ids)
            if s2_count + s3_count != len(ids):
                raise ValueError(f"Candidate ID has an invalid source prefix at line {line_number}")
            if cap_per_source is not None and (s2_count > cap_per_source or s3_count > cap_per_source):
                raise ValueError(
                    f"Candidate cap exceeded at line {line_number}: "
                    f"S2={s2_count}, S3={s3_count}, cap={cap_per_source}"
                )
            rows += 1
            empty_rows += not ids

    if required:
        examples = ", ".join(sorted(required)[:5])
        raise ValueError(f"Candidate file is missing {len(required):,} Source 1 rows; e.g. {examples}")

    cap_summary = f"at most {cap_per_source} candidates per target source" if cap_per_source else "unique, valid source IDs"
    print(f"  candidate_pairs.tsv: {rows:,} rows ({empty_rows:,} empty); {cap_summary}.")
    return rows, empty_rows


def main() -> int:
    matching = OUTPUT_ROOT / "matching_results.tsv"
    candidates = OUTPUT_ROOT / "candidate_pairs.tsv"
    metrics_path = OUTPUT_ROOT / "metrics.json"
    test_metrics_path = OUTPUT_ROOT / "test_metrics.json"
    required = [matching, candidates, metrics_path, DOCUMENTATION]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required submission files are missing: " + ", ".join(map(str, missing)))

    validator = STUDENT_ROOT / "utils" / "validate_submission.py"
    subprocess.run(
        [
            sys.executable,
            str(validator),
            "--matching",
            str(matching),
            "--test-dir",
            str(STUDENT_ROOT / "dataset" / "test"),
        ],
        cwd=STUDENT_ROOT,
        check=True,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "training_oof_operating_point" in metrics and not test_metrics_path.is_file():
        raise FileNotFoundError(f"Required V2 metric file is missing: {test_metrics_path}")
    test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8")) if test_metrics_path.is_file() else {}
    cap = None if "training_oof_operating_point" in metrics else int(metrics["selected_cap_per_source"])
    validate_candidate_file(
        candidates,
        STUDENT_ROOT / "dataset" / "test" / "test_source1.tsv",
        cap,
    )

    temporary_archive = ARCHIVE.with_suffix(".zip.tmp")
    try:
        documentation = render_documentation(metrics, test_metrics)
        with zipfile.ZipFile(
            temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            archive.write(matching, "output/matching_results.tsv")
            archive.write(candidates, "output/candidate_pairs.tsv")
            for path in PACKAGE_ROOT.rglob("*"):
                if not path.is_file() or path.name in EXCLUDED_FILES or "__pycache__" in path.parts:
                    continue
                if path.suffix == ".pyc":
                    continue
                archive.write(path, Path("code/business_entity_resolution") / path.relative_to(PACKAGE_ROOT))
            archive.writestr("Documentation_template.md", documentation)

        with zipfile.ZipFile(temporary_archive) as archive:
            bad_member = archive.testzip()
            names = set(archive.namelist())
            expected = {
                "output/matching_results.tsv",
                "output/candidate_pairs.tsv",
                "Documentation_template.md",
                "code/business_entity_resolution/aws/run-ec2.ps1",
            }
            if bad_member is not None or not expected.issubset(names):
                raise RuntimeError(f"Archive validation failed (bad member: {bad_member!r})")

        temporary_archive.replace(ARCHIVE)
    finally:
        temporary_archive.unlink(missing_ok=True)

    print(f"Validated submission archive: {ARCHIVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
