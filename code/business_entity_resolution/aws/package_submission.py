#!/usr/bin/env python3
"""Validate challenge outputs and build the DevCore submission archive."""

from __future__ import annotations

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
EXCLUDED_FILES = {"sagemaker-trust-policy.json", "run-glue.ps1", "package-submission.ps1"}


def render_documentation(metrics: dict[str, Any]) -> str:
    """Fill reportable run metrics into the methodology document."""
    text = DOCUMENTATION.read_text(encoding="utf-8")
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
        "**AWS EC2 runtime and estimated compute/storage cost:** To be filled from the\n"
        "completed job run.  ":
            f"**AWS EC2 runtime:** {metrics['runtime_seconds'] / 3600:.2f} hours.  ",
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


def main() -> int:
    matching = OUTPUT_ROOT / "matching_results.tsv"
    candidates = OUTPUT_ROOT / "candidate_pairs.tsv"
    metrics_path = OUTPUT_ROOT / "metrics.json"
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
            "--candidate",
            str(candidates),
            "--test-dir",
            str(STUDENT_ROOT / "dataset" / "test"),
        ],
        cwd=STUDENT_ROOT,
        check=True,
    )

    temporary_archive = ARCHIVE.with_suffix(".zip.tmp")
    try:
        import json

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        documentation = render_documentation(metrics)
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
