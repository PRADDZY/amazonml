# Amazon ML Challenge 2026: Business Entity Resolution

**Team Name:** DevCore  
**Team Members:** Pratik Daithankar  
**Submission Date:** 2026-09-26

## Solution summary

The solution first retrieves a small, country-matched candidate set for each
Source 2 and Source 3 record, then ranks candidate pairs with a CPU LightGBM
classifier. It writes the inverse candidate mapping to `candidate_pairs.tsv` and
the selected links to `matching_results.tsv`. Candidate generation uses bounded
top-k retrieval over Source 1; it never compares every pair of records.

## Data and text processing

The pipeline uses the challenge's training and test files. It does not query
external business registries, addresses, labels, or identity-resolution services.
Text processing preserves native scripts, folds Latin accents, and creates a
deterministic ASCII transliteration view. It retains raw and legal-suffix-cleaned
name views, normalizes common address terms, and indexes word tokens and character
trigrams.

Training, validation, indexing, and test inference run on AWS. Test Source 1 is
indexed separately from the training reference corpus, so test records and their
candidate IDs never enter model fitting.

## Candidate generation

Source 1 records are indexed separately by country with Tantivy. For each target
record, three BM25 queries retrieve bounded top-k candidates using combined name
and address evidence, name evidence, and address evidence. Each query uses raw and
cleaned text, transliteration, word tokens, and character trigrams. Reciprocal rank
fusion merges the three ranked lists, each with up to 24 results, then selects a
per-target cap from 1, 2, 3, 4, 6, or 8 candidates. The cap is selected during
cross-fitting to retain a small set while preserving measured macro F0.5.

`candidate_pairs.tsv` records the capped set presented to the pair classifier,
before its confidence threshold selects final matches. Candidate lists are
deduplicated per Source 1 entity and include empty lists.

**Selected candidates per target:** To be filled from `oof_metrics.json`.

**Test candidate-pair count:** To be filled from `test_metrics.json`.

**Mean / median / p95 / p99 / maximum candidates per Source 1:** To be filled
from `test_metrics.json`.

**Same-country search-space reduction:** To be filled from `test_metrics.json`.

## Pair model and validation

The pair classifier is CPU LightGBM. Its features describe native and
transliterated name/address similarity, token and trigram overlap, exact matches,
house-number and postal agreement, missing fields, country, target source, and
retrieval ranks and scores.

Five deterministic folds keep each target's candidate set together. Training
examples exclude the held-out target fold and candidate-reference fold. A fifth
Source 1 reference fold is reserved from all training splits for an entity-level
audit. The remaining folds select a per-target candidate cap and confidence
threshold against the challenge's macro F0.5 objective, including Source 1
entities with no true or predicted links. Each target produces at most one final
Source 1 match. The final classifier is fit on all training candidate pairs after
the operating point is selected.

**Selected candidate cap:** To be filled from `oof_metrics.json`.

**Selected confidence threshold:** To be filled from `oof_metrics.json`.

**Cross-fit calibration macro F0.5:** To be filled from `oof_metrics.json`.

**Reserved-fold audit macro F0.5:** To be filled from `oof_metrics.json`.

## AWS execution and artifacts

The full workflow runs on one `r6i.2xlarge` Amazon Linux 2023 EC2 worker in
`us-east-1`, with eight CPU workers and an encrypted 200 GB gp3 root volume. AWS
CLI stages code and data in S3, launches the worker, and retrieves its logs,
metrics, and output files. A finite process timeout and independent instance
shutdown limit bound the run. There is no hosted endpoint.

The final archive contains `output/matching_results.tsv`,
`output/candidate_pairs.tsv`, the runnable source and AWS launcher, and this
methodology. The organizer's submission validator runs before the archive is
created.
