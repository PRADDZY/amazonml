# Amazon ML Challenge 2026: Business Entity Resolution

**Team Name:** DevCore  
**Team Members:** Pratik Daithankar  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary

The solution uses country-constrained blocking followed by a Spark ML gradient-
boosted decision tree pair classifier. Validation selects the per-source candidate
cap and probability threshold against macro F0.5. `candidate_pairs.tsv` contains the
exact candidates scored by the final model, before thresholding.

## 2. Data and Constraints

The challenge supplies separate Source 1, Source 2, and Source 3 records and a
Source 1-to-target link list for training. Source 1 can match zero, one, or multiple
records in each target source. Training includes US and India; test adds France, so
the pipeline uses country labels as open-set strings and does not require training
examples for a country to generate test candidates.

All processing uses only the challenge files. The solution performs no external
business lookup, registry query, geocoding, or identity resolution service call.

## 3. Candidate Generation

Records are normalized with Unicode accent handling, trailing legal-suffix removal,
and common address abbreviation mapping. Within each country and target source, the
pipeline indexes exact normalized names and addresses, postal-code/house-number
combinations, informative name/address tokens, and character trigrams. It measures
posting-list frequency before joining and drops overly broad keys. Each query keeps
its rarest eligible keys per key type; retrieved records are merged and ranked from
exact-key, postal/house-number, token, trigram, and inverse-frequency evidence.

The candidate set is capped independently for Sources 2 and 3. Validation compares
caps of 4, 8, 12, 16, 24, and 32 per target source. The smallest cap with at least
99% held-out candidate recall and macro F0.5 within 0.002 of the best validation
score is selected. If no cap meets that recall gate, the cap with the best measured
F0.5 is selected. The final cap is at most 32 per source, or 64 total candidates per
Source 1 entity.

**Selected cap per target source:** To be filled from `output/metrics.json`.  
**Held-out candidate recall:** To be filled from `output/metrics.json`.  
**Test candidate pairs:** To be filled from `output/metrics.json`.  
**Test candidate reduction ratio:** To be filled from `output/metrics.json`.

## 4. Pair Model and Decision Rule

The model is Spark ML `GBTClassifier` using Spark 3.5.4 and Python 3.11 on Amazon
Linux 2023. Features cover normalized name and address edit
similarity, token Jaccard and containment, numeric and postal agreement, exact-key
and postal/house-number block hits, token and trigram evidence, inverse posting
frequency, missing-address indicators, and target source.

Training Source 1 IDs are assigned deterministically: 5% for model fitting, 5% for
validation, and the rest for retrieval-corpus coverage. For each fitting query, the
model uses all retrieved positive pairs and up to eight top-ranked negative pairs
per target source. Validation chooses the candidate cap and a probability threshold
using macro F0.5; correctly empty predictions for singletons contribute full credit.
The model does not force a top-1 match.

**Selected threshold:** To be filled from `output/metrics.json`.  
**Validation macro F0.5:** To be filled from `output/metrics.json`.  
**False-positive/false-negative review:** To be completed from held-out predictions.

## 5. AWS Execution and Results

The full batch job runs on an encrypted `r6i.2xlarge` EC2 instance in `us-east-1`,
alongside the challenge S3 bucket. Spark uses eight local worker threads and a 44 GiB
driver heap; the instance has 64 GiB RAM and a 200 GB gp3 root volume. AWS CLI stages
the source, launches and monitors the instance, and retrieves output parts from S3.
The EC2 role is scoped to the challenge bucket, the instance requires IMDSv2, and no
inbound network access is opened. The Spark process has a 12-hour timeout and the
instance terminates when its bootstrap script exits. Spark writes partitioned TSV
parts; the packaging helper streams them into flat submission files. Model fitting,
validation, and full test inference run on AWS. There is no hosted endpoint.

**AWS EC2 runtime and estimated compute/storage cost:** To be filled from the
completed job run.  
**Candidate count mean / median / p95 / p99 / max:** To be filled from
`output/metrics.json`.  
**Per-country counts:** To be filled from `output/metrics.json`.

## 6. Submission Artifacts

The archive contains `output/matching_results.tsv`, `output/candidate_pairs.tsv`,
the runnable Spark source and AWS EC2 launcher under `code/business_entity_resolution/`, its
runtime instructions, and this methodology document. The provided challenge
validator is run before packaging. Both output files contain exactly one row for
every test Source 1 entity, including empty lists for records with no candidates or
matches.
