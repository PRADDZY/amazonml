# Business Entity Resolution

This solution runs as a Spark batch job on Amazon EC2 in AWS. The AWS CLI stages the
source, launches and monitors the instance, and downloads the submission parts.
Full-corpus processing, model fitting, validation, and inference run in AWS; there
is no local full-data processing or hosted inference endpoint.

## Candidate generation and matching

Records are normalized with Unicode accent handling, business suffix removal, and
address abbreviations. Blocking indexes exact normalized names and addresses,
postal-code/house-number combinations, informative name/address tokens, and character
trigrams. Posting lists are measured before joins and broad keys are suppressed; each
Source 1 query emits a bounded number of keys. Retrieval ranks and deduplicates hits,
then keeps at most 32 candidates from each target source.

Training Source 1 IDs are deterministically split into model-fit, validation, and
retrieval-corpus groups. The Spark ML `GBTClassifier` trains on retrieved positives
and the eight strongest negatives per source and query. A validation sweep evaluates
per-source caps `{4, 8, 12, 16, 24, 32}` and decision thresholds with macro F0.5,
including singleton records. The selected cap is the smallest meeting 99% held-out
candidate recall and scoring within 0.002 of the best measured F0.5; if none meets
that recall gate, the highest-scoring cap is used.

`candidate_pairs.tsv` contains the exact final capped set scored by the model, with
one row per test Source 1 entity. `matching_results.tsv` contains the model's
above-threshold matches. Empty lists are retained in both files. The Spark job writes
partitioned outputs and metrics to S3; `materialize_output.py` streams the parts into
flat UTF-8 TSV files and `output/metrics.json`.

## AWS run

The launcher uses AWS CLI only and runs in `us-east-1`, beside the challenge S3
bucket. Full runs use one `r6i.2xlarge` EC2 instance with eight Spark local worker
threads; smoke runs use `r6i.xlarge`. The bootstrap installs Spark 3.5.4 and Python
3.11, reads input files from S3, runs Spark ML training and inference, and writes
candidate pairs, matching results, metrics, and logs under `output/glue/`. IMDSv2 is
required, no inbound network access is opened, the root EBS volume is encrypted, and
the instance terminates when the job exits. The full Spark process has a 12-hour
timeout. This route does not depend on SageMaker Processing or AWS Glue quotas.

From the workspace root, stage the source and validate the EC2 launcher without
starting compute:

```powershell
.\code\business_entity_resolution\aws\run-ec2.ps1 -StageOnly
```

Run the synthetic smoke test, then the full challenge data:

```powershell
.\code\business_entity_resolution\aws\run-ec2.ps1 -SmokeTest
.\code\business_entity_resolution\aws\run-ec2.ps1
```

After the full job succeeds, run the challenge validator and create the team archive:

```powershell
python .\code\business_entity_resolution\aws\package_submission.py
```

The archive is named `DevCore_submission.zip` and contains both required TSVs,
`code/business_entity_resolution/`, and `Documentation_template.md`. The Python
package script runs the provided validator on scored matches, then streams
`candidate_pairs.tsv` to check that every test Source 1 ID appears once, each
candidate ID has an S2/S3 prefix, each list is duplicate-free, and the selected
per-target-source cap is respected. The Spark job also checks that every final
match appears in the candidate set before writing either submission file.

Run the standard-library unit tests locally with:

```powershell
Push-Location .\code\business_entity_resolution
python -m unittest discover -s tests -v
Pop-Location
```
