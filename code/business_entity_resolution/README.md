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

## Higher-recall v2 path

`aws/launch-v2.ps1 -Mode probe` audits multilingual BM25 retrieval against the
complete training Source 1 index. `-Mode full` uses 64 CPU workers on an
`r6i.16xlarge` EC2 instance to run the cross-fitted LightGBM pipeline on AWS,
chooses the target candidate cap and score threshold using
macro F0.5, then writes both final TSVs. The reverse index is country-scoped;
each target queries bounded top-k Source 1 candidates, so the search does not
form a Cartesian product. The fifth entity fold remains an untouched operating
point audit. Worker status, logs, metrics, and submission files are saved under
the run's S3 `output/v2/` prefix; the worker shuts down automatically.

Run the new full path from the workspace root after the probe finishes:

```powershell
.\code\business_entity_resolution\aws\launch-v2.ps1 -Mode full -MaxHours 20
```

After a successful full run, download its files and metrics into a versioned
folder with the AWS CLI helper:

```powershell
.\code\business_entity_resolution\aws\fetch-v2.ps1
```

The helper leaves the current `output/` files untouched. Validate the fetched
files and create the final archive with:

```powershell
.\code\business_entity_resolution\aws\promote-v2.ps1 `
  -ResultsDirectory .\output\v2\results-<run-timestamp>
```

The promotion script runs the organizer's validator first, fills the methodology
metrics from the AWS run, backs up the current files and archive, then creates
`DevCore_submission.zip` from the validated TSVs.

The older Spark implementation below remains the baseline path. The v2 run is
promoted only after its held-out audit and official submission validation pass.

## Baseline Spark run

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

If the local launcher exits while the EC2 run is still active, continue with AWS
CLI using the run URI printed when the instance was started:

```powershell
$runUri = "s3://<bucket>/output/glue/<run-id>"
aws s3 cp "$runUri/status.txt" - --region us-east-1
aws s3 cp "$runUri/logs/job.log" - --region us-east-1 | Select-Object -Last 40
```

When the status is `SUCCEEDED`, retrieve the partitioned outputs and flatten them
for the submission package:

```powershell
New-Item -ItemType Directory -Force -Path .\output\candidate_parts, .\output\matching_parts, .\output\metrics_parts | Out-Null
aws s3 cp "$runUri/submission/candidate_pairs/" .\output\candidate_parts --recursive --region us-east-1 --only-show-errors
aws s3 cp "$runUri/submission/matching_results/" .\output\matching_parts --recursive --region us-east-1 --only-show-errors
aws s3 cp "$runUri/metrics/" .\output\metrics_parts --recursive --region us-east-1 --only-show-errors
python .\code\business_entity_resolution\src\materialize_output.py `
  --candidate-parts .\output\candidate_parts `
  --matching-parts .\output\matching_parts `
  --metrics-parts .\output\metrics_parts `
  --output-dir .\output
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
