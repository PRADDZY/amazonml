# DevCore Inspired V3

An isolated AWS Spark variant that upgrades the current entity-resolution pipeline with composite blocking inspired by publicly described ER methods. This implementation is original and uses no third-party model or copied source.

## Candidate generation

Records are partitioned by country and indexed in both a native-script, accent-folded view and an ASCII transliteration view. Bounded inverted keys include normalized full name and address, compact names, salient name-token pairs, name plus house number, address-token pairs, house number plus address anchor, postal plus house number, tokens, and sampled character trigrams. Posting lists are frequency-pruned by key type. Query keys are selected by rarity, then candidate evidence is ranked and capped independently for Source 2 and Source 3. The model sees at most 32 candidates per source per Source 1 record; the submitted `candidate_pairs.tsv` contains that pre-classifier set.

## Pair scoring and evaluation

The CPU LightGBM model uses native/transliterated name and address similarities, token overlap, number and postal agreement, retrieval evidence, composite-key hits, and candidate rank. Fitting, threshold calibration, and audit use three disjoint deterministic Source 1 partitions. The calibration objective is per-Source-1 macro F0.5, including entities with no true or predicted links. After selecting a candidate cap, the pipeline coordinate-calibrates separate confidence thresholds for Source 2 and Source 3. Final predictions enforce target exclusivity: each S2/S3 entity is assigned to at most one S1 entity, using score and deterministic tie breaks. A separate audit partition reports F0.5 with this same constraint, without fitting the model or tuning thresholds on its entities.

No score is promised by this code alone. The generated `metrics.json` reports both the calibration score and independent audit F0.5, candidate recall, false positives/negatives, and the test candidate-set size. Treat an independent audit score of 0.95 as the minimum evidence to consider this variant for upload; leaderboard performance can differ.

## AWS execution

Run `aws/run-inspired-v3.ps1` from the workspace PowerShell session. It stages this variant under its own S3 code and output prefixes, uses AWS CLI, and materializes results under `output/inspired_v3/<run-id>/`. It will refuse to launch if it sees an active EC2 instance in the selected region. `-StageOnly` only stages and prepares the AWS-side job; `-SmokeTest` uses the challenge smoke fixture. The full job runs on AWS and does not use local challenge data for model training or inference.

After a successful full run, create an independently validated archive with:

```powershell
python code/business_entity_resolution/inspired_v3/src/package_submission.py --run-dir output/inspired_v3/<run-id>
```

The generated archive is written inside that run directory and contains the two required TSVs, this variant's runnable code, and a metrics-filled methodology report.
