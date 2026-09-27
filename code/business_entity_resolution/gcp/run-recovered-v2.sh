#!/usr/bin/env bash
set -euo pipefail

APP=/home/daithankarpratik/app
RUN="$APP/recovery-v2"
PY="$APP/venv/bin/python"
CODE="$RUN/src"
FEATURES="$RUN/feature-shards"
FIT="$RUN/fit"
OUTPUT="$RUN/output"
LOGS="$RUN/logs"

export PYTHONPATH="$CODE"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$RUN/empty-index" "$FIT" "$OUTPUT" "$LOGS"
printf '{}\n' > "$RUN/empty-index/manifest.json"

for shard in 000 001 002 003 004 005; do
  test -s "$FEATURES/$shard/feature-shard-metrics.json"
  test -s "$FEATURES/$shard/train_features-shard-$shard-of-006.parquet"
  test -s "$FEATURES/$shard/train_targets-shard-$shard-of-006.parquet"
done

echo "FIT_STARTED $(date -Is)" > "$RUN/status.txt"
if ! "$PY" "$CODE/v2_pipeline.py" \
  --phase fit \
  --train-dir "$APP/data/train" \
  --test-dir "$APP/data/test" \
  --index-dir "$RUN/empty-index" \
  --output-dir "$FIT" \
  --feature-dir "$FEATURES" \
  --workers 12 \
  --shard-count 6 > "$LOGS/fit.log" 2>&1; then
  echo "FIT_FAILED $(date -Is)" > "$RUN/status.txt"
  tail -n 60 "$LOGS/fit.log"
  exit 1
fi

echo "TEST_STARTED $(date -Is)" > "$RUN/status.txt"
pids=()
for shard in 0 1 2 3 4 5; do
  shard_name=$(printf '%03d' "$shard")
  mkdir -p "$OUTPUT/test-shards/$shard_name"
  (
    "$PY" "$CODE/v2_pipeline.py" \
      --phase test \
      --train-dir "$APP/data/train" \
      --test-dir "$APP/data/test" \
      --index-dir "$FIT/test-index" \
      --output-dir "$OUTPUT/test-shards/$shard_name" \
      --fit-dir "$FIT" \
      --workers 2 \
      --shard-index "$shard" \
      --shard-count 6
  ) > "$LOGS/test-$shard_name.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  echo "TEST_FAILED $(date -Is)" > "$RUN/status.txt"
  tail -n 40 "$LOGS"/test-*.log
  exit 1
fi

echo "FINALIZE_STARTED $(date -Is)" > "$RUN/status.txt"
"$PY" "$CODE/v2_pipeline.py" \
  --phase finalize \
  --train-dir "$APP/data/train" \
  --test-dir "$APP/data/test" \
  --index-dir "$FIT/test-index" \
  --output-dir "$OUTPUT" \
  --fit-dir "$FIT" \
  --workers 4 \
  --shard-count 6 > "$LOGS/finalize.log" 2>&1

mkdir -p "$APP/output/recovered-v2"
cp "$OUTPUT/matching_results.tsv" "$APP/output/recovered-v2/"
cp "$OUTPUT/candidate_pairs.tsv" "$APP/output/recovered-v2/"
cp "$OUTPUT/metrics.json" "$APP/output/recovered-v2/"
echo "COMPLETED $(date -Is)" > "$RUN/status.txt"
cat "$RUN/status.txt"
ls -lh "$APP/output/recovered-v2"
