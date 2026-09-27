#!/usr/bin/env bash
set -euo pipefail

APP=/home/daithankarpratik/app
OUT="$APP/output/no-fit"
SPARK_TMP="$APP/no-fit-spark-tmp-retry2"
mkdir -p "$SPARK_TMP" "$OUT"

if [[ -s "$OUT/matching_results.tsv" ]]; then
  echo "Refusing to overwrite $OUT/matching_results.tsv" >&2
  exit 2
fi
if [[ -s "$OUT/job.pid" ]] && kill -0 "$(cat "$OUT/job.pid")" 2>/dev/null; then
  echo "No-fit job is already running with PID $(cat "$OUT/job.pid")"
  exit 0
fi

export PYSPARK_PYTHON="$APP/venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$APP/venv/bin/python"
export SPARK_LOCAL_DIRS="$SPARK_TMP"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

nohup "$APP/venv/bin/spark-submit" \
  --master 'local[6]' \
  --driver-memory 32g \
  --conf spark.driver.maxResultSize=4g \
  --conf "spark.local.dir=$SPARK_TMP" \
  --conf spark.default.parallelism=48 \
  --conf spark.sql.shuffle.partitions=48 \
  --conf spark.speculation=false \
  --py-files "$APP/code/er_core.py,$APP/code/sagemaker_spark_job.py" \
  "$APP/code/no_fit_submission.py" \
  --test-prefix "$APP/data/test" \
  --output-file "$OUT/matching_results.tsv" \
  --parallelism 48 > "$OUT/run.log" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$OUT/job.pid"
sleep 1
if kill -0 "$pid" 2>/dev/null; then
  echo "No-fit matcher started (PID $pid); log: $OUT/run.log"
else
  echo "No-fit matcher failed to start" >&2
  tail -n 40 "$OUT/run.log" >&2 || true
  exit 1
fi
