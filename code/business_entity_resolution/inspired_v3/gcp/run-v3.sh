#!/usr/bin/env bash
set -uo pipefail

APP="${APP:-$HOME/app}"
export PATH="$APP/venv/bin:$PATH"
export PYSPARK_PYTHON="$APP/venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$APP/venv/bin/python"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SPARK_LOCAL_DIRS="$APP/spark-tmp"

mkdir -p "$SPARK_LOCAL_DIRS" "$APP/output"
printf 'RUNNING %s\n' "$(date -Is)" > "$APP/run.status"

"$APP/venv/bin/spark-submit" \
  --master 'local[12]' \
  --driver-memory 64g \
  --conf spark.driver.maxResultSize=8g \
  --conf spark.local.dir="$SPARK_LOCAL_DIRS" \
  --conf spark.sql.files.maxPartitionBytes=16777216 \
  --conf spark.default.parallelism=96 \
  --conf spark.speculation=false \
  --py-files "$APP/code/er_core.py" \
  "$APP/code/sagemaker_spark_job.py" \
  --train-prefix "$APP/data/train" \
  --test-prefix "$APP/data/test" \
  --output-prefix "$APP/output" \
  --smoke-test false \
  --instance-type n2-custom-12-98304 \
  --compute-service 'Google Compute Engine Spark 3.5.4' \
  --s3-scheme s3 > "$APP/run.log" 2>&1
RC=$?

if [ "$RC" -eq 0 ]; then
  printf 'COMPLETED %s\n' "$(date -Is)" > "$APP/run.status"
else
  printf 'FAILED exit=%s %s\n' "$RC" "$(date -Is)" > "$APP/run.status"
fi

exit "$RC"
