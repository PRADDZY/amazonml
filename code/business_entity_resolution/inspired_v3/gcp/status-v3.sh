#!/usr/bin/env bash
set -u

APP=/home/daithankarpratik/app

printf 'Run status\n----------\n'
if [[ -f "$APP/run.status" ]]; then
    cat "$APP/run.status"
else
    printf 'Waiting for uploads or job start\n'
fi

printf '\nUploaded data\n-------------\n'
du -sh "$APP/data/train" "$APP/data/test" 2>/dev/null || true

printf '\nSpark processes\n---------------\n'
pgrep -af 'org.apache.spark.deploy.SparkSubmit|CoarseGrainedExecutorBackend|org.apache.spark.executor' || printf 'No Spark process found\n'

printf '\nRecent run log\n--------------\n'
if [[ -f "$APP/run.log" ]]; then
    grep -E 'ERROR|WARN|Stage [0-9]+|Entity resolution complete|Traceback|FAILED|COMPLETED' "$APP/run.log" | tail -n 12 || tail -n 8 "$APP/run.log"
else
    printf 'Run log has not been created\n'
fi

printf '\nOutput files\n------------\n'
find "$APP/output" -maxdepth 4 -type f -printf '%P (%s bytes)\n' 2>/dev/null | sort | tail -n 15 || true

printf '\nMemory and disk\n---------------\n'
free -h | head -n 2
df -h "$APP" | tail -n 1
