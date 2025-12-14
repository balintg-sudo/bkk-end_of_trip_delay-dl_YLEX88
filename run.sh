#!/bin/bash

WORK_DIR=$(pwd)

echo "--- Starting project ---"
echo "Work directory: $WORK_DIR"

if [ ! -d "$WORK_DIR/data" ]; then
    echo "ERROR: can't find 'data' directory in $WORK_DIR"
    exit 1
fi

if [ ! -d "$WORK_DIR/log" ]; then
    mkdir "$WORK_DIR/log"
fi

echo "Running docker container..."
docker run \
    --memory="8g" \
    --memory-swap="16g" \
    -v "$WORK_DIR/data":/app/data \
    -v "$WORK_DIR/log":/app/log \
    dl-project > "$WORK_DIR/log/run.log" 2>&1

echo "Finished running. Log file at 'log/run.log'"
