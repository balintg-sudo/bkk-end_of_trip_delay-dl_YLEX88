#!/bin/bash

WORK_DIR=$(pwd)

echo "--- Starting project ---"
echo "Work directory: $WORK_DIR"

if [ ! -d "$WORK_DIR/data" ]; then
    mkdir -p "$WORK_DIR/data"
fi

if [ ! -d "$WORK_DIR/log" ]; then
    mkdir "$WORK_DIR/log"
fi

echo "Running docker container..."
docker run \
    --memory="8g" \
    --memory-swap="-1" \
    -v "$WORK_DIR/data":/app/data \
    -v "$WORK_DIR/log":/app/log \
    dl-project > "$WORK_DIR/log/run.log" 2>&1

echo "Finished running. Log file at 'log/run.log'"
