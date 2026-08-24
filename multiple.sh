#!/bin/bash
# Submits multiple independent LaughlinEdge jobs to the cluster.
#
# LaughlinEdge always writes to ./output relative to its working directory,
# so each submission gets its own numbered folder under runs/ (runs/1, runs/2, ...,
# picking up wherever the last submission left off) and is sbatch'd from inside
# that folder. Since SLURM jobs default to the working directory at submission
# time, run.sh's "./output" and "./logs" then resolve inside that run's folder,
# so parallel runs never overwrite each other's output.
#
# The executable is symlinked into each run folder because run.sh invokes it as
# plain "LaughlinEdge" (no path), which is resolved relative to the job's
# working directory, not the location of run.sh or of this script.
#
# Usage: ./multiple.sh <num_runs>

set -euo pipefail

NUM_RUNS="${1:?Usage: $0 <num_runs>}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTABLE="$BASE_DIR/LaughlinEdge"
RUN_SCRIPT="$BASE_DIR/run.sh"
RUNS_DIR="$BASE_DIR/runs"

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "Error: executable not found at $EXECUTABLE (build it first)" >&2
    exit 1
fi

mkdir -p "$RUNS_DIR"

n=1
submitted=0
while (( submitted < NUM_RUNS )); do
    while [[ -e "$RUNS_DIR/$n" ]]; do
        n=$((n + 1))
    done

    RUN_DIR="$RUNS_DIR/$n"
    mkdir -p "$RUN_DIR/logs"
    ln -s "$EXECUTABLE" "$RUN_DIR/LaughlinEdge"

    (cd "$RUN_DIR" && sbatch -J "laughlin_edge_run${n}" "$RUN_SCRIPT")

    echo "Submitted run $n -> $RUN_DIR"
    n=$((n + 1))
    submitted=$((submitted + 1))
done
