#!/bin/bash
# Collects the .tsv sample files (0.tsv, 1.tsv, ...) produced under
# runs/*/output/... into a single collected/ folder.
#
# Every run writes files with the same names (0.tsv, 1.tsv, ...) inside the
# same category subfolders (e.g. N=25_m=2/dL=10/Statistics/...), since each
# job independently restarts its own rep counter at 0. This script mirrors
# that category structure under collected/, but renumbers files so samples
# from different runs land in the same folder without overwriting each
# other.
#
# collected/ is rebuilt from scratch on every invocation so it always
# matches exactly what's currently in runs/ (safe: it only ever touches
# collected/, and only ever reads from runs/ via cp -- files in runs/ are
# never modified, moved, or deleted).
#
# Usage: ./collect.sh

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$BASE_DIR/runs"
COLLECTED_DIR="$BASE_DIR/collected"

if [[ ! -d "$RUNS_DIR" ]]; then
    echo "Error: $RUNS_DIR does not exist" >&2
    exit 1
fi

rm -rf "$COLLECTED_DIR"
mkdir -p "$COLLECTED_DIR"

declare -A counter=()
total=0

# Walk run directories in numeric order (runs/1, runs/2, ...) and, within
# each, tsv files in numeric order, so the collected numbering is
# reproducible across invocations.
while IFS= read -r -d '' run_dir; do
    [[ -d "$run_dir/output" ]] || continue

    while IFS= read -r -d '' src; do
        rel="${src#"$run_dir"/output/}"
        category="$(dirname "$rel")"
        dest_dir="$COLLECTED_DIR/$category"
        mkdir -p "$dest_dir"

        idx="${counter[$category]:-0}"
        cp "$src" "$dest_dir/$idx.tsv"
        counter[$category]=$((idx + 1))
        total=$((total + 1))
    done < <(find "$run_dir/output" -type f -name '*.tsv' -print0 | sort -z -V)
done < <(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z -V)

echo "Collected $total files into $COLLECTED_DIR"
for category in "${!counter[@]}"; do
    echo "  $category: ${counter[$category]} files"
done
