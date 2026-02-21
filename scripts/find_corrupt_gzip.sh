#!/usr/bin/env bash
# Find corrupt/truncated gzip files that have valid headers but incomplete streams.
# Uses `gzip -t` to verify integrity in parallel.
#
# Usage:
#   bash scripts/find_corrupt_gzip.sh /g/data/hl36/mhf/monai/Task03_BrainTumourDx/imagesTr
#   bash scripts/find_corrupt_gzip.sh /g/data/hl36/mhf/monai/Task03_BrainTumourDx/labelsTr
#
# Control parallelism:
#   JOBS=32 bash scripts/find_corrupt_gzip.sh /path/to/dir

set -euo pipefail

DIR="${1:?Usage: $0 <directory>}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)}"

CORRUPT=$(mktemp)
NOTGZ=$(mktemp)
trap "rm -f $CORRUPT $NOTGZ" EXIT

echo "Scanning $(find "$DIR" -name '*.nii.gz' | wc -l) files with $JOBS jobs..."

check_file() {
    local f="$1"
    # Check magic bytes first
    local magic
    magic=$(od -A n -t x1 -N 2 "$f" | tr -d ' ')
    if [[ "$magic" != "1f8b" ]]; then
        echo "NOT_GZIP $f"
        return 0
    fi
    # Verify gzip integrity (decompress to /dev/null)
    if ! gzip -t "$f" 2>/dev/null; then
        echo "CORRUPT $f"
        return 0
    fi
}
export -f check_file

# Run checks in parallel
if command -v parallel &>/dev/null; then
    find "$DIR" -name '*.nii.gz' | parallel -j "$JOBS" check_file | tee >(grep '^CORRUPT' | awk '{print $2}' > "$CORRUPT") >(grep '^NOT_GZIP' | awk '{print $2}' > "$NOTGZ") > /dev/null
    # Wait for tee subprocesses
    wait
else
    find "$DIR" -name '*.nii.gz' | xargs -P "$JOBS" -I{} bash -c 'check_file "$@"' _ {} | tee >(grep '^CORRUPT' | awk '{print $2}' > "$CORRUPT") >(grep '^NOT_GZIP' | awk '{print $2}' > "$NOTGZ") > /dev/null
    wait
fi

n_corrupt=$(wc -l < "$CORRUPT")
n_notgz=$(wc -l < "$NOTGZ")

echo ""
echo "=== Results ==="
echo "Not gzip (wrong magic): $n_notgz"
echo "Corrupt (truncated):    $n_corrupt"

if [[ $n_notgz -gt 0 ]]; then
    echo ""
    echo "--- Not gzip (fix with fix_fake_gzip.sh) ---"
    cat "$NOTGZ"
fi

if [[ $n_corrupt -gt 0 ]]; then
    echo ""
    echo "--- Corrupt / truncated (need re-download or source copy) ---"
    cat "$CORRUPT"
fi

if [[ $n_corrupt -eq 0 && $n_notgz -eq 0 ]]; then
    echo "All files OK."
fi
