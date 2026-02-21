#!/usr/bin/env bash
# Fix files that have .nii.gz extension but are actually uncompressed .nii.
# Detects via magic bytes (gzip starts with 1f 8b), then gzips in-place.
#
# Uses GNU parallel (or xargs -P) to compress many files concurrently.
# pigz is preferred over gzip (~4x faster per file on multi-core).
#
# Usage:
#   bash scripts/fix_fake_gzip.sh /g/data/hl36/mhf/monai/Task03_BrainTumourDx/imagesTr
#   bash scripts/fix_fake_gzip.sh /g/data/hl36/mhf/monai/Task03_BrainTumourDx/labelsTr
#
# Dry run (just list affected files):
#   bash scripts/fix_fake_gzip.sh /g/data/hl36/mhf/monai/Task03_BrainTumourDx/imagesTr --dry-run
#
# Control parallelism (default: nproc):
#   JOBS=16 bash scripts/fix_fake_gzip.sh /path/to/dir

set -euo pipefail

DIR="${1:?Usage: $0 <directory> [--dry-run]}"
DRY_RUN=false
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=true

JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)}"

# Prefer pigz (parallel gzip) over gzip
if command -v pigz &>/dev/null; then
    GZIP_CMD="pigz"
else
    GZIP_CMD="gzip"
fi
echo "Using: $GZIP_CMD, jobs=$JOBS"

# ── Step 1: Find all fake gzip files (parallel scan) ─────────────────────
TMPFILE=$(mktemp)
ALLFILES=$(mktemp)
trap "rm -f $TMPFILE $ALLFILES" EXIT

# Collect file list
find "$DIR" -name '*.nii.gz' > "$ALLFILES"
count=$(wc -l < "$ALLFILES")
echo "Found: $count .nii.gz files, scanning in parallel..."

# Check magic bytes in parallel — outputs only non-gzip files
check_magic() {
    local f="$1"
    local magic
    magic=$(od -A n -t x1 -N 2 "$f" | tr -d ' ')
    [[ "$magic" != "1f8b" ]] && echo "$f"
    return 0
}
export -f check_magic

if command -v parallel &>/dev/null; then
    parallel -j "$JOBS" check_magic :::: "$ALLFILES" > "$TMPFILE"
else
    cat "$ALLFILES" | xargs -P "$JOBS" -I{} bash -c 'check_magic "$@"' _ {} > "$TMPFILE"
fi

fake=$(wc -l < "$TMPFILE")
echo "Fake gzip: $fake files"

if [[ $fake -eq 0 ]]; then
    echo "Nothing to fix."
    exit 0
fi

if $DRY_RUN; then
    echo ""
    echo "Affected files:"
    cat "$TMPFILE"
    echo ""
    echo "(dry run — no files were modified)"
    exit 0
fi

# ── Step 2: Fix in parallel ──────────────────────────────────────────────
# Each worker: mv foo.nii.gz foo.nii && pigz/gzip foo.nii
fix_one() {
    local f="$1"
    local tmp="${f%.gz}"
    mv "$f" "$tmp"
    $GZIP_CMD "$tmp"
}
export -f fix_one
export GZIP_CMD

if command -v parallel &>/dev/null; then
    parallel -j "$JOBS" --bar fix_one :::: "$TMPFILE"
else
    # Fallback: xargs -P
    cat "$TMPFILE" | xargs -P "$JOBS" -I{} bash -c 'fix_one "$@"' _ {}
fi

echo ""
echo "Fixed: $fake files (using $GZIP_CMD, $JOBS parallel jobs)"
