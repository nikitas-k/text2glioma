#!/bin/bash
# Submit K PBS jobs, each running scripts/run_synthesise_nonsense.py on a
# contiguous slice of CASE_INDICES. The full case index list should come from
# the stratification cell of scripts/gadi_synthesise_nonsense.ipynb (paste the
# printed CASE_INDICES here, or supply on the command line).
#
# Usage:
#     bash scripts/submit_synthesise_nonsense_blocks.sh \
#         --cases "1,7,15,22,38,47,55,68,84,101,119,142,157,168,187,203,221,247,278,305" \
#         --block 4
#
# Optional:
#     --steps 50               DDIM steps (default 50)
#     --cfg   1.0,4.5,7.0      CFG sweep (default '1.0,4.5,7.0')
#     --seed  42               seed (default 42)
#     --dry                    print qsub commands but don't submit
#     --name  synth_nonsense   PBS -N base name (default 'synth_nonsense')
#
# Each block becomes one PBS job on 1 GPU. For N=20 subjects with default
# CFG grid (~90 s per call, 84 calls per case), block=4 => ~5 h per job.

set -euo pipefail

CASES=""
BLOCK=4
STEPS=50
CFG="1.0,4.5,7.0"
SEED=42
DRY=0
NAME="synth_nonsense"
PBS_FILE="$(cd "$(dirname "$0")" && pwd)/gadi_synthesise_nonsense_block.pbs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cases) CASES="$2"; shift 2 ;;
        --block) BLOCK="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --cfg)   CFG="$2";   shift 2 ;;
        --seed)  SEED="$2";  shift 2 ;;
        --name)  NAME="$2";  shift 2 ;;
        --dry)   DRY=1;      shift 1 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${CASES}" ]]; then
    echo "error: --cases is required (comma-separated case indices)" >&2
    exit 2
fi
if [[ ! -f "${PBS_FILE}" ]]; then
    echo "error: PBS template not found: ${PBS_FILE}" >&2
    exit 2
fi

# Split comma-separated case list into a bash array.
IFS=',' read -r -a CASE_ARR <<< "${CASES}"
NCASES=${#CASE_ARR[@]}
NBLOCKS=$(( (NCASES + BLOCK - 1) / BLOCK ))

echo "cases         : ${CASES}"
echo "N cases       : ${NCASES}"
echo "block size    : ${BLOCK}"
echo "N blocks      : ${NBLOCKS}"
echo "cfg values    : ${CFG}"
echo "steps         : ${STEPS}"
echo "seed          : ${SEED}"
echo "PBS template  : ${PBS_FILE}"
[[ ${DRY} -eq 1 ]] && echo "*** DRY RUN: qsub commands will be printed but not submitted ***"

for (( i=0; i<NCASES; i+=BLOCK )); do
    SLICE=("${CASE_ARR[@]:$i:$BLOCK}")
    # qsub -v uses commas to separate KEY=VAL pairs, so we can't put commas
    # *inside* a value. Encode the case list with underscores and let
    # run_synthesise_nonsense.py accept either delimiter.
    SLICE_CSV=$(IFS=,; echo "${SLICE[*]}")
    SLICE_ENV=$(IFS=_; echo "${SLICE[*]}")
    # CFG values also contain commas; encode with underscores too.
    CFG_ENV=${CFG//,/_}
    BLOCK_IDX=$(( i / BLOCK ))
    JOB_NAME="${NAME}_b${BLOCK_IDX}"

    QSUB_CMD=(qsub
        -N "${JOB_NAME}"
        -v "T2G_CASE_INDICES=${SLICE_ENV},T2G_STEPS=${STEPS},T2G_CFG_VALUES=${CFG_ENV},T2G_SEED=${SEED}"
        "${PBS_FILE}"
    )

    echo
    echo "=== block ${BLOCK_IDX} : cases=${SLICE_CSV} ==="
    printf '  '; printf '%q ' "${QSUB_CMD[@]}"; echo
    if [[ ${DRY} -eq 0 ]]; then
        "${QSUB_CMD[@]}"
    fi
done
