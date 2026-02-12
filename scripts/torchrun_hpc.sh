#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# torchrun_hpc.sh — wrapper for launching distributed text2glioma
# training on PBS / Torque clusters or bare-metal nodes.
#
# Usage (interactive / bare-metal):
#   bash scripts/torchrun_hpc.sh \
#       --nproc_per_node 4 \
#       -m text2glioma.training.train_stage1_ddp \
#       --config configs/stage1.yaml --run_dir /scratch/runs/
#
# Usage (PBS qsub):
#   qsub scripts/torchrun_hpc.sh \
#       -v TRAIN_ARGS="--nproc_per_node 4 \
#           -m text2glioma.training.train_stage1_ddp \
#           --config configs/stage1.yaml --run_dir /scratch/runs/"
#
# PBS directives (adjust to your cluster):
#PBS -N t2g-train
#PBS -q gpu
#PBS -l nodes=1:ppn=16:gpus=4
#PBS -l mem=128gb
#PBS -l walltime=72:00:00
#PBS -o logs/t2g-train.o${PBS_JOBID}
#PBS -e logs/t2g-train.e${PBS_JOBID}
#PBS -j oe
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Change to submission directory when run via qsub ────────────────
if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    cd "$PBS_O_WORKDIR"
fi

# When submitted via ``qsub -v TRAIN_ARGS="..."`` the extra arguments
# are passed through the environment variable instead of $@.
if [[ $# -eq 0 && -n "${TRAIN_ARGS:-}" ]]; then
    set -- $TRAIN_ARGS
fi

# ── Cluster auto-detection ──────────────────────────────────────────
if [[ -n "${PBS_JOBID:-}" ]]; then
    echo "PBS detected.  Job ID: $PBS_JOBID"
    export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
    export MASTER_PORT=${MASTER_PORT:-29500}
    NNODES=$(sort -u "$PBS_NODEFILE" | wc -l)

    # Determine this node's rank among the unique nodes
    _THIS_HOST=$(hostname -s)
    NODE_RANK=0
    while IFS= read -r _h; do
        if [[ "$_h" == "$_THIS_HOST" ]]; then
            break
        fi
        NODE_RANK=$((NODE_RANK + 1))
    done < <(sort -u "$PBS_NODEFILE")

    echo "  MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
    echo "  NNODES=$NNODES  NODE_RANK=$NODE_RANK"
# Bare-metal / interactive
else
    echo "No scheduler detected — running locally."
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-29500}
    NNODES=${NNODES:-1}
    NODE_RANK=${NODE_RANK:-0}
fi

# ── Activate environment (edit to match your setup) ─────────────────
# Uncomment one of the following:
# module load cuda/12.1 anaconda3
# conda activate text2glioma
# source /path/to/venv/bin/activate

# ── Resolve nproc_per_node if not supplied ──────────────────────────
# If --nproc_per_node is in $@, torchrun will use it directly.
# Otherwise, count GPUs visible to this node via nvidia-smi.
if [[ ! " $* " =~ " --nproc_per_node " ]] && [[ ! " $* " =~ " --nproc-per-node " ]]; then
    if command -v nvidia-smi &>/dev/null; then
        NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        NPROC=1
    fi
    EXTRA_TORCHRUN="--nproc_per_node=${NPROC}"
else
    EXTRA_TORCHRUN=""
fi

# ── Launch ──────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────"
echo "torchrun  --nnodes=$NNODES  --node_rank=$NODE_RANK"
echo "          --master_addr=$MASTER_ADDR  --master_port=$MASTER_PORT"
echo "          $EXTRA_TORCHRUN"
echo "          $*"
echo "──────────────────────────────────────────────"

torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    ${EXTRA_TORCHRUN} \
    "$@"
