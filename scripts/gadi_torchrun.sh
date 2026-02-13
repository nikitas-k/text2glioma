#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# gadi_torchrun.sh — Gadi-specific torchrun wrapper that activates the
# monai venv on every node before launching.  Drop-in replacement for
# torchrun_nccl.sh from NCI-ai-ml.
#
# Usage (from a PBS script):
#   scripts/gadi_torchrun.sh scripts/launch_stage1_ddp.py --config ...
# ─────────────────────────────────────────────────────────────────────

VENV="/g/data/hl36/nk9793/venv/monai"

# ── Compute topology from PBS variables ─────────────────────────────
# PBS_NCI_NCPUS_PER_NODE is set by Gadi's PBS; fall back to PBS_NCPUS
CPUS_PER_NODE=${PBS_NCI_NCPUS_PER_NODE:-${PBS_NCPUS:-48}}
NNODES=$((${PBS_NCPUS:-48} / CPUS_PER_NODE))
if [ "$NNODES" -lt 1 ]; then
    NNODES=1
fi

PROC_PER_NODE=$((${PBS_NGPUS:-4} / NNODES))
MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
CWD=$(pwd)

echo "gadi_torchrun.sh: NNODES=$NNODES  PROC_PER_NODE=$PROC_PER_NODE  MASTER=$MASTER_ADDR"
echo "gadi_torchrun.sh: CPUS_PER_NODE=$CPUS_PER_NODE  PBS_NCPUS=${PBS_NCPUS:-unset}"
echo "gadi_torchrun.sh: args = $@"

# ── Build per-node launcher script ─────────────────────────────────
user_cfg="${PBS_O_WORKDIR}/.cfg_${PBS_JOBID}"
mkdir -p "${user_cfg}"

user_job="${user_cfg}/usrjob.sh"
script_name=$(basename "$1")

# Capture all arguments now so they are baked into the script
ALL_ARGS="$*"

cat > "${user_job}" << ENDSCRIPT
#!/bin/bash
source ~/.bashrc
module use /g/data/dk92/apps/Modules/modulefiles >& /dev/null
module load NCI-ai-ml/23.05 >& /dev/null
module load python3/3.9.2 >& /dev/null
source ${VENV}/bin/activate
cd ${CWD}
SRTHNAME=\${HOSTNAME%.gadi*}
echo "Node \${SRTHNAME}: starting torchrun at \$(date)"
torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${PROC_PER_NODE} \
    --rdzv_id=${PBS_JOBID%%.*} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:29400 \
    ${ALL_ARGS} \
    >& output.${script_name}.\${SRTHNAME}.log
echo "Node \${SRTHNAME}: torchrun exited with status \$? at \$(date)"
ENDSCRIPT

chmod u+x "${user_job}"

echo "gadi_torchrun.sh: generated ${user_job}"
cat "${user_job}"
echo "─────────────────────────────────────────────"

# ── Launch on every node via pbsdsh ─────────────────────────────────
for inode in $(seq 1 $CPUS_PER_NODE ${PBS_NCPUS:-48}); do
    echo "gadi_torchrun.sh: pbsdsh -n $inode ${user_job}"
    pbsdsh -n $inode "${user_job}" &
done
wait
