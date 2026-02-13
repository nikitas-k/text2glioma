#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# gadi_torchrun.sh — Gadi-specific torchrun wrapper that activates the
# monai venv on every node before launching.  Drop-in replacement for
# torchrun_nccl.sh from NCI-ai-ml.
#
# Usage (from a PBS script):
#   scripts/gadi_torchrun.sh scripts/launch_stage1_ddp.py --config ...
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV="/g/data/hl36/nk9793/venv/monai"

NNODES=$((PBS_NCPUS / PBS_NCI_NCPUS_PER_NODE))
if [ "$NNODES" -lt 1 ]; then
    NNODES=1
fi

PROC_PER_NODE=$((PBS_NGPUS / NNODES))
MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
CWD=$(pwd)

user_cfg="${PBS_O_WORKDIR}/.cfg_${PBS_JOBID}"
mkdir -p "${user_cfg}"

user_job="${user_cfg}/usrjob.sh"
script_name=$(basename "$1")

cat <<EOF > "${user_job}"
#!/bin/bash
source ~/.bashrc
module use /g/data/dk92/apps/Modules/modulefiles >& /dev/null
module load NCI-ai-ml/23.05 >& /dev/null
module load python3/3.9.2 >& /dev/null
source ${VENV}/bin/activate
cd ${CWD}
SRTHNAME=\${HOSTNAME%.gadi*}
torchrun \\
    --nnodes=${NNODES} \\
    --nproc_per_node=${PROC_PER_NODE} \\
    --rdzv_id=100 \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=${MASTER_ADDR}:29400 \\
    $@ \\
    >& output.${script_name}.\${SRTHNAME}.log
EOF

chmod u+x "${user_job}"

for inode in $(seq 1 "$PBS_NCI_NCPUS_PER_NODE" "$PBS_NCPUS"); do
    pbsdsh -n "$inode" "${user_job}" &
done
wait
