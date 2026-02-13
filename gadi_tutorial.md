# Training on NCI Gadi

This guide covers running distributed `text2glioma` training on
[NCI Gadi](https://nci.org.au/our-systems/hpc/gadi) using PBS Pro and the
`NCI-ai-ml/23.05` module stack.

> **Prerequisites** — a virtual environment with `text2glioma` installed
> and access to the `/g/data/dk92`, `/g/data/hl36`, `/scratch/vp06`
> (or equivalent) storage allocations.

---

## Environment setup (one-time)

```bash
# On a Gadi login node
module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-ai-ml/23.05
module load python3/3.9.2

python3 -m venv /g/data/hl36/$USER/venv/monai
source /g/data/hl36/$USER/venv/monai/bin/activate

# Install text2glioma into the venv
cd /path/to/text2glioma
pip install .
```

If you need the evaluation extras (FID, CLIP score, etc.):

```bash
pip install ".[eval]"
```

---

## How `torchrun_nccl.sh` works

Gadi provides `torchrun_nccl.sh` as part of the `NCI-ai-ml` module.
It is a wrapper that:

1. Reads PBS variables (`PBS_NCPUS`, `PBS_NGPUS`, `PBS_NCI_NCPUS_PER_NODE`,
   `PBS_NODEFILE`) to compute `NNODES`, `PROC_PER_NODE`, and `MASTER_ADDR`.
2. Writes a per-node launcher script.
3. Uses `pbsdsh` to execute that script on every allocated node, each
   calling `torchrun` with the appropriate rendezvous settings
   (`--rdzv_backend=c10d`, `--rdzv_endpoint=<master>:29400`).

All arguments you pass after `torchrun_nccl.sh` are forwarded directly
to `torchrun`.  This means you use `-m <module>` syntax to launch
`text2glioma` entry points:

```
torchrun_nccl.sh -m text2glioma.training.train_stage1_ddp --config ...
```

`torchrun` sets the `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`,
and `MASTER_PORT` environment variables that our DDP scripts read in
`setup_distributed()`.

---

## Stage 1 — VAE training

### Submit

```bash
qsub scripts/gadi_stage1.pbs
```

### What it does

- Requests **4 GPUs** on the `gpuhopper` queue (48 CPUs, 1 TB RAM).
- Activates the `monai` venv.
- Launches `train_stage1_ddp` via `torchrun_nccl.sh -m`.
- Logs to `~/job_logs/<JOBID>.log`.

### Template breakdown

```bash
#!/bin/bash
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=1022GB
#PBS -l jobfs=250GB
#PBS -q gpuhopper
#PBS -P vp06
#PBS -l walltime=48:00:00
#PBS -l storage=gdata/dk92+scratch/vp06+gdata/vp06+gdata/hl36+scratch/hl36
#PBS -l wd
#PBS -j oe

module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-ai-ml/23.05
module load python3/3.9.2

source /g/data/hl36/nk9793/venv/monai/bin/activate

torchrun_nccl.sh \
    -m text2glioma.training.train_stage1_ddp \
    --config  configs/stage1.yaml \
    --run_dir /scratch/vp06/$USER/runs \
    --data_dir /scratch/vp06/$USER/data \
    --batch_size 2 \
    --num_epochs 300 \
    --val_interval 5 \
    --pin_memory \
    2>&1 > ${HOME}/job_logs/${PBS_JOBID}.log
```

Edit the paths at the top of `scripts/gadi_stage1.pbs` to match your
allocation.

### Key arguments

| Flag | Default | Notes |
|------|---------|-------|
| `--config` | *(required)* | `configs/stage1.yaml` |
| `--run_dir` | *(required)* | Root output dir — use `/scratch/` for speed |
| `--data_dir` | `./data` | Where `DecathlonDataset` downloads BraTS |
| `--batch_size` | `2` | Per-GPU; effective = `batch_size × ngpus` |
| `--num_epochs` | `300` | |
| `--resume` | off | Add to resume from `checkpoint.pth` |

### Resume a failed/timed-out job

Add `--resume` to the `torchrun_nccl.sh` line in the PBS script
(or copy and edit):

```bash
torchrun_nccl.sh \
    -m text2glioma.training.train_stage1_ddp \
    --config  configs/stage1.yaml \
    --run_dir /scratch/vp06/$USER/runs \
    --data_dir /scratch/vp06/$USER/data \
    --resume \
    2>&1 > ${HOME}/job_logs/${PBS_JOBID}.log
```

---

## Stage 2 — LDM training

Stage 2 requires a **trained Stage-1 VAE** checkpoint.

### Submit

```bash
qsub scripts/gadi_stage2.pbs
```

### Template breakdown

```bash
#!/bin/bash
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=1022GB
#PBS -l jobfs=250GB
#PBS -q gpuhopper
#PBS -P vp06
#PBS -l walltime=48:00:00
#PBS -l storage=gdata/dk92+scratch/vp06+gdata/vp06+gdata/hl36+scratch/hl36
#PBS -l wd
#PBS -j oe

module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-ai-ml/23.05
module load python3/3.9.2

source /g/data/hl36/nk9793/venv/monai/bin/activate

torchrun_nccl.sh \
    -m text2glioma.training.train_stage2_ddp \
    --config       configs/ldm.yaml \
    --stage1_config configs/stage1.yaml \
    --stage1_uri   /scratch/vp06/$USER/runs/text2glioma/autoencoder_stage1/output/models/best_model.pth \
    --run_dir      /scratch/vp06/$USER/runs \
    --data_dir     /scratch/vp06/$USER/data \
    --batch_size 2 \
    --num_epochs 250 \
    --val_interval 5 \
    --pin_memory \
    2>&1 > ${HOME}/job_logs/${PBS_JOBID}.log
```

### Key arguments (Stage 2 only)

| Flag | Default | Notes |
|------|---------|-------|
| `--config` | *(required)* | `configs/ldm.yaml` |
| `--stage1_config` | *(required)* | `configs/stage1.yaml` |
| `--stage1_uri` | *(required)* | Path to pre-trained VAE `.pth` |
| `--train_spec` | `impression` | Text field: `impression` or `findings` |
| `--scale_factor` | `1.0` | Latent scaling |
| `--mask_dropout_p` | from config | Override mask dropout |
| `--text_dropout_p` | from config | Override text dropout |
| `--cache_dir` | `None` | HuggingFace cache (set if no internet) |

---

## Multi-node jobs

To run across **N nodes**, change the PBS resource request:

```bash
#PBS -l ncpus=96         # 48 × 2 nodes
#PBS -l ngpus=8          # 4 × 2 nodes
```

`torchrun_nccl.sh` will automatically:

- Detect `NNODES = PBS_NCPUS / PBS_NCI_NCPUS_PER_NODE`
- Set `PROC_PER_NODE = PBS_NGPUS / NNODES`
- Select `MASTER_ADDR` from the first line of `$PBS_NODEFILE`
- Launch on every node via `pbsdsh`

No changes are needed in the training arguments.

---

## Storage layout

Recommended directory structure on Gadi:

```
/scratch/vp06/$USER/
├── data/                          # DecathlonDataset (BraTS download)
│   └── Task01_BrainTumour/
└── runs/
    └── text2glioma/
        ├── autoencoder_stage1/    # Stage 1 outputs
        │   └── output/
        │       ├── models/        # best_model.pth, final_model.pth
        │       ├── logs/          # TensorBoard logs
        │       └── cache/
        └── ldm_stage2/            # Stage 2 outputs
            └── output/
                ├── models/
                └── logs/
```

Use `/scratch/` for training runs (fast parallel filesystem).
Back up final checkpoints to `/g/data/` for long-term storage.

---

## Monitoring

### Job status

```bash
qstat -u $USER
```

### Training logs

```bash
tail -f ~/job_logs/<JOBID>.gadi-pbs.log
```

`torchrun_nccl.sh` also writes per-node logs to the working directory as
`output.<script>.<hostname>.log`.

### TensorBoard

On a login node (or via SSH tunnel):

```bash
module load python3/3.9.2
source /g/data/hl36/$USER/venv/monai/bin/activate
tensorboard --logdir /scratch/vp06/$USER/runs/text2glioma/ --port 6006
```

Then forward port 6006 from your local machine:

```bash
ssh -L 6006:localhost:6006 nk9793@gadi.nci.org.au
```

---

## Using a custom dataset (JSON datalist)

If your data is not the MSD BraTS `DecathlonDataset`, you can point the
training scripts at any set of 4-channel NIfTI images + segmentation
labels via a **JSON datalist**.

### 1. Generate the datalist

Use the helper script on a login node:

```bash
source /g/data/hl36/nk9793/venv/monai/bin/activate

python scripts/make_datalist.py \
    --images '/g/data/hl36/mhf/monai/Task03_BrainTumourDx/imagesTr/nnUNetv2-*_full.nii' \
    --labels '/g/data/hl36/mhf/monai/Task03_BrainTumourDx/labelsTr/nnUNetv2-*.nii.gz' \
    --val_frac 0.2 \
    -o datalist_task03.json
```

The resulting JSON looks like:

```json
{
  "training": [
    {"image": "/g/data/.../nnUNetv2-00001_full.nii", "label": "/g/data/.../nnUNetv2-00001.nii.gz"},
    ...
  ],
  "validation": [...]
}
```

### 2. Update the PBS scripts

In `gadi_stage1.pbs` / `gadi_stage2.pbs`, set the `DATALIST` variable:

```bash
DATALIST=${HOME}/text2glioma/datalist_task03.json
```

The script will automatically pass `--datalist` and `--no_channel_reorder`
instead of `--data_dir`.

### 3. Channel ordering

| Flag | Behaviour |
|------|-----------|
| *(default)* | Reorders channels from MSD order (FLAIR/T1/T1CE/T2) → pipeline order (T1/T1CE/T2/FLAIR) |
| `--no_channel_reorder` | Skips reorder — use this when your NIfTIs are **not** in MSD order |

If your custom data already has channels in the order T1 / T1CE / T2 / FLAIR,
add `--no_channel_reorder`.  If your data follows MSD ordering, omit the flag.

### 4. Writing your own datalist by hand

You can also write the JSON manually — just follow this schema:

```json
{
  "training": [
    {"image": "/absolute/path/to/img1.nii.gz", "label": "/absolute/path/to/seg1.nii.gz"},
    {"image": "/absolute/path/to/img2.nii.gz", "label": "/absolute/path/to/seg2.nii.gz"}
  ],
  "validation": [
    {"image": "/absolute/path/to/img3.nii.gz", "label": "/absolute/path/to/seg3.nii.gz"}
  ]
}
```

> **Stage 1** only reads the `"image"` key; `"label"` is ignored but
> harmless. **Stage 2** uses both `"image"` and `"label"`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: text2glioma` | Re-run `pip install .` in the venv |
| NCCL timeout | Check `#PBS -l storage=` includes all required mounts |
| OOM on GPU | Reduce `--batch_size` to 1 |
| Checkpoint corrupt after timeout | Delete `checkpoint.pth`, restart, or use `best_model.pth` |
| `torchrun_nccl.sh: command not found` | Ensure `module load NCI-ai-ml/23.05` is in the PBS script |
| BraTS download hangs | Download on login node first: `python -c "from monai.apps import DecathlonDataset; DecathlonDataset('./data', 'Task01_BrainTumour', download=True)"` |
