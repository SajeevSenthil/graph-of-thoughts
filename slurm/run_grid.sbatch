#!/bin/bash
#SBATCH --job-name=got-grid
#SBATCH --partition=workq
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/dist_home/c_prajisha/sajeev/graph-of-thoughts/logs/%x-%j.out


# Generic single-variant grid run; all arguments are passed through to
# src/run_grid.py, e.g.
#   sbatch slurm/run_grid.sbatch --variant v4_triage --max-tokens 4096 --prompt-variant stagedefs --triage
# Resumable: resubmitting skips runs already in the variant's CSV.

set -euo pipefail
cd /dist_home/c_prajisha/sajeev/graph-of-thoughts
export HF_HUB_OFFLINE=1
PY=.venv/bin/python

echo "host=$(hostname) job=${SLURM_JOB_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader

# Fail fast if this node's driver can't run our torch build.
$PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

# Size vLLM's memory share from what is actually free on the allocated GPU,
# capped at UTIL_CAP (default 0.85), so a non-SLURM process already on the card
# can't make startup fail. The 70B model needs more: submit with
#   sbatch --export=ALL,UTIL_CAP=0.92 ...
read -r FREE TOTAL < <(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | head -1 | tr -d ',')
UTIL=$(awk -v f="$FREE" -v t="$TOTAL" -v c="${UTIL_CAP:-0.85}" 'BEGIN { u = (f - 2048) / t; if (u > c) u = c; printf "%.2f", u }')
echo "free=${FREE}MiB total=${TOTAL}MiB -> gpu_memory_utilization=${UTIL}"


$PY -u src/run_grid.py "$@" --gpu-mem-util "$UTIL"

echo "done: $(date)"
