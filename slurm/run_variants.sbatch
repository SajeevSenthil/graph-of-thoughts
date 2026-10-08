#!/bin/bash
#SBATCH --job-name=got-variants
#SBATCH --partition=workq
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/dist_home/c_prajisha/sajeev/graph-of-thoughts/logs/%x-%j.out

# Runs two full 63-run grids back to back:
#   v2_maxtok4096 : output limit 1536 -> 4096, prompt unchanged (isolates the truncation fix)
#   v3_stagedefs  : same as v2 plus MITRE stage definitions in the shared prompt
# Both are resumable: resubmitting this script skips runs already in each variant's CSV.

set -euo pipefail
cd /dist_home/c_prajisha/sajeev/graph-of-thoughts
export HF_HUB_OFFLINE=1
PY=.venv/bin/python

echo "host=$(hostname) job=${SLURM_JOB_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader

# Fail fast if this node's driver can't run our torch build.
$PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

# Size vLLM's memory share from what is actually free on the allocated GPU,
# capped at 0.85, so a non-SLURM process already on the card can't make startup fail.
read -r FREE TOTAL < <(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | head -1 | tr -d ',')
UTIL=$(awk -v f="$FREE" -v t="$TOTAL" 'BEGIN { u = (f - 2048) / t; if (u > 0.85) u = 0.85; printf "%.2f", u }')
echo "free=${FREE}MiB total=${TOTAL}MiB -> gpu_memory_utilization=${UTIL}"

COMMON=(--max-tokens 4096 --gpu-mem-util "$UTIL")
$PY -u src/run_grid.py --variant v2_maxtok4096 --prompt-variant base "${COMMON[@]}"
$PY -u src/run_grid.py --variant v3_stagedefs --prompt-variant stagedefs "${COMMON[@]}"

echo "done: $(date)"
