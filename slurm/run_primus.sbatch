#!/bin/bash
#SBATCH --job-name=got-primus
#SBATCH --partition=workq
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/dist_home/c_prajisha/sajeev/graph-of-thoughts/logs/%x-%j.out


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
$PY -u src/run_grid.py --variant primus_v4 --models llama-primus-reasoning --prompt-variant stagedefs --triage --triage-model llama3.1-8b "${COMMON[@]}"

echo "done: $(date)"
