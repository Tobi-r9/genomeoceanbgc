#!/bin/bash
# ===========================================================================
# GenomeOcean-4B-bgcFM smoke test -- JUPITER booster (1 node, 4x GH200 96GB)
#
#   cd /e/project1/scifi/thoeppe/genomeoceanbgc/jupiter
#   sbatch test_model.sbatch
#
# The same checks can be run straight on a login node (which has a GH200 and
# internet): `python test_bgc_model.py --with-vllm`. Use this job when the
# login-node GPU is busy, or when you want the 4-GPU tensor-parallel path --
# LLMUtils picks tensor_parallel_size from torch.cuda.device_count(), rounded
# down to a divisor of 12 (the attention-head count), so 4 GPUs -> TP=4.
#
# Prerequisites (login node, once):
#   bash /e/project1/scifi/thoeppe/envs/bgc/setup.sh
#   bash /e/project1/scifi/thoeppe/genomeoceanbgc/jupiter/download_model.sh
# ===========================================================================
#SBATCH --job-name=bgc-smoke
#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.out
#SBATCH --error=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.err

set -euo pipefail

REPO=/e/project1/scifi/thoeppe/genomeoceanbgc
ENV_ROOT=/e/project1/scifi/thoeppe/envs/bgc

source "${ENV_ROOT}/activate.sh"

# Compute nodes have no outbound internet: fail loudly instead of retry-hanging
# if download_model.sh was not run.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Only one GPU is needed for the transformers checks; the vLLM check below uses
# whatever CUDA_VISIBLE_DEVICES exposes. Restrict it to 1 GPU so TP=1 keeps the
# test fast and its memory footprint predictable.
export CUDA_VISIBLE_DEVICES=0

if [[ ! -d "${GO_BGC_MODEL}" ]]; then
    echo "!!! ${GO_BGC_MODEL} missing -- run jupiter/download_model.sh on a login node" >&2
    exit 1
fi

nvidia-smi
cd "${REPO}"
# -u: without it python block-buffers into the slurm .out file and the log stays
# empty until the job ends, which is indistinguishable from a hang.
srun python -u jupiter/test_bgc_model.py --with-vllm --gpu-mem-util 0.35
