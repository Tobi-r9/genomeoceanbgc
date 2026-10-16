#!/bin/bash
# ===========================================================================
# Frozen GenomeOcean-4B-bgcFM embeddings for the BGC constructs.
# One process per GPU, 4 disjoint shards in parallel on a single booster node.
#
#   cd /e/project1/scifi/thoeppe/genomeoceanbgc
#   sbatch experiments/embed.sbatch
#
# Prerequisites (login node):
#   python experiments/build_nt_dataset.py
#   python experiments/prepare_data.py
#
# Measured ~5.2 seq/s per GH200 at ~4.3k tokens/sequence, peak ~16 GiB, so
# 15,985 sequences over 4 GPUs is ~13 min. Nothing is trained here: the model is
# frozen and only forward passes run, which is why 4 independent processes on one
# node is the right shape.
# ===========================================================================
#SBATCH --job-name=bgc-embed
#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --output=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.out
#SBATCH --error=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.err

set -euo pipefail

REPO=/e/project1/scifi/thoeppe/genomeoceanbgc
source /e/project1/scifi/thoeppe/envs/bgc/activate.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false   # one process per GPU already saturates

cd "${REPO}"
NSHARDS=4
for i in $(seq 0 $((NSHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python -u experiments/embed_dataset.py \
        --shard "$i" --num-shards "$NSHARDS" &
done
wait

echo ">>> all shards done"
ls -la "${GO_SCRATCH}/bgc_clf/emb/"
