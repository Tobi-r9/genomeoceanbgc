#!/bin/bash
# ===========================================================================
# Validation sweep over feature blocks x lightweight classifiers.
# One process per feature block, all in parallel on a single node.
#
#   cd /e/project1/scifi/thoeppe/genomeoceanbgc
#   sbatch experiments/sweep.sbatch
#
# CPU-only (the embeddings are already extracted and frozen), but booster nodes
# are allocated exclusively and give 288 cores, which sklearn's OpenMP-threaded
# HistGradientBoosting uses well. The blocks differ hugely in cost -- 3 features
# for `len` versus 4456 for `emb_mean+modules` -- so running them concurrently
# rather than sequentially is what keeps this short.
#
# Each process writes results/val_sweep_<block>.csv; --final later selects the
# global winner across all of them.
# ===========================================================================
#SBATCH --job-name=bgc-sweep
#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.out
#SBATCH --error=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.err

set -euo pipefail

REPO=/e/project1/scifi/thoeppe/genomeoceanbgc
source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd "${REPO}"
BLOCKS=(modules_bigram modules+bigram)

# Divide the cores between the concurrent processes instead of letting each
# spawn 288 OpenMP threads and thrash.
export OMP_NUM_THREADS=$(( 128 / ${#BLOCKS[@]} ))
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} per block, ${#BLOCKS[@]} blocks in parallel"

for b in "${BLOCKS[@]}"; do
    python -u experiments/train_classifiers.py \
        --blocks "$b" --tag "_${b}" > "${GO_SCRATCH}/logs/sweep_${b}.log" 2>&1 &
done
wait

echo ">>> all blocks done"
for b in "${BLOCKS[@]}"; do
    echo "--- $b"
    tail -3 "${GO_SCRATCH}/logs/sweep_${b}.log"
done
