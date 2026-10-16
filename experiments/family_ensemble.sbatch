#!/bin/bash
# ===========================================================================
# The NRPS-Models-and-data simple method on all 29 leave-one-family-out folds.
#
#   cd /e/project1/scifi/thoeppe/genomeoceanbgc
#   sbatch experiments/family_ensemble.sbatch
#
# CPU-only. XGBoost is built with tree_method=hist on CPU here to match the
# thesis parameters exactly; no --gres is requested. Booster nodes are
# exclusive, so all 288 cores are allocated either way.
#
# NEVER run this on a login node.
#
# 6 shards rather than 8: the char 3-5-gram TF-IDF over ~5,000-residue proteins
# builds a large intermediate vocabulary before pruning to 1000 features, so
# each shard needs several GB.
#
# Re-runnable: folds with an existing <fold>_ensemble.json are skipped.
# ===========================================================================
#SBATCH --job-name=bgc-family-ensemble
#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --time=06:00:00
#SBATCH --output=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.out
#SBATCH --error=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.err

set -euo pipefail

REPO=/e/project1/scifi/thoeppe/genomeoceanbgc
source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd "${REPO}"
N=8
export OMP_NUM_THREADS=$(( 256 / N ))
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} per shard, ${N} shards in parallel"

for i in $(seq 0 $((N - 1))); do
    python -u experiments/family_ensemble.py --split-name "${SPLIT_NAME:-family}" --shard "$i" --num-shards "$N" \
        > "${GO_SCRATCH}/logs/family_ensemble_${SPLIT_NAME:-family}_shard${i}.log" 2>&1 &
done
wait

echo ">>> all shards done"
ls -1 "${REPO}/experiments/results/${SPLIT_NAME:-family}/" | grep -c '_ensemble.json$' \
    | xargs echo "folds with a thesis result:"
