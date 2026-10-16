#!/bin/bash
# ===========================================================================
# The contributed `symbolic` classifier on all 29 leave-one-family-out folds.
#
#   cd /e/project1/scifi/thoeppe/genomeoceanbgc
#   sbatch experiments/family_symbolic.sbatch
#
# Two slow sparse fits per fold (train, then train+val) at the frozen stage-3
# configuration n_kmers=150000 / l2=3.0 / max_iter=2500. No per-fold
# hyperparameter sweep -- see the deviation note in family_symbolic.py.
#
# CPU-only and unavoidably so: symbolic_nrps_logistic.py is hand-written numpy
# plus scipy L-BFGS, with no GPU path. No --gres is requested; booster nodes are
# exclusive, so all 288 cores are allocated either way.
#
# NEVER run this on a login node -- a single fold saturates the cores it is given.
#
# Re-runnable: folds with an existing <fold>_symbolic.json are skipped.
# ===========================================================================
#SBATCH --job-name=bgc-family-symbolic
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
    python -u experiments/family_symbolic.py --split-name "${SPLIT_NAME:-family}" --shard "$i" --num-shards "$N" \
        > "${GO_SCRATCH}/logs/family_symbolic_${SPLIT_NAME:-family}_shard${i}.log" 2>&1 &
done
wait

echo ">>> all shards done"
ls -1 "${REPO}/experiments/results/${SPLIT_NAME:-family}/" | grep -c '_symbolic.json$' \
    | xargs echo "folds with a symbolic result:"
