#!/bin/bash
# ===========================================================================
# REPORT.md's evaluation re-run on all 29 leave-one-family-out folds.
#
#   cd /e/project1/scifi/thoeppe/genomeoceanbgc
#   sbatch experiments/family_eval.sbatch
#
# CPU-only, and it has to be: the heads are scikit-learn LogisticRegression /
# MLPClassifier / HistGradientBoostingClassifier, none of which have a GPU
# backend, and this environment has no xgboost, lightgbm, cuML or cupy. No
# --gres is requested for that reason. It costs nothing -- booster nodes are
# allocated exclusively, so this job gets all 288 cores either way (verified
# with sbatch --test-only, identical allocation with and without --gres).
#
# NEVER run this sweep on a login node. It saturates every core it is given.
#
# Sharded over folds rather than over feature blocks (unlike sweep.sbatch)
# because each fold reloads the embeddings and rebuilds its own bigram
# vocabulary, so the fold is the natural unit of work.
#
# Re-runnable: family_eval.py skips folds whose <fold>_test.csv already exists,
# so a job that hits the walltime can simply be resubmitted.
# ===========================================================================
#SBATCH --job-name=bgc-family-eval
#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --time=04:00:00
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
    python -u experiments/family_eval.py --split-name "${SPLIT_NAME:-family}" --shard "$i" --num-shards "$N" \
        > "${GO_SCRATCH}/logs/family_eval_${SPLIT_NAME:-family}_shard${i}.log" 2>&1 &
done
wait

echo ">>> all shards done"
ls -1 "${REPO}/experiments/results/${SPLIT_NAME:-family}/" | grep -c '_test.csv$' \
    | xargs echo "folds with a test table:"
