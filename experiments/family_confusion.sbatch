#!/bin/bash
# ===========================================================================
# Confusion tables and balanced 50/50 subsets for the primary family folds.
#
#   cd /e/project1/scifi/thoeppe/genomeoceanbgc
#   sbatch experiments/family_confusion.sbatch
#
# CPU-only (scikit-learn logistic regression, no GPU backend), so no --gres.
# Booster nodes are exclusive, so all 288 cores are allocated regardless.
#
# NEVER run this on a login node.
#
# Single process: the run is dominated by two logistic-regression fits per fold
# on the 3072-d embedding, and sklearn already threads those internally.
# ===========================================================================
#SBATCH --job-name=bgc-family-confusion
#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --time=02:00:00
#SBATCH --output=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.out
#SBATCH --error=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.err

set -euo pipefail

REPO=/e/project1/scifi/thoeppe/genomeoceanbgc
source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=64

cd "${REPO}"
python -u experiments/family_confusion.py \
    > "${GO_SCRATCH}/logs/family_confusion.log" 2>&1

echo ">>> done"
tail -20 "${GO_SCRATCH}/logs/family_confusion.log"
