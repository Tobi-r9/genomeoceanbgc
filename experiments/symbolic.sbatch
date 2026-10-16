#!/bin/bash
# Hyperparameter sweep for the contributed symbolic NRPS classifier, on our split.
#   sbatch experiments/symbolic.sbatch
# 24 configurations, 6 shards in parallel. CPU-only.
#SBATCH --job-name=bgc-symbolic
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
source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
cd /e/project1/scifi/thoeppe/genomeoceanbgc

N=6
export OMP_NUM_THREADS=$(( 128 / N ))
for i in $(seq 0 $((N - 1))); do
    python -u experiments/run_symbolic.py --mode sweep --grid "${GRID:-stage1}" --shard "$i" --num-shards "$N" \
        > "${GO_SCRATCH}/logs/symbolic_${GRID:-stage1}_shard${i}.log" 2>&1 &
done
wait
echo ">>> sweep done"
