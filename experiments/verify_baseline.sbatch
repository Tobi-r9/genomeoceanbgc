#!/bin/bash
# Verify release_files/baseline.py reproduces the published modules+bigram number
# (median AP 0.408 over the 12 primary family folds).
#SBATCH --job-name=bgc-verify-baseline
#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:40:00
#SBATCH --output=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.out
#SBATCH --error=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.err
set -euo pipefail
source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
export OMP_NUM_THREADS=16
cd /e/project1/scifi/thoeppe/nrps_splits_v1
python -u baseline.py --primary-only --out /tmp/baseline_pred.csv
python -u baseline.py --primary-only --module-vocab trainval --out /tmp/baseline_pred_tv.csv
echo "=== scoring through evaluate.py ==="
python -u evaluate.py --pred /tmp/baseline_pred.csv --primary-only --compare thesis --n-boot 500
echo ">>> done"
