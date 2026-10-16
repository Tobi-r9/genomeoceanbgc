#!/bin/bash
# ===========================================================================
# Assemble the shareable split folder for external collaborators.
#
#   cd /e/project1/scifi/thoeppe/genomeoceanbgc
#   sbatch experiments/make_release.sbatch
#
# Single-core and only a couple of minutes, but it is submitted rather than run
# interactively: an earlier login-node attempt was SIGKILLed by the node
# watchdog (exit 137) while copying the sequence parquet.
#
# CPU-only, no --gres. Writes to /e/project1/scifi/thoeppe/nrps_splits_v1.
# Idempotent -- rerunning overwrites the folder's contents in place.
# ===========================================================================
#SBATCH --job-name=bgc-make-release
#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --output=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.out
#SBATCH --error=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.err

set -euo pipefail
REPO=/e/project1/scifi/thoeppe/genomeoceanbgc
source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
cd "${REPO}"
python -u experiments/make_release.py "$@"
echo ">>> done"
