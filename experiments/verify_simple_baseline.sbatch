#!/bin/bash
#SBATCH --job-name=bgc-verify-simple
#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.out
#SBATCH --error=/e/scratch/profound/thoeppe/genomeocean/logs/%x-%j.err
set -euo pipefail
source /e/project1/scifi/thoeppe/envs/bgc/activate.sh
export OMP_NUM_THREADS=16
cd /e/project1/scifi/thoeppe/nrps_splits_v1
for FOLD in TxlS XenS GxpS; do
  python - "$FOLD" <<'PY'
import sys, pandas as pd
fold=sys.argv[1]
con=pd.read_csv('constructs.csv'); fs=pd.read_csv('folds/family_folds.csv')
s=con['id'].map(dict(zip(fs[fs.family==fold]['id'], fs[fs.family==fold]['split'])))
out=con[['modules','produced']].copy(); out['split']=s.values
out.to_csv(f'/tmp/{fold}.csv', index=False)
print(f'--- {fold} ---', flush=True)
PY
  python -u baseline.py "/tmp/${FOLD}.csv"
done
echo ">>> done"
