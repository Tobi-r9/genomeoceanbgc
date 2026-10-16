#!/bin/bash
# ---------------------------------------------------------------------------
# Pre-stage GenomeOcean checkpoints onto scratch. LOGIN NODE ONLY -- compute
# nodes have no outbound internet.
#
#   bash jupiter/download_model.sh                    # the 4B BGC model (8.5 GB)
#   bash jupiter/download_model.sh 4B 100M            # extra checkpoints too
#
# Downloads into ${GO_SCRATCH}/hf_models/<name>/ as a plain directory (NOT a
# hub-cache blob tree), so the result can be handed to --model_dir and to vLLM
# directly, and so nothing is duplicated between cache and snapshot.
#
# Re-running is cheap: `hf download` verifies against the local metadata in
# <dir>/.cache/huggingface and skips files that are already complete.
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_DIR_ROOT="/e/project1/scifi/thoeppe/envs/bgc"
source "${ENV_DIR_ROOT}/activate.sh" >/dev/null

MODELS=("${@:-bgcFM}")

declare -A REPOS=(
    [bgcFM]="pGenomeOcean/GenomeOcean-4B-bgcFM"
    [4B]="pGenomeOcean/GenomeOcean-4B"
    [500M]="pGenomeOcean/GenomeOcean-500M"
    [100M]="pGenomeOcean/GenomeOcean-100M"
    [detector]="pGenomeOcean/GenomeOcean-4B-Artificial-Detector"
)

for key in "${MODELS[@]}"; do
    repo="${REPOS[$key]:-}"
    if [[ -z "${repo}" ]]; then
        echo "!!! Unknown model key '${key}'. Known: ${!REPOS[*]}" >&2
        exit 1
    fi
    dest="${GO_SCRATCH}/hf_models/$(basename "${repo}")"
    echo ">>> ${repo}  ->  ${dest}"
    hf download "${repo}" --local-dir "${dest}"
done

echo
echo ">>> Staged models:"
du -sh "${GO_SCRATCH}"/hf_models/* 2>/dev/null || true
