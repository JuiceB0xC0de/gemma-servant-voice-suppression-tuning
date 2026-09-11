#!/bin/bash
# Job entrypoint (inside docker.io/juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311,
# self-managed Python). Run from the delivered worktree root:
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/src/run_job.sh" [run_job.py args...]
#
# Placement: activation pools on container-local disk (/local-scratch, overlay root
# filesystem; 300 GiB write-probed at ~2.9 GB/s, /dev/shm is a fixed 40 GB so RAM is
# not an option); the per-job volume ($SILICO_EXPERIMENT_ARTIFACTS_DIR) receives
# sae.pt, meta.json, logs, results, the held-out token shards and the restart pool.
# Job resources requested at submit: 1x H100 80GB, 16 CPU, 256 GiB RAM.
set -euo pipefail
export PYTHONUNBUFFERED=1
export SAE_IMAGE_REF="${SAE_IMAGE_REF:-docker.io/juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311}"
export SAE_JOB_ID="${SAE_JOB_ID:-${MODAL_TASK_ID:-unknown}}"
EXP="${SILICO_EXPERIMENT_RELATIVE_DIR:?}"
mkdir -p /local-scratch
echo "== run_job.sh $(date -u +%FT%TZ) job=$SAE_JOB_ID image=$SAE_IMAGE_REF args=$*"
df -h /local-scratch /dev/shm "${SILICO_EXPERIMENT_ARTIFACTS_DIR}" || true
exec python3 -u "$EXP/src/run_job.py" "$@"
