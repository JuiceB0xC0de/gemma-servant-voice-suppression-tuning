#!/bin/bash
# Job entrypoint (inside docker.io/juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311,
# self-managed Python). Run from the delivered worktree root:
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/src/run_job.sh" [run_job.py args...]
#
# Resources requested at submit (production): 1x H100 80GB, 200 GiB RAM (204800 MiB),
# HF_TOKEN + WANDB_API_KEY forwarded. Activation pools live in process memory
# (SAE_POOL_BACKEND=ram); only token shards (131 MB) touch the job-local disk, and only
# results/logs go to $SILICO_EXPERIMENT_ARTIFACTS_DIR.
#
# Equivalent trainer command that run_job.py drives in-process (run_atlas.main()):
#   SAE_PRODUCE_ONLY=1 SAE_POOL_BACKEND=ram WANDB_MODE=online \
#   python run_atlas.py --config configs/gemma2_2b.yaml --layer-range 0,3 \
#       --capture rolling-hf --pool-retention 1 --no-push
# (the config env: block carries SAE_SEQ_LEN=1024; corpus monology/pile-uncopyrighted)
set -euo pipefail
export PYTHONUNBUFFERED=1
export SAE_IMAGE_REF="${SAE_IMAGE_REF:-docker.io/juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311}"
export SAE_JOB_ID="${SAE_JOB_ID:-${MODAL_TASK_ID:-unknown}}"
EXP="${SILICO_EXPERIMENT_RELATIVE_DIR:?}"
# The image ships PYTHONPATH=/opt/src/event-aware-SAE-trainer:/opt/src/gemma-triton-flash-attn:/pkg/:/root/.
# gemma-triton-flash-attn exposes a metadata-less `flash_attn` shim that makes transformers
# raise KeyError('flash_attn') when importing any modeling module unless its Gemma-4-only
# compatibility patch runs first. Gemma 2 uses sdpa, so drop that entry and the image's own
# trainer copy; the pinned+patched checkout under $EXP/trainer is prepended by run_job.py.
export PYTHONPATH="$(printf '%s' "${PYTHONPATH:-}" | tr ':' '\n' | grep -v -e '^/opt/src/gemma-triton-flash-attn$' -e '^/opt/src/event-aware-SAE-trainer$' | paste -sd: -)"
# zstandard is needed by `datasets` to stream the Pile's .jsonl.zst shards; installed into a
# job-local target only if the image lacks it (no other package is touched).
PKG_TARGET="$PWD/$EXP/scratch/job_pkgs"
if ! python3 -c "import zstandard" 2>/dev/null; then
  mkdir -p "$PKG_TARGET"
  python3 -m pip install -q --no-deps --target "$PKG_TARGET" "zstandard==0.23.0"
  export PYTHONPATH="$PKG_TARGET:$PYTHONPATH"
fi
echo "== PYTHONPATH=$PYTHONPATH"
echo "== run_job.sh $(date -u +%FT%TZ) job=$SAE_JOB_ID image=$SAE_IMAGE_REF args=$*"
nproc; free -g || true; cat /sys/fs/cgroup/memory.max 2>/dev/null || true
df -h /dev/shm "${SILICO_EXPERIMENT_ARTIFACTS_DIR}" || true
LOGDIR="${SILICO_EXPERIMENT_ARTIFACTS_DIR}/${SAE_LOG_TAG:-produce_only_L00-03_ram}/logs"
mkdir -p "$LOGDIR"
python3 -u "$EXP/src/run_job.py" "$@" 2>&1 | tee "$LOGDIR/job.log"
