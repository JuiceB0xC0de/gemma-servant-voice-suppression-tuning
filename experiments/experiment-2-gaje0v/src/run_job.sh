#!/bin/bash
# Job entrypoint for the Gemma 4 E4B SAE atlas (runs inside the researcher's public
# trainer image; self-managed Python environment, no Silico overlay).
#
# Usage (from the delivered worktree root):
#   bash "$SILICO_EXPERIMENT_RELATIVE_DIR/src/run_job.sh" <mode> [run_atlas args...]
#     mode = smoke | chain | verify
#
# Environment (set by the submit call):
#   SAE_IMAGE_REF   image reference recorded into every meta.json
#   RESTORE_FROM_REUSED_VOLUME=1  copy the previous job's scratch (resume pool, KV
#                   anchors, sidecars, token pool) from $SILICO_REUSED_VOLUME_DIR
#   HF_TOKEN / WANDB_API_KEY forwarded as job settings.
set -euo pipefail

MODE="${1:?mode required: smoke|chain|verify}"; shift || true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXP="${SILICO_EXPERIMENT_RELATIVE_DIR:?SILICO_EXPERIMENT_RELATIVE_DIR unset}"
ROOT="$(pwd)"
TRAINER="$ROOT/$EXP/trainer"
ART="${SILICO_EXPERIMENT_ARTIFACTS_DIR:?SILICO_EXPERIMENT_ARTIFACTS_DIR unset}"
mkdir -p "$ART"

# Durable (uploaded) outputs: SAE weights + meta. Scratch stays on the per-job volume.
export SAE_DATA_DIR="$ART/${SAE_DATA_SUBDIR:-sae_data}"
export SAE_SCRATCH_DIR="$ROOT/$EXP/scratch/rollcache"
export HF_HOME="$ROOT/$EXP/scratch/hf_home"
export TMPDIR="$ROOT/$EXP/scratch/tmp"
mkdir -p "$SAE_DATA_DIR" "$SAE_SCRATCH_DIR" "$HF_HOME" "$TMPDIR"
export SAE_TRAINER_DIR="$TRAINER"
export PYTHONPATH="$TRAINER${PYTHONPATH:+:$PYTHONPATH}"
export SAE_TRAINER_COMMIT="$(sed -n 's/^commit //p' "$TRAINER/TRAINER_COMMIT.txt")"
export SAE_JOB_ID="${SAE_JOB_ID:-${MODAL_TASK_ID:-unknown}}"
export SAE_IMAGE_REF="${SAE_IMAGE_REF:-unknown}"
JOBLOG="$ART/job_info_${MODE}_$(date -u +%Y%m%dT%H%M%SZ).json"

echo "== run_job.sh mode=$MODE  $(date -u +%FT%TZ)"
echo "   trainer=$TRAINER commit=$SAE_TRAINER_COMMIT image=$SAE_IMAGE_REF job=$SAE_JOB_ID"
echo "   SAE_DATA_DIR=$SAE_DATA_DIR  SAE_SCRATCH_DIR=$SAE_SCRATCH_DIR"
python3 --version; python3 -m pip --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

# ---- trainer at the current commit (image carries a pinned snapshot; override it) ----
cd "$TRAINER"
python3 -m pip install -q -e . --no-deps
cd "$ROOT"
python3 - <<'PY'
import torch, transformers, importlib
import sae_trainer_rolling, run_atlas, gemma_attention
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__)
print("trainer from", sae_trainer_rolling.__file__)
try:
    importlib.import_module("gemma_triton_flash_attn"); print("gemma_triton_flash_attn: present")
except ModuleNotFoundError:
    print("gemma_triton_flash_attn: ABSENT (Transformers default attention)")
PY

# ---- optional restore of scratch from a previous job's retained volume -------------
if [[ "${RESTORE_FROM_REUSED_VOLUME:-0}" == "1" ]]; then
  SRC="${SILICO_REUSED_VOLUME_DIR:?RESTORE requested but SILICO_REUSED_VOLUME_DIR unset}"
  PREV_SCRATCH="$SRC/$EXP/scratch/rollcache"
  echo "== restoring scratch from $PREV_SCRATCH"
  ls "$PREV_SCRATCH" || true
  shopt -s nullglob
  for d in "$PREV_SCRATCH"/tokens_* "$PREV_SCRATCH"/resume_pool_* "$PREV_SCRATCH"/kv_anchor_* "$PREV_SCRATCH"/kv_shared_*; do
    echo "   cp -r $(basename "$d")"; cp -r "$d" "$SAE_SCRATCH_DIR/"
  done
  shopt -u nullglob
  # Completed layers (sae.pt + meta.json) from the previous job's uploaded outputs are
  # expected to be staged by input_artifacts / a prior flow-back into $SAE_DATA_DIR.
fi

# ---- scratch hygiene on exit: keep only sae.pt + meta.json for completed layers ------
cleanup() {
  echo "== cleanup $(date -u +%FT%TZ)"
  shopt -s nullglob
  for d in "$SAE_DATA_DIR"/saes/*/layer_*_s*; do
    [[ -d "$d" ]] || continue
    case "$d" in *_latest) continue;; esac
    if [[ -f "$d/sae.pt" && -f "$d/meta.json" ]]; then
      rm -f "$d/checkpoint_full.pt" && echo "   dropped $d/checkpoint_full.pt (layer complete)"
      rm -rf "${d}_latest" 2>/dev/null || true
    else
      echo "   keeping $d (incomplete; checkpoint_full.pt is the resume point)"
    fi
  done
  shopt -u nullglob
  du -sh "$SAE_DATA_DIR" 2>/dev/null || true
  du -sh "$SAE_SCRATCH_DIR" 2>/dev/null || true
}
trap cleanup EXIT

START=$(date +%s)
RC=0
case "$MODE" in
  verify)
    python3 -u "$ROOT/$EXP/src/verify_rolling_forward.py" --out "$ART/forward_check_e4b.json" "$@" || RC=$?
    ;;
  smoke|chain)
    mkdir -p "$ART/logs"
    LOG="$ART/logs/${MODE}_$(echo "$*" | tr -c 'A-Za-z0-9,.' '_' | cut -c1-60)_$(date -u +%Y%m%dT%H%M%SZ).log"
    echo "   log copy: $LOG"
    set +e
    python3 -u "$TRAINER/run_atlas.py" --config "$TRAINER/configs/gemma4_e4b.yaml" "$@" 2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}
    set -e
    ;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
END=$(date +%s)
python3 - "$JOBLOG" "$MODE" "$RC" "$START" "$END" "$@" <<'PY'
import json, os, sys
p, mode, rc, s, e, *args = sys.argv[1:]
json.dump({"mode": mode, "rc": int(rc), "wall_s": int(e) - int(s), "args": args,
           "job_id": os.environ.get("SAE_JOB_ID"), "image_ref": os.environ.get("SAE_IMAGE_REF"),
           "trainer_commit": os.environ.get("SAE_TRAINER_COMMIT")}, open(p, "w"), indent=1)
print("wrote", p)
PY
echo "== done rc=$RC wall=$((END-START))s"
exit $RC
