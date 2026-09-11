#!/bin/sh
# Push trained Gemma-4-E4B SAE layers from staged store folders to the private HF
# dataset juiceb0xc0de/gemma-4-e4b-SAE.
# Usage: sh src/push_to_hub.sh <range_00_13|range_14_27|range_28_41|layer <range> <NN>|verify|private>
#   range_*  upload that folder's 14 layer dirs (layer_NN_s0/{sae.pt,meta.json})
#   layer <range> <NN>  upload a single layer dir (used after 22 GB range jobs died during input staging)
#   verify   upload results/atlas_summary.csv, then check the Hub listing and write hub_listing.json
#   private  set the repo private (update_repo_settings), then re-check the listing and rewrite hub_listing.json
# One range per job: the on-demand fabric caps declared input_artifacts at 32 GiB per job
# and each range folder is ~22 GB.
# Runs on python:3.11-slim (self-managed, CPU only). HF_TOKEN comes from job settings.
set -eu

MODE="${1:?mode required}"
REPO=juiceb0xc0de/gemma-4-e4b-SAE
IN="${SILICO_INPUT_ARTIFACTS_DIR:-/nonexistent}/experiments/exp_01m1y0m9rsfnasj5vm2egaje0v"
OUT="$SILICO_EXPERIMENT_ARTIFACTS_DIR"
mkdir -p "$OUT"

pip install -q "huggingface_hub[cli]"
python3 -c "import huggingface_hub as h; print('huggingface_hub', h.__version__)" | tee "$OUT/hf_version_$MODE.txt"

case "$MODE" in
  layer)
    # sh src/push_to_hub.sh layer <range_XX_YY> <NN>: upload one layer dir (~1.7 GB) as layer_NN_s0/
    RANGE="${2:?range required}"; NN="${3:?layer NN required}"
    SRC="$IN/$RANGE/saes/google_gemma-4-e4b-it/layer_${NN}_s0"
    ls -la "$SRC"
    echo "== uploading layer_${NN}_s0 =="; date -u
    hf upload "$REPO" "$SRC" "layer_${NN}_s0" --repo-type dataset --private | tee "$OUT/upload_layer_${NN}.txt"
    date -u
    ;;
  range_*)
    SRC="$IN/$MODE/saes/google_gemma-4-e4b-it"
    echo "== staged layers in $MODE =="
    ls "$SRC" | tee "$OUT/staged_layers_$MODE.txt"
    du -sh "$SRC"
    echo "== uploading $MODE =="; date -u
    hf upload "$REPO" "$SRC" . --repo-type dataset --private --commit-message "E4B SAE atlas $MODE" | tee "$OUT/upload_$MODE.txt"
    date -u
    ;;
  private)
    echo "== setting repo private =="
    python3 -c "
from huggingface_hub import HfApi
api = HfApi(); api.update_repo_settings('$REPO', repo_type='dataset', private=True)
print('private now:', api.dataset_info('$REPO').private)"
    ;;
  verify) : ;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
case "$MODE" in
  verify|private)
    if [ "$MODE" = verify ]; then
    echo "== uploading atlas_summary.csv =="
    hf upload "$REPO" "$IN/results/atlas_summary.csv" atlas_summary.csv --repo-type dataset --private | tee "$OUT/upload_summary.txt"
    fi
    echo "== verify =="
    python3 -c "
from huggingface_hub import HfApi; import json, re, sys, os
api = HfApi()
files = api.list_repo_files('$REPO', repo_type='dataset')
layers = sorted({m.group(1) for f in files for m in [re.match(r'(layer_\d\d_s0)/(sae\.pt|meta\.json)$', f)] if m})
ok = len(layers) == 42 and all(f'{l}/sae.pt' in files and f'{l}/meta.json' in files for l in layers) and 'atlas_summary.csv' in files
info = api.dataset_info('$REPO', files_metadata=True)
sizes = {s.rfilename: s.size for s in info.siblings}
json.dump({'repo': '$REPO', 'repo_type': 'dataset', 'private': info.private, 'sha': info.sha, 'n_layers': len(layers), 'layers': layers, 'ok': ok, 'files': sorted(files), 'sizes': sizes, 'total_bytes': sum(v or 0 for v in sizes.values())}, open(os.path.join('$OUT','hub_listing.json'),'w'), indent=1)
print('OK' if ok else 'FAIL', len(layers)); sys.exit(0 if ok else 1)"
    ;;
  *) : ;;
esac
