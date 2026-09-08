#!/bin/sh
# Upload the Bella-vs-Gemma direction files from exp_01m1xz587ze4t8np5xr6016475 to the
# directions/ folder of juiceb0xc0de/gemma-4-e4b-SAE, plus a short README, then verify.
# Runs in python:3.11-slim, CPU only, HF_TOKEN forwarded from job settings.
set -eu
REPO=juiceb0xc0de/gemma-4-e4b-SAE
IN="${SILICO_INPUT_ARTIFACTS_DIR:?}/experiments/exp_01m1xz587ze4t8np5xr6016475"
OUT="$SILICO_EXPERIMENT_ARTIFACTS_DIR"
mkdir -p "$OUT"

pip install -q huggingface_hub
python3 -c "import huggingface_hub as h; print('huggingface_hub', h.__version__)"
ls -la "$IN/E4B/run" "$IN/E2B/run"

up() { hf upload "$REPO" "$1" "$2" --repo-type dataset | tee -a "$OUT/upload_directions.txt"; }
up "$IN/E4B/run/directions.npz" directions/e4b_directions.npz
up "$IN/E4B/run/stage1.json"    directions/e4b_stage1.json
up "$IN/E2B/run/directions.npz" directions/e2b_directions.npz
up "$IN/E2B/run/stage1.json"    directions/e2b_stage1.json

cat > /tmp/README.md <<'EOF'
# Bella-vs-Gemma persona directions

Unit Bella-minus-Gemma difference-of-means direction per layer for Gemma-4-E4B-it
(`e4b_*`) and Gemma-4-E2B-it (`e2b_*`). Source experiment: exp_01m1xz587ze4t8np5xr6016475.

- `*_directions.npz`: keys `all` (all reply tokens) and `first` (first 32 reply tokens);
  each array has shape `[n_layers, d_model]`, rows are unit vectors. Layer `l` = output of block `l`.
- `*_stage1.json`: per-layer Cohen's d, bootstrap CI, shuffled null p95, cross-layer cosines,
  and logit-lens top tokens.
EOF
up /tmp/README.md directions/README.md

python3 -c "
from huggingface_hub import HfApi; import json, os, sys
api = HfApi(); info = api.dataset_info('$REPO', files_metadata=True)
want = ['directions/e4b_directions.npz','directions/e4b_stage1.json','directions/e2b_directions.npz','directions/e2b_stage1.json','directions/README.md']
sizes = {s.rfilename: s.size for s in info.siblings if s.rfilename.startswith('directions/')}
ok = all(w in sizes for w in want)
json.dump({'repo': '$REPO', 'private': info.private, 'sha': info.sha, 'directions_files': sizes, 'ok': ok}, open(os.path.join('$OUT','directions_listing.json'),'w'), indent=1)
print(json.dumps(sizes, indent=1)); print('OK' if ok else 'FAIL'); sys.exit(0 if ok else 1)"
