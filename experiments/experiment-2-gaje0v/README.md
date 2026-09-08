# Train the 42-layer JumpReLU SAE atlas for Gemma 4 E4B with the E2B recipe

## What this is
A per-layer JumpReLU SAE atlas for `google/gemma-4-E4B-it` (42 decoder layers,
residual stream d_in 2560), trained with the E2B atlas recipe: 32x expansion
(81,920 features), target L0 50, FineWeb-Edu, 500 pool shards x 32,768 tokens
(16.4 M tokens per layer pool), rolling-float pool production, early stop on EV
plateau. Trainer: `event-aware-SAE-trainer` at commit
`c796c2eeced4c9310066989c0fa304bdb88307b4` (copied into `trainer/`), run inside the
researcher's public image
`docker.io/juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311`
(`sha256:e3275d5ffdc44bcd06732e4b644dfd92efd5b5c8c0c754c03b5bddd0a4400dff`) on one H100
per job.

Published atlas: private HF dataset **`juiceb0xc0de/gemma-4-e4b-SAE`**, 42 folders
`layer_NN_s0/{sae.pt, meta.json}` plus `atlas_summary.csv`, 70.5 GB, verified through
the Hub API (42 layers, 87 files).

## Results
- 42/42 layers trained; none hit the 5000-step cap.
- Explained variance (EV): min 0.770, median 0.928, max 0.990; 30 layers EV >= 0.90,
  38 layers EV >= 0.85.
- Flagged (EV below the 0.85 floor, published and marked, not excluded):
  layer_17_s0 (0.834), layer_18_s0 (0.784), layer_19_s0 (0.770), layer_23_s0 (0.846).
- Mean L0: min 43.08, median 49.58, max 54.88; 41/42 layers within +-5 of the target
  50 (layer 9 is the exception).
- Dead features: max 0.88% of 81,920.
- Stop step: min 1001, median 1501, max 2251. 30 layers stopped on EV/L0
  convergence; 12 stopped by the trainer's dead-feature ceiling with rollback to the
  last good checkpoint (layers 8, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27).
- Compute: 686.7 GPU-minutes = 11.45 GPU-hours across the three chain jobs; about
  8.6 GPU-hours of SAE optimization after subtracting the forward-only pool walks
  that the 14-27 and 28-41 jobs run to reach layers 14 and 28.
- Forward check: the trainer's rolling per-layer forward is bit-exact against a full
  model forward on layers 0-40 (`results/forward_check_e4b_summary.json`).
- Per-layer table: `results/chain_results_table.csv` (EV, mean L0, dead %, recon loss,
  stop step and reason, wall minutes, W&B run URL). W&B project:
  https://wandb.ai/ricks-holmberg-juiceb0xc0de/gemma-4-e4b-SAE (runs `L{NN}_s0`).

## Deviations from the E2B recipe
1. `microbatch_tokens: 16384` instead of 32768 (`trainer/configs/gemma4_e4b.yaml`).
   `batch_tokens` stays at 32768, so each optimizer step accumulates two microbatches
   (2x gradient accumulation); the token budget, L0 target and schedule are unchanged.
   Reason: with d_in 2560 and 81,920 features, the JumpReLU activations of a
   32768-token microbatch exceed one 80 GB H100 (smoke job OOMed in `loss.backward()`
   at 74.5 GiB allocated).
2. Two edits to `trainer/sae_trainer_rolling.py` relative to commit `c796c2ee`; the
   full diff is `trainer/PATCH_vs_c796c2ee.diff` (34 lines, two hunks):
   - `meta.json` provenance: after the meta dict is built, the fields `trainer_commit`,
     `image_ref`, `job_id` are copied from the environment variables
     `SAE_TRAINER_COMMIT`, `SAE_IMAGE_REF`, `SAE_JOB_ID` when set. Reason: the plan
     requires every layer's meta to record trainer commit, container image, and job.
     No training logic is touched.
   - Skip-train branch (mid-chain start, e.g. `--layer-range 28,41`): after the pool for
     a below-start layer L is produced, delete the walk-up pool `L - pool_retention - 1`
     unless it is a KV anchor pool. Reason: the walk-up otherwise leaves one full
     16.4 M-token pool per skipped layer on disk (28 pools for the 28-41 job), which
     exceeds the per-job volume; pools below `L - pool_retention` are never read again
     because pool[L+1] is produced from pool[L]. KV anchor pools are exempt because the
     shared-KV layers (24 onward) read them.
   Everything else in `trainer/` is byte-identical to the checkout at `c796c2ee`.

## Layout
- `trainer/` copy of the trainer checkout (`TRAINER_COMMIT.txt`, `PATCH_vs_c796c2ee.diff`).
- `trainer/configs/gemma4_e4b.yaml` the E4B config.
- `src/run_job.sh` job entrypoint (`smoke | chain | join`), run from the delivered
  worktree root as `bash "$SILICO_EXPERIMENT_RELATIVE_DIR/src/run_job.sh" <mode> [run_atlas args]`.
  Sets `SAE_DATA_DIR`, `SAE_SCRATCH_DIR`, `HF_HOME`, installs the trainer copy, records
  `job_info_*.json`, and on exit deletes `checkpoint_full.pt` for completed layers.
- `src/join_atlas.py`, `src/make_summary.py` atlas assembly, `atlas_summary.csv`, and the
  EV / mean-L0 figure (the assembly job route was abandoned, see below; the summary and
  figure code still apply to the metas).
- `verify_atlas.py` the plan's verification command (local + Hub checks, encode spot check).
- `src/verify_rolling_forward.py` bit-exact forward check.
- `results/` per-layer table, W&B URLs, forward-check summary.
- `figures/checkpoint-renders/` renders used on the timeline cards.

## How to reproduce
Steps 1-2 ran as Silico on-demand jobs (Modal, 1 H100, the public image above,
`HF_TOKEN` and `WANDB_API_KEY` forwarded, `SAE_IMAGE_REF=<image@digest>`,
`SAE_DATA_SUBDIR=range_XX_YY`).

1. Smoke: `bash "$SILICO_EXPERIMENT_RELATIVE_DIR/src/run_job.sh" smoke --layer-range 0,0 --pool-batches 20 --max-steps 60 --no-push`,
   then the same for `--layer-range 24,24` (exercises the shared-KV path). Layer-0
   `activation_norm_probe` = 1.0194611549377441; it is pinned on every chain job with
   `--norm-ref` so the three ranges share one normalisation reference.
2. Chain (three parallel jobs):
   `bash "$SILICO_EXPERIMENT_RELATIVE_DIR/src/run_job.sh" chain --layer-range 0,13 --no-push --norm-ref 1.0194611549377441`
   and the same for `14,27` and `28,41`. Outputs land in the experiment store under
   `range_00_13/`, `range_14_27/`, `range_28_41/` (`saes/google_gemma-4-e4b-it/layer_NN_s0/`).
3. Publishing (what actually put the atlas on the Hub). The planned single "join" job
   could not run on this fabric: declared job inputs are capped at 32 GiB, and the
   reused-volume collect path failed four times (jobs declaring about 22 GB of
   `input_artifacts` were killed during input staging with an empty log). The working
   route was one CPU-only job per layer, each declaring only that layer's 1.7 GB folder
   in `input_artifacts` and running
   `hf upload juiceb0xc0de/gemma-4-e4b-SAE <layer dir> layer_NN_s0 --repo-type dataset --private`,
   up to about 12 jobs at a time, with a Hub-tree sweep after each wave that re-queued
   layers lost to concurrent-commit collisions. `atlas_summary.csv` was uploaded last.
   The repo therefore has 42 layer commits rather than one push.
4. Summary and figure (pod-side, needs `silico-figures`):
   `uv run --no-sync python src/make_summary.py --metas <atlas_metas.json> --out results/atlas_summary.csv --wall <wall_minutes.json>`
   where `atlas_metas.json` is `{"layer_NN": <meta.json>}` for the 42 layers.
5. Verification:
   `python3 verify_atlas.py --root <atlas dir> --hf-repo juiceb0xc0de/gemma-4-e4b-SAE --n-layers 42 --d-in 2560 --n-features 81920 --k 50 --ev-floor 0.85 --l0-window 40,60 --dead-max 0.01`.
   It exits non-zero because of the four flagged EV layers and lists them by name.

## Running externally (no Silico)
```bash
# 1. container with the pinned stack (torch 2.9.1, cu128, py3.11, Triton flash attention)
docker run --gpus all -it --shm-size 16g \
  -v $PWD:/work -v /data/sae:/data/sae \
  -e HF_TOKEN -e WANDB_API_KEY \
  docker.io/juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311 bash

# 2. trainer: either check out this branch's trainer/ copy, or clone upstream and apply the patch
git clone https://github.com/<owner>/event-aware-SAE-trainer && cd event-aware-SAE-trainer
git checkout c796c2eeced4c9310066989c0fa304bdb88307b4
patch -p1 < /work/experiments/experiment-2-gaje0v/trainer/PATCH_vs_c796c2ee.diff
cp /work/experiments/experiment-2-gaje0v/trainer/configs/gemma4_e4b.yaml configs/
pip install -e . --no-deps

# 3. locations (weights + meta are durable; scratch holds token and activation pools, ~640 GB for a 14-layer chain)
export SAE_DATA_DIR=/data/sae/out SAE_SCRATCH_DIR=/data/sae/scratch HF_HOME=/data/sae/hf_home
export SAE_TRAINER_COMMIT=c796c2eeced4c9310066989c0fa304bdb88307b4
export SAE_IMAGE_REF=docker.io/juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311
export SAE_JOB_ID=local-$(date +%s)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. smoke (layers 0 and 24, 20 pool shards, 60 steps)
python -u run_atlas.py --config configs/gemma4_e4b.yaml --layer-range 0,0 --pool-batches 20 --max-steps 60 --no-push
python -u run_atlas.py --config configs/gemma4_e4b.yaml --layer-range 24,24 --pool-batches 20 --max-steps 60 --no-push
# read final_metrics.activation_norm_probe from $SAE_DATA_DIR/saes/google_gemma-4-e4b-it/layer_00_s0/meta.json

# 5. chain: three GPUs in parallel ...
python -u run_atlas.py --config configs/gemma4_e4b.yaml --layer-range 0,13  --no-push --norm-ref <probe>
python -u run_atlas.py --config configs/gemma4_e4b.yaml --layer-range 14,27 --no-push --norm-ref <probe>
python -u run_atlas.py --config configs/gemma4_e4b.yaml --layer-range 28,41 --no-push --norm-ref <probe>
# ... or one sequential run on a single GPU (no forward-only walk-ups, ~8.6 GPU-h)
python -u run_atlas.py --config configs/gemma4_e4b.yaml --layer-range 0,41 --no-push --norm-ref <probe>

# 6. summary CSV (+ figure if silico-figures is installed; add --no-figure otherwise)
python /work/experiments/experiment-2-gaje0v/src/make_summary.py \
  --metas $SAE_DATA_DIR/saes/google_gemma-4-e4b-it --out atlas_summary.csv --no-figure

# 7. verify (exits non-zero and names the flagged layers if any EV < 0.85)
python /work/experiments/experiment-2-gaje0v/verify_atlas.py --root $SAE_DATA_DIR/saes/google_gemma-4-e4b-it \
  --hf-repo juiceb0xc0de/gemma-4-e4b-SAE --n-layers 42 --d-in 2560 --n-features 81920 --k 50 \
  --ev-floor 0.85 --l0-window 40,60 --dead-max 0.01 --skip-hf

# 8. upload (one commit per layer; rerun any layer that collides with a concurrent commit)
for L in $(seq -w 0 41); do
  hf upload juiceb0xc0de/gemma-4-e4b-SAE $SAE_DATA_DIR/saes/google_gemma-4-e4b-it/layer_${L}_s0 layer_${L}_s0 --repo-type dataset --private
done
hf upload juiceb0xc0de/gemma-4-e4b-SAE atlas_summary.csv atlas_summary.csv --repo-type dataset --private
```

## Outputs
- HF dataset (private): `juiceb0xc0de/gemma-4-e4b-SAE`, 42 x `layer_NN_s0/{sae.pt (1.68 GB), meta.json}`
  + `atlas_summary.csv`, 70.5 GB total; verified via the Hub API (42 layers, 87 files).
- Per-range weights in the experiment store (source of the upload):
  `artifact://juiceb0xc0de-15787e/experiments/exp_01m1y0m9rsfnasj5vm2egaje0v/range_00_13/`,
  `.../range_14_27/`, `.../range_28_41/` (each `saes/google_gemma-4-e4b-it/layer_NN_s0/`, 29 objects, 21.9 GB).
- Training jobs (2026-09-07 UTC): 826751606056 (layers 0-13, 2 h 13 m), 117948351885 (14-27,
  3 h 58 m), 435463686100 (28-41, 5 h 26 m); logs under `.../logs/chain___layer_range_*.log`.
- Model `google/gemma-4-E4B-it`; data FineWeb-Edu via the trainer's loader; config
  `trainer/configs/gemma4_e4b.yaml`; trainer commit `c796c2eeced4c9310066989c0fa304bdb88307b4`;
  image digest `sha256:e3275d5ffdc44bcd06732e4b644dfd92efd5b5c8c0c754c03b5bddd0a4400dff`.
- W&B: https://wandb.ai/ricks-holmberg-juiceb0xc0de/gemma-4-e4b-SAE (per-layer URLs in `results/wandb_runs.json`).
