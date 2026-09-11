# Calibrate the event-aware SAE trainer against Gemma Scope on Gemma 2 2B, with pools on local scratch

## What this is
Two deliverables from one H100 job. (1) Three JumpReLU SAEs (layers 5, 12, 20; residual stream at
block output; 16,128 features = 7 x 2304; target L0 50) trained by `event-aware-SAE-trainer` at
commit `c796c2eeced4c9310066989c0fa304bdb88307b4` (no `.py` changes; `trainer/` is the checkout copy plus
`trainer/configs/gemma2_2b.yaml`) on `google/gemma-2-2b`, scored on identical held-out tokens against
Gemma Scope (`google/gemma-scope-2b-pt-res`, `width_16k`, release nearest L0 50 per layer).
(2) A pool-placement timing record: the same walk and a 200-step training with activation pools on the
per-job network volume versus container-local disk, plus the full 0 -> 20 chain timings.

Results brief: `results/SUMMARY.md`. Details: `results/calibration.md`, `results/timing.md`.

## Layout
- `trainer/` copy of the trainer checkout (`TRAINER_COMMIT.txt`); only addition `configs/gemma2_2b.yaml`.
- `src/run_job.sh` / `src/run_job.py` job entrypoint and orchestrator (phases, W&B timing record, HF push, restart pool).
- `src/walk_layers.py` chain walk without training, built from the trainer's own producer primitives (with read/forward/write timers); also writes the token pool and the FineWeb-Edu held-out shards.
- `src/verify_rolling_forward.py` single-block walk vs full forward, all 26 layers -> `results/rolling_check.json`.
- `src/make_heldout_c4.py` second held-out source (C4 en validation), trainer shard recipe.
- `src/eval_calibration.py` ours vs Gemma Scope on identical tokens (EV, L0, dead, LM loss recovered, decoder overlap).
- `src/make_reports.py` pod-side: results/*.md|csv, SUMMARY.md, figure bundles.
- `results/` calibration.csv/.md, timing.csv/.md, rolling_check.json, SUMMARY.md, job_record.json.
- `figures/decoder_overlap_hist/`, `figures/produce_time_by_placement/` figure bundles.

## How to reproduce
Submitted as one Silico on-demand job (Modal, 1x H100 80GB, 16 CPU, 256 GiB; image
`docker.io/juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311@sha256:e3275d5ffdc44bcd06732e4b644dfd92efd5b5c8c0c754c03b5bddd0a4400dff`;
`HF_TOKEN`, `WANDB_API_KEY` forwarded):

    bash experiments/experiment-5-0x32n7/src/run_job.sh --mode full

which runs, in order: trainer install, model download, token pool (500 shards) + held-out shards, C4 held-out,
rolling check, A/B (walk 0-3 + 200-step layer-3 training on volume, then on local disk), chain walk 0->5, train 5,
walk 6->12, train 12, walk 13->20, train 20 (each `run_atlas.py --config configs/gemma2_2b.yaml --layer-range L,L`),
meta provenance + `hf upload` per layer, `eval_calibration.py`, restart manifest, W&B tables. Then pod-side:

    uv run --no-sync python src/make_reports.py --job-results <fetched results dir>

## Outputs
(filled in after the job; see the Outputs section below)
