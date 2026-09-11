# Gemma 2 2B SAE atlas, layers 0 to 25, one sequential job, in-RAM pools

## What this is
One H100 job running `event-aware-SAE-trainer` (commit `c796c2eeced4c9310066989c0fa304bdb88307b4`) through its own
orchestrator (`run_atlas.py` -> `run_atlas_rolling`) over `google/gemma-2-2b`, layers 0-25 in order. For each layer L:
produce pool L from pool L-1 in process memory (`rolling-hf` single-block walk over 500 shards of [32, 1024] Pile tokens),
release pool L-1, train the JumpReLU SAE for L on pool L with BOS/EOS/pad positions excluded from sampling and from the
`b_dec` estimate, save `layer_LL_s0/{sae.pt, meta.json, checkpoint_full.pt}`, push the folder to
`juiceb0xc0de/gemma-2-2b-SAE`, verify the Hub listing, advance. Next-layer pre-production is off; no activation pool is
ever written to disk; one W&B run for the whole job.

## Layout
- `trainer/` copy of the trainer checkout (`TRAINER_COMMIT.txt`) with Task 6's three env-gated changes
  (`SAE_SEQ_LEN`, `SAE_PRODUCE_ONLY`, `SAE_POOL_BACKEND=ram`) plus this task's four:
  `SAE_EXCLUDE_SPECIAL` (reader + `b_dec` masking, `accum_steps == 1` guard), `SAE_NO_PREPRODUCE` /
  `SAE_EAGER_POOL_CLEANUP`, `SAE_NO_RESUME_POOL`, `SAE_WANDB_SINGLE_RUN` (+ `SAE_WANDB_RUN_NAME`).
  `results/trainer.patch` is the exact diff vs `c796c2ee`; `results/delta_vs_task6.patch` is the diff vs Task 6's copy;
  `results/task6_trainer.patch` is Task 6's patch for reference. Defaults are unchanged when the variables are unset.
- `configs/gemma2_2b.yaml` run config (Pile corpus, expansion 7, 500 pool batches, target L0 50, max 5,000 steps,
  microbatch 32,768, `hf_sae_repo: juiceb0xc0de/gemma-2-2b-SAE`, `wandb_project: gemma-2-2b-SAE-atlas`,
  `env: SAE_SEQ_LEN: "1024"`).
- `src/run_job.sh` job entrypoint (image, PYTHONPATH fix, zstandard install, log tee); `src/run_job.py` drives
  `run_atlas.main()` in-process with the exact CLI, owns the single W&B run, wraps `_produce_pool_hf_rolling`,
  `train_sae_on_activations`, `HfApi.upload_folder`, `_rm_pool`, `_capture_token_pool` and `_save_full_checkpoint`
  (timing, special-token count, provenance block in each `meta.json`, per-layer Hub verification with retries), samples
  memory every 15 s, and writes `atlas_summary.csv`. `src/hub_progress.py` is the pod-side progress poll.
- `tests/test_atlas_patch.py` CPU tests for the four additions (run from `trainer/`: `apy -m pytest -q ../tests`).
- `results/` trainer.patch, delta_vs_task6.patch, task6_trainer.patch, test_results.txt, atlas_summary.csv,
  special_tokens.csv, job_record.json, mem_series.csv, hub_listing_{before,after}.json, hub_progress.json, job.log.

## How to reproduce
    bash experiments/experiment-7-39gmbf/src/run_job.sh --layer-range 0,25 --tag atlas_L00-25_ram
    # 1x H100 80GB, 8 CPU, 200 GiB RAM, HF_TOKEN + WANDB_API_KEY, image
    # docker.io/juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311
Equivalent trainer command (driven in-process by `run_job.py`):
    cd trainer && SAE_POOL_BACKEND=ram SAE_SEQ_LEN=1024 SAE_NO_PREPRODUCE=1 SAE_EAGER_POOL_CLEANUP=1 \
      SAE_NO_RESUME_POOL=1 SAE_EXCLUDE_SPECIAL=1 SAE_WANDB_SINGLE_RUN=1 \
      SAE_WANDB_RUN_NAME=gemma-2-2b_L00-25_pile1024_ram \
      python run_atlas.py --config ../configs/gemma2_2b.yaml --layer-range 0,25 --capture rolling-hf --pool-retention 1
Trainer tests (CPU): `cd trainer && apy -m pytest -q` -> `results/test_results.txt` (195 passed, 3 skipped) plus
`apy -m pytest -q ../tests/test_atlas_patch.py` (5 passed).

Smoke (interactive session 954019587525, 2026-09-11 19:44 UTC): layers 0-1, 20 shards, 300 steps, no push, W&B offline.
Masked fraction 0.00198, `b_dec` from 1,635,016 tokens (= 50 x 32,768 x (1 - 0.00198)), eager cleanup and no
pre-production confirmed in the log, no pool file on disk, 49 ms/step.

## Outputs
(filled in at completion)
