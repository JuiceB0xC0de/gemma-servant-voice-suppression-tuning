# Pool production only: Gemma 2 2B layers 0 to 3, in-process RAM pool, timed

## What this is
One H100 job running `event-aware-SAE-trainer` (commit `c796c2eeced4c9310066989c0fa304bdb88307b4`) through its own
orchestrator (`run_atlas.py` -> `run_atlas_rolling`) in a new produce-only mode: activation pools for layers 0-3 of
`google/gemma-2-2b` (residual stream, block output) are produced by the `rolling-hf` single-block walk over 500 shards of
[32, 1024] tokens from `monology/pile-uncopyrighted`, held in process memory (no pool files), released under
`pool_retention 1`, and timed per layer. No SAE is trained, nothing is pushed.

## Layout
- `trainer/` copy of the trainer checkout (`TRAINER_COMMIT.txt`) with the three approved, env-gated changes applied
  (`SAE_SEQ_LEN`, `SAE_PRODUCE_ONLY`, `SAE_POOL_BACKEND=ram`); `results/trainer.patch` is the exact diff vs `c796c2ee`.
- `configs/gemma2_2b.yaml` run config (corpus, `env: SAE_SEQ_LEN: "1024"`, pool_batches 500, retention 1).
- `src/run_job.sh` job entrypoint (image, PYTHONPATH fix, zstandard install, log tee); `src/run_job.py` drives
  `run_atlas.main()` in-process with the exact CLI, owns the single W&B run, wraps `_capture_token_pool` (special-token
  counts), `_produce_pool_hf_rolling` (wall/tokens/s/resident bytes/RSS) and `_rm_pool`, samples MemAvailable every 15 s,
  and records the Hub listing before/after.
- `results/` trainer.patch, test_results.txt, pool_timing.csv/.md, special_tokens.csv, job_record.json, mem_series.csv,
  hub_listing_{before,after}.json, job.log (copied from the job's artifact store upload).

## How to reproduce
    bash experiments/experiment-6-v30y84/src/run_job.sh          # 1x H100 80GB, 200 GiB RAM, HF_TOKEN + WANDB_API_KEY
Equivalent trainer command (driven in-process):
    SAE_PRODUCE_ONLY=1 SAE_POOL_BACKEND=ram WANDB_MODE=online python run_atlas.py --config configs/gemma2_2b.yaml \
        --layer-range 0,3 --capture rolling-hf --pool-retention 1 --no-push
Trainer tests (CPU): `cd trainer && apy -m pytest -q` -> `results/test_results.txt`.

## Outputs
(filled in after the job)
