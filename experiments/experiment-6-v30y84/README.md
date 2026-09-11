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
Job `615894493371` (Modal task `ta-01M28XXSMF2WWWRWX2E50SYN1S`), 2026-09-11 19:08-19:14 UTC, 1x H100 80GB, requested 8 CPU / 200 GiB;
the container reported 24 CPUs and MemTotal 1618 GB (limit not visible in-container; peak RSS 158.6 GB fit the 200 GiB request).
Image `docker.io/juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311@sha256:e3275d5ffdc44bcd06732e4b644dfd92efd5b5c8c0c754c03b5bddd0a4400dff`
(torch 2.9.1+cu128, transformers 5.5.4). Corpus `monology/pile-uncopyrighted` @ `3be90335b66f24456a5d6659d9c8d208c0357119`
(dataset_info sha resolved at job start). Trainer patch sha256 `b976f3e07f6a092a908a9b88b8b15907bf3120051fb21fcc0df1b4c5fa9e5c73`.

| layer | wall s | tokens/s | resident pool GB after produce / after cleanup | peak RSS GB |
|---|---|---|---|---|
| 0 | 35.0 | 467,475 | 75.5 / - | 83.1 |
| 1 | 44.3 | 369,955 | 151.0 / 75.5 | 158.6 |
| 2 | 43.2 | 379,127 | 151.0 / 75.5 | 158.6 |
| 3 | 43.6 | 375,714 | 151.0 / 75.5 | 158.6 |

Model fetch+load 38 s; 500 token shards [32,1024] in 98 s; trainer wall 308 s; job wall 316 s. Special tokens over 16,384,000 ids:
BOS 34,485 (16,000 at position 0, 18,485 mid-window), EOS 0, pad 0. Hub `juiceb0xc0de/gemma-2-2b-SAE` unchanged
(sha 9a20f466, 5 files, before and after). W&B: one run `produce_only_L00-03_ram` in `gemma-2-2b-SAE-timing`
(https://wandb.ai/ricks-holmberg-juiceb0xc0de/gemma-2-2b-SAE-timing/runs/ddikj8m9); no run added to `gemma-2-2b-SAE`.
Numbers describe performance under this configuration only (Pile, SEQ_LEN 1024, RAM backend); no RAM-vs-disk claim.

- `results/pool_timing.csv` / `.md`, `results/special_tokens.csv`, `results/job_record.json`, `results/mem_series.csv`,
  `results/hub_listing_after.json`, `results/job.log` (copies of the job's uploads).
- Durable job outputs: `artifact://juiceb0xc0de-15787e/experiments/exp_01m28x56geetftadf0d8v30y84/produce_only_L00-03_ram/`
  (`results/`: pool_timing, special_tokens, job_record, mem_series.{csv,json}, hub_listing_{before,after}; `logs/job.log`).
