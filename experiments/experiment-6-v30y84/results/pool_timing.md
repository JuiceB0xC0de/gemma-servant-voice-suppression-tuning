# Pool production timing: google/gemma-2-2b, layers 0-3, 500 shards x [32,1024], corpus monology/pile-uncopyrighted, in-process RAM pool, retention 1

Performance under this configuration only (Pile, SEQ_LEN 1024, RAM backend, 1x H100 80GB, job ta-01M28XXSMF2WWWRWX2E50SYN1S); no RAM-vs-disk claim.
Wall = wrapper wall-clock around `_produce_pool_hf_rolling` (includes GPU->CPU bf16 copies); tokens/s = 16,384,000 / wall.

| layer | wall s | tokens/s | resident pool GB after produce | after cleanup | peak RSS GB | MemAvailable GB after |
|---|---|---|---|---|---|---|
| 0 | 35.0 | 467,475 | 75.5 |  | 83.1 | 1535.5 |
| 1 | 44.3 | 369,955 | 151.0 | 75.5 | 158.6 | 1459.8 |
| 2 | 43.2 | 379,127 | 151.0 | 75.5 | 158.6 | 1459.8 |
| 3 | 43.6 | 375,714 | 151.0 | 75.5 | 158.6 | 1459.8 |

Other phases: model fetch + load 38 s; 500 token shards streamed and tokenized in 98 s; trainer wall 308 s; job wall 316 s (5.3 min) before flow-back.
