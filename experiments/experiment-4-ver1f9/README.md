# Do the Assistant-register features exist as a short list? Feature-level steering vs the raw direction at Gemma 4 E4B layer 10, plus how much of the Bella SFT moved along the axis

## What this is
Follow-up to the axis experiment (exp_01m1xz587ze4t8np5xr6016475). At `gemma-4-E4B-it` layer 10 we clamp the k most Assistant-ward JumpReLU SAE features (by decoder cosine with the Bella-minus-Gemma direction, and by token-level Cohen's d) to zero during generation and ask whether Bella-ness rises the way it does under the raw direction (Q1), whether clamping is gentler on neutral text (Q2), and how much of the Bella fine-tune's activation shift lies along the direction per layer (Q3).

## Layout
- `src/run_features.py`: the single H100 job (stage 1 rescoring, stage 4 SFT shift, stage 2 generation grid, stage 3 exemplars).
- `src/judge.py`: the axis experiment's judge, verbatim (gpt-5.4-mini, shared cache in `results/judge/cache.jsonl`).
- `results/data/`: prompts and pairs copied from the axis experiment (sha256 in `results/data_sha256.txt`).
- `results/prior/`: the axis experiment's E4B Gemma replies, stage-1 profile (median norms) and steering-cell table.
- `results/atlas/`, `results/sae_meta/`: atlas rlhf summary, SAE meta, direction stage-1 json.
- `results/smoke_summary.json`: interactive smoke result.

## How to reproduce
```
# GPU (1x H100, job-core image, HF_TOKEN):
HF_HUB_CACHE=$PWD/.hf_cache uv run python src/run_features.py --stages 1,4,2,3
# CPU:
apy src/judge.py run --gens <run>/generations.jsonl --model E4B
```

## Outputs
(filled at completion)
