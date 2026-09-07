# Locate the Assistant persona direction in Gemma 4 E2B and E4B and test whether scaling it down frees the Bella voice

## What this is

For each layer of `gemma-4-E2B-it` and `gemma-4-E4B-it` we fit a difference-of-means direction between Bella's replies and Gemma's own greedy replies to the same user prompts, measure how well it separates the two voices on held-out pairs (Cohen's d, paired bootstrap, permutation null), and test the plan's three questions: Q1 does the E2B profile crest at layers 4 and 13/14 with a trough between; Q2 do crests fall on global-attention layers in both models; Q3 does subtracting the direction move Gemma toward Bella's register while refusals, crisis handling and fluency hold over at least two contiguous coefficients. An E2B SAE atlas is used to ask whether the direction is a few features.

Verdicts: Q1 supported (crests at 4 and 13, dip at 12, both crest-minus-dip bootstrap intervals exclude zero). Q2 refuted as registered (both models peak at absolute layer 4; 1 of 3 E2B and 0 of 3 E4B strongest crests on global layers); post hoc, surviving crests sit one layer before a global-attention block (E4B 5 of 5, E2B 3 of 4). Q3 not supported: Bella-ness rises from 1.1 to 3.0 to 4.5 / 7 at coefficient −0.5 to −0.65 with refusals mostly intact, but every layer collapses into repetition within one or two grid steps; no two-coefficient window passes the registered rule or the post-hoc fluency variant.

## Layout

- `src/prepare_data.py`: cleans the Bella corpus, builds matched pairs (600, split by prompt 400/100/100), eval/crisis/red-team/neutral/authentic/corporate sets → `results/data/` (counts in `results/data/manifest.json`).
- `src/run_model.py`: GPU pipeline. Stage 1 direction fitting and layer profile (`stage1.json`, `acts_pairs.npz`, `directions.npz`, `gemma_replies.jsonl`); stage 2 steering (`steering_generations.jsonl`, `dose_response.json`, `neutral_perplexity.json`, `stage2_meta.json`); stage 3 E2B SAE analysis (`stage3.json`). `--coefs/--iteration/--seed_from` add cells to an existing run.
- `src/post_stage1.py`: CPU post-processing of `acts_pairs.npz`: per-layer paired bootstrap of d, crest-minus-trough intervals, proper permutation null, length confound → `stage1_boot.json`.
- `src/judge.py`: LLM judge (`gpt-5.4-mini`, OpenAI API, logprob expected scores) with pilot gate; `results/judge/{pilot.json, judge_meta.json, graded_sample_60.md}`; `judgments.jsonl` and `cache.jsonl` are git-ignored (large) and uploaded to the artifact store.
- `src/analyze.py`: decision rules, steering table, examples, figures → `results/analysis/summary.json`, `figures/<name>/`.
- `src/make_report_contract.py`: writes `report_summary.json` and `claims_manifest.json` from `summary.json`.
- `results/artifacts/<model>/{run,iter2}/`: small JSON copies of the job outputs used by `analyze.py`.
- `results/plan_sketch_e2b.svg`: the researcher's predicted-profile sketch from the plan.
- `scratch/`: one-off helpers.

## How to reproduce

```
apy src/prepare_data.py                                  # results/data/
# GPU (H100, job-core image, HF_TOKEN):
uv run python src/run_model.py --model E2B --stages 1,2,3
uv run python src/run_model.py --model E4B --stages 1,2
uv run python src/run_model.py --model E2B --stages 2 --coefs=-0.15,-0.35,-0.65,-0.8 --iteration 2 --seed_from <E2B/run> --out <root>/iter2
uv run python src/run_model.py --model E4B --stages 2 --coefs=-0.15,-0.35,-0.65,-0.8 --iteration 2 --seed_from <E4B/run> --out <root>/iter2
# CPU:
uv run python src/post_stage1.py --acts <run>/acts_pairs.npz --dirs <run>/directions.npz --stage1 <run>/stage1.json --out <run>/stage1_boot.json
apy src/judge.py pilot --gens results/artifacts/E2B/run/steering_generations.jsonl
apy src/judge.py run --gens results/artifacts/E2B/iter2/steering_generations.jsonl --model E2B --gens2 results/artifacts/E4B/iter2/steering_generations.jsonl --model2 E4B
uv run --no-sync python src/analyze.py
apy src/make_report_contract.py
```

Seed 42 throughout. Models: `google/gemma-4-E2B-it` @ `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, `google/gemma-4-E4B-it` @ `ee0ef6023621cff504d758262d4e04895a5af4a2`, bf16, sdpa. Layer l = output of decoder block l.

## Outputs

- GPU job outputs (artifact store, 2026-09-07): `artifact://juiceb0xc0de-15787e/experiments/exp_01m1xz587ze4t8np5xr6016475/E2B/run/` (stages 1 to 3, incl. `acts_pairs.npz` 412 MB), `.../E4B/run/` (stages 1, 2; `acts_pairs.npz` 868 MB), `.../iter2/E2B/run/` and `.../iter2/E4B/run/` (steering with all ten coefficients), `.../E2B/smoke/` (smoke test). `stage1_boot.json` in each `<model>/run/`.
- Judge outputs: `artifact://juiceb0xc0de-15787e/experiments/exp_01m1xz587ze4t8np5xr6016475/judge/judgments.jsonl` (26,240 rows) and `judge/cache.jsonl`; model `gpt-5.4-mini` via the researcher's `OPENAI_API_KEY` (plan named claude-sonnet-5; no Anthropic credential available).
- Analysis: `results/analysis/summary.json`, `results/steering_cells.json`, `results/profiles_with_uncertainty.json`, `figures/`.
- SAE atlas used: HF dataset `juiceb0xc0de/gemma-4-e2b-it-SAE`, `layer_NN_s0/sae.pt` (JumpReLU, layers 4, 9, 19, 28).
- Bella corpus: HF dataset registered in the thread resources (10,389 raw rows → 7,232 cleaned).
- Report: published Results page of this experiment.
