# Do the Assistant-register features exist as a short list? Feature-level steering vs the raw direction at Gemma 4 E4B layer 10, plus how much of the Bella SFT moved along the axis

## What this is
Follow-up to the axis experiment (exp_01m1xz587ze4t8np5xr6016475). At `gemma-4-E4B-it` layer 10 (and layer 4) we clamp the k most Assistant-ward JumpReLU SAE features (by decoder cosine with the Bella-minus-Gemma direction, and by Gemma-over-Bella per-token activation difference) to zero during generation and ask whether Bella-ness rises the way it does under the raw direction (Q1), whether clamping is gentler on neutral text (Q2), and how much of the Bella fine-tune's activation shift lies along the direction per layer (Q3).

## Result
- Q1 refuted: no feature cell (k = 5, 20, 50, 100, 200; three selection rules; Bella-ward amplification ×2/×4) reaches the Bella threshold (+1.26 = 70% of the direction's +1.80 gain at −0.35). Cells with k ≤ 100 deliver 1.5–9% of the direction's per-token shift along the Bella direction and gain ≤ +0.12. k = 200 cells gain +0.71 (A) / +1.21 (D) but collapse refusal (0.35 / 0.32 vs 0.99) and degenerate 36–55% of chat replies while delivering ≤ 10% of the direction's on-axis shift. Random matched sets of 200 features at the same edit norm (15.2/15.0 vs 15.3/18.2): gain −0.10/−0.09, refusal 0.94/0.92, 0% degeneration. The selected k = 200 sets do carry Bella-ward content (teacher-forced Bella-minus-Gemma reply log-prob +1.83/+2.15 nats per token vs base, more than the −0.35 direction's +1.40; random sets +0.49/+0.65) but cannot deliver it without breaking refusal, crisis handling and fluency; the judged gains may also be inflated by short degenerate replies (31.9/32.1 words vs 82.3 unsteered).
- Q2 not testable (no cell reaches the Bella threshold).
- Q3 refuted on the pre-registered cosine rule: cos(SFT shift, direction) = 0.52 at layer 4, 0.46 at layer 10, 0.30–0.78 everywhere (random-direction p95 0.034). Primary Q3 number: the fine-tune closes 3.1% of the Bella–Gemma gap at layer 4 and 2.6% at layer 10 (2–6% at layers 0–13, 29–43% at layers 29–41). Limitation: the cosine has not been separated from a shared component (a general chat-text or fine-tuning shift present in both the SFT delta and the Bella-minus-Gemma direction); `src/q3_mean_projected.py` shows it is unchanged after projecting out the per-layer mean residual or the 8 largest-|mean| coordinates (layer 4: 0.51, layer 10: 0.52), which rules out only a rank-1 component. The rise of gap closed with depth is a post hoc reading that tracks the shift norm, not confirmation of the predicted shape.

## Layout
- `src/run_features.py`: the H100 job (stage 1 rescoring, stage 4 SFT shift, stage 2 generation grid with exact per-token dose accounting, stage 3 exemplars). `--cells` selects cells, `--sub_bias_off` reproduces the layer-22 encoder-convention check.
- `src/judge.py`: the axis experiment's judge, verbatim (`gpt-5.4-mini`, shared cache in `results/judge/cache.jsonl`, gitignored).
- `src/analyze.py`: verdicts, per-cell table, plot-ready files (`--run results/run --run2 results/iter2`).
- `src/make_figures.py`, `figures/<name>/{plot.py,data.json,<name>.html}`: report figures (silico-figures).
- `src/make_report_summary.py`: writes `report_summary.json` and `claims_manifest.json` from the analysis files.
- `src/q3_mean_projected.py`: CPU shared-component check for Q3 (cosine after projecting out the per-layer mean residual / dropping massive-activation coordinates) from `run/sft_shift.npz`, `run/acts_means.npz`, `run/inputs/e4b_directions.npz` in the artifact store; writes `results/analysis/q3_mean_projected.{csv,json}`.
- `results/run/`: iteration-1 job outputs mirrored from the artifact store (generations, cells, dose response, stage1/3/4). `results/iter2/`: iteration-2 outputs (k=100 cells, Bella-ward amplification).
- `results/analysis/`: `cells.csv`/`cells.json` (27 cells), `verdicts.json`, `q1_curve.json`, `q2_scatter.json`, `q3_layers.csv`, `stage1_summary.json`, `feature_labels.json` (58 worker-written labels), `exemplar_sheet.txt`, `examples.json`.
- `results/data/`: prompts and pairs copied from the axis experiment (sha256 in `results/data_sha256.txt`). `results/prior/`: the axis experiment's E4B replies, stage-1 profile and steering-cell table. `results/atlas/`, `results/sae_meta/`: atlas rlhf summary, SAE meta.
- `results/smoke_summary.json`, `results/smoke2_review_fixes.json`: interactive smoke results, including the layer-22 encoder-convention numbers.
- `scratch/test_feature_edit_cpu.py`: CPU test of the FeatureEdit hook.

## How to reproduce
```
# GPU (1x H100, job-core image, HF_TOKEN), iteration 1:
HF_HUB_CACHE=$PWD/.hf_cache uv run python src/run_features.py --stages 1,4,2,3 --out "$SILICO_EXPERIMENT_ARTIFACTS_DIR/run"
# iteration 2 (adaptive cells):
uv run python src/run_features.py --stages 1,2 --cells amp:10:20:2,amp:10:20:4,A:10:100,D:10:100,R:10:100:0,R:10:100:1 --out "$SILICO_EXPERIMENT_ARTIFACTS_DIR/iter2"
# CPU:
apy src/judge.py run --gens results/run/generations.jsonl --model E4B-features --gens2 results/iter2/generations.jsonl --model2 E4B-features-iter2
apy src/analyze.py --run results/run --run2 results/iter2
uv run --no-sync python src/make_figures.py
apy src/q3_mean_projected.py --inputs <dir with sft_shift.npz, acts_means.npz, e4b_directions.npz>
apy src/make_report_summary.py
```

## Outputs
Date: 2026-09-09. Model `google/gemma-4-E4B-it` @ `ee0ef6023621cff504d758262d4e04895a5af4a2`; fine-tune `juiceb0xc0de/bella-bartender-gemma-e4b` @ `ac5ff55239c029c461a41bd2788d7423d0f2ac7c`; SAEs `juiceb0xc0de/gemma-4-e4b-SAE` @ `57bacb61a1bcc32be212a815b985a9bb42ff9a16` (layers 4/10/22); direction `directions/e4b_directions.npz` key `all`; atlas `atlas-gemma-4-e4b/rlhf`. Judge `gpt-5.4-mini` via the researcher's OpenAI key in the pod (10,530 generations, 8,640 judged items, 0 invalid).
- Iteration-1 job outputs (job 110128294020): `artifact://juiceb0xc0de-15787e/experiments/exp_01m21z6dx8e5r9zt6n0vver1f9/run/` (generations.jsonl, cells.json, dose_response.json, stage1.json, stage4.json, stage3_exemplars.json, acts_means.npz, features_L{4,10,22}.npz).
- Iteration-2 job outputs (job 615912779681): `artifact://juiceb0xc0de-15787e/experiments/exp_01m21z6dx8e5r9zt6n0vver1f9/iter2/run/`.
- Judge scores: `artifact://juiceb0xc0de-15787e/experiments/exp_01m21z6dx8e5r9zt6n0vver1f9/judge/judgments.jsonl`.
- Analysis and figures: `results/analysis/`, `figures/` in this directory; report contract `report_summary.json`, `claims_manifest.json`.
- Published report: the experiment's Results page.
