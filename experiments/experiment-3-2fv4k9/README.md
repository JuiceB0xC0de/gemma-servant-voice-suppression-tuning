# Push the 42 trained E4B SAE layers from the store to HF

## What this is
Task run: upload the 42 trained Gemma-4-E4B-it SAEs (layers 0 to 41, seed 0) produced by
`exp_01m1y0m9rsfnasj5vm2egaje0v` from the durable artifact store to the private Hugging Face
dataset `juiceb0xc0de/gemma-4-e4b-SAE`, plus the per-layer summary table `atlas_summary.csv`,
and verify the Hub listing.

## Layout
- `src/push_to_hub.sh`: the job script. Modes:
  `layer <range_XX_YY> <NN>` uploads one layer dir as `layer_NN_s0/` (the mode that worked);
  `range_*` uploads a whole 14-layer folder (worked once, for range_28_41);
  `verify` uploads the summary CSV and writes `hub_listing.json`;
  `private` sets the repo private and rewrites `hub_listing.json`.
- `src/hub_tree.py`: pod-side check of which layers are complete/partial/missing on the Hub.
- `results/hub_listing.json`: copy of the Hub file listing after upload (written by the verify job).

## How to reproduce
Inputs (read-only store refs, staged with `input_artifacts`):
`artifact://juiceb0xc0de-15787e/experiments/exp_01m1y0m9rsfnasj5vm2egaje0v/{range_00_13,range_14_27,range_28_41}/saes/google_gemma-4-e4b-it/layer_NN_s0/`
and `.../results/` (contains `atlas_summary.csv`).

All jobs: `python:3.11-slim`, CPU only (4 CPU, 8 GB), `HF_TOKEN` forwarded from job settings
(write-scoped), `huggingface_hub[cli]` 1.30.0 installed at job start, no other packages.

    # one job per layer, each staging one ~1.7 GB layer directory
    sh src/push_to_hub.sh layer range_00_13 00     # ... through range_28_41 41
    sh src/push_to_hub.sh verify                   # uploads atlas_summary.csv, writes hub_listing.json
    sh src/push_to_hub.sh private                  # only needed because the repo was created public

What did not work, and why the procedure looks like this:
- One job with all four inputs is refused: declared inputs are capped at 32 GiB per job (total 70 GB).
- One job per 14-layer range folder (21.9 GB staged) died three times during input staging with
  no log; only range_28_41 ever completed that way (commit 559bdede).
- The first attempt failed with `403 Forbidden ... create a dataset under the namespace` because the
  configured `HF_TOKEN` was read-only; the researcher swapped in a write token.
- `hf upload --private` did not make the repo private (visibility is fixed at creation), so a final
  `update_repo_settings(private=True)` job was needed.

## Outputs
- HF dataset (private): https://huggingface.co/datasets/juiceb0xc0de/gemma-4-e4b-SAE with
  `layer_NN_s0/{sae.pt,meta.json}` for NN = 00..41 (each `sae.pt` 1,678,389,613 bytes) and
  `atlas_summary.csv` (5,141 bytes) at the root; 70,492,474,314 bytes total. A stray 74-byte
  `atlas_marker_s0.json` at the root came along with the range_28_41 folder upload.
  Source: SAEs trained in `exp_01m1y0m9rsfnasj5vm2egaje0v` on `google/gemma-4-e4b-it`; uploaded 2026-09-07.
  Layers flagged in the source experiment for explained variance below 0.85: 17, 18, 19, 23
  (uploaded unchanged; quality was not re-verified here).
- `artifact://juiceb0xc0de-15787e/experiments/exp_01m1z0wefjes6r08s21t2fv4k9/hub_listing.json`
  (also `results/hub_listing.json`), plus per-layer `upload_layer_NN.txt` commit receipts in the same store folder.

## Follow-up: persona directions (2026-09-08)
`src/push_directions.sh` (job 407396606881) added `directions/{e4b,e2b}_{directions.npz,stage1.json}` and
`directions/README.md` from `exp_01m1xz587ze4t8np5xr6016475` (commit 57bacb61); listing in
`artifact://juiceb0xc0de-15787e/experiments/exp_01m1z0wefjes6r08s21t2fv4k9/directions_listing.json`.
Note: `hf upload` without `--private` flipped the repo back to public; `sh src/push_to_hub.sh private`
was rerun afterwards. Always re-check `dataset_info().private` after any upload to this repo.
