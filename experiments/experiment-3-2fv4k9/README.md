# Push the 42 trained E4B SAE layers from the store to HF

## What this is
Task run: upload the 42 trained Gemma-4-E4B-it SAEs (layers 0 to 41, seed 0) produced by
`exp_01m1y0m9rsfnasj5vm2egaje0v` from the durable artifact store to the private Hugging Face
dataset `juiceb0xc0de/gemma-4-e4b-SAE`, plus the per-layer summary table `atlas_summary.csv`,
and verify the Hub listing.

## Layout
- `src/push_to_hub.sh`: the job script. `range_*` mode uploads one 14-layer folder with
  `hf upload`; `verify` mode uploads the summary CSV and writes `hub_listing.json`.
- `results/hub_listing.json`: copy of the Hub file listing after upload (written by the verify job).

## How to reproduce
Inputs (read-only store refs, staged with `input_artifacts`):
`artifact://juiceb0xc0de-15787e/experiments/exp_01m1y0m9rsfnasj5vm2egaje0v/{range_00_13,range_14_27,range_28_41,results}/`.

Because the on-demand fabric caps declared inputs at 32 GiB per job and each range folder is
21.9 GB, the upload runs as three parallel CPU-only jobs (`python:3.11-slim`, 8 CPU, 16 GB,
`HF_TOKEN` forwarded from job settings, `huggingface_hub[cli]` installed at job start):

    sh src/push_to_hub.sh range_00_13   # job 690867472644
    sh src/push_to_hub.sh range_14_27   # job 198464546303
    sh src/push_to_hub.sh range_28_41   # job 579631959165
    sh src/push_to_hub.sh verify        # after the three finish

## Outputs
- HF dataset (private): https://huggingface.co/datasets/juiceb0xc0de/gemma-4-e4b-SAE with
  `layer_NN_s0/{sae.pt,meta.json}` for NN = 00..41 and `atlas_summary.csv` at the root.
  Source: SAEs trained in `exp_01m1y0m9rsfnasj5vm2egaje0v` on `google/gemma-4-e4b-it`; uploaded 2026-09-07.
- `$SILICO_EXPERIMENT_ARTIFACTS_DIR/hub_listing.json` (verify job output, also in the artifact store).
