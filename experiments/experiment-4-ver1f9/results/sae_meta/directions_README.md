# Bella-vs-Gemma persona directions

Unit Bella-minus-Gemma difference-of-means direction per layer for Gemma-4-E4B-it
(`e4b_*`) and Gemma-4-E2B-it (`e2b_*`). Source experiment: exp_01m1xz587ze4t8np5xr6016475.

- `*_directions.npz`: keys `all` (all reply tokens) and `first` (first 32 reply tokens);
  each array has shape `[n_layers, d_model]`, rows are unit vectors. Layer `l` = output of block `l`.
- `*_stage1.json`: per-layer Cohen's d, bootstrap CI, shuffled null p95, cross-layer cosines,
  and logit-lens top tokens.
