#!/usr/bin/env python3
"""Pod-side: turn the job's results (job_record.json, calibration/*.json|csv,
rolling_check.json, walk_*.json) into results/*.md|csv, SUMMARY.md and two figure
bundles (decoder-overlap distributions; produce time per layer by scratch placement).

    uv run --no-sync python src/make_reports.py --job-results <dir with results/> \
        --hf-repo juiceb0xc0de/gemma-2-2b-SAE
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def sig(x, n=4):
    """n significant figures, plain."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    if x == 0:
        return "0"
    d = n - int(math.floor(math.log10(abs(x)))) - 1
    return f"{round(x, d):.{max(d, 0)}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-results", required=True, help="local dir containing results/ (fetched)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--hf-repo", default="juiceb0xc0de/gemma-2-2b-SAE")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    R = Path(args.job_results) / "results"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rec = json.loads((R / "job_record.json").read_text())
    cal_rows = list(csv.DictReader(open(R / "calibration/calibration.csv")))
    cal_raw = json.loads((R / "calibration/calibration_raw.json").read_text())
    ov = json.loads((R / "calibration/decoder_overlap.json").read_text())
    rolling = json.loads((R / "rolling_check.json").read_text())
    tokprep = json.loads((R / "token_prep.json").read_text())
    train_layers = sorted(int(k) for k in rec["layers"].keys())
    env = rec["env"]
    phases = {p["phase"]: p["seconds"] for p in rec["phases"]}

    # ---------------------------------------------------------------- rolling check
    (out / "rolling_check.json").write_text(json.dumps(rolling, indent=1))

    # ---------------------------------------------------------------- calibration
    (out / "calibration.csv").write_text((R / "calibration/calibration.csv").read_text())
    ref = cal_raw["reference"]
    lines = ["# Calibration: event-aware SAE trainer vs Gemma Scope on Gemma 2 2B", ""]
    lines += [f"Model `google/gemma-2-2b` (26 layers, d_in 2304), trainer commit `{rec['trainer_commit']}`, "
              f"image `{rec['image_ref']}`, job `{rec['job_id']}`; torch {env['torch']}, transformers "
              f"{env['transformers']}, {env['gpu']} (driver {env['driver']}).", ""]
    lines += ["## Setup", "",
              "- Ours: JumpReLU SAE, 16,128 features (7 x 2304; the trainer exposes an integer expansion only), "
              "target L0 50, pool 500 shards x 32,768 tokens = 16.4 M tokens of FineWeb-Edu per layer, seed 0, "
              "one seed. Each layer trains for at most 5,000 steps x 32,768 tokens = 164 M token presentations "
              "(about 10 passes over the 16.4 M-token pool), stopping early on the trainer's EV/L0 gates.",
              "- Reference: `google/gemma-scope-2b-pt-res`, `width_16k` (16,384 features), release nearest L0 50 per layer; "
              "Gemma Scope SAEs were trained on 4 B tokens (Lieberum et al., 2024).",
              "- Held-out sets, identical tokens for both SAEs, position 0 (BOS) excluded from every metric: "
              f"(a) FineWeb-Edu shards {tokprep['heldout']['shard_ids'][0]}-{tokprep['heldout']['shard_ids'][-1]} "
              "of the trainer's own token stream (the 500 training shards are 0-499), 61 x 32,768 = 2.0 M tokens; "
              "(b) C4 English validation split, first 61 x 32,768 = 2.0 M tokens, tokenized with the same recipe.",
              "- BOS handling: the trainer prepends BOS to every 2048-token row and the pools keep position 0, so our SAE "
              "was trained *with* BOS-position activations (1/2048 of tokens). Gemma Scope excludes BOS. The comparison "
              "excludes position 0 for both.",
              "- LM loss recovered = (loss_zero - loss_sae) / (loss_zero - loss_clean) with the block-L residual replaced at "
              "positions >= 1; EV = 1 - sum||x - x_hat||^2 / sum||x - mean||^2 in fp32; dead = features never active on the 2 M tokens.",
              ""]
    lines += ["## Per-layer results", ""]
    hdr = "| layer | source | SAE | features | stated/target L0 | EV | mean L0 | dead % | LM loss recovered | CE increase (nats) | ref subfolder |"
    lines += [hdr, "|" + "---|" * 11]
    for r in cal_rows:
        L = r["layer"]
        lines.append(f"| {L} | {r['source']} | ours | {r['ours_n_features']} | 50 | {sig(float(r['ours_ev']))} | "
                     f"{sig(float(r['ours_l0']))} | {sig(100 * float(r['ours_dead_frac']), 3)} | "
                     f"{sig(float(r['ours_loss_recovered']))} | {sig(float(r['ours_ce_increase']))} | - |")
        lines.append(f"| {L} | {r['source']} | Gemma Scope | {r['ref_n_features']} | {r['ref_stated_l0']} | {sig(float(r['ref_ev']))} | "
                     f"{sig(float(r['ref_l0']))} | {sig(100 * float(r['ref_dead_frac']), 3)} | "
                     f"{sig(float(r['ref_loss_recovered']))} | {sig(float(r['ref_ce_increase']))} | `{r['ref_subfolder']}` |")
    lines += ["", "Clean LM loss and zero-ablation loss per source/layer:", ""]
    lines += ["| layer | source | loss clean | loss zero-ablation |", "|---|---|---|---|"]
    for r in cal_rows:
        lines.append(f"| {r['layer']} | {r['source']} | {sig(float(r['loss_clean']))} | {sig(float(r['loss_zero_ablation']))} |")
    lines += ["", "## Training budget and stop reasons (ours)", "",
              "| layer | steps | tokens seen | stop | final EV (train batch) | final L0 | final dead % | W&B |",
              "|---|---|---|---|---|---|---|---|"]
    for L in train_layers:
        d = rec["layers"][str(L)]["layers"][str(L)]
        m = ref[str(L)]["ours_train_final"]
        stop = "; ".join(d.get("stop_lines", [])) or "max steps"
        lines.append(f"| {L} | {ref[str(L)]['ours_train_steps']} | {ref[str(L)]['ours_train_tokens']:,} | {stop} | "
                     f"{sig(m.get('ev'))} | {sig(m.get('mean_l0'))} | {sig(m.get('dead_pct'), 3)} | {d.get('wandb_url', '')} |")
    lines += ["", "## Decoder-direction overlap", "",
              "For each feature, the max cosine between its decoder direction and any decoder direction of the other SAE.", "",
              "| layer | direction | frac > 0.7 | frac > 0.9 | mean | median |", "|---|---|---|---|---|---|"]
    for L in train_layers:
        for k, name in (("ours_to_ref", "ours -> Gemma Scope"), ("ref_to_ours", "Gemma Scope -> ours")):
            o = ov["layers"][str(L)][k]
            lines.append(f"| {L} | {name} | {sig(o['frac_gt_0.7'])} | {sig(o['frac_gt_0.9'])} | {sig(o['mean'])} | {sig(o['median'])} |")
    lines += ["", "## Mismatches stated", ""]
    for L in train_layers:
        rL = ref[str(L)]
        lines.append(f"- Layer {L}: width 16,128 (ours) vs 16,384 (reference); reference release `{rL['reference_subfolder']}` "
                     f"(stated L0 {rL['reference_stated_l0']}, available: {rL['reference_available_l0']}); measured L0 on held-out text in the table above.")
    lines += ["", "## Rolling-forward check", "",
              f"Capture path selected by the trainer for Gemma 2: **{rolling['trainer_capture_path']}** "
              f"(`_is_hf_rolling_supported`={rolling['trainer_hf_rolling_supported']}, generic={rolling['trainer_generic_rolling_supported']}); "
              f"attention implementation `{rolling['attn_implementation']}`, sliding window {rolling['sliding_window']} > SEQ_LEN 2048.",
              f"Max |diff| between the single-block walk and the model's own forward over {rolling['n_batches']} token shards x {rolling['probe_shape'][0]} sequences, all 26 layers: "
              f"**{max(r['max_abs_diff'] if 'max_abs_diff' in r else r['max_abs'] for r in rolling['layers']):.3e}** (all layers ok: {rolling['all_ok']}). "
              "Per-layer values in `results/rolling_check.json`.", "",
              "## Trainer limitations found on Gemma 2 (no source changes made)", "",
              "1. `--capture rolling-hf-float` fails on the first shard (`Expected all tensors to be on the same device`): the "
              "production loop only hoists the active block to GPU on the Gemma-4 branch, never on the HF branch. Model-agnostic, "
              "not Gemma-2-specific. Worked around by `rolling-hf` (full model resident on GPU, 5.2 GB).",
              "2. Under `rolling-hf`, a mid-chain start (`--layer-range 5,5`) bootstraps with a hooked full forward and never "
              "walks the chain, and the float variant that would keep the walk is item 1. The chain walk was therefore driven "
              "by `src/walk_layers.py`, which calls the trainer's own producer primitives in the same order (byte-identical "
              "shards, checked on 5 shards); the trainer then finds each target layer's pool and resumes at it.",
              "3. Skip-train pools are never deleted by the trainer (the E4B run patched this); the harness deletes each consumed "
              "source pool instead, keeping two pools on disk.",
              "4. `pool_forward_fusion` is ignored on the HF rolling path; `_produce_pool_hf_rolling` prints no read/forward/write split "
              "(the harness measures it).",
              "5. The trainer's HF push creates public repos (`private=False`); the private dataset repo was created beforehand and "
              "pushed with `hf upload` per layer.",
              "6. Training pools include the BOS-position activation (see BOS handling above).",
              ""]
    (out / "calibration.md").write_text("\n".join(lines))

    # ---------------------------------------------------------------- timing
    trows = []
    for kind, ab in rec["ab"].items():
        for row in ab["walk"]["layers"]:
            trows.append({"placement": kind, "phase": "produce", "layer": row["layer"], "seconds": row["total_s"],
                          "read_s": row["read_s"], "fwd_s": row["fwd_s"], "write_s": row["write_s"],
                          "tokens_per_s": row["tokens_per_s"], "gpu_util_mean": row.get("gpu_util_mean"),
                          "scratch_tag": f"scratch={kind}"})
        L = ab["walk"]["layers"][-1]["layer"]
        d = ab["train"]["layers"][str(L)]
        trows.append({"placement": kind, "phase": "train_200steps", "layer": L, "seconds": d.get("train_wall_s"),
                      "step_ms_mean": d.get("step_ms_mean"), "step_ms_p50": d.get("step_ms_p50"),
                      "step_ms_p95": d.get("step_ms_p95"), "steps_per_s": d.get("steps_per_s_from_step_time"),
                      "bdec_init_s": d.get("bdec_init_s"), "gpu_util_mean": d.get("gpu_util_mean_train"),
                      "scratch_tag": f"scratch={kind}"})
        trows.append({"placement": kind, "phase": "resume_pool_copy", "layer": L,
                      "seconds": d.get("post_save_resume_pool_s"), "scratch_tag": f"scratch={kind}"})
    for row in rec["chain"]["walk_layers"]:
        trows.append({"placement": "fast", "phase": "chain_produce", "layer": row["layer"], "seconds": row["total_s"],
                      "read_s": row["read_s"], "fwd_s": row["fwd_s"], "write_s": row["write_s"],
                      "tokens_per_s": row["tokens_per_s"], "gpu_util_mean": row.get("gpu_util_mean"),
                      "scratch_tag": "scratch=fast"})
    for L in train_layers:
        d = rec["layers"][str(L)]["layers"][str(L)]
        trows.append({"placement": "fast", "phase": "chain_train", "layer": L, "seconds": d.get("train_wall_s"),
                      "step_ms_mean": d.get("step_ms_mean"), "step_ms_p50": d.get("step_ms_p50"),
                      "step_ms_p95": d.get("step_ms_p95"), "steps_per_s": d.get("steps_per_s_from_step_time"),
                      "bdec_init_s": d.get("bdec_init_s"), "gpu_util_mean": d.get("gpu_util_mean_train"),
                      "steps": ref[str(L)]["ours_train_steps"], "scratch_tag": "scratch=fast"})
        trows.append({"placement": "fast->volume", "phase": "chain_resume_pool_copy", "layer": L,
                      "seconds": d.get("post_save_resume_pool_s"), "scratch_tag": "scratch=fast"})
    cols = ["placement", "phase", "layer", "seconds", "read_s", "fwd_s", "write_s", "tokens_per_s", "step_ms_mean",
            "step_ms_p50", "step_ms_p95", "steps_per_s", "steps", "bdec_init_s", "gpu_util_mean", "scratch_tag"]
    with open(out / "timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in trows:
            w.writerow({c: r.get(c) for c in cols})

    tl = ["# Timing: pools on the per-job volume vs container-local disk", ""]
    tl += [f"Job `{rec['job_id']}`, {env['gpu']}, {env['cpu_count']} CPU, {env['ram_total_gb']:.0f} GB RAM, /dev/shm "
           f"{env['shm_bytes'] / 1e9:.0f} GB (fixed; too small for a 75.5 GB pool, so RAM placement was not possible).", "",
           "- **volume** = the per-job Modal volume (`$SILICO_EXPERIMENT_ARTIFACTS_DIR`), measured 0.95 GB/s sequential write (dd, 2 GiB, fdatasync).",
           "- **fast** = container-local overlay root disk (`/local-scratch`), measured 2.9 GB/s sequential write; 300 GiB write-probed "
           "(the filesystem reports no meaningful size, so the trainer's free-space check prints `(overlay: unknown)` and is skipped).",
           "- Pool = 500 shards x 151 MB (bf16 [16, 2048, 2304]) = 75.5 GB per layer. Walk timing is from `src/walk_layers.py`, which "
           "runs the trainer's producer primitives with read / forward / write timers; training timing is the trainer's own "
           "`[STEP-TIME]` lines (every step for the A/B, every 10 steps for the chain) and wall clock between its banner and `Saved:`.",
           ""]
    tl += ["## A/B: walk layers 0-3 (produce only) then train layer 3 for 200 steps", "",
           "| placement | layer | produce s | read s | fwd s | write s | tokens/s | GPU util % |", "|---|---|---|---|---|---|---|---|"]
    for r in trows:
        if r["phase"] == "produce":
            tl.append(f"| {r['placement']} | {r['layer']} | {sig(r['seconds'])} | {sig(r['read_s'])} | {sig(r['fwd_s'])} | "
                      f"{sig(r['write_s'])} | {sig(r['tokens_per_s'], 4)} | {sig(r.get('gpu_util_mean'), 3)} |")
    tl += ["", "| placement | train wall s (200 steps incl. b_dec init) | b_dec init s | step ms mean | step ms p50 | step ms p95 | steps/s | GPU util % | resume-pool copy s (75.5 GB) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for kind in rec["ab"]:
        tr = next(r for r in trows if r["placement"] == kind and r["phase"] == "train_200steps")
        cp = next(r for r in trows if r["placement"] == kind and r["phase"] == "resume_pool_copy")
        tl.append(f"| {kind} | {sig(tr['seconds'])} | {sig(tr.get('bdec_init_s'))} | {sig(tr.get('step_ms_mean'))} | {sig(tr.get('step_ms_p50'))} | "
                  f"{sig(tr.get('step_ms_p95'))} | {sig(tr.get('steps_per_s'))} | {sig(tr.get('gpu_util_mean'), 3)} | {sig(cp['seconds'])} |")
    tl += ["", "## Full chain on fast scratch: walk 0 -> 20, train 5, 12, 20", "",
           "| layer | produce s | read s | fwd s | write s | tokens/s | GPU util % |", "|---|---|---|---|---|---|---|"]
    for r in trows:
        if r["phase"] == "chain_produce":
            tl.append(f"| {r['layer']} | {sig(r['seconds'])} | {sig(r['read_s'])} | {sig(r['fwd_s'])} | {sig(r['write_s'])} | "
                      f"{sig(r['tokens_per_s'], 4)} | {sig(r.get('gpu_util_mean'), 3)} |")
    tot_walk = sum(r["seconds"] for r in trows if r["phase"] == "chain_produce")
    tl += ["", f"Total walk 0 -> 20: **{tot_walk / 60:.1f} min** for 21 layers ({tot_walk / 21:.0f} s per layer).", "",
           "| layer | steps | train wall s | step ms mean | step ms p95 | steps/s | GPU util % | resume-pool copy to volume s |",
           "|---|---|---|---|---|---|---|---|"]
    for L in train_layers:
        tr = next(r for r in trows if r["phase"] == "chain_train" and r["layer"] == L)
        cp = next(r for r in trows if r["phase"] == "chain_resume_pool_copy" and r["layer"] == L)
        tl.append(f"| {L} | {tr.get('steps')} | {sig(tr['seconds'])} | {sig(tr.get('step_ms_mean'))} | {sig(tr.get('step_ms_p95'))} | "
                  f"{sig(tr.get('steps_per_s'))} | {sig(tr.get('gpu_util_mean'), 3)} | {sig(cp['seconds'])} |")
    tl += ["", "## Job phases (time/*)", "", "| phase | seconds |", "|---|---|"]
    for p in rec["phases"]:
        tl.append(f"| {p['phase']} | {sig(p['seconds'])} |")
    tl += ["", f"Job total {rec.get('job_total_s', 0) / 60:.1f} min; GPU busy (util >= 10%, 5 s samples) "
           f"{rec.get('gpu_busy_total_s', 0) / 60:.1f} min.", "",
           f"W&B timing run: {rec.get('timing_wandb_url')}", ""]
    (out / "timing.md").write_text("\n".join(tl))

    # ---------------------------------------------------------------- SUMMARY
    def cal(L, src, who, key):
        r = next(r for r in cal_rows if int(r["layer"]) == L and r["source"] == src)
        return float(r[f"{who}_{key}"])
    S = ["# Gemma 2 2B: event-aware SAE trainer calibrated against Gemma Scope", "",
         "**Token budget.** Our three JumpReLU SAEs each saw at most 164 M token presentations "
         "(<= 5,000 steps x 32,768 tokens, about 10 passes over a 16.4 M-token FineWeb-Edu pool; actual: "
         + ", ".join(f"layer {L} {ref[str(L)]['ours_train_tokens'] / 1e6:.0f} M" for L in train_layers)
         + "). Gemma Scope's release SAEs were trained on 4 B tokens each, roughly "
         f"{4e9 / max(ref[str(L)]['ours_train_tokens'] for L in train_layers):.0f}x more, on a 16,384-feature dictionary vs our 16,128.", "",
         "**What was run.** One H100 job: `event-aware-SAE-trainer` at commit `c796c2ee` (no source changes) on "
         "`google/gemma-2-2b`, residual-stream SAEs at the outputs of blocks 5, 12 and 20, target L0 50, 7x expansion, "
         "seed 0 (single seed). Activation pools were produced by the trainer's HF single-block rolling walk (verified bit-exact "
         "against the model's own forward on all 26 layers) on container-local disk; the same walk and a 200-step training "
         "were also timed with pools on the per-job network volume. Both SAEs were then scored on the same 2 x 2.0 M held-out "
         "tokens (FineWeb-Edu continuation of the training stream; C4 English validation), BOS excluded.", "",
         "## Calibration (held-out, identical tokens, BOS excluded)", "",
         "| layer | source | EV ours / Gemma Scope | L0 ours / GS | dead % ours / GS | LM loss recovered ours / GS |",
         "|---|---|---|---|---|---|"]
    for L in train_layers:
        for src in cal_raw["sources"]:
            S.append(f"| {L} | {src} | {sig(cal(L, src, 'ours', 'ev'))} / {sig(cal(L, src, 'ref', 'ev'))} | "
                     f"{sig(cal(L, src, 'ours', 'l0'), 3)} / {sig(cal(L, src, 'ref', 'l0'), 3)} | "
                     f"{sig(100 * cal(L, src, 'ours', 'dead_frac'), 3)} / {sig(100 * cal(L, src, 'ref', 'dead_frac'), 3)} | "
                     f"{sig(cal(L, src, 'ours', 'loss_recovered'))} / {sig(cal(L, src, 'ref', 'loss_recovered'))} |")
    S += ["", "Reference releases: " + ", ".join(f"layer {L} `{ref[str(L)]['reference_subfolder']}`" for L in train_layers)
          + " (nearest stated L0 to 50). Decoder-direction overlap (fraction of features with max cosine > 0.7 to the other SAE): "
          + "; ".join(f"L{L} ours->GS {sig(ov['layers'][str(L)]['ours_to_ref']['frac_gt_0.7'], 3)}, GS->ours "
                      f"{sig(ov['layers'][str(L)]['ref_to_ours']['frac_gt_0.7'], 3)}" for L in train_layers) + ".", ""]
    S += ["## Pool placement timing (same trainer code, same tokens)", "",
          "| placement | walk layers 0-3, s per layer (mean) | write share | tokens/s | 200-step train: step ms mean / p95 | resume-pool copy (75.5 GB) s |",
          "|---|---|---|---|---|---|"]
    for kind in rec["ab"]:
        pr = [r for r in trows if r["placement"] == kind and r["phase"] == "produce"]
        tr = next(r for r in trows if r["placement"] == kind and r["phase"] == "train_200steps")
        cp = next(r for r in trows if r["placement"] == kind and r["phase"] == "resume_pool_copy")
        S.append(f"| {kind} | {sig(sum(r['seconds'] for r in pr) / len(pr), 3)} | "
                 f"{100 * sum(r['write_s'] for r in pr) / sum(r['seconds'] for r in pr):.0f}% | "
                 f"{sig(sum(r['tokens_per_s'] for r in pr) / len(pr), 4)} | {sig(tr.get('step_ms_mean'), 3)} / {sig(tr.get('step_ms_p95'), 3)} | {sig(cp['seconds'], 3)} |")
    S += ["", f"Full chain on fast scratch: walk 0 -> 20 took {tot_walk / 60:.1f} min ({tot_walk / 21:.0f} s per layer); "
          + "; ".join(f"layer {L} trained {ref[str(L)]['ours_train_steps']} steps in "
                      f"{sig((rec['layers'][str(L)]['layers'][str(L)].get('train_wall_s') or 0) / 60, 3)} min" for L in train_layers)
          + f". Job total {rec.get('job_total_s', 0) / 60:.0f} min on one H100. volume = per-job Modal volume (0.95 GB/s); "
            "fast = container-local overlay disk (2.9 GB/s); /dev/shm is fixed at 40 GB so RAM placement was not possible.", "",
          "Single seed; one training run per layer; the A/B is one repetition per placement.", "",
          "## Links", "",
          f"- HF dataset (private): https://huggingface.co/datasets/{args.hf_repo} (`layer_05_s0`, `layer_12_s0`, `layer_20_s0`, each `sae.pt` + `meta.json`)",
          "- W&B showcase: https://wandb.ai/ricks-holmberg-juiceb0xc0de/gemma-2-2b-SAE (runs L05_s0, L12_s0, L20_s0, summary)",
          f"- W&B timing: {rec.get('timing_wandb_url')}",
          "- Details: `results/calibration.md`, `results/timing.md`, `results/rolling_check.json`", ""]
    (out / "SUMMARY.md").write_text("\n".join(S))

    if args.no_figures:
        return
    # ---------------------------------------------------------------- figures
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from silico_figures import apply_theme, save_figure_bundle, EDITORIAL_8
    layer_colors = {L: EDITORIAL_8[i % len(EDITORIAL_8)] for i, L in enumerate(train_layers)}
    edges = ov["layers"][str(train_layers[0])]["hist_bin_edges"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Ours -> Gemma Scope (16,128 features)", "Gemma Scope -> ours (16,384 features)"))
    data_out = {"bin_centers": centers, "series": {}}
    for row_i, key in ((1, "ours_to_ref"), (2, "ref_to_ours")):
        for L in train_layers:
            counts = ov["layers"][str(L)][key]["hist_counts"]
            tot = sum(counts)
            frac = [c / tot for c in counts]
            data_out["series"][f"L{L}_{key}"] = frac
            fig.add_trace(go.Scatter(x=centers, y=frac, mode="lines+markers", name=f"layer {L}",
                                     legendgroup=f"L{L}", showlegend=(row_i == 1),
                                     line=dict(color=layer_colors[L], shape="hvh")), row=row_i, col=1)
    fig.update_xaxes(title_text="Max cosine to the other SAE's decoder directions", row=2, col=1, range=[0, 1])
    fig.update_yaxes(title_text="Fraction of features", row=1, col=1)
    fig.update_yaxes(title_text="Fraction of features", row=2, col=1)
    apply_theme(fig, height=560)
    save_figure_bundle(fig, "decoder_overlap_hist", data=data_out,
                       alt="Step histograms of per-feature max decoder cosine between our SAE and Gemma Scope, "
                           "for layers 5, 12 and 20, in both directions.")

    fig2 = go.Figure()
    placements = list(rec["ab"].keys())
    pcol = {"volume": EDITORIAL_8[0], "fast": EDITORIAL_8[1]}
    d2 = {"layers": [], "placement": [], "read_s": [], "fwd_s": [], "write_s": []}
    for kind in placements:
        rows = [r for r in trows if r["placement"] == kind and r["phase"] == "produce"]
        xs = [f"L{r['layer']}" for r in rows]
        fig2.add_trace(go.Bar(name=f"{kind} scratch", x=xs, y=[r["seconds"] for r in rows],
                              marker_color=pcol.get(kind),
                              customdata=[[r["read_s"], r["fwd_s"], r["write_s"]] for r in rows],
                              hovertemplate="%{x} %{y:.0f} s<br>read %{customdata[0]:.0f} s, forward %{customdata[1]:.0f} s, "
                                            "write %{customdata[2]:.0f} s<extra>" + kind + "</extra>"))
        for r in rows:
            d2["layers"].append(r["layer"]); d2["placement"].append(kind)
            d2["read_s"].append(r["read_s"]); d2["fwd_s"].append(r["fwd_s"]); d2["write_s"].append(r["write_s"])
    fig2.update_layout(barmode="group")
    fig2.update_xaxes(title_text="Layer (pool produced)")
    fig2.update_yaxes(title_text="Produce time (s), 500 shards x 32,768 tokens")
    apply_theme(fig2, height=400)
    save_figure_bundle(fig2, "produce_time_by_placement", data=d2,
                       alt="Grouped bars of per-layer pool production time for layers 0-3 with pools on the network volume "
                           "versus container-local disk.")
    print("wrote", out, "and figures/")


if __name__ == "__main__":
    main()
