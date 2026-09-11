#!/usr/bin/env python3
"""Compare our JumpReLU SAEs with the Gemma Scope release SAEs on identical held-out
tokens, layer by layer.

For each layer L in --layers and each held-out source, the residual stream at the
output of block L (`hidden_states[L+1]`, i.e. the trainer's pool surface) is fed to
both SAEs. Position 0 (BOS) is excluded from every metric for both SAEs, matching
Gemma Scope's evaluation. Metrics:
  * explained variance  1 - sum||x - x_hat||^2 / sum||x - mean(x)||^2   (fp32)
  * mean L0             mean number of active features per token
  * dead fraction       features that never fire on this text
  * LM loss recovered   (loss_zero - loss_sae) / (loss_zero - loss_clean), where the
                        block-L residual is replaced by the SAE reconstruction (or by
                        zeros) at every non-BOS position
  * decoder overlap     per feature, max cosine to any decoder direction of the other
                        SAE (both directions); fractions > 0.7 and > 0.9; histogram

Our SAE:      pre = W_enc (x - b_dec) + b_enc ; z = pre * [pre > exp(log_threshold)]
Gemma Scope:  pre = x W_enc + b_enc           ; z = pre * [pre > threshold]
Both decode   x_hat = z W_dec + b_dec.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


def log(msg):
    print(f"[eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class OurSAE:
    def __init__(self, sae_dir: Path, device):
        import torch
        st = torch.load(sae_dir / "sae.pt", map_location="cpu", weights_only=True)
        self.meta = json.loads((sae_dir / "meta.json").read_text())
        self.W_enc = st["W_enc.weight"].float().to(device)            # [F, d]
        self.b_enc = st["W_enc.bias"].float().to(device)              # [F]
        self.W_dec = st["W_dec.weight"].float().to(device)            # [d, F]
        self.b_dec = st["b_dec"].float().to(device)                   # [d]
        self.threshold = st["log_threshold"].float().exp().to(device)  # [F]
        self.n_features = self.W_enc.shape[0]
        self.dec_dirs = self.W_dec.t()                                 # [F, d]

    def encode(self, x):
        import torch
        pre = torch.nn.functional.linear(x - self.b_dec, self.W_enc, self.b_enc)
        return pre * (pre > self.threshold)

    def decode(self, z):
        import torch
        return torch.nn.functional.linear(z, self.W_dec, self.b_dec)


class ScopeSAE:
    def __init__(self, npz_path: Path, device):
        import numpy as np
        import torch
        d = np.load(npz_path)
        self.W_enc = torch.from_numpy(d["W_enc"]).float().to(device)      # [d, F]
        self.b_enc = torch.from_numpy(d["b_enc"]).float().to(device)
        self.W_dec = torch.from_numpy(d["W_dec"]).float().to(device)      # [F, d]
        self.b_dec = torch.from_numpy(d["b_dec"]).float().to(device)
        self.threshold = torch.from_numpy(d["threshold"]).float().to(device)
        self.n_features = self.W_enc.shape[1]
        self.dec_dirs = self.W_dec                                        # [F, d]

    def encode(self, x):
        pre = x @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec


def pick_reference(layer: int, target_l0: int, token):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    prefix = f"layer_{layer}/width_16k/"
    files = api.list_repo_files("google/gemma-scope-2b-pt-res")
    cands = sorted({f.split("/")[2] for f in files if f.startswith(prefix)})
    l0s = {c: int(c.split("_")[-1]) for c in cands}
    best = min(cands, key=lambda c: (abs(l0s[c] - target_l0), l0s[c]))
    return prefix + best, l0s[best], {c: l0s[c] for c in cands}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/gemma-2-2b")
    ap.add_argument("--layers", default="5,12,20")
    ap.add_argument("--sae-root", required=True, help="dir with layer_NN_s0/{sae.pt,meta.json}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout", action="append", required=True,
                    help="name=dir of token shards ([n_seqs,2048] BOS-prepended); repeatable")
    ap.add_argument("--max-shards", type=int, default=0, help="cap shards per source (smoke)")
    ap.add_argument("--target-l0", type=int, default=50)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overlap-only", action="store_true")
    ap.add_argument("--seqs-per-forward", type=int, default=4,
                    help="sequences per model forward (logits are [B,2048,256k])")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, os.environ.get("SAE_TRAINER_DIR", "trainer"))
    import sae_trainer_rolling as t

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    hf_token = t._resolve_hf_token()
    layers = [int(x) for x in args.layers.split(",")]

    # ---- SAEs -------------------------------------------------------------
    ours, refs, ref_info = {}, {}, {}
    for L in layers:
        ours[L] = OurSAE(Path(args.sae_root) / f"layer_{L:02d}_s{args.seed}", device)
        sub, ref_l0, all_l0 = pick_reference(L, args.target_l0, hf_token)
        p = hf_hub_download("google/gemma-scope-2b-pt-res", sub + "/params.npz", token=hf_token)
        refs[L] = ScopeSAE(Path(p), device)
        ref_info[L] = {"reference_subfolder": sub, "reference_stated_l0": ref_l0,
                       "reference_available_l0": all_l0,
                       "reference_n_features": refs[L].n_features,
                       "ours_n_features": ours[L].n_features,
                       "ours_train_steps": ours[L].meta.get("n_steps"),
                       "ours_train_tokens": ours[L].meta.get("total_tokens"),
                       "ours_train_final": ours[L].meta.get("final_metrics", {})}
        log(f"L{L}: ours {ours[L].n_features} feats ({ours[L].meta.get('n_steps')} steps, "
            f"{ours[L].meta.get('total_tokens')} tokens); reference {sub} "
            f"({refs[L].n_features} feats, stated L0 {ref_l0})")

    # ---- decoder overlap -------------------------------------------------------
    overlap = {}
    bins = torch.linspace(0, 1, 21)
    for L in layers:
        A = F.normalize(ours[L].dec_dirs, dim=1)      # [Fo, d]
        B = F.normalize(refs[L].dec_dirs, dim=1)      # [Fr, d]
        cos = A @ B.t()                               # [Fo, Fr]
        o2r = cos.max(dim=1).values                   # for each of ours, best reference
        r2o = cos.max(dim=0).values
        overlap[L] = {
            "ours_to_ref": {"frac_gt_0.7": float((o2r > 0.7).float().mean()),
                            "frac_gt_0.9": float((o2r > 0.9).float().mean()),
                            "mean": float(o2r.mean()), "median": float(o2r.median()),
                            "hist_counts": torch.histc(o2r.cpu(), bins=20, min=0, max=1).tolist()},
            "ref_to_ours": {"frac_gt_0.7": float((r2o > 0.7).float().mean()),
                            "frac_gt_0.9": float((r2o > 0.9).float().mean()),
                            "mean": float(r2o.mean()), "median": float(r2o.median()),
                            "hist_counts": torch.histc(r2o.cpu(), bins=20, min=0, max=1).tolist()},
            "hist_bin_edges": bins.tolist(),
        }
        torch.save({"ours_to_ref_max_cos": o2r.cpu(), "ref_to_ours_max_cos": r2o.cpu()},
                   out_dir / f"decoder_overlap_L{L:02d}.pt")
        log(f"L{L} overlap: ours->ref >0.7 {overlap[L]['ours_to_ref']['frac_gt_0.7']:.3f} "
            f">0.9 {overlap[L]['ours_to_ref']['frac_gt_0.9']:.3f} | ref->ours >0.7 "
            f"{overlap[L]['ref_to_ours']['frac_gt_0.7']:.3f} >0.9 "
            f"{overlap[L]['ref_to_ours']['frac_gt_0.9']:.3f}")
        del cos
    (out_dir / "decoder_overlap.json").write_text(json.dumps(
        {"layers": {str(L): overlap[L] for L in layers}, "reference": {str(L): ref_info[L] for L in layers}},
        indent=1))
    if args.overlap_only:
        return

    # ---- model ------------------------------------------------------------------
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, token=hf_token, dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    n_layers = int(model.config.num_hidden_layers)
    text_model, _, decoder_layers = t._find_text_model(model, n_layers)
    log(f"model on GPU in {time.perf_counter() - t0:.0f}s")

    # Splice hook: replace block-L output at positions >= 1 with `state['repl']`.
    state = {"repl": None}

    def _splice(_m, _i, out):
        if state["repl"] is None:
            return None
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, 1:, :] = state["repl"].to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def lm_loss(logits, ids):
        return F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                               ids[:, 1:].reshape(-1), reduction="sum")

    results = {}
    for spec in args.heldout:
        name, d = spec.split("=", 1)
        paths = t._shard_paths(Path(d))
        if args.max_shards:
            paths = paths[: args.max_shards]
        shard_ids = [int(p.stem.split("_")[1]) for p in paths]
        acc = {L: {k: {"sse": 0.0, "sst": 0.0, "sum_x": None, "n": 0, "l0_sum": 0.0,
                       "fires": None, "loss_sae": 0.0}
                   for k in ("ours", "ref")} for L in layers}
        loss_clean = 0.0
        loss_zero = {L: 0.0 for L in layers}
        n_pred = 0
        # First pass for the mean (needed for the total variance) is avoided by using
        # the standard sum-of-squares identity in fp64: sst = sum||x||^2 - n*||mean||^2.
        sum_sq = {L: 0.0 for L in layers}
        sum_x = {L: None for L in layers}
        t0 = time.perf_counter()
        for si, p in enumerate(paths):
            ids_all = torch.load(p, map_location="cpu", weights_only=True).to(device)
            for b0 in range(0, ids_all.shape[0], args.seqs_per_forward):
              ids = ids_all[b0: b0 + args.seqs_per_forward]
              with torch.no_grad():
                out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
                loss_clean += lm_loss(out.logits, ids).item()
                n_pred += ids.shape[0] * (ids.shape[1] - 1)
                hs = out.hidden_states
                for L in layers:
                    x = hs[L + 1][:, 1:, :].float()                # exclude BOS position
                    xf = x.reshape(-1, x.shape[-1])
                    n_tok_L = xf.shape[0]
                    sum_sq[L] += float(xf.double().pow(2).sum())
                    sx = xf.double().sum(dim=0)
                    sum_x[L] = sx if sum_x[L] is None else sum_x[L] + sx
                    for k, sae in (("ours", ours[L]), ("ref", refs[L])):
                        z = sae.encode(xf)
                        xh = sae.decode(z)
                        a = acc[L][k]
                        a["sse"] += float((xf - xh).double().pow(2).sum())
                        a["l0_sum"] += float((z != 0).sum())
                        fires = (z != 0).sum(dim=0)
                        a["fires"] = fires if a["fires"] is None else a["fires"] + fires
                        a["n"] += n_tok_L
                        # LM loss with the reconstruction spliced in
                        state["repl"] = xh.reshape(x.shape)
                        h = decoder_layers[L].register_forward_hook(_splice)
                        try:
                            o2 = model(input_ids=ids, use_cache=False)
                        finally:
                            h.remove()
                            state["repl"] = None
                        a["loss_sae"] += lm_loss(o2.logits, ids).item()
                        del o2, z, xh
                    # zero ablation
                    state["repl"] = torch.zeros_like(x)
                    h = decoder_layers[L].register_forward_hook(_splice)
                    try:
                        o3 = model(input_ids=ids, use_cache=False)
                    finally:
                        h.remove()
                        state["repl"] = None
                    loss_zero[L] += lm_loss(o3.logits, ids).item()
                    del o3
                del out, hs
            if (si + 1) % 10 == 0 or si + 1 == len(paths):
                log(f"{name}: {si + 1}/{len(paths)} shards ({time.perf_counter() - t0:.0f}s)")
        rows = {}
        for L in layers:
            n = acc[L]["ours"]["n"]
            mean = sum_x[L] / n
            sst = sum_sq[L] - n * float(mean.pow(2).sum())
            lc = loss_clean / n_pred
            lz = loss_zero[L] / n_pred
            rows[L] = {"n_tokens_eval": n, "n_pred_positions": n_pred, "shard_ids": shard_ids,
                       "loss_clean": lc, "loss_zero_ablation": lz}
            for k in ("ours", "ref"):
                a = acc[L][k]
                ls = a["loss_sae"] / n_pred
                rows[L][k] = {
                    "explained_variance": 1.0 - a["sse"] / sst,
                    "mse_per_token": a["sse"] / n,
                    "mean_l0": a["l0_sum"] / n,
                    "dead_frac": float((a["fires"] == 0).float().mean()),
                    "n_dead": int((a["fires"] == 0).sum()),
                    "loss_with_sae": ls,
                    "loss_recovered": (lz - ls) / (lz - lc) if lz != lc else float("nan"),
                    "ce_increase": ls - lc,
                }
            log(f"{name} L{L}: ours EV {rows[L]['ours']['explained_variance']:.4f} L0 "
                f"{rows[L]['ours']['mean_l0']:.1f} dead {rows[L]['ours']['dead_frac']:.4f} "
                f"rec {rows[L]['ours']['loss_recovered']:.4f} | ref EV "
                f"{rows[L]['ref']['explained_variance']:.4f} L0 {rows[L]['ref']['mean_l0']:.1f} "
                f"dead {rows[L]['ref']['dead_frac']:.4f} rec {rows[L]['ref']['loss_recovered']:.4f}"
                f" | clean {lc:.4f} zero {lz:.4f}")
        results[name] = {"dir": d, "n_shards": len(paths), "layers": {str(L): rows[L] for L in layers}}
        (out_dir / "calibration_raw.json").write_text(json.dumps(
            {"model_id": args.model_id, "bos_excluded": True, "layers": layers,
             "reference": {str(L): ref_info[L] for L in layers}, "sources": results}, indent=1))

    # ---- flat CSV ------------------------------------------------------------------
    import csv
    with open(out_dir / "calibration.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "source", "n_tokens", "shard_ids",
                    "ours_ev", "ref_ev", "ours_l0", "ref_l0", "ours_dead_frac", "ref_dead_frac",
                    "ours_loss_recovered", "ref_loss_recovered", "ours_ce_increase", "ref_ce_increase",
                    "loss_clean", "loss_zero_ablation",
                    "ours_n_features", "ref_n_features", "ref_subfolder", "ref_stated_l0",
                    "ours_train_steps", "ours_train_tokens",
                    "overlap_ours_to_ref_gt0.7", "overlap_ours_to_ref_gt0.9",
                    "overlap_ref_to_ours_gt0.7", "overlap_ref_to_ours_gt0.9", "bos_excluded"])
        for name, r in results.items():
            for L in layers:
                row = r["layers"][str(L)]
                ov = overlap[L]
                w.writerow([L, name, row["n_tokens_eval"],
                            f"{row['shard_ids'][0]}-{row['shard_ids'][-1]}",
                            row["ours"]["explained_variance"], row["ref"]["explained_variance"],
                            row["ours"]["mean_l0"], row["ref"]["mean_l0"],
                            row["ours"]["dead_frac"], row["ref"]["dead_frac"],
                            row["ours"]["loss_recovered"], row["ref"]["loss_recovered"],
                            row["ours"]["ce_increase"], row["ref"]["ce_increase"],
                            row["loss_clean"], row["loss_zero_ablation"],
                            ref_info[L]["ours_n_features"], ref_info[L]["reference_n_features"],
                            ref_info[L]["reference_subfolder"], ref_info[L]["reference_stated_l0"],
                            ref_info[L]["ours_train_steps"], ref_info[L]["ours_train_tokens"],
                            ov["ours_to_ref"]["frac_gt_0.7"], ov["ours_to_ref"]["frac_gt_0.9"],
                            ov["ref_to_ours"]["frac_gt_0.7"], ov["ref_to_ours"]["frac_gt_0.9"], True])
    log(f"wrote {out_dir / 'calibration.csv'}")


if __name__ == "__main__":
    main()
