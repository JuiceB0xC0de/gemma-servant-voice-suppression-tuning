#!/usr/bin/env python3
"""Acceptance check for the Gemma 4 E4B JumpReLU SAE atlas.

    python3 verify_atlas.py --root <sae_dir> --hf-repo juiceb0xc0de/gemma-4-e4b-it-SAE \
        --n-layers 42 --d-in 2560 --n-features 81920 --k 50 \
        --ev-floor 0.85 --l0-window 40,60 --dead-max 0.01

Passes (exit 0) only if every check below holds; otherwise prints the failing layers by
name and exits 1. Layers that miss the EV floor or the L0 window are reported as flagged,
never dropped.

Checks
  1. layer_NN_s0/{sae.pt,meta.json} exists locally for NN in 0..n_layers-1.
  2. The same 42 folders exist in the HF repo (dataset), unless --skip-hf.
  3. meta.json: d_in, n_features, k, model_id match; trainer_commit present.
  4. final EV >= ev_floor, mean L0 within l0_window, dead fraction <= dead_max.
  5. One randomly chosen sae.pt per job range (--ranges) loads with only the standard
     library + torch and encodes a random [64, d_in] batch (scaled to the layer's
     recorded activation RMS); its mean L0 must be within --l0-tol of the recorded L0.

The report is written as JSON to --out.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def load_meta(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def final_metric(meta: dict, key: str):
    fm = meta.get("final_metrics") or {}
    v = fm.get(key)
    if v is None:
        curve = (meta.get("training_curve") or {}).get(key) or []
        v = curve[-1] if curve else None
    return v


def encode_check(layer_dir: Path, d_in: int, n_features: int, meta: dict, seed: int = 0):
    import torch
    torch.manual_seed(seed)
    state = torch.load(layer_dir / "sae.pt", map_location="cpu", weights_only=True)
    W_enc = state["W_enc.weight"].float()
    b_enc = state.get("W_enc.bias", torch.zeros(n_features)).float()
    b_dec = state.get("b_dec", torch.zeros(d_in)).float()
    thr = state["log_threshold"].float().exp()
    assert tuple(W_enc.shape) == (n_features, d_in), f"W_enc shape {tuple(W_enc.shape)}"
    assert tuple(state["W_dec.weight"].shape) == (d_in, n_features), \
        f"W_dec shape {tuple(state['W_dec.weight'].shape)}"
    rms = float((meta.get("preflight") or {}).get("activation_norm_probe") or 1.0)
    x = torch.randn(64, d_in) * rms
    pre = torch.nn.functional.linear(x - b_dec, W_enc, b_enc)
    l0 = (pre > thr).float().sum(-1).mean().item()
    return {"encode_mean_l0_random_batch": l0, "input_rms": rms,
            "nan_in_weights": bool(torch.isnan(W_enc).any() or torch.isnan(state["W_dec.weight"]).any()),
            "threshold_mean": thr.mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--hf-repo", default="juiceb0xc0de/gemma-4-e4b-it-SAE")
    ap.add_argument("--hf-repo-type", default="dataset")
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--n-layers", type=int, default=42)
    ap.add_argument("--d-in", type=int, default=2560)
    ap.add_argument("--n-features", type=int, default=81920)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--model-id", default="google/gemma-4-E4B-it")
    ap.add_argument("--ev-floor", type=float, default=0.85)
    ap.add_argument("--l0-window", default="40,60")
    ap.add_argument("--dead-max", type=float, default=0.01, help="fraction, e.g. 0.01 = 1%%")
    ap.add_argument("--ranges", default="0-13,14-27,28-41")
    ap.add_argument("--l0-tol", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-encode", action="store_true")
    ap.add_argument("--out", default="results/verify_atlas.json")
    args = ap.parse_args()

    root = Path(args.root)
    lo, hi = (float(x) for x in args.l0_window.split(","))
    failures: list[str] = []
    flagged: list[dict] = []
    rows = []

    # 1 + 3 + 4
    for L in range(args.n_layers):
        name = f"layer_{L:02d}_s{args.seed}"
        d = root / name
        row = {"layer": L, "dir": name, "present": d.is_dir(),
               "sae_pt": (d / "sae.pt").is_file(), "meta": (d / "meta.json").is_file()}
        if not (row["sae_pt"] and row["meta"]):
            failures.append(f"{name}: missing {'sae.pt' if not row['sae_pt'] else ''} "
                            f"{'meta.json' if not row['meta'] else ''}".strip())
            rows.append(row)
            continue
        m = load_meta(d / "meta.json")
        for key, want in (("d_in", args.d_in), ("n_features", args.n_features),
                          ("k", args.k), ("model_id", args.model_id)):
            if m.get(key) != want:
                failures.append(f"{name}: meta {key}={m.get(key)!r} != {want!r}")
        if not m.get("trainer_commit"):
            failures.append(f"{name}: meta has no trainer_commit")
        ev = final_metric(m, "ev"); l0 = final_metric(m, "mean_l0"); dead = final_metric(m, "dead_pct")
        row.update({"ev": ev, "mean_l0": l0, "dead_pct": dead, "n_steps": m.get("n_steps"),
                    "total_tokens": m.get("total_tokens"), "early_stopped": m.get("early_stopped"),
                    "best_ev": m.get("best_ev"), "trainer_commit": m.get("trainer_commit"),
                    "image_ref": m.get("image_ref"), "job_id": m.get("job_id")})
        problems = []
        if ev is None or ev < args.ev_floor:
            problems.append(f"EV {ev} < {args.ev_floor}")
        if l0 is None or not (lo <= l0 <= hi):
            problems.append(f"mean L0 {l0} outside [{lo:g}, {hi:g}]")
        # dead_pct in meta is a percentage (e.g. 0.18 == 0.18 %)
        if dead is None or dead / 100.0 > args.dead_max:
            problems.append(f"dead {dead}% > {args.dead_max * 100:g}%")
        if problems:
            flagged.append({"layer": name, "problems": problems})
            failures.append(f"{name}: " + "; ".join(problems))
        rows.append(row)

    # 2
    hf_missing = None
    if not args.skip_hf:
        from huggingface_hub import HfApi
        api = HfApi()
        present = set()
        for item in api.list_repo_tree(args.hf_repo, repo_type=args.hf_repo_type, recursive=True):
            parts = item.path.split("/")
            if len(parts) == 2 and parts[1] in ("sae.pt", "meta.json"):
                present.add(item.path)
        hf_missing = [f"layer_{L:02d}_s{args.seed}/{fn}" for L in range(args.n_layers)
                      for fn in ("sae.pt", "meta.json")
                      if f"layer_{L:02d}_s{args.seed}/{fn}" not in present]
        for p in hf_missing:
            failures.append(f"HF {args.hf_repo}: missing {p}")

    # 5
    encode_results = []
    if not args.skip_encode:
        rng = random.Random(args.seed)
        for rng_spec in args.ranges.split(","):
            a, b = (int(x) for x in rng_spec.split("-"))
            candidates = [L for L in range(a, b + 1)
                          if (root / f"layer_{L:02d}_s{args.seed}" / "sae.pt").is_file()]
            if not candidates:
                failures.append(f"range {rng_spec}: no sae.pt to spot-check")
                continue
            L = rng.choice(candidates)
            name = f"layer_{L:02d}_s{args.seed}"
            meta = load_meta(root / name / "meta.json")
            try:
                r = encode_check(root / name, args.d_in, args.n_features, meta, seed=args.seed)
            except Exception as e:  # noqa: BLE001
                failures.append(f"{name}: sae.pt failed to load/encode: {e}")
                encode_results.append({"range": rng_spec, "layer": name, "error": str(e)})
                continue
            rec = final_metric(meta, "mean_l0")
            r.update({"range": rng_spec, "layer": name, "recorded_mean_l0": rec,
                      "within_tol": rec is not None and abs(r["encode_mean_l0_random_batch"] - rec) <= args.l0_tol})
            if r["nan_in_weights"]:
                failures.append(f"{name}: NaN in weights")
            if not r["within_tol"]:
                failures.append(f"{name}: random-batch L0 {r['encode_mean_l0_random_batch']:.1f} "
                                f"vs recorded {rec} (tol ±{args.l0_tol:g})")
            encode_results.append(r)

    report = {"root": str(root), "hf_repo": None if args.skip_hf else args.hf_repo,
              "criteria": {"n_layers": args.n_layers, "d_in": args.d_in, "n_features": args.n_features,
                           "k": args.k, "ev_floor": args.ev_floor, "l0_window": [lo, hi],
                           "dead_max_fraction": args.dead_max, "l0_tol": args.l0_tol},
              "passed": not failures, "failures": failures, "flagged_layers": flagged,
              "hf_missing": hf_missing, "encode_checks": encode_results, "layers": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    n_ok = sum(1 for r in rows if r.get("sae_pt") and r.get("meta"))
    print(f"[verify] {n_ok}/{args.n_layers} layers present locally; "
          f"{len(flagged)} flagged; HF missing: {None if hf_missing is None else len(hf_missing)}")
    for f in failures:
        print(f"  FAIL {f}")
    for r in encode_results:
        print(f"  encode {r['layer']}: {r}")
    print(f"[verify] {'PASS' if not failures else 'FAIL'} -> {args.out}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
