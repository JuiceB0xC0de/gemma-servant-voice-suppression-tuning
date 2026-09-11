#!/usr/bin/env python3
"""Build atlas_summary.csv and the EV / mean-L0 per-layer figure from the meta.json
files of the assembled atlas.

    uv run python src/make_summary.py --metas results/metas/ --out results/atlas_summary.csv

--metas is a directory holding layer_NN_s0.meta.json (or the atlas root with
layer_NN_s0/meta.json). Pod-side only; needs silico-figures.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41]
KV_BOUNDARY = 24


def collect(metas_dir: Path, n_layers: int, seed: int = 0, wall: dict | None = None):
    """metas_dir: atlas root (layer_NN_s0/meta.json), a dir of layer_NN_s0.meta.json,
    or a JSON bundle file {"layer_NN": meta, ...}. wall: optional {"NN": minutes}."""
    rows = []
    bundle = json.loads(metas_dir.read_text()) if metas_dir.is_file() else None
    wall = wall or {}
    for L in range(n_layers):
        if bundle is not None:
            m = bundle.get(f"layer_{L:02d}")
            if m is None:
                rows.append({"layer": L, "present": False})
                continue
        else:
            cands = [metas_dir / f"layer_{L:02d}_s{seed}.meta.json",
                     metas_dir / f"layer_{L:02d}_s{seed}" / "meta.json"]
            p = next((c for c in cands if c.is_file()), None)
            if p is None:
                rows.append({"layer": L, "present": False})
                continue
            m = json.loads(p.read_text())
        fm = m.get("final_metrics") or {}
        curve = m.get("training_curve") or {}
        wall_meta = None
        for k in ("wall_minutes", "elapsed_min", "train_minutes"):
            if k in m:
                wall_meta = m[k]
        rows.append({
            "layer": L, "present": True,
            "steps": m.get("n_steps"), "tokens": m.get("total_tokens"),
            "ev": fm.get("ev", (curve.get("ev") or [None])[-1]),
            "best_ev": m.get("best_ev"),
            "mean_l0": fm.get("mean_l0", (curve.get("mean_l0") or [None])[-1]),
            "dead_pct": fm.get("dead_pct", (curve.get("dead_pct") or [None])[-1]),
            "early_stopped": m.get("early_stopped"),
            "wall_minutes": m.get("wall_minutes", wall_m if (wall_m := wall.get(f"{L:02d}", wall.get(str(L)))) is not None else wall_meta),
            "layer_type": "global" if L in GLOBAL_LAYERS else "sliding",
            "kv_shared": L >= KV_BOUNDARY,
            "job_id": m.get("job_id"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metas", required=True)
    ap.add_argument("--n-layers", type=int, default=42)
    ap.add_argument("--out", default="results/atlas_summary.csv")
    ap.add_argument("--figure-name", default="atlas_ev_l0_by_layer")
    ap.add_argument("--ev-floor", type=float, default=0.85)
    ap.add_argument("--wall", default=None, help="JSON {layer: wall_minutes} parsed from the chain logs")
    ap.add_argument("--no-figure", action="store_true", help="CSV only (job image lacks silico-figures)")
    args = ap.parse_args()

    wall = json.loads(Path(args.wall).read_text()) if args.wall else None
    rows = collect(Path(args.metas), args.n_layers, wall=wall)
    cols = ["layer", "layer_type", "kv_shared", "steps", "tokens", "ev", "best_ev", "mean_l0",
            "dead_pct", "early_stopped", "wall_minutes", "job_id"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if r.get("present"):
                w.writerow(r)
    present = [r for r in rows if r.get("present")]
    print(f"wrote {args.out} with {len(present)} layers")
    if args.no_figure:
        return

    # ---- figure: two stacked panels, EV and mean L0 by layer ----------------------
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from silico_figures import apply_theme, add_reference_line, save_figure_bundle

    layers = [r["layer"] for r in present]
    ev = [r["ev"] for r in present]
    l0 = [r["mean_l0"] for r in present]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Explained variance", "Mean L0"))
    fig.add_trace(go.Scatter(x=layers, y=ev, mode="lines+markers", name="EV",
                             hovertemplate="layer %{x}<br>EV %{y:.4f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=layers, y=l0, mode="lines+markers", name="mean L0",
                             hovertemplate="layer %{x}<br>L0 %{y:.1f}<extra></extra>"), row=2, col=1)
    gl = [L for L in GLOBAL_LAYERS if L in layers]
    fig.add_trace(go.Scatter(x=gl, y=[ev[layers.index(L)] for L in gl], mode="markers",
                             name="global attention", marker=dict(symbol="diamond", size=11, color="#2a6f4e"),
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=gl, y=[l0[layers.index(L)] for L in gl], mode="markers",
                             name="global attention", marker=dict(symbol="diamond", size=11, color="#2a6f4e"),
                             showlegend=False, hoverinfo="skip"), row=2, col=1)
    add_reference_line(fig, y=args.ev_floor, label=f"EV floor {args.ev_floor:g}", row=1, col=1)
    add_reference_line(fig, y=50, label="target L0 50", row=2, col=1)
    add_reference_line(fig, x=KV_BOUNDARY - 0.5, label="shared-KV boundary (layer 24)", row="all", col=1)
    fig.update_yaxes(title_text="Explained variance", range=[0.7, 1.0], row=1, col=1)
    fig.update_yaxes(title_text="Mean L0 (features/token)", row=2, col=1)
    fig.update_xaxes(title_text="Layer", row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=1, col=1)
    apply_theme(fig, height=620)
    save_figure_bundle(
        fig, args.figure_name,
        data={"layer": layers, "ev": ev, "mean_l0": l0,
              "global_attention_layers": gl, "kv_boundary": KV_BOUNDARY, "ev_floor": args.ev_floor},
        alt="Final explained variance (top) and mean L0 (bottom) of each layer's SAE for "
            "Gemma 4 E4B, with global-attention layers marked as diamonds and the shared-KV "
            "boundary at layer 24 drawn as a vertical line.")
    print(f"saved figures/{args.figure_name}/")


if __name__ == "__main__":
    main()
