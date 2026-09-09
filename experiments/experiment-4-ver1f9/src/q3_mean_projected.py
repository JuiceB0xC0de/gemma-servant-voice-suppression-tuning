"""Q3 shared-component check (CPU only).

Recomputes cos(SFT shift, Bella direction) per layer after removing components that two
mean-difference vectors could share for reasons unrelated to the Bella voice:
  (a) the per-layer mean residual of the Gemma replies (unit vector m_l), and
  (b) the K coordinates with the largest |mean residual| (massive-activation dimensions).
Also reports cos(shift, m_l) and cos(direction, m_l) directly.

Inputs (iteration-1 job outputs, artifact store refs under
artifact://juiceb0xc0de-15787e/experiments/exp_01m21z6dx8e5r9zt6n0vver1f9/run/):
  sft_shift.npz            shift[42,2560] = mean_prompt(ft_mean - base_mean) per layer
  acts_means.npz           gemma_mean/bella_mean[42,2560] from the 600-pair harvest
  inputs/e4b_directions.npz  'all'[42,2560] unit Bella-minus-Gemma direction used everywhere

Usage: apy src/q3_mean_projected.py --inputs /tmp --out results/analysis/q3_mean_projected.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

K_DROP = 8


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def perp(a, u):
    return a - (a @ u) * u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/tmp")
    ap.add_argument("--out", default="results/analysis/q3_mean_projected.csv")
    a = ap.parse_args()
    inp = Path(a.inputs)
    V = np.load(inp / "e4b_directions.npz")["all"].astype(np.float64)
    S = np.load(inp / "sft_shift.npz")["shift"].astype(np.float64)
    am = np.load(inp / "acts_means.npz")
    G = am["gemma_mean"].astype(np.float64)
    rng = np.random.default_rng(42)
    rows = []
    for l in range(42):
        s, v, g = S[l], V[l], G[l]
        m = g / np.linalg.norm(g)
        keep = np.ones_like(g, dtype=bool)
        keep[np.argsort(-np.abs(g))[:K_DROP]] = False
        rand = np.abs([cos(perp(s, m), perp(r, m)) for r in rng.standard_normal((200, len(s)))])
        rows.append({
            "layer": l,
            "cos": cos(s, v),
            "cos_shift_vs_mean": cos(s, m),
            "cos_dir_vs_mean": cos(v, m),
            "cos_after_mean_projected_out": cos(perp(s, m), perp(v, m)),
            "shift_frac_kept_after_projection": float(np.linalg.norm(perp(s, m)) / np.linalg.norm(s)),
            f"cos_drop_top{K_DROP}_mean_coords": cos(s[keep], v[keep]),
            "random_dir_abs_cos_p95_after_projection": float(np.percentile(rand, 95)),
        })
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {
        "k_drop": K_DROP,
        "max_abs_delta_cos_mean_projection_L0_13": max(abs(r["cos_after_mean_projected_out"] - r["cos"]) for r in rows[:14]),
        "max_abs_delta_cos_mean_projection_all": max(abs(r["cos_after_mean_projected_out"] - r["cos"]) for r in rows),
        "max_abs_delta_cos_drop_coords_L0_13": max(abs(r[f"cos_drop_top{K_DROP}_mean_coords"] - r["cos"]) for r in rows[:14]),
        "cos_after_projection_L4": rows[4]["cos_after_mean_projected_out"],
        "cos_after_projection_L10": rows[10]["cos_after_mean_projected_out"],
        "min_cos_after_projection": min(r["cos_after_mean_projected_out"] for r in rows),
        "max_abs_cos_shift_vs_mean_L0_13": max(abs(r["cos_shift_vs_mean"]) for r in rows[:14]),
        "max_random_p95_after_projection": max(r["random_dir_abs_cos_p95_after_projection"] for r in rows),
    }
    json.dump(summ, open(out.with_suffix(".json"), "w"), indent=1)
    for r in rows:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
