"""Post-hoc uncertainty and length-confound analysis of the stage-1 layer profile.

CPU-only. Reads acts_pairs.npz + directions.npz written by run_model.py stage 1 and the pairs split file,
and writes stage1_boot.json with, per pooling ("all", "first"):
  * per-layer test Cohen's d with a paired-bootstrap 95% interval (resampling test pairs),
  * for every interior local maximum above the shuffled null: the bootstrap interval of crest-minus-trough
    on each side and the fraction of bootstrap replicates in which the layer is an interior maximum,
  * the Q1 quantities (crest near 4, crest near 13/14, trough between) with bootstrap intervals,
  * within-side Spearman correlation of the per-pair projection with the reply's token count,
    and d after regressing the projection on log(token count) within side ("length-partialled d").

Usage: python src/post_stage1.py --acts <acts_pairs.npz> --dirs <directions.npz> --stage1 <stage1.json> --out <stage1_boot.json>
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "results" / "data"
SEED = 42
N_BOOT = 2000
PRED_CRESTS_E2B = [(3, 5), (12, 15)]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / (s + 1e-12))


def d_vec(PB, PG):
    """PB, PG: [n, L] projections -> d per layer (vectorised over layers)."""
    na, nb = PB.shape[0], PG.shape[0]
    s = np.sqrt(((na - 1) * PB.var(0, ddof=1) + (nb - 1) * PG.var(0, ddof=1)) / (na + nb - 2))
    return (PB.mean(0) - PG.mean(0)) / (s + 1e-12)


def interior_maxima(x, above=None):
    return [i for i in range(1, len(x) - 1)
            if x[i] >= x[i - 1] and x[i] > x[i + 1] and (above is None or x[i] > above[i])]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default=str(DATA / "pairs.jsonl"))
    args = ap.parse_args()
    s1 = json.load(open(args.stage1))
    A = np.load(args.acts)
    D = np.load(args.dirs)
    splits = np.array([json.loads(l)["split"] for l in open(args.pairs) if l.strip()])
    n = A["bella_all"].shape[0]
    splits = splits[:n]
    test = np.where(splits == "test")[0]
    rng = np.random.default_rng(SEED)
    L = A["bella_all"].shape[1]
    ntb, ntg = A["n_tok_bella"][test].astype(float), A["n_tok_gemma"][test].astype(float)
    res = {"model": s1["model"], "n_layers": L, "n_test_pairs": int(len(test)), "n_boot": N_BOOT, "pooling": {}}
    boot_idx = rng.integers(0, len(test), (N_BOOT, len(test)))
    for pool in ("all", "first"):
        V = D[pool].astype(np.float32)  # [L, d] unit directions (Bella - Gemma), fit on train
        B = A[f"bella_{pool}"][test].astype(np.float32)  # [n, L, d]
        G = A[f"gemma_{pool}"][test].astype(np.float32)
        PB = np.einsum("nld,ld->nl", B, V)
        PG = np.einsum("nld,ld->nl", G, V)
        del B, G
        d0 = d_vec(PB, PG)
        null = np.array(s1["pooling"][pool]["null_shuffled_d_p95"])
        # paired bootstrap of the whole profile
        boot = np.stack([d_vec(PB[ix], PG[ix]) for ix in boot_idx])  # [N_BOOT, L]
        lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
        se = boot.std(0, ddof=1)
        maxima = interior_maxima(d0, above=null)
        # fraction of replicates in which each layer is an interior local maximum (no null condition)
        is_max = np.zeros(L)
        for l in range(1, L - 1):
            is_max[l] = np.mean((boot[:, l] >= boot[:, l - 1]) & (boot[:, l] > boot[:, l + 1]))
        crests = []
        for k, l in enumerate(maxima):
            left_bound = maxima[k - 1] if k > 0 else 0
            right_bound = maxima[k + 1] if k + 1 < len(maxima) else L - 1
            tl = int(left_bound + np.argmin(d0[left_bound:l])) if l > left_bound else l
            tr = int(l + 1 + np.argmin(d0[l + 1:right_bound + 1])) if right_bound > l else l
            dl, dr = boot[:, l] - boot[:, tl], boot[:, l] - boot[:, tr]
            entry = {"layer": int(l), "d": float(d0[l]), "trough_left": tl, "trough_right": tr,
                     "drop_left": float(d0[l] - d0[tl]), "drop_right": float(d0[l] - d0[tr]),
                     "drop_left_ci": [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))],
                     "drop_right_ci": [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))],
                     "p_local_max": float(is_max[l])}
            entry["survives"] = bool((tl == l or entry["drop_left_ci"][0] > 0) and (tr == l or entry["drop_right_ci"][0] > 0))
            crests.append(entry)
        # Q1 quantities (E2B predictions; computed for any model for reference)
        c1 = [l for l in maxima if PRED_CRESTS_E2B[0][0] <= l <= PRED_CRESTS_E2B[0][1]]
        c2 = [l for l in maxima if PRED_CRESTS_E2B[1][0] <= l <= PRED_CRESTS_E2B[1][1]]
        q1 = {"crest_near_4": c1, "crest_near_13_14": c2}
        if c1 and c2:
            a, b = max(c1, key=lambda l: d0[l]), max(c2, key=lambda l: d0[l])
            t = int(a + 1 + np.argmin(d0[a + 1:b]))
            da, db = boot[:, a] - boot[:, t], boot[:, b] - boot[:, t]
            q1.update({"crest_a": int(a), "crest_b": int(b), "trough_layer": t,
                       "drop_a": float(d0[a] - d0[t]), "drop_a_ci": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
                       "drop_b": float(d0[b] - d0[t]), "drop_b_ci": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
                       "p_both_drops_ge_0.15": float(np.mean((da >= 0.15) & (db >= 0.15))),
                       "p_both_drops_gt_0": float(np.mean((da > 0) & (db > 0)))})
        # length confound
        rho_b = [spearman(PB[:, l], ntb) for l in range(L)]
        rho_g = [spearman(PG[:, l], ntg) for l in range(L)]
        # partial out log token count within side (separate fits per side), then d of residuals + side mean gap kept
        xb, xg = np.log(ntb + 1), np.log(ntg + 1)
        d_part = []
        for l in range(L):
            rb = PB[:, l] - np.polyval(np.polyfit(xb, PB[:, l], 1), xb) + PB[:, l].mean()
            rg = PG[:, l] - np.polyval(np.polyfit(xg, PG[:, l], 1), xg) + PG[:, l].mean()
            d_part.append(cohens_d(rb, rg))
        # single pooled regression across sides: how much of the side gap is explained by length alone?
        x_all = np.concatenate([xb, xg])
        d_len_only = []
        for l in range(L):
            y = np.concatenate([PB[:, l], PG[:, l]])
            fit = np.polyval(np.polyfit(x_all, y, 1), x_all)
            d_len_only.append(cohens_d(fit[: len(xb)], fit[len(xb):]))  # d that length alone would produce
        # d on pairs whose Gemma reply was not truncated (shorter replies)
        res["pooling"][pool] = {
            "test_d": d0.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist(), "se": se.tolist(),
            "p_interior_max": is_max.tolist(), "interior_maxima_above_null": [int(l) for l in maxima],
            "crests": crests, "n_crests_surviving": int(sum(c["survives"] for c in crests)),
            "surviving_layers": [c["layer"] for c in crests if c["survives"]],
            "q1": q1,
            "spearman_proj_ntok_bella": rho_b, "spearman_proj_ntok_gemma": rho_g,
            "d_length_partialled": d_part, "d_from_length_fit_only": d_len_only,
            "test_mean_proj_bella": PB.mean(0).tolist(), "test_mean_proj_gemma": PG.mean(0).tolist(),
        }
    res["n_tok_test"] = {"bella_mean": float(ntb.mean()), "gemma_mean": float(ntg.mean()),
                         "bella_median": float(np.median(ntb)), "gemma_median": float(np.median(ntg)),
                         "spearman_bella_gemma": spearman(ntb, ntg)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    P = res["pooling"][s1["pooling_choice"]["chosen"]]
    print(json.dumps({"model": res["model"], "pooling": s1["pooling_choice"]["chosen"], "surviving": P["surviving_layers"],
                      "maxima": P["interior_maxima_above_null"], "q1": P["q1"], "median_se": float(np.median(P["se"]))}, indent=1))


if __name__ == "__main__":
    main()
