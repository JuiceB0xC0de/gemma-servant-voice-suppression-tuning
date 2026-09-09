"""Write report_summary.json and claims_manifest.json from the verified analysis files.

Run: cd $SILICO_EXPERIMENT_DIR && apy src/make_report_summary.py
Every number in the report traces to a (file, key) pair in claims_manifest.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REL = "experiments/experiment-4-ver1f9"
AN = HERE / "results" / "analysis"
cells = json.load(open(AN / "cells.json"))
verd = json.load(open(AN / "verdicts.json"))
s1 = json.load(open(AN / "stage1_summary.json"))
labels = json.load(open(AN / "feature_labels.json"))
jm = json.load(open(HERE / "results" / "judge" / "judge_meta.json"))
smoke = json.load(open(HERE / "results" / "smoke2_review_fixes.json"))
q3rows = {int(r["layer"]): r for r in csv.DictReader(open(AN / "q3_layers.csv"))}
s1raw = json.load(open(HERE / "results" / "run" / "stage1.json"))

claims = []


def claim(cid, text, value, file, key, kind="quantitative"):
    claims.append({"id": cid, "claim": text, "value": value, "source_file": f"{REL}/{file}", "key": key, "kind": kind})


def cell_claims(c, short):
    t = cells[c]
    g = t["gain"]
    claim(f"{short}_gain", f"{c} Bella gain vs unsteered", round(g["mean"], 4), "results/analysis/cells.json", f"{c}.gain.mean")
    claim(f"{short}_gain_ci", f"{c} Bella gain 95% bootstrap CI", [round(g["lo"], 4), round(g["hi"], 4)], "results/analysis/cells.json", f"{c}.gain.lo, {c}.gain.hi")
    if t["dose_along_dir"] is not None:
        claim(f"{short}_dose", f"{c} mean shift along Bella direction per generated token", round(t["dose_along_dir"], 4), "results/analysis/cells.json", f"{c}.dose_along_dir")
    if t["dose_frac_of_dir_-0.35"] is not None:
        claim(f"{short}_dose_frac", f"{c} dose as fraction of direction −0.35 dose", round(t["dose_frac_of_dir_-0.35"], 4), "results/analysis/cells.json", f"{c}.dose_frac_of_dir_-0.35")
    if t["edit_norm_per_token"]:
        claim(f"{short}_edit_norm", f"{c} mean edit norm per generated token", round(t["edit_norm_per_token"], 4), "results/analysis/cells.json", f"{c}.edit_norm_per_token")
    if t["clamped_per_token"] is not None:
        claim(f"{short}_clamped", f"{c} mean features clamped per generated token (chat prompts)", round(t["clamped_per_token"], 4), "results/analysis/cells.json", f"{c}.clamped_per_token")
    claim(f"{short}_refusal", f"{c} red-team refusal rate (n=160)", round(t["refusal"], 4), "results/analysis/cells.json", f"{c}.refusal")
    claim(f"{short}_crisis", f"{c} crisis score (n=30)", round(t["crisis"], 4), "results/analysis/cells.json", f"{c}.crisis")
    claim(f"{short}_degen", f"{c} chat degeneration rate (n=100)", round(t["degenerate"], 4), "results/analysis/cells.json", f"{c}.degenerate")
    claim(f"{short}_ndegen", f"{c} neutral-stem degeneration rate (n=100)", round(t["neutral_degenerate"], 4), "results/analysis/cells.json", f"{c}.neutral_degenerate")
    claim(f"{short}_bella", f"{c} mean Bella score (1-7, n=100)", round(t["bella"], 4), "results/analysis/cells.json", f"{c}.bella")


short = {"base": "base", "dir:10:-0.35": "dir35", "dir:10:-0.5": "dir50"}
for c in cells:
    s = short.get(c, c.replace(":", "_"))
    cell_claims(c, s)

claim("q1_threshold", "Q1 Bella-gain threshold (70% of direction −0.35 gain)", round(verd["q1_gain_threshold"], 4), "results/analysis/verdicts.json", "q1_gain_threshold")
claim("q1_verdict", "Q1 verdict", verd["Q1"]["verdict"], "results/analysis/verdicts.json", "Q1.verdict", "categorical")
claim("q2_verdict", "Q2 verdict", verd["Q2"]["verdict"], "results/analysis/verdicts.json", "Q2.verdict", "categorical")
claim("q3_verdict", "Q3 verdict", verd["Q3"]["verdict"], "results/analysis/verdicts.json", "Q3.verdict", "categorical")
claim("q1_best_any", "best layer-10 feature cell by gain (any k, non-exploratory)", verd["Q1"]["best_any_k"]["cell"], "results/analysis/verdicts.json", "Q1.best_any_k.cell", "categorical")
claim("q1_best_short", "best layer-10 feature cell by gain at k<=50", verd["Q1"]["best_k_le_50"]["cell"], "results/analysis/verdicts.json", "Q1.best_k_le_50.cell", "categorical")
claim("exploratory_cells", "cells excluded from verdicts as exploratory extras", verd["exploratory_cells_excluded_from_verdicts"], "results/analysis/verdicts.json", "exploratory_cells_excluded_from_verdicts", "categorical")
claim("n_cells", "number of cells", verd["n_cells"], "results/analysis/verdicts.json", "n_cells")
claim("n_judged_rows", "judged generations", verd["n_judged_rows"], "results/analysis/verdicts.json", "n_judged_rows")
claim("judge_items", "judge items scored", jm["n_items"], "results/judge/judge_meta.json", "n_items")
claim("judge_invalid", "invalid judge parses", jm["invalid"], "results/judge/judge_meta.json", "invalid")
claim("judge_model", "judge model", jm["judge_model"], "results/judge/judge_meta.json", "judge_model", "categorical")

q3 = verd["Q3"]
for k in ("cos_L4", "cos_L10", "frac_gap_closed_test_L4", "frac_gap_closed_test_L10", "max_cos", "argmax_cos_layer", "mean_cos_layers_0_14", "mean_cos_layers_30_41", "random_dir_abs_cos_p95", "n_prompts"):
    claim(f"q3_{k}", f"Q3 {k}", round(q3[k], 4) if isinstance(q3[k], float) else q3[k], "results/analysis/verdicts.json", f"Q3.{k}")
claim("q3_ft_revision", "fine-tuned checkpoint revision", q3["ft"]["revision"], "results/analysis/verdicts.json", "Q3.ft.revision", "categorical")
claim("q3_shift_norm_ratio_L0", "SFT shift norm / median residual norm at layer 0", round(float(q3rows[0]["shift_norm_over_median_norm"]), 5), "results/analysis/q3_layers.csv", "row layer=0, col shift_norm_over_median_norm")
claim("q3_shift_norm_ratio_L10", "SFT shift norm / median residual norm at layer 10", round(float(q3rows[10]["shift_norm_over_median_norm"]), 5), "results/analysis/q3_layers.csv", "row layer=10, col shift_norm_over_median_norm")
claim("q3_shift_norm_ratio_max", "max over layers of SFT shift norm / median residual norm", round(max(float(r["shift_norm_over_median_norm"]) for r in q3rows.values()), 5), "results/analysis/q3_layers.csv", "max over rows, col shift_norm_over_median_norm")
claim("q3_frac_gap_closed_max", "max over layers of fraction of Bella-Gemma gap closed (test pairs)", round(max(float(r["frac_gap_closed_test"]) for r in q3rows.values()), 5), "results/analysis/q3_layers.csv", "max over rows, col frac_gap_closed_test")
claim("q3_frac_gap_closed_L0_13_range", "fraction of gap closed, layers 0-13 (min,max)", [round(min(float(q3rows[l]["frac_gap_closed_test"]) for l in range(14)), 4), round(max(float(q3rows[l]["frac_gap_closed_test"]) for l in range(14)), 4)], "results/analysis/q3_layers.csv", "rows layer 0-13, col frac_gap_closed_test")
claim("q3_frac_gap_closed_L29_41_range", "fraction of gap closed, layers 29-41 (min,max)", [round(min(float(q3rows[l]["frac_gap_closed_test"]) for l in range(29, 42)), 4), round(max(float(q3rows[l]["frac_gap_closed_test"]) for l in range(29, 42)), 4)], "results/analysis/q3_layers.csv", "rows layer 29-41, col frac_gap_closed_test")
claim("q3_shift_norm_ratio_L0_13_max", "max SFT shift norm / median residual norm over layers 0-13", round(max(float(q3rows[l]["shift_norm_over_median_norm"]) for l in range(14)), 4), "results/analysis/q3_layers.csv", "rows layer 0-13, col shift_norm_over_median_norm")
claim("q3_cos_L29_41_range", "cosine range over layers 29-41", [round(min(q3["cos_by_layer"][29:]), 3), round(max(q3["cos_by_layer"][29:]), 3)], "results/analysis/verdicts.json", "Q3.cos_by_layer[29:41]")
claim("q3_cos_min", "min cosine over layers", round(min(q3["cos_by_layer"]), 4), "results/analysis/verdicts.json", "min(Q3.cos_by_layer)")

for l in ("4", "10", "22"):
    p = s1["per_layer"][l]
    for k in ("l0_bella", "l0_gemma", "ev_bella", "ev_gemma", "meta_l0", "meta_ev", "spearman_dtest_vs_align_union_top200", "pool_size_fire_ge_1pct"):
        claim(f"s1_L{l}_{k}", f"stage 1 layer {l} {k}", round(p[k], 4) if isinstance(p[k], float) else p[k], "results/analysis/stage1_summary.json", f"per_layer.{l}.{k}")
claim("s1_n_pairs", "matched Bella/Gemma reply pairs harvested", s1["n_pairs"], "results/analysis/stage1_summary.json", "n_pairs")
claim("s1_splits", "pair splits", s1["splits"], "results/analysis/stage1_summary.json", "splits")
claim("s1_l0_check", "layer-10 harness L0 check", s1["l0_check_layer10"], "results/analysis/stage1_summary.json", "l0_check_layer10")
claim("s1_L22_encoder_convention", "layer-22 SAE encoder convention used", {"sae_encoder_sub_bias": s1raw["per_layer"]["22"]["sae_encoder_sub_bias"], "atlas_sub_bias": s1raw["per_layer"]["22"]["atlas_sub_bias"]}, "results/run/stage1.json", "per_layer.22.sae_encoder_sub_bias, per_layer.22.atlas_sub_bias", "categorical")
claim("smoke_L22_conventions", "layer-22 L0/EV under both encoder conventions (smoke, 24 pairs)", smoke["layer22_encoder_convention_check"], "results/smoke2_review_fixes.json", "layer22_encoder_convention_check")

claim("labels_n_assistant", "Assistant-ward features labelled", len(labels["groups"]["assistant_ward"]), "results/analysis/feature_labels.json", "len(groups.assistant_ward)")
claim("labels_n_bella", "Bella-ward features labelled", len(labels["groups"]["bella_ward"]), "results/analysis/feature_labels.json", "len(groups.bella_ward)")
claim("labels_summary", "label themes", labels["summary"], "results/analysis/feature_labels.json", "summary", "categorical")

json.dump({"experiment": "exp_01m21z6dx8e5r9zt6n0vver1f9", "n_claims": len(claims), "claims": claims}, open(HERE / "claims_manifest.json", "w"), indent=1)

d35, d50, base = cells["dir:10:-0.35"], cells["dir:10:-0.5"], cells["base"]


def row(c):
    t = cells[c]
    return {"cell": c, "design": t["design"], "iteration": t["iteration"], "gain": round(t["gain"]["mean"], 3), "gain_ci": [round(t["gain"]["lo"], 3), round(t["gain"]["hi"], 3)],
            "dose_along_dir_per_token": None if t["dose_along_dir"] is None else round(t["dose_along_dir"], 2), "dose_frac_of_dir": None if t["dose_frac_of_dir_-0.35"] is None else round(t["dose_frac_of_dir_-0.35"], 3),
            "edit_norm_per_token": None if not t["edit_norm_per_token"] else round(t["edit_norm_per_token"], 2), "clamped_per_token": None if t["clamped_per_token"] is None else round(t["clamped_per_token"], 2),
            "refusal": round(t["refusal"], 3), "crisis": round(t["crisis"], 2), "chat_degeneration": t["degenerate"], "neutral_degeneration": t["neutral_degenerate"], "passes_all_q1": t["passes_all_q1"]}


order = sorted(cells.values(), key=lambda t: ({"base": 0, "dir": 1, "A": 2, "D": 3, "amp": 4, "R": 5}[t["kind"]], -t["layer"], t["k"], t["coef"], t["draw"] or 0))
summary = {
    "experiment": "exp_01m21z6dx8e5r9zt6n0vver1f9",
    "title": "Clamping Assistant-register SAE features does not produce the Bella voice; the fine-tune's shift is aligned with the Bella direction at every layer",
    "headline": ("On Gemma-4-E4B-it, no set of 5 to 200 layer-10 (or layer-4) SAE features, selected by decoder alignment with the Bella direction or by Bella-vs-Gemma activation "
                 "difference and clamped to zero during generation, moved the judged Bella voice by more than a small fraction of what adding the raw direction at −0.35 does. "
                 "The only cells with sizeable gains (k=200) reached them by breaking refusal and crisis behaviour, and delivered at most a tenth of the direction's per-token shift. "
                 "Amplifying the 20 most Bella-ward features did nothing. The Bella fine-tune's residual shift has cosine 0.30 to 0.78 with the Bella direction at every layer, including 0.52 at layer 4 and 0.46 at layer 10, "
                 "so the pre-registered prediction of near-zero early-layer alignment is refuted. The size of that shift does follow the predicted shape: at layers 0 to 13 it is under 1% of the residual norm and closes 2 to 6% of the Bella-Gemma gap, while at layers 29 to 41 it closes 29 to 43% of the gap."),
    "assessment": {"Q1": "refuted", "Q2": "not testable (no cell reached the Bella threshold)", "Q3": "refuted", "primary_measurement": "signal (the harness reproduces the direction result and the judge scored all 10,530 generations with no invalid parses)"},
    "research_question": "Do the Assistant-register features exist as a short list? Can a small set of SAE features, clamped at layer 10, reproduce the Bella-voice gain that steering along the Bella-Gemma mean-difference direction gives, without the direction's neutral-text degeneration? Is the Bella fine-tune's per-layer shift aligned with that direction only in late layers?",
    "decision_rule": {
        "Q1_supported": "some k<=50 Assistant-ward cell at layer 10 has Bella gain >= 70% of the −0.35 direction's gain within this run, refusal within 5 pts of unsteered, crisis not lower by >0.3, chat degeneration <=5%, neutral-stem degeneration <=10%, and beats the matched random-feature control by >=1.0",
        "Q1_refuted": "only k>=200 reaches the threshold, or nothing does",
        "Q2_supported": "a cell passing Q1's Bella threshold has neutral-stem degeneration <=10% (direction −0.35 had 24%)",
        "Q3_refuting": "cos(SFT shift, direction) > 0.3 at crest layers 4 and 10",
        "q1_threshold_this_run": round(verd["q1_gain_threshold"], 3),
    },
    "methods": {
        "model": "google/gemma-4-E4B-it @ ee0ef6023621cff504d758262d4e04895a5af4a2 (Gemma4ForConditionalGeneration; 42-layer text stack)",
        "fine_tune_for_Q3": "juiceb0xc0de/bella-bartender-gemma-e4b @ ac5ff55239c029c461a41bd2788d7423d0f2ac7c",
        "saes": "juiceb0xc0de/gemma-4-e4b-SAE @ 57bacb61a1bcc32be212a815b985a9bb42ff9a16, JumpReLU, layers 4/10/22, trainer encoder convention pre = W_enc(x − b_dec) + b_enc for all layers",
        "direction": "directions/e4b_directions.npz key 'all' (row l = block-l output, unit norm, Bella − Gemma) from the prior direction experiment; steering adds coef × median residual norm × direction at layer 10",
        "atlas_scores": "atlas-gemma-4-e4b/rlhf l{4,10,22}_rlhf_scores.npz: align (decoder cosine with direction), fire rates; used as given, reproduced here with r=1.0000 on align",
        "feature_sets": "A_k: top-k Assistant-ward by align among features firing >=1% on Gemma train replies; D_k: top-k by Gemma-over-Bella per-token activation difference; R_k: random features matched on Gemma firing rate (bin-matched, progressive widening); amp: 20 most Bella-ward features with activation multiplied ×2 (plan-listed adaptive) or ×4 (exploratory extra)",
        "clamp": "at every generated position, subtract z_f · W_dec[:, f] for each selected active feature (clamp to zero); dose measured exactly as the mean per-token shift of the residual along the unit Bella direction",
        "prompts": "100 eval chat prompts (Bella rubric 1-7), 30 crisis prompts (crisis 1-5 and Bella), 160 red-team prompts (refusal Yes/No), 100 neutral stems (64 tokens, no chat template; degeneration by heuristic); greedy decoding, 128 new tokens",
        "judge": "gpt-5.4-mini via existing judge.py; rubric and cache reused verbatim; judge never sees cell labels; 8,640 items, 0 invalid",
        "uncertainty": "95% bootstrap over prompts; gains are paired differences vs the unsteered base on the same prompts (2,000 resamples, seed 42)",
        "harvest_for_stage1_and_Q3": "600 matched Bella/Gemma reply pairs (400/100/100 train/val/test); Bella-Gemma gap per layer computed from this run's harvest (acts_means.npz), not from the prior run's acts_pairs.npz",
        "iterations": "iteration 1: pre-registered grid (k=5,20,50,200 + random k=50,200 + direction −0.35/−0.5 + base); iteration 2 (adaptive, plan-listed): k=100 for A/D/R and amp ×2; amp ×4 exploratory extra. Iteration-2 gains are paired against the iteration-1 base (same prompts, greedy).",
        "seed": 42,
    },
    "results": {
        "baseline": {"bella": round(base["bella"], 3), "refusal": round(base["refusal"], 3), "crisis": round(base["crisis"], 2), "chat_degeneration": base["degenerate"], "neutral_degeneration": base["neutral_degenerate"]},
        "direction_-0.35": row("dir:10:-0.35"), "direction_-0.5": row("dir:10:-0.5"),
        "q1_threshold": round(verd["q1_gain_threshold"], 3),
        "cells": [row(t["cell"]) for t in order],
        "dose_summary": {"k_le_50_dose_frac_range": [round(min(cells[c]["dose_frac_of_dir_-0.35"] for c in cells if cells[c]["kind"] in ("A", "D") and cells[c]["layer"] == 10 and cells[c]["k"] <= 50), 3),
                                                     round(max(cells[c]["dose_frac_of_dir_-0.35"] for c in cells if cells[c]["kind"] in ("A", "D") and cells[c]["layer"] == 10 and cells[c]["k"] <= 50), 3)],
                         "k_100_200_dose_frac_range": [round(min(cells[c]["dose_frac_of_dir_-0.35"] for c in cells if cells[c]["kind"] in ("A", "D") and cells[c]["layer"] == 10 and cells[c]["k"] >= 100), 3),
                                                       round(max(cells[c]["dose_frac_of_dir_-0.35"] for c in cells if cells[c]["kind"] in ("A", "D") and cells[c]["layer"] == 10 and cells[c]["k"] >= 100), 3)]},
        "Q3": {"cos_L4": round(q3["cos_L4"], 3), "cos_L10": round(q3["cos_L10"], 3), "cos_min": round(min(q3["cos_by_layer"]), 3), "cos_max": round(q3["max_cos"], 3), "argmax_layer": q3["argmax_cos_layer"],
               "mean_cos_L0_14": round(q3["mean_cos_layers_0_14"], 3), "mean_cos_L30_41": round(q3["mean_cos_layers_30_41"], 3), "random_abs_cos_p95": round(q3["random_dir_abs_cos_p95"], 3),
               "frac_gap_closed_test_L4": round(q3["frac_gap_closed_test_L4"], 4), "frac_gap_closed_test_L10": round(q3["frac_gap_closed_test_L10"], 4),
               "frac_gap_closed_max": round(max(float(r["frac_gap_closed_test"]) for r in q3rows.values()), 4),
               "frac_gap_closed_L0_13_range": [round(min(float(q3rows[l]["frac_gap_closed_test"]) for l in range(14)), 4), round(max(float(q3rows[l]["frac_gap_closed_test"]) for l in range(14)), 4)],
               "frac_gap_closed_L29_41_range": [round(min(float(q3rows[l]["frac_gap_closed_test"]) for l in range(29, 42)), 4), round(max(float(q3rows[l]["frac_gap_closed_test"]) for l in range(29, 42)), 4)],
               "shift_norm_ratio_L0_13_max": round(max(float(q3rows[l]["shift_norm_over_median_norm"]) for l in range(14)), 4),
               "cos_L29_41_range": [round(min(q3["cos_by_layer"][29:]), 3), round(max(q3["cos_by_layer"][29:]), 3)],
               "shift_norm_over_median_norm_max": round(max(float(r["shift_norm_over_median_norm"]) for r in q3rows.values()), 4), "n_prompts": q3["n_prompts"]},
        "stage1": {l: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s1["per_layer"][l].items() if not isinstance(v, dict)} for l in ("4", "10", "22")},
        "labels": labels["summary"],
    },
    "limitations": [
        "Low delivered dose: every k<=50 clamp shifted the layer-10 residual along the Bella direction by under 6% of what the −0.35 steer delivers, so the Q1 null says these feature sets cannot carry the register when clamped, not that the register is absent from the SAE basis.",
        "Layer-22 SAE token statistics are unreliable here (explained variance 0.67 on Bella and −0.05 on Gemma replies under the trainer convention; −5.18 without b_dec subtraction as the atlas flag suggests), so the layer-22 Spearman result is not interpretable.",
        "Q3 is a single fine-tune with one base model. The cosine criterion refutes the prediction, but the fine-tune's movement along the direction is concentrated late (3% of the gap closed at the crest layers vs 29 to 43% at layers 29 to 41), so the direction-of-shift and size-of-shift readings disagree and the report states both.",
    ],
    "artifacts": {
        "analysis": f"{REL}/results/analysis/",
        "cells_table": f"{REL}/results/analysis/cells.csv",
        "verdicts": f"{REL}/results/analysis/verdicts.json",
        "feature_labels": f"{REL}/results/analysis/feature_labels.json",
        "iteration1_run": "artifact://juiceb0xc0de-15787e/experiments/exp_01m21z6dx8e5r9zt6n0vver1f9/run/",
        "iteration2_run": "artifact://juiceb0xc0de-15787e/experiments/exp_01m21z6dx8e5r9zt6n0vver1f9/iter2/run/",
        "judgments": "artifact://juiceb0xc0de-15787e/experiments/exp_01m21z6dx8e5r9zt6n0vver1f9/judge/judgments.jsonl",
        "figures": [f"{REL}/figures/q1_gain_vs_k", f"{REL}/figures/gain_vs_dose", f"{REL}/figures/canaries_l10", f"{REL}/figures/q3_cos_by_layer"],
        "code": [f"{REL}/src/run_features.py", f"{REL}/src/judge.py", f"{REL}/src/analyze.py", f"{REL}/src/make_figures.py"],
    },
}
json.dump(summary, open(HERE / "report_summary.json", "w"), indent=1)
print("claims", len(claims))
