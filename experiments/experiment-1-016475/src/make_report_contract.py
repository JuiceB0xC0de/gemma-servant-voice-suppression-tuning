"""Build report_summary.json and claims_manifest.json from results/analysis/summary.json (no transcription)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
S = json.load(open(HERE / "results/analysis/summary.json"))
SRC = "experiments/experiment-1-016475/results/analysis/summary.json"
PROF = {m: S["profiles"][m] for m in S["profiles"]}
Q1, Q2, Q3 = S["q1"], S["q2"], S["q3"]
TAB = {(r["model"], r["layer"], r["coef"]): r for r in S["steering_table"]}
claims = []


def claim(cid, text, value, key, source=SRC):
    claims.append({"id": cid, "claim": text, "value": value, "source": source, "key": key})
    return value


r3 = lambda x: round(x, 3)

# --- Q1
claim("q1_crest_a_d", "E2B layer 4 separation d", r3(Q1["d_a"]), "q1.d_a")
claim("q1_crest_b_d", "E2B layer 13 separation d", r3(Q1["d_b"]), "q1.d_b")
claim("q1_trough_layer", "E2B dip layer between the crests", Q1["trough_layer"], "q1.trough_layer")
claim("q1_trough_d", "E2B dip d", r3(Q1["trough_d"]), "q1.trough_d")
claim("q1_drop_a", "E2B crest 4 minus dip", r3(Q1["drop_a"]), "q1.drop_a")
claim("q1_drop_a_ci", "bootstrap 95% CI of crest 4 minus dip", [r3(x) for x in Q1["bootstrap"]["drop_a_ci"]], "q1.bootstrap.drop_a_ci")
claim("q1_drop_b", "E2B crest 13 minus dip", r3(Q1["drop_b"]), "q1.drop_b")
claim("q1_drop_b_ci", "bootstrap 95% CI of crest 13 minus dip", [r3(x) for x in Q1["bootstrap"]["drop_b_ci"]], "q1.bootstrap.drop_b_ci")
claim("q1_p_both", "bootstrap probability both drops >= 0.15", Q1["bootstrap"]["p_both_drops_ge_0.15"], "q1.bootstrap.p_both_drops_ge_0.15")
claim("q1_supported", "Q1 verdict (point rule and bootstrap rule)", Q1["supported"], "q1.supported")
claim("q1_supported_point", "Q1 verdict under the point rule alone", Q1["supported_point_rule"], "q1.supported_point_rule")
for m in PROF:
    P = PROF[m]
    claim(f"{m}_pooling", f"{m} pooling chosen on validation", P["pooling"], f"profiles.{m}.pooling")
    claim(f"{m}_min_d", f"{m} minimum layer d", r3(P["min_d"]), f"profiles.{m}.min_d")
    claim(f"{m}_max_d", f"{m} maximum layer d", r3(P["max_d"]), f"profiles.{m}.max_d")
    claim(f"{m}_argmax", f"{m} layer of maximum d", P["argmax_layer"], f"profiles.{m}.argmax_layer")
    claim(f"{m}_maxima", f"{m} interior local maxima above the permutation null", P["interior_maxima_above_null"], f"profiles.{m}.interior_maxima_above_null")
    claim(f"{m}_surviving", f"{m} crests surviving the crest-minus-trough bootstrap", P["uncertainty"]["surviving_layers"], f"profiles.{m}.uncertainty.surviving_layers")
    claim(f"{m}_median_se", f"{m} median bootstrap SE of d", r3(P["uncertainty"]["median_se"]), f"profiles.{m}.uncertainty.median_se")
    claim(f"{m}_null_max", f"{m} maximum permutation-null p95 across layers", r3(max(P["null_d_p95"])), f"profiles.{m}.null_d_p95 (max)")
    claim(f"{m}_null_run_model_max", f"{m} maximum contaminated shuffled-null p95 (run_model)", r3(max(P["null_shuffled_run_model_p95"])), f"profiles.{m}.null_shuffled_run_model_p95 (max)")
    claim(f"{m}_global", f"{m} global-attention layers", P["global_layers"], f"profiles.{m}.global_layers")
    claim(f"{m}_strongest3", f"{m} three strongest crests", P["strongest3"], f"profiles.{m}.strongest3")
    claim(f"{m}_strongest3_on_global", f"{m} strongest crests on global layers", P["strongest3_on_global"], f"profiles.{m}.strongest3_on_global")
    claim(f"{m}_ntok_gemma", f"{m} mean Gemma reply tokens (all 600 pairs)", r3(P["n_tok_gemma_mean"]), f"profiles.{m}.n_tok_gemma_mean")
    claim(f"{m}_ntok_bella", f"{m} mean Bella reply tokens", r3(P["n_tok_bella_mean"]), f"profiles.{m}.n_tok_bella_mean")
    LC = P["length_confound"]
    claim(f"{m}_d_len_only", f"{m} d of log token count between sides (test)", r3(LC["d_from_length_fit_only"][0]), f"profiles.{m}.length_confound.d_from_length_fit_only[0]")
    rb = LC["spearman_proj_ntok_bella"]
    claim(f"{m}_rho_bella_range", f"{m} within-Bella Spearman(projection, tokens) range", [r3(min(rb)), r3(max(rb))], f"profiles.{m}.length_confound.spearman_proj_ntok_bella (min,max)")
    rg = LC["spearman_proj_ntok_gemma"]
    claim(f"{m}_rho_gemma_range", f"{m} within-Gemma Spearman range", [r3(min(rg)), r3(max(rg))], f"profiles.{m}.length_confound.spearman_proj_ntok_gemma (min,max)")
    dp = LC["d_length_partialled"]
    claim(f"{m}_d_part_range", f"{m} length-partialled d range", [r3(min(dp)), r3(max(dp))], f"profiles.{m}.length_confound.d_length_partialled (min,max)")
    claim(f"{m}_ac_auroc_max", f"{m} authentic-vs-corporate snippet AUROC of pair direction (max over layers)", r3(max(P["ac_auroc"])), f"profiles.{m}.ac_auroc (max)")
    claim(f"{m}_ac_auroc_min", f"{m} authentic-vs-corporate snippet AUROC (min over layers)", r3(min(P["ac_auroc"])), f"profiles.{m}.ac_auroc (min)")
    claim(f"{m}_stage2_layers", f"{m} steering layers", P["stage2_layers"], f"profiles.{m}.stage2_layers")
claim("q2_supported", "Q2 verdict as registered", Q2["supported_point_rule"], "q2.supported_point_rule")
claim("q2_same_abs", "layers among the strongest three shared by both models", Q2["same_absolute_layers"], "q2.same_absolute_layers")
claim("q2_L9_d", "E2B hold-out layer 9 d", r3(Q2["e2b_L9_d"]), "q2.e2b_L9_d")
G = Q2["global_minus_one_posthoc"]
for k in ("E2B_surviving", "E2B_all_maxima", "E4B_surviving", "E4B_all_maxima", "E4B_first32_surviving"):
    claim(f"gm1_{k}_count", f"{k}: crests one layer before a global block / total", [G[k]["on_global_minus_1"], G[k]["n"]], f"q2.global_minus_one_posthoc.{k}.on_global_minus_1, .n")
    claim(f"gm1_{k}_p", f"{k}: hypergeometric P(>= that many) by chance", G[k]["p_on_global_minus_1_ge"], f"q2.global_minus_one_posthoc.{k}.p_on_global_minus_1_ge")
    claim(f"gm1_{k}_on_global", f"{k}: crests exactly on a global layer", G[k]["on_global"], f"q2.global_minus_one_posthoc.{k}.on_global")

# --- Q3
for m in ("E2B", "E4B"):
    b = TAB[(m, -1, 0.0)]
    for k in ("bella", "corporate", "refusal", "crisis", "degenerate", "n_words", "ppl"):
        claim(f"{m}_base_{k}", f"{m} unsteered {k}", r3(b[k]), f"steering_table[model={m},layer=-1,coef=0].{k}")
    claim(f"{m}_q3_any_pre", f"{m} any layer passes the pre-registered window rule", Q3[m]["supported_any_layer"], f"q3.{m}.supported_any_layer")
    claim(f"{m}_q3_any_post", f"{m} any layer passes the post-hoc window rule", Q3[m]["supported_any_layer_posthoc"], f"q3.{m}.supported_any_layer_posthoc")
    for l, v in Q3[m]["layers"].items():
        claim(f"{m}_L{l}_win_pre", f"{m} layer {l} pre-registered window", v["usable_window"], f"q3.{m}.layers.{l}.usable_window")
        claim(f"{m}_L{l}_win_post", f"{m} layer {l} post-hoc window", v["usable_window_posthoc"], f"q3.{m}.layers.{l}.usable_window_posthoc")
        claim(f"{m}_L{l}_voice_fluent", f"{m} layer {l} coefficients with Bella gain, corporate halved and fluent", v["voice_window_fluent"], f"q3.{m}.layers.{l}.voice_window_fluent")
cells = [("E4B", 7, -0.5), ("E4B", 7, -0.65), ("E4B", 7, -0.8), ("E4B", 7, -1.0), ("E2B", 9, -0.35), ("E2B", 9, -0.5), ("E2B", 9, -0.65), ("E2B", 4, -0.5), ("E2B", 4, -0.65), ("E4B", 11, -0.5), ("E4B", 11, -0.35), ("E2B", 28, -1.0), ("E2B", 19, -0.5),
         ("E4B", 10, -0.35), ("E4B", 10, -0.5), ("E4B", 10, -0.65), ("E4B", 10, -0.8), ("E2B", 13, -0.25), ("E2B", 13, -0.35), ("E2B", 13, -0.5), ("E2B", 13, -0.65), ("E4B", 4, -0.5), ("E4B", 4, -0.65), ("E4B", 4, -0.8), ("E4B", 22, -0.35), ("E4B", 22, -0.5)]
for (m, l, c) in (("E4B", 10, -0.35), ("E4B", 10, -0.5), ("E4B", 7, -0.5), ("E4B", 7, -0.65), ("E2B", 13, -0.25), ("E2B", 13, -0.35)):
    pc = Q3[m]["layers"][str(l)]["per_coef"][str(c)]
    claim(f"{m}_L{l}_c{c}_checks_pre", f"{m} layer {l} coef {c} pre-registered checks", pc["checks"], f"q3.{m}.layers.{l}.per_coef.{c}.checks")
    claim(f"{m}_L{l}_c{c}_checks_post", f"{m} layer {l} coef {c} post-hoc checks", pc["checks_posthoc"], f"q3.{m}.layers.{l}.per_coef.{c}.checks_posthoc")
    claim(f"{m}_L{l}_c{c}_ppl_ratio", f"{m} layer {l} coef {c} neutral perplexity ratio to unsteered", r3(pc["ppl_ratio"]), f"q3.{m}.layers.{l}.per_coef.{c}.ppl_ratio")
for m in PROF:
    claim(f"{m}_stage2_layers_iter1", f"{m} layers steered in iteration 1 (chosen with the contaminated run-time null)", PROF[m]["stage2_layers_iter1"], f"profiles.{m}.stage2_layers_iter1")
    claim(f"{m}_crest_layers_added", f"{m} valid-null crest layers steered in iteration 3", PROF[m]["crest_layers_added_iter3"], f"profiles.{m}.crest_layers_added_iter3")
    claim(f"{m}_steered_are_crests", f"{m} which steered layers are interior maxima / bootstrap-surviving crests", PROF[m]["stage2_layers_are_crests"], f"profiles.{m}.stage2_layers_are_crests")
SH = Q1["shape"]
claim("q1_trough_pred_min", "E2B minimum d over the predicted trough layers 7-11", r3(SH["predicted_trough_min_d"]), "q1.shape.predicted_trough_min_d")
claim("q1_max_after_14", "E2B maximum d after layer 14", r3(SH["max_d_after_layer_14"]), "q1.shape.max_d_after_layer_14")
claim("q1_plateau_range", "E2B range of d over layers 4-14", r3(SH["plateau_4_to_14_range_d"]), "q1.shape.plateau_4_to_14_range_d")
claim("q1_L4_survives", "E2B layer 4 survives the neighbour bootstrap", SH["layer4_survives_neighbour_bootstrap"], "q1.shape.layer4_survives_neighbour_bootstrap")
claim("q1_L4_drop_right_ci", "E2B layer 4 minus layer 5, bootstrap 95% CI", [r3(x) for x in SH["layer4_crest_record"]["drop_right_ci"]], "q1.shape.layer4_crest_record.drop_right_ci")
claim("q1_L13_survives", "E2B layer 13 survives the neighbour bootstrap", SH["layer13_survives_neighbour_bootstrap"], "q1.shape.layer13_survives_neighbour_bootstrap")
claim("q1_maxima_2_14", "E2B interior maxima between layers 2 and 14", SH["interior_maxima_between_2_and_14"], "q1.shape.interior_maxima_between_2_and_14")
for (m, l, c) in cells:
    r = TAB[(m, l, c)]
    for k in ("bella", "bella_lo", "bella_hi", "corporate", "refusal", "refusal_lo", "refusal_hi", "crisis", "crisis_lo", "crisis_hi", "degenerate", "n_words", "ppl", "dose_contrast"):
        if r.get(k) is not None:
            claim(f"{m}_L{l}_c{c}_{k}", f"{m} layer {l} coef {c} {k}", r3(r[k]), f"steering_table[model={m},layer={l},coef={c}].{k}")
claim("n_judgments", "total judge calls (unique items)", None, "results/judge/judge_meta.json n_items", source="experiments/experiment-1-016475/results/judge/judge_meta.json")
jm = json.load(open(HERE / "results/judge/judge_meta.json"))
claims[-1]["value"] = jm["n_items"]
pilot = json.load(open(HERE / "results/judge/pilot.json"))
claim("judge_pilot_auroc", "judge pilot AUROC Bella corpus vs unsteered Gemma", r3(pilot["auroc"]), "auroc", source="experiments/experiment-1-016475/results/judge/pilot.json")
claim("judge_model", "judge model", jm["judge_model"], "judge_model", source="experiments/experiment-1-016475/results/judge/judge_meta.json")
# --- SAE
for l, v in S["sae"].items():
    claim(f"sae_L{l}_need50", f"E2B SAE layer {l}: decoder features for 50% of the direction", v["features_needed"]["0.5"], f"sae.{l}.features_needed.0.5")
    claim(f"sae_L{l}_need80", f"E2B SAE layer {l}: features for 80%", v["features_needed"]["0.8"], f"sae.{l}.features_needed.0.8")
    claim(f"sae_L{l}_top_cos", f"E2B SAE layer {l}: max |cos| of a single decoder feature with the direction", r3(v["cos_abs_top1"]), f"sae.{l}.cos_abs_top1")
    claim(f"sae_L{l}_l0", f"E2B SAE layer {l}: mean L0 on our activations vs trainer meta", [r3(v["calibration"]["mean_l0"]), r3(v["calibration"]["meta_mean_l0"])], f"sae.{l}.calibration.mean_l0, meta_mean_l0")
    claim(f"sae_L{l}_ev", f"E2B SAE layer {l}: explained variance ours vs meta", [r3(v["calibration"]["ev"]), r3(v["calibration"]["meta_ev"])], f"sae.{l}.calibration.ev, meta_ev")
    claim(f"sae_L{l}_nfeat", f"E2B SAE layer {l}: features with |d|>1 between sides", v["n_features_abs_d_gt_1"], f"sae.{l}.n_features_abs_d_gt_1")
# --- logit lens (top promoted tokens, late layers)
for m, l in (("E2B", "28"), ("E4B", "28")):
    claim(f"lens_{m}_L{l}_promoted", f"{m} layer {l} logit-lens top promoted tokens", S["logit_lens"][m][l]["promoted"][:8], f"logit_lens.{m}.{l}.promoted[:8]")
# --- data
man = json.load(open(HERE / "results/data/manifest.json"))
claim("data_pairs", "matched pairs (train/val/test)", [400, 100, 100], "splits", source="experiments/experiment-1-016475/results/data/manifest.json")
claim("data_crisis_eval", "crisis eval prompts", 30, "crisis_eval", source="experiments/experiment-1-016475/results/data/manifest.json")
claim("data_redteam", "red-team prompts", 160, "red_team", source="experiments/experiment-1-016475/results/data/manifest.json")

report = {
    "title": "The Bella-vs-Gemma direction is decodable at every layer of Gemma 4 E2B and E4B; scaling it down at E4B layer 10 frees the voice over two coefficients but fails the registered perplexity check",
    "headline": ("A mean-difference direction between Bella's replies and Gemma's own replies separates the two voices at every layer of both models "
                 "(d 2.5 to 5.0 on 100 held-out pairs). E2B meets the registered two-crest rule literally (layers 4 and 13, dip at layer 12), but the shape is a rise, "
                 "a plateau from layer 4 to 14 (range 0.78 d) and a decline: the predicted trough at layers 7 to 11 did not appear (d 4.50 to 4.82, higher than every layer past 14), "
                 "and layer 4 is not separable from layers 5 to 6 by bootstrap (layer 13 is). The crests do not fall on global-attention layers (Q2 refuted). "
                 "Subtracting the direction at E4B layer 10, a bootstrap-surviving crest, moves Gemma toward Bella's register at coefficients -0.35 and -0.5 "
                 "(judged Bella-ness 1.13 to 2.98 and 4.06 / 7, corporate phrasing 0.20 to 0.10 and 0.07 hits per reply, refusal 0.994 to 0.969 and 0.968, crisis quality 3.03 to 3.16 and 3.24, "
                 "degeneration 0 %), and E4B layer 7 does the same at -0.5 and -0.65. Under the registered rule the layer-10 cells and layer 7 -0.5 fail only the neutral-perplexity check "
                 "(ratio 2.5 to 7.5 vs the 2x limit; the check penalises the voice change itself and rewards repetition) and, at -0.5, the two-sided crisis check because crisis quality improved by 0.21 to 0.28; "
                 "layer 7 -0.65 passes perplexity (ratio 1.9) but drops refusal by 5.1 points. Q3 is therefore not supported under the registered rule; under the post-hoc rule that swaps perplexity for a "
                 "degeneration rate it is supported at E4B layer 10 (two contiguous coefficients) and passes at layer 7 -0.5 only. E2B layer 13 (its surviving crest) gains voice at -0.25 and -0.35 (Bella-ness 3.68 and 4.32) but "
                 "misses the corporate halving by 0.01 at -0.25 and loses 23 refusal points at -0.5. Beyond about -0.65 every layer collapses into repetition or refusal loss."),
    "assessment": {"q1": "supported by the registered point rule and the paired bootstrap of the two crest-minus-dip drops, but on a one-layer dip at layer 12: layer 4 does not survive its own neighbour test (drop to layer 5 0.18, CI [-0.03, 0.43]), the predicted 7-11 trough is absent, and the profile is rise, plateau 4-14, decline rather than the sketched wave",
                   "q2": "refuted as registered (1 of 3 E2B crests and 0 of 3 E4B crests on global layers; both peak at layer 4). Post hoc: bootstrap-surviving crests sit one layer before a global block in E4B (5/5 with all-reply pooling, 3/4 with first-32 pooling); in E2B 3/4 surviving crests do so but the result is not significant once layer 4 is counted (all maxima 3/9, p=0.28). Suggestive only; adjacent layers are correlated and the crest list depends on pooling.",
                   "q3": "not supported under the registered rule: E4B layer 10 at -0.35 and -0.5 and layer 7 at -0.5 pass Bella gain (+1.9 to +2.9), corporate halving, refusal (within 2.6 points) and degeneration (0 %) but fail the neutral-perplexity check (ratio 2.5 to 7.5), which penalises the voice change itself, and the -0.5 cells also fail the two-sided crisis check because crisis quality improved by 0.21 to 0.28; layer 7 -0.65 passes perplexity but drops refusal by 5.1 points. Under the post-hoc rule (degeneration instead of perplexity, crisis one-sided) E4B layer 10 passes over two contiguous coefficients (-0.35, -0.5) and layer 7 at -0.5 only; no E2B layer passes two coefficients (layer 13 passes at -0.35 alone; at -0.25 corporate is 0.11 vs the 0.10 needed).",
                   "overall": "partial_signal"},
    "decision_rules": {
        "q1": "two interior local maxima above the null within +-1 of layers 4 and 13/14, each >= 0.15 d above the minimum between them; tightened: crest-minus-trough paired-bootstrap 95% CI excludes 0",
        "q2": ">= 2 of the 3 strongest crests on global-attention layers in both models and E2B layer 9 above null; refuted if the same absolute layers peak in both models",
        "q3": "Bella-ness +1.5, corporate hit rate halved, refusal and crisis within 5 points, neutral perplexity < 2x, over >= 2 contiguous coefficients; post-hoc variant swaps perplexity for a degeneration rate within 10 points and treats crisis one-sided",
    },
    "methods": {
        "models": {"E2B": "google/gemma-4-E2B-it @3e22461f (35 layers, d 1536)", "E4B": "google/gemma-4-E4B-it @ee0ef602 (42 layers, d 2560)"},
        "layer_convention": "layer l = output of decoder block l (0-indexed); global-attention layers E2B 4,9,14,19,24,29,34; E4B 5,11,17,23,29,35,41",
        "data": "7,232 cleaned Bella exchanges; 600 matched pairs (same user prompt, Bella reply vs Gemma greedy 96-token reply), split by prompt 400/100/100; 100 held-out eval prompts, 30 crisis prompts, 160 red-team prompts, 100 neutral stems, 500 authentic + 500 corporate snippets",
        "direction": "unit difference of means (Bella minus Gemma) of mean-pooled reply-token residuals per layer, fit on train; pooling (all reply tokens vs first 32) chosen on validation mean d",
        "uncertainty": "paired bootstrap over test pairs (2000), permutation null with labels flipped on all pairs (200), hypergeometric test for the post-hoc global-minus-one pattern",
        "steering": "h += c * median_norm_l * (Gemma minus Bella unit direction) at all positions of one layer; coefficients -0.15,-0.25,-0.35,-0.5,-0.65,-0.8,-1,-2,-4,+1; 128 new tokens greedy",
        "steered_layers": {m: {"iteration_1_selection": PROF[m]["stage2_layers_iter1"], "crest_layers_added_iteration_3": PROF[m]["crest_layers_added_iter3"], "all": PROF[m]["stage2_layers"],
                               "which_are_crests": PROF[m]["stage2_layers_are_crests"]} for m in PROF},
        "judge": f"{jm['judge_model']} with top-5 logprob expected scores; Bella-ness 1-7, refusal yes/no, crisis 1-5; corporate and swear regex; pilot AUROC {r3(pilot['auroc'])}; {jm['n_items']} judgments, 0 invalid",
        "sae": "E2B JumpReLU SAEs (juiceb0xc0de/gemma-4-e2b-it-SAE) at layers 4, 9, 13, 19, 28: least-squares reconstruction of the direction from top-k decoder features, per-feature d between sides on 100 test pairs",
        "seed": 42,
    },
    "results": {
        "profiles": {m: {"pooling": PROF[m]["pooling"], "test_d": PROF[m]["test_d"], "ci_lo": PROF[m]["uncertainty"]["ci_lo"], "ci_hi": PROF[m]["uncertainty"]["ci_hi"],
                         "null_p95": PROF[m]["null_d_p95"], "surviving": PROF[m]["uncertainty"]["surviving_layers"], "maxima": PROF[m]["interior_maxima_above_null"],
                         "global_layers": PROF[m]["global_layers"]} for m in PROF},
        "q1": {k: Q1[k] for k in ("supported", "supported_point_rule", "crest_a", "crest_b", "trough_layer", "d_a", "d_b", "trough_d", "drop_a", "drop_b", "bootstrap", "shape")},
        "q2": {"supported": Q2["supported_point_rule"], "E2B": Q2["E2B"], "E4B": Q2["E4B"], "same_absolute_layers": Q2["same_absolute_layers"], "global_minus_one_posthoc": G},
        "q3": {m: {"supported_any_layer": Q3[m]["supported_any_layer"], "supported_any_layer_posthoc": Q3[m]["supported_any_layer_posthoc"],
                   "layers": {l: {k: v[k] for k in ("usable_window", "usable_window_posthoc", "voice_window", "voice_window_fluent")} for l, v in Q3[m]["layers"].items()}} for m in Q3},
        "steering_table": S["steering_table"],
        "examples": S["examples"],
        "sae": {l: {"features_needed": v["features_needed"], "calibration": v["calibration"], "cos_abs_top1": v["cos_abs_top1"], "n_features_abs_d_gt_1": v["n_features_abs_d_gt_1"]} for l, v in S["sae"].items()},
        "logit_lens": S["logit_lens"],
    },
    "limitations": [
        "The direction is a voice-plus-length direction: Gemma replies average 89 to 94 tokens (mostly truncated at 96) vs 37 to 45 for Bella; log token count alone separates the sides at d 1.7; within-Bella projections correlate with length at Spearman -0.4 to -0.8.",
        "The Q3 perplexity criterion (neutral continuations scored under the unsteered model) rewards repetition and penalises short, lowercase, slang-heavy continuations, i.e. the voice change itself; every otherwise-passing cell fails on it, so the post-hoc verdict rests on a degeneration rate instead.",
        "The iteration-1 steering layers (E2B 19, 28, 4, 9; E4B 28, 7, 23, 11) were selected at run time with the contaminated shuffled null, so the plan's crest layers were not steered until a third iteration added E2B 13 and E4B 4, 10, 22; E4B layer 7, the best iteration-1 layer, is not a crest under the valid null. The SAE decomposition covers E2B layers 4, 9, 13, 19, 28.",
        "Layer-position claims (Q1 crests, the post-hoc global-minus-one pattern) inherit the length confound: length partialling changes the E4B maxima (layer 4 disappears) and the E2B first-32 vs all-token pooling changes which crests survive.",
        "The global-minus-one pattern is post hoc and would read as the registered prediction under a block-input indexing convention; it needs a pre-registered test on a third model.",
        "SAE calibration: at layers 9, 19 and 28 the E2B SAEs fire 40 to 60 % more features on assistant-reply tokens than in training (EV 0.75 to 0.92 vs 0.81 to 0.94), so feature-level statements there are approximate.",
        "The judge is gpt-5.4-mini (plan named claude-sonnet-5; no Anthropic credential was available); it scores repeated non-English tokens as Bella-like, which the degeneration filter addresses.",
        "Crisis pool: only 12 strict crisis prompts exist in the corpus; the 30-prompt crisis set adds 18 first-person distress prompts.",
    ],
    "deviations": [
        "Judge model gpt-5.4-mini via the researcher's OpenAI key instead of claude-sonnet-5.",
        "Crisis evaluation set widened to first-person distress prompts (12 strict + 18 distress).",
        "Coefficient grid extended in a second iteration (-0.15, -0.35, -0.65, -0.8), allowed by the plan's adaptive-input clause; all ten coefficients reported.",
        "Shuffled-label null from the GPU script found to be contaminated and replaced by a proper permutation null computed post hoc; both reported. Because that null also drove the run-time layer selection, a third steering iteration added the valid-null crest layers (E2B 13; E4B 4, 10, 22) at all ten coefficients and the E2B SAE decomposition at layer 13.",
        "Crest counting tightened with a paired bootstrap; both the registered point rule and the tightened rule are reported.",
    ],
    "artifacts": {
        "summary": SRC,
        "steering_cells": "experiments/experiment-1-016475/results/steering_cells.json",
        "profiles_with_uncertainty": "experiments/experiment-1-016475/results/profiles_with_uncertainty.json",
        "figures": "experiments/experiment-1-016475/figures/",
        "judge_outputs": "experiments/experiment-1-016475/results/judge/ (judgments.jsonl in artifact store)",
        "gpu_outputs": ["artifact://juiceb0xc0de-15787e/experiments/exp_01m1xz587ze4t8np5xr6016475/E2B/run/", "artifact://juiceb0xc0de-15787e/experiments/exp_01m1xz587ze4t8np5xr6016475/E4B/run/",
                        "artifact://juiceb0xc0de-15787e/experiments/exp_01m1xz587ze4t8np5xr6016475/iter2/E2B/run/", "artifact://juiceb0xc0de-15787e/experiments/exp_01m1xz587ze4t8np5xr6016475/iter2/E4B/run/",
                        "artifact://juiceb0xc0de-15787e/experiments/exp_01m1xz587ze4t8np5xr6016475/iter3/E2B/run/", "artifact://juiceb0xc0de-15787e/experiments/exp_01m1xz587ze4t8np5xr6016475/iter3/E4B/run/"],
        "code": ["experiments/experiment-1-016475/src/prepare_data.py", "experiments/experiment-1-016475/src/run_model.py", "experiments/experiment-1-016475/src/post_stage1.py",
                 "experiments/experiment-1-016475/src/judge.py", "experiments/experiment-1-016475/src/analyze.py"],
    },
}
(HERE / "report_summary.json").write_text(json.dumps(report, indent=1))
(HERE / "claims_manifest.json").write_text(json.dumps({"claims": claims, "n_claims": len(claims)}, indent=1))
print(len(claims), "claims")
