"""Report figures from results/analysis (pod-side, silico-figures).

Run: cd $SILICO_EXPERIMENT_DIR && uv run --no-sync python src/make_figures.py
Bundles land in figures/<name>/ (html render, data.json, plot.py).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import PLOTLY_CONFIG, add_reference_line, apply_theme, save_figure_bundle

HERE = Path(__file__).resolve().parent.parent
AN = HERE / "results" / "analysis"
_ec = HERE / "figures" / "entity_colors.json"
# extend the checkpoint agent's entity map (never reshuffle): amp cells get the next unused EDITORIAL_8 slot
COLORS = json.load(open(_ec if _ec.exists() else HERE / "figures" / "checkpoint-renders" / "entity_colors.json"))
if "amp (Bella-ward amplified)" not in COLORS:
    COLORS["amp (Bella-ward amplified)"] = "#31362E"
    json.dump(COLORS, open(_ec, "w"), indent=1)
C_A, C_D, C_R, C_DIR, C_AMP, C_BASE = (COLORS["A (Assistant-ward)"], COLORS["D (Gemma-over-Bella)"], COLORS["R (random matched)"],
                                       COLORS["direction"], COLORS["amp (Bella-ward amplified)"], COLORS["base"])

table = json.load(open(AN / "cells.json"))
verd = json.load(open(AN / "verdicts.json"))
thresh = verd["q1_gain_threshold"]
dir35 = table["dir:10:-0.35"]


def cells(kind, layer):
    return sorted([t for t in table.values() if t["kind"] == kind and t["layer"] == layer], key=lambda t: (t["k"], t["coef"], t["draw"] or 0))


def err(ts):
    return dict(type="data", array=[t["gain"]["hi"] - t["gain"]["mean"] for t in ts], arrayminus=[t["gain"]["mean"] - t["gain"]["lo"] for t in ts], thickness=1)


def canary_ok(t):
    ch = t["checks"]
    return ch["refusal_within_5pts"] and ch["crisis_not_lower_by_0p3"] and ch["chat_degeneration_le_5pct"] and ch["neutral_degeneration_le_10pct"]


def hover(t):
    return (f"{t['cell']}<br>gain {t['gain']['mean']:+.2f} [{t['gain']['lo']:+.2f}, {t['gain']['hi']:+.2f}]"
            f"<br>dose along direction {t['dose_along_dir']:.2f}/token<br>refusal {t['refusal']:.3f}, chat degeneration {t['degenerate']:.0%}, neutral {t['neutral_degenerate']:.0%}")


# ---------------- Figure 1: Bella gain vs number of clamped features
fig = go.Figure()
series = [("A", 10, "Assistant-ward, layer 10", C_A, "solid", 3), ("D", 10, "Gemma-over-Bella, layer 10", C_D, "solid", 2)]
data1 = {"threshold": thresh, "dir_-0.35_gain": dir35["gain"]["mean"], "series": {}}
for kind, layer, name, col, dash, w in series:
    ts = cells(kind, layer)
    data1["series"][name] = [{"k": t["k"], "gain": t["gain"]["mean"], "lo": t["gain"]["lo"], "hi": t["gain"]["hi"], "canaries_intact": canary_ok(t)} for t in ts]
    fig.add_trace(go.Scatter(x=[t["k"] for t in ts], y=[t["gain"]["mean"] for t in ts], error_y=err(ts), mode="lines+markers", name=name,
                             line=dict(color=col, dash=dash, width=w), marker=dict(color=col, size=9, symbol=["circle" if canary_ok(t) else "circle-open" for t in ts], line=dict(width=2, color=col)),
                             text=[hover(t) for t in ts], hoverinfo="text"))
rs = cells("R", 10)
data1["series"]["Random matched, layer 10"] = [{"k": t["k"], "gain": t["gain"]["mean"], "lo": t["gain"]["lo"], "hi": t["gain"]["hi"], "draw": t["draw"]} for t in rs]
fig.add_trace(go.Scatter(x=[t["k"] for t in rs], y=[t["gain"]["mean"] for t in rs], error_y=err(rs), mode="markers", name="Random matched, layer 10",
                         marker=dict(color=C_R, size=8, symbol="diamond"), text=[hover(t) for t in rs], hoverinfo="text"))
am = cells("amp", 10)
data1["series"]["Bella-ward amplified (k=20)"] = [{"k": t["k"], "coef": t["coef"], "gain": t["gain"]["mean"], "lo": t["gain"]["lo"], "hi": t["gain"]["hi"], "exploratory": t["exploratory"]} for t in am]
fig.add_trace(go.Scatter(x=[t["k"] for t in am], y=[t["gain"]["mean"] for t in am], error_y=err(am), mode="markers", name="Bella-ward amplified ×2, ×4 (k=20)",
                         marker=dict(color=C_AMP, size=9, symbol="square"), text=[hover(t) for t in am], hoverinfo="text"))
add_reference_line(fig, y=dir35["gain"]["mean"], label=f"direction −0.35: {dir35['gain']['mean']:+.2f}")
add_reference_line(fig, y=thresh, label=f"Q1 threshold: {thresh:+.2f}")
add_reference_line(fig, y=0, label="unsteered")
fig.update_xaxes(type="log", title="Features clamped, k", tickvals=[5, 20, 50, 100, 200], ticktext=["5", "20", "50", "100", "200"])
fig.update_yaxes(title="Bella voice gain (judge points)", range=[-0.4, 2.1])
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
apply_theme(fig, height=460)
save_figure_bundle(fig, "q1_gain_vs_k", data=data1, root=HERE / "figures",
                   alt="Bella voice gain against number of clamped features for three selection rules, random controls and amplification cells, with the direction gain and Q1 threshold as reference lines.")

# ---------------- Figure 2: gain vs delivered dose along the Bella direction
fig = go.Figure()
groups = [("dir", "Direction steer", C_DIR, "star", 13), ("A", "Assistant-ward clamp", C_A, "circle", 10), ("D", "Gemma-over-Bella clamp", C_D, "circle", 10),
          ("amp", "Bella-ward amplified", C_AMP, "square", 10), ("R", "Random matched clamp", C_R, "diamond", 9)]
data2 = []
for kind, name, col, sym, size in groups:
    ts = sorted([t for t in table.values() if t["kind"] == kind], key=lambda t: t["dose_along_dir"])
    for t in ts:
        data2.append({"cell": t["cell"], "kind": kind, "layer": t["layer"], "dose_along_dir": t["dose_along_dir"], "edit_norm": t["edit_norm_per_token"], "gain": t["gain"]["mean"], "lo": t["gain"]["lo"], "hi": t["gain"]["hi"], "canaries_intact": canary_ok(t)})
    fig.add_trace(go.Scatter(x=[t["dose_along_dir"] for t in ts], y=[t["gain"]["mean"] for t in ts], error_y=err(ts), mode="markers", name=name,
                             marker=dict(color=col, size=size, symbol=[sym if canary_ok(t) else sym + "-open" for t in ts], line=dict(width=2, color=col)),
                             text=[hover(t) for t in ts], hoverinfo="text"))
add_reference_line(fig, y=thresh, label=f"Q1 threshold: {thresh:+.2f}")
add_reference_line(fig, x=0, label="no shift along direction")
fig.update_xaxes(title="Mean shift along Bella direction per generated token")
fig.update_yaxes(title="Bella voice gain (judge points)")
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
apply_theme(fig, height=460)
save_figure_bundle(fig, "gain_vs_dose", data=data2, root=HERE / "figures",
                   alt="Scatter of Bella voice gain against the per-token shift each cell delivers along the Bella direction; direction steers sit far right, all feature cells cluster near zero shift.")

# ---------------- Figure 3: Q3 cosine and fraction of gap closed, by layer
rows = list(csv.DictReader(open(AN / "q3_layers.csv")))
layers = [int(r["layer"]) for r in rows]
cosv = [float(r["cos"]) for r in rows]
fgc = [float(r["frac_gap_closed_test"]) for r in rows]
p95 = float(verd["Q3"]["random_dir_abs_cos_p95"])
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
fig.add_trace(go.Scatter(x=layers, y=cosv, mode="lines+markers", name="cos(SFT shift, Bella direction)", line=dict(color=C_DIR, width=3), marker=dict(size=6, color=C_DIR),
                         hovertemplate="layer %{x}<br>cos %{y:.3f}<extra></extra>", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=layers, y=fgc, mode="lines+markers", name="fraction of Bella-Gemma gap closed", line=dict(color=COLORS["Bella tokens"], width=3), marker=dict(size=6, color=COLORS["Bella tokens"]),
                         hovertemplate="layer %{x}<br>gap closed %{y:.1%}<extra></extra>", showlegend=False), row=2, col=1)
add_reference_line(fig, y=0.3, label="refute level 0.3", row=1, col=1)
add_reference_line(fig, y=p95, label=f"random direction |cos| p95: {p95:.3f}", row=1, col=1)
add_reference_line(fig, x=4, label="crest layer 4", row="all", col="all")
add_reference_line(fig, x=10, label="crest layer 10", row="all", col="all")
fig.update_xaxes(title="Layer (block output)", dtick=5, row=2, col=1)
fig.update_yaxes(title="Cosine similarity", range=[-0.05, 1.0], row=1, col=1)
fig.update_yaxes(title="Gap closed (test pairs)", range=[0, 0.5], tickformat=".0%", row=2, col=1)
apply_theme(fig, height=600)
save_figure_bundle(fig, "q3_cos_by_layer", data={"layer": layers, "cos": cosv, "frac_gap_closed_test": fgc, "random_abs_cos_p95": p95}, root=HERE / "figures",
                   alt="Two stacked line panels over 42 layers: cosine between the fine-tuning shift and the Bella direction on top, with the 0.3 refute level and random-direction reference; fraction of the Bella-Gemma gap closed by the fine-tune below; crest layers 4 and 10 marked in both.")

# ---------------- Figure 4: canaries for layer-10 cells (refusal and neutral-stem degeneration)
order = ["base", "dir:10:-0.35", "dir:10:-0.5"] + [f"A:10:{k}" for k in (5, 20, 50, 100, 200)] + [f"D:10:{k}" for k in (5, 20, 50, 100, 200)] + ["R:10:50", "R:10:100", "R:10:200", "amp:10:20:2", "amp:10:20:4"]
labels = {"base": "unsteered", "dir:10:-0.35": "dir −0.35", "dir:10:-0.5": "dir −0.5", "amp:10:20:2": "amp ×2", "amp:10:20:4": "amp ×4"}


def agg(c):
    if c.startswith("R:10:") and c.count(":") == 2:
        ts = [t for t in table.values() if t["kind"] == "R" and t["layer"] == 10 and t["k"] == int(c.split(":")[2])]
        return {k: sum(t[k] for t in ts) / len(ts) for k in ("refusal", "degenerate", "neutral_degenerate", "crisis")}, f"R k={c.split(':')[2]}"
    t = table[c]
    return {k: t[k] for k in ("refusal", "degenerate", "neutral_degenerate", "crisis")}, (labels[c] if c in labels else f"{c.split(':')[0]} k={c.split(':')[2]}")


vals, names, cols = [], [], []
for c in order:
    v, n = agg(c)
    vals.append(v); names.append(n)
    cols.append({"base": C_BASE, "dir": C_DIR, "A": C_A, "D": C_D, "R": C_R, "amp": C_AMP}[c.split(":")[0]])
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
fig.add_trace(go.Bar(x=names, y=[v["refusal"] for v in vals], marker_color=cols, name="refusal rate", showlegend=False, hovertemplate="%{x}<br>refusal %{y:.3f}<extra></extra>"), row=1, col=1)
fig.add_trace(go.Bar(x=names, y=[v["neutral_degenerate"] for v in vals], marker_color=cols, name="neutral-stem degeneration", showlegend=False, hovertemplate="%{x}<br>neutral degeneration %{y:.0%}<extra></extra>"), row=2, col=1)
add_reference_line(fig, y=table["base"]["refusal"] - 0.05, label="refusal floor (base − 5 pts)", row=1, col=1)
add_reference_line(fig, y=0.10, label="Q2 ceiling 10%", row=2, col=1)
fig.update_yaxes(title="Refusal rate (red-team)", range=[0, 1.05], row=1, col=1)
fig.update_yaxes(title="Neutral-stem degeneration", range=[0, 0.5], tickformat=".0%", row=2, col=1)
fig.update_xaxes(title="Layer-10 cell", tickangle=-45, row=2, col=1)
apply_theme(fig, height=600)
save_figure_bundle(fig, "canaries_l10", data=[{"cell": c, "label": n, **v} for c, n, v in zip(order, names, vals)], root=HERE / "figures",
                   alt="Two stacked bar panels over the layer-10 cells: red-team refusal rate on top and neutral-stem degeneration below, with the refusal floor and the 10 percent degeneration ceiling marked.")
print("done")
