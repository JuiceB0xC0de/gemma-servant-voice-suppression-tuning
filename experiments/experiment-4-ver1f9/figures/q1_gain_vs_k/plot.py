"""Rebuild q1_gain_vs_k from the co-located data.json."""
import json
from pathlib import Path

import plotly.graph_objects as go
from silico_figures import add_reference_line, apply_theme, save_figure_bundle

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "data.json"))
COL = {"Assistant-ward, layer 10": "#988453", "Gemma-over-Bella, layer 10": "#B9605B", "Random matched, layer 10": "#7495AB", "Bella-ward amplified (k=20)": "#31362E"}
LEGEND = {"Bella-ward amplified (k=20)": "Bella-ward amplified ×2, ×4 (k=20)"}
SYM = {"Random matched, layer 10": "diamond", "Bella-ward amplified (k=20)": "square"}
fig = go.Figure()
for name, pts in D["series"].items():
    e = dict(type="data", array=[p["hi"] - p["gain"] for p in pts], arrayminus=[p["gain"] - p["lo"] for p in pts], thickness=1)
    line = name in ("Assistant-ward, layer 10", "Gemma-over-Bella, layer 10")
    fig.add_trace(go.Scatter(x=[p["k"] for p in pts], y=[p["gain"] for p in pts], error_y=e, mode="lines+markers" if line else "markers", name=LEGEND.get(name, name),
                             line=dict(color=COL[name], width=3 if "Assistant" in name else 2),
                             marker=dict(color=COL[name], size=9, symbol=[("circle" if p.get("canaries_intact", True) else "circle-open") if line else SYM[name] for p in pts], line=dict(width=2, color=COL[name]))))
add_reference_line(fig, y=D["dir_-0.35_gain"], label=f"direction −0.35: {D['dir_-0.35_gain']:+.2f}")
add_reference_line(fig, y=D["threshold"], label=f"Q1 threshold: {D['threshold']:+.2f}")
add_reference_line(fig, y=0, label="unsteered")
fig.update_xaxes(type="log", title="Features clamped, k", tickvals=[5, 20, 50, 100, 200], ticktext=["5", "20", "50", "100", "200"])
fig.update_yaxes(title="Bella voice gain (judge points)", range=[-0.4, 2.1])
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
apply_theme(fig, height=460)
save_figure_bundle(fig, HERE.name, data=D, root=HERE.parent, alt=(HERE / "alt.txt").read_text().strip())
