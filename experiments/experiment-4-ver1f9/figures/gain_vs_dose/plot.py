"""Rebuild gain_vs_dose from the co-located data.json."""
import json
from pathlib import Path

import plotly.graph_objects as go
from silico_figures import add_reference_line, apply_theme, save_figure_bundle

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "data.json"))
G = {"dir": ("Direction steer", "#2E6E4E", "star", 13), "A": ("Assistant-ward clamp", "#988453", "circle", 10), "D": ("Gemma-over-Bella clamp", "#B9605B", "circle", 10),
     "amp": ("Bella-ward amplified", "#31362E", "square", 10), "R": ("Random matched clamp", "#7495AB", "diamond", 9)}
fig = go.Figure()
for kind, (name, col, sym, size) in G.items():
    pts = [p for p in D if p["kind"] == kind]
    e = dict(type="data", array=[p["hi"] - p["gain"] for p in pts], arrayminus=[p["gain"] - p["lo"] for p in pts], thickness=1)
    fig.add_trace(go.Scatter(x=[p["dose_along_dir"] for p in pts], y=[p["gain"] for p in pts], error_y=e, mode="markers", name=name,
                             marker=dict(color=col, size=size, symbol=[sym if p["canaries_intact"] else sym + "-open" for p in pts], line=dict(width=2, color=col)),
                             text=[p["cell"] for p in pts], hoverinfo="text+x+y"))
thr = 0.70 * next(p["gain"] for p in D if p["cell"] == "dir:10:-0.35")
add_reference_line(fig, y=thr, label=f"Q1 threshold: {thr:+.2f}")
add_reference_line(fig, x=0, label="no shift along direction")
fig.update_xaxes(title="Mean shift along Bella direction per generated token")
fig.update_yaxes(title="Bella voice gain (judge points)")
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
apply_theme(fig, height=460)
save_figure_bundle(fig, HERE.name, data=D, root=HERE.parent, alt=(HERE / "alt.txt").read_text().strip())
