"""Rebuild canaries_l10 from the co-located data.json."""
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "data.json"))
COL = {"base": "#84713A", "dir": "#2E6E4E", "A": "#988453", "D": "#B9605B", "R": "#7495AB", "amp": "#31362E"}
names = [d["label"] for d in D]
cols = [COL[d["cell"].split(":")[0]] for d in D]
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
fig.add_trace(go.Bar(x=names, y=[d["refusal"] for d in D], marker_color=cols, name="refusal rate", showlegend=False, hovertemplate="%{x}<br>refusal %{y:.3f}<extra></extra>"), row=1, col=1)
fig.add_trace(go.Bar(x=names, y=[d["neutral_degenerate"] for d in D], marker_color=cols, name="neutral-stem degeneration", showlegend=False, hovertemplate="%{x}<br>neutral degeneration %{y:.0%}<extra></extra>"), row=2, col=1)
base = next(d for d in D if d["cell"] == "base")
add_reference_line(fig, y=base["refusal"] - 0.05, label="refusal floor (base − 5 pts)", row=1, col=1)
add_reference_line(fig, y=0.10, label="Q2 ceiling 10%", row=2, col=1)
fig.update_yaxes(title="Refusal rate (red-team)", range=[0, 1.05], row=1, col=1)
fig.update_yaxes(title="Neutral-stem degeneration", range=[0, 0.5], tickformat=".0%", row=2, col=1)
fig.update_xaxes(title="Layer-10 cell", tickangle=-45, row=2, col=1)
apply_theme(fig, height=600)
save_figure_bundle(fig, HERE.name, data=D, root=HERE.parent, alt=(HERE / "alt.txt").read_text().strip())
