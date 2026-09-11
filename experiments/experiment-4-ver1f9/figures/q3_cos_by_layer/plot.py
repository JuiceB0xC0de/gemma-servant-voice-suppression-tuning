"""Rebuild q3_cos_by_layer from the co-located data.json."""
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "data.json"))
C_DIR, C_BELLA = "#2E6E4E", "#C4650D"
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
fig.add_trace(go.Scatter(x=D["layer"], y=D["cos"], mode="lines+markers", name="cos(SFT shift, Bella direction)", line=dict(color=C_DIR, width=3), marker=dict(size=6, color=C_DIR),
                         hovertemplate="layer %{x}<br>cos %{y:.3f}<extra></extra>", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=D["layer"], y=D["frac_gap_closed_test"], mode="lines+markers", name="fraction of Bella-Gemma gap closed", line=dict(color=C_BELLA, width=3), marker=dict(size=6, color=C_BELLA),
                         hovertemplate="layer %{x}<br>gap closed %{y:.1%}<extra></extra>", showlegend=False), row=2, col=1)
add_reference_line(fig, y=0.3, label="refute level 0.3", row=1, col=1)
add_reference_line(fig, y=D["random_abs_cos_p95"], label=f"random direction |cos| p95: {D['random_abs_cos_p95']:.3f}", row=1, col=1)
add_reference_line(fig, x=4, label="crest layer 4", row="all", col="all")
add_reference_line(fig, x=10, label="crest layer 10", row="all", col="all")
fig.update_xaxes(title="Layer (block output)", dtick=5, row=2, col=1)
fig.update_yaxes(title="Cosine similarity", range=[-0.05, 1.0], row=1, col=1)
fig.update_yaxes(title="Gap closed (test pairs)", range=[0, 0.5], tickformat=".0%", row=2, col=1)
apply_theme(fig, height=600)
save_figure_bundle(fig, HERE.name, data=D, root=HERE.parent, alt=(HERE / "alt.txt").read_text().strip())
