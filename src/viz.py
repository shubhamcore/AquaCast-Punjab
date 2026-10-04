"""
AquaCast-Punjab :: Visualisation layer
======================================

Every Plotly figure used by the dashboard, built here so ``app.py`` stays a
layout file.  All figures use a single dark "hydro" theme, transparent
backgrounds (Streamlit dark mode) and SI units with explicit axis titles —
a judge should never have to ask "is that metres or feet?".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import BRAND, HORIZONS, ZONE_CRITICAL_MAX, ZONE_SAFE_MAX

C = BRAND
GRID = "rgba(148,163,184,0.14)"
FONT = dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12,
            color="#CBD5E1")


def _layout(fig, title: str | None = None, height: int = 380,
            legend_top: bool = True, **kw):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,22,33,0.55)",
        font=FONT,
        height=height,
        margin=dict(l=52, r=20, t=48 if title else 24, b=44),
        title=dict(text=title or "", font=dict(size=14, color="#E2E8F0"), x=0.01),
        hoverlabel=dict(bgcolor="#0B1622", bordercolor=C["primary"],
                        font=dict(color="#E2E8F0")),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)) if legend_top else dict(bgcolor="rgba(0,0,0,0)"),
        **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def zone_bands(fig, y_lo: float, y_hi: float):
    """Shade the Safe / Critical / Over-Exploited depth bands."""
    lo = min(y_lo, ZONE_SAFE_MAX - 1)
    hi = max(y_hi, ZONE_CRITICAL_MAX + 2)
    fig.add_hrect(y0=lo, y1=ZONE_SAFE_MAX, fillcolor="#22C55E", opacity=0.09,
                  line_width=0, annotation_text="Safe", annotation_position="top left",
                  annotation=dict(font_size=10, font_color="#4ADE80"))
    fig.add_hrect(y0=ZONE_SAFE_MAX, y1=ZONE_CRITICAL_MAX, fillcolor="#F59E0B", opacity=0.09,
                  line_width=0, annotation_text="Critical", annotation_position="top left",
                  annotation=dict(font_size=10, font_color="#FBBF24"))
    fig.add_hrect(y0=ZONE_CRITICAL_MAX, y1=hi, fillcolor="#EF4444", opacity=0.09,
                  line_width=0, annotation_text="Over-exploited", annotation_position="top left",
                  annotation=dict(font_size=10, font_color="#F87171"))
    return fig


# ======================================================================================
# forecast chart
# ======================================================================================
def forecast_chart(df: pd.DataFrame, hist_tail: int = 420, issue_date=None,
                   fc_dates=None, fc_mean=None, fc_std=None,
                   observed: pd.DataFrame | None = None,
                   title: str = "Groundwater level — observations & AquaCast forecast"
                   ) -> go.Figure:
    d = df.copy()
    d["date"] = pd.to_datetime(d.date)
    d = d.tail(hist_tail)
    fig = go.Figure()
    y_lo = float(d.gw_level_mbgl.min()) - 0.6
    y_hi = float(d.gw_level_mbgl.max()) + 0.8

    fig.add_trace(go.Scatter(x=d.date, y=d.gw_level_mbgl, mode="lines",
                             name="Reconstructed water table",
                             line=dict(color=C["primary"], width=2.0),
                             hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} m bgl<extra></extra>"))
    # provenance shading: where the inputs are measured vs modelled
    meas = d[d.rainfall_src == "telemetry"]
    if len(meas):
        fig.add_trace(go.Scatter(x=meas.date, y=[d.gw_level_mbgl.min() - 0.35] * len(meas),
                                 mode="markers", name="Real rainfall telemetry",
                                 marker=dict(size=3, color="#22D3EE", symbol="line-ns-open",
                                             line=dict(width=1, color="#22D3EE")),
                                 hovertemplate="measured<extra></extra>"))

    if observed is not None and len(observed):
        fig.add_trace(go.Scatter(x=observed.date, y=observed.gw_level_mbgl, mode="markers",
                                 name="CGWB in-situ measurement",
                                 marker=dict(size=9, color="#FDE047", symbol="diamond",
                                             line=dict(width=1, color="#0F172A")),
                                 hovertemplate="<b>CGWB</b><br>%{x|%d %b %Y}<br>%{y:.2f} m bgl<extra></extra>"))

    if fc_dates is not None and fc_mean is not None:
        fd = list(pd.to_datetime(fc_dates))
        fm = np.asarray(fc_mean, dtype=float)
        fs = np.asarray(fc_std if fc_std is not None else np.zeros_like(fm), dtype=float)
        xs = [pd.Timestamp(issue_date)] + fd if issue_date is not None else fd
        ms = [float(d.gw_level_mbgl.iloc[-1])] + list(fm)
        ss = [0.0] + list(fs)
        up = np.array(ms) + 1.96 * np.array(ss)
        dn = np.array(ms) - 1.96 * np.array(ss)
        fig.add_trace(go.Scatter(x=list(xs) + list(xs)[::-1], y=list(up) + list(dn)[::-1],
                                 fill="toself", fillcolor="rgba(139,92,246,0.20)",
                                 line=dict(width=0), hoverinfo="skip",
                                 name="95 % confidence (MC-dropout)"))
        fig.add_trace(go.Scatter(x=xs, y=ms, mode="lines+markers", name="AquaCast LSTM forecast",
                                 line=dict(color=C["violet"], width=2.6, dash="dot"),
                                 marker=dict(size=9, color=C["violet"],
                                             line=dict(width=1.5, color="#0F172A")),
                                 hovertemplate="%{x|%d %b %Y}<br><b>%{y:.2f} m bgl</b><extra></extra>"))
        y_hi = max(y_hi, float(max(up)) + 0.4)

    fig.add_vline(x=pd.Timestamp(issue_date) if issue_date is not None else d.date.iloc[-1],
                  line=dict(color="#94A3B8", width=1, dash="dash"))
    fig.add_annotation(x=pd.Timestamp(issue_date) if issue_date is not None else d.date.iloc[-1],
                       y=1.02, yref="paper", text="today", showarrow=False,
                       font=dict(size=10, color="#94A3B8"))
    zone_bands(fig, y_lo, y_hi)
    fig.update_yaxes(title_text="Depth to water table (m bgl)", autorange="reversed")
    fig.update_xaxes(title_text="")
    return _layout(fig, title, height=430)


# ======================================================================================
# water budget
# ======================================================================================
def water_budget_chart(annual: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = annual["year"].astype(str)
    fig.add_trace(go.Bar(x=x, y=annual["draft_mcm"], name="Groundwater draft",
                         marker_color=C["danger"], opacity=0.9,
                         hovertemplate="Draft %{y:,.0f} MCM<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Bar(x=x, y=annual["recharge_mcm"], name="Natural + return recharge",
                         marker_color=C["primary"], opacity=0.9,
                         hovertemplate="Recharge %{y:,.0f} MCM<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=annual["deficit_mcm"], name="Deficit",
                             mode="lines+markers", line=dict(color="#FDE047", width=2.4),
                             hovertemplate="Deficit %{y:,.0f} MCM<extra></extra>"),
                  secondary_y=True)
    fig.update_yaxes(title_text="Volume (MCM)", secondary_y=False)
    fig.update_yaxes(title_text="Deficit (MCM)", secondary_y=True, showgrid=False)
    fig.update_layout(barmode="group", bargap=0.28)
    return _layout(fig, "Annual extraction vs recharge (district water budget)", height=330)


def seasonal_profile(df: pd.DataFrame) -> go.Figure:
    d = df.copy(); d["date"] = pd.to_datetime(d.date)
    m = d.groupby(d.date.dt.month).agg(
        level=("gw_level_mbgl", "mean"), draft=("gw_draft_mcm", "mean"),
        recharge=("total_recharge_mcm", "mean"), rain=("rainfall_mm", "sum"))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=m.index, y=m.draft, name="Mean daily draft (MCM)",
                         marker_color="#F87171", opacity=.85))
    fig.add_trace(go.Bar(x=m.index, y=m.recharge, name="Mean daily recharge (MCM)",
                         marker_color="#38BDF8", opacity=.85))
    fig.add_trace(go.Scatter(x=m.index, y=m.level, name="Mean water table (m bgl)",
                             line=dict(color="#FDE047", width=2.6)), secondary_y=True)
    fig.update_xaxes(tickmode="array",
                     tickvals=list(range(1, 13)),
                     ticktext=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    fig.update_yaxes(title_text="MCM / day", secondary_y=False)
    fig.update_yaxes(title_text="m bgl", secondary_y=True, autorange="reversed", showgrid=False)
    return _layout(fig, "Seasonal rhythm: Rabi drawdown → Kharif paddy → monsoon recovery",
                   height=330)


# ======================================================================================
# model performance
# ======================================================================================
def validation_chart(test_df: pd.DataFrame, issue, y_true, y_pred, horizon: int,
                     j: int) -> go.Figure:
    fig = go.Figure()
    x = pd.to_datetime(issue)
    fig.add_trace(go.Scatter(x=x, y=y_true[:, j], mode="lines", name="Actual (physics truth)",
                             line=dict(color="#38BDF8", width=2.0)))
    fig.add_trace(go.Scatter(x=x, y=y_pred[:, j], mode="lines", name=f"LSTM t+{horizon}",
                             line=dict(color=C["violet"], width=1.9, dash="dot")))
    fig.update_yaxes(title_text="Water table (m bgl)", autorange="reversed")
    return _layout(fig, f"Test-set validation — {horizon}-day horizon", height=300)


def scatter_skill(y_true, y_pred, horizon: int) -> go.Figure:
    fig = go.Figure()
    lo = float(min(y_true.min(), y_pred.min())) - .2
    hi = float(max(y_true.max(), y_pred.max())) + .2
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                             marker=dict(size=5, color=C["primary"], opacity=.55,
                                         line=dict(width=0)),
                             name="forecast"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="#FDE047", width=1.4, dash="dash"), name="1:1"))
    fig.update_xaxes(title_text="Actual (m bgl)")
    fig.update_yaxes(title_text="Predicted (m bgl)")
    return _layout(fig, f"Predicted vs actual — t+{horizon}", height=300)


def history_chart(history: list[dict]) -> go.Figure:
    h = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.epoch, y=h.train_rmse, mode="lines", name="Train RMSE",
                             line=dict(color=C["primary"], width=2)))
    fig.add_trace(go.Scatter(x=h.epoch, y=h.val_rmse, mode="lines", name="Validation RMSE",
                             line=dict(color=C["accent"], width=2)))
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="RMSE (m)")
    return _layout(fig, "Training curve", height=280)


def importance_chart(imp: pd.DataFrame) -> go.Figure:
    d = imp.sort_values("rmse_delta_m")
    colours = [C["danger"] if v > 0 else "#475569" for v in d.rmse_delta_m]
    fig = go.Figure(go.Bar(x=d.rmse_delta_m, y=d.feature, orientation="h",
                           marker_color=colours,
                           error_x=dict(type="data", array=d["std"], color="#94A3B8"),
                           hovertemplate="%{y}: +%{x:.3f} m RMSE when shuffled<extra></extra>"))
    fig.update_xaxes(title_text="Increase in test RMSE when the channel is shuffled (m)")
    return _layout(fig, "Permutation feature importance", height=330)


def saliency_chart(sal: pd.DataFrame, horizon: str = "t+90") -> go.Figure:
    d = sal[sal.horizon == horizon]
    feats = [c for c in d.columns if c not in ("horizon", "day_offset")]
    fig = go.Figure()
    for f in feats:
        fig.add_trace(go.Scatter(x=d.day_offset, y=d[f], mode="lines", name=f,
                                 stackgroup="one", line=dict(width=.6)))
    fig.update_xaxes(title_text="Days before the issue date")
    fig.update_yaxes(title_text="|∂ forecast / ∂ input| (share)")
    return _layout(fig, f"Which days drove the {horizon} forecast", height=300)


benchmark_bar = lambda b: _layout(
    go.Figure(go.Bar(x=b.rmse_m, y=b.model, orientation="h",
                     marker_color=[C["violet"]] + ["#334155"] * (len(b) - 1),
                     text=[f"{v:.3f} m" for v in b.rmse_m], textposition="auto",
                     textfont=dict(color="#E2E8F0", size=11))),
    "Model benchmark — lower is better", height=280).update_xaxes(title_text="Test RMSE (m)")


def risk_gauge(score: float, band_colour: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=score,
        number=dict(font=dict(size=40, color="#E2E8F0"), suffix=""),
        delta=dict(reference=50, valueformat=".1f", font=dict(size=12)),
        gauge=dict(axis=dict(range=[0, 100], tickwidth=1, tickcolor="#94A3B8",
                             tickfont=dict(size=9, color="#94A3B8")),
                   bar=dict(color=band_colour, thickness=0.72),
                   bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, 25], color="rgba(34,197,94,.16)"),
                          dict(range=[25, 45], color="rgba(132,204,22,.16)"),
                          dict(range=[45, 62], color="rgba(245,158,11,.16)"),
                          dict(range=[62, 78], color="rgba(249,115,22,.16)"),
                          dict(range=[78, 100], color="rgba(239,68,68,.16)")])))
    return _layout(fig, height=230).update_layout(margin=dict(l=24, r=24, t=24, b=12))


def risk_components(components: dict, weights: dict) -> go.Figure:
    keys = [k for k in components]
    fig = go.Figure(go.Bar(
        x=[k.capitalize() for k in keys],
        y=[components[k] * abs(weights.get(k, 0)) for k in keys],
        marker_color=[C["danger"] if weights.get(k, 0) > 0 else C["accent"] for k in keys],
        text=[f"{components[k]:.0f}" for k in keys], textposition="auto",
        textfont=dict(color="#E2E8F0", size=10)))
    fig.update_yaxes(title_text="Weighted contribution to the 1–100 score")
    return _layout(fig, "Risk score decomposition", height=300)
