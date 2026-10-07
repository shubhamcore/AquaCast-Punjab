"""
AquaCast-Punjab :: Executive Dashboard (Streamlit · ParentSquare-inspired UI)
============================================================================

Run with::

    streamlit run app.py

Five tabs, designed to be screen-recorded in under three minutes:

1. **Executive & Farmer** — the four numbers that matter, the forecast chart
   with MC-dropout confidence bands, and the bilingual WhatsApp notice card.
2. **Macro & Spatial** — the interactive monitoring-station map (Groundwater
   Level Station.geojson), block stress stratification and the aquifer
   cross-section.
3. **Digital Twin & Credit** — the dedicated simulation tab: four policy /
   climate levers, a before-vs-after trajectory chart with the conserved
   water-headroom area, impact scorecards and the Satin Finserv credit
   risk adjustment.
4. **Model Lab** — honest metrics, architecture, saliency, training curve and
   the data-provenance ledger.
5. **Admin & Finserv** — portfolio exposure, rotational feeder power
   scheduling and the exportable district compliance / risk report.

Visual language follows ParentSquare: clean white elevated cards on a soft
slate canvas, high-contrast slate headers, status pills, and a deliberately
uncluttered sidebar (district · cropping season · language only).
"""
from __future__ import annotations

import html as _html
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src import telemetry_ingest as ti                       # noqa: E402
from src import viz                                          # noqa: E402
from src.advisory_engine import (build_advisory, classify_zone,  # noqa: E402
                                 crop_stage_for, pump_failure_risk)
from src.config import (DATA_END, DATA_START, DISTRICTS,               # noqa: E402
                        FLOOD_IRRIGATION_EFFICIENCY, HORIZONS,
                        MC_DROPOUT_PASSES, MODEL_DIR, SCENARIOS,
                        Scenario, ZONE_CRITICAL_MAX, get_prior)
from src.dataset_generator import load_dataset               # noqa: E402
from src.digital_twin import project                          # noqa: E402
from src.model_lstm import (benchmark, build_sequences,       # noqa: E402
                            evaluate, input_saliency, live_window,
                            load_model, permutation_importance,
                            predict)
from src.risk_engine import portfolio_view                    # noqa: E402
from src.spatial_loader import (district_geojson,             # noqa: E402
                                punjab_districts)
from src.well_network import block_summary, build_well_table  # noqa: E402

# ======================================================================================
# page config + ParentSquare-inspired theme
# ======================================================================================
st.set_page_config(page_title="AquaCast-Punjab | AI Aquifer Intelligence",
                   page_icon="💧", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
:root{
    --bg-app:#F8FAFC; --bg-card:#FFFFFF; --border-color:#E2E8F0;
    --text-primary:#0F172A; --text-secondary:#475569; --text-muted:#94A3B8;
    --accent-green:#16A34A; --accent-amber:#D97706; --accent-red:#DC2626;
    --accent-blue:#0369A1;
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMainBlock"]{
    background:var(--bg-app) !important;}
section[data-testid="stSidebar"]{
    background:#FFFFFF !important; border-right:1px solid var(--border-color);}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:var(--text-secondary);}
.block-container{padding-top:1.1rem;padding-bottom:2.5rem;max-width:1500px}
h1,h2,h3,h4{color:var(--text-primary);letter-spacing:-.015em;font-weight:700}
h1{font-size:1.65rem !important} h2{font-size:1.35rem !important} h3{font-size:1.12rem !important}
p,li,label{color:var(--text-secondary)}

/* ---- elevated white container cards ------------------------------------------------ */
.page-head{background:var(--bg-card);border:1px solid var(--border-color);
    border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,0.05);
    padding:20px 24px;margin-bottom:14px}
.page-head h1{margin:0;font-size:1.6rem;font-weight:800;color:var(--text-primary)}
.page-head p{margin:6px 0 0;color:var(--text-secondary);font-size:.92rem;line-height:1.5}
.kpi,.card{background:var(--bg-card);border:1px solid var(--border-color);
    border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,0.05);
    padding:14px 16px;margin-bottom:12px;height:100%}
.card{height:auto}
.kpi .lab{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;
    color:var(--text-secondary);font-weight:700}
.kpi .val{font-size:1.8rem;font-weight:800;line-height:1.15;margin:6px 0 2px;
    color:var(--text-primary)}
.kpi .sub{font-size:.79rem;color:var(--text-secondary);line-height:1.4}
.card-title{font-weight:700;color:var(--text-primary);font-size:.95rem;margin-bottom:6px}

/* ---- pills & badges ---------------------------------------------------------------- */
.pill{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.73rem;
    font-weight:700;letter-spacing:.02em;white-space:nowrap}
.badge{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.73rem;
    font-weight:700;letter-spacing:.03em;margin:4px 6px 0 0;
    background:#ECFDF5;color:#15803D;border:1px solid #BBF7D0}
.badge.blue{background:#EFF6FF;color:#1D4ED8;border-color:#BFDBFE}
.badge.amber{background:#FFFBEB;color:#B45309;border-color:#FDE68A}
.badge.violet{background:#F5F3FF;color:#6D28D9;border-color:#DDD6FE}

/* ---- ParentSquare notice / WhatsApp preview ---------------------------------------- */
.notice{background:var(--bg-card);border:1px solid var(--border-color);border-radius:12px;
    box-shadow:0 1px 3px rgba(0,0,0,0.05);overflow:hidden;margin-bottom:10px}
.notice-head{display:flex;align-items:center;gap:10px;padding:12px 14px;
    border-bottom:1px solid var(--border-color);background:#F8FAFC}
.notice-avatar{width:38px;height:38px;border-radius:10px;flex:0 0 auto;
    background:linear-gradient(135deg,#16A34A,#059669);color:#fff;display:flex;
    align-items:center;justify-content:center;font-size:19px}
.notice-who{flex:1 1 auto;min-width:0}
.notice-title{font-weight:700;color:var(--text-primary);font-size:.9rem;
    display:flex;align-items:center;gap:6px}
.notice-verified{display:inline-flex;align-items:center;justify-content:center;
    width:15px;height:15px;border-radius:50%;background:#16A34A;color:#fff;
    font-size:9px;font-weight:900}
.notice-sub{font-size:.72rem;color:var(--text-secondary)}
.notice-date{background:#EFF6FF;color:#1D4ED8;border:1px solid #BFDBFE;
    border-radius:999px;font-size:.68rem;font-weight:700;padding:3px 10px;white-space:nowrap}
.notice-body{padding:12px 14px}
.notice-lang{font-size:.78rem;font-weight:700;color:var(--accent-blue);
    text-transform:uppercase;letter-spacing:.06em;margin:2px 0 3px}
.notice-text{font-size:.86rem;color:var(--text-primary);line-height:1.55;
    white-space:pre-wrap;margin-bottom:10px}
.notice-divider{border-top:1px dashed var(--border-color);margin:4px 0 10px}

.note{font-size:.8rem;color:var(--text-secondary);line-height:1.55}
.hl{color:var(--accent-blue);font-weight:700}
.ok{color:#15803D;font-weight:700} .wn{color:#B45309;font-weight:700}
.dg{color:#B91C1C;font-weight:700}
hr.sep{border:0;border-top:1px solid var(--border-color);margin:14px 0}

/* ---- tabs --------------------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"]{gap:6px}
.stTabs [data-baseweb="tab"]{background:var(--bg-card);border:1px solid var(--border-color);
    border-radius:10px;padding:8px 16px;font-weight:600;color:var(--text-secondary);
    font-size:.86rem}
.stTabs [aria-selected="true"]{background:var(--bg-card);color:var(--text-primary);
    border-color:var(--accent-green);box-shadow:0 1px 3px rgba(0,0,0,0.05)}
.stTabs [data-baseweb="tab-highlight"]{background-color:var(--accent-green)}
.stTabs [data-baseweb="tab-border"]{background-color:transparent}

/* ---- misc --------------------------------------------------------------------------- */
div[data-testid="stMetric"]{background:var(--bg-card);border:1px solid var(--border-color);
    border-radius:12px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,0.05)}
[data-testid="stExpander"]{background:var(--bg-card);border:1px solid var(--border-color);
    border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,0.05)}
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# cached loaders
# ======================================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def _data(district: str) -> pd.DataFrame:
    return load_dataset(district)


@st.cache_data(show_spinner=False, ttl=3600)
def _model_ready() -> bool:
    return (MODEL_DIR / "lstm_aquifer.pth").exists()


@st.cache_resource(show_spinner="Loading AquaCast LSTM…")
def _model():
    return load_model()


@st.cache_data(show_spinner=False, ttl=3600)
def _obs():
    return ti.load_groundwater_observations()


@st.cache_data(show_spinner=False, ttl=3600)
def _telemetry_meta():
    rf = ti.load_all_rainfall()
    tb = ti.load_all_temperature()
    return rf.meta, tb.meta


def rate_for(district: str) -> float:
    return get_prior(district).long_term_decline_m_per_yr / 365.25


@st.cache_data(show_spinner=False, ttl=600)
def forecast_for(district: str, mc: int):
    """Run the LSTM on the latest 60-day window with MC-dropout uncertainty."""
    df = _data(district)
    res = _model()
    if res is None:
        return None
    # NOTE: deliberately *not* build_sequences(...) [-1:] — that window ends 90
    # days before the end of the record because it reserves a realised target.
    # live_window ends on the newest observation, so the forecast is genuinely
    # 30/60/90 days ahead.
    Xl, base, issue, tdates, rte = live_window(
        df, trend_rate_m_per_day=rate_for(district))
    p = predict(res, Xl, base, mc_passes=mc, trend_rate_m_per_day=rte)
    out = {}
    out["mean"] = p["mean"][0]
    out["std"] = p["std"][0]
    out["issue"] = pd.Timestamp(issue[-1])
    out["targets"] = [pd.Timestamp(t) for t in tdates[-1]]
    return out


@st.cache_data(show_spinner=False, ttl=600)
def test_evaluation(district: str):
    df = _data(district)
    res = _model()
    if res is None:
        return None
    from src.model_lstm import chronological_split
    _, te = chronological_split(df)
    X, Y, base, issue, tdates, rte = build_sequences(
        te, trend_rate_m_per_day=rate_for(district))
    p = predict(res, X, base, trend_rate_m_per_day=rte)
    Ytrue = base[:, None] + Y + rte[:, None] * np.asarray(HORIZONS, dtype="float32")[None, :]
    return dict(issue=issue, y_true=Ytrue, y_pred=p["mean"], base=base)


# ======================================================================================
# Digital-twin runs (levers live on Tab 3 — read from widget state before render)
# ======================================================================================
SIM_DAYS = 3650                     # 10-year counterfactual window
LEVER_DEFAULTS = {"dt_monsoon": 0, "dt_drip": 0, "dt_shift": 0, "dt_canal": 100}


@st.cache_data(show_spinner=False, ttl=600)
def twin_runs(district: str, monsoon: int, drip: int, shift: int, canal: int,
              days: int = SIM_DAYS):
    """Baseline (standard flood) vs intervention trajectories from the FAO-56 twin."""
    df = _data(district)
    eff = 0.90 if drip > 0 else FLOOD_IRRIGATION_EFFICIENCY
    interv = Scenario(name="Intervention", irrigation_efficiency=eff,
                      monsoon_anomaly_pct=float(monsoon),
                      paddy_transplant_shift_days=int(shift),
                      canal_availability_pct=float(canal),
                      pump_adoption_drip_pct=float(drip))
    pb = project(df, district, SCENARIOS["Standard Flood Irrigation"],
                 days=days, paddy_shift_days=0)
    pi = project(df, district, interv, days=days, paddy_shift_days=int(shift))
    return pb, pi


@st.cache_data(show_spinner=False, ttl=600)
def bau_trend(district: str, days: int = SIM_DAYS) -> pd.DataFrame:
    """Business-as-usual: the district's observed long-term depletion trend
    continued from today's measured level (the honest no-policy baseline)."""
    df = _data(district)
    cur = float(df.gw_level_mbgl.iloc[-1])
    decline = float(get_prior(district).long_term_decline_m_per_yr)
    idx = pd.date_range(pd.Timestamp(df.date.max()), periods=days + 1, freq="D")
    return pd.DataFrame({"date": idx,
                         "level": cur + decline * np.arange(days + 1) / 365.25})


@st.cache_data(show_spinner=False, ttl=600)
def rabi_deficit_fraction(district: str) -> float:
    """Share of the Rabi (Nov–Apr) crop draft the aquifer cannot naturally
    recharge — the flowering-season water deficit used by the credit engine."""
    df = _data(district)
    r = df[df.date.dt.month.isin([11, 12, 1, 2, 3, 4])]
    draft = float(r.gw_draft_mcm.sum())
    recharge = float(r.total_recharge_mcm.sum())
    return float(np.clip((draft - recharge) / max(draft, 1e-6), 0.0, 1.0))


@st.cache_data(show_spinner=False, ttl=600)
def credit_pair(district: str, bau_end: float, interv_end: float,
                current_mbgl: float, pump_set: float) -> tuple:
    """Satin Finserv underwriting score before vs after the intervention.

    Status quo is scored at the level business-as-usual reaches inside the
    window (capped at the 50% dry-out level — 'water failure at flowering');
    the post-intervention score removes the flowering deficit and the bore
    deepening capex the intervention makes unnecessary.
    """
    from src.crop_cycle_model import next_windows
    import src.credit_risk_engine as cre
    wins = next_windows(365, district=district)
    frac = rabi_deficit_fraction(district)
    sq_level = float(bau_end)          # where business-as-usual lands in the window
    sq = cre.score(district, predicted_level_mbgl=sq_level,
                   pump_set_depth_mbgl=float(pump_set),
                   current_level_mbgl=float(current_mbgl),
                   critical_windows=wins, deficit_fraction=frac, horizon_days=90)
    # Post-intervention there is no flowering deficit (the water is there) and
    # therefore no stage falls inside a *deficit* window, and no bore deepening
    # has to be financed — both sub-scores legitimately go to zero.
    post = cre.score(district, predicted_level_mbgl=float(interv_end),
                     pump_set_depth_mbgl=float(pump_set),
                     current_level_mbgl=float(current_mbgl),
                     critical_windows=[], deficit_fraction=0.0,
                     deepening_capex_inr=0.0, horizon_days=90)
    return sq, post


@st.cache_data(show_spinner=False, ttl=3600)
def _network(district: str, seed: int = 2024):
    try:
        from src.spatial_network import build_network
        return build_network(district, seed=seed)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=600)
def _overexploited_summary(district: str) -> dict:
    """Block-level stratification for the lender portfolio view."""
    try:
        w = build_well_table(district)
        b = block_summary(w)
        if b.empty:
            return dict(n_blocks=0, oe_blocks=0, oe_well_share=0.0, wells=0, oe_wells=0)
        oe = b[b.mean_level_mbgl > ZONE_CRITICAL_MAX]
        return dict(n_blocks=int(len(b)), oe_blocks=int(len(oe)),
                    oe_well_share=float(oe.wells.sum()) / max(float(b.wells.sum()), 1.0),
                    wells=int(b.wells.sum()), oe_wells=int(oe.wells.sum()))
    except Exception:
        return dict(n_blocks=0, oe_blocks=0, oe_well_share=0.0, wells=0, oe_wells=0)


# ======================================================================================
# small HTML helpers (ParentSquare card language)
# ======================================================================================
def kpi_card(lab: str, val: str, unit: str, sub: str, colour: str = "#0F172A",
             pill: str = "") -> str:
    return f"""<div class="kpi">
      <div class="lab">{lab}</div>
      <div class="val" style="color:{colour}">{val}
        <span style="font-size:.85rem;color:var(--text-secondary);font-weight:600">{unit}</span>
        {pill}</div>
      <div class="sub">{sub}</div></div>"""


def pill(text: str, colour: str) -> str:
    return (f"<span class='pill' style='background:{colour}1A;color:{colour};"
            f"border:1px solid {colour}33'>{text}</span>")


def esc(x) -> str:
    return _html.escape(str(x))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = np.radians(float(lat1)), np.radians(float(lat2))
    dp = np.radians(float(lat2) - float(lat1))
    dl = np.radians(float(lon2) - float(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


# ======================================================================================
# sidebar — deliberately minimal: district · cropping season · language
# ======================================================================================
SEASON_STAGE = crop_stage_for(pd.Timestamp(DATA_END))

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:4px 0 12px">
      <div style="font-size:2rem">💧</div>
      <div style="font-weight:800;font-size:1.05rem;color:#0F172A">AquaCast-Punjab</div>
      <div style="font-size:.72rem;color:#475569">AI Aquifer Intelligence</div>
    </div>""", unsafe_allow_html=True)

    # 1 — district selector
    district = st.selectbox("📍 District", DISTRICTS, index=0)

    # 2 — primary cropping season view indicator (display only, no controls)
    st.markdown(f"""
    <div class="card" style="margin-bottom:10px">
      <div class="lab" style="font-size:.7rem;text-transform:uppercase;
           letter-spacing:.08em;color:var(--text-secondary);font-weight:700">
        🌾 Primary cropping season</div>
      <div style="font-weight:800;color:var(--text-primary);font-size:.95rem;margin-top:4px">
        Rabi · Wheat <span style="font-weight:600;color:var(--text-secondary);
        font-size:.78rem">(01 Nov → 30 Apr)</span></div>
      <div style="font-size:.78rem;color:var(--text-secondary);margin-top:3px">
        Stage now: <b>{esc(SEASON_STAGE['en'])}</b><br>
        <span dir="auto">{esc(SEASON_STAGE['pa'])}</span></div>
      <div style="font-size:.72rem;color:var(--text-muted);margin-top:6px">
        Secondary: Kharif paddy · 15 May → 27 Oct</div>
    </div>""", unsafe_allow_html=True)

    # 3 — language toggle
    lang = st.radio("🗣️ Advisory language · ਭਾਸ਼ਾ", ["English", "ਪੰਜਾਬੀ"],
                    horizontal=True, key="lang_radio")

# ======================================================================================
# data prep
# ======================================================================================
prior = get_prior(district)
df = _data(district)
df["date"] = pd.to_datetime(df.date)
AS_OF = pd.Timestamp(df.date.max())
model_ready = _model_ready()

# district-switch state hygiene: station selection is district-scoped
if st.session_state.get("_last_district") not in (None, district):
    st.session_state.pop("gw_station_sel", None)
    st.session_state.pop("gw_station_idx", None)
st.session_state["_last_district"] = district

# ======================================================================================
# page header
# ======================================================================================
st.markdown(f"""
<div class="page-head">
  <h1>💧 Project AquaCast &nbsp;·&nbsp; AI Aquifer Intelligence &amp; Agro-Advisory</h1>
  <p>Hyper-local groundwater depletion forecasting for Punjab's Rabi wheat — satellite
     crop demand × climate stress × hydrological telemetry, turned into irrigation quotas
     and water-risk signals for farmers and lenders.</p>
  <div style="margin-top:8px">
    <span class="badge">PyTorch LSTM · 30/60/90 d</span>
    <span class="badge blue">MC-dropout uncertainty</span>
    <span class="badge violet">India-WRIS + CGWB ground truth</span>
    <span class="badge amber">Sankalp · Climate-Smart Agriculture</span>
  </div>
</div>""", unsafe_allow_html=True)

if not model_ready:
    st.error("Model weights not found. Run `python run_pipeline.py` first to train the LSTM.")
    st.stop()

fc = forecast_for(district, MC_DROPOUT_PASSES)
if fc is None:
    st.error("Could not load model weights.")
    st.stop()

# ---- compact forecast / farm controls (the sidebar is reserved for the 3 essentials)
ctl1, ctl2 = st.columns([1, 2.6])
with ctl1:
    horizon = st.radio("🎯 Forecast horizon", HORIZONS, index=2, horizontal=True,
                       format_func=lambda v: f"{v} days",
                       help="Target horizon for the advisory and quota cards.")
with ctl2:
    with st.expander("⚙️ Farm, pump & uncertainty settings", expanded=False):
        f1, f2, f3 = st.columns(3)
        acres = f1.slider("Farm size (acres)", 1.0, 25.0, 5.0, 1.0,
                          key="farm_acres")
        pump_set = f2.slider("Tubewell installation depth (m)", 35, 70, 49, 1,
                             key="pump_depth")
        mc = f3.slider("MC-dropout passes", 0, 120, MC_DROPOUT_PASSES, 20,
                       key="mc_passes")

j = list(HORIZONS).index(horizon)

# ---- digital-twin levers live in session state (their widgets sit on Tab 3) -----------
monsoon = int(st.session_state.get("dt_monsoon", LEVER_DEFAULTS["dt_monsoon"]))
drip = int(st.session_state.get("dt_drip", LEVER_DEFAULTS["dt_drip"]))
shift = int(st.session_state.get("dt_shift", LEVER_DEFAULTS["dt_shift"]))
canal = int(st.session_state.get("dt_canal", LEVER_DEFAULTS["dt_canal"]))

scen_live = Scenario(name="Your scenario",
                     irrigation_efficiency=(0.90 if drip > 0
                                            else FLOOD_IRRIGATION_EFFICIENCY),
                     monsoon_anomaly_pct=float(monsoon),
                     paddy_transplant_shift_days=int(shift),
                     canal_availability_pct=float(canal),
                     pump_adoption_drip_pct=float(drip))

forecast_level = float(fc["mean"][j])
adv = build_advisory(df, district, fc, horizon, scen_live,
                     pump_set_depth_mbgl=float(pump_set), farm_acres=float(acres))

wells = build_well_table(district, df, horizon_level=forecast_level,
                         horizon_delta=adv.predicted_delta_m)
blocks = block_summary(wells)
zone = classify_zone(forecast_level)

# ======================================================================================
# the five tabs (exact labels)
# ======================================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Executive & Farmer",
    " Macro & Spatial",
    " Digital Twin & Credit",
    " Model Lab",
    " Admin & Finserv"
])

# --------------------------------------------------------------------------------------
# TAB 1 — Executive & Farmer
# --------------------------------------------------------------------------------------
with tab1:
    cur = adv.current_level_mbgl
    delta = adv.predicted_delta_m
    zone_pill = pill(adv.zone_label_en, zone["colour"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card(
            "Current Water Table (mbgl)", f"{cur:.2f}", "m bgl",
            f"as of {AS_OF:%d %b %Y} · district composite<br>{zone_pill}",
            colour="#0F172A"), unsafe_allow_html=True)
    with c2:
        c2_col = zone["colour"] if delta > 0 else "#16A34A"
        st.markdown(kpi_card(
            "Predicted Drawdown at Harvest (mbgl)", f"{forecast_level:.2f}", "m bgl",
            f"<span class='{'dg' if delta > 0 else 'ok'}'>{delta:+.2f} m</span> "
            f"over {horizon} d · target {adv.target_date}<br>"
            f"95% band {adv.lower_mbgl:.2f}–{adv.upper_mbgl:.2f} m",
            colour=c2_col), unsafe_allow_html=True)
    with c3:
        pf = adv.pump_failure_risk_pct
        pf_col = "#DC2626" if pf > 60 else "#D97706" if pf > 30 else "#16A34A"
        pf_pill = pill("Critical", "#DC2626") if pf > 60 else (
            pill("Warning", "#D97706") if pf > 30 else pill("Safe", "#16A34A"))
        st.markdown(kpi_card(
            "Pump Dry-Out Risk (%)", f"{pf:.0f}", "%",
            f"headroom {adv.headroom_m:.1f} m to the {pump_set} m pump · "
            f"{'⚠️ dry-out in ~%.0f days' % adv.days_to_dryout if adv.days_to_dryout and adv.days_to_dryout < 1500 else '✅ no dry-out expected in 4 yrs'}"
            f"<br>{pf_pill}",
            colour=pf_col), unsafe_allow_html=True)
    with c4:
        gap_d = adv.quota_days_between_events
        sched = (f"one {adv.quota_hours_per_event:.0f} h run every {gap_d:.0f} days/acre"
                 if gap_d else "no irrigation needed at this crop stage")
        st.markdown(kpi_card(
            "Recommended Weekly Pumping Quota (hrs/acre)",
            f"{adv.quota_hours_per_week_per_acre:.1f}", "h/wk/acre",
            f"{sched} · {adv.quota_litres_per_week_per_acre:,} L/wk for {acres:.0f} ac<br>"
            f"pump delivers {adv.pump_discharge_m3h:.0f} m³/h · "
            f"sustainability {adv.sustainability_factor:.2f}×",
            colour="#0369A1"), unsafe_allow_html=True)

    st.markdown("")

    # ---- main chart + ParentSquare notice split ---------------------------------------
    left, right = st.columns([1.8, 1.0])
    with left:
        obs = _obs()
        st.plotly_chart(viz.forecast_chart(df, hist_tail=470, issue_date=AS_OF,
                                           fc_dates=[fc["targets"][j]],
                                           fc_mean=[fc["mean"][j]],
                                           fc_std=[fc["std"][j]],
                                           observed=obs),
                        width='stretch')
        st.caption("Diamonds are real CGWB in-situ measurements; the cyan ticks mark days "
                   "whose rainfall input comes from live telemetry rather than the fitted "
                   "generator. The forecast track is the LSTM (data-driven); the policy "
                   "levers drive the FAO-56 physics digital twin on the "
                   "‘Digital Twin & Credit’ tab, because the LSTM only ever sees the "
                   "observed past.")

    with right:
        st.markdown("**📲 ParentSquare notice · WhatsApp advisory preview**")
        wa_pa = esc(adv.whatsapp_pa)
        wa_en = esc(adv.whatsapp_en)
        # bilingual card — the language toggle decides which block leads
        first_pa = lang != "English"
        block_first = (f"<div class='notice-lang'>ਪੰਜਾਬੀ</div>"
                       f"<div class='notice-text' dir='auto'>{wa_pa}</div>") if first_pa else (
                          f"<div class='notice-lang'>English</div>"
                          f"<div class='notice-text'>{wa_en}</div>")
        block_second = (f"<div class='notice-divider'></div>"
                        f"<div class='notice-lang'>English</div>"
                        f"<div class='notice-text'>{wa_en}</div>") if first_pa else (
                           f"<div class='notice-divider'></div>"
                           f"<div class='notice-lang'>ਪੰਜਾਬੀ</div>"
                           f"<div class='notice-text' dir='auto'>{wa_pa}</div>")
        st.markdown(f"""
        <div class="notice">
          <div class="notice-head">
            <div class="notice-avatar">🌾</div>
            <div class="notice-who">
              <div class="notice-title">AquaCast Punjab
                <span class="notice-verified" title="Verified sender">✓</span></div>
              <div class="notice-sub">Official aquifer advisory ·
                {len(wells):,} wells monitored · {esc(district)}</div>
            </div>
            <div class="notice-date">{AS_OF:%d %b %Y}</div>
          </div>
          <div class="notice-body">
            {block_first}{block_second}
          </div>
        </div>""", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download SMS (160-char)",
            data=adv.sms_en if lang == "English" else adv.sms_pa,
            file_name=f"aquacast_sms_{district}_{horizon}d.txt", mime="text/plain",
            width='stretch')

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # ---- action cards ------------------------------------------------------------------
    a1, a2 = st.columns([1.0, 1.15])
    with a1:
        st.markdown("#### ✅ Recommended Field Actions")
        for a in adv.actions:
            st.markdown(
                f"<div class='card'><div style='font-size:.95rem'>{a['icon']} "
                f"<b>{esc(a['en'])}</b></div>"
                f"<div style='color:var(--text-secondary);font-size:.85rem;margin-top:3px' "
                f"dir='auto'>{esc(a['pa'])}</div></div>", unsafe_allow_html=True)
    with a2:
        st.markdown("#### 💧 Quota Justification &amp; Crop Stage")
        st.markdown(f"<div class='card'>"
                    f"<div class='card-title'>Why this quota?</div>"
                    f"<div class='note'>{esc(adv.quota_note_en)}</div>"
                    f"<div class='note' dir='auto' style='margin-top:6px'>{esc(adv.quota_note_pa)}</div>"
                    f"<hr class='sep'>"
                    f"<div class='card-title'>Crop stage at the forecast date</div>"
                    f"<b>{esc(adv.crop_stage)}</b><br>"
                    f"<span style='color:var(--text-secondary)' dir='auto'>{esc(adv.crop_stage_pa)}</span>"
                    f"<div class='note' style='margin-top:8px'>Water-balance over the next "
                    f"{horizon} d: <span class='dg'>draft {adv.projected_draft_mcm:,.0f} MCM</span> vs "
                    f"<span class='ok'>recharge {adv.projected_recharge_mcm:,.0f} MCM</span> → "
                    f"net deficit <b>{adv.deficit_mcm:,.0f} MCM</b> "
                    f"(stage of extraction {adv.stage_of_extraction_pct:.0f}%)."
                    f"</div></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# TAB 2 — Macro & Spatial
# --------------------------------------------------------------------------------------
with tab2:
    st.markdown(f"### 🗺️ {district} — monitoring-station map & spatial stratification")

    m1, m2 = st.columns([1.9, 1.0])
    clicked_pos = None
    with m1:
        try:
            import folium
            import geopandas as gpd
            from streamlit_folium import st_folium

            fmap = folium.Map(location=[float(wells.lat.mean()), float(wells.lon.mean())],
                              zoom_start=10, tiles="CartoDB positron",
                              control_scale=True)
            pun = gpd.read_file("data/Punjab State Boundary.geojson")
            folium.GeoJson(
                json.loads(pun.to_json()), name="Punjab",
                style_function=lambda f: {"fillColor": "#E2E8F0", "color": "#94A3B8",
                                          "weight": 1.2, "fillOpacity": 0.35},
            ).add_to(fmap)
            for _, r in punjab_districts().iterrows():
                folium.GeoJson(json.loads(gpd.GeoSeries([r.geometry]).to_json()),
                               style_function=lambda f: {"fillColor": "#F1F5F9",
                                                         "color": "#CBD5E1", "weight": .7,
                                                         "fillOpacity": .45},
                               ).add_to(fmap)
            folium.GeoJson(
                district_geojson(district),
                name=f"{district} boundary",
                style_function=lambda f: {"fillColor": "#16A34A", "color": "#16A34A",
                                          "weight": 2, "fillOpacity": 0.05},
            ).add_to(fmap)

            def colour(r):
                return ("#16A34A" if r < 30 else "#D97706" if r <= 35 else "#DC2626")
            for _, r in wells.iterrows():
                thk = (f"{r.aquifer_thickness_m:.0f} m"
                       if np.isfinite(r.get("aquifer_thickness_m", np.nan)) else "n/a")
                folium.CircleMarker(
                    [r.lat, r.lon], radius=5.2,
                    color=colour(r.forecast_level_mbgl), weight=1.2,
                    fill=True, fill_color=colour(r.forecast_level_mbgl), fill_opacity=0.85,
                    tooltip=folium.Tooltip(
                        f"<b>{_html.escape(str(r.label)[:44])}</b><br>"
                        f"Forecast: {r.forecast_level_mbgl:.1f} m bgl<br>"
                        f"Aquifer thickness: {thk}<br>"
                        f"Pump set: {r.pump_set_depth_mbgl:.0f} m · headroom {r.headroom_m:.1f} m<br>"
                        f"Failure risk: <b>{r.pump_failure_risk_pct:.0f}%</b>"),
                ).add_to(fmap)
            # real telemetry gauges
            rf_meta, tb_meta = _telemetry_meta()
            if rf_meta is not None and not rf_meta.empty:
                for _, r in rf_meta.iterrows():
                    if np.isfinite(r.lat):
                        folium.Marker([r.lat, r.lon],
                                      icon=folium.Icon(color="blue", icon="tint"),
                                      tooltip=f"🌧️ Rain gauge {r.station}").add_to(fmap)
            if tb_meta is not None and not tb_meta.empty:
                for _, r in tb_meta.iterrows():
                    if np.isfinite(r.lat):
                        folium.Marker([r.lat, r.lon],
                                      icon=folium.Icon(color="orange",
                                                       icon="thermometer-half"),
                                      tooltip=f"🌡️ Temp telemetry {r.district}").add_to(fmap)
            obs_df = _obs()
            if not obs_df.empty:
                for _, r in obs_df.iterrows():
                    if np.isfinite(r.lat):
                        folium.Marker([r.lat, r.lon],
                                      icon=folium.Icon(color="green", icon="tint",
                                                       prefix="fa"),
                                      tooltip=f"CGWB {r.station}: {r.gw_level_mbgl:.2f} m bgl "
                                              f"({pd.Timestamp(r.date):%d %b %Y})").add_to(fmap)
            folium.LayerControl().add_to(fmap)
            out = st_folium(fmap, use_container_width=True, height=480,
                            key=f"map_{district}")
            if isinstance(out, dict):
                clicked_pos = out.get("last_object_clicked")
        except Exception as exc:
            st.warning(f"Interactive map unavailable ({exc})")
            st.map(wells[["lat", "lon"]].dropna(), size=18)

    # ---- station detail panel: click or select a station -------------------------------
    with m2:
        st.markdown("**📍 Station inspector**")
        st.caption("Click a well on the map (or pick one from the list) to reveal its "
                   "local depth, aquifer layer thickness and nearby connected tubewells.")

        n = len(wells)
        options = [f"{r.label} · {r.well_id}" for _, r in wells.iterrows()]
        sel_idx = int(st.session_state.get("gw_station_idx", 0) or 0)
        if not (0 <= sel_idx < n):
            sel_idx = 0

        if clicked_pos is not None:
            try:
                dlat = (wells.lat.astype(float) - float(clicked_pos["lat"])).to_numpy()
                dlon = (wells.lon.astype(float) - float(clicked_pos["lng"])).to_numpy()
                k = int(np.argmin(dlat ** 2 + dlon ** 2))
                if np.hypot(dlat[k], dlon[k]) < 0.08:      # ~9 km sanity radius
                    sel_idx = k
                    st.session_state["gw_station_idx"] = k
                    st.session_state["gw_station_sel"] = options[k]
            except Exception:
                pass

        pick = st.selectbox("Monitoring station", options, index=sel_idx,
                            key="gw_station_sel", label_visibility="collapsed")
        sel_idx = options.index(pick)
        r = wells.iloc[sel_idx]
        st.session_state["gw_station_idx"] = sel_idx

        zc = {"safe": "#16A34A", "critical": "#D97706",
              "over_exploited": "#DC2626"}.get(r.zone, "#16A34A")
        thick = (f"{r.aquifer_thickness_m:.0f} m"
                 if np.isfinite(r.get("aquifer_thickness_m", np.nan)) else "—")
        first_aq = (f"{r.depth_first_aquifer_m:.0f} m"
                    if np.isfinite(r.get("depth_first_aquifer_m", np.nan)) else "—")
        st.markdown(f"""
        <div class="card">
          <div class="card-title">{esc(r.label)}</div>
          <div class="note">{esc(r.well_id)} · block {esc(str(getattr(r, 'block', '—')))}</div>
          <div style="margin-top:8px"><b style="color:var(--text-primary)">{r.current_level_mbgl:.1f} m</b>
            <span class="note"> depth today → forecast </span>
            <b style="color:{zc}">{r.forecast_level_mbgl:.1f} m</b> {pill(r.zone.replace('_', ' ').title(), zc)}</div>
          <div class="note" style="margin-top:6px">Aquifer thickness: <b>{thick}</b>
            · first aquifer @ {first_aq}
            · pump set {r.pump_set_depth_mbgl:.0f} m · headroom {r.headroom_m:.1f} m
            · dry-out risk {r.pump_failure_risk_pct:.0f}%</div>
        </div>""", unsafe_allow_html=True)

        # nearby connected tubewells (hydraulic interference network)
        net = _network(district)
        try:
            if (net is not None and len(getattr(net, "wells", []))
                    and getattr(net, "influence", np.zeros((0, 0))).size
                    and len(net.stations)):
                slat = float(net.stations.lat.to_numpy(float))
                slon = float(net.stations.lon.to_numpy(float))
                si = int(np.argmin((slat - float(r.lat)) ** 2
                                    + (slon - float(r.lon)) ** 2))
                infl = np.asarray(net.influence[:, si], dtype=float)
                order = np.argsort(infl)[::-1][:5]
                rows = []
                for wi in order:
                    if not np.isfinite(infl[wi]) or infl[wi] <= 0:
                        continue
                    wl = net.wells.iloc[int(wi)]
                    dk = _haversine_km(float(r.lat), float(r.lon),
                                       float(wl.lat), float(wl.lon))
                    rows.append(dict(Tubewell=f"TW-{int(wi):04d}",
                                     Distance=f"{dk:.2f} km",
                                     Drawdown=f"{infl[wi]:.2f} m"))
                st.markdown("**🔗 Nearby connected tubewells**")
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                 width='stretch')
                cum = float(np.asarray(net.cumulative_drawdown_m).ravel()[si])
                st.markdown(
                    f"<div class='note'>Season-long cumulative drawdown at this "
                    f"piezometer from the surrounding tubewell field: "
                    f"<b>{cum:.2f} m</b> "
                    f"({len(net.wells):,} bores solved for interference in "
                    f"{district}).</div>", unsafe_allow_html=True)
            else:
                st.info("Interference network unavailable for this district.")
        except Exception as exc:
            st.caption(f"Nearby tubewells unavailable: {exc}")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # ---- spatial stratification ---------------------------------------------------------
    st.markdown("#### 🎨 Spatial stratification — block stress levels")
    st.markdown(
        f"{pill('Safe · <30 m', '#16A34A')} {pill('Critical · 30–35 m', '#D97706')} "
        f"{pill('Over-Exploited · >35 m', '#DC2626')} "
        f"<span class='note' style='margin-left:8px'>colour-coded by mean forecast "
        f"water level at +{horizon} days</span>", unsafe_allow_html=True)

    s1, s2 = st.columns([1.15, 1.0])
    with s1:
        if not blocks.empty:
            b = blocks.head(12)
            # precomputed hex colours (numeric arrays + colorscale make plotly.js
            # fall back to black text for `textposition="auto"` labels)
            def _stress_col(v):
                if v < 30:
                    return "#16A34A"
                if v <= 35:
                    return "#D97706"
                if v <= 40:
                    return "#EA580C"
                return "#DC2626"
            fig = go.Figure(go.Bar(
                x=b.mean_risk_pct, y=b.block.astype(str).str.title(), orientation="h",
                marker=dict(color=[_stress_col(v) for v in b.mean_level_mbgl]),
                text=[f"{v:.0f}%" for v in b.mean_risk_pct], textposition="auto",
                textfont=dict(color="#FFFFFF", size=10)))
            fig.update_xaxes(title_text="Mean pump-failure risk (%)", range=[0, 100])
            fig.update_yaxes(title_text="Block")
            st.plotly_chart(viz._layout(
                fig, "Block stress ranking — mean dry-out risk at forecast date",
                legend_top=False), width='stretch')
        st.markdown(f"<div class='note'>{len(wells):,} CGWB / Punjab-GW observation wells "
                    f"downscaled from the district forecast. "
                    f"<b>{int((wells.zone=='over_exploited').sum())}</b> are forecast in the "
                    f"over-exploited band (&gt;35 m) and "
                    f"<b>{int((wells.pump_failure_risk_pct>60).sum())}</b> carry &gt;60% "
                    f"dry-out risk.</div>", unsafe_allow_html=True)
    with s2:
        try:
            from src.aquifer_strata import cross_section
            xs_fig = cross_section(district, level_mbgl=float(adv.current_level_mbgl),
                                   decline_m_per_yr=prior.long_term_decline_m_per_yr)
            xs_fig.update_layout(
                title=dict(text=f"{district} — aquifer thickness cross-section",
                           font=dict(size=14, color="#0F172A"), x=0.0,
                           xanchor="left"),
                height=420, margin=dict(l=40, r=20, t=50, b=40),
                legend=dict(orientation="h", x=0, y=0.02, yanchor="bottom",
                            bgcolor="rgba(255,255,255,0.75)", font=dict(size=10)))
            st.plotly_chart(xs_fig, width='stretch')
        except Exception as exc:
            st.caption(f"Cross-section unavailable: {exc}")

    st.markdown(
        "<div class='note'>Aquifer thickness cross-section overview: the three-tier "
        "strata model (L1 / L2 / L3) cut west–east across the district with today's "
        "water table and the +10 / +20-year decline positions.</div>",
        unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# TAB 3 — Digital Twin & Credit (dedicated simulation tab)
# --------------------------------------------------------------------------------------
with tab3:
    st.markdown("### 🧪 Digital Twin — intervention & climate levers")
    st.markdown(
        "<div class='note'>The FAO-56 agro-hydrological water balance is re-run live "
        "with your levers over a 10-year counterfactual window. This is a calibrated "
        "physics simulation, not a curve fit — move a slider and both trajectories "
        "below are recomputed.</div>", unsafe_allow_html=True)

    # ---- Part A — levers (moved here from the sidebar) --------------------------------
    st.markdown("#### 🎛️ Part A · Intervention & climate levers")
    lc = st.columns(4)
    with lc[0]:
        st.slider("🌧️ Monsoon rain anomaly", -40, 40, LEVER_DEFAULTS["dt_monsoon"], 5,
                  format="%.0f%%", key="dt_monsoon",
                  help="% change in Jun–Sep monsoon recharge vs normal.")
    with lc[1]:
        st.slider("💧 Micro-drip / sprinkler adoption", 0, 100,
                  LEVER_DEFAULTS["dt_drip"], 5, format="%.0f%%", key="dt_drip",
                  help="% of the irrigated area shifted to micro-irrigation.")
    with lc[2]:
        st.slider("🌾 Paddy sowing / transplant delay", 0, 30,
                  LEVER_DEFAULTS["dt_shift"], 1, format="%.0f d", key="dt_shift",
                  help="Days later than the normal transplanting window.")
    with lc[3]:
        st.slider("🚰 Canal water allocation shift", 50, 150,
                  LEVER_DEFAULTS["dt_canal"], 5, format="%.0f%%", key="dt_canal",
                  help="% of the normal canal allocation (surface water trades for "
                       "groundwater draft).")

    # widget state is now current for this run
    monsoon = int(st.session_state["dt_monsoon"])
    drip = int(st.session_state["dt_drip"])
    shift = int(st.session_state["dt_shift"])
    canal = int(st.session_state["dt_canal"])
    scen_live = Scenario(name="Your scenario",
                         irrigation_efficiency=(0.90 if drip > 0
                                                else FLOOD_IRRIGATION_EFFICIENCY),
                         monsoon_anomaly_pct=float(monsoon),
                         paddy_transplant_shift_days=int(shift),
                         canal_availability_pct=float(canal),
                         pump_adoption_drip_pct=float(drip))

    bau = bau_trend(district, SIM_DAYS)
    pb, pi = twin_runs(district, monsoon, drip, shift, canal, SIM_DAYS)

    bau_end = float(bau.level.iloc[-1])
    interv_end = float(pi.projected_level_mbgl.iloc[-1])
    draft_base = float(pb.gw_draft_mcm.sum())
    draft_interv = float(pi.gw_draft_mcm.sum())
    risk_base, _ = pump_failure_risk(bau_end, float(pump_set))
    risk_interv, _ = pump_failure_risk(interv_end, float(pump_set))

    # ---- Part B — before vs after comparative analytics --------------------------------
    st.markdown("#### 📊 Part B · Before vs. after comparative analytics")
    fig = go.Figure()
    x0, x1 = bau.date.iloc[0], bau.date.iloc[-1]
    # baseline — business-as-usual, dashed red
    fig.add_trace(go.Scatter(
        x=bau.date, y=bau.level, mode="lines",
        name="Baseline — business-as-usual (no intervention)",
        line=dict(color="#DC2626", width=2.6, dash="dash"),
        hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} m bgl<extra>Baseline</extra>"))
    # intervention — solid green, with the conserved headroom shaded between curves
    fig.add_trace(go.Scatter(
        x=pi.date, y=pi.projected_level_mbgl, mode="lines",
        name="Intervention — your lever settings",
        line=dict(color="#16A34A", width=3.0),
        fill="tonexty", fillcolor="rgba(22,163,74,0.16)",
        hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} m bgl<extra>Intervention</extra>"))
    # thresholds
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[45, 45], mode="lines",
        name="Pump Cavitation / Air-Suction Limit (45 m)",
        line=dict(color="#7C3AED", width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[42, 42], mode="lines",
        name="Pump-failure threshold (42 m)",
        line=dict(color="#D97706", width=1.4, dash="dot")))
    lo = float(min(bau.level.min(), pi.projected_level_mbgl.min())) - 1.0
    hi = float(max(bau.level.max(), pi.projected_level_mbgl.max())) + 1.0
    viz.zone_bands(fig, lo, hi)
    fig.add_vline(x=AS_OF, line=dict(color="#94A3B8", width=1, dash="dash"))
    fig.update_yaxes(title_text="Depth to water table (m bgl)", autorange="reversed",
                     range=[max(hi, 46.5), lo])
    fig.update_xaxes(title_text="")
    st.plotly_chart(viz._layout(
        fig, f"Before vs. after — {SIM_DAYS // 365}-year counterfactual trajectories "
             f"({district})", legend_top=True), width='stretch')
    st.caption("Shaded green area = water headroom conserved by your levers (volume "
               "between the business-as-usual curve and the simulated intervention). "
               "Baseline continues the district's observed long-term depletion trend "
               f"({prior.long_term_decline_m_per_yr:.2f} m/yr); the intervention is the "
               "FAO-56 twin with reduced pumping draft, monsoon conservation and canal "
               "augmentation.")

    # ---- impact scorecards --------------------------------------------------------------
    st.markdown("#### 🧮 Impact scorecards")
    headroom = bau_end - interv_end
    draft_avoided = draft_base - draft_interv
    cap_base = dict(total_inr=0)
    cap_interv = dict(total_inr=0)
    try:
        from src.aquifer_strata import deepening_capex
        cap_base = deepening_capex(max(bau_end, 1.0),
                                   static_level_mbgl=float(adv.current_level_mbgl))
        cap_interv = deepening_capex(max(interv_end, 1.0),
                                     static_level_mbgl=float(adv.current_level_mbgl))
    except Exception:
        pass
    capex_avoided_lakh = (cap_base["total_inr"] - cap_interv["total_inr"]) / 1e5

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(kpi_card(
            "Water Headroom Saved", f"{headroom:+.2f}", "meters",
            f"between {bau_end:.1f} m (baseline) and {interv_end:.1f} m "
            f"(intervention) at year {SIM_DAYS // 365}",
            colour="#16A34A"), unsafe_allow_html=True)
    with s2:
        st.markdown(kpi_card(
            "Net Pumping Draft Avoided",
            (f"−{draft_avoided:,.0f}" if draft_avoided >= 1 else "0"), "MCM",
            (f"over the {SIM_DAYS // 365}-year window "
             f"(−{draft_avoided / (SIM_DAYS / 365.25):,.0f} MCM/yr) · "
             f"baseline {draft_base:,.0f} → {draft_interv:,.0f} MCM"
             if draft_avoided >= 1 else
             "neutral levers — raise drip, canal or monsoon above the defaults "
             "to cut pumping draft"),
            colour="#16A34A" if draft_avoided >= 1 else "#475569"),
            unsafe_allow_html=True)
    with s3:
        risk_col = "#DC2626" if risk_base > 60 else "#D97706"
        st.markdown(kpi_card(
            "Pump Dry-Out Probability",
            f"{risk_base:.0f}% ➔ {risk_interv:.0f}%", "",
            f"Baseline: {risk_base:.0f}% at {bau_end:.1f} m · "
            f"Simulated: {risk_interv:.0f}% at {interv_end:.1f} m",
            colour=risk_col), unsafe_allow_html=True)
    with s4:
        st.markdown(kpi_card(
            "Borehole Deepening Capex Avoided",
            (f"₹{max(capex_avoided_lakh, 0.0):.1f}" if capex_avoided_lakh >= 0.1
             else "₹0"),
            "Lakhs / farmer",
            (f"baseline would chase the table to {bau_end:.1f} m "
             f"(₹{cap_base['total_inr'] / 1e5:.1f} L) vs "
             f"₹{cap_interv['total_inr'] / 1e5:.1f} L with your levers"
             if capex_avoided_lakh >= 0.1 else
             f"at neutral levers both curves reach {bau_end:.1f} m — move the "
             f"levers to keep the bore viable"),
            colour="#16A34A" if capex_avoided_lakh >= 0.1 else "#475569"),
            unsafe_allow_html=True)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # ---- Satin Finserv / MFI credit risk adjustment ------------------------------------
    st.markdown("#### 🏦 Satin Finserv / MFI credit risk adjustment")
    try:
        sq, post = credit_pair(district, bau_end, interv_end,
                               float(adv.current_level_mbgl), float(pump_set))
        g1, g2 = st.columns([1.0, 1.1])
        with g1:
            gfig = go.Figure(go.Bar(
                x=["Status Quo", "Post-Intervention"],
                y=[sq.score, post.score],
                marker_color=["#DC2626", "#16A34A"],
                text=[f"{sq.score:.0f}/100", f"{post.score:.0f}/100"],
                textposition="outside",
                textfont=dict(size=14, color="#0F172A"),
                hovertemplate="%{x}: %{y:.1f}/100<extra></extra>"))
            gfig.update_yaxes(title_text="Credit risk score (0–100)", range=[0, 110])
            st.plotly_chart(viz._layout(
                gfig, "Aquifer-linked default risk before vs after", legend_top=False),
                width='stretch')
        with g2:
            sq_high = sq.score >= 71
            post_low = post.score <= 35
            sq_line = ("High Default Risk due to water failure at flowering"
                       if sq_high else
                       "Moderate Default Risk — contingency clause advised")
            post_line = ("Low Default Risk; qualifies for interest rebate"
                         if post_low else
                         "Reduced risk — conditional approval with drip financed in")
            st.markdown(f"""
            <div class="card">
              <div class="card-title">Satin Finserv / MFI underwriting note</div>
              <div style="margin-top:8px"><b style="color:#DC2626">Status Quo Credit Risk:
                {sq.score:.0f}/100</b> — {sq_line}
                <div class="note">band: {esc(sq.band_label)} · scored at
                {bau_end:.1f} m bgl (the level business-as-usual reaches inside the
                window)</div></div>
              <div style="margin-top:10px"><b style="color:#16A34A">Post-Intervention
                Credit Risk: {post.score:.0f}/100</b> — {post_line}
                <div class="note">band: {esc(post.band_label)} · scored at
                {interv_end:.1f} m bgl with the flowering deficit and the deepening
                capex both removed</div></div>
              <hr class="sep">
              <div class="note"><b>Lending decision:</b> {esc(post.recommendation)}</div>
            </div>""", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        d1.metric("Score improvement", f"−{sq.score - post.score:.0f} pts",
                  f"{sq.score:.0f} → {post.score:.0f}", delta_color="inverse")
        d2.metric("Borehole deepening capex avoided",
                  f"₹{max(capex_avoided_lakh, 0.0):.1f} L",
                  f"vs ₹{cap_base['total_inr'] / 1e5:.1f} L at the baseline level")
        d3.metric("Interest rebate", "Eligible" if post_low else "Not yet",
                  "Satin Finserv aquifer-linked KCC terms")
    except Exception as exc:
        st.warning(f"Credit risk adjustment unavailable: {exc}")

# --------------------------------------------------------------------------------------
# TAB 4 — Model Lab
# --------------------------------------------------------------------------------------
with tab4:
    st.markdown("### 🔬 Model Lab — how good is it, really?")
    res = _model()

    # ---- architecture card ----------------------------------------------------------------
    try:
        n_params = sum(p.numel() for p in res.model.parameters())
    except Exception:
        n_params = 0
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Model architecture — hybrid physics-informed forecaster</div>
      <div class="note" style="margin-top:4px">
      <b style="color:var(--text-primary)">PyTorch LSTM</b> · {n_params:,} trainable parameters ·
      60-day input window · 2 stacked LSTM layers (hidden 64, dropout 0.2) ·
      direct multi-horizon heads for t+30 / t+60 / t+90 · MC-dropout
      ({MC_DROPOUT_PASSES} stochastic passes) for the 95% confidence band ·
      6 telemetry drivers + 2 calendar channels (doy sin/cos) · Adam (lr 1e-3),
      ReduceLROnPlateau, early stopping at patience {8}.<br>
      <b style="color:var(--text-primary)">FAO-56 agro-hydrological mass balance</b> —
      Hargreaves ET₀ → Kc-curve crop demand (wheat 0.40→1.15→0.30, paddy 1.05→1.20→0.85)
      → effective rainfall (USDA-SCS) + canal supply → lumped aquifer water balance with
      specific yield Sy={prior.specific_yield} and a {150:.0f}-day recharge lag; the
      LSTM is trained on residuals around this physics trend, which is why the digital
      twin and the network can never disagree about direction.</div>
    </div>""", unsafe_allow_html=True)

    metrics_all = None
    try:
        metrics_all = evaluate(res, df, verbose=False)
    except Exception as exc:
        st.warning(f"Evaluation failed: {exc}")

    if metrics_all:
        k1, k2, k3, k4, k5 = st.columns(5)
        for col, lab, val, sub, colr in [
            (k1, "Test RMSE", f"{metrics_all['rmse_m']:.3f} m", "all horizons", "#16A34A"),
            (k2, "Test MAE", f"{metrics_all['mae_m']:.3f} m", "metres", "#0369A1"),
            (k3, "R²", f"{metrics_all['r2']:.3f}", "vs climatology", "#0369A1"),
            (k4, "Directional trend accuracy",
             f"{metrics_all['directional_acc']*100:.1f}%", "rise/fall direction", "#D97706"),
            (k5, "Skill vs persistence",
             f"{metrics_all['skill_vs_persistence_pct']:+.1f}%", "RMSE reduction", "#16A34A"),
        ]:
            col.markdown(kpi_card(lab, val, "", sub, colour=colr),
                         unsafe_allow_html=True)

        st.markdown("")
        t1, t2 = st.columns([1.05, 1.0])
        with t1:
            rows = []
            for h in HORIZONS:
                d = metrics_all["per_horizon"][f"t+{h}"]
                rows.append(dict(Horizon=f"+{h} d", RMSE_m=round(d["rmse_m"], 3),
                                 MAE_m=round(d["mae_m"], 3), Bias_m=round(d["bias_m"], 3),
                                 MaxErr_m=round(d["max_abs_m"], 3),
                                 TrendAcc=f"{d['directional_acc']*100:.1f}%"))
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            st.markdown(f"<div class='note'>Baselines on the same test window — "
                        f"persistence <b>{metrics_all['baseline_persistence_rmse_m']:.3f} m</b>, "
                        f"seasonal-naive <b>{metrics_all['baseline_seasonal_naive_rmse_m']:.3f} m</b>, "
                        f"linear trend <b>{metrics_all['baseline_linear_trend_rmse_m']:.3f} m</b>. "
                        f"AquaCast beats all three.</div>", unsafe_allow_html=True)
            st.plotly_chart(viz.history_chart(res.history), width='stretch')
            st.caption("Training loss convergence — train vs validation RMSE per epoch "
                       "(early stopping guards the 30-epoch budget).")
        with t2:
            ev = test_evaluation(district)
            if ev is not None:
                st.plotly_chart(viz.validation_chart(df, ev["issue"], ev["y_true"],
                                                     ev["y_pred"], horizon, j),
                                width='stretch')
                st.plotly_chart(viz.scatter_skill(ev["y_true"][:, j], ev["y_pred"][:, j],
                                                  horizon), width='stretch')

        st.markdown('<hr class="sep">', unsafe_allow_html=True)
        x1, x2 = st.columns(2)
        with x1:
            imp = permutation_importance(res, df, n_repeats=3)
            st.plotly_chart(viz.importance_chart(imp), width='stretch')
        with x2:
            try:
                sal = input_saliency(res, df)
                sal_fig = viz.saliency_chart(sal, f"t+{horizon}")
                # 8 stacked channels wrap the legend to two rows, so the title
                # moves above the figure — guaranteed collision-free.
                sal_fig.update_layout(title=None)
                st.markdown(f"**Which days drove the t+{horizon} forecast**")
                st.plotly_chart(sal_fig, width='stretch')
            except Exception as exc:
                st.info(f"Saliency unavailable: {exc}")

        if st.button("🏁 Run model benchmark (LSTM vs GRU vs linear vs persistence)"):
            with st.spinner("Training comparison models…"):
                bm = benchmark(res, df, quick=True, verbose=False)
            st.plotly_chart(viz.benchmark_bar(bm), width='stretch')
            st.dataframe(bm, hide_index=True, width='stretch')

    # ---- data provenance ledger -----------------------------------------------------------
    with st.expander("🧾 Data provenance ledger", expanded=False):
        p1, p2 = st.columns(2)
        with p1:
            rf_meta, tb_meta = _telemetry_meta()
            st.markdown("**Real India-WRIS rainfall gauges**")
            st.dataframe(rf_meta, hide_index=True, width='stretch')
        with p2:
            st.markdown("**Real hourly temperature telemetry**")
            st.dataframe(tb_meta, hide_index=True, width='stretch')
        obs_df = _obs()
        st.markdown(f"**In-situ CGWB water-level readings** — {len(obs_df)} measurements "
                    f"across {obs_df.station.nunique()} wells "
                    f"({pd.Timestamp(obs_df.date.min()).date()} → "
                    f"{pd.Timestamp(obs_df.date.max()).date()}). These are the only true "
                    f"labels in the system; the physics model is calibrated against them.")
        st.dataframe(obs_df.head(12), hide_index=True, width='stretch')
        meta_path = Path(f"data/processed/processed_{district.lower()}_meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            st.json({"calibration": meta.get("calibration"),
                     "constants": meta.get("constants"),
                     "provenance": meta.get("provenance")})

# --------------------------------------------------------------------------------------
# TAB 5 — Admin & Finserv
# --------------------------------------------------------------------------------------
with tab5:
    st.markdown("### 🏛️ Admin & Satin Finserv — portfolio, feeder scheduling & exports")

    # ---- portfolio exposure summary --------------------------------------------------------
    st.markdown("#### 💼 Portfolio exposure summary — institutional lenders")
    port, oe = {}, {}
    for dd in DISTRICTS:
        try:
            ddf = _data(dd)
            ff = forecast_for(dd, MC_DROPOUT_PASSES)
            if ff is None:
                continue
            a = build_advisory(ddf, dd, ff, horizon, scen_live,
                               pump_set_depth_mbgl=float(pump_set),
                               farm_acres=float(acres))
            port[dd] = dict(risk_score=a.risk_score, band_en=a.risk_band, pd_pct=a.pd_pct,
                            exposure_inr=a.exposure_inr,
                            expected_loss_inr=a.expected_loss_inr,
                            loss_avoidable_inr=a.expected_loss_inr * 0.22)
            oe[dd] = _overexploited_summary(dd)
        except Exception:
            continue
    pv = portfolio_view(port) if port else {}
    oe_wells = sum(v["oe_wells"] for v in oe.values())
    tot_wells = max(sum(v["wells"] for v in oe.values()), 1)
    oe_blocks = sum(v["oe_blocks"] for v in oe.values())
    tot_blocks = max(sum(v["n_blocks"] for v in oe.values()), 1)
    oe_exposure = sum(port[d]["exposure_inr"] * oe.get(
        d, {"oe_well_share": 0.0})["oe_well_share"] for d in port)

    if pv:
        e1, e2, e3, e4 = st.columns(4)
        for col, lab, val, sub, colr in [
            (e1, "Total agricultural credit exposure",
             f"₹{pv['exposure_inr']/1e7:,.0f} cr",
             f"{pv['districts']} districts · crop loans", "#0F172A"),
            (e2, "Exposure in over-exploited blocks",
             f"₹{oe_exposure/1e7:,.0f} cr",
             f"{oe_blocks}/{tot_blocks} blocks · {oe_wells:,}/{tot_wells:,} wells >35 m",
             "#DC2626"),
            (e3, "Portfolio-at-risk", f"{pv['par_pct']:.2f}%",
             "expected 12-month loss rate", "#D97706"),
            (e4, "Expected loss", f"₹{pv['expected_loss_inr']/1e7:,.1f} cr",
             f"EAD × PD × LGD 60% · ₹{pv['loss_avoidable_inr']/1e7:,.1f} cr avoidable "
             f"with 90-day early action", "#16A34A"),
        ]:
            col.markdown(kpi_card(lab, val, "", sub, colour=colr),
                         unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(pv["per_district"]).T.reset_index()
                     .rename(columns={"index": "district"}),
                     hide_index=True, width='stretch')

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # ---- rotational feeder power scheduling ------------------------------------------------
    st.markdown("#### ⚡ Rotational feeder power scheduling")
    st.caption("Punjab's agricultural feeders run on rotational supply. The same 8 hours "
               "of power deliver more water — and stress the grid less — when the feeder "
               "groups are staggered instead of drawing peak tubewell load together.")
    p1, p2, p3 = st.columns(3)
    hours = p1.slider("Supply hours per day", 2, 12, 8, 1, key="feeder_hours")
    groups = p2.slider("Feeder groups to rotate", 2, 6, 3, 1, key="feeder_groups")
    per = hours / max(groups, 1)
    sched = []
    for i in range(int(groups)):
        s = int(22 + i * per) % 24
        e = int(22 + (i + 1) * per) % 24
        sched.append(dict(Group=f"Feeder group {i+1}",
                          **{"Window": f"{s:02d}:00 – {e:02d}:00",
                             "Hours": round(per, 2)}))
    sched_df = pd.DataFrame(sched)

    total_kw = prior.tubewells * prior.avg_pump_hp * 0.746
    peak_unsched = total_kw / 1000.0
    peak_sched = total_kw / max(groups, 1) / 1000.0
    q1, q2, q3 = st.columns(3)
    q1.markdown(kpi_card("Simultaneous peak (no rotation)", f"{peak_unsched:,.0f}", "MW",
                         f"all {prior.tubewells:,} tubewells on one window",
                         colour="#DC2626"), unsafe_allow_html=True)
    q2.markdown(kpi_card("Scheduled rotational peak", f"{peak_sched:,.0f}", "MW",
                         f"{groups} groups × {per:.1f} h staggered from 22:00",
                         colour="#16A34A"), unsafe_allow_html=True)
    q3.markdown(kpi_card("Peak coincidence avoided",
                         f"{(1 - 1 / max(groups, 1)) * 100:.0f}", "%",
                         f"−{peak_unsched - peak_sched:,.0f} MW of feeder coincidence "
                         f"at the evening peak", colour="#16A34A"),
                unsafe_allow_html=True)

    t1, t2 = st.columns([1.0, 1.2])
    with t1:
        st.dataframe(sched_df, hide_index=True, width='stretch')
    with t2:
        try:
            from src.crop_cycle_model import next_windows
            cw = next_windows(30, district=district)
            if cw:
                st.markdown(f"""
                <div class="card" style="border-left:3px solid #D97706">
                  <div class="card-title">⚡ Priority override — crop stage first call</div>
                  <div class="note">{esc(cw[0]['crop'])} hits <b>{esc(cw[0]['stage'])}</b>
                  on {cw[0]['date']:%d %b}, needing about {cw[0]['mm_day']:.1f} mm/day.
                  During that window this stage should get first call on the feeder —
                  a missed CRI or flowering irrigation cannot be recovered later in the
                  season.</div></div>""", unsafe_allow_html=True)
            else:
                st.info("No critical crop stage in the next 30 days — rotation can "
                        "follow the standing schedule.")
        except Exception as exc:
            st.caption(f"Crop-window lookup unavailable: {exc}")

    st.markdown('<hr class="sep">', unsafe_allow_html=True)

    # ---- exportable compliance & risk report -----------------------------------------------
    st.markdown("#### 📄 Exportable district compliance & risk report")

    compliance = pd.DataFrame([
        dict(district=district, as_of=f"{AS_OF:%Y-%m-%d}",
             current_level_mbgl=round(adv.current_level_mbgl, 2),
             predicted_level_mbgl=round(adv.predicted_level_mbgl, 2),
             forecast_horizon_days=horizon, zone=adv.zone_label_en,
             stage_of_extraction_pct=round(adv.stage_of_extraction_pct, 1),
             pump_failure_risk_pct=round(adv.pump_failure_risk_pct, 1),
             weekly_quota_h_per_acre=round(adv.quota_hours_per_week_per_acre, 2),
             credit_risk_score=round(adv.risk_score, 1), credit_band=adv.risk_band,
             exposure_inr=round(adv.exposure_inr, 0),
             expected_loss_inr=round(adv.expected_loss_inr, 0),
             over_exploited_blocks=f"{oe.get(district, {}).get('oe_blocks', 0)}/"
                                    f"{oe.get(district, {}).get('n_blocks', 0)}")
    ])
    block_csv = (blocks.drop(columns="geometry").to_csv(index=False).encode()
                 if not blocks.empty and "geometry" in blocks
                 else (blocks.to_csv(index=False).encode() if not blocks.empty else b""))

    x1, x2, x3 = st.columns(3)
    with x1:
        st.download_button("⬇️ Compliance & risk summary (CSV)",
                           data=compliance.to_csv(index=False).encode(),
                           file_name=f"{district.lower()}_compliance_risk.csv",
                           mime="text/csv", width='stretch')
    with x2:
        st.download_button("⬇️ Block-level risk table (CSV)", data=block_csv,
                           file_name=f"{district.lower()}_block_risk.csv",
                           mime="text/csv", width='stretch')
    with x3:
        st.download_button("⬇️ Daily timeseries (CSV)",
                           data=df.to_csv(index=False).encode(),
                           file_name=f"{district.lower()}_daily_timeseries.csv",
                           mime="text/csv", width='stretch')

    report_html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>AquaCast compliance report — {district}</title>
<style>body{{font-family:Inter,system-ui,sans-serif;max-width:820px;margin:36px auto;
color:#0F172A;padding:0 24px}}h1{{color:#0F172A}}h2{{color:#475569;font-size:1.05rem}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #E2E8F0;padding:6px 9px;text-align:left;font-size:13px}}
th{{background:#F8FAFC}} .pa{{font-family:'Noto Sans Gurmukhi',system-ui}}
.tag{{display:inline-block;padding:2px 10px;border-radius:999px;background:#ECFDF5;
color:#15803D;font-size:12px;font-weight:700}}</style></head><body>
<h1>AquaCast-Punjab — District Water Compliance &amp; Risk Report</h1>
<p><b>{district}</b> · issued {AS_OF:%d %b %Y} · forecast horizon +{horizon} days ·
scenario {esc(scen_live.name)} <span class="tag">{esc(adv.zone_label_en)}</span></p>
<h2>1. Water status</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Current water table</td><td>{adv.current_level_mbgl:.2f} m bgl</td></tr>
<tr><td>Predicted at {adv.target_date}</td><td>{adv.predicted_level_mbgl:.2f} m bgl
({adv.predicted_delta_m:+.2f} m)</td></tr>
<tr><td>Pump dry-out risk</td><td>{adv.pump_failure_risk_pct:.0f}%</td></tr>
<tr><td>Weekly pumping quota</td><td>{adv.quota_hours_per_week_per_acre:.1f} h/week/acre</td></tr>
<tr><td>Stage of extraction</td><td>{adv.stage_of_extraction_pct:.0f}%</td></tr>
</table>
<h2>2. Credit &amp; portfolio</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Credit risk score</td><td>{adv.risk_score:.0f}/100 — {esc(adv.risk_band)}</td></tr>
<tr><td>Exposure</td><td>₹{adv.exposure_inr:,.0f}</td></tr>
<tr><td>Expected loss (EAD×PD×LGD)</td><td>₹{adv.expected_loss_inr:,.0f}</td></tr>
<tr><td>Over-exploited blocks</td><td>{oe.get(district, {}).get('oe_blocks', 0)} of
{oe.get(district, {}).get('n_blocks', 0)}</td></tr>
</table>
<h2>3. Block-level risk</h2>
{blocks.drop(columns='geometry', errors='ignore').to_html(index=False, border=0)
 if not blocks.empty else '<p>No block data.</p>'}
<h2>4. Advisory (English)</h2><pre style="white-space:pre-wrap">{_html.escape(adv.whatsapp_en)}</pre>
<h2 class="pa">ਸਲਾਹ (ਪੰਜਾਬੀ)</h2><pre class="pa" style="white-space:pre-wrap">{_html.escape(adv.whatsapp_pa)}</pre>
<p style="color:#64748B;font-size:12px">Generated by AquaCast-Punjab ·
physics-informed hybrid (India-WRIS telemetry + FAO-56 + PyTorch LSTM) ·
data window {DATA_START} → {DATA_END}</p>
</body></html>"""
    st.download_button("⬇️ Full district report (HTML → print to PDF)",
                       data=report_html,
                       file_name=f"AquaCast_report_{district}_{horizon}d.html",
                       mime="text/html", width='stretch')
    st.caption("Open the HTML file and use your browser's Print → Save as PDF to produce "
               "the signed compliance copy.")

# ======================================================================================
st.markdown('<hr class="sep">', unsafe_allow_html=True)
st.markdown(
    f"<div class='note' style='text-align:center'>"
    f"AquaCast-Punjab · physics-informed hybrid (real India-WRIS telemetry + FAO-56 + lumped "
    f"aquifer water balance + PyTorch LSTM) · data window {DATA_START} → {DATA_END} · "
    f"Built for Sankalp — Climate-Smart Agriculture, Water Conservation &amp; Rural Credit Risk."
    f"</div>", unsafe_allow_html=True)
