"""
Module E — Persona views
========================

One system, three rooms. The same model output is rendered three ways, because
the same number means three different things to three different people:

    1. Kisan Mobile      — 375 px, high contrast, Punjabi/English, one dial and
                           one instruction: how many hours to run the pump, and
                           when. Ends in a WhatsApp push.
    2. Farmer Desktop    — the water budget behind that instruction, the
                           neighbour-pump interference calculator, a rotation
                           simulator, and a drilling finance calculator.
    3. District Admin &
       Satin Finserv     — the network map, block stress heatmap, portfolio at
                           risk in ₹ Crore, and the power-supply optimiser.

Bilingual strings live in ``KISAN_I18N``. Punjabi is Gurmukhi, written out
literally (not transliterated), and every key has both languages so the toggle
can never leave a user staring at an English fallback.
"""
from __future__ import annotations

import html
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Bilingual copy for the kisan view
# ---------------------------------------------------------------------------
KISAN_I18N = {
    "title": {"en": "AquaCast", "pa": "ਐਕੁਆਕਾਸਟ"},
    "subtitle": {"en": "Your tubewell, today", "pa": "ਤੁਹਾਡਾ ਟਿਊਬਵੈੱਲ, ਅੱਜ"},
    "water_level": {"en": "Water level", "pa": "ਪਾਣੀ ਦਾ ਪੱਧਰ"},
    "metres_below": {"en": "metres below ground", "pa": "ਮੀਟਰ ਧਰਤੀ ਹੇਠਾਂ"},
    "status": {"en": "Status", "pa": "ਸਥਿਤੀ"},
    "pump_today": {"en": "Run your pump today", "pa": "ਅੱਜ ਪੰਪ ਚਲਾਓ"},
    "best_window": {"en": "Best window", "pa": "ਸਭ ਤੋਂ ਵਧੀਆ ਸਮਾਂ"},
    "why": {"en": "Why", "pa": "ਕਾਰਨ"},
    "next_alert": {"en": "Next critical stage", "pa": "ਅਗਲੀ ਨਾਜ਼ੁਕ ਅਵਸਥਾ"},
    "no_alert": {"en": "No critical crop stage in the next 30 days",
                 "pa": "ਅਗਲੇ 30 ਦਿਨਾਂ ਵਿੱਚ ਕੋਈ ਨਾਜ਼ੁਕ ਅਵਸਥਾ ਨਹੀਂ"},
    "send": {"en": "Send as WhatsApp", "pa": "ਵਟਸਐਪ 'ਤੇ ਭੇਜੋ"},
    "sent": {"en": "Pushed to this kisan's phone",
             "pa": "ਇਸ ਕਿਸਾਨ ਦੇ ਫ਼ੋਨ 'ਤੇ ਭੇਜਿਆ ਗਿਆ"},
    "district": {"en": "District", "pa": "ਜ਼ਿਲ੍ਹਾ"},
    "safe": {"en": "Safe", "pa": "ਸੁਰੱਖਿਅਤ"},
    "warning": {"en": "Warning", "pa": "ਚੇਤਾਵਨੀ"},
    "failure": {"en": "Pump failure risk", "pa": "ਪੰਪ ਖ਼ਰਾਬ ਹੋਣ ਦਾ ਖ਼ਤਰਾ"},
    "hours": {"en": "hours", "pa": "ਘੰਟੇ"},
    "litres": {"en": "litres per acre", "pa": "ਲੀਟਰ ਪ੍ਰਤੀ ਏਕੜ"},
    "litres_today": {"en": "litres today (whole holding)",
                     "pa": "ਅੱਜ ਲੀਟਰ (ਸਾਰੀ ਜ਼ਮੀਨ)"},
    "depth_now": {"en": "Water is at", "pa": "ਪਾਣੀ ਇੱਥੇ ਹੈ"},
    "falls_to": {"en": "falls to", "pa": "ਡਿੱਗ ਕੇ"},
    "advice": {"en": "Advice", "pa": "ਸਲਾਹ"},
}

BAND_I18N = {
    "safe": {"en": "SAFE", "pa": "ਸੁਰੱਖਿਅਤ", "col": "#22C55E"},
    "warning": {"en": "WARNING", "pa": "ਚੇਤਾਵਨੀ", "col": "#F59E0B"},
    "failure": {"en": "PUMP FAILURE RISK", "pa": "ਪੰਪ ਖ਼ਰਾਬ ਹੋਣ ਦਾ ਖ਼ਤਰਾ", "col": "#EF4444"},
}


def t(key: str, lang: str = "en") -> str:
    d = KISAN_I18N.get(key)
    if not d:
        return key
    return d.get(lang) or d.get("en") or key


def kisan_band(pump_failure_risk_pct: float, stage_of_extraction_pct: float,
               zone_key: str = "safe") -> str:
    """
    Safe / Warning / Pump-failure, from the advisory's own classification.

    The aquifer zone comes from the advisory rather than a threshold we invent
    here, so the dial and the SMS can never disagree. A district can be
    over-exploited while any individual pump still has headroom — the pump
    failure risk is what escalates to red, not the zone.
    """
    if pump_failure_risk_pct >= 50.0:
        return "failure"
    if zone_key in ("over_exploited", "critical") or pump_failure_risk_pct >= 20.0:
        return "warning"
    return "safe"


# ---------------------------------------------------------------------------
def _dial(band: str, level: float, lang: str) -> go.Figure:
    """270° dial gauge — Safe / Warning / Pump failure."""
    col = BAND_I18N[band]["col"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(level),
        number=dict(suffix=" m", font=dict(size=34, color="#F1F7FF")),
        gauge=dict(
            axis=dict(range=[0, 70], tickwidth=1, tickcolor="#41556B",
                      tickfont=dict(size=10, color="#8FA6BF")),
            bar=dict(color=col, thickness=0.72),
            bgcolor="rgba(255,255,255,.04)",
            borderwidth=0,
            steps=[dict(range=[0, 30], color="rgba(34,197,94,.16)"),
                   dict(range=[30, 45], color="rgba(245,158,11,.18)"),
                   dict(range=[45, 70], color="rgba(239,68,68,.20)")],
            threshold=dict(line=dict(color=col, width=3), thickness=0.85,
                           value=float(level))),
    ))
    fig.update_layout(height=210, margin=dict(l=22, r=22, t=8, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E6F4FF"))
    return fig


def _phone_css() -> str:
    return """
<style>
.kisan-wrap{display:flex;justify-content:center}
.kisan{max-width:375px;width:100%;background:#08131f;border:12px solid #16222f;
  border-radius:38px;box-shadow:0 24px 60px rgba(0,0,0,.6);overflow:hidden}
.kisan .bar{background:linear-gradient(90deg,#075E54,#128C7E);color:#fff;
  padding:10px 14px;display:flex;align-items:center;gap:10px}
.kisan .av{width:34px;height:34px;border-radius:50%;background:#25D366;display:flex;
  align-items:center;justify-content:center;font-size:18px;font-weight:800;color:#04231a}
.kisan .who{font-weight:800;font-size:.95rem;line-height:1.15}
.kisan .st{font-size:.72rem;opacity:.85}
.kisan .body{padding:12px}
.kisan .row{display:flex;gap:8px;margin-bottom:9px}
.kisan .box{flex:1;background:rgba(255,255,255,.055);border:1px solid rgba(148,163,184,.16);
  border-radius:14px;padding:10px 11px}
.kisan .lab{font-size:.66rem;text-transform:uppercase;letter-spacing:.07em;
  color:#8FA6BF;font-weight:800}
.kisan .big{font-size:1.5rem;font-weight:800;line-height:1.15;color:#F1F7FF;margin-top:3px}
.kisan .sml{font-size:.75rem;color:#93ACc4;margin-top:2px;line-height:1.4}
.kisan .bub{background:#122d3d;border-radius:13px 13px 13px 4px;padding:11px 13px;
  color:#E6EDF5;font-size:.86rem;line-height:1.6;white-space:pre-wrap;
  border:1px solid rgba(37,211,102,.20);margin-top:9px}
.kisan .meta{text-align:right;color:#5f7d92;font-size:.66rem;margin-top:5px}
.kisan .pill{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.72rem;
  font-weight:800;letter-spacing:.05em}
.pa{font-family:'Nirmala UI','Raavi','Gurbani Akhar',system-ui,sans-serif}
</style>
"""


# ======================================================================================
# VIEW 1 — Kisan mobile
# ======================================================================================
def render_kisan(adv, district: str, critical_windows: list | None = None,
                 key_prefix: str = "kisan", farm_acres: float = 5.0) -> None:
    st.markdown(_phone_css(), unsafe_allow_html=True)

    lang = st.session_state.get(f"{key_prefix}_lang", "en")
    c_l, c_r = st.columns([1, 1])
    with c_l:
        st.markdown("**📱 Kisan Mobile View** · 375 px")
    with c_r:
        new_lang = st.radio("Language / ਭਾਸ਼ਾ", ["en", "pa"],
                            index=0 if lang == "en" else 1,
                            horizontal=True, key=f"{key_prefix}_lang_radio",
                            label_visibility="collapsed",
                            format_func=lambda x: "English" if x == "en" else "ਪੰਜਾਬੀ")
        if new_lang != lang:
            st.session_state[f"{key_prefix}_lang"] = new_lang
            lang = new_lang
    pacls = " pa" if lang == "pa" else ""

    band = kisan_band(adv.pump_failure_risk_pct, adv.stage_of_extraction_pct,
                      getattr(adv, "zone", "safe"))
    bi = BAND_I18N[band]
    # The advisory quotes a per-acre weekly quota; the farmer runs one pump for
    # the whole holding, so show the farm total, capped at the 8 h the feeder
    # actually delivers.
    hours_today = (float(adv.quota_hours_per_week_per_acre)
                   * float(max(farm_acres, 0.5)) / 7.0)
    hours_today = float(np.clip(hours_today, 0.0, 8.0))
    win_start, win_end = _pump_window(hours_today)
    status_txt = bi[lang]
    level = float(adv.current_level_mbgl)

    st.markdown(f"""
<div class="kisan-wrap"><div class="kisan{pacls}">
  <div class="bar">
    <div class="av">💧</div>
    <div><div class="who">{t('title', lang)}</div>
         <div class="st">{t('subtitle', lang)} · {district}</div></div>
  </div>
  <div class="body">
    <div class="row">
      <div class="box"><div class="lab">{t('water_level', lang)}</div>
        <div class="big">{level:.1f}<span style="font-size:.8rem"> m</span></div>
        <div class="sml">{t('metres_below', lang)}</div></div>
      <div class="box"><div class="lab">{t('status', lang)}</div>
        <div style="margin-top:6px"><span class="pill" style="background:{bi['col']}22;
          color:{bi['col']};border:1px solid {bi['col']}66">{status_txt}</span></div>
        <div class="sml">{adv.pump_failure_risk_pct:.0f}% {t('failure', lang).lower()}</div></div>
    </div>
    <div class="row">
      <div class="box"><div class="lab">{t('pump_today', lang)}</div>
        <div class="big">{hours_today:.1f}<span style="font-size:.8rem"> {t('hours', lang)}</span></div>
        <div class="sml">{t('best_window', lang)}: {win_start}–{win_end}<br>
          {farm_acres:.1f} acre holding</div></div>
      <div class="box"><div class="lab">{t('litres_today', lang)}</div>
        <div class="big">{int(adv.quota_litres_per_week_per_acre*farm_acres/7):,}</div>
        <div class="sml">{adv.crop_stage_pa if lang=='pa' else adv.crop_stage}<br>
          extraction {adv.stage_of_extraction_pct:.0f}%</div></div>
    </div>
    <div class="bub">{html.escape(adv.sms_pa if lang == 'pa' else adv.sms_en)}</div>
    <div class="meta">— {t('title', lang)} · {t('district', lang)} {district}</div>
  </div>
</div></div>
""", unsafe_allow_html=True)

    # ---- the dial ----------------------------------------------------------
    st.plotly_chart(_dial(band, level, lang), width='stretch',
                    config={"displayModeBar": False})

    # ---- next critical stage ----------------------------------------------
    st.markdown(f"#### ⏰ {t('next_alert', lang)}")
    if critical_windows:
        for w in critical_windows[:3]:
            days = int((w["date"] - pd.Timestamp.today().normalize()).days)
            st.markdown(
                f"""<div class="card" style="border-left:3px solid {w.get('colour','#38BDF8')}">
<b>{w['date']:%d %b %Y}</b> &nbsp;·&nbsp; {days} days away<br>
{w['crop']} — <b>{w['stage']}</b><br>
<span class="note">Needs about {w['mm_day']:.1f} mm/day at this stage
(Kc {w['kc']}). Missing this irrigation cannot be made up later.</span></div>""",
                unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="note">{t("no_alert", lang)}</div>',
                    unsafe_allow_html=True)

    # ---- WhatsApp push drawer ---------------------------------------------
    with st.expander(f"📲 {t('send', lang)}", expanded=False):
        msg = adv.whatsapp_pa if lang == "pa" else adv.whatsapp_en
        st.code(msg, language=None)
        pa_txt = adv.sms_pa
        st.markdown("**SMS fallback (≤160 chars, Gurmukhi)**")
        st.code(pa_txt, language=None)
        st.caption(f"SMS length: {len(pa_txt)} characters")
        if st.button(f"📤 {t('send', lang)}", key=f"{key_prefix}_push"):
            st.success(f"✅ {t('sent', lang)}")
            st.balloons()


def _pump_window(hours: float) -> tuple[str, str]:
    """
    Choose the pumping window.

    Punjab's agricultural feeders get a rotational supply, and the least-waste
    hours are the ones with the lowest evaporative loss and the least
    competition on the feeder — overnight. We anchor the window at 22:00 and
    extend it by however many hours the quota allows.
    """
    h = float(np.clip(hours, 0.5, 8.0))
    start_h = 22
    end_h = int((start_h + h) % 24)
    return f"{start_h}:00", f"{end_h:02d}:00"


# ======================================================================================
# VIEW 2 — Farmer desktop advisory
# ======================================================================================
def render_farmer(df: pd.DataFrame, adv, district: str,
                  net=None, key_prefix: str = "farmer") -> None:
    from .aquifer_strata import (marginal_cost_curve, deepening_capex,
                                 litholog_stats, salinity_risk, cross_section)
    from .config import get_prior
    from .crop_cycle_model import rotation, diversify, critical_windows, next_windows

    prior = get_prior(district)
    st.markdown("### 🧑‍🌾 Farmer Desktop Advisory")
    st.caption(f"{district} · everything below is computed from this district's "
               f"own data, live")

    T1, T2, T3, T4 = st.tabs(["💧 Water budget", "🔗 Neighbour interference",
                              "🔄 Rotation simulator", "🛠️ Drilling calculator"])

    # ---------------- water budget ----------------
    with T1:
        st.markdown("#### Where your water goes, and where it comes back")
        r = rotation(district, df if df is not None and len(df) else None)
        m = r.meta
        k1, k2, k3, k4 = st.columns(4)
        for col, lab, val, sub in [
            (k1, "Rabi wheat draft", f"{r.seasonal_mcm['wheat']:,.0f} MCM",
             f"{m['seasonal_gw_mm']['wheat']:.0f} mm over {m['wheat_ha']:,.0f} ha"),
            (k2, f"Kharif {m['kharif_crop']} draft",
             f"{r.seasonal_mcm[m['kharif_crop']]:,.0f} MCM",
             f"{m['seasonal_gw_mm'][m['kharif_crop']]:.0f} mm over {m['kharif_ha']:,.0f} ha"),
            (k3, "Zaid (summer) draft", f"{r.seasonal_mcm['zaid']:,.0f} MCM",
             f"{m['zaid_ha']:,.0f} ha at {m['zaid_share']:.0%} of holdings"),
            (k4, "Annual groundwater draft", f"{r.annual_draft_mcm:,.0f} MCM",
             f"canal supplies {m['canal_share_pct']:.0f}% of the rest"),
        ]:
            col.markdown(f"""<div class="kpi"><div class="lab">{lab}</div>
              <div class="val">{val}</div><div class="sub">{sub}</div></div>""",
                         unsafe_allow_html=True)

        st.markdown(f"""
<div class="card"><b>{r.pre_sowing_low_mcm:,.0f} MCM is pumped out between
1 June and 1 November</b> — {100*r.pre_sowing_low_mcm/max(r.annual_draft_mcm,1e-9):.0f}%
of the year's draft, all of it before a single wheat seed goes in.<br>
<span class="note">Calibration: κ<sub>kharif</sub> {m['kappa_kharif']:.3f},
κ<sub>rabi</sub> {m['kappa_rabi']:.3f}. The gap between them is the monsoon
rainfall credit paddy gets and wheat does not.</span></div>""",
                    unsafe_allow_html=True)

        monthly = r.frame.set_index("date")["draft_mcm_day"].resample("MS").sum()
        fig = go.Figure(go.Bar(x=monthly.index, y=monthly.values,
                               marker_color="#0EA5E9"))
        fig.update_layout(title="Monthly groundwater draft (MCM)", height=280,
                          template="plotly_white",
                          margin=dict(l=44, r=16, t=42, b=32),
                          yaxis_title="MCM", xaxis_title=None)
        st.plotly_chart(fig, width='stretch')

        tw = None
        try:
            if df is not None and len(df):
                from .crop_cycle_model import seasonal_trough
                tw = seasonal_trough(df)
        except Exception:
            tw = None
        if tw:
            st.markdown(f"""
<div class="card"><b>The annual cycle, detrended:</b> deepest in
<b>{tw['deepest_label']}</b>, shallowest in <b>{tw['shallowest_label']}</b>,
amplitude <b>{tw['amplitude_m']:.2f} m</b>, on top of a
<b>{tw['trend_m_per_day']*365:.2f} m/yr</b> decline.<br>
<span class="note">The monsoon <i>refills</i> the aquifer; it is the Rabi wheat
season that drains it into the summer trough. The monsoon does not restore —
each year starts deeper than the last.</span></div>""", unsafe_allow_html=True)

    # ---------------- neighbour interference ----------------
    with T2:
        st.markdown("#### Your pump, your neighbour's water level")
        if net is None:
            st.info("Spatial network not loaded for this district.")
        else:
            stn = net.stations.copy()
            if "cumulative_drawdown_m" in stn:
                stn = stn.sort_values("cumulative_drawdown_m", ascending=False)
            top = stn.head(12)
            fig = go.Figure(go.Bar(
                y=[f"{i}" for i in range(len(top))],
                x=top["cumulative_drawdown_m"].to_numpy() if "cumulative_drawdown_m" in top
                else np.zeros(len(top)),
                orientation="h", marker_color="#8B5CF6"))
            fig.update_layout(title="Superposed drawdown at the 12 worst "
                                    "monitoring piezometers (m)",
                              height=300, template="plotly_white",
                              margin=dict(l=50, r=16, t=42, b=28),
                              xaxis_title="metres")
            st.plotly_chart(fig, width='stretch')

            meta = getattr(net, "meta", {}) or {}
            st.markdown(f"""
<div class="card"><b>The number that surprises every farmer:</b> one tubewell
moves a neighbour's piezometer by <b>{meta.get('max_single_coupling_m', 0):.3f} m</b>.
A whole village of {meta.get('bores_per_cluster', 0)} bores moves it by
<b>{meta.get('max_cumulative_drawdown_m', 0):.2f} m</b>.<br>
<span class="note">Nobody's individual pump is the problem and everybody's is.
{len(net.wells):,} bores modelled across {meta.get('n_clusters', 0)} village
clusters ({meta.get('local_bore_density_per_km2', 0):.1f} bores/km², derived
from the district's real tubewell count) — a
{100*meta.get('sample_fraction', 0):.1f}% sample of the
{meta.get('real_tubewells', 0):,} bores actually in the ground, so the true
interference is worse than shown.</span></div>""", unsafe_allow_html=True)

            st.markdown("**What one bore does to its neighbours**")
            d_m = st.slider("Distance to your neighbour's bore (m)", 50, 1500, 300,
                            50, key=f"{key_prefix}_dist")
            try:
                from .spatial_network import single_well_cone
                cone = single_well_cone(district, distance_m=float(d_m))
                st.metric("Drawdown at that distance",
                          f"{cone['drawdown_m']:.3f} m",
                          f"after {cone['days']} days at {cone['Q_m3d']:.0f} m³/d")
            except Exception as exc:
                st.caption(f"Cone calculation unavailable: {exc}")

    # ---------------- rotation simulator ----------------
    with T3:
        st.markdown("#### What if I take some land out of paddy?")
        c1, c2 = st.columns([1, 2])
        with c1:
            switch_ha = st.slider("Hectares shifted out of paddy", 0, 20000, 2000,
                                  100, key=f"{key_prefix}_switch")
            to_crop = st.selectbox("Into", ["maize", "cotton"], key=f"{key_prefix}_to")
            zaid_share = st.slider("Summer (Zaid) crop on % of holdings",
                                   0, 40, 8, 1, key=f"{key_prefix}_zaid")
        with c2:
            dv = diversify(district, df if df is not None and len(df) else None,
                           switch_ha=float(switch_ha), to_crop=to_crop,
                           zaid_share=zaid_share / 100.0)
            st.markdown(f"""
<div class="kpi"><div class="lab">Water saved</div>
<div class="val" style="color:#4ADE80">{dv['saved_mcm']:,.1f} MCM</div>
<div class="sub">{dv['saved_pct']:.1f}% of {district}'s annual groundwater draft ·
{dv['saved_mcm']*1e6/max(dv['switch_ha'],1):,.0f} m³ per hectare
shifted<br><b>+{dv['head_recovered_m']:.3f} m</b> of head recovered across the
district each year</div></div>""", unsafe_allow_html=True)
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute", "relative", "total"],
                x=["Paddy baseline", f"{dv['switch_ha']:,.0f} ha → {to_crop}", "New total"],
                y=[dv["baseline_kharif_mcm"],
                   -(dv["baseline_kharif_mcm"] - dv["switched_kharif_mcm"]),
                   None],
                connector=dict(line=dict(color="#41556B")),
                decreasing=dict(marker=dict(color="#4ADE80")),
                totals=dict(marker=dict(color="#0EA5E9"))))
            fig.update_layout(title="Kharif groundwater draft (MCM)", height=300,
                              template="plotly_white",
                              margin=dict(l=44, r=16, t=42, b=32))
            st.plotly_chart(fig, width='stretch')

        st.markdown("##### Critical stages in the next 60 days")
        cw = next_windows(60, district=district)
        if cw:
            st.dataframe(pd.DataFrame([{
                "Date": w["date"].strftime("%d %b %Y"), "Crop": w["crop"],
                "Stage": w["stage"], "Kc": w["kc"], "mm/day": w["mm_day"],
                "Days away": int((w["date"] - pd.Timestamp.today().normalize()).days)}
                for w in cw]), hide_index=True, width='stretch')
        else:
            st.caption("No critical phenological stage falls in the next 60 days.")

    # ---------------- drilling calculator ----------------
    with T4:
        st.markdown("#### What does it cost to chase the water down?")
        c1, c2 = st.columns([1, 2])
        with c1:
            target = st.slider("Finish the bore at (m)", 20, 200, 75, 5,
                               key=f"{key_prefix}_depth")
            acres = st.slider("Operated area (acres)", 1.0, 25.0,
                              float(prior.avg_pump_hp and 5.0), 0.5,
                              key=f"{key_prefix}_acres")
            hp = st.slider("Pump rating (HP)", 3.0, 30.0, float(prior.avg_pump_hp),
                           0.5, key=f"{key_prefix}_hp")
        with c2:
            cap = deepening_capex(float(target), static_level_mbgl=float(
                adv.current_level_mbgl))
            st.markdown(f"""
<div class="kpi"><div class="lab">Bore finished at {target} m</div>
<div class="val">₹{cap['total_inr']/1e5:.2f} L</div>
<div class="sub">{cap['stratum']} — {cap['stratum_name']} ·
{cap['pump_class']}<br>Needs <b>{cap['required_hp']:.1f} HP</b> ·
{cap['inr_per_ft_total']:.0f} ₹/ft all-in</div></div>""",
                        unsafe_allow_html=True)
            br = pd.DataFrame([
                {"Component": "Drilling", "₹": cap["drilling_inr"]},
                {"Component": "Casing", "₹": cap["casing_inr"]},
                {"Component": "Development & test pump", "₹": cap["development_inr"]},
                {"Component": f"Pump ({cap['required_hp']:.1f} HP)", "₹": cap["pump_inr"]},
                {"Component": "Civil & electrical", "₹": cap["civil_inr"]},
            ])
            fig = go.Figure(go.Bar(x=br["Component"], y=br["₹"],
                                   marker_color="#F59E0B",
                                   text=[f"₹{v:,}" for v in br["₹"]],
                                   textposition="outside"))
            fig.update_layout(height=300, template="plotly_white",
                              margin=dict(l=44, r=16, t=24, b=60),
                              yaxis_title="₹")
            st.plotly_chart(fig, width='stretch')
            if not cap["centrifugal_possible"]:
                st.markdown("""
<div class="card" style="border-left:3px solid #F59E0B"><b>Your centrifugal
pump will not work at this depth.</b> At more than about 8 m of suction lift a
surface pump cannot prime, so this depth forces a submersible — that forced
swap, not the extra drilling, is what makes the step expensive.</div>""",
                            unsafe_allow_html=True)

        mc = marginal_cost_curve()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mc["target_depth_mbgl"], y=mc["total_inr"],
                                 mode="lines", name="Total capex",
                                 line=dict(color="#0EA5E9", width=3)))
        fig.add_trace(go.Scatter(x=mc["target_depth_mbgl"],
                                 y=mc["marginal_inr_per_ft"], mode="lines",
                                 name="Marginal ₹/ft", yaxis="y2",
                                 line=dict(color="#EF4444", width=2, dash="dash")))
        fig.update_layout(title="Cost of a bore, and the marginal ₹/ft of going deeper",
                          height=320, template="plotly_white",
                          yaxis=dict(title="Total ₹"),
                          yaxis2=dict(title="Marginal ₹/ft", overlaying="y",
                                      side="right"),
                          margin=dict(l=52, r=52, t=44, b=40),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width='stretch')

        st.plotly_chart(cross_section(district), width='stretch')

        logs = litholog_stats(district)
        sal = salinity_risk(district, float(target))
        st.markdown(f"""
<div class="note">Measured, from <b>{logs['n_logs']} real CGWB lithologs</b>
({logs['scope']}): median drilled depth
<b>{logs['median_drilled_depth_m']:.0f} m</b>, median transmissivity
<b>{logs['transmissivity']['median']:,.0f} m²/d</b>, median EC
<b>{logs['electrical_conductivity']['median']:.0f} µS/cm</b>.
Saline share (>2,000 µS/cm): <b>{logs['saline_fraction_pct']:.0f}%</b>.<br>
<b>We do not claim deep water is saltier.</b> In this dataset EC correlates with
drilled depth at r = {logs['ec_vs_depth_r']:+.3f} — no relationship. Salinity is
a real hazard, but it is not depth-dependent in the measured record.</div>""",
                    unsafe_allow_html=True)


# ======================================================================================
# VIEW 3 — Administrator / lender extras
# ======================================================================================
def render_admin_extras(df: pd.DataFrame, adv, district: str, net=None,
                        key_prefix: str = "admin") -> None:
    from .benchmarks import benchmark_matrix, headline_improvements
    from .credit_risk_engine import capex_comparison

    st.markdown("### 🏛️ District Administrator & Satin Finserv Portal")
    A1, A2, A3 = st.tabs(["💼 Portfolio at risk", "⚡ Power-supply optimiser",
                          "🏆 Competitive benchmark"])

    with A1:
        st.markdown("#### Portfolio at risk by block")
        try:
            from .well_network import wells_for_district, block_summary
            from .risk_engine import credit_risk, portfolio_view
            wells = wells_for_district(district)
            bs = block_summary(district, wells, pd.DataFrame(), 0.0)
            bs = bs[bs["block"].astype(str).str.lower() != "null"] \
                if "block" in bs else bs
            if "mean_risk_pct" in bs:
                bs = bs.sort_values("mean_risk_pct", ascending=False)
                fig = go.Figure(go.Bar(
                    x=bs["block"].astype(str), y=bs["mean_risk_pct"],
                    marker=dict(color=bs["mean_risk_pct"],
                                colorscale=[[0, "#22C55E"], [.5, "#F59E0B"],
                                            [1, "#EF4444"]]),
                    text=[f"{v:.1f}%" for v in bs["mean_risk_pct"]],
                    textposition="outside"))
                fig.update_layout(title=f"{district} — mean aquifer credit risk by block",
                                  height=340, template="plotly_white",
                                  margin=dict(l=44, r=16, t=44, b=70),
                                  yaxis_title="mean risk score")
                st.plotly_chart(fig, width='stretch')
                st.dataframe(bs, hide_index=True, width='stretch')
        except Exception as exc:
            st.caption(f"Block risk unavailable: {exc}")

        st.markdown("#### Deepening vs drip — which do we finance?")
        cmp_ = capex_comparison(district, current_level_mbgl=float(adv.current_level_mbgl),
                                pump_set_depth_mbgl=float(adv.pump_set_depth_mbgl))
        rows = []
        for tag, k in (("Deepen the bore", "deepen"), ("Drip (on grid)", "drip_only"),
                       ("Drip + solar pump", "drip")):
            r = cmp_[k]
            rows.append(dict(Option=tag, **{"Capex ₹": r["capex_inr"],
                         "Payback": (f"{r['payback_years']:.1f} yr"
                                     if r.get("payback_years") else "—"),
                         "NPV ₹": r["npv_inr"],
                         "Lasts": (f"{r['years_of_relief']:.1f} yr"
                                   if k == "deepen"
                                   else f"+{r['bore_life_extension_yrs']:.1f} yr bore life")}))
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.markdown(f"""
<div class="card"><b>Verdict: {cmp_['verdict'].replace('_',' ')}.</b>
The bore fails in <b>{cmp_['years_to_bore_failure']:.1f} years</b> at
{cmp_['decline_m_per_yr']:.2f} m/yr. Deepening costs ₹
{cmp_['deepen']['capex_inr']:,} but its benefit does not start until year
{cmp_['deepen']['benefit_starts_in_yr']:.1f} — by which time the discount rate
has eaten it. Drip on the existing grid connection costs
₹{cmp_['drip_only']['capex_inr']:,}, pays back in
{cmp_['drip_only']['payback_years']:.1f} years and adds
{cmp_['drip_only']['bore_life_extension_yrs']:.1f} years of bore life.<br>
<span class="note"><b>The solar pump is the worst of the three</b>
(NPV ₹{cmp_['drip']['npv_inr']:,}) — not because solar is bad, but because
Punjab's agricultural power is subsidised to ~zero, so it displaces electricity
the farmer was not paying for.</span></div>""", unsafe_allow_html=True)

    with A2:
        st.markdown("#### Rotational 8-hour supply optimiser")
        st.caption("Punjab's agricultural feeders run on a rotational supply. "
                   "The same 8 hours of power deliver more water when they are "
                   "scheduled against the crop stage and the feeder, not the clock.")
        hours = st.slider("Supply hours per day available", 2, 12, 8, 1,
                          key=f"{key_prefix}_hrs")
        blocks = st.slider("Feeder groups to rotate", 2, 6, 3, 1,
                           key=f"{key_prefix}_grp")
        per = hours / max(blocks, 1)
        sched = []
        for i in range(int(blocks)):
            s = int(22 + i * per) % 24
            e = int(22 + (i + 1) * per) % 24
            sched.append(dict(Group=f"Feeder group {i+1}",
                              **{"Window": f"{s:02d}:00 – {e:02d}:00",
                                 "Hours": round(per, 2)}))
        st.dataframe(pd.DataFrame(sched), hide_index=True, width='stretch')
        from .crop_cycle_model import next_windows
        cw = next_windows(30, district=district)
        if cw:
            st.markdown(f"""
<div class="card" style="border-left:3px solid #F59E0B"><b>Priority override:</b>
{cw[0]['crop']} hits <b>{cw[0]['stage']}</b> on
{cw[0]['date']:%d %b}, needing about {cw[0]['mm_day']:.1f} mm/day. During that
window this stage should get first call on the feeder — a missed CRI or
flowering irrigation cannot be recovered later in the season.</div>""",
                        unsafe_allow_html=True)
        else:
            st.caption("No critical stage in the next 30 days — rotation can "
                       "follow the standing schedule.")

    with A3:
        h = headline_improvements(district)
        k1, k2, k3 = st.columns(3)
        k1.metric("Our spatial input", f"{h['our_raster_cell_m']} m",
                  f"GRACE cell is {h['grace_cell_km2']:,} km²")
        k2.metric("Linear gain vs GRACE", f"{h['linear_gain_vs_grace']}×",
                  f"{h['linear_gain_vs_grace_effective']}× vs smoothed GRACE")
        k3.metric("Forecast lead time", "30 / 60 / 90 d",
                  "GRACE and India-WRIS are retrospective")
        st.dataframe(benchmark_matrix(district).drop(columns=["verdict"]),
                     hide_index=True, width='stretch')
        st.markdown(f"""
<div class="card"><span class="note">{h['note']}</span></div>""",
                    unsafe_allow_html=True)
