"""
AquaCast-Punjab :: Module 4 — Advisory & Agrarian Risk Decision Engine
======================================================================

Turns a 30/60/90-day LSTM forecast into four things a human can act on:

1. **Aquifer Health Zone** — Safe / Critical / Over-Exploited (mbgl thresholds).
2. **Irrigation quota** — recommended tubewell pumping hours per week per acre,
   derived from the FAO-56 demand, the pump's discharge and an aquifer
   sustainability cap.
3. **Pump-failure & days-to-dry-out** — how close the forecast puts the water
   table to the district's typical submersible installation depth.
4. **Micro-finance risk score (1-100)** for the lending partner — see
   :mod:`src.risk_engine`.

Every formula is explicit and documented so that a credit officer or an
agriculture department reviewer can audit the number, not just trust it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import (DISTRICT_PRIORS, WHEAT_STAGES, ZONE_CRITICAL_MAX, ZONE_SAFE_MAX,
                     Scenario, draft_multiplier, get_prior)
from .dataset_generator import rabi_day_of_season
from .i18n import ACTION_COPY, BAND, LABELS, band_key, zone_key
from .risk_engine import credit_risk

# --------------------------------------------------------------------------------------
# engineering constants
# --------------------------------------------------------------------------------------
ACRE_M2 = 4046.86
HA_MM_TO_M3 = 10.0              # 1 mm over 1 hectare == 10 m^3
PUMP_SUCTION_MARGIN_M = 6.0     # submersible is normally set this far below static level
DEFAULT_PUMP_SET_MBGL = 49.0    # typical installation depth for a central-Punjab tubewell
                                # (depth-to-first-aquifer ~19 m + ~30 m of screen)
SUSTAINABLE_STAGE = 1.00        # CGWB stage-of-extraction target (draft == recharge)
MIN_QUOTA_FRACTION = 0.55       # never cut a farmer below 55 % of crop requirement
DIESEL_L_PER_HOUR_HP = 0.20     # ~0.2 L/h per HP for a 7.5 HP diesel set
DIESEL_INR_PER_L = 96.0

# --- pump hydraulics ------------------------------------------------------------------
# A centrifugal/submersible set is rated at *free delivery*, not at 40 m of lift.
# Discharge at the head a Punjab tubewell actually works against is far lower, and
# it falls every year as the water table drops — which is precisely why power
# consumption per acre climbs even when the crop does not change.
PUMP_WIRE_TO_WATER_EFF = 0.55   # combined motor + pump efficiency (well-maintained set)
HP_TO_W = 745.7
PUMP_TDH_MARGIN_M = 10.0        # friction losses + delivery head above the static level
TYPICAL_EVENT_MM = 60.0         # one flood irrigation on a Punjab wheat field
ELECTRICITY_KWH_PER_HP_H = 0.85
ELECTRICITY_INR_PER_KWH = 0.0   # Punjab agricultural power is subsidised to ~zero


@dataclass
class Advisory:
    """Everything the dashboard / SMS / PDF needs. JSON-serialisable."""
    district: str
    issue_date: str
    horizon_days: int
    target_date: str
    scenario: str
    # state
    current_level_mbgl: float
    predicted_level_mbgl: float
    predicted_delta_m: float
    lower_mbgl: float
    upper_mbgl: float
    # classification
    zone: str
    zone_label_en: str
    zone_label_pa: str
    zone_colour: str
    # pumping
    required_depth_mm_week: float
    pump_discharge_m3h: float
    quota_hours_per_week_per_acre: float
    quota_litres_per_week_per_acre: int
    quota_hours_per_event: float
    quota_events_per_week: float
    quota_days_between_events: float
    sustainability_factor: float
    quota_note_en: str
    quota_note_pa: str
    # failure
    pump_set_depth_mbgl: float
    headroom_m: float
    pump_failure_risk_pct: float
    days_to_dryout: float | None
    # water budget
    projected_draft_mcm: float
    projected_recharge_mcm: float
    deficit_mcm: float
    stage_of_extraction_pct: float
    # crop / money
    crop_stage: str
    crop_stage_pa: str
    water_saved_pct: float
    cost_saved_inr_per_acre: float
    # credit
    risk_score: float
    risk_band: str
    risk_band_pa: str
    risk_band_colour: str
    pd_pct: float
    expected_loss_inr: float
    exposure_inr: float
    credit_action_en: str
    credit_action_pa: str
    # messaging
    actions: list[dict]
    sms_en: str
    sms_pa: str
    whatsapp_en: str
    whatsapp_pa: str
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ======================================================================================
# helpers
# ======================================================================================
def classify_zone(level_mbgl: float) -> dict:
    k = zone_key(level_mbgl)
    from .i18n import ZONES
    z = ZONES[k]
    return dict(key=k, label_en=z["en"], label_pa=z["pa"], colour=z["colour"],
                emoji=z["emoji"])


def crop_stage_for(date: pd.Timestamp) -> dict:
    """
    Human-readable wheat phenology for the forecast target date.

    Driven by ``rabi_day_of_season`` — the *same* day-of-season index that builds
    the FAO-56 Kc curve in Module 2b — so the label always agrees with the crop
    coefficient that produced the water demand.  The boundaries are taken from
    ``WHEAT_STAGES`` (initial / development / mid / late).

    Anchored on calendar day-of-year it silently mislabelled late-November
    through December ("Fallow") because the Rabi season wraps the year end.
    """
    dos = float(rabi_day_of_season(np.array([int(date.dayofyear)]))[0])
    if np.isnan(dos):
        return dict(en="Fallow / Kharif paddy", pa="ਖਾਲੀ / ਸਾਉਣੀ ਝੋਨਾ", stage="Fallow")
    if dos <= WHEAT_STAGES["initial_end"]:
        return dict(en="Crown-root initiation (CRI)", pa="ਕਰਾਊਨ-ਰੂਟ (CRI) ਪੜਾਅ", stage="CRI")
    if dos <= WHEAT_STAGES["dev_end"]:
        return dict(en="Tillering / canopy development", pa="ਕੱਲੀਆਂ ਨਿਕਲਣਾ", stage="Tillering")
    if dos <= 110:
        return dict(en="Jointing / stem elongation", pa="ਗੰਢ ਬਣਨਾ", stage="Jointing")
    if dos <= WHEAT_STAGES["mid_end"]:
        return dict(en="Flowering — most water-sensitive", pa="ਫੁੱਲ ਆਉਣਾ — ਸਭ ਤੋਂ ਸੰਵੇਦਨਸ਼ੀਲ",
                    stage="Flowering")
    return dict(en="Milk / dough grain fill", pa="ਦਾਣਾ ਭਰਨਾ", stage="Grain fill")


def weekly_requirement_mm(df: pd.DataFrame, target_date: pd.Timestamp,
                          prior, scenario: Scenario) -> float:
    """
    Crop water that must actually be *pumped*, mm/week, at the forecast target date.

    The dataset's demand columns are built under the baseline flood scenario, so
    the scenario lever enters here as a draft multiplier: micro-irrigation lifts
    application efficiency, so the same crop needs less water out of the ground.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d.date)
    # climatological week centred on the target date (+/- 3 d), averaged over years
    doy = int(target_date.dayofyear)
    sel = d[(d.date.dt.dayofyear - doy).abs() <= 3]
    if sel.empty:
        sel = d
    depth_mm_day = float((sel.crop_water_demand_mcm + sel.paddy_demand_mcm).mean() * 1e6
                         / max(prior.wheat_area_m2, 1.0) * 1000.0)
    return max(depth_mm_day * 7.0 * draft_multiplier(scenario), 0.0)


def quota_hours(required_mm_week: float, pump_discharge_m3h: float,
                sustainability: float, farm_acres: float = 1.0) -> tuple[float, float, int]:
    """
    Pumping hours per week for one acre.

    hours = (mm/week x 10 m3 per ha-mm) x (acre in ha) / pump discharge
    then scaled by the aquifer sustainability factor.
    """
    ha = farm_acres * ACRE_M2 / 1e4
    m3 = required_mm_week * HA_MM_TO_M3 * ha
    hours_raw = m3 / max(pump_discharge_m3h, 1.0)
    factor = float(np.clip(0.60 + 0.40 * sustainability, MIN_QUOTA_FRACTION, 1.0))
    hours = hours_raw * factor
    return float(hours), float(factor), int(m3 * factor * 1000.0)


def effective_pump_discharge_m3h(prior, water_level_mbgl: float) -> float:
    """
    Discharge (m³/h) the district's average pump set can actually deliver **at
    today's head** — Q = η·P / (ρ·g·H).

    Rating a 7.5 HP set at 45 m³/h (its free-delivery nameplate) overstates
    delivery by ~80 % against a 36 m water table, which silently halves every
    pumping-hour and diesel-cost figure the advisory quotes.  Deeper water means
    higher total dynamic head, less discharge, more hours and more energy per
    acre — the mechanism that makes depletion a *cost* story, not just a
    hydrology story.  Never exceeds the nameplate rating.
    """
    hp = float(getattr(prior, "avg_pump_hp", 7.5) or 7.5)
    head = max(float(water_level_mbgl), 3.0) + PUMP_TDH_MARGIN_M
    q = PUMP_WIRE_TO_WATER_EFF * hp * HP_TO_W / (1000.0 * 9.81 * head) * 3600.0
    nameplate = float(getattr(prior, "avg_pump_discharge_m3h", 45.0) or 45.0)
    return float(np.clip(q, 5.0, nameplate))


def irrigation_schedule(hours_per_week_per_acre: float, discharge_m3h: float,
                        event_mm: float = TYPICAL_EVENT_MM) -> tuple[float, float, float]:
    """
    Translate an abstract "hours per week" into the unit a farmer actually plans
    in: *how many hours to run the pump each time, and how many days to wait*.

    Returns ``(hours_per_event, events_per_week, days_between_events)``.
    """
    q = max(float(discharge_m3h), 1.0)
    m3_per_event_per_acre = event_mm * HA_MM_TO_M3 * (ACRE_M2 / 1e4)
    hours_per_event = m3_per_event_per_acre / q
    events_per_week = hours_per_week_per_acre / max(hours_per_event, 1e-6)
    days_between = 7.0 / events_per_week if events_per_week > 1e-6 else float("inf")
    return float(hours_per_event), float(events_per_week), float(days_between)


def pump_failure_risk(predicted_level: float, pump_set_depth: float) -> tuple[float, float]:
    """
    Probability the tubewell loses suction before the forecast date.

    Risk ramps linearly from 0 % at 12 m of headroom to 100 % at 0 m.
    """
    headroom = pump_set_depth - predicted_level
    if headroom <= 0:
        return 100.0, headroom
    return float(np.clip((1.0 - headroom / 12.0) * 100.0, 0.0, 100.0)), float(headroom)


def days_to_dryout(current_level: float, pump_set_depth: float,
                   rate_m_per_day: float) -> float | None:
    if rate_m_per_day <= 1e-6:
        return None
    return float(max((pump_set_depth - current_level) / rate_m_per_day, 0.0))


# ======================================================================================
# main entry point
# ======================================================================================
def build_advisory(
    df: pd.DataFrame,
    district: str,
    forecast: dict,
    horizon_days: int,
    scenario: Scenario,
    pump_set_depth_mbgl: float | None = None,
    farm_acres: float = 1.0,
    language: str = "both",
) -> Advisory:
    """
    Convert one forecast (see ``model_lstm.predict``) into a full advisory.

    Parameters
    ----------
    df          : the district daily frame (used for crop demand & water budget)
    forecast    : dict with ``mean``/``std`` arrays of shape (H,) for one issue date
    horizon_days: 30 / 60 / 90
    """
    prior = get_prior(district)
    d = df.copy()
    d["date"] = pd.to_datetime(d.date)

    issue_date = pd.Timestamp(d.date.max())
    target_date = issue_date + pd.Timedelta(days=int(horizon_days))
    j = list([30, 60, 90]).index(int(horizon_days))

    current_level = float(d.gw_level_mbgl.iloc[-1])
    pred = float(np.ravel(forecast["mean"])[j])
    sd = float(np.ravel(forecast.get("std", np.zeros_like(forecast["mean"])))[j])
    delta = pred - current_level

    lower, upper = pred - 1.96 * sd, pred + 1.96 * sd
    zone = classify_zone(pred)

    # ---------------- water budget over the horizon ---------------------------------
    horizon_slice = d[d.date > issue_date]          # future: use scenario-scaled demand
    future_days = int(horizon_days)
    if len(horizon_slice) >= future_days:
        fut = horizon_slice.iloc[:future_days]
    else:
        fut = d.iloc[-future_days:]
    eff_gain = 0.60 / max(scenario.irrigation_efficiency, 0.30)
    adoption = np.clip(scenario.pump_adoption_drip_pct, 0, 100) / 100.0
    draft_mult = (1 - adoption) + adoption * eff_gain
    monsoon_mult = 1.0 + scenario.monsoon_anomaly_pct / 100.0
    is_monsoon = fut.month.isin([6, 7, 8, 9]).to_numpy()
    draft_mcm = float((fut.gw_draft_mcm.sum()) * draft_mult)
    recharge_mcm = float((fut.total_recharge_mcm.to_numpy()
                          * np.where(is_monsoon, monsoon_mult, 1.0)).sum()
                         * (scenario.canal_availability_pct / 100.0) ** 0.35)
    deficit = draft_mcm - recharge_mcm
    stage = 100.0 * draft_mcm / max(recharge_mcm, 1e-6)
    sustainability = float(np.clip(SUSTAINABLE_STAGE * recharge_mcm / max(draft_mcm, 1e-6),
                                   0.0, 1.6))

    # ---------------- irrigation quota ------------------------------------------------
    req_mm = weekly_requirement_mm(d, target_date, prior, scenario)
    # Discharge is derated to the head the pump actually works against at the
    # *forecast* water table (7.5 HP set, ~0.55 wire-to-water efficiency).
    # Using the free-delivery nameplate silently halves every pumping-hour and
    # energy-cost figure the advisory quotes.
    q_m3h = effective_pump_discharge_m3h(prior, pred)
    hours, factor, litres = quota_hours(req_mm, q_m3h, sustainability, farm_acres)
    h_event, ev_week, d_between = irrigation_schedule(hours, q_m3h)

    # ---------------- pump failure ----------------------------------------------------
    pump_set = float(pump_set_depth_mbgl or DEFAULT_PUMP_SET_MBGL)
    pf_risk, headroom = pump_failure_risk(pred, pump_set)
    rate_m_per_day = delta / max(horizon_days, 1)
    dryout_days = days_to_dryout(current_level, pump_set, max(rate_m_per_day, 1e-6))

    # ---------------- savings vs flood -------------------------------------------------
    water_saved_pct = float(np.clip((1 - draft_mult) * 100, 0, 100))
    diesel_saved_l = (DIESEL_L_PER_HOUR_HP * prior.avg_pump_hp
                      * max(hours * (1 - factor) if factor < 1 else 0.0, 0.0))
    cost_saved = float(diesel_saved_l * DIESEL_INR_PER_L
                       + ELECTRICITY_KWH_PER_HP_H * prior.avg_pump_hp
                       * max(hours * (1 - factor), 0.0) * ELECTRICITY_INR_PER_KWH)

    # ---------------- credit risk -------------------------------------------------------
    cr = credit_risk(district=district, predicted_level=pred, current_level=current_level,
                     zone=zone["key"], pump_failure_risk_pct=pf_risk,
                     drawdown_m=delta, deficit_mcm=deficit, stage_pct=stage,
                     days_to_dryout=dryout_days, horizon_days=horizon_days,
                     crop_stage=crop_stage_for(target_date)["stage"],
                     canal_share_pct=prior.canal_share_pct,
                     drip_adoption_pct=scenario.pump_adoption_drip_pct)

    # ---------------- action list --------------------------------------------------------
    st = crop_stage_for(target_date)
    actions = _actions(pred, zone["key"], pf_risk, sustainability, cr, st, dryout_days)

    sms_en, sms_pa = _sms(district, current_level, pred, target_date, hours, pf_risk,
                          zone, actions, "en", h_event, d_between), None
    sms_pa = _sms(district, current_level, pred, target_date, hours, pf_risk, zone,
                  actions, "pa", h_event, d_between)
    wa_en = _whatsapp(district, issue_date, target_date, horizon_days, current_level,
                      pred, lower, upper, zone, hours, pf_risk, cr, actions, st, "en",
                      h_event, d_between, q_m3h)
    wa_pa = _whatsapp(district, issue_date, target_date, horizon_days, current_level,
                      pred, lower, upper, zone, hours, pf_risk, cr, actions, st, "pa",
                      h_event, d_between, q_m3h)

    note_en, note_pa = _quota_note(factor, sustainability, req_mm)
    band = BAND[band_key(cr["risk_score"])]

    return Advisory(
        district=district, issue_date=str(issue_date.date()), horizon_days=int(horizon_days),
        target_date=str(target_date.date()), scenario=scenario.name,
        current_level_mbgl=round(current_level, 2), predicted_level_mbgl=round(pred, 2),
        predicted_delta_m=round(delta, 2), lower_mbgl=round(float(lower), 2),
        upper_mbgl=round(float(upper), 2),
        zone=zone["key"], zone_label_en=zone["label_en"], zone_label_pa=zone["label_pa"],
        zone_colour=zone["colour"],
        required_depth_mm_week=round(req_mm, 1),
        pump_discharge_m3h=round(q_m3h, 1),
        quota_hours_per_week_per_acre=round(hours, 1),
        quota_litres_per_week_per_acre=int(litres),
        quota_hours_per_event=round(h_event, 1),
        quota_events_per_week=round(ev_week, 2),
        quota_days_between_events=(round(d_between, 1)
                                   if np.isfinite(d_between) and d_between < 400 else None),
        sustainability_factor=round(factor, 2),
        quota_note_en=note_en, quota_note_pa=note_pa,
        pump_set_depth_mbgl=round(pump_set, 1), headroom_m=round(headroom, 2),
        pump_failure_risk_pct=round(pf_risk, 1),
        days_to_dryout=None if dryout_days is None else round(float(dryout_days), 0),
        projected_draft_mcm=round(draft_mcm, 1),
        projected_recharge_mcm=round(recharge_mcm, 1),
        deficit_mcm=round(deficit, 1), stage_of_extraction_pct=round(stage, 0),
        crop_stage=st["en"], crop_stage_pa=st["pa"],
        water_saved_pct=round(water_saved_pct, 1),
        cost_saved_inr_per_acre=round(cost_saved, 0),
        risk_score=round(cr["risk_score"], 1), risk_band=cr["band_en"],
        risk_band_pa=cr["band_pa"], risk_band_colour=band["colour"],
        pd_pct=round(cr["pd_pct"], 2), expected_loss_inr=round(cr["expected_loss_inr"], 0),
        exposure_inr=round(cr["exposure_inr"], 0),
        credit_action_en=cr["action_en"], credit_action_pa=cr["action_pa"],
        actions=actions, sms_en=sms_en, sms_pa=sms_pa,
        whatsapp_en=wa_en, whatsapp_pa=wa_pa,
        extras=dict(components=cr["components"], confidence_sd_m=round(sd, 3),
                    acres=farm_acres, horizon_days=int(horizon_days)),
    )


# ======================================================================================
# copy generation
# ======================================================================================
def _actions(pred, zone_key_, pf_risk, sustainability, cr, stage, dryout_days) -> list[dict]:
    out = []
    if pf_risk >= 60 or (dryout_days is not None and dryout_days < 400):
        out.append(dict(id="deepen", icon="🚨", **ACTION_COPY["deepen"]))
    if zone_key_ == "over_exploited":
        out.append(dict(id="reduce", icon="⛔", **ACTION_COPY["reduce"]))
        out.append(dict(id="drip", icon="💧", **ACTION_COPY["drip"]))
    elif zone_key_ == "critical":
        out.append(dict(id="drip", icon="💧", **ACTION_COPY["drip"]))
    else:
        out.append(dict(id="ok", icon="✅", **ACTION_COPY["ok"]))
    if stage["stage"] in ("CRI", "Flowering", "Jointing"):
        out.append(dict(id="irrigate", icon="🌾", **ACTION_COPY["irrigate"]))
    if cr["risk_score"] >= 62:
        out.append(dict(id="credit_watch", icon="🏦", **ACTION_COPY["credit_watch"]))
    return out[:4]


def _quota_note(factor, sustainability, req_mm) -> tuple[str, str]:
    if factor < 0.8:
        return (f"Restricted quota: the aquifer can only sustain {sustainability:.0%} of the "
                f"projected draft. Priority to the {req_mm:.0f} mm/week crop need at the most "
                f"water-sensitive stage.",
                f"ਸੀਮਤ ਕੋਟਾ: ਧਰਤੀ ਹੇਠਲਾ ਪਾਣੀ ਸਿਰਫ਼ {sustainability:.0%} ਖਿੱਚ ਸਹਿ ਸਕਦਾ ਹੈ।")
    if factor > 0.98:
        return (f"Full crop requirement of {req_mm:.0f} mm/week can be met — draft is within "
                f"sustainable recharge.",
                f"{req_mm:.0f} ਮਿਮੀ/ਹਫ਼ਤਾ ਪੂਰੀ ਲੋੜ ਪੂਰੀ ਕੀਤੀ ਜਾ ਸਕਦੀ ਹੈ।")
    return (f"Slightly reduced quota — draft is {1/max(sustainability,1e-6):.0%} of recharge.",
            f"ਥੋੜ੍ਹੀ ਕਟੌਤੀ — ਖਿੱਚ ਰੀਚਾਰਜ ਦਾ {1/max(sustainability,1e-6):.0%} ਹੈ।")


def _sms(district, cur, pred, target_date, hours, pf, zone, actions, lang,
         h_event=0.0, d_between=0.0) -> str:
    L = lambda k: LABELS[k][lang]
    sched = bool(h_event) and np.isfinite(d_between) and d_between < 400
    gap_en = f" ({h_event:.0f} h every {d_between:.0f} d)" if sched else ""
    gap_pa = f" ({h_event:.0f} ਘੰਟੇ ਹਰ {d_between:.0f} ਦਿਨਾਂ ਬਾਅਦ)" if sched else ""
    if lang == "pa":
        return (f"ਐਕੁਆਕਾਸਟ | {district}\n"
                f"ਪਾਣੀ ਦਾ ਪੱਧਰ: {cur:.1f} → {pred:.1f} ਮੀਟਰ ਹੇਠਾਂ ({target_date:%d %b})\n"
                f"ਹਾਲਤ: {zone['label_pa']} {zone['emoji']}\n"
                f"ਟਿਊਬਵੈੱਲ: {hours:.1f} ਘੰਟੇ/ਹਫ਼ਤਾ/ਏਕੜ{gap_pa}\n"
                f"ਪੰਪ ਖ਼ਤਰਾ: {pf:.0f}%\n"
                f"{actions[0]['pa']}\n"
                f"ਬੰਦ ਕਰਨ ਲਈ STOP ਭੇਜੋ")
    return (f"AquaCast | {district}\n"
            f"Water table: {cur:.1f} -> {pred:.1f} m bgl ({target_date:%d %b})\n"
            f"Zone: {zone['label_en']} {zone['emoji']}\n"
            f"Pump quota: {hours:.1f} h/week/acre{gap_en}\n"
            f"Pump risk: {pf:.0f}%\n"
            f"{actions[0]['en']}\n"
            f"Reply STOP to opt out")


def _whatsapp(district, issue, target, horizon, cur, pred, lo, hi, zone, hours, pf, cr,
              actions, stage, lang, h_event=0.0, d_between=0.0, q_m3h=45.0) -> str:
    sched = bool(h_event) and np.isfinite(d_between) and d_between < 400
    sched_en = (f" (~{h_event:.0f} h run, every {d_between:.0f} d)" if sched else "")
    sched_pa = (f" (~{h_event:.0f} \u0a18\u0a70\u0a1f\u0a47, \u0a39\u0a30 {d_between:.0f} \u0a26\u0a3f\u0a28\u0a3e\u0a02 \u0a2c\u0a3e\u0a05\u0a26)" if sched else "")
    if lang == "pa":
        return (
            f"*{LABELS['aqua_alert']['pa']} — {district}*\n"
            f"📅 {issue:%d %b %Y} → {target:%d %b %Y} (+{horizon} ਦਿਨ)\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💧 {LABELS['water_level']['pa']}: *{cur:.1f} → {pred:.1f}* {LABELS['metres_below']['pa']}\n"
            f"📊 95% ਰੇਂਜ: {lo:.1f} – {hi:.1f} ਮੀਟਰ\n"
            f"{zone['emoji']} {LABELS['zone']['pa']}: *{zone['label_pa']}*\n"
            f"🚰 {LABELS['pump_hours']['pa']}: *{hours:.1f}* {LABELS['hours_per_week']['pa']}{sched_pa}\n"
            f"   ਇਸ ਡੂੰਘਾਈ ਤੇ ਪੰਪ ਡਿਲੀਵਰੀ: {q_m3h:.0f} ਮੀ³/ਘੰਟਾ\n"
            f"⚠️ {LABELS['pump_risk']['pa']}: *{pf:.0f}%*\n"
            f"🌾 {stage['pa']}\n"
            f"🏦 {LABELS['credit']['pa']}: {cr['risk_score']:.0f}/100 ({cr['band_pa']})\n"
            f"━━━━━━━━━━━━━━━\n"
            + "\n".join(f"{a['icon']} {a['pa']}" for a in actions)
            + f"\n📞 {LABELS['helpline']['pa']}: 1800-XXX-XXXX"
        )
    return (
        f"*{LABELS['aqua_alert']['en']} — {district}*\n"
        f"📅 {issue:%d %b %Y} → {target:%d %b %Y} (+{horizon} d)\n"
        f"━━━━━━━━━━━━━━━\n"
        f"💧 {LABELS['water_level']['en']}: *{cur:.1f} → {pred:.1f}* m bgl\n"
        f"📊 95% range: {lo:.1f} – {hi:.1f} m\n"
        f"{zone['emoji']} {LABELS['zone']['en']}: *{zone['label_en']}*\n"
        f"🚰 {LABELS['pump_hours']['en']}: *{hours:.1f}* h/week/acre{sched_en}\n"
        f"   pump delivery at this depth: {q_m3h:.0f} m³/h\n"
        f"⚠️ {LABELS['pump_risk']['en']}: *{pf:.0f}%*\n"
        f"🌾 {stage['en']}\n"
        f"🏦 {LABELS['credit']['en']}: {cr['risk_score']:.0f}/100 ({cr['band_en']})\n"
        f"━━━━━━━━━━━━━━━\n"
        + "\n".join(f"{a['icon']} {a['en']}" for a in actions)
        + f"\n📞 {LABELS['helpline']['en']}: 1800-XXX-XXXX"
    )


if __name__ == "__main__":
    from .dataset_generator import load_dataset
    from .model_lstm import load_model, predict, live_window
    df = load_dataset("Sangrur")
    res = load_model()
    Xl, base, issue, tdates, rte = live_window(
        df, trend_rate_m_per_day=get_prior("Sangrur").long_term_decline_m_per_yr / 365.25)
    fc = predict(res, Xl, base, mc_passes=40, trend_rate_m_per_day=rte)
    adv = build_advisory(df, "Sangrur", fc, 90, Scenario())
    print(adv.whatsapp_en)
    print()
    print(adv.whatsapp_pa)
