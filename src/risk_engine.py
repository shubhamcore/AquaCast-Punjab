"""
AquaCast-Punjab :: Micro-Finance Risk Engine (Satin Finserv evaluation track)
=============================================================================

Translates a hydrogeological forecast into a **credit** decision.

Why a lender should care
------------------------
In central Punjab the collateral behind a crop loan is not land title — it is
*the water under it*.  When a tubewell loses suction mid-Rabi, the wheat fails,
the borrower's cash flow stops, and a performing KCC turns into a restructured
account.  AquaCast forecasts that failure 30-90 days early, which is exactly the
window a lender needs to intervene (top-up, drip-irrigation refinance, or
pre-harvest monitoring) instead of writing the account off.

Scoring model (fully auditable, no black box)
---------------------------------------------
Six weighted components, each normalised to 0-100, summed and clipped to 1-100:

============  ===========================================  ======
weight        component                                    source
============  ===========================================  ======
0.26          aquifer depth & zone                         LSTM t+90 level
0.22          pump-failure probability                     headroom vs install depth
0.16          forecast drawdown over the horizon           LSTM delta
0.14          water-budget deficit / stage of extraction   physics water balance
0.12          crop-stage sensitivity                       FAO-56 phenology
0.10          monsoon recharge anomaly                     rainfall telemetry
-0.06         adaptive capacity (canal + drip)             district priors
============  ===========================================  ======

Score to probability of default (PD)::

    PD(score) = PD_floor + (score/100)^2 * (PD_ceiling - PD_floor)

a deliberately convex mapping: risk accelerates once the water table is already
past the critical threshold.  Expected loss = exposure x PD x LGD.
"""
from __future__ import annotations

import numpy as np

from .config import DISTRICT_PRIORS, ZONE_CRITICAL_MAX, ZONE_SAFE_MAX, get_prior
from .i18n import BAND, band_key

# --- calibration ------------------------------------------------------------------------
PD_FLOOR = 0.9            # % PD for a very safe aquifer
PD_CEILING = 23.0         # % PD when the aquifer is fully over-exploited
LGD_CROP_LOAN = 0.60      # loss-given-default on unsecured/partly secured crop loans
EARLY_WARNING_EFFICACY = 0.22   # share of expected loss a 90-day warning can avert

CROP_STAGE_SENSITIVITY = {
    "CRI": 0.75,           # crown-root initiation: irrigation timing decides tillering
    "Tillering": 0.70,
    "Jointing": 0.85,
    "Flowering": 1.00,     # terminal drought at anthesis is the worst case
    "Grain fill": 0.65,
    "Fallow": 0.25,
}

CREDIT_ACTIONS = {
    "low": {
        "en": "No action — standard KCC renewal cycle. Offer a water-smart top-up at renewal.",
        "pa": "ਕੋਈ ਕਾਰਵਾਈ ਨਹੀਂ — ਸਧਾਰਨ KCC ਨਵੀਨੀਕਰਨ।",
    },
    "moderate": {
        "en": "Watchlist — include the account in the quarterly water-risk review; nudge "
              "drip/sprinkler adoption.",
        "pa": "ਨਿਗਰਾਨੀ ਸੂਚੀ — ਤਿਮਾਹੀ ਪਾਣੀ-ਜੋਖਮ ਸਮੀਖਿਆ ਵਿੱਚ ਸ਼ਾਮਲ ਕਰੋ।",
    },
    "elevated": {
        "en": "Pre-harvet monitoring + micro-irrigation refinance offer; cap fresh top-ups "
              "at 60% of the ticket.",
        "pa": "ਵਾਢੀ ਤੋਂ ਪਹਿਲਾਂ ਨਿਗਰਾਨੀ + ਸੂਖਮ-ਸਿੰਚਾਈ ਰਿਫ਼ਾਈਨੈਂਸ।",
    },
    "high": {
        "en": "Restructure: re-phase the instalment to post-harvest, release a contingent "
              "tube-well repair / deepening facility.",
        "pa": "ਪੁਨਰਗਠਨ: ਕਿਸ਼ਤ ਨੂੰ ਵਾਢੀ ਤੋਂ ਬਾਅਦ ਲਈ ਮੁਲਤਵੀ ਕਰੋ।",
    },
    "severe": {
        "en": "Escalate to the regional risk committee; freeze fresh exposure, prioritise "
              "borewell-deepening / canal-link emergency credit.",
        "pa": "ਖੇਤਰੀ ਜੋਖਮ ਕਮੇਟੀ ਨੂੰ ਭੇਜੋ; ਨਵੀਂ ਐਕਸਪੋਜ਼ਰ ਰੋਕੋ।",
    },
}


def _clip(x, lo=0.0, hi=100.0) -> float:
    return float(np.clip(x, lo, hi))


def _depth_component(level_mbgl: float) -> float:
    """0 at <=20 m, 55 at the 30 m 'critical' line, 88 at the 35 m 'over-exploited'
    line, 100 at >=45 m."""
    pts = [(20, 0), (30, 55), (35, 88), (45, 100)]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return _clip(float(np.interp(level_mbgl, xs, ys)))


def credit_risk(district: str, predicted_level: float, current_level: float,
                zone: str, pump_failure_risk_pct: float, drawdown_m: float,
                deficit_mcm: float, stage_pct: float, days_to_dryout,
                horizon_days: int, crop_stage: str = "Fallow",
                canal_share_pct: float = 0.0, drip_adoption_pct: float = 0.0,
                monsoon_anomaly_pct: float = 0.0) -> dict:
    """Return the full credit-risk verdict for one district-horizon pair."""
    prior = get_prior(district)

    # ---- components ------------------------------------------------------------------
    c_depth = _depth_component(predicted_level)
    c_pump = _clip(pump_failure_risk_pct)
    # 1.5 m of drawdown inside a 90-day horizon is a full-risk signal
    c_draw = _clip(abs(drawdown_m) / max(horizon_days, 30) * 30.0 / 1.5 * 100.0)
    c_stage = _clip((stage_pct - 90.0) / 80.0 * 100.0)          # 90% -> 0, 170% -> 100
    c_crop = _clip(CROP_STAGE_SENSITIVITY.get(crop_stage, 0.5) * 100.0)
    c_monsoon = _clip(50.0 - monsoon_anomaly_pct * 1.6)         # -30% rain -> 98
    c_adapt = _clip(np.clip(canal_share_pct * 0.9 + drip_adoption_pct * 0.35, 0, 100))

    weights = dict(depth=0.26, pump=0.22, drawdown=0.16, stage=0.14,
                   crop=0.12, monsoon=0.10, adapt=-0.06)
    comps = dict(depth=c_depth, pump=c_pump, drawdown=c_draw, stage=c_stage,
                 crop=c_crop, monsoon=c_monsoon, adapt=c_adapt)
    raw = (weights["depth"] * c_depth + weights["pump"] * c_pump
           + weights["drawdown"] * c_draw + weights["stage"] * c_stage
           + weights["crop"] * c_crop + weights["monsoon"] * c_monsoon
           + weights["adapt"] * c_adapt)

    # hard overrides: a dry pump inside the horizon is a near-certain crop failure
    if days_to_dryout is not None and days_to_dryout < horizon_days:
        raw = max(raw, 88.0)
    if zone == "over_exploited":
        raw = max(raw, 62.0)

    score = float(np.clip(raw, 1.0, 100.0))
    band = band_key(score)

    pd_pct = PD_FLOOR + (score / 100.0) ** 2 * (PD_CEILING - PD_FLOOR)
    exposure = prior.active_farm_loans * prior.avg_ticket_inr
    expected_loss = exposure * (pd_pct / 100.0) * LGD_CROP_LOAN
    loss_avoidable = expected_loss * EARLY_WARNING_EFFICACY

    act = CREDIT_ACTIONS[band]
    return dict(
        risk_score=round(score, 1),
        band=band, band_en=BAND[band]["en"], band_pa=BAND[band]["pa"],
        band_colour=BAND[band]["colour"],
        components={k: round(v, 1) for k, v in comps.items()},
        weights=weights,
        pd_pct=round(pd_pct, 2),
        exposure_inr=round(exposure, 0),
        expected_loss_inr=round(expected_loss, 0),
        loss_avoidable_inr=round(loss_avoidable, 0),
        lgd=LGD_CROP_LOAN,
        action_en=act["en"], action_pa=act["pa"],
    )


def portfolio_view(district_risk: dict[str, dict]) -> dict:
    """Roll per-district verdicts up into a lender's portfolio picture."""
    if not district_risk:
        return {}
    exposure = sum(v["exposure_inr"] for v in district_risk.values())
    el = sum(v["expected_loss_inr"] for v in district_risk.values())
    avoid = sum(v["loss_avoidable_inr"] for v in district_risk.values())
    wavg_score = sum(v["risk_score"] * v["exposure_inr"] for v in district_risk.values()) / max(exposure, 1)
    return dict(
        districts=len(district_risk),
        exposure_inr=round(exposure, 0),
        expected_loss_inr=round(el, 0),
        par_pct=round(100 * el / max(exposure, 1), 2),
        loss_avoidable_inr=round(avoid, 0),
        weighted_avg_risk_score=round(wavg_score, 1),
        per_district={k: dict(score=v["risk_score"], band=v["band_en"],
                              pd_pct=v["pd_pct"], exposure_inr=v["exposure_inr"],
                              expected_loss_inr=v["expected_loss_inr"])
                      for k, v in district_risk.items()},
    )
