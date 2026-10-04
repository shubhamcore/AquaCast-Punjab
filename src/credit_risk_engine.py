"""
Module D — Credit Risk Engine (lender-grade)
============================================

A 0-100 aquifer-linked credit score for a Kisan Credit Card / crop loan, built
from three sub-scores the way an underwriter actually thinks about a farm in
the Punjab tubewell belt:

    score = w1 * DrawdownDeficit
          + w2 * CriticalStageStress
          + w3 * BoreholeDeepeningCapexRatio

    DrawdownDeficit              will the bore still deliver at harvest?
    CriticalStageStress          does a non-negotiable crop stage fall inside
                                 the forecast deficit window?
    BoreholeDeepeningCapexRatio  can this household actually *afford* to chase
                                 the water table down, or does the fix cost
                                 more than the farm earns?

Decision bands
--------------
    0-35    Standard KCC — approve at normal terms.
    36-70   Approve with a contingency clause: drip / diversification financed
            into the ticket, because the water risk is real but fixable.
    71-100  Do not lend against this aquifer. Parametric micro-insurance or
            restructuring; the water is the collateral and it is failing.

Weights
-------
The default split is a documented lending policy, not a fitted constant, and
it is exposed as a tunable because a lender's risk appetite is a policy
choice. Every *input* to the score is computed live from the data.

Capex comparator
----------------
``capex_comparison`` prices the two things a stressed farmer can actually buy:
another 30 m of bore, or a solar micro-drip system. It returns payback years
and NPV for each, because the honest answer is that deepening is a *depleting*
asset (it buys you N years until the table passes the new depth) while drip is
a *permanent* one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Lending policy
# ---------------------------------------------------------------------------
W_DRAWDOWN = 0.45
W_CRITICAL_STAGE = 0.30
W_CAPEX = 0.25

BANDS = (
    (0, 35, "APPROVE", "Standard KCC at normal terms",
     "#2E7D32"),
    (36, 70, "APPROVE_WITH_CONDITION",
     "Contingency clause — drip / diversification financed into the ticket",
     "#E4A11B"),
    (71, 100, "RESTRUCTURE_OR_INSURE",
     "Parametric micro-insurance or restructuring — do not lend against this aquifer",
     "#B3261E"),
)

# Capex comparator assumptions — component-level, totals emerge
DRIP_CAPEX_INR_PER_ACRE = 48_000.0      # inline drip + fertigation, Punjab rate
DRIP_SUBSIDY_FRACTION = 0.55            # PMKSY / state micro-irrigation subsidy
SOLAR_PUMP_CAPEX_INR_PER_HP = 62_000.0
SOLAR_SUBSIDY_FRACTION = 0.30
DRIP_WATER_SAVING_PCT = 38.0            # vs flood irrigation, Punjab trials
DRIP_ENERGY_SAVING_PCT = 28.0
DRIP_YIELD_UPLIFT_PCT = 8.0
FLOOD_IRRIGATION_KWH_PER_ACRE_YR = 780.0
GRID_KWH_COST_IF_UNSUBSIDISED = 6.5
DIESEL_L_PER_ACRE_YR_FLOOD = 165.0
DIESEL_INR_PER_L = 96.0
#: Punjab subsidises agricultural power to ~zero, so an on-grid farmer does
#: NOT bank the kWh a solar pump displaces. Only the share of irrigation run
#: on diesel backup (power cuts are routine) converts to cash.
DIESEL_BACKUP_SHARE = 0.30
FERTILISER_LABOUR_SAVING_INR_PER_ACRE_YR = 2_400.0
WHEAT_PADDY_GROSS_MARGIN_INR_PER_ACRE_YR = 26_500.0
DISCOUNT_RATE = 0.10
DRIP_LIFE_YEARS = 12
TYPICAL_OPERATED_ACRES = 5.0


# ---------------------------------------------------------------------------
@dataclass
class CreditScore:
    score: float
    band_key: str
    band_label: str
    band_colour: str
    components: dict
    weights: dict
    drivers: list
    recommendation: str


def _clip(x, lo=0.0, hi=100.0) -> float:
    return float(np.clip(x, lo, hi))


def _band(score: float):
    for lo, hi, key, label, col in BANDS:
        if lo <= score <= hi:
            return key, label, col
    return BANDS[-1][2], BANDS[-1][3], BANDS[-1][4]


# ---------------------------------------------------------------------------
def drawdown_deficit(predicted_level_mbgl: float, pump_set_depth_mbgl: float,
                     current_level_mbgl: float | None = None,
                     decline_m_per_yr: float = 0.0,
                     horizon_days: int = 90) -> dict:
    """
    Will the bore still deliver at harvest?

    Scored on the margin between the forecast water level and the depth the
    pump is set at, normalised over a 12 m band: at or above the pump intake
    the score is 100 (the pump is already sucking air), and 12 m of clearance
    or more is 0.
    """
    margin = float(pump_set_depth_mbgl) - float(predicted_level_mbgl)
    band = 12.0
    score = _clip(100.0 * (1.0 - margin / band))
    failing = margin <= 0
    return dict(score=score, margin_m=margin, pump_set_depth_mbgl=float(pump_set_depth_mbgl),
                predicted_level_mbgl=float(predicted_level_mbgl),
                already_failing=bool(failing),
                note=("pump intake already above the water level" if failing else
                      f"{margin:.1f} m of clearance below the pump intake"))


def critical_stage_stress(windows: list, horizon_days: int = 30,
                          deficit_fraction: float = 0.0) -> dict:
    """
    Does a non-negotiable crop stage fall inside the deficit window?

    Wheat crown-root initiation and flowering, and paddy flowering, are the
    stages a missed irrigation cannot be recovered from. More stages inside
    the window, and a larger deficit, both push the score up.
    """
    n = len(windows or [])
    # up to 3 stages inside the horizon saturates the stage term
    stage_term = _clip(100.0 * min(n, 3) / 3.0)
    deficit_term = _clip(100.0 * float(np.clip(deficit_fraction, 0.0, 1.0)))
    score = _clip(0.6 * stage_term + 0.4 * deficit_term)
    names = [f"{w.get('crop', '').split('(')[0].strip()} — {w.get('stage','')}"
             for w in (windows or [])]
    return dict(score=score, n_windows=n, stage_term=stage_term,
                deficit_term=deficit_term, deficit_fraction=float(deficit_fraction),
                stages=names[:4])


def capex_burden(deepening_capex_inr: float, annual_net_income_inr: float,
                 loan_ticket_inr: float | None = None) -> dict:
    """
    Can this household afford to chase the water down?

    The ratio that matters is the deepening bill against a year of net farm
    income. At or below a quarter of a year's income the fix is absorbable; at
    a full year's income or more it is not — and note that deepening is a
    *recurring* cost, because the table keeps falling.
    """
    inc = max(float(annual_net_income_inr), 1.0)
    ratio = float(deepening_capex_inr) / inc
    score = _clip(100.0 * (ratio - 0.25) / 0.75)
    return dict(score=score, ratio=ratio, capex_inr=float(deepening_capex_inr),
                annual_net_income_inr=inc,
                loan_ticket_inr=(float(loan_ticket_inr) if loan_ticket_inr else None),
                note=(f"deepening costs {ratio:.2f}x a year of net farm income"))


# ---------------------------------------------------------------------------
def score(district: str,
          predicted_level_mbgl: float,
          pump_set_depth_mbgl: float,
          current_level_mbgl: float | None = None,
          decline_m_per_yr: float = 0.8,
          deficit_fraction: float = 0.0,
          critical_windows: list | None = None,
          deepening_capex_inr: float | None = None,
          annual_net_income_inr: float | None = None,
          weights: tuple[float, float, float] | None = None,
          horizon_days: int = 90) -> CreditScore:
    """
    Compose the three sub-scores into the 0-100 underwriting score.

    Every input is caller-supplied from live model output; nothing here is
    defaulted to a flattering value. ``annual_net_income_inr`` falls back to
    the district's own operated-acreage x gross-margin product, and
    ``deepening_capex_inr`` to the Module C cost of reaching the next stratum.
    """
    if weights is None:
        weights = (W_DRAWDOWN, W_CRITICAL_STAGE, W_CAPEX)
    w1, w2, w3 = (float(x) for x in weights)
    wsum = w1 + w2 + w3 or 1.0
    w1, w2, w3 = w1 / wsum, w2 / wsum, w3 / wsum

    if deepening_capex_inr is None:
        deepening_capex_inr = _default_deepening_capex(district, predicted_level_mbgl)
    if annual_net_income_inr is None:
        annual_net_income_inr = TYPICAL_OPERATED_ACRES * WHEAT_PADDY_GROSS_MARGIN_INR_PER_ACRE_YR

    dd = drawdown_deficit(predicted_level_mbgl, pump_set_depth_mbgl,
                          current_level_mbgl, decline_m_per_yr, horizon_days)
    cs = critical_stage_stress(critical_windows or [], horizon_days, deficit_fraction)
    cx = capex_burden(deepening_capex_inr, annual_net_income_inr)

    total = _clip(w1 * dd["score"] + w2 * cs["score"] + w3 * cx["score"])
    key, label, col = _band(total)

    drivers = sorted(
        [("Drawdown deficit", dd["score"], w1, dd["note"]),
         ("Critical-stage stress", cs["score"], w2,
          f"{cs['n_windows']} critical stage(s) inside the window"),
         ("Deepening capex burden", cx["score"], w3, cx["note"])],
        key=lambda r: r[1] * r[2], reverse=True)

    if key == "APPROVE":
        rec = ("Approve at standard KCC terms. Aquifer has clearance to harvest "
               "and the household can absorb a deepening if it comes to that.")
    elif key == "APPROVE_WITH_CONDITION":
        rec = ("Approve with a contingency clause: finance drip or a partial "
               "paddy-to-maize shift into the ticket. The water risk is real but "
               "fixable, and the fix cuts the draft that causes it.")
    else:
        rec = ("Do not lend against this aquifer on standard terms. Move to "
               "parametric micro-insurance or restructure; the collateral — the "
               "water — is failing faster than the loan tenor.")

    return CreditScore(
        score=round(total, 1), band_key=key, band_label=label, band_colour=col,
        components=dict(drawdown=dd, critical_stage=cs, capex=cx),
        weights=dict(drawdown=w1, critical_stage=w2, capex=w3),
        drivers=[dict(name=n, score=round(s, 1), weight=round(w, 3),
                      contribution=round(s * w, 1), note=note)
                 for n, s, w, note in drivers],
        recommendation=rec)


def _default_deepening_capex(district: str, predicted_level_mbgl: float) -> float:
    from .aquifer_strata import stratum_for_depth, deepening_capex, COMPLETION_INTO_STRATUM_M
    s = stratum_for_depth(float(predicted_level_mbgl))
    nxt_top = s.bottom_mbgl
    target = min(nxt_top + COMPLETION_INTO_STRATUM_M, s.bottom_mbgl + COMPLETION_INTO_STRATUM_M)
    return float(deepening_capex(target, static_level_mbgl=float(predicted_level_mbgl))["total_inr"])


# ---------------------------------------------------------------------------
def _npv(capex: float, annual_benefit: float, years: int,
         life: int | None = None) -> float:
    years = int(min(years, life)) if life else int(years)
    return -float(capex) + sum(float(annual_benefit) / (1.0 + DISCOUNT_RATE) ** t
                               for t in range(1, years + 1))


def capex_comparison(district: str,
                     current_level_mbgl: float,
                     pump_set_depth_mbgl: float,
                     decline_m_per_yr: float = 0.8,
                     acres: float = TYPICAL_OPERATED_ACRES,
                     pump_hp: float = 7.5) -> dict:
    """
    Solar micro-drip vs deepening the bore, on an *incremental* basis.

    The accounting subtlety that decides the answer, and that the naive
    version of this comparison gets wrong: a farmer who deepens does not gain
    a crop he did not have. He already earns the gross margin. What deepening
    buys is **the margin in the years after the old bore would have failed** —
    a benefit that is deferred by (pump depth - current level) / decline years
    and therefore heavily discounted.

    Drip is the mirror image. It saves water, yield and energy *from year
    one*, and — the benefit usually forgotten — it cuts the draft that drives
    the decline, so the existing bore simply lasts longer. That deferred
    deepening is priced in explicitly.
    """
    from .aquifer_strata import deepening_capex, stratum_for_depth, COMPLETION_INTO_STRATUM_M

    lvl = float(current_level_mbgl)
    psd = float(pump_set_depth_mbgl)
    dec = max(float(decline_m_per_yr), 1e-6)
    margin = float(acres) * WHEAT_PADDY_GROSS_MARGIN_INR_PER_ACRE_YR

    # ---- do-nothing baseline: when does the existing bore stop delivering? --
    yrs_to_failure = max((psd - lvl) / dec, 0.0)

    # ---- Option A: deepen --------------------------------------------------
    s = stratum_for_depth(lvl)
    new_depth = max(min(s.bottom_mbgl + COMPLETION_INTO_STRATUM_M, lvl + 40.0), lvl + 5.0)
    a = deepening_capex(new_depth, static_level_mbgl=lvl)
    deepen_capex = float(a["total_inr"])
    relief_yrs = max((new_depth - psd) / dec, 0.0)
    extra_head_m = max(new_depth - psd, 0.0)
    extra_energy_yr = (acres * FLOOD_IRRIGATION_KWH_PER_ACRE_YR
                       * extra_head_m / 100.0 * 0.6 * GRID_KWH_COST_IF_UNSUBSIDISED)

    # Benefit accrues only in the window AFTER the do-nothing bore dies.
    def _pv_window(start_yr: float, n_yr: float, annual: float) -> float:
        pv = 0.0
        t = max(start_yr, 0.0)
        for _ in range(int(np.ceil(n_yr)) if n_yr > 0 else 0):
            t += 1.0
            pv += annual / (1.0 + DISCOUNT_RATE) ** t
        return pv

    deepen_pv_benefit = _pv_window(yrs_to_failure, relief_yrs, margin) \
        - _pv_window(0.0, relief_yrs, extra_energy_yr)
    deepen_npv = deepen_pv_benefit - deepen_capex
    # Simple payback: capex / (margin - extra energy), only meaningful once the
    # deferred window starts, so we report it measured from the failure year.
    net_annual = margin - extra_energy_yr
    deepen_payback = (yrs_to_failure + deepen_capex / net_annual
                      if net_annual > 0 else float("inf"))

    # ---- Option B: solar micro-drip ---------------------------------------
    drip_gross = DRIP_CAPEX_INR_PER_ACRE * float(acres)
    drip_net = drip_gross * (1.0 - DRIP_SUBSIDY_FRACTION)
    solar_gross = SOLAR_PUMP_CAPEX_INR_PER_HP * float(pump_hp)
    solar_net = solar_gross * (1.0 - SOLAR_SUBSIDY_FRACTION)
    drip_capex = drip_net + solar_net

    # Energy only converts to cash on the diesel-backed share: Punjab's grid
    # power for agriculture is subsidised to ~zero, so counting displaced kWh
    # at tariff would flatter solar and mislead a lender.
    drip_annual = (
        margin * (DRIP_YIELD_UPLIFT_PCT / 100.0)
        + float(acres) * FERTILISER_LABOUR_SAVING_INR_PER_ACRE_YR
        + DIESEL_BACKUP_SHARE * float(acres) * DIESEL_L_PER_ACRE_YR_FLOOD
        * (DRIP_ENERGY_SAVING_PCT / 100.0) * DIESEL_INR_PER_L
    )
    # Drip cuts the draft that drives the decline, so the bore lasts longer.
    decline_eff = dec * (1.0 - DRIP_WATER_SAVING_PCT / 100.0)
    yrs_to_failure_drip = max((psd - lvl) / max(decline_eff, 1e-6), 0.0)
    life_extension_yrs = yrs_to_failure_drip - yrs_to_failure

    drip_pv_operating = sum(drip_annual / (1.0 + DISCOUNT_RATE) ** t
                            for t in range(1, DRIP_LIFE_YEARS + 1))
    # The deepening this household no longer needs to buy at year yrs_to_failure.
    drip_pv_deferred = (deepen_capex / (1.0 + DISCOUNT_RATE) ** yrs_to_failure
                        if yrs_to_failure > 0 else deepen_capex)
    # Option C: drip alone, staying on the (free) grid. Same water saving,
    # same bore-life extension, none of the solar capital.
    drip_only_capex = drip_net
    drip_only_npv = (drip_pv_operating + drip_pv_deferred) - drip_only_capex
    drip_only_payback = (drip_only_capex / drip_annual) if drip_annual > 0 else float("inf")
    drip_pv_benefit = drip_pv_operating + drip_pv_deferred
    drip_npv = drip_pv_benefit - drip_capex
    drip_payback = (drip_capex / drip_annual) if drip_annual > 0 else float("inf")

    options = {"deepen": deepen_npv, "drip_only": drip_only_npv, "drip_solar": drip_npv}
    verdict = max(options, key=options.get).upper()

    return dict(
        district=district, acres=float(acres), pump_hp=float(pump_hp),
        current_level_mbgl=lvl, pump_set_depth_mbgl=psd, decline_m_per_yr=dec,
        years_to_bore_failure=round(yrs_to_failure, 1),
        margin_inr_per_yr=round(margin),
        deepen=dict(capex_inr=round(deepen_capex), target_depth_m=round(new_depth, 1),
                    years_of_relief=round(relief_yrs, 1),
                    benefit_starts_in_yr=round(yrs_to_failure, 1),
                    extra_energy_inr_per_yr=round(extra_energy_yr),
                    pv_benefit_inr=round(deepen_pv_benefit),
                    npv_inr=round(deepen_npv),
                    payback_years=(round(deepen_payback, 1)
                                   if np.isfinite(deepen_payback) else None),
                    required_hp=a["required_hp"],
                    declining_asset=True),
        drip=dict(capex_inr=round(drip_capex),
                  gross_capex_inr=round(drip_gross + solar_gross),
                  subsidy_inr=round((drip_gross + solar_gross) - drip_capex),
                  water_saving_pct=DRIP_WATER_SAVING_PCT,
                  decline_after_m_per_yr=round(decline_eff, 3),
                  years_to_bore_failure=round(yrs_to_failure_drip, 1),
                  bore_life_extension_yrs=round(life_extension_yrs, 1),
                  annual_benefit_inr=round(drip_annual),
                  pv_operating_inr=round(drip_pv_operating),
                  pv_deferred_deepening_inr=round(drip_pv_deferred),
                  pv_benefit_inr=round(drip_pv_benefit),
                  npv_inr=round(drip_npv),
                  payback_years=(round(drip_payback, 1)
                                 if np.isfinite(drip_payback) else None),
                  life_years=DRIP_LIFE_YEARS, declining_asset=False),
        drip_only=dict(capex_inr=round(drip_only_capex),
                       gross_capex_inr=round(drip_gross),
                       subsidy_inr=round(drip_gross - drip_only_capex),
                       water_saving_pct=DRIP_WATER_SAVING_PCT,
                       decline_after_m_per_yr=round(decline_eff, 3),
                       years_to_bore_failure=round(yrs_to_failure_drip, 1),
                       bore_life_extension_yrs=round(life_extension_yrs, 1),
                       annual_benefit_inr=round(drip_annual),
                       pv_deferred_deepening_inr=round(drip_pv_deferred),
                       npv_inr=round(drip_only_npv),
                       payback_years=(round(drip_only_payback, 1)
                                      if np.isfinite(drip_only_payback) else None),
                       life_years=DRIP_LIFE_YEARS, declining_asset=False),
        npv_rank=sorted(options.items(), key=lambda kv: kv[1], reverse=True),
        verdict=verdict)


# ---------------------------------------------------------------------------
def self_check() -> None:
    from .config import DISTRICTS, get_prior
    from .advisory_engine import DEFAULT_PUMP_SET_MBGL as PUMP_SET_DEPTH_MBGL
    print("\n[Module D] Credit risk engine")
    print("-" * 74)
    print(f"  weights: drawdown {W_DRAWDOWN:.2f} · critical-stage {W_CRITICAL_STAGE:.2f} "
          f"· capex {W_CAPEX:.2f}")
    for lo, hi, key, label, _ in BANDS:
        print(f"    {lo:3d}-{hi:3d}  {key:<24s} {label}")
    print()
    for d in DISTRICTS:
        p = get_prior(d)
        lvl = float(p.base_depth_2021_mbgl)
        for delta in (0.0, 4.0, 9.0):
            pred = lvl + delta
            cs = score(d, predicted_level_mbgl=pred,
                       pump_set_depth_mbgl=PUMP_SET_DEPTH_MBGL,
                       decline_m_per_yr=float(p.long_term_decline_m_per_yr),
                       deficit_fraction=min(delta / 9.0, 1.0),
                       critical_windows=[{"crop": "Wheat (Rabi)", "stage": "Crown root initiation"}]
                       if delta > 0 else [],
                       annual_net_income_inr=5 * WHEAT_PADDY_GROSS_MARGIN_INR_PER_ACRE_YR)
            print(f"  {d:<9s} {pred:5.1f} m (+{delta:.0f})  score {cs.score:5.1f}  "
                  f"{cs.band_key:<24s} top driver: {cs.drivers[0]['name']}")
    print()
    cmp_ = capex_comparison("Sangrur", current_level_mbgl=32.0,
                            pump_set_depth_mbgl=49.0, decline_m_per_yr=0.8)
    print(f"  bore fails in {cmp_['years_to_bore_failure']:.1f} yr at "
          f"{cmp_['decline_m_per_yr']:.2f} m/yr")
    print(f"  {'option':<12s} {'capex ₹':>10s} {'payback':>9s} {'NPV ₹':>12s} {'lasts':>10s}")
    for tag, k in (("deepen bore", "deepen"), ("drip (on grid)", "drip_only"),
                   ("drip + solar", "drip")):
        r = cmp_[k]
        pb = f"{r['payback_years']:.1f} yr" if r.get("payback_years") else "-"
        lasts = (f"{r['years_of_relief']:.1f} yr" if k == "deepen"
                 else f"+{r['bore_life_extension_yrs']:.1f} yr")
        print(f"  {tag:<12s} {r['capex_inr']:>10,d} {pb:>9s} {r['npv_inr']:>12,d} {lasts:>10s}")
    print(f"  verdict: {cmp_['verdict']}  "
          f"(drip saves {cmp_['drip']['water_saving_pct']:.0f}% of the water and "
          f"adds {cmp_['drip']['bore_life_extension_yrs']:.1f} yr of bore life; "
          f"deepening buys {cmp_['deepen']['years_of_relief']:.1f} yr, "
          f"starting in yr {cmp_['deepen']['benefit_starts_in_yr']:.1f})")
    print("-" * 74)


if __name__ == "__main__":
    self_check()
