"""
Module C — Stratified Aquifer Model & Deepening Economics
=========================================================

Three operational hydrostratigraphic units, the pump technology each one
demands, and the cash a farmer must find to chase the water table down into
the next one.

    L1  Shallow unconfined      0 - 40 mbgl    centrifugal / 5-10 HP
    L2  Semi-confined          40 - 90 mbgl    submersible / 15-25 HP
    L3  Deep confined            > 90 mbgl     high-head submersible / 25+ HP

The unit *boundaries* are operational — they are the depths at which the pump
technology and the bore design change, which is what determines capex. The
unit *properties* are measured: transmissivity, storage coefficient, static
water level and electrical conductivity are pulled from the real CGWB
**Litholog** layer (305 bore logs, 25 in Sangrur, 21 in Ludhiana), not assumed.

Two honest findings that contradict the brief, both surfaced in the UI
-----------------------------------------------------------------------
1. **Real Punjab bores are far deeper than the schematic.** The lithologs give
   a median drilled depth of ~207-306 m, not 90 m. The 0-40 / 40-90 / >90 m
   tiers describe the *shallow* system that smallholders actually pump from;
   the aquifer system itself continues several hundred metres down.

2. **Salinity does not increase with depth in this dataset.** EC vs drilled
   depth correlates at r = 0.03 (n = 96) — i.e. not at all. 7% of logs exceed
   the 2,000 µS/cm saline threshold, but they are not concentrated in the deep
   bores. Deep salinity is therefore flagged as a literature risk, never as a
   finding of this dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Operational strata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stratum:
    key: str
    name: str
    top_mbgl: float
    bottom_mbgl: float
    pump_class: str
    hp_range: tuple[float, float]
    description: str
    colour: str

    def contains(self, depth_mbgl: float) -> bool:
        return self.top_mbgl <= depth_mbgl < self.bottom_mbgl

    @property
    def thickness_m(self) -> float:
        return self.bottom_mbgl - self.top_mbgl


L1 = Stratum("L1", "Shallow unconfined", 0.0, 40.0,
             "Centrifugal / shallow jet", (5.0, 10.0),
             "The phreatic aquifer every Punjab village started with. Seasonally "
             "recharged by the monsoon, and the first to go dry. Below ~8 m "
             "suction lift a surface centrifugal pump simply stops priming.",
             "#7FB069")

L2 = Stratum("L2", "Semi-confined", 40.0, 90.0,
             "Submersible", (15.0, 25.0),
             "Reached by most of central Punjab already. Needs a bore with a "
             "proper gravel pack and a submersible set; the capital step from "
             "L1 is the single largest on-farm investment a smallholder makes.",
             "#E4A11B")

L3 = Stratum("L3", "Deep confined", 90.0, 400.0,
             "High-head submersible", (25.0, 40.0),
             "Below the fresh-water lens in places; expensive to drill, slow to "
             "recharge, and effectively a one-way decision — you cannot come "
             "back up when the water table recovers.",
             "#B4553F")

STRATA = (L1, L2, L3)
STRATA_BY_KEY = {s.key: s for s in STRATA}

# ---------------------------------------------------------------------------
# Cost model — every figure is a component; totals emerge, they are not asserted
# ---------------------------------------------------------------------------
M_PER_FT = 0.3048

DRILLING_INR_PER_FT = {"L1": 150.0, "L2": 230.0, "L3": 320.0}
CASING_INR_PER_FT = 420.0          # 6" uPVC, the Punjab standard
CASING_TOP_ONLY_M = 15.0           # shallow bores case only the unconsolidated top
DEVELOPMENT_INR = 15_000.0         # gravel shrouding, surging, test pumping
PUMP_INR_PER_HP = {"Centrifugal / shallow jet": 2_000.0,
                   "Submersible": 3_600.0,
                   "High-head submersible": 4_200.0}
CIVIL_ELECTRICAL_INR = 25_000.0    # plinth, starter, cable, connection

PUMP_EFFICIENCY = 0.55             # combined motor + pump, wire-to-water
SUCTION_LIFT_LIMIT_M = 8.0         # practical limit for a surface centrifugal set
DRAWDOWN_ALLOWANCE_M = 8.0         # design drawdown inside the bore
DELIVERY_MARGIN_M = 10.0           # friction + delivery head above static level
DESIGN_DISCHARGE_M3H = 25.0        # a working Punjab tubewell, not a nameplate figure

SALINE_EC_THRESHOLD = 2_000.0      # µS/cm — above this, irrigation water is marginal
COMPLETION_INTO_STRATUM_M = 35.0   # bores finish this far into a stratum, not at its base


def stratum_for_depth(depth_mbgl: float) -> Stratum:
    for s in STRATA:
        if s.contains(depth_mbgl):
            return s
    return STRATA[-1]


def required_hp(static_level_mbgl: float, discharge_m3h: float = DESIGN_DISCHARGE_M3H,
                drawdown_m: float = DRAWDOWN_ALLOWANCE_M,
                margin_m: float = DELIVERY_MARGIN_M) -> float:
    """Pump shaft power from hydraulic demand: P = rho g Q H, metric form."""
    head = float(static_level_mbgl) + float(drawdown_m) + float(margin_m)
    return float(discharge_m3h) * head / (270.0 * PUMP_EFFICIENCY)


def deepening_capex(target_depth_mbgl: float,
                    discharge_m3h: float = DESIGN_DISCHARGE_M3H,
                    static_level_mbgl: float | None = None) -> dict:
    """
    Itemised cost of a bore finished at ``target_depth_mbgl``.

    Returns the component breakdown, the total, the rupees per foot of *total*
    bore, the pump duty required, and whether a centrifugal set is still
    physically possible at that depth (it is not once the static level drops
    past ~8 m of suction lift).
    """
    target = float(max(target_depth_mbgl, 1.0))
    swl = float(static_level_mbgl) if static_level_mbgl is not None else min(target - 2.0, target * 0.9)
    swl = max(swl, 1.0)

    drilling = 0.0
    for s in STRATA:
        top = max(s.top_mbgl, 0.0)
        bot = min(s.bottom_mbgl, target)
        if bot > top:
            drilling += (bot - top) / M_PER_FT * DRILLING_INR_PER_FT[s.key]

    stratum = stratum_for_depth(target)
    case_len_m = min(target, CASING_TOP_ONLY_M) if stratum.key == "L1" else target
    casing = case_len_m / M_PER_FT * CASING_INR_PER_FT

    hp = required_hp(swl, discharge_m3h)
    pump_key = stratum.pump_class
    # A centrifugal set can only be used while the water is within suction
    # lift. Once the table drops past ~8 m it will not prime at all, and the
    # farmer is forced onto a submersible — that forced swap, not the extra
    # drilling, is what makes the L1->L2 step so expensive for a smallholder.
    centrifugal_ok = (stratum.key == "L1") and (swl <= SUCTION_LIFT_LIMIT_M)
    if not centrifugal_ok and pump_key == "Centrifugal / shallow jet":
        pump_key = "Submersible"
    pump = hp * PUMP_INR_PER_HP[pump_key]

    total = drilling + casing + DEVELOPMENT_INR + pump + CIVIL_ELECTRICAL_INR
    return dict(target_depth_mbgl=target, stratum=stratum.key,
                stratum_name=stratum.name, pump_class=pump_key,
                centrifugal_possible=bool(centrifugal_ok),
                required_hp=round(hp, 2),
                drilling_inr=round(drilling), casing_inr=round(casing),
                development_inr=round(DEVELOPMENT_INR), pump_inr=round(pump),
                civil_inr=round(CIVIL_ELECTRICAL_INR), total_inr=round(total),
                inr_per_ft_total=round(total / (target / M_PER_FT), 1),
                inr_lakh=round(total / 1e5, 2))


def marginal_cost_curve(depth_grid: np.ndarray | None = None,
                        discharge_m3h: float = DESIGN_DISCHARGE_M3H,
                        static_level_mbgl: float | None = None) -> pd.DataFrame:
    """Capex and rupees-per-marginal-foot across a depth sweep."""
    if depth_grid is None:
        depth_grid = np.arange(20.0, 200.1, 2.5)
    rows = []
    prev_d, prev_c = None, None
    for d in depth_grid:
        r = deepening_capex(float(d), discharge_m3h,
                            static_level_mbgl=static_level_mbgl if static_level_mbgl is not None
                            else min(d - 2.0, d * 0.9))
        marg = np.nan if prev_d is None else (r["total_inr"] - prev_c) / ((d - prev_d) / M_PER_FT)
        rows.append({**r, "marginal_inr_per_ft": marg})
        prev_d, prev_c = d, r["total_inr"]
    return pd.DataFrame(rows)


def transition_cost(from_stratum: str = "L1", to_stratum: str = "L2",
                    discharge_m3h: float = DESIGN_DISCHARGE_M3H) -> dict:
    """
    What it costs a farmer to chase the table from one stratum into the next:
    the deepening itself, plus the mandatory pump swap.
    """
    a, b = STRATA_BY_KEY[from_stratum], STRATA_BY_KEY[to_stratum]
    # Bores are completed some way *into* a stratum, not at its base — pricing
    # to the bottom of L3 (400 m) would invent a 10 Lakh bore nobody drills.
    da = min(a.bottom_mbgl, a.top_mbgl + COMPLETION_INTO_STRATUM_M)
    db = min(b.bottom_mbgl, b.top_mbgl + COMPLETION_INTO_STRATUM_M)
    ca = deepening_capex(da, discharge_m3h, da - 2.0)
    cb = deepening_capex(db, discharge_m3h, db - 2.0)
    delta = cb["total_inr"] - ca["total_inr"]
    return dict(from_stratum=from_stratum, to_stratum=to_stratum,
                from_inr=ca["total_inr"], to_inr=cb["total_inr"],
                delta_inr=delta, delta_lakh=round(delta / 1e5, 2),
                from_pump=ca["pump_class"], to_pump=cb["pump_class"],
                from_hp=ca["required_hp"], to_hp=cb["required_hp"],
                pump_swap_required=ca["pump_class"] != cb["pump_class"])


# ---------------------------------------------------------------------------
# Real litholog measurements
# ---------------------------------------------------------------------------
_LITH_NUM = ("Depth_of_Drill__m_", "Depth_of_Construction__mbgl_", "Depth_of_Bedrock__mbgl_",
             "Static_Water_Level__mbgl_", "Transmissivity__m2_day_", "Storage_coefficient",
             "Electrical_Conductivity__Micro_Seimens__cm_", "Chloride__mg_l_",
             "Discharge__lps_", "Draw_Down__m_", "Specific_Capacity__lpm_m_DD_",
             "Nitrate__mg_l_", "Fluoride__mg_l_", "Iron__mg_l_")


@lru_cache(maxsize=8)
def _litholog_raw() -> pd.DataFrame:
    from .spatial_loader import load_layer
    g = load_layer("litholog")
    d = pd.DataFrame(g.drop(columns="geometry"))
    for c in _LITH_NUM:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.replace(0, np.nan)


@lru_cache(maxsize=32)
def litholog_stats(district: str | None = None) -> dict:
    """
    Measured aquifer properties from the real CGWB Litholog layer.

    Falls back from the district to the whole state when a district has too few
    logs to be meaningful (Moga has 5), and says so in ``scope``.
    """
    d = _litholog_raw()
    if district:
        sub = d[d["District"].astype(str).str.strip().str.lower() == district.lower()]
    else:
        sub = d
    scope = f"{district} logs" if district and len(sub) >= 10 else "all Punjab logs"
    if len(sub) < 10:
        sub = d

    def stat(col):
        s = sub[col].dropna() if col in sub.columns else pd.Series(dtype=float)
        return dict(n=int(len(s)),
                    median=float(s.median()) if len(s) else np.nan,
                    p25=float(s.quantile(.25)) if len(s) else np.nan,
                    p75=float(s.quantile(.75)) if len(s) else np.nan)

    ec = sub["Electrical_Conductivity__Micro_Seimens__cm_"].dropna() \
        if "Electrical_Conductivity__Micro_Seimens__cm_" in sub.columns else pd.Series(dtype=float)
    dd = sub["Depth_of_Drill__m_"].dropna()
    swl = sub["Static_Water_Level__mbgl_"].dropna()
    common = sub.dropna(subset=["Electrical_Conductivity__Micro_Seimens__cm_", "Depth_of_Drill__m_"]) \
        if len(ec) else sub.iloc[0:0]
    ec_depth_r = float(common["Depth_of_Drill__m_"].corr(
        common["Electrical_Conductivity__Micro_Seimens__cm_"])) if len(common) > 3 else np.nan

    return dict(
        scope=scope, n_logs=int(len(sub)),
        drilled_depth=stat("Depth_of_Drill__m_"),
        static_water_level=stat("Static_Water_Level__mbgl_"),
        transmissivity=stat("Transmissivity__m2_day_"),
        storage_coefficient=stat("Storage_coefficient"),
        electrical_conductivity=stat("Electrical_Conductivity__Micro_Seimens__cm_"),
        discharge_lps=stat("Discharge__lps_"),
        bedrock=stat("Depth_of_Bedrock__mbgl_"),
        saline_fraction_pct=(100.0 * float((ec > SALINE_EC_THRESHOLD).mean())
                             if len(ec) else np.nan),
        ec_n=int(len(ec)),
        ec_vs_depth_r=ec_depth_r,
        median_drilled_depth_m=float(dd.median()) if len(dd) else np.nan,
        median_swl_m=float(swl.median()) if len(swl) else np.nan,
    )


def salinity_risk(district: str | None = None, target_depth_mbgl: float = 120.0) -> dict:
    """
    Salinity risk for drilling to ``target_depth_mbgl``.

    Deliberately honest: the measured EC record shows **no** depth dependence
    (r is returned), so we report the population risk from real logs and label
    the deep-water salinity hazard as a literature risk rather than dressing up
    an assumption as a measurement.
    """
    st = litholog_stats(district)
    ec = st["electrical_conductivity"]
    r = st["ec_vs_depth_r"]
    base = st["saline_fraction_pct"]
    measured = bool(np.isfinite(base))
    # A modest, clearly-labelled literature uplift for going below the fresh
    # lens, applied ONLY as a scenario knob — never presented as measured.
    depth_uplift = 0.0
    if target_depth_mbgl > 90:
        depth_uplift = min(15.0, (target_depth_mbgl - 90.0) / 10.0 * 2.0)
    return dict(district=district, target_depth_mbgl=target_depth_mbgl,
                measured_saline_pct=base if measured else np.nan,
                ec_median=ec["median"], ec_p75=ec["p75"], ec_n=st["ec_n"],
                threshold=SALINE_EC_THRESHOLD,
                ec_vs_depth_r=r,
                depth_dependence_supported=bool(np.isfinite(r) and abs(r) > 0.2),
                literature_uplift_pct=round(depth_uplift, 1),
                scenario_saline_pct=(base + depth_uplift) if measured else np.nan,
                scope=st["scope"], n_logs=st["n_logs"])


# ---------------------------------------------------------------------------
def strata_profile(district: str, current_level_mbgl: float) -> dict:
    """
    Where the district's water table sits in the three-tier system, what it
    would take to follow it down, and what the real logs say about the rock.
    """
    from .config import get_prior
    cur = float(current_level_mbgl)
    here = stratum_for_depth(cur)
    nxt = next((s for s in STRATA if s.top_mbgl > here.top_mbgl), None)
    logs = litholog_stats(district)
    capex_now = deepening_capex(max(cur + 15.0, here.top_mbgl + 10.0), static_level_mbgl=cur)
    step = (transition_cost(here.key, nxt.key) if nxt else None)
    decline = float(getattr(get_prior(district), "long_term_decline_m_per_yr", 0.0) or 0.0)
    gap = (nxt.top_mbgl - cur) if nxt else 0.0
    yrs = (gap / decline) if (nxt and decline > 1e-6) else None
    return dict(district=district, current_level_mbgl=cur,
                current_stratum=here.key, current_stratum_name=here.name,
                current_pump_class=here.pump_class,
                metres_to_next_stratum=gap,
                next_stratum=(nxt.key if nxt else None),
                decline_m_per_yr=decline,
                years_to_next_stratum=yrs,
                current_capex=capex_now, transition=step,
                litholog=logs,
                saline=salinity_risk(district, max(cur + 20.0, 100.0)))


def self_check(districts=("Sangrur", "Ludhiana", "Moga")) -> None:
    print("\n[Module C] Aquifer strata & deepening economics")
    print("-" * 74)
    st = litholog_stats("Sangrur")
    print(f"  real lithologs           : {st['n_logs']} ({st['scope']})")
    print(f"  drilled depth (median)   : {st['median_drilled_depth_m']:.0f} m")
    print(f"  static water level       : {st['median_swl_m']:.1f} m")
    print(f"  transmissivity (median)  : {st['transmissivity']['median']:,.0f} m²/d")
    print(f"  storage coefficient      : {st['storage_coefficient']['median']:.5f}")
    print(f"  EC median / saline share : {st['electrical_conductivity']['median']:.0f} µS/cm · "
          f"{st['saline_fraction_pct']:.0f}% > {SALINE_EC_THRESHOLD:.0f}")
    print(f"  EC vs depth correlation  : r = {st['ec_vs_depth_r']:+.3f}  "
          f"-> depth dependence {'SUPPORTED' if abs(st['ec_vs_depth_r'])>0.2 else 'NOT supported'}")
    print()
    print(f"  {'target':>8s}  {'stratum':>6s}  {'pump':>28s}  {'HP':>5s}  "
          f"{'total ₹':>10s}  {'₹/ft':>7s}")
    for d in (25, 40, 60, 90, 120, 160):
        r = deepening_capex(float(d), static_level_mbgl=min(d - 2, d * 0.9))
        print(f"  {d:7.0f}m  {r['stratum']:>6s}  {r['pump_class']:>28s}  "
              f"{r['required_hp']:5.1f}  {r['total_inr']:>10,d}  {r['inr_per_ft_total']:7.0f}")
    print()
    for a, b in (("L1", "L2"), ("L2", "L3")):
        t = transition_cost(a, b)
        print(f"  {a} -> {b}  deepening + pump swap : ₹{t['delta_inr']:,} "
              f"(₹{t['delta_lakh']} L) · {t['from_pump']} {t['from_hp']:.1f} HP -> "
              f"{t['to_pump']} {t['to_hp']:.1f} HP")
    print()
    print(f"  {'district':<10s} {'level':>7s}  {'stratum':>7s}  {'to next':>8s}  "
          f"{'years':>6s}  {'step ₹L':>8s}")
    for dname in districts:
        from .config import get_prior
        lvl = float(get_prior(dname).base_depth_2021_mbgl)
        p = strata_profile(dname, lvl)
        t = p["transition"]
        yrs = p["years_to_next_stratum"]
        print(f"  {dname:<10s} {lvl:6.1f}m  {p['current_stratum']:>7s}  "
              f"{p['metres_to_next_stratum']:7.1f}m  "
              f"{(f'{yrs:.1f}' if yrs else '-'):>6s}  "
              f"{(t['delta_lakh'] if t else 0):>8.2f}")
    print("-" * 74)


if __name__ == "__main__":
    self_check()


# ---------------------------------------------------------------------------
def cross_section(district: str, n_points: int = 90,
                  level_mbgl: float | None = None,
                  decline_m_per_yr: float | None = None) -> "object":
    """
    Vertical cross-section along a west-east transect through the district:
    the three strata as bands, the water table today, where it will be in 10
    and 20 years at the district's own decline rate, and the rupee cost of
    following it down.

    The transect is cut through the real district polygon, so the section has
    the district's true width; strata tops are the operational boundaries and
    the groundwater surface is the district level, not a synthetic wave.
    """
    import plotly.graph_objects as go
    from shapely.geometry import Point

    from .config import get_prior
    from .spatial_loader import district_geometry

    prior = get_prior(district)
    if level_mbgl is None:
        level_mbgl = float(getattr(prior, "base_depth_2021_mbgl", 30.0))
    if decline_m_per_yr is None:
        decline_m_per_yr = float(getattr(prior, "long_term_decline_m_per_yr", 0.8) or 0.8)

    geom = district_geometry(district)
    if geom is not None:
        minx, miny, maxx, maxy = geom.bounds
        mid_lat = (miny + maxy) / 2.0
        # sample along latitude through the widest part we can find
        best, best_len = mid_lat, 0.0
        for f in np.linspace(miny + 0.02, maxy - 0.02, 21):
            xs = np.linspace(minx, maxx, 400)
            inside = np.array([geom.contains(Point(x, f)) for x in xs])
            if inside.sum() > best_len:
                best_len, best = int(inside.sum()), float(f)
        xs_all = np.linspace(minx, maxx, 400)
        keep = np.array([geom.contains(Point(x, best)) for x in xs_all])
        if keep.sum() < 3:
            keep = np.ones_like(xs_all, dtype=bool)
        lon = xs_all[keep]
        width_km = float((lon.max() - lon.min()) * 111.0 * np.cos(np.radians(best)))
    else:
        lon = np.linspace(75.55, 76.20, n_points)
        width_km = float((lon.max() - lon.min()) * 111.0 * np.cos(np.radians(30.2)))

    x = np.linspace(0.0, max(width_km, 1.0), max(len(lon), 20))
    max_depth = 180.0
    fig = go.Figure()

    # ---- strata bands ------------------------------------------------------
    for s in STRATA:
        bot = min(s.bottom_mbgl, max_depth)
        if bot <= s.top_mbgl:
            continue
        cap = deepening_capex(min(s.top_mbgl + COMPLETION_INTO_STRATUM_M, bot),
                              static_level_mbgl=min(s.top_mbgl + COMPLETION_INTO_STRATUM_M, bot) - 2)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([np.full_like(x, s.top_mbgl), np.full_like(x, bot)[::-1]]),
            fill="toself", fillcolor=s.colour, opacity=0.30,
            line=dict(width=0), mode="lines", hoverinfo="skip",
            name=f"{s.key} — {s.name}"))
        fig.add_annotation(x=x[len(x) // 2], y=(s.top_mbgl + bot) / 2,
                           text=(f"<b>{s.key} · {s.name}</b><br>"
                                 f"{s.top_mbgl:.0f}–{bot:.0f} m · {s.pump_class}<br>"
                                 f"{s.hp_range[0]:.0f}–{s.hp_range[1]:.0f} HP · "
                                 f"bore ≈ ₹{cap['total_inr']/1e5:.2f} L"),
                           showarrow=False, font=dict(size=10, color="#1F2A37"),
                           bgcolor="rgba(255,255,255,0.72)", borderpad=4)

    # ---- water table: now, +10 yr, +20 yr ----------------------------------
    lv = float(level_mbgl)
    for yrs, dash, col, label in ((0, "solid", "#0B5FA5", "today"),
                                  (10, "dash", "#E4A11B", "+10 years"),
                                  (20, "dot", "#B4553F", "+20 years")):
        y = lv + decline_m_per_yr * yrs
        if y > max_depth:
            continue
        fig.add_trace(go.Scatter(
            x=x, y=np.full_like(x, y), mode="lines",
            line=dict(color=col, width=3 if yrs == 0 else 2, dash=dash),
            name=f"Water table — {label} ({y:.1f} m)",
            hovertemplate=f"{label}: %{{y:.1f}} m bgl<extra></extra>"))

    # ---- the crossing point ------------------------------------------------
    gap = L2.top_mbgl - lv
    yrs_to_l2 = gap / decline_m_per_yr if decline_m_per_yr > 1e-6 else None
    if yrs_to_l2 and 0 < yrs_to_l2 < 60:
        fig.add_annotation(x=x[int(len(x) * 0.06)], y=L2.top_mbgl,
                           ax=0, ay=-28, text=(f"crosses into <b>L2</b> in "
                                               f"~{yrs_to_l2:.0f} years"),
                           showarrow=True, arrowhead=2, arrowcolor="#B4553F",
                           font=dict(size=11, color="#B4553F"))

    fig.update_layout(
        title=(f"{district} — hydrostratigraphic cross-section "
               f"({width_km:.0f} km W–E transect, {lv:.1f} m bgl, "
               f"−{decline_m_per_yr:.2f} m/yr)"),
        xaxis_title="Distance across district (km)",
        yaxis_title="Depth below ground level (m)",
        yaxis=dict(autorange="reversed", range=[max_depth, 0], gridcolor="rgba(0,0,0,.07)"),
        height=470, margin=dict(l=64, r=18, t=58, b=44),
        legend=dict(orientation="h", y=-0.16, x=0),
        template="plotly_white", hovermode="x unified")
    return fig
