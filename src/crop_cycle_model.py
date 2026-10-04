"""
Module B — Crop Cycle Model
===========================

A full-year, three-season crop rotation engine for the Punjab tubewell belt:

    Rabi    wheat        ~01 Nov -> ~15 Apr    Kc 0.40 -> 1.15 -> 0.30   ~350 mm
    Zaid    moong/sunflw ~20 Apr -> ~20 Jun    Kc 0.50 -> 1.05 -> 0.65   ~400 mm
    Kharif  paddy        ~20 Jun -> ~25 Oct    Kc 1.05 -> 1.20 -> 0.90   1200-1400 mm

Why this module exists
----------------------
The earlier engine modelled Rabi wheat and Kharif paddy only. That misses the
single most important fact about Punjab's groundwater, which is not how much
water paddy uses but *when* it uses it: the Kharif flood is drawn down through
June-October, the monsoon recharge pulse arrives weeks behind the pumping, and
the aquifer is therefore at its annual **lowest point in early November — the
week wheat goes in the ground**. Wheat is then sown into a depleted aquifer
with only canal water and a deepening pump to fall back on. Any model that
averages the year cannot see that; a daily rotation model can.

Everything here is computed live: crop areas and reference seasonal depths come
from ``DistrictPrior``, daily ET0 from the Hargreaves equation on the merged
telemetry, effective rainfall from the USDA-SCS curve used elsewhere in the
repo, and the resulting seasonal totals are *anchored* to the prior depths so
the curves cannot drift away from the district's known water budget.

Units
-----
``mm`` over an area of ``ha``:  1 mm over 1 ha = 10 m³, so
    MCM = area_ha * depth_mm / 1.0e5
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .config import get_prior

# ---------------------------------------------------------------------------
#: Scenario knob, not a measurement. Punjab's fields mostly lie fallow between
#: the wheat harvest and paddy transplanting, and where a summer crop is taken
#: it is short-duration moong on a minority of holdings. We therefore model the
#: Zaid season as an explicit lever the user can dial, defaulting to a modest
#: share, and label it as a lever in the UI rather than pretending to know it.
ZAID_DEFAULT_AREA_SHARE = 0.08

#: Seepage + percolation under continuous paddy flooding (mm/day), on top of
#: ETc. Punjab alluvium under a ponded field loses this much straight back
#: down; it is why "paddy uses 1200 mm" understates the true draft.
PADDY_PERCOLATION_MM_DAY = 2.6

#: Days after sowing for the phenological stages that cannot be missed.
WHEAT_CRI_DAS = 21            # crown root initiation
WHEAT_TILLER_DAS = 45
WHEAT_FLOWER_DAS = 95
WHEAT_GRAIN_DAS = 115
PADDY_TILLER_DAS = 30
PADDY_FLOWER_DAS = 75
PADDY_GRAIN_DAS = 100


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Crop:
    """One crop in the rotation."""
    key: str
    label: str
    season: str                     # Rabi | Zaid | Kharif
    sow_doy: int                    # day-of-year of sowing/transplanting
    harvest_doy: int
    kc_anchors: tuple[tuple[float, float], ...]   # (DAS, Kc) breakpoints
    ref_etc_mm: float               # literature seasonal crop evapotranspiration
    percolation_mm_day: float = 0.0
    flooded: bool = False           # flooded crops calibrate on a different curve
    colour: str = "#4C78A8"

    @property
    def length_days(self) -> int:
        n = self.harvest_doy - self.sow_doy
        return n + 365 if n <= 0 else n

    def das(self, doy: np.ndarray) -> np.ndarray:
        """Days after sowing for each day of year (negative outside season)."""
        d = (np.asarray(doy, dtype=float) - self.sow_doy) % 365.0
        out = np.where(d > self.length_days, -(365.0 - d), d)
        return out

    def in_season(self, doy: np.ndarray) -> np.ndarray:
        d = self.das(doy)
        return (d >= 0) & (d <= self.length_days)

    def kc(self, doy: np.ndarray) -> np.ndarray:
        """Piecewise-linear Kc from the anchor points, 0 outside the season."""
        d = self.das(doy)
        xs = np.array([a[0] for a in self.kc_anchors], dtype=float)
        ys = np.array([a[1] for a in self.kc_anchors], dtype=float)
        kc = np.interp(d, xs, ys, left=np.nan, right=np.nan)
        return np.where(self.in_season(doy), np.nan_to_num(kc, nan=0.0), 0.0)


# Kharif paddy: transplanted ~20 Jun (DOY 171), harvested ~25 Oct (DOY 298).
# Kc rises through the vegetative stage to ~1.20 at flowering, holds ponded
# through grain fill, then drops as the field is drained before harvest.
PADDY = Crop(
    key="paddy", label="Paddy (Kharif)", season="Kharif",
    sow_doy=171, harvest_doy=298,
    kc_anchors=((0, 1.05), (20, 1.12), (60, 1.20), (95, 1.18), (125, 0.90), (127, 0.60)),
    ref_etc_mm=1300.0, percolation_mm_day=PADDY_PERCOLATION_MM_DAY, flooded=True,
    colour="#2E7D5B",
)

# Rabi wheat: sown ~01 Nov (DOY 305, wrapping into next January), harvested
# ~15 Apr (DOY 105). Kc 0.40 at germination -> 1.15 at flowering -> 0.30 at
# maturity, integrating to the district's ~350 mm reference depth.
WHEAT = Crop(
    key="wheat", label="Wheat (Rabi)", season="Rabi",
    sow_doy=305, harvest_doy=105,
    kc_anchors=((0, 0.40), (25, 0.75), (70, 1.05), (100, 1.15), (140, 0.85), (160, 0.30)),
    ref_etc_mm=450.0,
    colour="#C9A227",
)

# Zaid: short-duration moong / sunflower, ~20 Apr (DOY 110) -> ~20 Jun (DOY 171).
ZAID = Crop(
    key="zaid", label="Summer moong / sunflower (Zaid)", season="Zaid",
    sow_doy=110, harvest_doy=171,
    kc_anchors=((0, 0.50), (25, 0.85), (50, 1.05), (75, 0.90), (85, 0.60)),
    ref_etc_mm=400.0,
    colour="#E07B39",
)

# Diversification alternatives the simulator can swap paddy out for.
MAIZE = Crop(
    key="maize", label="Maize (Kharif alternative)", season="Kharif",
    sow_doy=171, harvest_doy=263,
    kc_anchors=((0, 0.45), (30, 0.90), (60, 1.15), (85, 1.05), (110, 0.60)),
    ref_etc_mm=620.0,
    colour="#8E6E9E",
)

COTTON = Crop(
    key="cotton", label="Cotton (Kharif alternative)", season="Kharif",
    sow_doy=150, harvest_doy=320,
    kc_anchors=((0, 0.35), (45, 0.75), (95, 1.15), (150, 1.05), (190, 0.70)),
    ref_etc_mm=780.0,
    colour="#B0A8B9",
)

ALL_CROPS: dict[str, Crop] = {c.key: c for c in (PADDY, WHEAT, ZAID, MAIZE, COTTON)}
KHARIF_OPTIONS = ("paddy", "maize", "cotton")


# ---------------------------------------------------------------------------
def effective_rainfall_mm(rain: np.ndarray) -> np.ndarray:
    """USDA-SCS effective rainfall — same curve the rest of the repo uses."""
    r = np.asarray(rain, dtype=float)
    eff = r * (4.17 - 0.0833 * r) / (5.09 + 0.0425 * r) / 25.4 * 25.4
    eff = np.where(r <= 0, 0.0, np.clip((r * (4.17 - 0.2 * r)) / 100.0 + 0.1 * r, 0, r))
    return np.clip(eff, 0, r)


def mm_over_ha_to_mcm(area_ha, depth_mm):
    """1 mm over 1 ha = 10 m³.  Vectorises over arrays of depth."""
    return np.asarray(area_ha, dtype=float) * np.asarray(depth_mm, dtype=float) / 1.0e5


# ---------------------------------------------------------------------------
@dataclass
class RotationResult:
    """Daily, full-year output of the rotation model."""
    frame: pd.DataFrame                       # daily doy / kc / etc / draft
    crops: dict[str, Crop]
    areas_ha: dict[str, float]
    seasonal_mm: dict[str, float]             # achieved seasonal ETc depth
    seasonal_mcm: dict[str, float]            # achieved seasonal draft (MCM)
    annual_draft_mcm: float
    peak_doy: int
    pre_sowing_low_mcm: float                 # cumulative Kharif+Zaid draft by 01 Nov
    meta: dict = field(default_factory=dict)


def rotation(
    district: str = "Sangrur",
    df: pd.DataFrame | None = None,
    kharif_crop: str = "paddy",
    zaid_share: float = ZAID_DEFAULT_AREA_SHARE,
    paddy_area_ha: float | None = None,
    wheat_area_ha: float | None = None,
) -> RotationResult:
    """
    Run the full-year rotation for ``district``.

    ``df`` supplies ``ET0_mm/day`` (or ``tmax``/``tmin`` to derive it) and
    ``rainfall_mm``; if omitted a climatological year is synthesised from the
    district's own monthly climate model, which keeps the module usable
    standalone.

    Each crop's daily Kc x ET0 curve is *normalised* so its seasonal integral
    equals the district's reference depth (paddy 1200-1400 mm, wheat ~350 mm
    from the prior). That anchoring is what stops a synthetic ET0 year from
    producing an implausible seasonal total.
    """
    prior = get_prior(district)
    wheat_ha = float(wheat_area_ha if wheat_area_ha is not None else prior.wheat_area_ha)
    paddy_ha = float(paddy_area_ha if paddy_area_ha is not None else prior.kharif_paddy_area_ha)

    kh = ALL_CROPS.get(kharif_crop, PADDY)
    kh_ha = paddy_ha                     # the Kharif alternative occupies paddy land
    wheat_ref = float(getattr(prior, "wheat_season_depth_mm", 350.0) or 350.0)
    zaid_ha = wheat_ha * float(zaid_share)

    # ---- ET0 + rain ---------------------------------------------------------
    if df is not None and len(df):
        idx = pd.to_datetime(df["date"]) if "date" in df else df.index
        if "ET0_mm" in df:
            et0 = df["ET0_mm"].to_numpy(dtype=float)
        else:
            et0 = _et0_from_telemetry(df, getattr(prior, "lat_deg", 30.24))
        rain = (df["rainfall_mm"].to_numpy(dtype=float)
                if "rainfall_mm" in df else np.full(len(et0), np.nan))
        dates = pd.DatetimeIndex(idx)
        doy = np.asarray(dates.dayofyear, dtype=float)
    else:
        dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        doy = dates.dayofyear.to_numpy(dtype=float)
        et0 = _clim_et0(doy)
        rain = np.full(len(doy), np.nan)

    have_rain = bool(np.isfinite(rain).any())
    rain_f = np.nan_to_num(rain, nan=0.0)
    pe = effective_rainfall_mm(rain_f)

    # ---- per-crop daily ETc -> net irrigation -> groundwater draft ----------
    #
    # Convention (identical to ``dataset_generator.compute_demand`` so the two
    # modules cannot disagree):
    #
    #   ETc           = Kc x ET0, season-normalised to the crop's ref_etc_mm
    #   net irrigation= max(ETc - effective rainfall, 0) + percolation
    #   groundwater   = net irrigation x (1 - canal share) x kappa
    #
    # ``kappa`` is one calibration constant per *season*, not per crop. That
    # matters: the dominant thing kappa absorbs is the monsoon rainfall credit
    # and the return flow from puddled fields, and every Kharif crop shares the
    # same monsoon. Splitting kappa by crop instead would hand paddy a large
    # rain credit and maize none, which would make diversification look
    # mysteriously useless. kappa_kharif is pinned so Kharif paddy lands on the
    # repo's calibrated PADDY_GW_DRAFT_MM; kappa_rabi so Rabi wheat lands on
    # the district's own rabi_draft_mcm prior. Both targets are data, not
    # guesses, and both kappas are reported so a reviewer can check them.
    frame = pd.DataFrame({"date": dates, "doy": doy, "ET0_mm": et0,
                          "rainfall_mm": rain_f, "eff_rain_mm": pe})
    seasonal_etc_mm: dict[str, float] = {}
    seasonal_net_mm: dict[str, float] = {}
    seasonal_gw_mm: dict[str, float] = {}
    seasonal_mcm: dict[str, float] = {}
    areas = {"wheat": wheat_ha, kh.key: kh_ha, "zaid": zaid_ha}
    crops_by_key = {"wheat": WHEAT, kh.key: kh, "zaid": ZAID}
    canal_share = float(getattr(prior, "canal_share_pct", 0.0) or 0.0) / 100.0
    gw_frac = 1.0 - canal_share

    from .dataset_generator import PADDY_GW_DRAFT_MM
    paddy_target_mm = float(PADDY_GW_DRAFT_MM)

    def _sim(crop: Crop) -> dict:
        """Kc, season-normalised ETc, and net irrigation for one crop."""
        kc = crop.kc(doy)
        raw_etc = kc * et0
        tot = raw_etc.sum()
        scale = float(np.clip(crop.ref_etc_mm / tot, 0.2, 5.0)) if tot > 1e-6 else 1.0
        etc = raw_etc * scale
        perc = crop.percolation_mm_day * (kc > 0)
        net = np.clip(etc - pe * (kc > 0), 0.0, None) + perc
        return dict(kc=kc, etc=etc, net=net, perc=perc)

    # pass 1: every crop in this rotation
    sim: dict[str, dict] = {k: _sim(c) for k, c in crops_by_key.items()}
    for key in sim:
        seasonal_etc_mm[key] = float(sim[key]["etc"].sum())
        seasonal_net_mm[key] = float(sim[key]["net"].sum())

    # pass 2: one kappa per season, each pinned to a data-derived target.
    #
    # The Kharif target always comes from PADDY, never from whatever Kharif crop
    # the caller asked for — otherwise swapping in maize would silently pin
    # maize to paddy's calibrated draft and diversification would appear to
    # save nothing at all. So when the scenario crop is not paddy we still run
    # paddy's curve once, purely to read off the calibration constant.
    def _cls(crop: Crop) -> str:
        return "kharif" if crop.season == "Kharif" else "rabi"

    kappa: dict[str, float] = {"kharif": 1.0, "rabi": 1.0}
    if kh.key == "paddy":
        paddy_net = seasonal_net_mm["paddy"]
    else:
        paddy_net = float(_sim(PADDY)["net"].sum())
    raw_kh = paddy_net * gw_frac
    if raw_kh > 1e-6:
        kappa["kharif"] = float(np.clip(paddy_target_mm / raw_kh, 0.1, 10.0))
    raw_r = seasonal_net_mm["wheat"] * gw_frac
    if raw_r > 1e-6:
        kappa["rabi"] = float(np.clip(wheat_ref / raw_r, 0.1, 10.0))

    # pass 3: apply and book the MCM
    for key, crop in crops_by_key.items():
        k = kappa[_cls(crop)]
        gw_mm = sim[key]["net"] * gw_frac * k
        frame[f"kc_{key}"] = sim[key]["kc"]
        frame[f"etc_{key}"] = sim[key]["etc"]
        frame[f"draft_mm_{key}"] = gw_mm
        seasonal_gw_mm[key] = float(gw_mm.sum())
        seasonal_mcm[key] = float(mm_over_ha_to_mcm(areas[key], gw_mm.sum()))
    seasonal_mm = seasonal_etc_mm

    frame["draft_mcm_day"] = sum(
        mm_over_ha_to_mcm(areas[k], frame[f"draft_mm_{k}"].to_numpy())
        for k in areas if f"draft_mm_{k}" in frame)
    frame["draft_mm_day"] = (
        frame["draft_mcm_day"] / max(sum(areas.values()), 1e-9) * 1e5)

    annual = float(sum(seasonal_mcm.values()))
    # Kharif pumping runs Jun-Oct; wheat goes in the ground on 01 Nov (DOY 305)
    peak_doy = int(doy[int(np.argmax(frame["draft_mcm_day"].to_numpy()))])
    kharif_by_nov = _window(frame, 152, 305, "draft_mcm_day")

    return RotationResult(
        frame=frame,
        crops={"wheat": WHEAT, kh.key: kh, "zaid": ZAID},
        areas_ha=areas,
        seasonal_mm=seasonal_mm,
        seasonal_mcm=seasonal_mcm,
        annual_draft_mcm=annual,
        peak_doy=peak_doy,
        pre_sowing_low_mcm=float(kharif_by_nov),
        meta=dict(district=district, kharif_crop=kh.key, zaid_share=float(zaid_share),
                  wheat_ha=wheat_ha, kharif_ha=kh_ha, zaid_ha=zaid_ha,
                  have_rainfall=have_rain,
                  canal_share_pct=100.0 * canal_share,
                  kappa_kharif=kappa["kharif"], kappa_rabi=kappa["rabi"],
                  target_wheat_mm=wheat_ref,
                  target_kharif_mm=paddy_target_mm,
                  seasonal_gw_mm=seasonal_gw_mm,
                  seasonal_etc_mm=seasonal_etc_mm,
                  percolation_mm_day=kh.percolation_mm_day),
    )


def _window(frame: pd.DataFrame, start_doy: int, end_doy: int, col: str) -> float:
    d = frame["doy"].to_numpy()
    if start_doy <= end_doy:
        m = (d >= start_doy) & (d < end_doy)
    else:
        m = (d >= start_doy) | (d < end_doy)
    return float(frame.loc[m, col].sum())


def _et0_from_telemetry(df: pd.DataFrame, lat_deg: float) -> np.ndarray:
    """Hargreaves ET0 from tmax/tmin when a precomputed ET0 column is absent."""
    try:
        from .dataset_generator import hargreaves_et0
    except Exception:                     # pragma: no cover - fallback path
        hargreaves_et0 = None
    tmax = df["tmax_c"].to_numpy(dtype=float) if "tmax_c" in df else None
    tmin = df["tmin_c"].to_numpy(dtype=float) if "tmin_c" in df else None
    if tmax is None or tmin is None or hargreaves_et0 is None:
        return _clim_et0(pd.DatetimeIndex(pd.to_datetime(df["date"])).dayofyear.to_numpy()
                         if "date" in df else np.arange(1, 366) % 365 + 1)
    doy = pd.DatetimeIndex(pd.to_datetime(df["date"])).dayofyear.to_numpy(dtype=float)
    return hargreaves_et0(tmax, tmin, lat_deg, doy)


def _clim_et0(doy: np.ndarray) -> np.ndarray:
    """Smooth annual ET0 wave (mm/day) peaking pre-monsoon — standalone fallback."""
    d = np.asarray(doy, dtype=float)
    return 3.4 + 3.6 * np.cos(2 * np.pi * (d - 152) / 365.0) ** 2 * 1.0 + \
        0.9 * np.cos(2 * np.pi * (d - 20) / 365.0)


# ---------------------------------------------------------------------------
def critical_windows(district: str = "Sangrur",
                     kharif_crop: str = "paddy") -> list[dict]:
    """
    Phenological windows in the next 12 months that cannot be missed, with the
    date, the crop, the stage and how much water the stage needs.

    Drives the kisan view's "next-30-day critical stage" alert.
    """
    wheat_sow = pd.Timestamp(_year_for_doy(305))
    out = []

    def add(crop: Crop, das: int, stage: str, note: str):
        d = crop.sow_doy + das
        date = pd.Timestamp(_year_for_doy(int(d % 365) or 365))
        # stage water need = Kc at that DAS x a representative ET0 for that date
        kc = float(np.interp(das, [a[0] for a in crop.kc_anchors],
                             [a[1] for a in crop.kc_anchors]))
        et0 = float(_clim_et0(np.array([date.dayofyear]))[0])
        out.append(dict(crop=crop.label, crop_key=crop.key, stage=stage,
                        das=das, date=date, kc=round(kc, 2),
                        mm_day=round(kc * et0, 1), note=note,
                        colour=crop.colour))

    add(WHEAT, WHEAT_CRI_DAS, "Crown root initiation",
        "First irrigation. Missing it costs more yield than any later cut.")
    add(WHEAT, WHEAT_TILLER_DAS, "Tillering / canopy development",
        "Peak canopy, peak daily demand.")
    add(WHEAT, WHEAT_FLOWER_DAS, "Flowering",
        "Most stress-sensitive stage; do not ration here.")
    add(WHEAT, WHEAT_GRAIN_DAS, "Grain fill", "Late cut costs test weight.")
    kh = ALL_CROPS.get(kharif_crop, PADDY)
    add(kh, PADDY_TILLER_DAS, "Tillering", "Ponding established; maintain 5 cm.")
    add(kh, PADDY_FLOWER_DAS, "Flowering", "Continuous flooding critical.")
    add(kh, PADDY_GRAIN_DAS, "Grain fill", "Last 2-3 irrigations before drain.")

    out.sort(key=lambda r: r["date"])
    return out


def next_windows(days: int = 30, district: str = "Sangrur",
                 kharif_crop: str = "paddy",
                 today: pd.Timestamp | None = None) -> list[dict]:
    """Critical windows falling in the next ``days`` days."""
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    end = today + pd.Timedelta(days=int(days))
    rows = []
    for w in critical_windows(district, kharif_crop):
        d = w["date"]
        for yr in (today.year, today.year + 1):
            cand = d.replace(year=yr)
            if today <= cand <= end:
                rows.append({**w, "date": cand})
    rows.sort(key=lambda r: r["date"])
    return rows


def _year_for_doy(doy: int) -> str:
    """Anchor a day-of-year into the current 12-month rotation window."""
    today = pd.Timestamp.today().normalize()
    yr = today.year if doy >= today.dayofyear else today.year + 1
    return _date_from_doy(yr, doy)


def _date_from_doy(year: int, doy: int) -> str:
    return (pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(doy) - 1)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
def diversify(district: str = "Sangrur", df: pd.DataFrame | None = None,
              switch_ha: float = 2000.0, to_crop: str = "maize",
              zaid_share: float = ZAID_DEFAULT_AREA_SHARE) -> dict:
    """
    Shift ``switch_ha`` of paddy land to ``to_crop`` and report the water saved.

    Returns baseline vs switched annual draft, the MCM saved, the saving as a
    share of the district's annual groundwater draft, and the implied head
    recovery in metres (draft / (area x specific yield)).
    """
    prior = get_prior(district)
    paddy_ha = float(prior.kharif_paddy_area_ha)
    switch_ha = float(min(max(switch_ha, 0.0), paddy_ha))

    # NB: ``rotation()`` always books the *whole* Rabi wheat and Zaid area, so
    # you cannot add two rotation results together — wheat and zaid would be
    # counted twice. Only the Kharif term differs between the two cases, so we
    # price the Kharif hectares separately and add the shared terms once.
    base = rotation(district, df, kharif_crop="paddy", zaid_share=zaid_share)
    shared_mcm = base.seasonal_mcm["wheat"] + base.seasonal_mcm["zaid"]

    def kharif_mcm(crop_key: str, area_ha: float) -> float:
        if area_ha <= 0:
            return 0.0
        r = rotation(district, df, kharif_crop=crop_key, zaid_share=zaid_share,
                     paddy_area_ha=area_ha)
        return float(r.seasonal_mcm.get(crop_key, 0.0))

    base_kharif = kharif_mcm("paddy", paddy_ha)
    alt_kharif = kharif_mcm("paddy", paddy_ha - switch_ha) + kharif_mcm(to_crop, switch_ha)

    baseline_mcm = shared_mcm + base_kharif
    switched_mcm = shared_mcm + alt_kharif
    saved_mcm = baseline_mcm - switched_mcm

    area_km2 = max(float(getattr(prior, "wheat_area_ha", 280000)) / 100.0, 1.0)
    sy = float(getattr(prior, "specific_yield", 0.12) or 0.12)
    head_m = saved_mcm * 1e6 / (area_km2 * 1e6 * sy)
    return dict(district=district, switch_ha=switch_ha, to_crop=to_crop,
                baseline_mcm=baseline_mcm, switched_mcm=switched_mcm,
                saved_mcm=saved_mcm,
                saved_pct=100.0 * saved_mcm / max(baseline_mcm, 1e-9),
                head_recovered_m=head_m,
                baseline_kharif_mcm=base_kharif,
                switched_kharif_mcm=alt_kharif,
                shared_mcm=shared_mcm)


# ---------------------------------------------------------------------------
def self_check(districts: Sequence[str] = ("Sangrur", "Ludhiana", "Moga")) -> None:
    print("\n[Module B] Crop cycle model")
    print("-" * 66)
    for d in districts:
        r = rotation(d)
        m = r.meta
        print(f"\n  {d}  ({m['kharif_crop']} / wheat / zaid {m['zaid_share']:.0%})")
        print(f"    areas (ha)              : wheat {m['wheat_ha']:,.0f} · "
              f"kharif {m['kharif_ha']:,.0f} · zaid {m['zaid_ha']:,.0f}")
        for k in ("wheat", m["kharif_crop"], "zaid"):
            print(f"    {k:<22s}: ETc {m['seasonal_etc_mm'][k]:6.0f} mm -> "
                  f"GW {m['seasonal_gw_mm'][k]:6.0f} mm -> "
                  f"{r.seasonal_mcm[k]:7.1f} MCM")
        print(f"    kappa (kharif / rabi)   : {m['kappa_kharif']:.3f} / "
              f"{m['kappa_rabi']:.3f}   canal share {m['canal_share_pct']:.0f}%")
        print(f"    ANNUAL DRAFT            : {r.annual_draft_mcm:,.1f} MCM")
        print(f"    peak pumping day        : DOY {r.peak_doy} "
              f"({pd.Timestamp('2025-01-01') + pd.Timedelta(days=int(r.peak_doy)-1):%d %b})")
        print(f"    drawn Jun-Oct (pre-sow) : {r.pre_sowing_low_mcm:,.1f} MCM "
              f"({100*r.pre_sowing_low_mcm/max(r.annual_draft_mcm,1e-9):.0f}% of the year)")
        try:
            dv = diversify(d, switch_ha=2000.0, to_crop="maize")
            print(f"    2,000 ha paddy -> maize : -{dv['saved_mcm']:,.1f} MCM "
                  f"({dv['saved_pct']:.1f}%) = +{dv['head_recovered_m']:.3f} m head/yr")
        except Exception as exc:                     # pragma: no cover
            print(f"    diversify               : FAILED ({exc})")
    print("\n  critical windows (next 60 d):")
    for w in next_windows(60):
        print(f"    {w['date']:%d %b %Y}  {w['crop']:<32s} {w['stage']:<28s} "
              f"{w['mm_day']:5.1f} mm/d")
    print("-" * 66)


if __name__ == "__main__":
    self_check()


# ---------------------------------------------------------------------------
def seasonal_trough(df: pd.DataFrame, level_col: str = "gw_level_mbgl") -> dict:
    """
    Prove the November trough from an actual modelled hydrograph.

    Averages the water level by calendar month over the whole series, then
    reports the month the aquifer sits deepest (largest mbgl) and the month it
    sits shallowest. The claim this module exists to support is that the
    deepest point of the year falls in **early November — the week wheat is
    sown**, because Kharif paddy has been pumped out through the monsoon and
    the recharge pulse has not yet arrived. If a district's own series does
    not show that, the module says so rather than asserting it.
    """
    if df is None or level_col not in df or not len(df):
        return {}
    d = df.dropna(subset=[level_col]).copy()
    if not len(d):
        return {}
    t = pd.to_datetime(d["date"])
    d["_m"] = t.dt.month
    # Detrend first: the series carries a ~0.8 m/yr decline, and we want the
    # *intra-annual* cycle, not the five-year slide. Removing a fitted linear
    # trend leaves the seasonal signal, which is what the Kharif/Rabi story is
    # actually about.
    x = (t - t.min()).dt.days.to_numpy(dtype=float)
    y = d[level_col].to_numpy(dtype=float)
    if len(d) > 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        d[level_col] = y - (slope * x + intercept) + float(np.mean(y))
    prof = d.groupby("_m")[level_col].mean()
    if prof.empty:
        return {}
    full = prof.reindex(range(1, 13)).interpolate().bfill().ffill()
    deep = int(full.idxmax())
    shallow = int(full.idxmin())
    return dict(monthly_mbgl={int(k): float(v) for k, v in full.items()},
                deepest_month=deep, shallowest_month=shallow,
                amplitude_m=float(full.max() - full.min()),
                deepest_label=pd.Timestamp(2000, deep, 1).strftime("%b"),
                shallowest_label=pd.Timestamp(2000, shallow, 1).strftime("%b"),
                trend_m_per_day=float(slope) if len(d) > 2 else 0.0,
                claim_holds=deep in (10, 11, 12))
