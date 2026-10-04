"""
AquaCast-Punjab :: Digital-Twin "What-If" Simulator
===================================================

The LSTM answers *"what will the water table do?"*
The digital twin answers *"what would it do if we changed the policy?"*

It re-runs the **same** lumped aquifer water-balance engine that generated the
training data, but with the user's levers applied, and projects the trajectory
forward from today's observed state:

* monsoon anomaly            -> scales rainfall-recharge over Jun-Sep
* micro-irrigation adoption  -> cuts crop groundwater draft (efficiency gain)
* paddy transplant shift     -> moves the Kharif demand window (the single
                                biggest Punjab policy lever: a 2-week delay in
                                transplanting saves ~1 irrigation and lets the
                                monsoon arrive before peak pumping)
* canal availability         -> trades groundwater draft for surface supply

Everything runs in ~30 ms on the full 1 828-day record, so sliders stay live.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (DATA_END, PADDY_END_DOY, PADDY_START_DOY, PROCESSED_DIR,
                     SEED, Scenario, get_prior)
from .dataset_generator import (compute_demand, fit_climate_model,
                                generate_rainfall, generate_temperature,
                                water_balance)
from .spatial_loader import district_area_km2


@lru_cache(maxsize=1)
def _climate():
    return fit_climate_model()


@lru_cache(maxsize=8)
def _meta(district: str) -> dict:
    p = PROCESSED_DIR / f"processed_{district.lower()}_meta.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"calibration": {"mean_lag_days": 150.0, "recharge_scale": 1.4}}


def _future_weather(index: pd.DatetimeIndex, cm, rng) -> pd.DataFrame:
    rain = generate_rainfall(index, cm, rng)
    tmax, tmin = generate_temperature(index, cm, rng)
    return pd.DataFrame({"date": index, "doy": index.dayofyear, "month": index.month,
                         "year": index.year, "rainfall_mm": rain,
                         "temp_max_c": tmax, "temp_min_c": tmin,
                         "rainfall_src": "generated", "temp_src": "generated"})


def project(df: pd.DataFrame, district: str, scenario: Scenario,
            days: int = 120, seed: int = SEED,
            paddy_shift_days: int = 0) -> pd.DataFrame:
    """
    Project the water table ``days`` ahead under ``scenario``.

    Returns a frame indexed by date with ``projected_level_mbgl`` (anchored to
    the observed level on the last historical day) and the driver columns.
    """
    prior = get_prior(district)
    area = district_area_km2(district)
    meta = _meta(district)
    lag = float(meta.get("calibration", {}).get("mean_lag_days", 150.0))
    k = float(meta.get("calibration", {}).get("recharge_scale", 1.4))

    hist = df.copy()
    hist["date"] = pd.to_datetime(hist.date)
    last = hist.date.max()

    cm = _climate()
    rng = np.random.default_rng(seed)
    fut_idx = pd.date_range(last + pd.Timedelta(days=1), periods=days, freq="D")
    fut = _future_weather(fut_idx, cm, rng)
    fut["district"] = district
    fut["is_rabi"] = 0

    full = pd.concat([hist.drop(columns=[c for c in ("gw_level_mbgl", "cumulative_decline_m",
                                                     "rolling_7d_rainfall", "rolling_30d_temp",
                                                     "rolling_14d_demand", "rain_anom_30d",
                                                     "stress_index", "net_flux_mcm",
                                                     "total_recharge_mcm", "rain_recharge_mcm",
                                                     "gw_draft_mcm", "canal_supply_mcm",
                                                     "total_crop_draft_mcm", "paddy_demand_mcm",
                                                     "crop_water_demand_mcm", "etc_wheat_mm",
                                                     "kc_paddy", "kc_wheat", "et0_mm")
                                         if c in hist.columns]), fut], ignore_index=True)
    full = full.sort_values("date").reset_index(drop=True)

    full = compute_demand(full, prior, scenario, paddy_shift_days=int(paddy_shift_days))
    full, _ = water_balance(full, prior, scenario, area, lag, k)

    # --- anchor to the observed present ---------------------------------------------------
    split = len(hist)
    base_decline = float(full.cumulative_decline_m.iloc[split - 1])
    current_level = float(hist.gw_level_mbgl.iloc[-1])
    out = full.iloc[split - 1:].copy()
    out["projected_level_mbgl"] = current_level + (out.cumulative_decline_m - base_decline)
    out["is_future"] = np.arange(len(out)) > 0
    out = out[["date", "projected_level_mbgl", "rainfall_mm", "temp_max_c",
               "crop_water_demand_mcm", "paddy_demand_mcm", "gw_draft_mcm",
               "total_recharge_mcm", "net_flux_mcm", "is_future"]]
    return out.reset_index(drop=True)


def compare_scenarios(df: pd.DataFrame, district: str, scenarios: dict[str, Scenario],
                      days: int = 120, paddy_shift_days: int | None = None) -> pd.DataFrame:
    """
    Run several scenarios and return a wide frame of projected trajectories.

    ``paddy_shift_days`` is an *override* applied to every scenario.  Leave it as
    ``None`` (the default) and each scenario uses its own
    ``paddy_transplant_shift_days`` — which is what makes a "drip + delayed
    transplant" scenario different from a plain "drip" one.  Passing a single
    value silently applied one shift to the baseline *and* the counterfactual,
    which made the two indistinguishable.
    """
    out = None
    for name, sc in scenarios.items():
        shift = (int(paddy_shift_days) if paddy_shift_days is not None
                 else int(getattr(sc, "paddy_transplant_shift_days", 0) or 0))
        p = project(df, district, sc, days=days, paddy_shift_days=shift)
        s = p.set_index("date").projected_level_mbgl.rename(name)
        out = s.to_frame() if out is None else out.join(s, how="outer")
    return out.reset_index()


def scenario_deltas(projections: pd.DataFrame, baseline_col: str) -> pd.DataFrame:
    """End-of-horizon difference (metres) of each scenario vs the baseline."""
    end = projections.iloc[-1]
    rows = []
    for c in projections.columns:
        if c == "date" or c == baseline_col:
            continue
        rows.append(dict(scenario=c,
                         end_level_mbgl=round(float(end[c]), 2),
                         delta_vs_baseline_m=round(float(end[c] - end[baseline_col]), 2),
                         saved_m=round(float(end[baseline_col] - end[c]), 2)))
    return pd.DataFrame(rows).sort_values("saved_m", ascending=False).reset_index(drop=True)
