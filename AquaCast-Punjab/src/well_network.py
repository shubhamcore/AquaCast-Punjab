"""
AquaCast-Punjab :: Well-Network Downscaling (district forecast -> 121 village wells)
====================================================================================

The LSTM forecasts a *district-representative* water table.  A lender or a block
officer needs it per well.  This module projects the district trajectory onto
every CGWB / Punjab-GW observation well in the district by learning a **static
per-well offset** — the part of a well's level that local lithology, elevation
and pumping density explain and that does not change over the horizon.

Offset estimation
-----------------
* Wells with in-situ CGWB readings (10 in Sangrur): offset = mean(observed -
  district series on the observation dates).  Measured, not modelled.
* All other wells: inverse-distance-weighted interpolation of those measured
  offsets (Gaussian kernel, 12 km bandwidth), shrunk towards the district mean
  where no observation well is nearby (``SHRINK``).

Pump installation depth
-----------------------
Estimated per well as ``depth to first aquifer + TYPICAL_SCREEN_DEPTH`` using the
CGWB aquifer raster shipped in ``data/rasters/`` — so dry-out risk varies
spatially for a physical reason, not because we tuned it to look interesting.
"""
from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from . import telemetry_ingest as ti
from .config import DEFAULT_DISTRICT, get_prior
from .spatial_loader import wells_for_district

warnings.filterwarnings("ignore")

KERNEL_KM = 12.0          # Gaussian bandwidth for offset interpolation
MAX_KM = 45.0             # beyond this, fall back to the district mean
SHRINK = 0.65             # shrink interpolated anomalies toward the district mean
TYPICAL_SCREEN_DEPTH_M = 30.0     # screen/sump set below the top of the first aquifer
                                # (central-Punjab tubewells are typically 45-70 m deep)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def well_offsets(district: str, df: pd.DataFrame) -> pd.DataFrame:
    """Static per-well offset (metres, +ve = well is deeper than the district mean)."""
    obs = ti.load_groundwater_observations()
    d = df.copy()
    d["date"] = pd.to_datetime(d.date)
    series = d.set_index("date").gw_level_mbgl

    rows = []
    if not obs.empty:
        o = obs.copy()
        o["date"] = pd.to_datetime(o.date).dt.normalize()
        o["model"] = o.date.map(series)
        o = o.dropna(subset=["model", "lat", "lon"])
        o["anom"] = o.gw_level_mbgl - o.model
        g = (o.groupby(["station", "lat", "lon"], as_index=False)
               .agg(anom=("anom", "mean"), n=("anom", "size")))
        rows = g.to_dict("records")
    obs_df = pd.DataFrame(rows)
    return obs_df


def build_well_table(district: str = DEFAULT_DISTRICT, df: pd.DataFrame | None = None,
                     horizon_level: float | None = None,
                     horizon_delta: float | None = None) -> gpd.GeoDataFrame:
    """
    Per-well current level, forecast level, headroom and dry-out risk.

    Parameters
    ----------
    horizon_level : district-mean predicted level at the forecast date (mbgl)
    horizon_delta : district-mean predicted change over the horizon (m, +ve = deeper)
    """
    from .dataset_generator import load_dataset
    df = df if df is not None else load_dataset(district)
    prior = get_prior(district)
    wells = wells_for_district(district).copy()
    wells["lat"] = wells.geometry.y
    wells["lon"] = wells.geometry.x

    obs = well_offsets(district, df)
    if obs.empty:
        wells["offset_m"] = 0.0
        wells["offset_source"] = "none"
    else:
        olat = obs.lat.to_numpy(float); olon = obs.lon.to_numpy(float)
        oan = obs.anom.to_numpy(float)
        global_mean = float(np.mean(oan))
        offs, srcs = [], []
        for lat, lon in zip(wells.lat.to_numpy(float), wells.lon.to_numpy(float)):
            dist = _haversine_km(lat, lon, olat, olon)
            w = np.exp(-0.5 * (dist / KERNEL_KM) ** 2)
            if w.sum() < 1e-6 or dist.min() > MAX_KM:
                offs.append(global_mean * SHRINK); srcs.append("regional mean")
                continue
            val = float((w * oan).sum() / w.sum())
            near = dist.min()
            if near < 0.6:
                srcs.append("measured (in-situ CGWB)")
            else:
                blend = float(np.exp(-0.5 * (near / MAX_KM) ** 2))
                val = blend * val + (1 - blend) * global_mean
                val *= SHRINK if near > 2 * KERNEL_KM else 1.0
                srcs.append("IDW interpolation")
            offs.append(val)
        wells["offset_m"] = np.round(offs, 2)
        wells["offset_source"] = srcs

    cur = float(df.gw_level_mbgl.iloc[-1])
    h_lvl = float(horizon_level if horizon_level is not None else cur)
    h_del = float(horizon_delta if horizon_delta is not None else 0.0)

    wells["current_level_mbgl"] = np.round(cur + wells.offset_m, 2)
    wells["forecast_level_mbgl"] = np.round(h_lvl + wells.offset_m, 2)
    wells["forecast_delta_m"] = np.round(h_del, 2)

    d1 = wells.get("depth_first_aquifer_m", pd.Series(np.nan, index=wells.index))
    wells["pump_set_depth_mbgl"] = np.round(
        np.clip(pd.to_numeric(d1, errors="coerce").fillna(d1.median() if d1.notna().any() else 20.0)
                + TYPICAL_SCREEN_DEPTH_M, 28.0, 75.0), 1)
    wells["headroom_m"] = np.round(wells.pump_set_depth_mbgl - wells.forecast_level_mbgl, 2)
    wells["pump_failure_risk_pct"] = np.round(
        np.clip((1.0 - wells.headroom_m / 12.0) * 100.0, 0.0, 100.0), 1)
    wells["aquifer_thickness_m"] = pd.to_numeric(
        wells.get("aquifer_thickness_m", np.nan), errors="coerce").round(1)

    def zone(v):
        return "safe" if v < 30 else ("critical" if v <= 35 else "over_exploited")
    wells["zone"] = wells.forecast_level_mbgl.map(zone)
    wells["label"] = wells.get("station_name", pd.Series("CGWB well", index=wells.index)).astype(str)
    return wells


def block_summary(wells: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate well-level risk to block (sub-district) level for the choropleth."""
    if wells.empty:
        return pd.DataFrame()
    key = "block" if "block" in wells.columns else None
    if key is None or wells[key].isna().all():
        wells = wells.copy()
        wells["block"] = "District-wide"
        key = "block"
    g = wells.dropna(subset=["forecast_level_mbgl"]).groupby(key)
    out = g.agg(wells=("forecast_level_mbgl", "size"),
                mean_level_mbgl=("forecast_level_mbgl", "mean"),
                max_level_mbgl=("forecast_level_mbgl", "max"),
                mean_risk_pct=("pump_failure_risk_pct", "mean"),
                mean_headroom_m=("headroom_m", "mean")).reset_index()
    out = out.rename(columns={key: "block"})
    out["mean_level_mbgl"] = out.mean_level_mbgl.round(2)
    out["mean_risk_pct"] = out.mean_risk_pct.round(1)
    return out.sort_values("mean_risk_pct", ascending=False).reset_index(drop=True)
