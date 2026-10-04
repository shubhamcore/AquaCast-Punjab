"""
AquaCast-Punjab :: Module 1 — Spatial Ingestion & Punjab Crop Baseline
=====================================================================

Loads every India-WRIS / CGWB vector + raster asset shipped under ``data/``,
clips it to Punjab, joins the raster covariates (aquifer thickness, depth to the
first aquifer, soil texture/slope/productivity, Rabi-wheat mask) onto the
observation-well network, and merges the calibrated Rabi-wheat crop baseline.

Design notes
------------
* India-WRIS ships *national* GeoJSONs (70-150 MB).  ``scripts/build_spatial_assets.py``
  clips them once to the Punjab envelope; this module only ever reads the small
  clipped copies, so the Streamlit app boots in < 1 s.
* Several WRIS boundary files are stored as ``[lat, lon]`` instead of the GeoJSON
  ``[lon, lat]`` contract.  The builder script flips them; ``_assert_wgs84``
  re-checks at load time and auto-heals if a raw file is ever dropped in.
* Every layer is loaded lazily + memoised (``functools.lru_cache``) because
  Streamlit re-executes the script on every widget interaction.
"""
from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.sample import sample_gen
from shapely.geometry import Point, box

from .config import (DATA_DIR, DEFAULT_DISTRICT, DISTRICT_PRIORS, GEO_LAYERS,
                     RASTERS, DistrictPrior)

warnings.filterwarnings("ignore", category=UserWarning)
PUNJAB_BBOX = (73.8, 29.4, 77.0, 32.7)   # lon/lat envelope


# ----------------------------------------------------------------------------------
# low-level helpers
# ----------------------------------------------------------------------------------
def _assert_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """WRIS boundary files sometimes ship as [lat, lon]; flip them if so."""
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    x0, y0, x1, y1 = gdf.total_bounds
    if x0 > 40 and y1 > 40:                       # x looks like latitude
        gdf = gdf.set_geometry(gdf.geometry.apply(
            lambda g: None if g is None else _flip_geom(g)))
        gdf = gdf[~gdf.geometry.isna()]
    return gdf


def _flip_geom(geom):
    from shapely.ops import transform
    return transform(lambda x, y: (y, x), geom)


@lru_cache(maxsize=16)
def load_layer(key: str) -> gpd.GeoDataFrame:
    """Load (and memoise) one clipped vector layer by registry key."""
    path = GEO_LAYERS[key]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Spatial asset missing: {path}\n"
            "Run `python scripts/build_spatial_assets.py` (or `python run_pipeline.py --rebuild-spatial`) "
            "to rebuild the Punjab clips from the India-WRIS source files.")
    gdf = gpd.read_file(path)
    return _assert_wgs84(gdf)


@lru_cache(maxsize=1)
def punjab_districts() -> gpd.GeoDataFrame:
    """22 Punjab district polygons with computed geodesic area (km²)."""
    gdf = load_layer("districts").copy()
    gdf["area_km2"] = gdf.geometry.to_crs(32643).area / 1e6
    gdf["district"] = gdf["District"].astype(str).str.strip().str.title()
    return gdf[["district", "State", "area_km2", "geometry"]].sort_values("district")


def district_geometry(name: str):
    gdf = punjab_districts()
    hit = gdf[gdf.district.str.lower() == name.lower()]
    if hit.empty:
        raise KeyError(f"District '{name}' not found. Available: {sorted(gdf.district)}")
    return hit.geometry.union_all()


def district_area_km2(name: str) -> float:
    gdf = punjab_districts()
    hit = gdf[gdf.district.str.lower() == name.lower()]
    return float(hit.area_km2.iloc[0]) if len(hit) else float("nan")


# ----------------------------------------------------------------------------------
# raster covariates
# ----------------------------------------------------------------------------------
def sample_rasters(points: gpd.GeoDataFrame, keys: Sequence[str] | None = None
                   ) -> pd.DataFrame:
    """Bilinear/nearest sampling of every clipped raster at the given points."""
    keys = list(keys or RASTERS.keys())
    if points.empty:
        return pd.DataFrame(index=points.index, columns=keys, dtype=float)
    coords = list(zip(points.geometry.x, points.geometry.y))
    out = pd.DataFrame(index=points.index)
    for key in keys:
        path = RASTERS.get(key)
        if path is None or not Path(path).exists():
            out[key] = np.nan
            continue
        with rasterio.open(path) as src:
            nodata = src.nodata
            vals = np.array([v[0] for v in sample_gen(src, coords)], dtype="float64")
            if nodata is not None:
                vals = np.where(np.isclose(vals, nodata), np.nan, vals)
            out[key] = vals
    return out


# ----------------------------------------------------------------------------------
# observation-well network
# ----------------------------------------------------------------------------------
_STATION_RENAMES = {
    "district__name": "district", "district_name": "district",
    "block__name": "block", "block_name": "block",
}

@lru_cache(maxsize=4)
def groundwater_wells(source: str = "gw_stations") -> gpd.GeoDataFrame:
    """All CGWB / Punjab-GW observation wells inside Punjab, with covariates."""
    gdf = load_layer(source).copy()
    gdf = gdf.rename(columns=_STATION_RENAMES)
    if "district" in gdf.columns:
        gdf["district"] = gdf["district"].astype(str).str.strip().str.title()
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.within(box(*PUNJAB_BBOX))]
    gdf = gdf.reset_index(drop=True)

    # de-duplicate: CGWB and Punjab-GW publish the same physical well twice
    key = (gdf["lat"].round(4).astype(str) + "_" + gdf["long"].round(4).astype(str)
           if {"lat", "long"}.issubset(gdf.columns)
           else gdf.geometry.x.round(4).astype(str) + "_" + gdf.geometry.y.round(4).astype(str))
    gdf = gdf.assign(_key=key).drop_duplicates(subset="_key").drop(columns="_key")
    gdf = gdf.reset_index(drop=True)

    cov = sample_rasters(gdf)
    gdf = pd.concat([gdf.drop(columns=[c for c in cov.columns if c in gdf.columns]), cov], axis=1)
    return gdf


def wells_for_district(name: str, min_thickness_m: float = 0.0) -> gpd.GeoDataFrame:
    """Observation wells inside a district polygon, with raster covariates attached."""
    gdf = groundwater_wells()
    poly = district_geometry(name)
    sub = gdf[gdf.geometry.within(poly)].copy()
    if sub.empty:                       # relax to intersects for edge wells
        sub = gdf[gdf.geometry.intersects(poly)].copy()
    sub = sub[~sub.geometry.isna()].reset_index(drop=True)
    if "aquifer_thickness_m" in sub and min_thickness_m:
        sub = sub[sub.aquifer_thickness_m.fillna(0) >= min_thickness_m]
    sub["well_id"] = [f"{name[:3].upper()}-W{i:03d}" for i in range(1, len(sub) + 1)]
    sub["station_label"] = sub.get("station_name", pd.Series(index=sub.index, dtype=str)).fillna("CGWB Well")
    return sub


@lru_cache(maxsize=8)
def rainfall_stations(district: str | None = None) -> gpd.GeoDataFrame:
    gdf = load_layer("rainfall_stations").copy().rename(columns=_STATION_RENAMES)
    if district and "district" in gdf.columns:
        gdf = gdf[gdf.district.astype(str).str.title() == district.title()]
    return gdf.reset_index(drop=True)


# ----------------------------------------------------------------------------------
# Module 1 public API
# ----------------------------------------------------------------------------------
def get_prior(district: str = DEFAULT_DISTRICT) -> DistrictPrior:
    if district not in DISTRICT_PRIORS:
        raise KeyError(f"No calibrated priors for '{district}'. Known: {list(DISTRICT_PRIORS)}")
    return DISTRICT_PRIORS[district]


def crop_baseline(district: str = DEFAULT_DISTRICT) -> dict:
    """The calibrated Rabi-wheat baseline for a district, fully derived."""
    p = get_prior(district)
    area_km2 = district_area_km2(district)
    gross_area_ha = area_km2 * 100
    return {
        "district": district,
        "geographic_area_km2": round(area_km2, 1),
        "gross_cropped_area_ha": round(gross_area_ha, 0),
        "wheat_area_ha": p.wheat_area_ha,
        "wheat_area_share_pct": round(100 * p.wheat_area_ha / gross_area_ha, 1),
        "wheat_season_depth_mm": p.wheat_season_depth_mm,
        "rabi_draft_mcm": p.rabi_draft_mcm,
        "kharif_paddy_area_ha": p.kharif_paddy_area_ha,
        "specific_yield": p.specific_yield,
        "aquifer_storage_mcm_per_m": round(p.specific_yield * area_km2 * 1e6 / 1e6, 1),
        "baseline_depth_2021_mbgl": p.base_depth_2021_mbgl,
        "long_term_decline_m_per_yr": p.long_term_decline_m_per_yr,
        "tubewells": p.tubewells,
        "canal_share_pct": p.canal_share_pct,
        "active_farm_loans": p.active_farm_loans,
        "avg_ticket_inr": p.avg_ticket_inr,
    }


def spatial_summary(district: str = DEFAULT_DISTRICT) -> dict:
    """Everything Module 1 knows about a district — used by the dashboard header."""
    wells = wells_for_district(district)
    base = crop_baseline(district)
    thick = wells.get("aquifer_thickness_m", pd.Series(dtype=float)).dropna()
    depth1 = wells.get("depth_first_aquifer_m", pd.Series(dtype=float)).dropna()
    return {
        **base,
        "n_observation_wells": int(len(wells)),
        "median_aquifer_thickness_m": round(float(thick.median()), 1) if len(thick) else float("nan"),
        "median_depth_first_aquifer_m": round(float(depth1.median()), 1) if len(depth1) else float("nan"),
        "well_depth_p10_m": round(float(thick.quantile(.1)), 1) if len(thick) else float("nan"),
        "well_depth_p90_m": round(float(thick.quantile(.9)), 1) if len(thick) else float("nan"),
    }


def district_geojson(district: str) -> dict:
    """GeoJSON FeatureCollection for a single district (for map layers)."""
    gdf = punjab_districts()
    return json.loads(gdf[gdf.district.str.lower() == district.lower()].to_json())


def self_check(district: str = DEFAULT_DISTRICT, verbose: bool = True) -> dict:
    """Fast, human-readable validation of every spatial asset."""
    rows = {}
    for key in GEO_LAYERS:
        try:
            g = load_layer(key)
            rows[key] = f"{len(g):>6d} features | {list(g.columns)[0]}.."
        except Exception as exc:                                    # pragma: no cover
            rows[key] = f"MISSING ({type(exc).__name__})"
    for key, path in RASTERS.items():
        rows[f"raster::{key}"] = ("ok" if Path(path).exists() else "MISSING")
    s = spatial_summary(district)
    if verbose:
        print("\n[Module 1] Spatial ingestion self-check")
        print("-" * 66)
        for k, v in rows.items():
            print(f"  {k:34s} {v}")
        print("-" * 66)
        print(f"  District                 : {district}")
        print(f"  Geographic area          : {s['geographic_area_km2']:,.1f} km²")
        print(f"  Wheat area               : {s['wheat_area_ha']:,.0f} ha ({s['wheat_area_share_pct']}% of GCA)")
        print(f"  Rabi seasonal draft      : {s['rabi_draft_mcm']:,.0f} MCM over 01-Nov -> 30-Apr")
        print(f"  Specific yield Sy        : {s['specific_yield']}")
        print(f"  Aquifer storage          : {s['aquifer_storage_mcm_per_m']:,.1f} MCM per metre of drawdown")
        print(f"  Observation wells        : {s['n_observation_wells']}")
        print(f"  Median aquifer thickness : {s['median_aquifer_thickness_m']} m")
        print(f"  Median 1st-aquifer depth : {s['median_depth_first_aquifer_m']} m")
        print("-" * 66 + "\n")
    return {"layers": rows, "summary": s}


if __name__ == "__main__":
    self_check()
