#!/usr/bin/env python3
"""
Rebuild the Punjab spatial clips from the *national* India-WRIS downloads.

The raw WRIS exports are India-wide (District Boundary 81 MB, Groundwater Level
Station 75 MB, WIMS 69 MB …).  Loading them on every Streamlit re-run is not an
option, so this script runs once and writes small Punjab-only copies into
``data/`` plus down-sampled, reprojected GeoTIFFs into ``data/rasters/``.

Usage
-----
    python scripts/build_spatial_assets.py            # needs data_raw/ + data_ext/
    python run_pipeline.py --rebuild-spatial

Gotchas handled here
--------------------
* ``District Boundary``, ``State Boundary``, ``International Boundary`` and
  ``District Headquarter`` are stored as **[lat, lon]** instead of the GeoJSON
  ``[lon, lat]`` contract — they are flipped before use.
* The station layers *are* ``[lon, lat]``.  Both conventions are asserted, not
  assumed.
* The soil / aquifer rasters are state-wide or national (up to 49 000 px wide);
  they are windowed + reprojected to a 0.004° (~440 m) Punjab grid so the app can
  sample them instantly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import geopandas as gpd                                    # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
import pyogrio                                             # noqa: E402
import rasterio                                            # noqa: E402
from rasterio.transform import from_origin                 # noqa: E402
from rasterio.warp import Resampling, reproject            # noqa: E402
from shapely.ops import transform as shp_transform         # noqa: E402

from src.config import DATA_DIR, RASTER_DIR                # noqa: E402

RAW = ROOT / "data_raw"
EXT = ROOT / "data_ext"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RASTER_DIR.mkdir(parents=True, exist_ok=True)

PUNJAB_LL = (73.70, 29.40, 77.05, 32.65)      # lon/lat bbox for [lon,lat] layers
PUNJAB_YX = (29.40, 73.70, 32.65, 77.05)      # lat/lon bbox for the flipped layers


def flip(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda g: None if g is None else shp_transform(lambda x, y: (y, x), g))
    return gdf[~gdf.geometry.isna()].set_geometry("geometry")


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def build_vectors() -> None:
    log("vector layers")
    for src, dst, bbox, do_flip in [
        ("District Boundary.geojson", "District Boundary.geojson", PUNJAB_YX, True),
        ("State Boundary.geojson", "State Boundary.geojson", PUNJAB_YX, True),
        ("International Boundary.geojson", "International Boundary.geojson", PUNJAB_YX, True),
        ("District Headquarter.geojson", "District Headquarter.geojson", PUNJAB_YX, True),
        ("Groundwater Level Station.geojson", "Groundwater Level Station.geojson", PUNJAB_LL, False),
        ("Groundwater Station.geojson", "Groundwater Station.geojson", PUNJAB_LL, False),
        ("WIMS Station.geojson", "WIMS Station.geojson", PUNJAB_LL, False),
        ("Rainfall Station.geojson", "Rainfall Station.geojson", PUNJAB_LL, False),
        ("Litholog.geojson", "Litholog.geojson", PUNJAB_LL, False),
    ]:
        p = RAW / src
        if not p.exists():
            log(f"!  missing {src} — skipped")
            continue
        g = pyogrio.read_dataframe(p, bbox=bbox)
        g = g.set_crs(4326)
        if do_flip:
            g = flip(g)
        g.to_file(DATA_DIR / dst, driver="GeoJSON")
        log(f"{src:38s} {len(g):>6d} -> {dst}  ({os.path.getsize(DATA_DIR/dst)/1e6:.2f} MB)")

    # ---- Punjab-only subsets ------------------------------------------------------------
    st = gpd.read_file(DATA_DIR / "State Boundary.geojson").set_crs(4326)
    punjab = st[st["State"].astype(str).str.upper().str.strip() == "PB"]
    if punjab.empty:
        raise RuntimeError("Could not isolate Punjab in the state-boundary layer")
    punjab.to_file(DATA_DIR / "Punjab State Boundary.geojson", driver="GeoJSON")
    poly = punjab.union_all()

    d = gpd.read_file(DATA_DIR / "District Boundary.geojson").set_crs(4326)
    d = d[d.geometry.intersects(poly)]
    d = d[d.State.astype(str).str.upper().str.strip() == "PB"]
    cols = [c for c in ("District", "State", "st_area_shape_") if c in d.columns]
    d[cols + ["geometry"]].to_file(DATA_DIR / "District Boundary.geojson", driver="GeoJSON")
    log(f"Punjab districts: {len(d)} -> {sorted(d.District.unique())}")

    for name, filt in [("Groundwater Level Station.geojson", "state_name"),
                       ("Groundwater Station.geojson", "state_name"),
                       ("WIMS Station.geojson", "state_name"),
                       ("Rainfall Station.geojson", "state_name"),
                       ("Litholog.geojson", "State")]:
        p = DATA_DIR / name
        if not p.exists() or filt not in gpd.list_layers(p):
            continue
        g = gpd.read_file(p).set_crs(4326)
        inside = g[g.geometry.within(poly)]
        named = g[g[filt].astype(str).str.contains("Punjab", case=False, na=False)]
        out = pd.concat([inside, named]).drop_duplicates(
            subset=[g.columns[0]]).reset_index(drop=True)
        out.to_file(p, driver="GeoJSON")
        log(f"{name:38s} {len(g):>6d} -> {len(out):>5d} inside Punjab")


def build_rasters() -> None:
    log("raster layers (windowed + reprojected to a 440 m Punjab grid)")
    PB = (73.85, 29.50, 76.98, 32.56)
    RES = 0.004
    w = int((PB[2] - PB[0]) / RES)
    h = int((PB[3] - PB[1]) / RES)
    dst_transform = from_origin(PB[0], PB[3], RES, RES)

    def find(name: str) -> Path | None:
        hits = sorted(EXT.rglob(name))
        return hits[0] if hits else None

    jobs = [
        ("PB_AQ_THICK.tif", "aquifer_thickness_m.tif", "Aquifer thickness (m)", "float"),
        ("PB_AQ_START.tif", "depth_first_aquifer_m.tif", "Depth to first aquifer (m)", "float"),
        ("Punjab_Rabi_Wheat_2023_2024.img", "rabi_wheat_mask.tif", "Rabi wheat mask 2023-24", "cat"),
        ("SOILDEPTH.tif", "soil_depth.tif", "Soil depth class", "cat"),
        ("SOILTEXTURE.tif", "soil_texture.tif", "Soil texture class", "cat"),
        ("SOILSLOPE.tif", "soil_slope.tif", "Soil slope class", "cat"),
        ("SOILPRODUCTIVITY.tif", "soil_productivity.tif", "Soil productivity class", "cat"),
        ("SOILEROSION.tif", "soil_erosion.tif", "Soil erosion class", "cat"),
    ]
    for src_name, out_name, desc, kind in jobs:
        src = find(src_name)
        if src is None:
            log(f"!  missing raster {src_name}")
            continue
        with rasterio.open(src) as s:
            nod = s.nodata if s.nodata is not None else -9999.0
            dst = np.full((h, w), nod, dtype="float32")
            reproject(source=rasterio.band(s, 1), destination=dst,
                      src_transform=s.transform, src_crs=s.crs,
                      dst_transform=dst_transform, dst_crs="EPSG:4326",
                      src_nodata=s.nodata, dst_nodata=nod,
                      resampling=(Resampling.bilinear if kind == "float"
                                  else Resampling.nearest))
        out = RASTER_DIR / out_name
        with rasterio.open(out, "w", driver="GTiff", height=h, width=w, count=1,
                           dtype="float32", crs="EPSG:4326", transform=dst_transform,
                           nodata=nod, compress="deflate",
                           predictor=(3 if kind == "float" else 1)) as d:
            d.write(dst.astype("float32"), 1)
            d.set_band_description(1, desc)
        with rasterio.open(out) as d:
            a = d.read(1, masked=True)
            log(f"{desc:30s} -> {out_name:30s} {out.stat().st_size/1e6:5.2f} MB  "
                f"valid={a.count():>7d}  range {a.min():.1f}–{a.max():.1f}")


if __name__ == "__main__":
    print("\n=== AquaCast :: building Punjab spatial assets ===")
    if not RAW.exists():
        print(f"  ! {RAW} not found — download the India-WRIS exports first.")
        raise SystemExit(1)
    build_vectors()
    build_rasters()
    print("\n  done.\n")
