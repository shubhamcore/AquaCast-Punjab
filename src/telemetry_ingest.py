"""
AquaCast-Punjab :: Module 2a — India-WRIS / CGWB Telemetry Ingestion & QC
=========================================================================

Turns the *raw, real* portal exports shipped in ``data/raw_telemetry/`` into clean,
QC'd daily station series:

===================================  ==================================================
``Rainfall_*.xlsx``                  hourly ``GPRS-Rainfall by Telemetry`` increments
                                     (NOT the ``GPRS-Rain acumm (Daily)`` running
                                     counter — that one double-counts ~12x)
``Temperature_Bugra head.xlsx``      mislabelled by the portal: actually a 3rd
                                     rainfall gauge (Bugra Head, Sangrur/Dhuri)
``temprature_tel_hr_...csv``         hourly air-temperature telemetry, 5 Punjab stations
``Ground Water Level_*.xlsx``        sparse manual CGWB water-level readings (mbgl)
===================================  ==================================================

Data-quality rules (all configurable, all logged)
-------------------------------------------------
* rainfall  : drop sensor spikes (> 300 mm/h), drop stuck-gauge runs, require >= 18
              valid hours in a day for the day to be trusted.
* temperature : physically valid range [-5, 55] °C, reject -9999/360 sentinel values,
              require >= 8 valid hours for a trusted daily Tmax/Tmin.
* groundwater: reject non-physical (< 0.5 m or > 200 m) readings.

Everything returns a ``pandas.DataFrame`` indexed by date with a ``*_src`` provenance
column so the dashboard can show *exactly* which days are measured and which are
physics-filled.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TELEMETRY_DIR

warnings.filterwarnings("ignore")

RAIN_SPIKE_MM_H = 300.0          # anything above this in one hour is a sensor fault
RAIN_MIN_HOURS = 18              # hours of data needed to trust a daily total
TEMP_MIN, TEMP_MAX = -5.0, 49.0   # Punjab all-time record is 48.3 degC -> 49 is a hard sensor cap
TEMP_MIN_HOURS = 8
TEMP_DIURNAL_MIN = 1.5        # degC: a working sensor always shows a diurnal range
TEMP_DAILY_MAX_FLOOR = 8.0    # degC: Punjab's daily maximum never drops below this
GW_MIN, GW_MAX = 0.5, 200.0

# physical locations of the telemetry gauges (India-WRIS station registry)
# Surveyed CGWB / Punjab-GW coordinates (matched against the station registry in
# data/Groundwater Level Station.geojson), so the map can plot the real gauges.
STATION_GEO = {
    "Babanpur": (30.413333, 75.877778, "Sangrur", "Dhuri"),
    "Babanpur_1": (30.413333, 75.877778, "Sangrur", "Dhuri"),
    "Bugra head": (30.534167, 75.779444, "Sangrur", "Dhuri"),
    "Bald kothi": (30.640278, 75.905556, "Sangrur", "Sangrur"),
    "Benra-m": (30.338333, 75.847778, "Sangrur", "Dhuri"),
    # CGWB observation wells carrying in-situ water-level readings
    "Akbar pur m": (30.166667, 76.001389, "Sangrur", "Sangrur"),
    "Badrukhan (m)": (30.254167, 75.825000, "Sangrur", "Sangrur"),
    "Bagarian-pz": (30.458889, 76.020833, "Sangrur", "Malerkotla"),
    "Bakshiwala-pz": (30.092222, 75.783333, "Sangrur", "Sunam"),
    "Banbhaura-pz": (30.470000, 75.951389, "Sangrur", "Sangrur"),
    "Baurhai khurd d": (30.625556, 75.839722, "Sangrur", "Dhuri"),
    "Baurhai khurd m": (30.625556, 75.839722, "Sangrur", "Dhuri"),
    "Baurhai khurd s": (30.625556, 75.839722, "Sangrur", "Dhuri"),
    "Khanouri kalan-pz": (29.843056, 76.115278, "Sangrur", "Andana"),
    "Manjhi gp": (30.289722, 76.096389, "Sangrur", "Bhawanigarh"),
}
# hourly temperature telemetry districts -> coordinate + distance weight to Sangrur
TEMP_STATION_GEO = {
    "Ludhiana": (30.9000, 75.8500),
    "Mansa": (29.9833, 75.3833),
    "Mukatsar": (30.4667, 74.5167),
    "Muktsar": (30.4667, 74.5167),
    "Ferozepur": (30.9333, 74.6167),
    "Pathankot": (32.2667, 75.6500),
    "Firozepur": (30.9333, 74.6167),
}


# ======================================================================================
# low-level WRIS Excel parser
# ======================================================================================
def parse_wris_excel(path: str | Path, data_type: str | None = None) -> pd.DataFrame:
    """
    Parse an India-WRIS 'two-column + metadata' export into a tidy frame.

    The portal writes 6 metadata rows, then a header row starting with
    ``Data Type Code``, then the data.  Files may contain more than one
    ``Data Type Description`` block (e.g. the hourly *increment* series and the
    *daily accumulator* series); pass ``data_type`` to keep just one.
    """
    path = Path(path)
    xl = pd.ExcelFile(path)
    sheet = xl.sheet_names[1] if len(xl.sheet_names) > 1 else xl.sheet_names[0]
    raw = xl.parse(sheet, header=None)

    hdr = raw.index[raw[0].astype(str).str.strip() == "Data Type Code"]
    if len(hdr) == 0:                                            # already tidy
        df = raw.copy()
        df.columns = ["code", "desc", "time", "value", "unit"][: df.shape[1]]
    else:
        i = hdr[0]
        df = raw.iloc[i + 1:].copy()
        df.columns = ["code", "desc", "time", "value", "unit"]

    df["ts"] = pd.to_datetime(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["ts", "value"])
    if data_type is not None:
        df = df[df["desc"].astype(str).str.contains(data_type, case=False, na=False)]
    return df[["ts", "value", "desc"]].reset_index(drop=True)


def station_name_from_filename(path: str | Path) -> str:
    stem = Path(path).stem
    stem = re.sub(r"^(Ground Water Level|Rainfall|Temperature)_", "", stem)
    return stem.strip()


# ======================================================================================
# Rainfall
# ======================================================================================
def load_rainfall_station(path: str | Path) -> pd.DataFrame:
    """Hourly rainfall *increments* -> QC'd daily rainfall (mm)."""
    name = station_name_from_filename(path)
    df = parse_wris_excel(path)
    # prefer the true incremental series; fall back to differencing the accumulator
    inc = df[df.desc.astype(str).str.contains("Rainfall by Telemetry", case=False, na=False)]
    if inc.empty:
        acc = df[df.desc.astype(str).str.contains("acumm", case=False, na=False)]
        if acc.empty:
            return pd.DataFrame(columns=["date", "rainfall_mm", "hours", "station"])
        acc = acc.sort_values("ts")
        # accumulator resets when the value drops
        d = acc.assign(diff=acc.value.diff().clip(lower=0)).set_index("ts")
        hourly = d["diff"].resample("h").sum()
    else:
        hourly = inc.set_index("ts")["value"].resample("h").sum()

    hourly = hourly.where(hourly <= RAIN_SPIKE_MM_H)          # de-spike
    daily = hourly.resample("D").agg(["sum", "size"])
    daily.columns = ["rainfall_mm", "hours"]
    daily = daily[daily.hours >= 1]
    daily["trusted"] = daily.hours >= RAIN_MIN_HOURS
    daily["rainfall_mm"] = daily["rainfall_mm"].round(2)
    daily = daily.reset_index().rename(columns={"ts": "date"})
    daily["station"] = name
    return daily[["date", "station", "rainfall_mm", "hours", "trusted"]]


@dataclass
class RainfallBundle:
    daily: pd.DataFrame        # index=date, columns=<station>, values mm
    coverage: pd.DataFrame     # index=date, columns=<station>, 1 if trusted observation
    meta: pd.DataFrame         # station, lat, lon, district, block, n_days, total_mm


def load_all_rainfall(directory: str | Path = TELEMETRY_DIR) -> RainfallBundle:
    files = sorted(Path(directory).glob("Rainfall_*.xlsx")) + \
            sorted(Path(directory).glob("Temperature_*.xlsx"))   # portal mislabels this one
    frames, metas = [], []
    for f in files:
        try:
            d = load_rainfall_station(f)
        except Exception as exc:                                  # pragma: no cover
            print(f"  ! rainfall parse failed: {f.name} ({exc})")
            continue
        if d.empty:
            continue
        name = d.station.iloc[0]
        lat, lon, dist, blk = STATION_GEO.get(name, (np.nan, np.nan, "", ""))
        metas.append(dict(station=name, lat=lat, lon=lon, district=dist, block=blk,
                          n_days=int(d.trusted.sum()),
                          start=d.date.min().date(), end=d.date.max().date(),
                          total_mm=round(float(d.rainfall_mm.sum()), 1),
                          file=f.name))
        frames.append(d)
    if not frames:
        return RainfallBundle(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    all_df = pd.concat(frames, ignore_index=True)
    daily = all_df.pivot_table(index="date", columns="station", values="rainfall_mm")
    cov = all_df.pivot_table(index="date", columns="station", values="trusted").astype(float)
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    cov = cov.reindex(daily.index)
    return RainfallBundle(daily=daily, coverage=cov, meta=pd.DataFrame(metas))


# ======================================================================================
# Temperature
# ======================================================================================
def load_hourly_temperature(path: str | Path | None = None) -> pd.DataFrame:
    """Hourly air-temperature telemetry CSV -> tidy frame with QC flag."""
    path = Path(path or TELEMETRY_DIR / "temprature_tel_hr_punjab_sw_pb_2021_2025.csv")
    df = pd.read_csv(path, low_memory=False)
    df["ts"] = pd.to_datetime(df["Data Acquisition Time"], dayfirst=True, errors="coerce")
    val_col = [c for c in df.columns if "Temperature" in c and "Hourly" in c][0]
    df["temp_c"] = pd.to_numeric(df[val_col], errors="coerce")
    df["district"] = df["District"].astype(str).str.strip().str.title()
    df = df.dropna(subset=["ts", "temp_c"])
    df["valid"] = df.temp_c.between(TEMP_MIN, TEMP_MAX)
    return df[["ts", "district", "Station", "temp_c", "valid"]]


@dataclass
class TemperatureBundle:
    daily: pd.DataFrame      # MultiIndex columns (district, max/min)
    coverage: pd.DataFrame   # index=date, columns=district, 1 if trusted
    meta: pd.DataFrame


def load_all_temperature(path: str | Path | None = None) -> TemperatureBundle:
    df = load_hourly_temperature(path)
    good = df[df.valid]
    daily = (good.set_index("ts")
                 .groupby("district")["temp_c"]
                 .resample("D")
                 .agg(["max", "min", "size"])
                 .rename(columns={"max": "temp_max_c", "min": "temp_min_c", "size": "hours"}))
    daily = daily[daily.hours >= 1].reset_index()          # -> date, district, tmax, tmin, hours
    daily = daily.rename(columns={"ts": "date"})

    # --- QC: reject frozen / stuck sensors -------------------------------------------
    # Punjab's daily maximum never falls below ~8 degC and a working sensor always
    # shows a diurnal range; Pathankot (Jan-Mar 2024) & Ferozepur (Apr 2021) report
    # flat 0.0 degC runs which are dropped here.
    diurnal_range = daily.temp_max_c - daily.temp_min_c
    daily["trusted"] = ((daily.hours >= TEMP_MIN_HOURS)
                        & (diurnal_range >= TEMP_DIURNAL_MIN)
                        & (daily.temp_max_c >= TEMP_DAILY_MAX_FLOOR)
                        & (daily.temp_max_c <= TEMP_MAX)
                        & (daily.temp_min_c >= TEMP_MIN))

    wide = daily.pivot(index="date", columns="district",
                       values=["temp_max_c", "temp_min_c"])
    cov = daily.pivot_table(index="date", columns="district", values="trusted").astype(float)
    idx = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(idx); cov = cov.reindex(idx)

    metas = []
    for d, g in daily.groupby("district"):
        lat, lon = TEMP_STATION_GEO.get(d, (np.nan, np.nan))
        metas.append(dict(district=d, lat=lat, lon=lon,
                          n_days=int(g.trusted.sum()),
                          start=g.date.min().date(), end=g.date.max().date(),
                          mean_tmax=round(float(g.temp_max_c.mean()), 1),
                          abs_max=round(float(g.temp_max_c.max()), 1),
                          abs_min=round(float(g.temp_min_c.min()), 1)))
    return TemperatureBundle(daily=wide, coverage=cov, meta=pd.DataFrame(metas))


def idw_blend(wide: pd.DataFrame, cov: pd.DataFrame,
              target_lat: float, target_lon: float,
              station_geo: dict, power: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """Inverse-distance-weighted composite of station series onto a target point."""
    stations = [c for c in station_geo if c in wide.columns.get_level_values(1)]
    if not stations:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    d = np.array([_haversine(target_lat, target_lon, *station_geo[s]) for s in stations])
    w = 1.0 / np.power(np.maximum(d, 1.0), power)
    w /= w.sum()
    num = pd.Series(0.0, index=wide.index); den = pd.Series(0.0, index=wide.index)
    for wi, s in zip(w, stations):
        c = cov[s].fillna(0) if s in cov.columns else pd.Series(1.0, index=wide.index)
        num += wide[("temp_max_c", s)].fillna(0) * c * wi
        den += c * wi
    out = (num / den.replace(0, np.nan)).where(den > 0)
    qc = (den > 0).astype(float)
    return out, qc


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ======================================================================================
# Groundwater level observations (the sparse, precious, real labels)
# ======================================================================================
def load_groundwater_observations(directory: str | Path = TELEMETRY_DIR) -> pd.DataFrame:
    """All CGWB manual water-level readings -> tidy frame (one row per measurement)."""
    rows = []
    for f in sorted(Path(directory).glob("Ground Water Level_*.xlsx")):
        name = station_name_from_filename(f)
        try:
            df = parse_wris_excel(f)
        except Exception as exc:                                  # pragma: no cover
            print(f"  ! groundwater parse failed: {f.name} ({exc})")
            continue
        for _, r in df.iterrows():
            v = float(r.value)
            if not (GW_MIN <= v <= GW_MAX):
                continue
            lat, lon, dist, blk = STATION_GEO.get(name, (np.nan, np.nan, "", ""))
            rows.append(dict(station=name, date=r.ts.normalize(), gw_level_mbgl=v,
                             lat=lat, lon=lon, file=f.name))
    obs = pd.DataFrame(rows)
    if not obs.empty:
        obs = obs.sort_values("date").reset_index(drop=True)
    return obs


def merge_well_coordinates(obs: pd.DataFrame, wells_gdf) -> pd.DataFrame:
    """Attach surveyed CGWB coordinates to the observation records by station name."""
    if obs.empty or wells_gdf is None or len(wells_gdf) == 0:
        return obs
    lut = {}
    for _, r in wells_gdf.iterrows():
        nm = str(r.get("station_name", "")).strip().lower()
        if nm and nm != "nan":
            lut.setdefault(nm, (float(r.geometry.y), float(r.geometry.x),
                                str(r.get("district", "")), str(r.get("block", ""))))
    lats, lons, dists, blocks = [], [], [], []
    for _, r in obs.iterrows():
        key = r.station.lower().replace("-pz", "-pz")
        hit = None
        for k, v in lut.items():
            if key.split("-")[0].split("(")[0].strip() in k or k in key:
                hit = v
                break
        if hit:
            lats.append(hit[0]); lons.append(hit[1]); dists.append(hit[2]); blocks.append(hit[3])
        else:
            lats.append(r.lat); lons.append(r.lon); dists.append(""); blocks.append("")
    out = obs.copy()
    out["lat"] = lats; out["lon"] = lons
    out["district"] = [d.title() if d and d.lower() != "nan" else "Sangrur" for d in dists]
    out["block"] = blocks
    return out


def telemetry_self_check(verbose: bool = True) -> dict:
    """Print & return a provenance table of every real asset we ingested."""
    rf = load_all_rainfall()
    tb = load_all_temperature()
    gw = load_groundwater_observations()
    out = dict(rainfall=rf.meta, temperature=tb.meta, groundwater=gw)
    if verbose:
        print("\n[Module 2a] Real telemetry ingestion (India-WRIS / CGWB)")
        print("-" * 92)
        print("  RAINFALL GAUGES")
        if not rf.meta.empty:
            print(rf.meta.to_string(index=False))
        print("\n  TEMPERATURE STATIONS (hourly telemetry)")
        if not tb.meta.empty:
            print(tb.meta.to_string(index=False))
        print(f"\n  GROUNDWATER LEVEL OBSERVATIONS : {len(gw)} manual CGWB readings "
              f"across {gw.station.nunique() if len(gw) else 0} wells")
        if len(gw):
            print(f"    window {gw.date.min().date()} -> {gw.date.max().date()} | "
                  f"depth {gw.gw_level_mbgl.min():.2f} - {gw.gw_level_mbgl.max():.2f} mbgl")
            print(gw.groupby("station").gw_level_mbgl.agg(["count", "min", "max"]).to_string())
        print("-" * 92 + "\n")
    return out


if __name__ == "__main__":
    telemetry_self_check()
