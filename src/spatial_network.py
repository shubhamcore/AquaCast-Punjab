"""
MODULE A — Interactive Tubewell Network & Spatial Dependency Engine
===================================================================

Answers the question a block-level officer actually asks: *"if these thirty
tubewells all run at once, how far does the cone of depression reach, and which
monitoring piezometers will feel it?"*

Physics
-------
Well interference is computed with the **Cooper–Jacob** logarithmic
approximation to the Theis solution, valid for small r and large t
(u = r²S / 4Tt < 0.01)::

    s(r, t) = Q / (4 π T) · ln( 2.25 T t / (r² S) )

with

* ``Q``  pumping rate                                [m³/day]
* ``T``  transmissivity  = K · b                     [m²/day]
* ``b``  saturated aquifer thickness                 [m]   (real CGWB raster)
* ``K``  hydraulic conductivity                      [m/day] (from soil texture)
* ``t``  pumping duration                            [days]
* ``r``  distance to the observation point           [m]
* ``S``  storage coefficient                         [–] (specific yield, unconfined)

Because the Theis equation is **linear in Q**, drawdown from a field of wells
superposes. That matters: in Punjab's high-transmissivity alluvium a *single*
7.5 HP tubewell barely nudges its neighbour 300 m away (centimetres), but a
dense village cluster of 30–40 bores running together produces the metres of
regional drawdown that actually desiccates shallow wells. Both numbers are
reported separately, because conflating them is how people talk themselves into
"my pump isn't the problem".

Provenance
----------
* Well coordinates, aquifer thickness, first-aquifer depth and soil texture are
  **real** India-WRIS / CGWB assets read from ``data/``.
* Village-level tubewell positions are **synthetic** (seeded, deterministic),
  because no public registry of private boreholes exists. They are placed on
  the real wheat mask, inside the real district polygon, and are labelled as
  synthetic everywhere they surface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:                                    # graceful degradation (see README)
    import geopandas as gpd
    from shapely.geometry import Point
    _HAVE_GEO = True
except Exception:                       # pragma: no cover
    gpd = None
    Point = None
    _HAVE_GEO = False

from .config import DISTRICTS, RASTERS, get_prior

# ---------------------------------------------------------------- constants --
DEFAULT_PUMP_HOURS_PER_DAY = 8.0    # typical run-time during a irrigation turn
WELL_RADIUS_M = 0.15                # nominal bore radius (Cooper-Jacob singularity)
MIN_R_M = 1.0                       # clamp: the log form blows up as r -> 0

#: Days of continuous seasonal pumping used as the Cooper-Jacob ``t``.
#: A Rabi wheat season needs ~5 irrigation turns; a Kharif paddy season is
#: flooded more or less continuously, so the drawdown cone is far wider.
SEASON_PUMPING_DAYS = {"Rabi": 45.0, "Zaid": 30.0, "Kharif": 110.0}


# =============================================================================
# 1.  Hydraulic properties from the real rasters
# =============================================================================
def hydraulic_properties(lon, lat, prior=None) -> pd.DataFrame:
    """
    Sample the real rasters at (lon, lat) and derive K, b, T and S.

    ``b`` (saturated thickness) is the *confining* geometric control: the
    aquifer-thickness raster gives the total alluvial column, and the
    first-aquifer-depth raster gives how far down you must go before you hit
    saturated material.  Saturated thickness available to a bore is the
    difference, floored at a few metres so thin-hill voxels stay physical.
    """
    lon = np.atleast_1d(np.asarray(lon, dtype="float64"))
    lat = np.atleast_1d(np.asarray(lat, dtype="float64"))
    out = pd.DataFrame({"lon": lon, "lat": lat})

    if not _HAVE_GEO:
        out["K_m_day"] = DEFAULT_K_M_PER_DAY
        out["thickness_m"] = 100.0
        out["depth_first_aquifer_m"] = 20.0
    else:
        gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(lon, lat)],
                               crs="EPSG:4326")
        from .spatial_loader import sample_rasters
        try:
            samp = sample_rasters(gdf, keys=["aquifer_thickness_m",
                                             "depth_first_aquifer_m",
                                             "soil_texture"])
            out["thickness_m"] = _num(samp.get("aquifer_thickness_m"), 120.0, 5.0, 400.0)
            out["depth_first_aquifer_m"] = _num(samp.get("depth_first_aquifer_m"), 20.0, 0.0, 250.0)
            tex = _num(samp.get("soil_texture"), 3.0, 1.0, 4.0)
            out["soil_texture"] = np.round(tex).astype(int)
            out["K_m_day"] = [TEXTURE_K_M_PER_DAY.get(int(t), DEFAULT_K_M_PER_DAY)
                              for t in out.soil_texture]
        except Exception:
            out["K_m_day"] = DEFAULT_K_M_PER_DAY
            out["thickness_m"] = 120.0
            out["depth_first_aquifer_m"] = 20.0
            out["soil_texture"] = 3

    # saturated thickness actually available between the water table and bedrock
    out["saturated_thickness_m"] = (out.thickness_m - out.depth_first_aquifer_m).clip(lower=8.0)
    out["T_m2_day"] = (out.K_m_day * out.saturated_thickness_m).round(1)
    sy = float(getattr(prior, "specific_yield", 0.12) or 0.12) if prior else 0.12
    out["S"] = sy
    return out


def _num(series, default, lo, hi) -> np.ndarray:
    """Coerce a possibly-missing / masked raster sample to a sane float array."""
    if series is None:
        return np.full(1, default)
    a = pd.to_numeric(pd.Series(np.asarray(series).ravel()), errors="coerce")
    a = a.replace([np.inf, -np.inf], np.nan).fillna(default)
    return a.clip(lo, hi).to_numpy()


# =============================================================================
# 2.  Cooper–Jacob interference
# =============================================================================
def cooper_jacob_drawdown(Q_m3_day, T_m2_day, t_days, r_m, S) -> np.ndarray:
    """
    Drawdown s (m) at distance ``r`` after pumping at rate ``Q`` for time ``t``.

    Returns 0 where the logarithmic argument falls below 1 — i.e. outside the
    cone of influence, where the approximation (and the physics) says the
    disturbance has not yet arrived.  That is not a fudge: it is the finite
    propagation speed of a pressure pulse through a real aquifer.
    """
    Q = np.asarray(Q_m3_day, dtype="float64")
    T = np.asarray(T_m2_day, dtype="float64")
    t = np.asarray(t_days, dtype="float64")
    r = np.maximum(np.asarray(r_m, dtype="float64"), MIN_R_M)
    S = np.asarray(S, dtype="float64")

    arg = 2.25 * T * t / (r ** 2 * np.maximum(S, 1e-6))
    s = (Q / (4.0 * math.pi * np.maximum(T, 1e-6))) * np.log(np.maximum(arg, 1.0 + 1e-12))
    return np.where(arg > 1.0, np.maximum(s, 0.0), 0.0)


def influence_radius(Q_m3_day, T_m2_day, t_days, S, threshold_m=0.05) -> float:
    """
    Distance (m) at which drawdown falls to ``threshold_m`` — the closed-form
    inverse of Cooper–Jacob, used to size the drawdown-cone overlay.

        r = sqrt( 2.25 T t / (S · exp(4πT s / Q)) )
    """
    Q = float(max(Q_m3_day, 1e-9)); T = float(max(T_m2_day, 1e-6))
    t = float(max(t_days, 1e-6)); S = float(max(S, 1e-6))
    denom = S * math.exp(4.0 * math.pi * T * threshold_m / Q)
    return float(math.sqrt(2.25 * T * t / denom))


def cone_radii(Q_m3_day, T_m2_day, t_days, S,
               thresholds=(0.05, 0.10, 0.25, 0.50, 1.00)) -> dict:
    """Radii (m) for each drawdown contour — the nested cone polygons."""
    return {f"{th:.2f}": influence_radius(Q_m3_day, T_m2_day, t_days, S, th)
            for th in thresholds}


# =============================================================================
# 3.  Tubewell network
# =============================================================================
@dataclass
class TubewellNetwork:
    """A district's tubewell field, its monitoring piezometers and their coupling."""
    district: str
    wells: pd.DataFrame                  # synthetic village tubewells
    stations: pd.DataFrame               # real CGWB monitoring piezometers
    influence: np.ndarray                # (n_wells x n_stations) metres
    station_influence: np.ndarray        # (n_stations x n_stations) metres
    cumulative_drawdown_m: np.ndarray    # per station, superposed
    meta: dict = field(default_factory=dict)

    def top_dependencies(self, k: int = 60):
        """Strongest (well -> station) couplings, as a tidy frame for the map."""
        if self.influence.size == 0:
            return pd.DataFrame(columns=["wi", "si", "drawdown_m", "dist_m"])
        n = self.influence.shape[0]
        k = min(k, self.influence.size)
        flat = np.argsort(self.influence.ravel())[::-1][:k]
        wi, si = np.unravel_index(flat, self.influence.shape)
        rows = []
        for a, b in zip(wi, si):
            if a >= n:
                continue
            rows.append(dict(
                wi=int(a), si=int(b),
                drawdown_m=float(self.influence[a, b]),
                dist_m=float(_haversine_m(self.wells.lat[a], self.wells.lon[a],
                                          self.stations.lat[b], self.stations.lon[b])),
            ))
        return pd.DataFrame(rows)


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def _distance_matrix_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance matrix (metres) between two point sets."""
    R = 6371008.8
    la1 = np.radians(np.atleast_1d(lat1))[:, None]
    lo1 = np.radians(np.atleast_1d(lon1))[:, None]
    la2 = np.radians(np.atleast_1d(lat2))[None, :]
    lo2 = np.radians(np.atleast_1d(lon2))[None, :]
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def load_monitoring_stations(district: str) -> pd.DataFrame:
    """
    Real CGWB monitoring piezometers inside ``district``.

    Prefers ``Groundwater Level Station`` (the network with level time-series)
    and falls back to ``Groundwater Station`` when the clip is empty.
    """
    if not _HAVE_GEO:
        return pd.DataFrame(columns=["lat", "lon", "station", "source"])
    from .spatial_loader import district_geometry, load_layer

    geom = district_geometry(district)
    frames = []
    for key in ("gw_stations", "groundwater_stations"):
        try:
            g = load_layer(key)
        except Exception:
            continue
        if g is None or g.empty:
            continue
        try:
            g = g[g.geometry.within(geom)] if geom is not None else g
        except Exception:
            try:
                g = g[g.intersects(geom)]
            except Exception:
                pass
        if g.empty:
            continue
        g = g.copy()
        g["lat"] = g.geometry.y
        g["lon"] = g.geometry.x
        name = next((c for c in ("Station_Name", "station", "STATION", "Name", "WellName",
                                 "name", "GmlID", "id") if c in g.columns), None)
        g["station"] = (g[name].astype(str) if name else
                        [f"{key[:2].upper()}-{i:04d}" for i in range(len(g))])
        frames.append(g[["lat", "lon", "station"]].assign(source=key))
        if len(g) >= 12:
            break
    if not frames:
        return pd.DataFrame(columns=["lat", "lon", "station", "source"])
    out = pd.concat(frames, ignore_index=True).dropna(subset=["lat", "lon"])
    return out.reset_index(drop=True)


#: Tubewell density inside a village cluster. Punjab's central districts run
#: 15-40 bores per km² of cultivated area; a cluster of ~30 wells inside a
#: ~700 m radius reproduces that, and it is the *local* density — not the
#: district average — that governs interference.
CLUSTER_SIGMA_DEG = 0.0030      # ~330 m spread of bores around the village centre
CLUSTER_MIN_SEP_DEG = 0.055     # ~6 km between village centres
KM2_PER_VILLAGE = 240.0         # one village cluster per ~240 km² of district

#: Fallback local bore density inside a village (tubewells per km²), used only
#: when the district prior carries no tubewell count. In the normal path the
#: density is *derived* from ``DistrictPrior.tubewells`` divided by the net
#: sown area implied by the wheat/paddy priors — see ``bore_density()``.
LOCAL_BORE_DENSITY_PER_KM2 = 30.0
MAX_TUBEWELLS = 3000            # render/CPU guard for the influence matrix

#: Hydraulic conductivity K (m/day) by NBSS&LUP soil-texture class — alluvial
#: Indo-Gangetic values (sand >> sandy loam > loam >> clay).
TEXTURE_K_M_PER_DAY = {1: 40.0, 2: 25.0, 3: 15.0, 4: 4.0}
DEFAULT_K_M_PER_DAY = 18.0


def bore_density(district: str) -> tuple[float, float, int]:
    """
    Return (local bores per km² of net sown area, net sown km², real tubewells).

    Derived, not assumed: ``DistrictPrior.tubewells`` is the district's real
    Minor-Irrigation tubewell count, and net sown area is taken as the larger
    of the wheat and paddy priors (they occupy the same land in rotation, so
    max() is the net sown footprint — summing them would double-count). Punjab
    works out at 25-40 bores/km², among the highest tubewell densities on
    earth, and that local density is what makes interference matter.
    """
    from .config import get_prior
    prior = get_prior(district)
    n_tw = int(getattr(prior, "tubewells", 0) or 0)
    net_sown_km2 = max(float(getattr(prior, "wheat_area_ha", 0) or 0),
                       float(getattr(prior, "kharif_paddy_area_ha", 0) or 0)) / 100.0
    if n_tw <= 0 or net_sown_km2 <= 0:
        return LOCAL_BORE_DENSITY_PER_KM2, net_sown_km2, n_tw
    return float(n_tw) / net_sown_km2, net_sown_km2, n_tw


def _bores_per_cluster(density_km2: float = LOCAL_BORE_DENSITY_PER_KM2,
                       sigma_deg: float = CLUSTER_SIGMA_DEG) -> int:
    """Village footprint (2 sigma radius, in km²) x local bore density."""
    r_km = 2.0 * sigma_deg * 111.0
    area_km2 = float(np.pi * r_km ** 2)
    return int(max(10, round(density_km2 * area_km2)))


def generate_tubewells(district: str, per_cluster: int | None = None,
                       seed: int = 2024, blocks=None) -> pd.DataFrame:
    """
    Place a deterministic, seeded field of **village tubewell clusters**.

    Synthetic bore positions, and labelled so everywhere they surface.

    Why clusters and not a uniform scatter: spreading 360 bores evenly across
    3,603 km² puts them ~3 km apart, where Cooper–Jacob interference is ~1 cm
    and the whole module says nothing. Punjab's tubewells are not distributed
    that way — they sit 20–40 to a village, a few hundred metres apart, because
    that is where the fields and the three-phase line are. Local density is
    the physics; district-average density is a red herring.

    Cluster centres are rejection-sampled inside the real district polygon and
    (where the mask covers the point) on the real Rabi wheat mask. Each cluster
    inherits the block label of the nearest *real* CGWB well, so the block
    attribution is real even though the bore positions are not.
    """
    cols = ["lon", "lat", "block", "cluster", "synth"]
    dens_km2, net_sown_km2, n_real_tw = bore_density(district)
    if per_cluster is None:
        per_cluster = _bores_per_cluster(dens_km2)
    if not _HAVE_GEO:
        return pd.DataFrame(columns=cols)
    # how many villages would tile the net sown area at this footprint?
    if net_sown_km2 > 0:
        r_km = 2.0 * CLUSTER_SIGMA_DEG * 111.0
        vill_need = int(round(net_sown_km2 / (np.pi * r_km ** 2)))
    else:
        vill_need = 999
    from .spatial_loader import district_geometry, district_area_km2

    geom = district_geometry(district)
    if geom is None:
        return pd.DataFrame(columns=cols)
    minx, miny, maxx, maxy = geom.bounds

    # ---- how many villages? -------------------------------------------------
    try:
        area = float(district_area_km2(district))
    except Exception:
        area = (maxx - minx) * (maxy - miny) * 111 * 111 * 0.85
    n_clusters = int(max(4, min(vill_need, round(area / KM2_PER_VILLAGE))))
    # render/CPU guard: cap the modelled fleet, and remember the shortfall so
    # the UI can state the sample fraction honestly.
    if n_clusters * per_cluster > MAX_TUBEWELLS:
        n_clusters = max(4, MAX_TUBEWELLS // max(per_cluster, 1))

    # ---- real block labels, anchored on real wells --------------------------
    ref_lat = ref_lon = ref_blk = None
    try:
        from .well_network import wells_for_district
        w = wells_for_district(district)
        if "block" in w.columns and len(w):
            w = w.dropna(subset=["block"])
            w = w[w.block.astype(str).str.lower() != "null"]
            if len(w):
                ref_lat = w.geometry.y.to_numpy()
                ref_lon = w.geometry.x.to_numpy()
                ref_blk = w.block.astype(str).to_numpy()
    except Exception:
        pass
    if ref_blk is None:
        ref_blk = np.array([f"CLUSTER-{i+1}" for i in range(n_clusters)])
        ref_lat = ref_lon = None
    # a deterministic rotation so successive clusters get different blocks
    block_cycle = list(ref_blk)

    # ---- wheat mask: bores sit on cropped land ------------------------------
    try:
        import rasterio
        with rasterio.open(RASTERS["rabi_wheat_mask"]) as src:
            mask = src.read(1); nodata = src.nodata
            tr = src.transform; H, Wd = mask.shape
    except Exception:
        mask, nodata, tr, H, Wd = None, None, None, 0, 0

    def on_wheat(lon: float, lat: float) -> bool:
        if mask is None:
            return True
        try:
            col = int((lon - tr.c) / tr.a)
            row = int((lat - tr.f) / tr.e)
            if not (0 <= row < H and 0 <= col < Wd):
                return False
            v = mask[row, col]
            return not (nodata is not None and v == nodata) and float(v) > 0
        except Exception:
            return True

    rng = np.random.default_rng(seed)

    # ---- 1. village centres -------------------------------------------------
    # Weighted, not uniform: we draw a large candidate pool inside the polygon
    # and sample centres with probability ~ 1/(distance to nearest real
    # monitoring station).  Rationale — CGWB does not site piezometers at
    # random either, it sites them in the villages where the pumps are.  A
    # uniform scatter of synthetic villages would leave 95% of the real
    # piezometers sitting in empty countryside and the interference matrix
    # would be near-zero everywhere, which is an artefact of our placement,
    # not a finding about Punjab.
    cand, tries = [], 0
    while len(cand) < 4000 and tries < 40000:
        tries += 1
        lon = rng.uniform(minx, maxx); lat = rng.uniform(miny, maxy)
        if geom.contains(Point(lon, lat)) and on_wheat(lon, lat):
            cand.append((lon, lat))
    if not cand:
        cand = [(geom.centroid.x, geom.centroid.y)]

    if ref_lat is not None and len(ref_lat):
        ca = np.asarray(cand)
        dkm = np.hypot((ca[:, 1:2] - ref_lat[None, :]) * 111.0,
                       (ca[:, 0:1] - ref_lon[None, :]) * 111.0 *
                       np.cos(np.radians(ca[:, 1:2]))).min(axis=1)
        wts = 1.0 / (dkm + 0.75)                       # +0.75 km softens the 1/x
        wts = wts ** 1.5
    else:
        wts = np.ones(len(cand))

    centres = []
    for sep in (CLUSTER_MIN_SEP_DEG, CLUSTER_MIN_SEP_DEG / 2,
                CLUSTER_MIN_SEP_DEG / 4, 0.0):
        centres = []
        order = rng.choice(len(cand), size=len(cand), replace=False,
                           p=wts / wts.sum())
        for idx in order:
            lon, lat = cand[int(idx)]
            if sep > 0 and any((c[0] - lon) ** 2 + (c[1] - lat) ** 2 < sep ** 2
                               for c in centres):
                continue
            centres.append((lon, lat))
            if len(centres) >= n_clusters:
                break
        if len(centres) >= n_clusters or sep == 0.0:
            break
    if not centres:
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        centres = [(cx, cy)]

    # ---- 2. bores around each centre ---------------------------------------
    rows = []
    for ci, (clon, clat) in enumerate(centres):
        # nearest real well -> real block label
        if ref_lat is not None:
            d = np.hypot((ref_lat - clat) * 111.0, (ref_lon - clon) * 111.0)
            block = str(ref_blk[int(np.argmin(d))])
        else:
            block = block_cycle[ci % len(block_cycle)]

        placed, guard = 0, 0
        while placed < per_cluster and guard < per_cluster * 40:
            guard += 1
            lon = clon + rng.normal(0, CLUSTER_SIGMA_DEG)
            lat = clat + rng.normal(0, CLUSTER_SIGMA_DEG)
            if not (minx <= lon <= maxx and miny <= lat <= maxy):
                continue
            if not geom.contains(Point(lon, lat)):
                continue
            if not on_wheat(lon, lat):
                continue
            rows.append(dict(lon=lon, lat=lat, block=block, cluster=f"C{ci:02d}",
                             synth=True))
            placed += 1

    if not rows:                      # last-resort fallback
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        rows = [dict(lon=cx + rng.uniform(-.02, .02), lat=cy + rng.uniform(-.02, .02),
                     block=block_cycle[i % len(block_cycle)], cluster="C00", synth=True)
                for i in range(min(per_cluster * n_clusters, 120))]

    return pd.DataFrame(rows)


# =============================================================================
# 4.  Assemble + solve
# =============================================================================
def build_network(district: str, per_cluster: int | None = None, seed: int = 2024,
                  season: str = "Kharif", hours_per_day: float = DEFAULT_PUMP_HOURS_PER_DAY,
                  discharge_m3h: float | None = None) -> TubewellNetwork:
    """
    Build the tubewell field and solve the interference problem.

    ``discharge_m3h`` defaults to the district's *head-derated* pump delivery
    (see ``advisory_engine.effective_pump_discharge_m3h``) — the same number the
    advisory quotes, so the map and the farmer card can never disagree.
    """
    prior = get_prior(district)

    stations = load_monitoring_stations(district)
    wells = generate_tubewells(district, per_cluster=per_cluster, seed=seed)

    empty = pd.DataFrame(columns=["lat", "lon", "station", "source"])
    if wells.empty or stations.empty:
        return TubewellNetwork(district, wells, stations if not stations.empty else empty,
                               np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0),
                               meta={"ok": False, "reason": "no geometry / no stations"})

    dens_km2, net_sown_km2, n_real_tw = bore_density(district)

    # ---- hydraulic properties at both sets of points ------------------------
    # NB: ``hydraulic_properties`` echoes lon/lat back; drop them before the
    # concat or the duplicate column names make ``.lat`` return a DataFrame.
    wh = hydraulic_properties(wells.lon.to_numpy(), wells.lat.to_numpy(), prior)
    sh = hydraulic_properties(stations.lon.to_numpy(), stations.lat.to_numpy(), prior)
    wells = pd.concat([wells.reset_index(drop=True),
                       wh.drop(columns=["lon", "lat"]).reset_index(drop=True)], axis=1)
    stations = pd.concat([stations.reset_index(drop=True),
                          sh.drop(columns=["lon", "lat"]).reset_index(drop=True)], axis=1)

    # ---- pumping rate -------------------------------------------------------
    if discharge_m3h is None:
        try:
            from .advisory_engine import effective_pump_discharge_m3h
            q_h = effective_pump_discharge_m3h(prior, float(prior.base_depth_2021_mbgl))
        except Exception:
            q_h = float(getattr(prior, "avg_pump_discharge_m3h", 45.0) or 45.0)
    else:
        q_h = float(discharge_m3h)
    Q = q_h * float(hours_per_day)
    wells["pump_m3h"] = q_h
    wells["Q_m3_day"] = Q

    # ---- distances ----------------------------------------------------------
    D_ws = _distance_matrix_m(wells.lat.to_numpy(), wells.lon.to_numpy(),
                              stations.lat.to_numpy(), stations.lon.to_numpy())
    D_ss = _distance_matrix_m(stations.lat.to_numpy(), stations.lon.to_numpy(),
                              stations.lat.to_numpy(), stations.lon.to_numpy())
    t_days = float(SEASON_PUMPING_DAYS.get(season, 60.0))

    # geometric mean of T at source and receiver — the standard way to couple a
    # pumping well in one formation to an observation point in another
    T_w = wells.T_m2_day.to_numpy()[:, None]
    T_s = stations.T_m2_day.to_numpy()[None, :]
    T_ws = np.sqrt(np.maximum(T_w, 1.0) * np.maximum(T_s, 1.0))
    S_ws = 0.5 * (wells.S.to_numpy()[:, None] + stations.S.to_numpy()[None, :])
    infl = cooper_jacob_drawdown(Q, T_ws, t_days, D_ws, S_ws)

    T_ss = np.sqrt(np.outer(stations.T_m2_day.to_numpy(), stations.T_m2_day.to_numpy()))
    S_ss = stations.S.to_numpy()[None, :]
    infl_ss = cooper_jacob_drawdown(Q, T_ss, t_days, D_ss, S_ss)
    np.fill_diagonal(infl_ss, 0.0)

    cum = infl.sum(axis=0)
    stations["cumulative_drawdown_m"] = np.round(cum, 3)
    stations["n_influencing_wells"] = (infl > 0.001).sum(axis=0)
    stations["nearest_well_m"] = np.round(D_ws.min(axis=0), 0)

    meta = {
        "ok": True,
        "district": district,
        "n_wells": int(len(wells)),
        "n_stations": int(len(stations)),
        "season": season,
        "t_days": t_days,
        "pump_m3h": round(float(q_h), 1),
        "Q_m3_day": round(float(Q), 1),
        "hours_per_day": float(hours_per_day),
        "median_T_m2_day": float(np.median(wells.T_m2_day)),
        "median_K_m_day": float(np.median(wells.K_m_day)),
        "median_saturated_thickness_m": float(np.median(wells.saturated_thickness_m)),
        "S": float(wells.S.iloc[0]),
        "max_cumulative_drawdown_m": float(cum.max()),
        "mean_cumulative_drawdown_m": float(cum.mean()),
        "p95_cumulative_drawdown_m": float(np.percentile(cum, 95)) if cum.size else 0.0,
        "stations_over_0p1m": int((cum > 0.10).sum()),
        "stations_over_0p5m": int((cum > 0.50).sum()),
        "n_clusters": int(wells.cluster.nunique()) if "cluster" in wells else 0,
        "bores_per_cluster": int(wells.groupby("cluster").size().median())
                             if "cluster" in wells and len(wells) else 0,
        "local_bore_density_per_km2": float(dens_km2),
        "real_tubewells": int(n_real_tw),
        "net_sown_km2": float(net_sown_km2),
        "sample_fraction": (len(wells) / n_real_tw) if n_real_tw else float("nan"),
        "villages_needed": int(round(net_sown_km2 / (np.pi * (2.0 * CLUSTER_SIGMA_DEG * 111.0) ** 2)))
                           if net_sown_km2 > 0 else 0,
        "wells_are_synthetic": True,
        "stations_are_real": True,
    }
    return TubewellNetwork(district, wells, stations, infl, infl_ss, cum, meta)


def single_well_cone(district: str, lat: float, lon: float,
                     Q_m3_day: float, season: str = "Kharif") -> dict:
    """Drawdown cone parameters for one bore — powers the map overlay."""
    prior = get_prior(district)
    h = hydraulic_properties([lon], [lat], prior).iloc[0]
    t = float(SEASON_PUMPING_DAYS.get(season, 60.0))
    return {
        "lat": float(lat), "lon": float(lon),
        "Q_m3_day": float(Q_m3_day), "t_days": t,
        "K_m_day": float(h.K_m_day),
        "saturated_thickness_m": float(h.saturated_thickness_m),
        "T_m2_day": float(h.T_m2_day), "S": float(h.S),
        "soil_texture": int(getattr(h, "soil_texture", 3)),
        "radii_m": cone_radii(Q_m3_day, h.T_m2_day, t, h.S),
        "self_drawdown_m": float(cooper_jacob_drawdown(Q_m3_day, h.T_m2_day, t,
                                                       WELL_RADIUS_M, h.S)),
    }


def interference_at(Q_m3_day, T_m2_day, t_days, S, distances_m) -> np.ndarray:
    """Convenience wrapper used by the farmer-side 'what if my neighbour pumps?' tool."""
    return cooper_jacob_drawdown(Q_m3_day, T_m2_day, t_days, distances_m, S)


# =============================================================================
# 5.  Self-check
# =============================================================================
def self_check(district: str = "Sangrur", verbose: bool = True) -> dict:
    net = build_network(district)
    if verbose:
        m = net.meta
        print(f"\n[Module A] Tubewell network — {district}")
        print("-" * 66)
        if not m.get("ok"):
            print(f"  unavailable: {m.get('reason')}")
            return m
        print(f"  tubewells (synthetic)     : {m['n_wells']}")
        print(f"  monitoring piezometers    : {m['n_stations']}  (real CGWB)")
        print(f"  pump                      : {m['pump_m3h']:.1f} m³/h × "
              f"{m['hours_per_day']:.0f} h/d = {m['Q_m3_day']:.0f} m³/d")
        print(f"  median K / thickness / T  : {m['median_K_m_day']:.1f} m/d · "
              f"{m['median_saturated_thickness_m']:.0f} m · {m['median_T_m2_day']:,.0f} m²/d")
        print(f"  season                    : {m['season']} ({m['t_days']:.0f} d of pumping)")
        print(f"  village clusters          : {m['n_clusters']} x "
              f"{m['bores_per_cluster']} bores "
              f"({m['local_bore_density_per_km2']:.1f} bores/km² derived) "
              f"= {m['n_wells']} of {m['real_tubewells']:,} real bores "
              f"({100*m['sample_fraction']:.1f}% sample)")
        print(f"  superposed drawdown       : max {m['max_cumulative_drawdown_m']:.2f} m · "
              f"p95 {m['p95_cumulative_drawdown_m']:.2f} m · "
              f"mean {m['mean_cumulative_drawdown_m']:.2f} m")
        print(f"  piezometers affected      : {m['stations_over_0p1m']} over 0.10 m · "
              f"{m['stations_over_0p5m']} over 0.50 m  (of {m['n_stations']})")
        single = net.influence.max() if net.influence.size else 0.0
        print(f"  strongest SINGLE coupling : {single:.3f} m "
              f"— one tubewell on one piezometer")
        print("  (the gap between those two lines is the point: individual bores")
        print("   barely register; the cluster is what drains the piezometer)")
        print("-" * 66)
    return net.meta


if __name__ == "__main__":
    for d in DISTRICTS:
        self_check(d)
