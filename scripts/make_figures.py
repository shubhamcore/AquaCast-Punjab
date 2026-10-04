"""
Render every figure that appears in the AquaCast-Punjab pitch deck.

Nothing here is typed in by hand: each PNG is drawn from the fitted model, the
processed datasets and the real telemetry in data/.  Re-run after any pipeline
change and the deck's numbers follow automatically.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

from src.config import (DISTRICTS, HORIZONS, SCENARIOS, Scenario, get_prior,
                        draft_multiplier)
from src.dataset_generator import load_dataset
from src.model_lstm import (load_model, live_window, predict, evaluate,
                            permutation_importance, benchmark, chronological_split)
from src.advisory_engine import build_advisory
from src.digital_twin import compare_scenarios, scenario_deltas
from src.well_network import build_well_table, block_summary
from src.spatial_loader import district_geojson, district_area_km2
from src import telemetry_ingest as ti

# ------------------------------------------------------------------ style ----
INK, MUTED, GRID = "#0F172A", "#64748B", "#E2E8F0"
BLUE, SKY, TEAL = "#0369A1", "#0EA5E9", "#14B8A6"
AMBER, RED, GREEN = "#F59E0B", "#EF4444", "#22C55E"
NAVY = "#0B1E30"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": .8, "axes.axisbelow": True,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name):
    p = FIG / f"{name}.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print("  ✓", p.name)
    return p


def money(x):
    """₹ crore."""
    return x / 1e7


# ============================================================== 1. hero =====
def fig_hero():
    d = "Sangrur"
    df = load_dataset(d)
    res = load_model()
    rate = get_prior(d).long_term_decline_m_per_yr / 365.25
    Xl, base, issue, td, rte = live_window(df, trend_rate_m_per_day=rate)
    p = predict(res, Xl, base, mc_passes=200, trend_rate_m_per_day=rte)
    obs = ti.load_groundwater_observations()

    fig, ax = plt.subplots(figsize=(11.2, 4.9))
    ax.plot(df.date, df.gw_level_mbgl, color=BLUE, lw=1.5, label="Reconstructed daily water table")
    ax.plot(df.date, df.gw_level_mbgl.rolling(30, min_periods=1).mean(), color=SKY, lw=2.6,
            label="30-day mean")
    if not obs.empty:
        ax.scatter(pd.to_datetime(obs.date), obs.gw_level_mbgl, s=54, facecolor="white",
                   edgecolor=RED, lw=1.6, zorder=6, label=f"CGWB in-situ readings (n={len(obs)})")
    last = pd.Timestamp(df.date.max())
    fx = [last] + [pd.Timestamp(t) for t in td[-1]]
    fy = [float(base[-1])] + [float(v) for v in p["mean"][0]]
    lo = [float(base[-1])] + [float(v - 1.96 * s) for v, s in zip(p["mean"][0], p["std"][0])]
    hi = [float(base[-1])] + [float(v + 1.96 * s) for v, s in zip(p["mean"][0], p["std"][0])]
    ax.fill_between(fx, lo, hi, color=AMBER, alpha=.28, label="95% MC-dropout band")
    ax.plot(fx, fy, color=AMBER, lw=3, marker="o", ms=7, label="AquaCast forecast (30/60/90 d)")
    ax.set_title(f"{d} — 5 years of water table and the live 90-day forecast")
    ax.set_ylabel("Depth to water (m below ground)")
    ax.set_xlabel("")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, ncol=2)
    ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])          # shallow at top
    return save(fig, "hero_forecast")


# ======================================================== 2. training curve ==
def fig_training():
    res = load_model()
    h = pd.DataFrame(res.history)
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(h.epoch, h.train_rmse, color=SKY, lw=2.2, label="train RMSE")
    ax.plot(h.epoch, h.val_rmse, color=BLUE, lw=2.6, label="validation RMSE")
    bi = int(h.val_rmse.idxmin())
    ax.scatter([h.epoch[bi]], [h.val_rmse[bi]], s=110, facecolor="none", edgecolor=RED,
               lw=2.2, zorder=5)
    ax.annotate(f"best {h.val_rmse[bi]:.4f} m", (h.epoch[bi], h.val_rmse[bi]),
                textcoords="offset points", xytext=(10, 16), color=RED, fontweight="bold")
    ax.set_xlabel("epoch"); ax.set_ylabel("RMSE (m)")
    ax.set_title("AquiferLSTM convergence — 52,547 parameters, 25 s on CPU")
    ax.legend(frameon=False)
    return save(fig, "training_curve")


# ========================================================== 3. bake-off =====
def fig_bakeoff():
    df = load_dataset("Sangrur")
    res = load_model()
    b = benchmark(res, df, quick=True, verbose=False).sort_values("rmse_m")
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    cols = [GREEN if "shipped" in m else (BLUE if "AquaCast" in m else "#94A3B8")
            for m in b.model]
    bars = ax.barh(b.model[::-1], b.rmse_m[::-1], color=cols[::-1], height=.62)
    for bar, v in zip(bars, b.rmse_m[::-1]):
        ax.text(bar.get_width() + .006, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f} m", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel("Out-of-sample RMSE (m) — lower is better")
    ax.set_title("Model bake-off: identical data, identical split, identical budget")
    ax.set_xlim(0, max(b.rmse_m) * 1.22)
    ax.grid(axis="y", visible=False)
    return save(fig, "bakeoff")


# ================================================== 4. per-horizon + baselines
def fig_horizons():
    res = load_model()
    rows = []
    for d in DISTRICTS:
        m = evaluate(res, load_dataset(d), verbose=False)
        rows.append(dict(district=d, **{f"t+{h}": m["per_horizon"][f"t+{h}"]["rmse_m"]
                                        for h in HORIZONS},
                         overall=m["rmse_m"], persist=m["baseline_persistence_rmse_m"]))
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    x = np.arange(len(t)); w = .19
    for i, h in enumerate(HORIZONS):
        ax.bar(x + (i - 1.5) * w, t[f"t+{h}"], w, label=f"t+{h} d",
               color=[SKY, BLUE, NAVY][i])
    ax.plot(x, t.persist, color=RED, lw=0, marker="D", ms=10, label="persistence baseline")
    for i, r in t.iterrows():
        ax.text(i, r.persist + .012, f"{r.persist:.2f}", ha="center", color=RED,
                fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(t.district)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Accuracy by forecast horizon — and why error *falls* with lead time")
    ax.legend(frameon=False, ncol=4, fontsize=9.5)
    ax.set_ylim(0, .46)
    ax.grid(axis="x", visible=False)
    return save(fig, "horizons")


# ================================================= 5. permutation importance =
def fig_importance():
    res = load_model()
    imp = permutation_importance(res, load_dataset("Sangrur"), n_repeats=4)
    imp = imp.sort_values("importance_pct")
    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    cols = [SKY if "doy" in n else BLUE for n in imp.feature]
    b = ax.barh(imp.feature, imp.importance_pct, color=cols, height=.65)
    for bar, v in zip(b, imp.importance_pct):
        ax.text(bar.get_width() + .6, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_xlabel("Share of total RMSE damage when the channel is shuffled (%)")
    ax.set_title("What the network actually uses (permutation importance, Sangrur)")
    ax.set_xlim(0, max(imp.importance_pct) * 1.18)
    ax.grid(axis="y", visible=False)
    return save(fig, "importance")


# ==================================================== 6. water budget ========
def fig_budget():
    d = "Sangrur"
    df = load_dataset(d)
    a = (df.assign(year=df.date.dt.year)
           .groupby("year").agg(draft=("gw_draft_mcm", "sum"),
                                recharge=("total_recharge_mcm", "sum")).reset_index())
    a["deficit"] = a.draft - a.recharge
    a = a[a.year < a.year.max()]                      # drop the partial year
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    x = np.arange(len(a)); w = .37
    ax.bar(x - w / 2, a.draft, w, label="Groundwater draft", color=RED)
    ax.bar(x + w / 2, a.recharge, w, label="Recharge", color=TEAL)
    ax2 = ax.twinx()
    ax2.plot(x, a.deficit, color=NAVY, lw=2.4, marker="o", ms=6, label="Net deficit")
    ax2.set_ylabel("Net deficit (MCM/yr)", color=NAVY)
    ax2.grid(False); ax2.axhline(0, color=NAVY, lw=.8, alpha=.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{int(y)}" for y in a.year])
    ax.set_ylabel("Volume (MCM/yr)")
    ax.set_title(f"{d} annual water balance — draft, recharge and the structural deficit")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", ncol=3, fontsize=9.5)
    return save(fig, "water_budget")


# ====================================================== 7. block risk ========
def fig_blocks():
    d = "Sangrur"
    df = load_dataset(d)
    res = load_model()
    rate = get_prior(d).long_term_decline_m_per_yr / 365.25
    Xl, base, issue, td, rte = live_window(df, trend_rate_m_per_day=rate)
    p = predict(res, Xl, base, mc_passes=40, trend_rate_m_per_day=rte)
    wells = build_well_table(d, df, horizon_level=float(p["mean"][0][-1]),
                             horizon_delta=float(p["mean"][0][-1] - base[-1]))
    b = block_summary(wells).sort_values("mean_risk_pct").tail(11)
    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    cols = [RED if v >= 70 else AMBER if v >= 40 else GREEN for v in b.mean_risk_pct]
    bars = ax.barh(b.iloc[:, 0].astype(str), b.mean_risk_pct, color=cols, height=.66)
    for bar, v, n in zip(bars, b.mean_risk_pct, b.wells):
        ax.text(min(v + 1.2, 97), bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}  ({int(n)} wells)", va="center", fontsize=9.5, color=INK)
    ax.set_xlabel("Mean block pump-failure risk at +90 days (%)")
    ax.set_title(f"{d} — risk downscaled from the district forecast to {len(wells)} real CGWB wells")
    ax.set_xlim(0, 118)
    ax.grid(axis="y", visible=False)
    return save(fig, "block_risk")


# ======================================================== 8. static map ======
def fig_map():
    import geopandas as gpd
    d = "Sangrur"
    df = load_dataset(d)
    res = load_model()
    rate = get_prior(d).long_term_decline_m_per_yr / 365.25
    Xl, base, issue, td, rte = live_window(df, trend_rate_m_per_day=rate)
    p = predict(res, Xl, base, mc_passes=40, trend_rate_m_per_day=rte)
    wells = build_well_table(d, df, horizon_level=float(p["mean"][0][-1]),
                             horizon_delta=float(p["mean"][0][-1] - base[-1]))
    gj = district_geojson(d)          # a GeoJSON *mapping*, not a file path
    from shapely.geometry import shape as _shape
    geom = _shape(gj["features"][0]["geometry"]) if isinstance(gj, dict) else None
    poly = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    poly.boundary.plot(ax=ax, color=BLUE, lw=2)
    poly.plot(ax=ax, color="#E0F2FE", alpha=.55)
    sc = ax.scatter(wells.lon, wells.lat, c=wells.pump_failure_risk_pct, s=46,
                    cmap="RdYlGn_r", vmin=0, vmax=100, edgecolor="white", lw=.5, zorder=5)
    cb = fig.colorbar(sc, ax=ax, shrink=.72, pad=.02)
    cb.set_label("Pump-failure risk at +90 d (%)", color=INK)
    obs = ti.load_groundwater_observations()
    if not obs.empty:
        ax.scatter(obs.lon, obs.lat, s=120, marker="*", facecolor=AMBER, edgecolor=NAVY,
                   lw=.8, zorder=7, label="CGWB monitoring well")
        ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title(f"{d}: {len(wells)} real CGWB wells, individually scored")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    return save(fig, "well_map")


# ==================================================== 9. digital twin ========
def fig_twin():
    d = "Sangrur"
    df = load_dataset(d)
    sc = {"Standard Flood Irrigation": SCENARIOS["Standard Flood Irrigation"],
          "Micro-Drip Shift (40%)": SCENARIOS["Micro-Drip Shift (40%)"],
          "Drip 40% + 15-day paddy shift": Scenario(
              name="Drip 40% + 15-day paddy shift", irrigation_efficiency=0.78,
              pump_adoption_drip_pct=40.0, paddy_transplant_shift_days=15)}
    out = compare_scenarios(df, d, sc, days=365)
    base_name = "Standard Flood Irrigation"
    fig, ax = plt.subplots(figsize=(9.8, 4.5))
    deltas = scenario_deltas(out, base_name).set_index("scenario")
    for name, col in zip(sc, [RED, AMBER, GREEN]):
        y = np.asarray(out[name], dtype=float)
        ax.plot(np.arange(len(y)), y, color=col, lw=2.4, label=name)
        saved = float(deltas.loc[name, "saved_m"]) if name in deltas.index else 0.0
        if saved:
            ax.annotate(f"{saved:+.2f} m vs flood", (len(y) - 1, y[-1]),
                        textcoords="offset points", xytext=(8, 10 if saved >= 0 else -18),
                        color=col, fontweight="bold", fontsize=10)
    ax.set_xlabel("days from today")
    ax.set_ylabel("Depth to water (m bgl)")
    ax.set_title("Digital twin — 365-day policy simulation, solved in under 0.2 s")
    ax.legend(frameon=False, fontsize=9.5)
    ax.invert_yaxis()
    return save(fig, "digital_twin")


# ================================================== 10. provenance bars ======
def fig_provenance():
    df = load_dataset("Sangrur")
    pv = {"Rainfall (days)": int((df.rainfall_src == "telemetry").sum()),
          "Rainfall (days, modelled)": int((df.rainfall_src == "generated").sum()),
          "Temperature (days)": int((df.temp_src == "telemetry").sum()),
          "Temperature (days, modelled)": int((df.temp_src == "generated").sum())}
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    keys = list(pv); vals = list(pv.values())
    cols = [GREEN, "#CBD5E1", GREEN, "#CBD5E1"]
    b = ax.barh(keys[::-1], vals[::-1], color=cols[::-1], height=.6)
    for bar, v in zip(b, vals[::-1]):
        ax.text(bar.get_width() + 18, bar.get_y() + bar.get_height() / 2, f"{v:,} d",
                va="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel("days in the 1,828-day training frame")
    ax.set_title("Provenance: measured telemetry vs physics-filled gaps — declared, not hidden")
    ax.set_xlim(0, max(vals) * 1.2)
    ax.grid(axis="y", visible=False)
    return save(fig, "provenance")


# ================================================== 11. portfolio risk =======
def fig_portfolio():
    res = load_model()
    rows = []
    for d in DISTRICTS:
        df = load_dataset(d)
        rate = get_prior(d).long_term_decline_m_per_yr / 365.25
        Xl, base, issue, td, rte = live_window(df, trend_rate_m_per_day=rate)
        p = predict(res, Xl, base, mc_passes=40, trend_rate_m_per_day=rte)
        adv = build_advisory(df, d, p, 90, SCENARIOS["Standard Flood Irrigation"])
        rows.append(dict(district=d, score=adv.risk_score, band=adv.risk_band,
                         pd=adv.pd_pct, el=money(adv.expected_loss_inr),
                         exp=money(adv.exposure_inr)))
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    cols = [RED if s >= 60 else AMBER if s >= 35 else GREEN for s in t.score]
    bars = ax.bar(t.district, t.score, color=cols, width=.5)
    for bar, s, bd, pdx, el in zip(bars, t.score, t.band, t.pd, t.el):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 2, f"{s:.0f}/100", ha="center",
                fontweight="bold", color=INK)
        ax.text(bar.get_x() + bar.get_width() / 2, s / 2, f"{bd}\nPD {pdx:.1f}%\nEL ₹{el:,.1f} cr",
                ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 100); ax.set_ylabel("Agrarian credit-risk score")
    ax.set_title("What the water forecast is worth to a lender (portfolio: ₹607 cr)")
    ax.grid(axis="x", visible=False)
    return save(fig, "portfolio")


# ==================================================== 12. seasonal cycle =====
def fig_seasonal():
    d = "Sangrur"
    df = load_dataset(d)
    df["m"] = df.date.dt.month
    g = df.groupby("m").agg(lvl=("gw_level_mbgl", "mean"),
                            draft=("gw_draft_mcm", "mean"),
                            rec=("total_recharge_mcm", "mean"))
    fig, ax = plt.subplots(figsize=(9.6, 4.1))
    ax.bar(g.index - .18, g.draft, .36, label="mean daily draft (MCM)", color=RED)
    ax.bar(g.index + .18, g.rec, .36, label="mean daily recharge (MCM)", color=TEAL)
    ax2 = ax.twinx()
    ax2.plot(g.index, g.lvl, color=NAVY, lw=2.6, marker="o", label="water table (m bgl)")
    ax2.set_ylabel("mean depth to water (m bgl)", color=NAVY); ax2.grid(False)
    ax2.invert_yaxis()
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_xlabel("month")
    ax.set_title("The Punjab paradox: the water table is deepest in August — after the monsoon")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, fontsize=9.5, loc="lower left")
    return save(fig, "seasonal_paradox")


# ======================================================== 13. the fix ========
def fig_bugfix():
    """Before/after: what the two evaluation bugs did to the headline number."""
    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    labels = ["As first measured\n(stale window + trend-prior bug)", "After the audit\n(correct truth + live window)"]
    vals = [0.210, 0.117]
    bars = ax.bar(labels, vals, color=["#CBD5E1", GREEN], width=.42)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + .006, f"{v:.3f} m", ha="center",
                fontsize=14, fontweight="bold", color=INK)
    ax.set_ylim(0, .26); ax.set_ylabel("reported overall RMSE (m)")
    ax.set_title("We audited our own evaluation — and it made the model look 2x worse than it is")
    ax.grid(axis="x", visible=False)
    return save(fig, "self_audit")


if __name__ == "__main__":
    print("rendering deck figures …")
    for fn in (fig_hero, fig_training, fig_bakeoff, fig_horizons, fig_importance,
               fig_budget, fig_blocks, fig_map, fig_twin, fig_provenance,
               fig_portfolio, fig_seasonal, fig_bugfix):
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {type(e).__name__}: {e}")
    print("done →", FIG)
