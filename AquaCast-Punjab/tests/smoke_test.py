"""Headless smoke test — exercises every code path the Streamlit app touches."""
import sys, traceback, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd

from src.config import (DISTRICTS, SCENARIOS, Scenario, draft_multiplier,
                       FLOOD_IRRIGATION_EFFICIENCY)
from src.spatial_loader import get_prior
from src.dataset_generator import load_dataset
from src.model_lstm import (HORIZONS, load_model, build_sequences, live_window,
                            chronological_split, predict, evaluate,
                            permutation_importance, input_saliency, benchmark)
from src.advisory_engine import (build_advisory, crop_stage_for,
                                 effective_pump_discharge_m3h, irrigation_schedule)
from src.digital_twin import compare_scenarios, scenario_deltas
from src.well_network import build_well_table, block_summary
from src.spatial_loader import spatial_summary, district_geojson
from src import telemetry_ingest as ti, viz

ok = fail = 0
def run(name, fn):
    global ok, fail
    t = time.time()
    try:
        fn(); ok += 1
        print(f"  PASS  {name:52s} {time.time()-t:6.2f}s")
    except Exception as e:
        fail += 1
        print(f"  FAIL  {name:52s} {type(e).__name__}: {e}")
        traceback.print_exc()

print("\n=== AquaCast-Punjab smoke test ===\n")
res = load_model(); assert res is not None, "model not trained"
print(f"  model loaded (best val RMSE {res.best_val:.4f} m)\n")

for d in DISTRICTS:
    df = load_dataset(d); df["date"] = pd.to_datetime(df.date)
    rate = get_prior(d).long_term_decline_m_per_yr / 365.25
    X, Y, base, issue, td, rte = build_sequences(df, trend_rate_m_per_day=rate)
    fc = predict(res, X[-1:], base[-1:], mc_passes=30, trend_rate_m_per_day=rate)
    def _a(d=d, df=df, fc=fc):
        adv = build_advisory(df, d, fc, 90, SCENARIOS["Micro-Drip Shift (40%)"],
                             pump_set_depth_mbgl=49.0, farm_acres=5.0)
        assert 0 < adv.risk_score <= 100, adv.risk_score
        assert adv.quota_hours_per_week_per_acre > 0
    run(f"[{d}] advisory engine (90d, drip)", _a)
    def _w(d=d, df=df, fc=fc):
        w = build_well_table(d, df, horizon_level=float(np.ravel(fc["mean"])[2]))
        assert w.pump_failure_risk_pct.between(0, 100).all()
        block_summary(w)
    run(f"[{d}] well network + blocks", _w)
    def _t(d=d, df=df):
        return scenario_deltas(compare_scenarios(
            df, d, {**SCENARIOS, "custom": Scenario(pump_adoption_drip_pct=60.0,
                    monsoon_anomaly_pct=-20, canal_availability_pct=80)},
            days=365, paddy_shift_days=15), "Standard Flood Irrigation")
    run(f"[{d}] digital twin (3 scenarios)", _t)
    run(f"[{d}] spatial summary", lambda d=d: spatial_summary(d))
    run(f"[{d}] district geojson", lambda d=d: district_geojson(d))

df = load_dataset("Sangrur")
def live_window_is_today():
    """The operational forecast must be issued from the newest observation."""
    Xl, base, issue, td, rte = live_window(df, trend_rate_m_per_day=0.8/365.25)
    assert Xl.shape[0] == 1, Xl.shape
    assert pd.Timestamp(issue[-1]) == pd.Timestamp(df.date.max()), (
        f"forecast issued {pd.Timestamp(issue[-1]).date()}, "
        f"not {pd.Timestamp(df.date.max()).date()}")
    assert abs(float(base[-1]) - float(df.gw_level_mbgl.iloc[-1])) < 1e-4
    for h, t in zip((30, 60, 90), td[-1]):
        assert pd.Timestamp(t) == pd.Timestamp(df.date.max()) + pd.Timedelta(days=int(h))
    # and it must agree with the trained network: Oct -> Dec is a monsoon recovery
    p = predict(res, Xl, base, trend_rate_m_per_day=rte)
    d90 = float(p["mean"][0, -1] - base[-1])
    clim = (df.assign(m=df.date.dt.month)
              .groupby("m").gw_level_mbgl.mean())
    observed_oct_dec = float(clim.loc[12] - clim.loc[10])
    assert d90 < 0, f"Oct->Dec should recover, got {d90:+.2f} m"
    assert abs(d90 - observed_oct_dec) < 0.30, (
        f"live 90-day delta {d90:+.2f} m vs climatological {observed_oct_dec:+.2f} m")


run("live forecast window ends on the newest datum", live_window_is_today)


def rabi_wraps_the_year():
    """21 Nov - 04 Jan is wheat, not 'Fallow' (regression: doy-based staging)."""
    got = {str(pd.Timestamp(d).date()): crop_stage_for(pd.Timestamp(d))["stage"]
           for d in ("2026-11-05", "2026-11-25", "2026-12-31", "2027-01-20",
                     "2027-02-20", "2027-04-10", "2026-07-15")}
    assert got["2026-12-31"] == "Tillering", got
    assert got["2026-11-05"] == "CRI", got
    assert got["2026-11-25"] == "Tillering", got
    assert got["2027-01-20"] == "Jointing", got
    assert got["2027-02-20"] == "Flowering", got
    assert got["2027-04-10"] == "Grain fill", got
    assert got["2026-07-15"] == "Fallow", got


run("Rabi phenology wraps the year end", rabi_wraps_the_year)


def pump_hydraulics():
    """Discharge must be derated to the head the pump actually works against."""
    prior = get_prior("Sangrur")
    shallow = effective_pump_discharge_m3h(prior, 10.0)
    deep = effective_pump_discharge_m3h(prior, 45.0)
    assert 20.0 <= deep < shallow <= prior.avg_pump_discharge_m3h, (shallow, deep)
    h_ev, ev_wk, gap = irrigation_schedule(2.0, deep)
    assert h_ev > 0 and ev_wk > 0 and 1.0 < gap < 200.0, (h_ev, ev_wk, gap)


run("pump discharge derated by head", pump_hydraulics)


def scenario_lever_is_consistent():
    """The drip lever must move the quota the same way it moves the water balance."""
    assert abs(draft_multiplier(SCENARIOS["Standard Flood Irrigation"]) - 1.0) < 1e-9
    drip = draft_multiplier(SCENARIOS["Micro-Drip Shift (40%)"])
    assert 0.5 < drip < 1.0, drip
    # higher application efficiency => strictly less water pumped
    a = Scenario(name="a", irrigation_efficiency=0.60, pump_adoption_drip_pct=40.0)
    b = Scenario(name="b", irrigation_efficiency=0.90, pump_adoption_drip_pct=40.0)
    assert draft_multiplier(b) < draft_multiplier(a)
    # and it must actually move the advisory quota
    from src.model_lstm import live_window
    Xl, base, issue, td, rte = live_window(df, trend_rate_m_per_day=0.8/365.25)
    fc = predict(res, Xl, base, mc_passes=10, trend_rate_m_per_day=rte)
    for scen_name in ("Standard Flood Irrigation", "Micro-Drip Shift (40%)"):
        adv = build_advisory(df, "Sangrur", fc, 90, SCENARIOS[scen_name])
        if scen_name.startswith("Standard"):
            flood = adv.quota_hours_per_week_per_acre
        else:
            assert adv.quota_hours_per_week_per_acre < flood, "drip did not cut the quota"
            assert adv.water_saved_pct > 0


run("scenario lever is consistent end-to-end", scenario_lever_is_consistent)

def trend_prior_reconciles():
    """evaluate()'s truth must include the drift predict() re-adds (rate x horizon)."""
    m = evaluate(res, df, verbose=False)
    _, te = chronological_split(df)
    r = get_prior("Sangrur").long_term_decline_m_per_yr / 365.25
    X, Y, b, i, t, rte = build_sequences(te, trend_rate_m_per_day=r)
    Yh = predict(res, X, b, trend_rate_m_per_day=rte)["mean"]
    Yt = b[:, None] + Y + rte[:, None] * np.asarray(HORIZONS, dtype="float32")[None, :]
    ref = float(np.sqrt(((Yh - Yt) ** 2).mean()))
    assert abs(ref - m["rmse_m"]) < 1e-3, f"evaluate {m['rmse_m']:.4f} vs manual {ref:.4f}"
    # a residual +rate*h bias would show up as a bias comparable to the trend term
    assert abs(m["bias_m"]) < 0.10, f"bias {m['bias_m']:+.3f} m — trend prior mismatch?"
    assert m["rmse_m"] < 0.16, m["rmse_m"]


run("trend prior reconciles in evaluate()", trend_prior_reconciles)


run("model evaluation (Sangrur)", lambda: evaluate(res, df, verbose=False))
run("permutation importance", lambda: permutation_importance(res, df, n_repeats=2))
run("input saliency", lambda: input_saliency(res, df))

def charts():
    seq = build_sequences(df, trend_rate_m_per_day=0.8/365.25)
    fc = predict(res, seq[0][-1:], seq[2][-1:], mc_passes=20,
                 trend_rate_m_per_day=0.8/365.25)
    viz.forecast_chart(df, issue_date=pd.Timestamp(df.date.max()),
                       fc_dates=[pd.Timestamp(df.date.max())+pd.Timedelta(days=90)],
                       fc_mean=[1.0], fc_std=[0.2],
                       observed=ti.load_groundwater_observations())
    annual = df.assign(year=df.date.dt.year).groupby("year").agg(
        draft_mcm=("gw_draft_mcm","sum"), recharge_mcm=("total_recharge_mcm","sum")).reset_index()
    annual["deficit_mcm"] = annual.draft_mcm - annual.recharge_mcm
    viz.water_budget_chart(annual); viz.seasonal_profile(df)
    viz.history_chart(res.history); viz.risk_gauge(55, "#F59E0B")
    viz.risk_components({"depth":50,"pump":40,"drawdown":30,"stage":20,"crop":60,
                         "monsoon":50,"adapt":10},
                        {"depth":.26,"pump":.22,"drawdown":.16,"stage":.14,
                         "crop":.12,"monsoon":.10,"adapt":-.06})
    viz.importance_chart(permutation_importance(res, df, n_repeats=1))
    viz.saliency_chart(input_saliency(res, df))
    viz.benchmark_bar(benchmark(res, df, quick=True, verbose=False))
run("all plotly figures", charts)

rf, tb = ti.load_all_rainfall(), ti.load_all_temperature()
print(f"\n  provenance: {len(rf.meta)} rain gauges | {len(tb.meta)} temp stations | "
      f"{len(ti.load_groundwater_observations())} CGWB readings")
print(f"\n=== {ok} passed, {fail} failed ===\n")
sys.exit(1 if fail else 0)
