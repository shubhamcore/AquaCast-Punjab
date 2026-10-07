#!/usr/bin/env python3
"""
AquaCast-Punjab :: single entrypoint
====================================

    python run_pipeline.py                 # full pipeline, then launch Streamlit
    python run_pipeline.py --no-app        # pipeline only
    python run_pipeline.py --rebuild-spatial   # re-clip the India-WRIS assets
    python run_pipeline.py --retrain       # force a fresh LSTM training run
    python run_pipeline.py --epochs 60 --quick

Stages
------
0. pre-flight   — verify every asset under ``data/`` is present
1. Module 1     — spatial ingestion & Punjab crop baseline
2. Module 2a    — real India-WRIS / CGWB telemetry ingestion + QC
3. Module 2b    — hybrid time-series synthesis (physics + measured + gap-filled)
4. Module 3     — PyTorch LSTM training, evaluation, benchmark, explainability
5. Module 4     — advisory & agrarian-risk engine smoke run
6. launch       — ``streamlit run app.py``
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BANNER = r"""
    _                        ____           _       ____                       _
   / \   __ _  ___ _   _    / ___|__ _  ___| | __  |  _ \ _   _ _ __  _ __   ___ | |
  / _ \ / _` |/ __| | | |  | |   / _` |/ __| |/ /  | |_) | | | | '_ \| '_ \ / _ \| |
 / ___ \ (_| | (__| |_| |  | |__| (_| | (__|   <   |  __/| |_| | |_) | |_) | (_) | |
/_/   \_\__,_|\___|\__,_|   \____\__,_|\___|_|\_\  |_|    \__,_| .__/| .__/ \___/|_|
                                                               |_|   |_|
            AI-Powered Aquifer Depletion & Irrigation Advisory System
            Sankalp · Climate-Smart Agriculture · Water · Rural Credit Risk
"""


def hr(title: str = "") -> None:
    print("\n" + "=" * 92)
    if title:
        print(f"  {title}")
        print("=" * 92)


def stage(n: int, title: str) -> None:
    print(f"\n\033[1;36m[{n}] {title}\033[0m")


# ======================================================================================
def preflight(rebuild_spatial: bool) -> bool:
    stage(0, "Pre-flight — verifying assets")
    from src.config import DATA_DIR, GEO_LAYERS, RASTERS, TELEMETRY_DIR

    missing = [str(p) for p in GEO_LAYERS.values() if not Path(p).exists()]
    missing += [str(p) for p in RASTERS.values() if not Path(p).exists()]
    if missing or rebuild_spatial:
        print("  rebuilding Punjab spatial clips from the India-WRIS source files…")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_spatial_assets.py")],
                           cwd=ROOT)
        if r.returncode != 0:
            print("  ! spatial build failed — is data_raw/ present with the WRIS downloads?")
            if missing:
                print("  ! missing:", *missing, sep="\n      ")
                return False
    ok = True
    for k, p in GEO_LAYERS.items():
        print(f"  {'✓' if Path(p).exists() else '✗'} {k:24s} {Path(p).name}")
        ok &= Path(p).exists()
    for k, p in RASTERS.items():
        e = Path(p).exists()
        print(f"  {'✓' if e else '✗'} raster:{k:17s} {Path(p).name}")
    telem = sorted(TELEMETRY_DIR.glob("*.xlsx")) + sorted(TELEMETRY_DIR.glob("*.csv"))
    print(f"  ✓ raw_telemetry              {len(telem)} India-WRIS / CGWB exports")
    return ok


def module1():
    stage(1, "Module 1 — Spatial ingestion & Punjab crop baseline")
    from src.spatial_loader import self_check
    self_check("Sangrur")


def module2a():
    stage(2, "Module 2a — Real telemetry ingestion (India-WRIS / CGWB)")
    from src.telemetry_ingest import telemetry_self_check
    telemetry_self_check()


def module2b(districts, force=False):
    stage(3, "Module 2b — Hybrid time-series synthesis")
    from src.dataset_generator import PROCESSED_DIR, build_dataset, save_dataset
    metas = {}
    for d in districts:
        p = PROCESSED_DIR / f"processed_{d.lower()}_timeseries.csv"
        if p.exists() and not force:
            print(f"  {d:10s} cached ({p.name}) — use --force-data to rebuild")
            continue
        df, meta = build_dataset(d)
        save_dataset(df, meta, d)
        metas[d] = meta
    return metas


def module3(districts, epochs, quick, retrain):
    stage(4, "Module 3 — PyTorch LSTM")
    from src.config import MODEL_DIR
    from src.dataset_generator import load_dataset
    from src.model_lstm import (benchmark, evaluate, load_model, permutation_importance,
                                save_model, train_model)
    from src.spatial_loader import get_prior

    res = None if retrain else load_model()
    if res is None:
        dfs = [load_dataset(d) for d in districts]
        rates = [get_prior(d).long_term_decline_m_per_yr / 365.25 for d in districts]
        res = train_model(dfs, rates, epochs=epochs, quick=quick)
        save_model(res)
    else:
        print(f"  loaded trained weights from {MODEL_DIR/'lstm_aquifer.pth'}")
        print(f"  (best validation RMSE {res.best_val:.4f} m, {len(res.history)} epochs)")

    metrics = {}
    for d in districts:
        m = evaluate(res, load_dataset(d))
        metrics[d] = m
    print("\n  Permutation feature importance (Sangrur):")
    print(permutation_importance(res, load_dataset(districts[0]), n_repeats=3)
          .to_string(index=False))
    print()
    benchmark(res, load_dataset(districts[0]), quick=quick)
    return res, metrics


def module4(districts):
    stage(5, "Module 4 — Advisory & agrarian-risk engine")
    from src.config import HORIZONS, SCENARIOS, get_prior
    from src.advisory_engine import build_advisory
    from src.dataset_generator import load_dataset
    from src.model_lstm import live_window, predict
    from src.risk_engine import portfolio_view

    res_metrics = {}
    for d in districts:
        df = load_dataset(d)
        rate = get_prior(d).long_term_decline_m_per_yr / 365.25
        from src.model_lstm import load_model
        mdl = load_model()
        # live_window: the forecast window ends on the newest observation, so the
        # advisory is issued for today rather than for (today - 90 days).
        Xl, base, issue, td, rte = live_window(df, trend_rate_m_per_day=rate)
        fc = predict(mdl, Xl, base, mc_passes=40, trend_rate_m_per_day=rte)
        adv = build_advisory(df, d, fc, HORIZONS[-1], SCENARIOS["Standard Flood Irrigation"])
        res_metrics[d] = dict(risk_score=adv.risk_score, band_en=adv.risk_band,
                              pd_pct=adv.pd_pct, exposure_inr=adv.exposure_inr,
                              expected_loss_inr=adv.expected_loss_inr,
                              loss_avoidable_inr=adv.expected_loss_inr * 0.22)
        print(f"\n  ── {d} ── as of {adv.issue_date}, +{adv.horizon_days} days → {adv.target_date}")
        print(f"     water table   {adv.current_level_mbgl:.2f} → {adv.predicted_level_mbgl:.2f} m bgl "
              f"({adv.predicted_delta_m:+.2f} m)   zone: {adv.zone_label_en}")
        print(f"     pump quota    {adv.quota_hours_per_week_per_acre:.1f} h/week/acre "
              f"(sustainability ×{adv.sustainability_factor:.2f})")
        print(f"     pump risk     {adv.pump_failure_risk_pct:.0f}%  "
              f"(headroom {adv.headroom_m:.1f} m)")
        print(f"     credit risk   {adv.risk_score:.0f}/100 ({adv.risk_band}) · "
              f"PD {adv.pd_pct:.2f}% · expected loss ₹{adv.expected_loss_inr/1e7:,.1f} cr")
    pv = portfolio_view(res_metrics)
    if pv:
        print(f"\n  PORTFOLIO: exposure ₹{pv['exposure_inr']/1e7:,.0f} cr · "
              f"PAR {pv['par_pct']:.2f}% · expected loss ₹{pv['expected_loss_inr']/1e7:,.1f} cr · "
              f"avoidable ₹{pv['loss_avoidable_inr']/1e7:,.1f} cr")
    return res_metrics


def launch(port: int = 8501):
    stage(6, f"Launching Streamlit dashboard on port {port}")
    print("  → open the Live Preview, or http://localhost:%d\n" % port)
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                    "--server.address", "0.0.0.0", "--server.port", str(port),
                    "--server.headless", "true"], cwd=ROOT)


# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="AquaCast-Punjab pipeline")
    ap.add_argument("--districts", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="6-epoch smoke run")
    ap.add_argument("--retrain", action="store_true", help="force retraining")
    ap.add_argument("--force-data", action="store_true", help="rebuild the time-series")
    ap.add_argument("--rebuild-spatial", action="store_true")
    ap.add_argument("--no-app", action="store_true")
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args()

    t0 = time.time()
    print(BANNER)
    from src.config import DISTRICTS, EPOCHS
    districts = args.districts or DISTRICTS
    epochs = args.epochs or EPOCHS

    if not preflight(args.rebuild_spatial):
        print("\n  ! pre-flight failed — see the messages above.")
        return 1
    module1()
    module2a()
    module2b(districts, force=args.force_data)
    module3(districts, epochs, args.quick, args.retrain)
    module4(districts)

    hr("PIPELINE COMPLETE")
    print(f"  finished in {time.time()-t0:.1f}s")
    print("  artefacts:")
    for p in ["data/processed/processed_sangrur_timeseries.csv",
              "data/processed/climate_model.json",
              "models/lstm_aquifer.pth", "models/lstm_aquifer.json"]:
        print(f"    {'✓' if Path(ROOT/p).exists() else '✗'} {p}")
    hr()
    if not args.no_app:
        launch(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
