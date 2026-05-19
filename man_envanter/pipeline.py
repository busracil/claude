"""
Ana Pipeline (Orchestrator)
============================
ADIM 1 -> 6'yı uçtan uca çalıştırır:
  1. Veri yükleme
  3. ML talep tahmini (4 model + Optuna)
  4. 7 yöntem karşılaştırma -> şampiyon seçimi
  5. (r,Q) optimizasyonu (Grid + Optuna + SimPy)  — joblib ile paralel
  6. EOQ klasik politika karşılaştırması -> tasarruf
Tüm çıktılar outputs/ klasörüne yazılır.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
import data_loader as dl
from forecasting.champion import run_forecasting
from optimization.optimizer import optimize_part
from optimization.eoq import evaluate_eoq, savings
from optimization.sim_engine import monthly_to_daily


def _process_part(part, monthly_fc, opt_row, hist_monthly):
    """Tek parça için ADIM 5 + ADIM 6 — paralel çalıştırılır."""
    try:
        opt = optimize_part(part, monthly_fc, opt_row)
        daily = monthly_to_daily(monthly_fc)
        eoq = evaluate_eoq(part, hist_monthly, opt_row, daily)
        abs_save, pct_save = savings(opt["ml_total_cost"], eoq["eoq_total_cost"])

        cost_map = opt.pop("cost_map")
        merged = {**opt, **{k: v for k, v in eoq.items() if k != C.COL_PART}}
        merged["abs_saving"] = abs_save
        merged["pct_saving"] = pct_save
        return merged, cost_map
    except Exception as exc:  # parça atlanır, pipeline durmaz
        return {C.COL_PART: part, "error": str(exc)}, []


def run_pipeline(max_parts=C.MAX_PARTS, ml_trials=C.ML_OPTUNA_TRIALS):
    start = time.time()
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    print("=" * 64)
    print("  MAN TÜRKİYE — ML + SİMÜLASYON ENVANTER OPTİMİZASYONU")
    print("=" * 64)

    # ── ADIM 1: Veri ────────────────────────────────────────────
    print("\n[ADIM 1] Veri yükleniyor...")
    data = dl.load_all()
    feature_cols = dl.feature_columns(data["ml"])
    parts = dl.get_parts(data, max_parts)
    master = dl.build_part_master(data)
    print(f"  {len(parts)} parça, {len(feature_cols)} feature.")

    # ── ADIM 3-4: Tahmin + Şampiyon ─────────────────────────────
    print("\n[ADIM 3-4] Talep tahmini ve şampiyon seçimi...")
    forecasts_long, champions, method_metrics, report = run_forecasting(
        data, feature_cols, parts, ml_trials)

    # Şampiyon aylık tahmin serileri
    champ_map = dict(zip(champions[C.COL_PART], champions["champion"]))
    champ_fc = {}
    for part, sub in forecasts_long.groupby(C.COL_PART):
        cm = champ_map.get(part)
        rows = sub[sub["method"] == cm].sort_values(C.COL_MONTH_IDX)
        champ_fc[part] = rows["pred"].values

    # Tarihsel aylık seri (Train) ve optimizasyon parametreleri
    train_df = data["ml"][data["ml"][C.COL_SPLIT] == C.TRAIN_LABEL]
    hist_map = {p: g.sort_values(C.COL_MONTH_IDX)[C.COL_TARGET].values
                for p, g in train_df.groupby(C.COL_PART)}
    opt_rows = {r[C.COL_PART]: r.to_dict() for _, r in master.iterrows()}

    # ── ADIM 5-6: Optimizasyon (paralel) ────────────────────────
    print(f"\n[ADIM 5-6] {len(parts)} parça için (r,Q) optimizasyonu + EOQ "
          f"karşılaştırması (paralel)...")
    jobs = [
        delayed(_process_part)(p, champ_fc.get(p, np.zeros(C.TEST_HORIZON)),
                               opt_rows[p], hist_map.get(p, np.array([])))
        for p in parts
    ]
    results = Parallel(n_jobs=C.N_JOBS, verbose=5)(jobs)

    opt_records = [r[0] for r in results]
    cost_map_rows = []
    for (rec, cmap) in results:
        for c in cmap:
            cost_map_rows.append({C.COL_PART: rec[C.COL_PART], **c})

    df_opt = pd.DataFrame(opt_records)
    df_opt = df_opt.merge(
        master[[C.COL_PART, "ABC", "XYZ", "Segment", "Model_Grubu"]],
        on=C.COL_PART, how="left")
    df_opt = df_opt.merge(
        champions[[C.COL_PART, "champion", "WAPE", "sMAPE"]],
        on=C.COL_PART, how="left")

    # ── Çıktıları yaz ───────────────────────────────────────────
    print("\n[ÇIKTI] Sonuçlar yazılıyor...")
    forecasts_long.to_parquet(C.FORECASTS_PATH, index=False)
    champions.to_csv(C.CHAMPIONS_PATH, index=False, encoding="utf-8-sig")
    df_opt.to_csv(C.OPTIMIZATION_PATH, index=False, encoding="utf-8-sig")
    method_metrics.to_parquet(
        os.path.join(C.OUTPUT_DIR, "method_metrics.parquet"), index=False)
    pd.DataFrame(cost_map_rows).to_parquet(
        os.path.join(C.OUTPUT_DIR, "cost_maps.parquet"), index=False)

    _write_dashboard(df_opt, champions, report, master)

    with open(C.MODEL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    elapsed = time.time() - start
    _print_summary(df_opt, elapsed)
    return df_opt


def _write_dashboard(df_opt, champions, report, master):
    """Streamlit arayüzü için özet dashboard.json üretir."""
    valid = df_opt[df_opt.get("ml_total_cost").notna()] if "ml_total_cost" in df_opt else df_opt

    summary = {
        "total_parts": int(len(df_opt)),
        "optimized_parts": int(len(valid)),
        "total_eoq_cost": float(valid["eoq_total_cost"].sum()),
        "total_ml_cost": float(valid["ml_total_cost"].sum()),
        "total_saving": float(valid["abs_saving"].sum()),
        "avg_pct_saving": float(valid["pct_saving"].mean()),
        "avg_ml_service": float(valid["ml_service_level"].mean()),
        "avg_eoq_service": float(valid["eoq_service_level"].mean()),
    }
    dashboard = {
        "summary": summary,
        "champion_distribution": champions["champion"].value_counts().to_dict(),
        "segment_distribution": master["Segment"].value_counts().to_dict(),
        "model_group_distribution": master["Model_Grubu"].value_counts().to_dict(),
        "method_overall_metrics": report["overall"],
        "tuning_info": {k: v["val_wape"] for k, v in report["tuning"].items()},
    }
    with open(C.DASHBOARD_PATH, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False, default=str)


def _print_summary(df_opt, elapsed):
    print("\n" + "=" * 64)
    print("  SONUÇ ÖZETİ")
    print("=" * 64)
    valid = df_opt[df_opt["ml_total_cost"].notna()] if "ml_total_cost" in df_opt else df_opt
    if len(valid):
        print(f"  Optimize edilen parça : {len(valid)}")
        print(f"  Ort. ML servis düzeyi : {valid['ml_service_level'].mean():.1%}")
        print(f"  Ort. EOQ servis düzeyi: {valid['eoq_service_level'].mean():.1%}")
        print(f"  Ort. maliyet tasarrufu: {valid['pct_saving'].mean():.1f}%")
        print(f"  Toplam tasarruf       : {valid['abs_saving'].sum():,.0f} TL")
    if "error" in df_opt.columns:
        n_err = df_opt["error"].notna().sum()
        if n_err:
            print(f"  Atlanan parça (hata)  : {n_err}")
    print(f"  Çalışma süresi        : {elapsed:.1f} sn")
    print("=" * 64)


if __name__ == "__main__":
    run_pipeline()
