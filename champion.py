"""
ADIM 4 — Yöntem Karşılaştırması ve Şampiyon Seçimi
===================================================
Her parça için 7 yöntem (4 ML + 3 klasik) test ufkunda karşılaştırılır.
Dört metrik (MAE, RMSE, WAPE, sMAPE) hesaplanır; her metrikte en iyi yönteme
1 oy verilir. En çok oyu alan yöntem o parçanın ŞAMPİYONU olur (eşitlik WAPE
ile bozulur). Şampiyonun test tahminleri ADIM 5 optimizasyonuna girdidir.
"""

from collections import Counter
import numpy as np
import pandas as pd

import config as C
from data_loader import split_train_test
from forecasting.ml_models import train_all_models
from forecasting.classical import classical_forecasts
from forecasting.metrics import all_metrics


def _build_forecast_table(data, feature_cols, parts, ml_trials):
    """Tüm yöntemlerin test tahminlerini uzun-format tabloda toplar."""
    ml_df = data["ml"][data["ml"][C.COL_PART].isin(parts)].copy()
    train_df, test_df = split_train_test(ml_df)

    print("ADIM 3 — ML modelleri eğitiliyor...")
    ml_preds, tuning_info = train_all_models(train_df, test_df, feature_cols, ml_trials)

    test_base = (test_df[[C.COL_PART, C.COL_MONTH_IDX, C.COL_DATE, C.COL_TARGET_RAW]]
                 .rename(columns={C.COL_TARGET_RAW: "actual"})
                 .reset_index(drop=True))

    frames = []
    for name, pred in ml_preds.items():
        tmp = test_base.copy()
        tmp["method"] = name
        tmp["pred"] = pred
        frames.append(tmp)

    print("ADIM 3 — Klasik yöntemler hesaplanıyor...")
    train_by_part = {p: g.sort_values(C.COL_MONTH_IDX)[C.COL_TARGET].values
                     for p, g in train_df.groupby(C.COL_PART)}
    test_by_part = {p: g.sort_values(C.COL_MONTH_IDX)
                    for p, g in test_base.groupby(C.COL_PART)}

    for part in parts:
        ptest = test_by_part.get(part)
        if ptest is None or len(ptest) == 0:
            continue
        series = train_by_part.get(part, np.array([]))
        cf = classical_forecasts(series, horizon=len(ptest))
        for name, fc in cf.items():
            tmp = ptest.copy()
            tmp["method"] = name
            tmp["pred"] = np.asarray(fc[:len(ptest)], float)
            frames.append(tmp)

    forecasts_long = pd.concat(frames, ignore_index=True)
    return forecasts_long, tuning_info


def _select_champions(forecasts_long):
    """Parça bazlı metrik oylaması ile şampiyon yöntemi belirler."""
    champ_rows = []
    method_metric_rows = []

    for part, sub in forecasts_long.groupby(C.COL_PART):
        per_method = {}
        for method, msub in sub.groupby("method"):
            per_method[method] = all_metrics(msub["actual"].values, msub["pred"].values)
            method_metric_rows.append({
                C.COL_PART: part, "method": method, **per_method[method]
            })

        methods = list(per_method.keys())
        votes = Counter()
        for metric in C.METRICS:
            best = min(methods, key=lambda m: per_method[m][metric])
            votes[best] += 1

        champion = min(methods, key=lambda m: (-votes[m], per_method[m]["WAPE"]))
        champ_rows.append({
            C.COL_PART: part,
            "champion": champion,
            "votes": votes[champion],
            "MAE": per_method[champion]["MAE"],
            "RMSE": per_method[champion]["RMSE"],
            "WAPE": per_method[champion]["WAPE"],
            "sMAPE": per_method[champion]["sMAPE"],
        })

    champions = pd.DataFrame(champ_rows)
    method_metrics = pd.DataFrame(method_metric_rows)
    return champions, method_metrics


def _overall_model_metrics(forecasts_long):
    """Yöntem başına genel (tüm parça-ay) metrikler — dashboard için."""
    out = {}
    for method, sub in forecasts_long.groupby("method"):
        out[method] = all_metrics(sub["actual"].values, sub["pred"].values)
    return out


def run_forecasting(data, feature_cols, parts, ml_trials=C.ML_OPTUNA_TRIALS):
    """
    Talep tahmini aşamasını uçtan uca çalıştırır.

    Returns:
        forecasts_long : [Parça_Kodu, Ay_Sirasi, Tarih, actual, method, pred]
        champions      : parça bazlı şampiyon yöntem + metrikleri
        method_metrics : parça x yöntem metrik tablosu
        report         : {"tuning": ..., "overall": ...}
    """
    forecasts_long, tuning_info = _build_forecast_table(
        data, feature_cols, parts, ml_trials)

    print("ADIM 4 — Şampiyon yöntem seçimi...")
    champions, method_metrics = _select_champions(forecasts_long)
    overall = _overall_model_metrics(forecasts_long)

    report = {"tuning": tuning_info, "overall": overall}
    print(f"ADIM 4 — {len(champions)} parça için şampiyon belirlendi.")
    print("Şampiyon dağılımı:")
    for m, c in champions["champion"].value_counts().items():
        print(f"  {m:20s}: {c}")

    return forecasts_long, champions, method_metrics, report
