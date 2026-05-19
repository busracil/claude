"""
Hata Metrikleri
===============
MAE, RMSE, WAPE(%) ve sMAPE. Sıfır taleplerde MAPE sonsuza gittiği için
şampiyon seçiminde WAPE ve sMAPE tercih edilir (poster ADIM 4).
"""

import numpy as np


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error (%) — sıfır talebe dayanıklı."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0 if np.sum(np.abs(y_pred)) == 0 else 100.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def smape(y_true, y_pred):
    """Symmetric MAPE (%) — 0/0 durumunda 0 kabul edilir."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denom == 0, 0.0, 2.0 * np.abs(y_true - y_pred) / denom)
    return float(np.mean(terms) * 100.0)


def all_metrics(y_true, y_pred):
    """Dört metriği sözlük olarak döndürür."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }
