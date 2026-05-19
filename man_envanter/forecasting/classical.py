"""
Klasik Talep Tahmin Yöntemleri
==============================
Her parçanın Train serisinden (ay 1-30) test ufku (6 ay) için tahmin üretir:
  - Croston-SBA  : aralıklı talep — SBA bias düzeltmeli Croston
  - HareketliOrtalama : son N ayın aritmetik ortalaması
  - UstelDuzeltme : α optimize edilen basit üstel düzeltme
"""

import numpy as np

import config as C


def moving_average(train_series, horizon=C.TEST_HORIZON, window=C.MA_WINDOW):
    """Son `window` ayın ortalaması, ufuk boyunca sabit."""
    arr = np.asarray(train_series, float)
    if len(arr) == 0:
        return np.zeros(horizon)
    w = min(window, len(arr))
    level = float(np.mean(arr[-w:]))
    return np.full(horizon, max(level, 0.0))


def exponential_smoothing(train_series, horizon=C.TEST_HORIZON):
    """Basit üstel düzeltme; α [0.05, 0.95] aralığında MSE ile seçilir."""
    arr = np.asarray(train_series, float)
    if len(arr) == 0:
        return np.zeros(horizon)
    if len(arr) == 1:
        return np.full(horizon, max(arr[0], 0.0))

    best_alpha, best_mse = 0.3, np.inf
    for alpha in np.arange(0.05, 0.96, 0.05):
        level = arr[0]
        sq_err = 0.0
        for x in arr[1:]:
            pred = level
            sq_err += (x - pred) ** 2
            level = alpha * x + (1 - alpha) * level
        mse = sq_err / (len(arr) - 1)
        if mse < best_mse:
            best_mse, best_alpha = mse, alpha

    level = arr[0]
    for x in arr[1:]:
        level = best_alpha * x + (1 - best_alpha) * level
    return np.full(horizon, max(level, 0.0))


def croston_sba(train_series, horizon=C.TEST_HORIZON, alpha=C.CROSTON_ALPHA):
    """
    Croston yöntemi + SBA (Syntetos-Boylan Approximation) bias düzeltmesi.
    Talep büyüklüğü ve talepler arası aralık ayrı üstel düzeltme ile izlenir;
    SBA çarpanı (1 - alpha/2) sistematik yanlılığı düzeltir.
    """
    arr = np.asarray(train_series, float)
    nonzero_idx = np.where(arr > 0)[0]
    if len(nonzero_idx) == 0:
        return np.zeros(horizon)
    if len(nonzero_idx) == 1:
        rate = arr[nonzero_idx[0]] / len(arr)
        return np.full(horizon, max(rate, 0.0))

    z = arr[nonzero_idx[0]]               # talep büyüklüğü tahmini
    x = float(nonzero_idx[0] + 1)         # aralık tahmini
    last = nonzero_idx[0]
    for idx in nonzero_idx[1:]:
        interval = idx - last
        z = alpha * arr[idx] + (1 - alpha) * z
        x = alpha * interval + (1 - alpha) * x
        last = idx

    forecast_rate = (1 - alpha / 2.0) * (z / x) if x > 0 else 0.0
    return np.full(horizon, max(forecast_rate, 0.0))


def classical_forecasts(train_series, horizon=C.TEST_HORIZON):
    """Üç klasik yöntemin tahminlerini sözlük olarak döndürür."""
    return {
        "Croston-SBA": croston_sba(train_series, horizon),
        "HareketliOrtalama": moving_average(train_series, horizon),
        "UstelDuzeltme": exponential_smoothing(train_series, horizon),
    }
