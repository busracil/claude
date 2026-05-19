"""
ADIM 3 — Makine Öğrenmesi Tabanlı Talep Tahmini
================================================
Dört model (RandomForest, XGBoost, LightGBM, CatBoost) tüm Train satırlarıyla
GLOBAL olarak eğitilir. Her model için Optuna (TPE) hiperparametre optimizasyonu
yapılır. Sentetik satırlar (is_synthetic=1) düşük ağırlıkla eğitilir. Eğitilen
modeller Test satırları (ay 31-36) üzerinde 6 aylık tahmin üretir.
"""

import warnings
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import config as C
from forecasting.metrics import wape

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Optuna hiperparametre tuning'inde kullanılan ay-bazlı holdout sınırı
_VAL_START_MONTH = 25


def _prepare_xy(df, feature_cols):
    """Feature matrisi (NaN -> 0), hedef ve örnek ağırlıklarını hazırlar."""
    X = df[feature_cols].fillna(0.0).astype(float).values
    y = df[C.COL_TARGET].astype(float).values
    if "is_synthetic" in df.columns:
        w = np.where(df["is_synthetic"].values == 1, C.SYNTHETIC_WEIGHT, 1.0)
    else:
        w = np.ones(len(df))
    return X, y, w


def _build_model(name, params):
    if name == "RandomForest":
        return RandomForestRegressor(random_state=C.RANDOM_SEED, n_jobs=-1, **params)
    if name == "XGBoost":
        return xgb.XGBRegressor(random_state=C.RANDOM_SEED, n_jobs=-1,
                                verbosity=0, **params)
    if name == "LightGBM":
        return lgb.LGBMRegressor(random_state=C.RANDOM_SEED, n_jobs=-1,
                                 verbose=-1, **params)
    if name == "CatBoost":
        return CatBoostRegressor(random_state=C.RANDOM_SEED, verbose=False,
                                 allow_writing_files=False, **params)
    raise ValueError(name)


def _suggest_params(name, trial):
    if name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 1.0]),
        }
    if name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 15, 120),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        }
    if name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
    raise ValueError(name)


def _tune(name, train_df, feature_cols, n_trials):
    """Ay-bazlı holdout ile Optuna hiperparametre optimizasyonu."""
    fit_df = train_df[train_df[C.COL_MONTH_IDX] < _VAL_START_MONTH]
    val_df = train_df[train_df[C.COL_MONTH_IDX] >= _VAL_START_MONTH]

    Xf, yf, wf = _prepare_xy(fit_df, feature_cols)
    Xv = val_df[feature_cols].fillna(0.0).astype(float).values
    yv = val_df[C.COL_TARGET_RAW].astype(float).values

    def objective(trial):
        params = _suggest_params(name, trial)
        model = _build_model(name, params)
        model.fit(Xf, yf, sample_weight=wf)
        pred = np.clip(model.predict(Xv), 0, None)
        return wape(yv, pred)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=C.RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def train_all_models(train_df, test_df, feature_cols, n_trials=C.ML_OPTUNA_TRIALS):
    """
    4 ML modelini global eğitir ve test satırlarında tahmin üretir.

    Returns:
        predictions: {model_name: np.array (test satırlarıyla hizalı)}
        tuning_info: {model_name: {"params":..., "val_wape":...}}
    """
    X_train, y_train, w_train = _prepare_xy(train_df, feature_cols)
    X_test = test_df[feature_cols].fillna(0.0).astype(float).values

    predictions = {}
    tuning_info = {}

    for name in C.ML_METHODS:
        print(f"  [{name}] hiperparametre optimizasyonu ({n_trials} trial)...")
        best_params, val_wape = _tune(name, train_df, feature_cols, n_trials)

        model = _build_model(name, best_params)
        model.fit(X_train, y_train, sample_weight=w_train)
        pred = np.clip(model.predict(X_test), 0, None)

        predictions[name] = pred
        tuning_info[name] = {"params": best_params, "val_wape": float(val_wape)}
        print(f"  [{name}] tamam — doğrulama WAPE: {val_wape:.2f}%")

    return predictions, tuning_info
