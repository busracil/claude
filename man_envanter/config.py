"""
MAN Türkiye — ML + Simülasyon ile Dinamik Envanter Optimizasyonu
Konfigürasyon: tüm sabitler, dosya yolları ve parametreler.
"""

import os

# ─── Dosya Yolları ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
EXCEL_PATH = os.path.join(PROJECT_ROOT, "tüketim_v5.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Çıktı dosyaları
FORECASTS_PATH = os.path.join(OUTPUT_DIR, "forecasts.parquet")
CHAMPIONS_PATH = os.path.join(OUTPUT_DIR, "champions.csv")
OPTIMIZATION_PATH = os.path.join(OUTPUT_DIR, "optimization.csv")
MODEL_METRICS_PATH = os.path.join(OUTPUT_DIR, "model_metrics.json")
DASHBOARD_PATH = os.path.join(OUTPUT_DIR, "dashboard.json")

# ─── Excel Sayfa Adları ──────────────────────────────────────────
SHEET_ML = "ML_Hazir_Veri"
SHEET_SEGMENT = "ABC_XYZ_Segmentasyon"
SHEET_OPT = "Optimizasyon_Parametreleri"
SHEET_MPS = "MPS_Long_Format"
SHEET_STRATEJI = "Model_Strateji"

# ─── Veri Sütunları ──────────────────────────────────────────────
COL_PART = "Parça_Kodu"
COL_DATE = "Tarih"
COL_MONTH_IDX = "Ay_Sirasi"
COL_SPLIT = "Split"
COL_TARGET = "Talep"            # winsorize — eğitim hedefi
COL_TARGET_RAW = "Talep_ham"    # ham — hata ölçümü

# ML feature dışı bırakılacak sütunlar (kimlik + hedef + string segment)
NON_FEATURE_COLS = [
    COL_PART, COL_DATE, COL_MONTH_IDX, COL_SPLIT,
    COL_TARGET, COL_TARGET_RAW, "ABC", "XYZ", "Segment",
]
# lag_12 sentetik tekrarı öğretiyor (korelasyon 0.99) — düşürmek için True yap
DROP_LAG12 = False

TRAIN_LABEL = "Train"
TEST_LABEL = "Test"
TEST_HORIZON = 6                # Test ufku (ay 31-36)

# ─── ML Model Eğitimi (ADIM 3) ───────────────────────────────────
RANDOM_SEED = 42
ML_OPTUNA_TRIALS = 30           # Her ML modeli için hiperparametre denemesi
SYNTHETIC_WEIGHT = 0.3          # is_synthetic satırların eğitim ağırlığı

# ─── Klasik Yöntemler ────────────────────────────────────────────
MA_WINDOW = 3                   # Hareketli ortalama penceresi (ay)
CROSTON_ALPHA = 0.1             # Croston/SBA düzeltme katsayısı

# ─── Optimizasyon (ADIM 5) ───────────────────────────────────────
SIM_HORIZON_DAYS = 360          # Simülasyon ufku (gün)
SIM_WARMUP_DAYS = 60            # Isınma dönemi (tedarik hattının dolması için)
SIM_REPLICATIONS = 20           # SimPy doğrulama replikasyonu (poster)
DEMAND_NOISE_STD = 0.25         # Günlük talep stokastik gürültü oranı
LEAD_TIME_NOISE = 0.10          # Tedarik süresi stokastik oranı

GRID_STEPS = 12                 # Grid Search r ve Q için adım sayısı
OPT_OPTUNA_TRIALS = 40          # (r,Q) ince ayar deneme sayısı

TARGET_SERVICE_LEVEL = 0.95     # Hedef fill rate
SERVICE_Z = 1.645               # %95 servis için z değeri (SS hesabı)

# ─── Çalıştırma Kapsamı ──────────────────────────────────────────
MAX_PARTS = None                # None = tüm 2.135 parça; smoke test için ör. 20
N_JOBS = -1                     # joblib paralel çekirdek (-1 = tümü)

# ─── Yöntem İsimleri ─────────────────────────────────────────────
ML_METHODS = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
CLASSICAL_METHODS = ["Croston-SBA", "HareketliOrtalama", "UstelDuzeltme"]
ALL_METHODS = ML_METHODS + CLASSICAL_METHODS    # 7 yöntem
METRICS = ["MAE", "RMSE", "WAPE", "sMAPE"]
