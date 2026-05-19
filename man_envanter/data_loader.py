"""
ADIM 1 — Veri Yükleme
=====================
tüketim_v5.xlsx içindeki ML'e hazır sayfaları okur, Train/Test böler ve
parça bazlı parametre/segment bilgilerini tek "parça master" tablosunda
birleştirir. Posterdeki veri hazırlama (ADIM 1) ve segmentasyon (ADIM 2)
veri setinde tamamlandığı için bu modül yalnızca yükleme yapar.
"""

import os
import pandas as pd

import config as C


# Optimizasyon parametre sayfasındaki Türkçe sütunları sade adlara çevir
OPT_RENAME = {
    "Tedarik Süresi (gün)": "lead_time",
    "Birim Maliyet (TL)": "unit_cost",
    "Sipariş Maliyeti (TL)": "order_cost",
    "Elde Tutma (TL/adet/ay)": "holding_cost_month",
    "Stoksuz Maliyet (TL)": "stockout_cost",
    "Başlangıç Stok": "initial_stock",
}


def _strip_cols(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_all():
    """Tüm sayfaları yükler ve sözlük olarak döndürür."""
    if not os.path.exists(C.EXCEL_PATH):
        raise FileNotFoundError(f"Veri seti bulunamadı: {C.EXCEL_PATH}")

    xls = pd.ExcelFile(C.EXCEL_PATH, engine="openpyxl")

    ml = _strip_cols(xls.parse(C.SHEET_ML))
    segment = _strip_cols(xls.parse(C.SHEET_SEGMENT))
    opt = _strip_cols(xls.parse(C.SHEET_OPT))
    mps = _strip_cols(xls.parse(C.SHEET_MPS))
    strateji = _strip_cols(xls.parse(C.SHEET_STRATEJI))

    opt = opt.rename(columns=OPT_RENAME)

    return {
        "ml": ml,
        "segment": segment,
        "opt": opt,
        "mps": mps,
        "strateji": strateji,
    }


def feature_columns(ml_df):
    """ML model girdisi feature sütunlarını belirler."""
    cols = [c for c in ml_df.columns if c not in C.NON_FEATURE_COLS]
    if C.DROP_LAG12 and "lag_12" in cols:
        cols.remove("lag_12")
    return cols


def split_train_test(ml_df):
    """ML verisini Train (ay 1-30) ve Test (ay 31-36) olarak böler."""
    train = ml_df[ml_df[C.COL_SPLIT] == C.TRAIN_LABEL].copy()
    test = ml_df[ml_df[C.COL_SPLIT] == C.TEST_LABEL].copy()
    return train, test


def build_part_master(data):
    """Parça bazlı segment + optimizasyon parametre + strateji tablosu."""
    seg = data["segment"].copy()
    opt = data["opt"].copy()
    strat = data["strateji"][[C.COL_PART, "Model_Grubu"]].copy()

    master = seg.merge(opt, on=C.COL_PART, how="left")
    master = master.merge(strat, on=C.COL_PART, how="left")
    return master


def get_parts(data, max_parts=None):
    """Analiz edilecek parça kodlarının listesi."""
    parts = sorted(data["ml"][C.COL_PART].unique().tolist())
    if max_parts:
        parts = parts[:max_parts]
    return parts


def part_series(ml_df, part_code):
    """Tek parçanın aylık talep serisini (Train+Test, sıralı) döndürür."""
    s = ml_df[ml_df[C.COL_PART] == part_code].sort_values(C.COL_MONTH_IDX)
    return s


if __name__ == "__main__":
    data = load_all()
    print("Sayfa boyutları:")
    for k, v in data.items():
        print(f"  {k:10s}: {v.shape}")

    feats = feature_columns(data["ml"])
    print(f"\nFeature sayısı: {len(feats)}")

    train, test = split_train_test(data["ml"])
    print(f"Train satır: {len(train)}  |  Test satır: {len(test)}")

    parts = get_parts(data)
    print(f"Parça sayısı: {len(parts)}")

    master = build_part_master(data)
    print(f"Parça master: {master.shape}")
    print(f"Model_Grubu dağılımı:\n{master['Model_Grubu'].value_counts()}")
