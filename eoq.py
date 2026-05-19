"""
ADIM 6 — EOQ Klasik Politika ve Karşılaştırma
==============================================
Geleneksel EOQ politikası tarihsel ortalama talebe dayanır:
    EOQ = sqrt(2 * D * S / h)          (yıllık D, sipariş maliyeti S, elde tutma h)
    SS  = z * sigma_L * sqrt(LT)
    ROP = D_LT + SS
EOQ politikası (r=ROP, Q=EOQ) ile ML+SimPy ile optimize edilen politika AYNI
simülasyon evreninde karşılaştırılır. Tasarruf = EOQ maliyeti - ML+SimPy maliyeti.
"""

import numpy as np

import config as C
from optimization.sim_engine import make_costs, monthly_to_daily, simulate_simpy


def eoq_parameters(hist_monthly, lead_time, costs):
    """Tarihsel aylık talep serisinden EOQ politika parametrelerini hesaplar."""
    hist = np.asarray(hist_monthly, float)
    mu_month = float(np.mean(hist)) if len(hist) else 0.0
    std_month = float(np.std(hist)) if len(hist) else 0.0

    D_annual = mu_month * 12.0
    S = costs.order_cost
    h_annual = costs.holding_per_day * 365.0

    if D_annual > 0 and h_annual > 0:
        eoq = np.sqrt(2.0 * D_annual * S / h_annual)
    else:
        eoq = max(mu_month, 1.0)

    mu_daily = mu_month / 30.0
    sigma_daily = std_month / np.sqrt(30.0)
    d_lt = mu_daily * lead_time
    safety_stock = C.SERVICE_Z * sigma_daily * np.sqrt(lead_time)
    rop = d_lt + safety_stock

    return {
        "r_eoq": int(round(max(rop, 0))),
        "Q_eoq": int(round(max(eoq, 1))),
        "SS_eoq": round(float(safety_stock), 1),
        "D_annual": D_annual,
    }


def evaluate_eoq(part_code, hist_monthly, opt_row, daily_demand):
    """EOQ politikasını ML ile aynı talep serisinde SimPy ile değerlendirir."""
    costs = make_costs(opt_row)
    lead_time = float(opt_row["lead_time"])
    params = eoq_parameters(hist_monthly, lead_time, costs)

    val = simulate_simpy(params["r_eoq"], params["Q_eoq"],
                         daily_demand, lead_time, costs)

    return {
        C.COL_PART: part_code,
        "r_eoq": params["r_eoq"],
        "Q_eoq": params["Q_eoq"],
        "SS_eoq": params["SS_eoq"],
        "eoq_total_cost": val["total_cost"],
        "eoq_holding_cost": val["holding_cost"],
        "eoq_ordering_cost": val["ordering_cost"],
        "eoq_stockout_cost": val["stockout_cost"],
        "eoq_service_level": val["service_level"],
        "eoq_num_orders": val["num_orders"],
        "eoq_avg_inventory": val["avg_inventory"],
    }


def savings(ml_cost, eoq_cost):
    """EOQ'ya kıyasla mutlak ve yüzde tasarruf."""
    abs_save = eoq_cost - ml_cost
    pct = (abs_save / eoq_cost * 100.0) if eoq_cost > 0 else 0.0
    return abs_save, pct
