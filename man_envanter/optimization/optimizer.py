"""
ADIM 5 — (r, Q) Parametre Optimizasyonu
========================================
Aşama 1: Grid Search — geniş Q×r ızgarasında maliyet haritası çıkarılır.
Aşama 2: Optuna (TPE) — en iyi bölgenin çevresinde hassas (r*, Q*) aranır.
Aşama 3: SimPy — bulunan (r*, Q*) 20 replikasyonla doğrulanır.
Servis seviyesi kısıtı ceza fonksiyonu ile uygulanır.
"""

import numpy as np
import optuna

import config as C
from optimization.sim_engine import (
    make_costs, monthly_to_daily, simulate_fast, simulate_simpy,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _ranges(daily_demand, lead_time):
    """Parçaya özel r ve Q arama aralıklarını belirler."""
    mu = float(np.mean(daily_demand)) if len(daily_demand) else 0.0
    d_lt = mu * lead_time
    r_max = max(d_lt * 3.0, 10.0)
    q_min = max(1.0, mu * 3.0)
    q_max = max(mu * 180.0, d_lt * 2.0, q_min * 8.0, 20.0)
    r_grid = np.unique(np.linspace(0, r_max, C.GRID_STEPS).round().astype(int))
    q_grid = np.unique(np.linspace(q_min, q_max, C.GRID_STEPS).round().astype(int))
    q_grid = q_grid[q_grid >= 1]
    return r_grid, q_grid


def _penalized_cost(res):
    """Servis seviyesi hedefin altındaysa maliyete ceza ekler."""
    cost = res["total_cost"]
    sl = res["service_level"]
    if sl < C.TARGET_SERVICE_LEVEL:
        cost += (C.TARGET_SERVICE_LEVEL - sl) * max(cost, 1.0) * 10.0
    return cost


def grid_search(daily_demand, lead_time, costs):
    """Aşama 1 — ızgara taraması, maliyet haritası ve en iyi (r,Q)."""
    r_grid, q_grid = _ranges(daily_demand, lead_time)
    cost_map = []
    best = None
    for r in r_grid:
        for Q in q_grid:
            res = simulate_fast(int(r), int(Q), daily_demand, lead_time, costs)
            score = _penalized_cost(res)
            cost_map.append({"r": int(r), "Q": int(Q),
                             "cost": res["total_cost"],
                             "service_level": res["service_level"]})
            if best is None or score < best[0]:
                best = (score, int(r), int(Q))
    return best[1], best[2], cost_map


def optuna_tune(daily_demand, lead_time, costs, r0, Q0):
    """Aşama 2 — grid sonucu çevresinde Optuna ince ayarı."""
    r_lo, r_hi = max(0, int(r0 * 0.7)), int(r0 * 1.3) + 5
    q_lo, q_hi = max(1, int(Q0 * 0.7)), int(Q0 * 1.3) + 5

    def objective(trial):
        r = trial.suggest_int("r", r_lo, r_hi)
        Q = trial.suggest_int("Q", q_lo, q_hi)
        res = simulate_fast(r, Q, daily_demand, lead_time, costs)
        return _penalized_cost(res)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=C.RANDOM_SEED))
    study.optimize(objective, n_trials=C.OPT_OPTUNA_TRIALS, show_progress_bar=False)
    return int(study.best_params["r"]), int(study.best_params["Q"])


def optimize_part(part_code, monthly_forecast, opt_row):
    """
    Tek parça için tam optimizasyon: Grid Search -> Optuna -> SimPy doğrulama.

    Returns: r*, Q*, SS, maliyet bileşenleri ve servis seviyesi içeren sözlük.
    """
    costs = make_costs(opt_row)
    lead_time = float(opt_row["lead_time"])
    daily_demand = monthly_to_daily(monthly_forecast)
    mu_daily = float(np.mean(daily_demand))

    r_grid_best, q_grid_best, cost_map = grid_search(daily_demand, lead_time, costs)
    r_opt, q_opt = optuna_tune(daily_demand, lead_time, costs,
                               r_grid_best, q_grid_best)

    # SimPy doğrulama (20 replikasyon)
    val = simulate_simpy(r_opt, q_opt, daily_demand, lead_time, costs)

    safety_stock = max(0.0, r_opt - mu_daily * lead_time)

    return {
        C.COL_PART: part_code,
        "r_optimal": r_opt,
        "Q_optimal": q_opt,
        "SS_optimal": round(safety_stock, 1),
        "grid_r": r_grid_best,
        "grid_Q": q_grid_best,
        "ml_total_cost": val["total_cost"],
        "ml_holding_cost": val["holding_cost"],
        "ml_ordering_cost": val["ordering_cost"],
        "ml_stockout_cost": val["stockout_cost"],
        "ml_service_level": val["service_level"],
        "ml_num_orders": val["num_orders"],
        "ml_avg_inventory": val["avg_inventory"],
        "cost_map": cost_map,
    }
