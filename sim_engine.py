"""
ADIM 5 — Stok Simülasyon Motoru
================================
(r, Q) sürekli gözden geçirmeli stok politikasının simülasyonu.

Model: Envanter pozisyonu (IP = eldeki stok + yoldaki sipariş) r seviyesine
düştüğünde Q'luk sipariş verilir; sipariş tedarik süresi sonra teslim alınır.
Karşılanamayan talep KAYIP SATIŞ olarak modellenir (geri ısmarlama yok) ve
stoksuz maliyetine yansır.

  * simulate_fast : NumPy tabanlı hızlı/deterministik tek koşu — Grid Search ve
                    Optuna araması bu motoru kullanır.
  * simulate_simpy: SimPy tabanlı ayrık olay simülasyonu, stokastik talep ve
                    tedarik süresi ile çok replikasyonlu — şampiyon doğrulaması.
"""

from dataclasses import dataclass
import numpy as np
import simpy

import config as C


@dataclass
class PartCosts:
    """Parça bazlı maliyet parametreleri."""
    unit_cost: float
    order_cost: float
    holding_per_day: float      # TL/adet/gün (aylık değerin /30'u)
    stockout_per_unit: float    # TL/adet (karşılanamayan birim başına)


def make_costs(row):
    """Optimizasyon parametre satırından PartCosts üretir."""
    return PartCosts(
        unit_cost=float(row["unit_cost"]),
        order_cost=float(row["order_cost"]),
        holding_per_day=float(row["holding_cost_month"]) / 30.0,
        stockout_per_unit=float(row["stockout_cost"]),
    )


def monthly_to_daily(monthly, horizon_days=C.SIM_HORIZON_DAYS):
    """Aylık tahminleri günlük talebe yayar; ufku doldurmak için tekrarlar."""
    monthly = np.asarray(monthly, float)
    if len(monthly) == 0:
        return np.zeros(horizon_days)
    daily = np.repeat(np.maximum(monthly, 0.0) / 30.0, 30)
    if len(daily) < horizon_days:
        reps = int(np.ceil(horizon_days / len(daily)))
        daily = np.tile(daily, reps)
    return daily[:horizon_days]


def _result(holding, ordering, stockout, demand, served, n_orders,
            inv_sum, inv_days, stockout_days):
    total = holding + ordering + stockout
    return {
        "total_cost": total,
        "holding_cost": holding,
        "ordering_cost": ordering,
        "stockout_cost": stockout,
        "service_level": (served / demand) if demand > 0 else 1.0,
        "num_orders": n_orders,
        "avg_inventory": (inv_sum / inv_days) if inv_days > 0 else 0.0,
        "stockout_days": stockout_days,
    }


# Maksimum eşzamanlı sipariş (sonsuz döngü koruması)
_MAX_ORDERS_PER_DAY = 5000


def simulate_fast(r, Q, daily_demand, lead_time, costs,
                  horizon=C.SIM_HORIZON_DAYS, warmup=C.SIM_WARMUP_DAYS):
    """Deterministik tek koşu — hızlı maliyet değerlendirmesi (arama için)."""
    r = max(0.0, float(r))
    Q = max(1.0, float(Q))
    lt = max(1, int(round(lead_time)))
    n = horizon + warmup
    arrivals = np.zeros(n + lt + 2)

    on_hand = r + Q
    on_order = 0.0
    holding = ordering = stockout = 0.0
    demand_tot = served_tot = 0.0
    n_orders = 0
    inv_sum = 0.0
    inv_days = stockout_days = 0

    for day in range(n):
        # Teslim alımları
        arr = arrivals[day]
        on_hand += arr
        on_order -= arr

        # Talebi karşıla (kayıp satış)
        d = daily_demand[day] if day < len(daily_demand) else daily_demand[-1]
        served = on_hand if on_hand < d else d
        on_hand -= served
        lost = d - served
        scored = day >= warmup

        # (r,Q) yeniden sipariş — envanter pozisyonu r'ye düşene kadar
        ip = on_hand + on_order
        guard = 0
        while ip <= r and guard < _MAX_ORDERS_PER_DAY:
            arrivals[day + lt] += Q
            on_order += Q
            ip += Q
            n_orders += 1
            guard += 1
            if scored:
                ordering += costs.order_cost

        # Maliyetler
        if scored:
            demand_tot += d
            served_tot += served
            if lost > 0:
                stockout += lost * costs.stockout_per_unit
                stockout_days += 1
            holding += on_hand * costs.holding_per_day
            inv_sum += on_hand
            inv_days += 1

    return _result(holding, ordering, stockout, demand_tot, served_tot,
                   n_orders, inv_sum, inv_days, stockout_days)


def _simpy_single(r, Q, daily_demand, lead_time, costs, rng, horizon, warmup):
    """Tek SimPy replikasyonu — stokastik talep ve tedarik süresi."""
    r = max(0.0, float(r))
    Q = max(1.0, float(Q))
    env = simpy.Environment()
    s = {
        "on_hand": r + Q, "on_order": 0.0,
        "holding": 0.0, "ordering": 0.0, "stockout": 0.0,
        "demand": 0.0, "served": 0.0, "n_orders": 0,
        "inv_sum": 0.0, "inv_days": 0, "stockout_days": 0,
    }

    def delivery(env, qty):
        actual_lt = max(1, int(rng.normal(lead_time, lead_time * C.LEAD_TIME_NOISE)))
        yield env.timeout(actual_lt)
        s["on_hand"] += qty
        s["on_order"] -= qty

    def demand_proc(env):
        day = 0
        while True:
            base = daily_demand[day] if day < len(daily_demand) else daily_demand[-1]
            noise = rng.normal(0, max(base * C.DEMAND_NOISE_STD, 0.05))
            d = max(0.0, base + noise)
            scored = env.now >= warmup

            served = min(s["on_hand"], d)
            s["on_hand"] -= served
            lost = d - served

            ip = s["on_hand"] + s["on_order"]
            guard = 0
            while ip <= r and guard < _MAX_ORDERS_PER_DAY:
                s["on_order"] += Q
                ip += Q
                s["n_orders"] += 1
                guard += 1
                if scored:
                    s["ordering"] += costs.order_cost
                env.process(delivery(env, Q))

            if scored:
                s["demand"] += d
                s["served"] += served
                if lost > 0:
                    s["stockout"] += lost * costs.stockout_per_unit
                    s["stockout_days"] += 1
                s["holding"] += s["on_hand"] * costs.holding_per_day
                s["inv_sum"] += s["on_hand"]
                s["inv_days"] += 1

            day += 1
            yield env.timeout(1)

    env.process(demand_proc(env))
    env.run(until=horizon + warmup)
    return _result(s["holding"], s["ordering"], s["stockout"], s["demand"],
                   s["served"], s["n_orders"], s["inv_sum"], s["inv_days"],
                   s["stockout_days"])


def simulate_simpy(r, Q, daily_demand, lead_time, costs,
                   n_replications=C.SIM_REPLICATIONS,
                   horizon=C.SIM_HORIZON_DAYS, warmup=C.SIM_WARMUP_DAYS,
                   seed=C.RANDOM_SEED):
    """SimPy ile çok replikasyonlu doğrulama — ortalama sonuçları döndürür."""
    runs = []
    for rep in range(n_replications):
        rng = np.random.RandomState(seed + rep)
        runs.append(_simpy_single(r, Q, daily_demand, lead_time, costs,
                                  rng, horizon, warmup))
    keys = runs[0].keys()
    avg = {k: float(np.mean([run[k] for run in runs])) for k in keys}
    avg["num_orders"] = int(round(avg["num_orders"]))
    return avg
