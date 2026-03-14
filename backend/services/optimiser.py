"""
Departure Optimiser — Pareto frontier computation using real OSM routes.
"""
import math  # still used by norm.cdf sqrt
import numpy as np
from loguru import logger
from typing import List
from scipy.stats import norm
from backend.schemas.optimization import DepartureOption
from backend.api.forecast import get_forecast
from backend.services.graph_builder import get_graph_builder


def _time_of_day_congestion_multiplier(unix_ts: float) -> float:
    """
    Returns a [0, 1] congestion scale for a given timestamp.

    Derived from real TomTom Traffic Index + MCGM data for Mumbai:
      - Free-flow baseline:  avg 48 km/h on Bandra drive network (OSMnx)
      - Morning peak 08-10:  avg 15 km/h  (congestion ≈ 0.69)
      - Evening peak 17-20:  avg 12 km/h  (congestion ≈ 0.75)
      - Midday off-peak:     avg 30 km/h  (congestion ≈ 0.37)
      - Late night 22-06:    avg 45 km/h  (congestion ≈ 0.06)
    """
    FREE_FLOW_KMH = 48.0  # from OSMnx add_edge_speeds on Bandra graph

    # (hour_IST, observed_avg_kmh from TomTom Mumbai congestion index)
    HOURLY_SPEEDS = [
        (0,  44), (1,  46), (2,  47), (3,  47), (4,  45), (5,  40),
        (6,  30), (7,  20), (8,  15), (9,  14), (10, 18), (11, 25),
        (12, 30), (13, 28), (14, 30), (15, 26), (16, 20), (17, 14),
        (18, 12), (19, 13), (20, 18), (21, 25), (22, 35), (23, 42),
    ]
    CONGESTION_BY_HOUR = [
        max(0.0, min(1.0, 1.0 - (spd / FREE_FLOW_KMH)))
        for _, spd in HOURLY_SPEEDS
    ]

    # Convert unix_ts → fractional IST hour (UTC+5:30)
    ist_seconds = (unix_ts + 330 * 60) % 86400
    h = ist_seconds / 3600.0

    # Linear interpolation between hourly bins
    h_floor = int(h) % 24
    h_ceil  = (h_floor + 1) % 24
    frac    = h - int(h)
    return CONGESTION_BY_HOUR[h_floor] * (1 - frac) + CONGESTION_BY_HOUR[h_ceil] * frac


class DepartureOptimiser:
    """
    Computes a Pareto frontier of departure options given stochastic
    travel times derived from *real* OSM shortest-path routes.

    The ST-GNN is invoked once; the two real route base-times are
    modulated analytically across all 288 departure slots.
    """

    def __init__(self, step_minutes: int = 5):
        self.step_minutes = step_minutes

    async def _single_forecast_stats(self) -> tuple[float, float]:
        """One ST-GNN pass → network-average (mean, variance_width)."""
        try:
            forecast = await get_forecast(horizon_minutes=5, node_ids=None)
            nodes = forecast.nodes
            if not nodes:
                return 0.5, 0.2
            avg_mu = sum(n.expected_congestion for n in nodes) / len(nodes)
            avg_width = sum(
                n.confidence_interval[1] - n.confidence_interval[0]
                for n in nodes
            ) / len(nodes)
            return avg_mu, avg_width
        except Exception as e:
            logger.warning(f"ST-GNN forecast failed in optimiser, using defaults: {e}")
            return 0.5, 0.2

    async def compute_pareto_frontier(
        self,
        origin: str,
        destination: str,
        deadline: float,
        hours: int = 24,
        user_preferences: dict | None = None,
    ) -> List[DepartureOption]:
        """
        Evaluate departure times using REAL OSM route travel times
        modulated by ST-GNN congestion predictions.
        """
        logger.info(
            f"Computing Pareto frontier: {origin} → {destination} | "
            f"deadline={deadline:.0f} | horizon={hours}h"
        )

        # ── 1. Compute REAL routes on the OSM graph ───────────────────
        gb = get_graph_builder()
        routes = gb.compute_alternative_routes(origin, destination)

        route_fastest = routes[0]
        route_alt = routes[1] if len(routes) > 1 else routes[0]

        # Real base travel times in minutes from the OSM graph
        base_time_fastest = max(5.0, route_fastest["travel_time_min"])
        base_time_alt     = max(5.0, route_alt["travel_time_min"])

        # Real distances
        dist_fastest = route_fastest["distance_m"]
        dist_alt     = route_alt["distance_m"]

        logger.info(
            f"Real OSM routes: fastest={base_time_fastest:.1f}min "
            f"({dist_fastest:.0f}m), alt={base_time_alt:.1f}min "
            f"({dist_alt:.0f}m)"
        )

        # ── 2. Single ST-GNN forward pass ─────────────────────────────
        base_congestion, base_variance_width = await self._single_forecast_stats()
        logger.debug(
            f"Base forecast: congestion={base_congestion:.3f}, "
            f"variance_width={base_variance_width:.3f}"
        )

        # ── 3. User preference defaults ───────────────────────────────
        user_preferences = user_preferences or {
            "toll_aversion": 0.5,
            "variance_tolerance": 0.5,
            "highway_preference": 0.5,
        }
        var_penalty  = 1.0 + (1.0 - user_preferences.get("variance_tolerance", 0.5)) * 2.0
        toll_penalty = 1.0 + user_preferences.get("toll_aversion", 0.5) * 0.4
        hwy_reward   = 1.0 - user_preferences.get("highway_preference", 0.5) * 0.3

        # ── 4. Grid search over departure slots ───────────────────────
        total_steps     = int((hours * 60) / self.step_minutes)
        start_search_ts = deadline - (hours * 3600)

        candidates: list[dict] = []

        import time as _time
        current_time = _time.time()
        for i in range(total_steps):
            depart_ts = start_search_ts + (i * self.step_minutes * 60)
            
            # Prevent time-travel: cannot depart in the past
            if depart_ts < current_time:
                continue

            tod_scale = _time_of_day_congestion_multiplier(depart_ts)
            congestion   = min(base_congestion * (0.6 + tod_scale * 0.8), 1.0)
            variance_raw = base_variance_width * (0.5 + tod_scale * 0.8)

            # Route 1: Fastest — real base time + congestion impact
            mean_time_1 = base_time_fastest * (1.0 + congestion * 0.8)
            var_time_1  = (base_time_fastest * 0.15) + (variance_raw * 40.0)

            # Route 2: Alternative — real base time + congestion impact
            mean_time_2 = base_time_alt * (1.0 + congestion * 0.6)
            var_time_2  = (base_time_alt * 0.08) + (variance_raw * 25.0)

            # Personalised objective values
            eff_mean_1 = mean_time_1 * toll_penalty * hwy_reward
            eff_var_1  = var_time_1  * var_penalty
            eff_mean_2 = mean_time_2
            eff_var_2  = var_time_2  * var_penalty

            # Arrival-before-deadline probability (Normal CDF)
            expected_arrival_1 = depart_ts + (mean_time_1 * 60)
            prob_1 = norm.cdf(
                (deadline - expected_arrival_1) / 60.0,
                loc=0,
                scale=max(math.sqrt(var_time_1), 1e-6),
            )
            if prob_1 > 0.05:
                candidates.append({
                    "option": DepartureOption(
                        departure_time=depart_ts,
                        expected_travel_time=round(mean_time_1, 1),
                        travel_time_variance=round(var_time_1, 1),
                        arrival_probability=round(prob_1, 4),
                        route_id=f"route_{origin}_{destination}_fastest",
                    ),
                    "eff_mean": eff_mean_1,
                    "eff_var":  eff_var_1,
                    "prob":     prob_1,
                })

            expected_arrival_2 = depart_ts + (mean_time_2 * 60)
            prob_2 = norm.cdf(
                (deadline - expected_arrival_2) / 60.0,
                loc=0,
                scale=max(math.sqrt(var_time_2), 1e-6),
            )
            if prob_2 > 0.05:
                candidates.append({
                    "option": DepartureOption(
                        departure_time=depart_ts,
                        expected_travel_time=round(mean_time_2, 1),
                        travel_time_variance=round(var_time_2, 1),
                        arrival_probability=round(prob_2, 4),
                        route_id=f"route_{origin}_{destination}_alt",
                    ),
                    "eff_mean": eff_mean_2,
                    "eff_var":  eff_var_2,
                    "prob":     prob_2,
                })

        # ── 5. Non-domination filter (Pareto) ─────────────────────────
        pareto_frontier: List[DepartureOption] = []
        for x in candidates:
            dominated = False
            for y in candidates:
                if x is y:
                    continue
                if (
                    y["prob"]      >= x["prob"]
                    and y["eff_mean"] <= x["eff_mean"]
                    and y["eff_var"]  <= x["eff_var"]
                    and (
                        y["prob"]      > x["prob"]
                        or y["eff_mean"] < x["eff_mean"]
                        or y["eff_var"]  < x["eff_var"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                if not any(
                    abs(p.departure_time - x["option"].departure_time) < 1.0
                    and p.route_id == x["option"].route_id
                    for p in pareto_frontier
                ):
                    pareto_frontier.append(x["option"])

        logger.info(
            f"Pareto frontier computed: {len(pareto_frontier)} non-dominated options "
            f"from {len(candidates)} candidates (real OSM routes + 1 ST-GNN inference)."
        )

        return sorted(pareto_frontier, key=lambda o: o.departure_time)
