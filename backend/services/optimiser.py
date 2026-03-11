import time
import math
import numpy as np
from loguru import logger
from typing import List
from scipy.stats import norm
from backend.schemas.optimization import DepartureOption

# We import only the forecast helper, not the full HTTP endpoint,
# so we can call the underlying logic directly (1 ST-GNN pass total).
from backend.api.forecast import get_forecast


def _time_of_day_congestion_multiplier(unix_ts: float) -> float:
    """
    Returns a [0, 1] congestion scale factor for a given timestamp.
    Peaks are calibrated to Mumbai patterns:
      - Morning rush:  08:00–10:00 IST  (UTC+5:30 offset applied)
      - Lunchtime dip: 12:30–14:00
      - Evening rush: 17:00–20:00 IST
    We use a superposition of Gaussians over the 24-hour cycle.
    """
    # Convert to fractional hour in IST (UTC + 5:30 = 330 min)
    ist_seconds = (unix_ts + 330 * 60) % 86400
    h = ist_seconds / 3600.0  # 0.0–24.0

    peak_hours = [9.0, 18.5]   # morning and evening rush centroids
    widths     = [1.2, 1.5]    # σ in hours

    scale = 0.25  # baseline off-peak multiplier
    for mu, sigma in zip(peak_hours, widths):
        scale += 0.4 * math.exp(-((h - mu) ** 2) / (2 * sigma ** 2))

    return min(scale, 1.0)


class DepartureOptimiser:
    """
    Computes a Pareto frontier of departure options given stochastic travel times.

    The key optimisation over the prior implementation:
    – The ST-GNN inference is executed **once** per call to
      `compute_pareto_frontier`, producing a baseline (avg_congestion,
      avg_variance) for the current moment.
    – All 288 × 2 departure slot evaluations then apply a time-of-day
      modulation curve around that baseline — O(1) model calls, not O(N).
    """

    def __init__(self, step_minutes: int = 5):
        self.step_minutes = step_minutes

    async def _single_forecast_stats(self) -> tuple[float, float]:
        """
        Runs exactly one ST-GNN forecast and returns the network-average
        (mean_congestion, mean_variance_width).  Uses a short 5-minute horizon
        to represent current conditions only.
        """
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
        Evaluate departure times across the planning horizon.
        Returns non-dominated options (Pareto optimal) trading off expected
        travel time, variance, and deadline arrival probability.

        ST-GNN is called **once**; time-of-day modulation is applied analytically.
        """
        logger.info(
            f"Computing Pareto frontier: {origin} → {destination} | "
            f"deadline={deadline:.0f} | horizon={hours}h"
        )

        # ── 1. Single ST-GNN forward pass ──────────────────────────────────
        base_congestion, base_variance_width = await self._single_forecast_stats()
        logger.debug(
            f"Base forecast: congestion={base_congestion:.3f}, "
            f"variance_width={base_variance_width:.3f}"
        )

        # ── 2. User preference defaults ────────────────────────────────────
        user_preferences = user_preferences or {
            "toll_aversion": 0.5,
            "variance_tolerance": 0.5,
            "highway_preference": 0.5,
        }
        var_penalty  = 1.0 + (1.0 - user_preferences.get("variance_tolerance", 0.5)) * 2.0
        toll_penalty = 1.0 + user_preferences.get("toll_aversion", 0.5) * 0.4
        hwy_reward   = 1.0 - user_preferences.get("highway_preference", 0.5) * 0.3

        # ── 3. Grid search over departure slots ────────────────────────────
        total_steps     = int((hours * 60) / self.step_minutes)
        start_search_ts = deadline - (hours * 3600)

        candidates: list[dict] = []

        for i in range(total_steps):
            depart_ts = start_search_ts + (i * self.step_minutes * 60)

            # Time-of-day scale applied to base forecast: analytically cheap
            tod_scale = _time_of_day_congestion_multiplier(depart_ts)
            congestion   = min(base_congestion * (0.6 + tod_scale * 0.8), 1.0)
            variance_raw = base_variance_width * (0.5 + tod_scale * 0.8)

            # Route 1: Fastest – highways/tolls, higher variance
            mean_time_1 = 35.0 + (congestion * 40.0)
            var_time_1  = 10.0 + (variance_raw * 60.0)

            # Route 2: Alternative – city roads, lower variance, slower
            mean_time_2 = 45.0 + (congestion * 30.0)
            var_time_2  = 5.0  + (variance_raw * 40.0)

            # Effective (personalised) objective values
            eff_mean_1 = mean_time_1 * toll_penalty * hwy_reward
            eff_var_1  = var_time_1  * var_penalty
            eff_mean_2 = mean_time_2
            eff_var_2  = var_time_2  * var_penalty

            # Arrival-before-deadline probability via Normal CDF
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
                        expected_travel_time=mean_time_1,
                        travel_time_variance=var_time_1,
                        arrival_probability=prob_1,
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
                        expected_travel_time=mean_time_2,
                        travel_time_variance=var_time_2,
                        arrival_probability=prob_2,
                        route_id=f"route_{origin}_{destination}_alt",
                    ),
                    "eff_mean": eff_mean_2,
                    "eff_var":  eff_var_2,
                    "prob":     prob_2,
                })

        # ── 4. Non-domination filter (personalised Pareto) ─────────────────
        pareto_frontier: List[DepartureOption] = []
        for x in candidates:
            dominated = False
            for y in candidates:
                if x is y:
                    continue
                # y weakly dominates x on all three objectives AND strictly on at least one
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
                # Deduplicate near-identical departure times for same route
                if not any(
                    abs(p.departure_time - x["option"].departure_time) < 1.0
                    and p.route_id == x["option"].route_id
                    for p in pareto_frontier
                ):
                    pareto_frontier.append(x["option"])

        logger.info(
            f"Pareto frontier computed: {len(pareto_frontier)} non-dominated options "
            f"from {len(candidates)} candidates (1 ST-GNN inference used)."
        )

        # Sort chronologically for the UI
        return sorted(pareto_frontier, key=lambda o: o.departure_time)
