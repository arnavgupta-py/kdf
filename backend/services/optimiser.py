import time
import math
import numpy as np
from loguru import logger
from typing import List
from scipy.stats import norm
from backend.schemas.optimization import DepartureOption
from backend.api.forecast import get_forecast

class DepartureOptimiser:
    """
    Computes a Pareto frontier of departure options given stochastic travel times.
    In real deployment, this would use OR-Tools to solve constraints on a real 
    routing formulation. We implement a rapid dynamic programming/grid search 
    to evaluate all possible departure times (every 5 mins).
    """
    def __init__(self, step_minutes=5):
        self.step_minutes = step_minutes

    async def compute_pareto_frontier(self, origin: str, destination: str, deadline: float, hours: int = 24, user_preferences: dict = None) -> List[DepartureOption]:
        """
        Evaluate departure times across a 24-hour horizon.
        Return non-dominated options (Pareto optimal) trading off expected travel time, 
        variance, and deadline arrival probability.
        Weights user preferences implicitly into the domination loop.
        """
        logger.info(f"Computing Pareto frontier for {origin} -> {destination} before {deadline}.")
        
        # Calculate maximum possible search range
        total_steps = int((hours * 60) / self.step_minutes)
        start_search_time = deadline - (hours * 3600)
        
        user_preferences = user_preferences or {"toll_aversion": 0.5, "variance_tolerance": 0.5, "highway_preference": 0.5}
        
        candidates = []
        for i in range(total_steps):
            depart_ts = start_search_time + (i * self.step_minutes * 60)
            
            # Determine forecast horizon (capped at 120 mins for API)
            horizon_mins = int((depart_ts - time.time()) / 60)
            # Fetch ST-GNN synthetic routing node forecasts
            forecast = await get_forecast(horizon_minutes=max(15, min(120, horizon_mins)), node_ids=None)
            
            # Use ST-GNN node congestion estimates to simulate expected path travel time
            avg_congestion = sum(n.expected_congestion for n in forecast.nodes) / len(forecast.nodes) if forecast.nodes else 0.5
            avg_variance = sum(n.confidence_interval[1] - n.confidence_interval[0] for n in forecast.nodes) / len(forecast.nodes) if forecast.nodes else 0.2
            
            # Path expected delay driven causally by the ST-GNN rather than arbitrary sine-wave
            # Route 1: Fastest (Highways, Tolls, Higher variance)
            mean_time_1 = 35.0 + (avg_congestion * 40.0)
            var_time_1 = 10.0 + (avg_variance * 60.0)
            
            # Route 2: Alternative (No Tolls, City Roads, Lower Variance / Slower)
            mean_time_2 = 45.0 + (avg_congestion * 30.0)
            var_time_2 = 5.0 + (avg_variance * 40.0)
            
            # Impose Personalised Preference Penalties to determine EFF (Effective) objective values
            var_penalty = 1.0 + (1.0 - user_preferences.get("variance_tolerance", 0.5)) * 2.0
            toll_penalty = 1.0 + user_preferences.get("toll_aversion", 0.5) * 0.4
            hwy_reward = 1.0 - user_preferences.get("highway_preference", 0.5) * 0.3
            
            eff_mean_1 = mean_time_1 * toll_penalty * hwy_reward
            eff_var_1 = var_time_1 * var_penalty
            
            eff_mean_2 = mean_time_2 # No tolls or hwys here
            eff_var_2 = var_time_2 * var_penalty
            
            # Calculate arrival probability based on Normal Distribution N(mean, var)
            expected_arrival_1 = depart_ts + (mean_time_1 * 60)
            prob_1 = norm.cdf((deadline - expected_arrival_1) / 60.0, loc=0, scale=math.sqrt(var_time_1))
            
            if prob_1 > 0.05:
                candidates.append({
                    "option": DepartureOption(
                        departure_time=depart_ts, expected_travel_time=mean_time_1,
                        travel_time_variance=var_time_1, arrival_probability=prob_1,
                        route_id=f"route_{origin}_{destination}_fastest"
                    ),
                    "eff_mean": eff_mean_1, "eff_var": eff_var_1, "prob": prob_1
                })
                
            expected_arrival_2 = depart_ts + (mean_time_2 * 60)
            prob_2 = norm.cdf((deadline - expected_arrival_2) / 60.0, loc=0, scale=math.sqrt(var_time_2))
            
            if prob_2 > 0.05:
                candidates.append({
                    "option": DepartureOption(
                        departure_time=depart_ts, expected_travel_time=mean_time_2,
                        travel_time_variance=var_time_2, arrival_probability=prob_2,
                        route_id=f"route_{origin}_{destination}_alt"
                    ),
                    "eff_mean": eff_mean_2, "eff_var": eff_var_2, "prob": prob_2
                })

        # Filter for Personalised Pareto optimality
        pareto_frontier = []
        for x in candidates:
            dominated = False
            for y in candidates:
                if x == y:
                    continue
                # y strictly dominates x based on personalised effective metrics
                if (y["prob"] >= x["prob"] and 
                    y["eff_mean"] <= x["eff_mean"] and 
                    y["eff_var"] <= x["eff_var"]):
                    
                    if (y["prob"] > x["prob"] or 
                        y["eff_mean"] < x["eff_mean"] or 
                        y["eff_var"] < x["eff_var"]):
                        dominated = True
                        break
            if not dominated:
                # Deduplicate very similar items
                if not any(abs(p.departure_time - x["option"].departure_time) < 1.0 and p.route_id == x["option"].route_id for p in pareto_frontier):
                    pareto_frontier.append(x["option"])
                    
        # Sort by departure time for return
        return sorted(pareto_frontier, key=lambda o: o.departure_time, reverse=True)
