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

    async def compute_pareto_frontier(self, origin: str, destination: str, deadline: float, hours: int = 24) -> List[DepartureOption]:
        """
        Evaluate departure times across a 24-hour horizon.
        Return non-dominated options (Pareto optimal) trading off expected travel time, 
        variance, and deadline arrival probability.
        """
        logger.info(f"Computing Pareto frontier for {origin} -> {destination} before {deadline}.")
        
        # Calculate maximum possible search range
        total_steps = int((hours * 60) / self.step_minutes)
        start_search_time = deadline - (hours * 3600)
        
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
            mean_time_min = 35.0 + (avg_congestion * 40.0)
            var_time_min2 = 10.0 + (avg_variance * 50.0)
            
            # Calculate arrival probability based on Normal Distribution N(mean_time_min, var_time_min2)
            expected_arrival = depart_ts + (mean_time_min * 60)
            diff_from_deadline_min = (deadline - expected_arrival) / 60.0
            
            # Prob(Arrival <= Deadline)
            arrival_prob = norm.cdf(diff_from_deadline_min, loc=0, scale=math.sqrt(var_time_min2))
            
            # We enforce a hard constraint that user needs at least some chance of making it
            if arrival_prob > 0.05:
                candidates.append(
                    DepartureOption(
                        departure_time=depart_ts,
                        expected_travel_time=mean_time_min,
                        travel_time_variance=var_time_min2,
                        arrival_probability=arrival_prob,
                        route_id=f"route_{origin}_{destination}_fastest"
                    )
                )

        # Filter for Pareto optimality
        # Objectives: maximize arrival_prob, minimize expected_travel_time, minimize variance
        # For simplicity, we filter dominated solutions.
        pareto_frontier = []
        for x in candidates:
            dominated = False
            for y in candidates:
                if x == y:
                    continue
                # y strictly dominates x if it's better or equal on all objectives and strictly better on one
                if (y.arrival_probability >= x.arrival_probability and 
                    y.expected_travel_time <= x.expected_travel_time and 
                    y.travel_time_variance <= x.travel_time_variance):
                    
                    if (y.arrival_probability > x.arrival_probability or 
                        y.expected_travel_time < x.expected_travel_time or 
                        y.travel_time_variance < x.travel_time_variance):
                        dominated = True
                        break
            if not dominated:
                # Deduplicate very similar items
                if not any(abs(p.departure_time - x.departure_time) < 1.0 for p in pareto_frontier):
                    pareto_frontier.append(x)
                    
        # Sort by departure time for return
        return sorted(pareto_frontier, key=lambda o: o.departure_time, reverse=True)
