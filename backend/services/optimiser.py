import time
import math
import numpy as np
from loguru import logger
from typing import List
from scipy.stats import norm
from backend.schemas.optimization import DepartureOption

class DepartureOptimiser:
    """
    Computes a Pareto frontier of departure options given stochastic travel times.
    In real deployment, this would use OR-Tools to solve constraints on a real 
    routing formulation. We implement a rapid dynamic programming/grid search 
    to evaluate all possible departure times (every 5 mins).
    """
    def __init__(self, step_minutes=5):
        self.step_minutes = step_minutes

    def compute_pareto_frontier(self, origin: str, destination: str, deadline: float, hours: int = 24) -> List[DepartureOption]:
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
            
            # Predict mean travel time based on distance (mock base) and simple temporal cycle
            # Morning/Evening bumps simulated via sine wave
            time_of_day_hr = (depart_ts % 86400) / 3600
            
            # Simulated base traffic + sine wave for rush hour (peaks at 9AM and 6PM)
            rush_hour_factor = (math.sin((time_of_day_hr - 9) * math.pi/4) + math.sin((time_of_day_hr - 18) * math.pi/4)) / 2
            
            # Simulated routing distribution: Mean ~45 mins, with added rush hour delay
            mean_time_min = 45.0 + max(0, rush_hour_factor * 25.0)
            
            # Variance increases proportionately with traffic congestion
            var_time_min2 = 15.0 + max(0, rush_hour_factor * 40.0) 
            
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
