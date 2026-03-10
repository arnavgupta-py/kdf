import time
import math
import random
from loguru import logger
from typing import List, Dict, Any
from backend.schemas.optimization import ParkingAlternative

class ParkingIntelligence:
    """
    Synthetic occupancy model simulating distinct commercial, residential, 
    and transit cluster patterns based on expected destination geometry 
    using probabilistic outputs.
    """
    def __init__(self):
        # We classify land zones into peak profile multipliers
        self.LAND_USAGE_PROFILES = {
            "commercial": {"peak_hours": [9, 13, 17], "variance": 0.2, "baseline": 0.4},
            "residential": {"peak_hours": [7, 19, 21], "variance": 0.15, "baseline": 0.6},
            "transit": {"peak_hours": [8, 18], "variance": 0.3, "baseline": 0.5}
        }
        
    def _compute_occupancy_prob(self, zone_type: str, time_hr: float) -> float:
        profile = self.LAND_USAGE_PROFILES.get(zone_type, self.LAND_USAGE_PROFILES["commercial"])
        base = profile["baseline"]
        peaks = profile["peak_hours"]
        
        # Superposition of Gaussian peaks over base occupancy
        active = sum([math.exp(-(((time_hr - p)**2) / (2 * profile["variance"]**2))) for p in peaks])
        
        # Squish to probability range
        prob = min(0.99, base + (active * 0.45))
        return max(0.01, prob + random.uniform(-0.05, 0.05))

    def evaluate_parking(self, destination: str, arrival_time: float, zone_type: str) -> Dict[str, Any]:
        """
        Evaluate synthetic parking occupancy probabilistically.
        Returns the primary probability and a set of local alternatives sorted by score.
        """
        logger.info(f"Evaluating parking for {destination} at ts={arrival_time} inside {zone_type}")
        time_hr = (arrival_time % 86400) / 3600
        
        primary_prob = self._compute_occupancy_prob(zone_type, time_hr)
        
        alternatives = []
        if primary_prob > 0.8:
            logger.info("Destination occupied probability high, computing alternatives.")
            # Surface 3 fallback options
            alts_zones = [
                (f"{destination}_alt_garage_1", "commercial", 450, 5.0),
                (f"{destination}_alt_street_a", "residential", 800, 0.0),
                (f"{destination}_alt_transit_p", "transit", 1200, 2.0)
            ]
            
            for alts_id, ztype, dist, cost in alts_zones:
                prob = self._compute_occupancy_prob(ztype, time_hr)
                alternatives.append(
                    ParkingAlternative(
                        location_id=alts_id,
                        occupancy_probability=prob,
                        walking_distance_meters=dist,
                        cost=cost
                    )
                )
        
        # Sort alternatives by a composite score of (probability penalty + distance penalty)
        alternatives.sort(key=lambda x: (x.occupancy_probability * 10) + (x.walking_distance_meters / 100))
        
        return {
            "primary_occupancy_probability": round(primary_prob, 3),
            "alternatives": alternatives
        }
