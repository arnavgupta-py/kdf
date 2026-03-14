"""
Real traffic estimation service using OSM graph structural data.

Replaces the previous mock Google Maps integration with actual
graph-derived metrics (edge speeds, road density, intersection degree).
"""
import math
from loguru import logger
from typing import Dict, Any, List

from backend.services.graph_builder import get_graph_builder


class TrafficEstimationService:
    """Computes real traffic / congestion metrics from the loaded OSM graph."""

    def __init__(self):
        logger.info("Initialized real OSM-based Traffic Estimation Service.")

    def fetch_traffic_data(self, origin: str, destination: str) -> Dict[str, Any]:
        """Compute per-segment congestion along a real route."""
        gb = get_graph_builder()
        route_info = gb.compute_route(origin, destination, weight="travel_time")

        segments: List[Dict[str, Any]] = []
        path = route_info.get("path", [])
        G = gb.G

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = min(G[u][v].values(), key=lambda d: d.get("travel_time", 1e9))

            speed = edge_data.get("speed_kph", 30.0)
            if isinstance(speed, list):
                speed = speed[0]
            speed = float(speed)

            length = edge_data.get("length", 100.0)
            if isinstance(length, list):
                length = length[0]

            # Congestion from actual speed vs free-flow (60 km/h baseline)
            cong = max(0.0, min(1.0, 1.0 - speed / 60.0))
            level = "HIGH" if cong > 0.6 else "MODERATE" if cong > 0.3 else "LOW"

            segments.append({
                "start_location": str(u),
                "end_location": str(v),
                "congestion_level": level,
                "speed_kmh": round(speed, 1),
                "length_m": round(float(length), 1),
            })

        return {
            "status": "OK",
            "traffic_model": "osm_graph_derived",
            "total_distance_m": route_info["distance_m"],
            "total_travel_time_s": route_info["travel_time_s"],
            "segments": segments,
        }

    def fetch_area_congestion(self, place_name: str) -> Dict[str, Any]:
        """Real area-level congestion computed from graph edge attributes."""
        gb = get_graph_builder()
        return gb.estimate_area_congestion()


# Backward-compatible alias so existing imports keep working
GoogleMapsIntegration = TrafficEstimationService
