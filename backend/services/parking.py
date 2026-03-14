"""
Parking Intelligence — real OSM parking amenity queries with
probabilistic occupancy estimation.
"""
import math
import time
import random
import osmnx as ox
from loguru import logger
from typing import List, Dict, Any

from backend.schemas.optimization import ParkingAlternative
from backend.services.graph_builder import get_graph_builder


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class ParkingIntelligence:
    """
    Queries real parking locations from OpenStreetMap and estimates
    occupancy probabilistically using land-use zone profiles.
    """

    def __init__(self):
        self.LAND_USAGE_PROFILES = {
            "commercial":  {"peak_hours": [9, 13, 17], "variance": 0.2, "baseline": 0.4},
            "residential": {"peak_hours": [7, 19, 21], "variance": 0.15, "baseline": 0.6},
            "transit":     {"peak_hours": [8, 18],     "variance": 0.3,  "baseline": 0.5},
        }
        # Cache OSM parking queries keyed by rounded (lat, lon)
        self._parking_cache: Dict[tuple, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Occupancy model (kept — the math is reasonable, no real sensor data)
    # ------------------------------------------------------------------
    def _compute_occupancy_prob(self, zone_type: str, time_hr: float) -> float:
        profile = self.LAND_USAGE_PROFILES.get(zone_type, self.LAND_USAGE_PROFILES["commercial"])
        base = profile["baseline"]
        peaks = profile["peak_hours"]
        active = sum(
            math.exp(-(((time_hr - p) ** 2) / (2 * profile["variance"] ** 2)))
            for p in peaks
        )
        prob = min(0.99, base + active * 0.45)
        return max(0.01, prob + random.uniform(-0.05, 0.05))

    # ------------------------------------------------------------------
    # Real parking location queries
    # ------------------------------------------------------------------
    def _fetch_real_parking(self, lat: float, lon: float, radius: int = 1500) -> List[Dict[str, Any]]:
        """Query OSM Overpass API for parking amenities near (lat, lon)."""
        cache_key = (round(lat, 3), round(lon, 3))
        if cache_key in self._parking_cache:
            return self._parking_cache[cache_key]

        results: List[Dict[str, Any]] = []
        try:
            gdf = ox.features_from_point((lat, lon), tags={"amenity": "parking"}, dist=radius)
            logger.info(f"OSM parking query returned {len(gdf)} results near ({lat:.4f}, {lon:.4f})")

            for idx, row in gdf.iterrows():
                # Extract centroid for the parking feature
                geom = row.geometry
                if geom is None:
                    continue
                centroid = geom.centroid
                p_lat, p_lon = centroid.y, centroid.x

                name = row.get("name", None)
                if name is None or (hasattr(name, '__class__') and name.__class__.__name__ == 'float'):
                    name = f"Parking @ ({p_lat:.4f}, {p_lon:.4f})"

                capacity = row.get("capacity", None)
                try:
                    capacity = int(capacity) if capacity is not None else None
                except (ValueError, TypeError):
                    capacity = None

                access_type = row.get("access", "public")
                fee = row.get("fee", "unknown")
                parking_type = row.get("parking", "surface")

                walking_dist = _haversine(lat, lon, p_lat, p_lon)

                results.append({
                    "location_id": str(name),
                    "lat": round(p_lat, 6),
                    "lon": round(p_lon, 6),
                    "capacity": capacity,
                    "access": str(access_type) if access_type else "public",
                    "fee": str(fee) if fee else "unknown",
                    "parking_type": str(parking_type) if parking_type else "surface",
                    "walking_distance_m": round(walking_dist, 1),
                })
        except Exception as e:
            logger.warning(f"OSM parking query failed, using graph-based fallback: {e}")
            results = self._fallback_parking_from_graph(lat, lon)

        # Sort by walking distance
        results.sort(key=lambda x: x["walking_distance_m"])
        self._parking_cache[cache_key] = results
        return results

    def _fallback_parking_from_graph(self, lat: float, lon: float) -> List[Dict[str, Any]]:
        """If Overpass query fails, use nearby graph nodes as candidate parking."""
        gb = get_graph_builder()
        if gb.G is None:
            return []

        center_node = gb.find_nearest_node(lat, lon)
        # Get nodes within ~800m
        candidates = []
        for node, data in gb.G.nodes(data=True):
            n_lat = data.get("y", 0.0)
            n_lon = data.get("x", 0.0)
            dist = _haversine(lat, lon, n_lat, n_lon)
            if dist < 1200:
                degree = gb.G.degree(node)
                # Higher-degree nodes are intersections — likely near parking
                if degree >= 3:
                    candidates.append({
                        "location_id": f"Node {node} (intersection)",
                        "lat": round(n_lat, 6),
                        "lon": round(n_lon, 6),
                        "capacity": None,
                        "access": "public",
                        "fee": "unknown",
                        "parking_type": "street",
                        "walking_distance_m": round(dist, 1),
                    })
        candidates.sort(key=lambda x: x["walking_distance_m"])
        return candidates[:10]

    # ------------------------------------------------------------------
    # Main evaluation (uses real locations now)
    # ------------------------------------------------------------------
    def evaluate_parking(self, destination: str, arrival_time: float, zone_type: str) -> Dict[str, Any]:
        """Evaluate parking with real OSM locations and probabilistic occupancy."""
        logger.info(f"Evaluating parking for '{destination}' at ts={arrival_time} zone={zone_type}")

        # Geocode destination to real coordinates
        gb = get_graph_builder()
        dest_lat, dest_lon = gb.geocode_place(destination)

        # Convert Unix timestamp to IST fractional hour (UTC+5:30 = +19800s)
        time_hr = ((arrival_time + 19800) % 86400) / 3600
        primary_prob = self._compute_occupancy_prob(zone_type, time_hr)

        # Fetch real parking locations from OSM
        parking_spots = self._fetch_real_parking(dest_lat, dest_lon)

        alternatives: List[ParkingAlternative] = []

        if parking_spots:
            # First spot is the primary (closest to destination)
            primary = parking_spots[0]
            primary_prob = self._compute_occupancy_prob(zone_type, time_hr)

            # Remaining spots become alternatives
            for spot in parking_spots[1:8]:  # up to 7 alternatives
                spot_prob = self._compute_occupancy_prob(zone_type, time_hr)
                # Spots farther away tend to have lower occupancy
                distance_discount = max(0.0, 1.0 - spot["walking_distance_m"] / 2000.0)
                adjusted_prob = spot_prob * (0.5 + 0.5 * distance_discount)

                cost = 0.0
                if spot.get("fee") in ("yes", "Yes"):
                    cost = 5.0 + random.uniform(0, 5)
                elif spot.get("parking_type") == "multi-storey":
                    cost = 8.0 + random.uniform(0, 4)

                alternatives.append(
                    ParkingAlternative(
                        location_id=spot["location_id"],
                        occupancy_probability=round(adjusted_prob, 3),
                        walking_distance_meters=spot["walking_distance_m"],
                        cost=round(cost, 2),
                    )
                )
        else:
            logger.info("No real parking data available, using occupancy model only.")

        # Sort alternatives: low occupancy & short walk first
        alternatives.sort(
            key=lambda x: (x.occupancy_probability * 10) + (x.walking_distance_meters / 100)
        )

        return {
            "primary_occupancy_probability": round(primary_prob, 3),
            "alternatives": alternatives,
        }
