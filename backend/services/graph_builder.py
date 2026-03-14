import os
import math
import pickle
import osmnx as ox
import networkx as nx
import torch
from torch_geometric.data import Data
from loguru import logger
from typing import Tuple, Optional, List, Dict, Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DIR = os.path.join(_REPO_ROOT, "cache")
_GRAPH_CACHE_PATH = os.path.join(_CACHE_DIR, "bandra_graph.pkl")

# Pre-cached coordinates for common Mumbai landmarks so geocoding is instant
# during hackathon demos (falls back to OSMnx geocoding for unknown places).
MUMBAI_LANDMARKS: Dict[str, Tuple[float, float]] = {
    "bandra west":   (19.0596, 72.8295),
    "bandra east":   (19.0655, 72.8510),
    "bandra":        (19.0596, 72.8295),
    "lower parel":   (19.0048, 72.8306),
    "andheri":       (19.1136, 72.8697),
    "andheri west":  (19.1197, 72.8464),
    "andheri east":  (19.1136, 72.8697),
    "dadar":         (19.0178, 72.8478),
    "kurla":         (19.0726, 72.8845),
    "worli":         (19.0176, 72.8155),
    "juhu":          (19.0883, 72.8264),
    "santacruz":     (19.0815, 72.8411),
    "churchgate":    (18.9355, 72.8275),
    "cst":           (18.9398, 72.8354),
    "bkc":           (19.0630, 72.8684),
    "powai":         (19.1176, 72.9060),
    "prabhadevi":    (19.0128, 72.8286),
    "mahim":         (19.0404, 72.8399),
    "sion":          (19.0437, 72.8641),
    "matunga":       (19.0275, 72.8526),
    "khar":          (19.0713, 72.8354),
    "vile parle":    (19.0963, 72.8484),
    "goregaon":      (19.1550, 72.8494),
    "malad":         (19.1862, 72.8484),
    "kandivali":     (19.2048, 72.8523),
    "borivali":      (19.2288, 72.8570),
}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two coordinates."""
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class GraphBuilder:
    def __init__(self, place_name: str = "Bandra, Mumbai, India", use_cache: bool = True):
        self.place_name = place_name
        self.use_cache = use_cache
        self.G: Optional[nx.MultiDiGraph] = None
        self._pyg_cache: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def build_network_graph(self) -> nx.MultiDiGraph:
        """Fetch the OSM drive graph, enrich with speed/travel-time, and cache."""

        # 1. Try project-local pickle cache
        if os.path.exists(_GRAPH_CACHE_PATH):
            try:
                with open(_GRAPH_CACHE_PATH, "rb") as f:
                    cached = pickle.load(f)
                self.G = cached["graph"]
                logger.info(
                    f"Graph loaded from local cache ({_GRAPH_CACHE_PATH}): "
                    f"{len(self.G.nodes)} nodes, {len(self.G.edges)} edges."
                )
                self._ensure_enriched()
                return self.G
            except Exception as e:
                logger.warning(f"Local graph cache corrupt, rebuilding: {e}")

        # 2. Network fetch
        logger.info(f"Building/Fetching OSM graph for {self.place_name} via network…")
        try:
            ox.settings.log_console = False
            ox.settings.use_cache = self.use_cache
            self.G = ox.graph_from_place(self.place_name, network_type="drive", simplify=True)
            logger.info(f"Graph fetched: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges.")
        except Exception as e:
            logger.error(f"Failed to fetch OSM graph: {e}")
            logger.info("Falling back to small synthetic grid network for demonstration.")
            self.G = nx.grid_2d_graph(5, 5, create_using=nx.MultiDiGraph)
            mapping = {n: f"{n[0]}_{n[1]}" for n in self.G.nodes}
            self.G = nx.relabel_nodes(self.G, mapping)

        # 3. Enrich with speed & travel-time edge attributes
        self._ensure_enriched()

        # 4. Persist
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            with open(_GRAPH_CACHE_PATH, "wb") as f:
                pickle.dump({"graph": self.G, "place_name": self.place_name}, f)
            logger.info(f"Graph persisted to local cache: {_GRAPH_CACHE_PATH}")
        except Exception as e:
            logger.warning(f"Could not persist graph cache: {e}")

        return self.G

    def _ensure_enriched(self) -> None:
        """Add speed_kph and travel_time (seconds) to every edge if missing."""
        if self.G is None:
            return
        sample = next(iter(self.G.edges(data=True)), None)
        if sample and "travel_time" in sample[2]:
            return  # already enriched
        try:
            self.G = ox.add_edge_speeds(self.G)
            self.G = ox.add_edge_travel_times(self.G)
            logger.info("Enriched graph edges with speed_kph and travel_time.")
        except Exception as e:
            logger.warning(f"Could not enrich graph edges (adding fallback): {e}")
            for u, v, k, d in self.G.edges(keys=True, data=True):
                length = d.get("length", 100.0)
                if isinstance(length, list):
                    length = length[0]
                d.setdefault("speed_kph", 30.0)
                d.setdefault("travel_time", float(length) / (30.0 / 3.6))

    # ------------------------------------------------------------------
    # Geocoding helpers
    # ------------------------------------------------------------------
    def geocode_place(self, place_name: str) -> Tuple[float, float]:
        """Return (lat, lon) for a place name.  Tries the local landmark
        dict first, then falls back to OSMnx geocoding."""
        key = place_name.replace("_", " ").strip().lower()
        if key in MUMBAI_LANDMARKS:
            return MUMBAI_LANDMARKS[key]
        try:
            return ox.geocode(f"{place_name}, Mumbai, India")
        except Exception:
            # Ultimate fallback: centroid of loaded graph
            if self.G is not None:
                lats = [d.get("y", 0) for _, d in self.G.nodes(data=True)]
                lons = [d.get("x", 0) for _, d in self.G.nodes(data=True)]
                if lats and lons:
                    return (sum(lats) / len(lats), sum(lons) / len(lons))
            return (19.0596, 72.8295)  # Bandra default

    def find_nearest_node(self, lat: float, lon: float) -> Any:
        """Return the OSM node ID closest to the given coordinates."""
        if self.G is None:
            self.build_network_graph()
        return ox.distance.nearest_nodes(self.G, X=lon, Y=lat)

    # ------------------------------------------------------------------
    # Real route computation
    # ------------------------------------------------------------------
    def compute_route(
        self,
        origin: str,
        destination: str,
        weight: str = "travel_time",
    ) -> Dict[str, Any]:
        """Compute a real shortest-path route on the OSM graph.

        Returns dict with path, distance_m, travel_time_s, coords, etc.
        """
        if self.G is None:
            self.build_network_graph()

        orig_lat, orig_lon = self.geocode_place(origin)
        dest_lat, dest_lon = self.geocode_place(destination)

        orig_node = self.find_nearest_node(orig_lat, orig_lon)
        dest_node = self.find_nearest_node(dest_lat, dest_lon)

        try:
            path = nx.shortest_path(self.G, orig_node, dest_node, weight=weight)
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {origin} and {destination}")
            straight_dist = _haversine(orig_lat, orig_lon, dest_lat, dest_lon)
            return {
                "path": [],
                "distance_m": straight_dist,
                "travel_time_s": straight_dist / (20.0 / 3.6),  # assume 20 km/h
                "travel_time_min": (straight_dist / (20.0 / 3.6)) / 60.0,
                "coords": [(orig_lat, orig_lon), (dest_lat, dest_lon)],
                "num_segments": 0,
                "origin_node": orig_node,
                "dest_node": dest_node,
            }

        # Sum edge attributes along the path
        total_dist = 0.0
        total_time = 0.0
        coords: List[Tuple[float, float]] = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # MultiDiGraph: pick the shortest edge between u and v
            edge_data = min(self.G[u][v].values(), key=lambda d: d.get(weight, 1e9))
            length = edge_data.get("length", 100.0)
            if isinstance(length, list):
                length = length[0]
            total_dist += float(length)
            total_time += float(edge_data.get("travel_time", float(length) / (30.0 / 3.6)))

        # Collect lat/lon of each node in the path
        for n in path:
            nd = self.G.nodes[n]
            coords.append((nd.get("y", 0.0), nd.get("x", 0.0)))

        return {
            "path": path,
            "distance_m": round(total_dist, 1),
            "travel_time_s": round(total_time, 1),
            "travel_time_min": round(total_time / 60.0, 2),
            "coords": coords,
            "num_segments": len(path) - 1,
            "origin_node": orig_node,
            "dest_node": dest_node,
        }

    def compute_alternative_routes(
        self, origin: str, destination: str
    ) -> List[Dict[str, Any]]:
        """Return two real routes: fastest (by travel_time) and shortest (by length)."""
        fastest = self.compute_route(origin, destination, weight="travel_time")
        shortest = self.compute_route(origin, destination, weight="length")
        fastest["route_label"] = "fastest"
        shortest["route_label"] = "shortest"
        return [fastest, shortest]

    # ------------------------------------------------------------------
    # Traffic estimation (replaces old Google Maps mock)
    # ------------------------------------------------------------------
    def estimate_area_congestion(self) -> Dict[str, Any]:
        """Compute real congestion metrics from OSM graph edge attributes."""
        if self.G is None:
            self.build_network_graph()

        speeds = []
        for _, _, d in self.G.edges(data=True):
            spd = d.get("speed_kph", 30.0)
            if isinstance(spd, list):
                spd = spd[0]
            try:
                speeds.append(float(spd))
            except (TypeError, ValueError):
                speeds.append(30.0)

        avg_speed = sum(speeds) / max(len(speeds), 1)
        # Mumbai city average is ~15-20 km/h peaks, 30-40 off-peak
        congestion = max(0.0, min(1.0, 1.0 - (avg_speed / 60.0)))

        # Hotspots = nodes with highest degree (busiest intersections)
        degrees = sorted(self.G.degree(), key=lambda x: x[1], reverse=True)
        hotspots = [str(n) for n, _ in degrees[:5]]

        return {
            "avg_congestion": round(congestion, 3),
            "avg_speed_kph": round(avg_speed, 1),
            "total_segments": self.G.number_of_edges(),
            "hotspots": hotspots,
        }

    # ------------------------------------------------------------------
    # PyTorch Geometric conversion (unchanged logic)
    # ------------------------------------------------------------------
    def get_pytorch_geometric_data(self) -> Tuple[Data, dict]:
        if self._pyg_cache is not None:
            return self._pyg_cache

        if self.G is None:
            self.build_network_graph()

        node_list = list(self.G.nodes())
        node_idx_map = {n_id: idx for idx, n_id in enumerate(node_list)}
        idx_node_map = {idx: n_id for n_id, idx in node_idx_map.items()}

        edge_index_u, edge_index_v, edge_weights = [], [], []

        for u, v, data in self.G.edges(data=True):
            edge_index_u.append(node_idx_map[u])
            edge_index_v.append(node_idx_map[v])
            length = data.get("length", 10.0)
            if isinstance(length, list):
                length = length[0]
            edge_weights.append(1.0 / max(1.0, float(length)))

        edge_index = torch.tensor([edge_index_u, edge_index_v], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        x = torch.ones((len(node_list), 1), dtype=torch.float32)

        pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self._pyg_cache = (pyg_data, idx_node_map)
        return pyg_data, idx_node_map


# ---------------------------------------------------------------------------
# Module-level singleton — all services share one graph instance
# ---------------------------------------------------------------------------
_shared_builder: Optional[GraphBuilder] = None


def get_graph_builder() -> GraphBuilder:
    """Return a lazily-initialised, application-wide GraphBuilder."""
    global _shared_builder
    if _shared_builder is None:
        _shared_builder = GraphBuilder("Bandra, Mumbai, India", use_cache=True)
        _shared_builder.build_network_graph()
    return _shared_builder
