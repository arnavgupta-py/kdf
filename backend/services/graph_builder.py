import osmnx as ox
import networkx as nx
import torch
from torch_geometric.data import Data
from loguru import logger
from typing import Tuple

class GraphBuilder:
    def __init__(self, place_name: str = "Bandra, Mumbai, India", use_cache: bool = True):
        self.place_name = place_name
        self.use_cache = use_cache
        self.G = None

    def build_network_graph(self) -> nx.MultiDiGraph:
        """
        Retrieves the OpenStreetMap graph for the given place.
        """
        logger.info(f"Building/Fetching OSM graph for {self.place_name}")
        try:
            # We configure OSMnx appropriately
            ox.settings.log_console = False
            ox.settings.use_cache = self.use_cache
            
            # For demonstration we pull a driveable network
            # This can be heavy, but osmnx caches it if use_cache is True
            self.G = ox.graph_from_place(self.place_name, network_type="drive", simplify=True)
            logger.info(f"Graph instantiated with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges.")
            return self.G
        except Exception as e:
            logger.error(f"Failed to fetch OSM graph: {e}")
            logger.info("Falling back to small synthetic grid network for demonstration.")
            self.G = nx.grid_2d_graph(5, 5, create_using=nx.MultiDiGraph)
            # Re-label nodes to strings rather than tuples so they act like real osmid strings
            mapping = {n: f"{n[0]}_{n[1]}" for n in self.G.nodes}
            self.G = nx.relabel_nodes(self.G, mapping)
            return self.G

    def get_pytorch_geometric_data(self) -> Tuple[Data, dict]:
        """
        Converts the NetworkX structure into a PyTorch Geometric Data object
        with edge indices and simple constant weights for ST-GNN processing.
        Returns the data and a mapping from PyTorch index to OSM node ID.
        """
        if self.G is None:
            self.build_network_graph()

        node_list = list(self.G.nodes())
        node_idx_map = {n_id: idx for idx, n_id in enumerate(node_list)}
        idx_node_map = {idx: n_id for n_id, idx in node_idx_map.items()}

        edge_index_u = []
        edge_index_v = []
        edge_weights = []

        for u, v, data in self.G.edges(data=True):
            edge_index_u.append(node_idx_map[u])
            edge_index_v.append(node_idx_map[v])
            
            # Simple inverse of length as a weight proxy, defaulting to 1.0
            length = data.get("length", 10.0)
            if isinstance(length, list):
                length = length[0]
            weight = 1.0 / max(1.0, float(length))
            edge_weights.append(weight)

        edge_index = torch.tensor([edge_index_u, edge_index_v], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        
        # We can create a basic node feature matrix (e.g., all 1s as initial state)
        x = torch.ones((len(node_list), 1), dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data, idx_node_map
