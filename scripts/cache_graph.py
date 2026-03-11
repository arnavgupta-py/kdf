#!/usr/bin/env python3
"""
cache_graph.py — Pre-warm the Bandra road network cache.

Run once before starting the server so the first request sees
<10 ms graph load instead of a ~3-5 s OSMnx HTTP fetch.

Usage:
    uv run python scripts/cache_graph.py
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.graph_builder import GraphBuilder, _GRAPH_CACHE_PATH
from loguru import logger


def main() -> None:
    if os.path.exists(_GRAPH_CACHE_PATH):
        logger.info(f"Cache already exists at {_GRAPH_CACHE_PATH} — skipping fetch.")
        logger.info("Delete the file and re-run to force a refresh.")
        return

    logger.info("Building Bandra road network graph and persisting to cache…")
    builder = GraphBuilder(place_name="Bandra, Mumbai, India", use_cache=True)
    G = builder.build_network_graph()
    data, idx_node_map = builder.get_pytorch_geometric_data()

    logger.success(
        f"Graph cached successfully.\n"
        f"  Nodes : {G.number_of_nodes()}\n"
        f"  Edges : {G.number_of_edges()}\n"
        f"  PyG nodes : {data.num_nodes}\n"
        f"  Cache path: {_GRAPH_CACHE_PATH}"
    )


if __name__ == "__main__":
    main()
