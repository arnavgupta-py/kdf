import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from backend.schemas.forecast import ForecastResponse, ForecastNode, CausalFactor
from backend.services.graph_builder import GraphBuilder
from backend.services.causal_inference import CausalInferenceEngine
import torch
from loguru import logger

router = APIRouter()
causal_engine = CausalInferenceEngine()

# Initialize the network graph once
try:
    graph_builder = GraphBuilder("Bandra, Mumbai, India", use_cache=True)
    # We can fetch the PyTorch Data instantly in memory as a mock initialization
    pyg_data, node_mapping = graph_builder.get_pytorch_geometric_data()
    logger.info(f"Graph initialized with {pyg_data.num_nodes} nodes.")
except Exception as e:
    logger.error(f"Failed to initialize graph immediately: {e}")
    pyg_data = None
    node_mapping = {}

@router.get("/predict", response_model=ForecastResponse)
async def get_forecast(
    horizon_minutes: int = Query(90, description="Forecast horizon in minutes"),
    node_ids: Optional[str] = Query(None, description="Comma-separated list of OSM node IDs")
):
    """
    Returns probabilistic traffic state forecasts at the requested horizon
    along with the primary causal triggers derived from the ST-GNN and DoWhy engines.
    """
    if horizon_minutes > 120:
        raise HTTPException(status_code=400, detail="Forecast horizon cannot exceed 120 minutes.")

    # Determine nodes to forecast
    requested_nodes = []
    if node_ids:
        requested_nodes = node_ids.split(",")
    elif pyg_data is not None:
        # Default to predicting 10 central nodes in the subset representation
        requested_nodes = [node_mapping[i] for i in range(min(10, pyg_data.num_nodes))]
    else:
        requested_nodes = ["node_1", "node_2", "bridge_east_3"]

    response_nodes = []
    
    # In a fully deployed setup, we would run:
    # mu, sigma = stgnn_model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
    # Here, we generate realistic synthetic forecasts based on node identities.
    
    for count, n_id in enumerate(requested_nodes):
        # We simulate the ST-GNN's probabilistic output (mu: 0.0=free, 1.0=standstill)
        # Using hash variation for realistic but stable differences per node ID.
        hasher = hash(n_id) % 100
        mu = 0.3 + (hasher / 200.0) # Varies from 0.3 to 0.8
        
        # Adding horizon noise: further ahead implies higher uncertainty (larger sigma)
        sigma = 0.05 + (horizon_minutes / 1200.0)
        
        # Bound limits logically
        lower_bound = max(0.0, mu - 1.96 * sigma)
        upper_bound = min(1.0, mu + 1.96 * sigma)

        # Retrieve causal insights (mock functionality mapping the logic)
        raw_factors = causal_engine.get_mocked_causal_factors(mu)
        schema_factors = [CausalFactor(**factor) for factor in raw_factors]

        response_node = ForecastNode(
            node_id=str(n_id),
            expected_congestion=round(mu, 3),
            confidence_interval=[round(lower_bound, 3), round(upper_bound, 3)],
            causal_factors=schema_factors
        )
        response_nodes.append(response_node)

    return ForecastResponse(
        timestamp=time.time(),
        horizon_minutes=horizon_minutes,
        nodes=response_nodes
    )
