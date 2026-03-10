from pydantic import BaseModel, Field
from typing import List

class CausalFactor(BaseModel):
    factor: str = Field(..., description="The cause, e.g., 'Weather' or 'Accident'")
    impact_score: float = Field(..., description="Estimated impact score typically from 0.0 to 1.0")
    description: str = Field(..., description="Human-readable rationale for this factor")

class ForecastNode(BaseModel):
    node_id: str = Field(..., description="The ID of the graph node (intersection)")
    expected_congestion: float = Field(..., description="Probability or severity of congestion between 0.0 to 1.0")
    confidence_interval: List[float] = Field(..., description="Tuple of [lower_bound, upper_bound] for confidence")
    causal_factors: List[CausalFactor] = []

class ForecastResponse(BaseModel):
    timestamp: float = Field(..., description="Time of forecast generation (Unix epoch)")
    horizon_minutes: int = Field(..., description="How many minutes into the future this forecast predicts")
    nodes: List[ForecastNode] = Field(..., description="List of node forecasts in the network graph")
