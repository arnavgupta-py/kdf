from pydantic import BaseModel, Field
from typing import List, Optional

class DepartureOption(BaseModel):
    departure_time: float = Field(..., description="Unix timestamp for departure")
    expected_travel_time: float = Field(..., description="Expected travel time in minutes")
    travel_time_variance: float = Field(..., description="Variance of travel time in minutes^2")
    arrival_probability: float = Field(..., description="Probability of arriving before the deadline (0.0 to 1.0)")
    route_id: Optional[str] = Field(None, description="Recommended route ID or summary")

class OptimizationRequest(BaseModel):
    origin: str = Field(..., description="Origin node ID or address")
    destination: str = Field(..., description="Destination node ID or address")
    deadline: float = Field(..., description="Unix timestamp for the hard arrival deadline")
    planning_horizon_hours: int = Field(24, description="Planning horizon in hours")
    user_id: Optional[str] = Field(None, description="Optional User ID to load personal routing behaviour weights")

class OptimizationResponse(BaseModel):
    options: List[DepartureOption] = Field(..., description="Pareto frontier of departure candidates")

class ParkingAlternative(BaseModel):
    location_id: str = Field(..., description="Node ID or descriptor of the parking spot")
    occupancy_probability: float = Field(..., description="Probability of this spot being full (0.0 to 1.0)")
    walking_distance_meters: float = Field(..., description="Distance to final destination in meters")
    cost: float = Field(..., description="Cost metric (e.g., USD or abstract units)")

class ParkingRequest(BaseModel):
    destination: str = Field(..., description="Destination node ID where parking is needed")
    arrival_time: float = Field(..., description="Estimated arrival time as Unix epoch")
    zone_type: str = Field("commercial", description="Land-use zone type (e.g., commercial, residential, transit)")

class ParkingResponse(BaseModel):
    primary_occupancy_probability: float = Field(..., description="Occupancy probability of the requested destination")
    alternatives: List[ParkingAlternative] = Field(..., description="Alternative suggestions if primary is likely full")
