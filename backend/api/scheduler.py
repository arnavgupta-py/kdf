from fastapi import APIRouter, HTTPException
from backend.schemas.optimization import OptimizationRequest, OptimizationResponse, ParkingRequest, ParkingResponse
from backend.services.optimiser import DepartureOptimiser
from backend.services.parking import ParkingIntelligence
from loguru import logger
import time

router = APIRouter()

optimiser_service = DepartureOptimiser(step_minutes=5)
parking_service = ParkingIntelligence()

@router.post("/departure-optimiser", response_model=OptimizationResponse)
async def get_departure_frontier(request: OptimizationRequest):
    """
    Computes a Pareto frontier array of departure options reflecting the optimal 
    trade-off between expected travel time and hard-arrival probability.
    """
    if request.deadline < time.time():
        raise HTTPException(status_code=400, detail="Departure deadline must be in the future.")
        
    try:
        options = optimiser_service.compute_pareto_frontier(
            origin=request.origin,
            destination=request.destination,
            deadline=request.deadline,
            hours=request.planning_horizon_hours
        )
        return OptimizationResponse(options=options)
    except Exception as e:
        logger.error(f"Failed to compute optimized departures: {e}")
        raise HTTPException(status_code=500, detail="Optimization failed.")

@router.post("/parking-intel", response_model=ParkingResponse)
async def get_parking_occupancy(request: ParkingRequest):
    """
    Generates a continuous probabilistic metric modeling target congestion patterns 
    and alternatives for real-time commercial/residential usage points.
    """
    if request.arrival_time < time.time() - 3600:
        raise HTTPException(status_code=400, detail="Parking prediction cannot be for past events.")
        
    try:
        parking_data = parking_service.evaluate_parking(
            destination=request.destination,
            arrival_time=request.arrival_time,
            zone_type=request.zone_type.lower()
        )
        return ParkingResponse(
            primary_occupancy_probability=parking_data["primary_occupancy_probability"],
            alternatives=parking_data["alternatives"]
        )
    except Exception as e:
        logger.error(f"Failed to evaluate parking metrics: {e}")
        raise HTTPException(status_code=500, detail="Parking intelligence failed.")
