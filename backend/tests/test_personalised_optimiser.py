import pytest
import math
from backend.services.optimiser import DepartureOptimiser
from backend.schemas.optimization import DepartureOption
import time

import asyncio

def test_personalised_optimiser():
    asyncio.run(run_personalised_optimiser())

async def run_personalised_optimiser():
    optimiser = DepartureOptimiser(step_minutes=15)
    deadline = time.time() + 7200 # 2 hours from now
    
    # 1. Baseline testing without preferences
    default_frontier = await optimiser.compute_pareto_frontier(
        origin="bandra", destination="andheri", deadline=deadline, hours=4,
        user_preferences=None
    )
    assert len(default_frontier) > 0

    # 2. Testing with extreme toll aversion (should shift towards alternative route)
    toll_averse_prefs = {
        "toll_aversion": 1.0, 
        "variance_tolerance": 0.5, 
        "highway_preference": 0.1
    }
    averse_frontier = await optimiser.compute_pareto_frontier(
        origin="bandra", destination="andheri", deadline=deadline, hours=4,
        user_preferences=toll_averse_prefs
    )
    
    # Check that alt route is favored due to penalising tolls
    # Either the size changes or the composition of route_ids changes
    
    # 3. Testing with high highway preference (should shift towards fastest route)
    highway_prefs = {
        "toll_aversion": 0.1, 
        "variance_tolerance": 0.9, 
        "highway_preference": 1.0
    }
    highway_frontier = await optimiser.compute_pareto_frontier(
        origin="bandra", destination="andheri", deadline=deadline, hours=4,
        user_preferences=highway_prefs
    )
    
    assert len(averse_frontier) > 0
    assert len(highway_frontier) > 0
    # There should be a shift in edges or total size of the pareto frontier
    # We just ensure it runs and successfully applies weighting to the routes without crashing.
