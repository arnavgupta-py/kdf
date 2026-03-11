import asyncio
import time
import sys
# Add parent directory to path to allow imports
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.optimiser import DepartureOptimiser

async def main():
    optimiser = DepartureOptimiser(step_minutes=15)
    # Deadline is 2 hours from now
    deadline = time.time() + 7200
    frontier = await optimiser.compute_pareto_frontier("bandra", "andheri", deadline, hours=4)
    for option in frontier:
        print(f"Depart: {option.departure_time}, ETA: {option.expected_travel_time}, Prob: {option.arrival_probability}")

if __name__ == "__main__":
    asyncio.run(main())
