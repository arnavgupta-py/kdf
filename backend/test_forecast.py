import asyncio
import sys
import os

# Ensure the backend module is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.api.forecast import get_forecast

async def run_test():
    print("Executing forecast...")
    res = await get_forecast(horizon_minutes=60, node_ids=None)
    for node in res.nodes:
        print(f"Node {node.node_id}: expected={node.expected_congestion}, val={node.confidence_interval}, causal={' '.join([c.factor for c in node.causal_factors])}")

if __name__ == "__main__":
    asyncio.run(run_test())
