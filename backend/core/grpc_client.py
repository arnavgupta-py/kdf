import grpc
from backend.core import telemetry_pb2
from backend.core import telemetry_pb2_grpc

class TelemetryClient:
    def __init__(self, host='localhost', port=50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = telemetry_pb2_grpc.TelemetryServiceStub(self.channel)

    def get_occupancy_map(self):
        req = telemetry_pb2.OccupancyRequest()
        try:
            response = self.stub.GetOccupancyMap(req)
            # Convert protobuf response to native dictionary for Python API
            return {
                "timestamp_ms": response.timestamp_ms,
                "segments": [
                    {
                        "segment_id": s.segment_id,
                        "occupancy_ratio": s.occupancy_ratio,
                        "active_vehicles": s.active_vehicles
                    } for s in response.segments
                ]
            }
        except grpc.RpcError as e:
            print(f"gRPC call failed: {e.details()}")
            return None

# Singleton instance
telemetry_client = TelemetryClient()
