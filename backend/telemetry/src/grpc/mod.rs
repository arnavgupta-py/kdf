use std::time::{SystemTime, UNIX_EPOCH};
use tonic::{Request, Response, Status};

use crate::state::OccupancyMap;
pub use telemetry_proto::telemetry_service_server::{TelemetryService, TelemetryServiceServer};
pub use telemetry_proto::{OccupancyRequest, OccupancyResponse, SegmentState};

pub mod telemetry_proto {
    // The name 'telemetry' comes from the package name in telemetry.proto
    tonic::include_proto!("telemetry");
}

pub struct OccupancyService {
    pub map: OccupancyMap,
}

#[tonic::async_trait]
impl TelemetryService for OccupancyService {
    async fn get_occupancy_map(
        &self,
        _request: Request<OccupancyRequest>,
    ) -> Result<Response<OccupancyResponse>, Status> {
        let mut segments = Vec::new();
        
        for entry in self.map.iter() {
            let segment_id = *entry.key();
            let data = entry.value();
            segments.push(SegmentState {
                segment_id,
                occupancy_ratio: data.occupancy_ratio,
                active_vehicles: data.active_vehicles,
            });
        }
        
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
            
        Ok(Response::new(OccupancyResponse {
            timestamp_ms,
            segments,
        }))
    }
}
