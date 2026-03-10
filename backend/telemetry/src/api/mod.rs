use axum::{
    extract::{State, Json},
    response::IntoResponse,
    http::StatusCode,
    routing::post,
    Router,
};
use serde::Deserialize;
use crate::state::{OccupancyMap, SegmentData};
use tracing::debug;

#[derive(Deserialize)]
pub struct ProbeEvent {
    pub segment_id: u32,
    pub speed_kmh: f32,
    pub is_commercial: bool,
}

pub fn create_router(map: OccupancyMap) -> Router {
    Router::new()
        .route("/ingest", post(handle_ingest))
        .with_state(map)
}

async fn handle_ingest(
    State(map): State<OccupancyMap>,
    Json(payload): Json<Vec<ProbeEvent>>,
) -> impl IntoResponse {
    // In a real system, we would perform complex filtering, deduplication, 
    // and EWMA (Exponential Weighted Moving Average) updates here.
    // For MVP, we simply tally active vehicles and naively update occupancy ratio.
    
    debug!("Received {} probe events", payload.len());
    
    for event in payload {
        let mut entry = map.entry(event.segment_id).or_insert(SegmentData {
            occupancy_ratio: 0.0,
            active_vehicles: 0,
        });
        
        entry.active_vehicles += 1;
        
        // Naive heuristic: higher volume = higher occupancy ratio up to 1.0
        let new_ratio = (entry.active_vehicles as f32 / 100.0).min(1.0);
        entry.occupancy_ratio = new_ratio;
    }

    StatusCode::ACCEPTED
}
