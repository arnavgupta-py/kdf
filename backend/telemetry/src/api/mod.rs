use axum::{
    extract::{State, Json},
    response::IntoResponse,
    http::StatusCode,
    routing::post,
    Router,
};
use serde::Deserialize;
use tokio::time::{Instant, Duration};
use crate::state::{OccupancyMap, DedupMap, SegmentData};
use tracing::debug;

#[derive(Deserialize)]
pub struct ProbeEvent {
    pub segment_id: u32,
    pub speed_kmh: f32,
    pub is_commercial: bool,
    pub device_id: Option<String>,
}

#[derive(Clone)]
pub struct AppState {
    pub occupancy: OccupancyMap,
    pub dedup: DedupMap,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/ingest", post(handle_ingest))
        .with_state(state)
}

async fn handle_ingest(
    State(state): State<AppState>,
    Json(payload): Json<Vec<ProbeEvent>>,
) -> impl IntoResponse {
    debug!("Received {} probe events", payload.len());
    
    let now = Instant::now();
    let dedup_window = Duration::from_secs(30);
    
    for event in payload {
        // Enforce 30-second deduplication per device
        if let Some(device_id) = event.device_id {
            if !device_id.is_empty() {
                let mut is_duplicate = false;
                if let Some(mut last_seen) = state.dedup.get_mut(&device_id) {
                    if now.duration_since(*last_seen) < dedup_window {
                        is_duplicate = true;
                    } else {
                        *last_seen = now;
                    }
                } else {
                    state.dedup.insert(device_id, now);
                }
                
                if is_duplicate {
                    continue; // Skip processing this duplicate payload
                }
            }
        }
        
        let mut entry = state.occupancy.entry(event.segment_id).or_insert(SegmentData {
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
