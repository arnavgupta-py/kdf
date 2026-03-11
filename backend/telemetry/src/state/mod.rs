use dashmap::DashMap;
use std::sync::Arc;
use tokio::time::Instant;

#[derive(Debug, Clone)]
pub struct SegmentData {
    pub occupancy_ratio: f32,
    pub active_vehicles: u32,
}

// Global thread-safe map holding the real-time simulation state
pub type OccupancyMap = Arc<DashMap<u32, SegmentData>>;

// Global thread-safe map holding deduplication timestamps per device_id
pub type DedupMap = Arc<DashMap<String, Instant>>;

pub fn create_occupancy_map() -> OccupancyMap {
    Arc::new(DashMap::new())
}

pub fn create_dedup_map() -> DedupMap {
    Arc::new(DashMap::new())
}
