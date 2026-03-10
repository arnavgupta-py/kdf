use dashmap::DashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SegmentData {
    pub occupancy_ratio: f32,
    pub active_vehicles: u32,
}

// Global thread-safe map holding the real-time simulation state
pub type OccupancyMap = Arc<DashMap<u32, SegmentData>>;

pub fn create_occupancy_map() -> OccupancyMap {
    Arc::new(DashMap::new())
}
