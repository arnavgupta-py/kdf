# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Phase 4: Causal Traffic Engine**
  - Designed probabilistic `STGNNModel` using PyTorch Geometric integrating spatial GCN convolutions and GRU for temporal forecasting.
  - Implemented `CausalInferenceEngine` leveraging `dowhy` to identify and quantify systemic causal triggers of congestion like accidents and weather.
  - Built `GraphBuilder` fetching real street networks on-the-fly (`Bandra, Mumbai`) through `osmnx` and processing it into PyTorch `Data`.
  - Stood up `/api/v1/forecast/predict` probabilistic endpoint exposing causal traffic intel with dynamic horizons and bounded stochastic confidence intervals.
- **Phase 3: Rust Telemetry Engine**
  - Configured Tokio and Axum workspace in Rust for high-throughput HTTP ingestion routing.
  - Built thread-safe OccupancyMap using `dashmap` to track live vehicle densities segment-wise.
  - Implemented `get_occupancy_map` via `tonic` gRPC to expose internal states to Python ML pipeline.
  - Generated and finalized Python proxy bindings leveraging `grpcio`.
- **Phase 2: Core Data Models & APIs**
  - Defined SQLAlchemy schemas for `User` and `Journey` tracking.
  - Defined PyDantic request/response validation schemas.
  - Implemented foundational CRUD logic for creating users and updating preferences.
  - Setup basic API endpoints for User registration mounted to the FastAPI router.
- **Phase 1: Project Setup & Infrastructure**
  - Initialized Python environment and installed backend dependencies via `uv`.
  - Initialized Rust cargo workspace for telemetry at `backend/telemetry/`.
  - Established project folder structure (`backend/`, `frontend/`).
  - Implemented `backend/core/config.py` using `pydantic-settings`.
  - Setup DB clients (`backend/db/sqlite.py`, `backend/db/qdrant.py`).
  - Added centralized JSON/Stdout logging in `backend/core/logger.py` with `loguru`.
  - Built `frontend/templates/base.html` with Tailwind CSS, Alpine.js, and HTMX.
