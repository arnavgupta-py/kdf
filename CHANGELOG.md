# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
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
