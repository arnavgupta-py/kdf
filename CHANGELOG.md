# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Phase 1: Project Setup & Infrastructure**
  - Initialized Python environment and installed backend dependencies via `uv`.
  - Initialized Rust cargo workspace for telemetry at `backend/telemetry/`.
  - Established project folder structure (`backend/`, `frontend/`).
  - Implemented `backend/core/config.py` using `pydantic-settings`.
  - Setup DB clients (`backend/db/sqlite.py`, `backend/db/qdrant.py`).
  - Added centralized JSON/Stdout logging in `backend/core/logger.py` with `loguru`.
  - Built `frontend/templates/base.html` with Tailwind CSS, Alpine.js, and HTMX.
