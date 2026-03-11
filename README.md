# CRONOS
**City-scale Route & Occupancy Network with Optimised Scheduling**

*An AI-powered predictive urban navigation system for Mumbai metropolitan commuters.*

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        CRONOS                                │
│                                                              │
│  ┌─────────────────────────┐   ┌────────────────────────┐   │
│  │  Python ML Layer         │   │  Rust Telemetry Engine │   │
│  │  (FastAPI · PyTorch)     │◄──│  (Tokio · Axum)        │   │
│  │                          │   │                        │   │
│  │  ST-GNN Forecasting      │   │  GPS Probe Ingestion   │   │
│  │  Causal Inference (DoWhy)│   │  10k events/sec        │   │
│  │  Pareto Departure Opt.   │   │  OccupancyMap (DashMap)│   │
│  │  Parking Intelligence    │   │  gRPC Interface        │   │
│  └─────────────────────────┘   └────────────────────────┘   │
│              │                                               │
│  ┌─────────────────────────┐                                 │
│  │  Frontend (Jinja + HTMX)│                                 │
│  │  Trip Planner            │                                 │
│  │  Live Telemetry View     │                                 │
│  │  User Preferences        │                                 │
│  └─────────────────────────┘                                 │
└──────────────────────────────────────────────────────────────┘
```

## Setup

### Prerequisites
- Python ≥ 3.14 with [`uv`](https://github.com/astral-sh/uv)
- Rust toolchain (for the telemetry engine)

### Install dependencies
```bash
uv sync
```

### Pre-warm the graph cache (recommended — eliminates cold-start)
```bash
uv run python scripts/cache_graph.py
```

### Train the ST-GNN model
```bash
uv run python backend/train_stgnn.py
```

### Start the API server
```bash
uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the Rust telemetry engine (separate terminal)
```bash
cd backend/telemetry && cargo run --release
```
The Rust engine exposes:
- **HTTP** on `0.0.0.0:8080` — GPS probe ingestion
- **gRPC** on `0.0.0.0:50051` — occupancy map query

---

## Core Components

| Component | Tech | Description |
|---|---|---|
| **ST-GNN** | PyTorch Geometric | GCNConv × 2 + GRU, probabilistic (μ, σ) output |
| **Causal Engine** | DoWhy | Backdoor linear regression over Weather/Accidents/TimeOfDay |
| **Departure Optimiser** | SciPy + NumPy | Pareto frontier via Normal CDF; **1 ST-GNN inference per call** |
| **Parking Intelligence** | NumPy | Gaussian kernel occupancy over land-use profiles |
| **Graph Cache** | OSMnx + Pickle | Bandra MMR network persisted to `cache/bandra_graph.pkl` |
| **Telemetry Engine** | Rust (Tokio + Axum + DashMap) | Lock-free occupancy map, gRPC export |
| **Frontend** | Jinja2 + HTMX + Alpine.js | Glassmorphism dashboard, real-time HTMX partials |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/forecast/predict` | Probabilistic ST-GNN traffic forecast |
| `POST` | `/api/v1/scheduler/departure-optimiser` | Pareto frontier of departure windows |
| `POST` | `/api/v1/scheduler/parking-intel` | Parking occupancy estimate + alternatives |
| `POST` | `/api/v1/users/` | Create user |
| `PUT` | `/api/v1/users/me/preferences` | Update routing preferences |

---

*VCET CRONOS — Horizon 1.0 — March 2026*
