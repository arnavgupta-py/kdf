# CRONOS

**City-scale Route & Occupancy Network with Optimised Scheduling**

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135.1-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C.svg?style=flat&logo=PyTorch)](https://pytorch.org)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg?logo=Rust&logoColor=white)](https://www.rust-lang.org)

An **AI-Driven Predictive Urban Navigation and Mobility Optimization System** designed to solve the increasingly complex challenges of metropolitan commutes. 

---

## 📖 The Problem

Urban commuters in metropolitan cities (like Mumbai) face significant challenges due to:
- **Unpredictable Traffic Congestion:** Existing navigation platforms primarily rely on real-time traffic monitoring and short-term routing based only on the "now". They react to congestion rather than anticipating it.
- **Inefficient Departure Planning:** Users struggle to plan optimal departure times for upcoming trips without knowing the probability of on-time arrival given the inherent delay variances.
- **Parking Unavailability:** Commuters frequently spend excess time circling for parking spaces blindly, which increases travel delays, carbon emissions, and reduces overall mobility efficiency.
- **Generic Routing:** Standard route planners often ignore individual traveler nuances (e.g., toll aversion, highway preference, or variance tolerance).

## 💡 The Solution

**CRONOS** proposes an intelligent transportation solution that provides proactive travel assistance. Rather than just reacting, CRONOS leverages Spatio-Temporal Graph Neural Networks (ST-GNN), Causal Inference, and Reinforcement Learning to look into the future. By integrating four powerful AI capabilities into a unified architecture, the system provides context-aware recommendations tailored to each user.

---

## 🚀 Core Components

### 1. Predictive Traffic Forecasting (ST-GNN & Causal Inference)
* **How it works:** Reconstructs road networks using **OSMnx** & **PyTorch Geometric**. It runs a trained **Spatio-Temporal Graph Neural Network (ST-GNN)** to forecast local congestion indices up to 120 minutes into the future.
* **Causal Inference Engine:** Powered by **DoWhy**, it performs backdoor-adjustment & linear regression over real-world data distributions (Peak commute windows, Monsoon weather datasets, and Road Accident probabilities) to explain *why* a specific segment is predicted to be congested, not just *that* it will be congested.

### 2. Smart Departure Planning (Pareto Frontier Optimisation)
* **How it works:** Instead of a single "leave now" instruction, the **Departure Optimiser** generates a complete **Pareto frontier** of departure options.
* **Math in Action:** It evaluates real OSM shortest-path routes mixed with the ST-GNN's predicted congestion variance. It calculates the statistical probability of arriving before the user's deadline across 288 departure slots, trading off expected travel time against variance/reliability.

### 3. Parking Intelligence
* **How it works:** Queries real-world OpenStreetMap (OSM) data for parking amenities via the Overpass API near the user's geocoded destination. 
* **Probabilistic Occupancy:** Applies analytical land-use occupancy models (Commercial, Residential, Transit) with Gaussian variance layers to calculate expected availability of parking spots, weighting results based on walking distance to the actual destination.

### 4. Personalised Learning (PPO Reinforcement Learning)
* **How it works:** Every completed journey acts as a feedback loop. Using the actual vs. predicted arrival time errors as the reward signal, a **Proximal Policy Optimization (PPO)** Reinforcement Learning agent continuously adapts to each user's unique profile.
* **Result:** Dynamically infers hidden user preferences such as toll aversion, variance tolerance, and highway preference, tweaking the router engine over time without the user ever needing to touch a settings toggle.

---

## 🛠 Technology Stack

### Backend & AI 
* **Framework:** [FastAPI](https://fastapi.tiangolo.com) - High performance, asynchronous Python web framework.
* **Deep Learning Framework:** [PyTorch](https://pytorch.org) & [PyTorch Geometric](https://pytorch-geometric.readthedocs.io) for complex graph-based network learning.
* **Causal Inference:** [DoWhy](https://microsoft.github.io/dowhy/) for discovering causal attribution of traffic triggers.
* **Geospatial & Routing:** [OSMnx](https://osmnx.readthedocs.io/en/stable/) / OpenAPI / NetworkX.
* **Database & ORM:** [SQLite](https://sqlite.org) managed asynchronously via [SQLAlchemy](https://www.sqlalchemy.org/).

### Telemetry / Edge 
* **Microservice:** High-speed telemetry ingestion powered by **Rust** (Cargo-based deployment).

### Frontend
* **Stack:** Pure HTML/CSS powered by [Jinja2](https://jinja.palletsprojects.com/) templating and **HTMX** for dynamic partial-page updates matching an SPA-like feel while retaining total server authority.

---

## ⚙️ Architecture Workflow

1. **Ingest & Mapping:** The Graph Builder constructs the underlying topological graph structure mapping nodes to physical spatial latitude/longitudes using NetworkX.
2. **Forecast Trigger:** The API queries the PyTorch ST-GNN `predict` layer while simultaneously piping real-time telemetry from the Rust-based edge daemon. Let the engine establish upper and lower prediction bounds via structural DoWhy models.
3. **Optimiser Handshake:** User constraints (Destination, Deadline) combined with the forecasted network topology are passed to the Departure Optimiser.
4. **Learning Evaluation:** Upon trip completion, the schedule adherence delta pushes into the PPO Agent, tuning the future loss parameters for `user_id`.

---

## 🏗 Setup & Installation

### Prerequisites
- [uv](https://docs.astral.sh/uv/) / Python 3.14+
- [Rust](https://www.rust-lang.org/tools/install) >= 1.70 (for the telemetry engine)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd kdf
   ```

2. **Run the Rust Telemetry Daemon:**
   ```bash
   cd backend/telemetry
   cargo run --release
   ```
   *(Keep this terminal window running)*

3. **Install dependencies and launch the Backend/Frontend:**
   ```bash
   # From the project root
   uv install
   uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
   ```

4. **Access the Dashboard:**
   Open a web browser and navigate to `http://127.0.0.1:8000`

> Note: The system automatically seeds a demo guest user (`guest@cronos.com`) on startup for immediate testing.

---

## 📄 License
This original project, CRONOS, is provided as an active prototype for Advanced Agentic Hackathons.
