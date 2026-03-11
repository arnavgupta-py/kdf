from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

@router.get("/trip", response_class=HTMLResponse)
async def trip_planner_view(request: Request):
    """Returns the partial HTML for the Trip Planner."""
    return templates.TemplateResponse("pages/trip.html", {"request": request})

@router.get("/live", response_class=HTMLResponse)
async def live_telemetry_view(request: Request):
    """Returns the partial HTML for the Live Telemetry."""
    return templates.TemplateResponse("pages/live.html", {"request": request})

from backend.db.sqlite import get_db
from backend.services.user_service import get_user_by_email, parse_user_preferences
from fastapi import Depends
from sqlalchemy.orm import Session

@router.get("/settings", response_class=HTMLResponse)
async def settings_view(request: Request, db: Session = Depends(get_db)):
    """Returns the partial HTML for User Preferences."""
    # MVP: Hardcoded user ID to demonstrate config binding
    user = get_user_by_email(db, email="guest@cronos.com")
    prefs = parse_user_preferences(user) if user else {}
    
    return templates.TemplateResponse("pages/settings.html", {
        "request": request,
        "preferences": prefs
    })

from backend.schemas.optimization import OptimizationRequest
from backend.services.optimiser import DepartureOptimiser
import time

# Create a module-level optimizer instance for the UI
ui_optimiser = DepartureOptimiser(step_minutes=5)

@router.post("/trip/calculate", response_class=HTMLResponse)
async def ui_trip_calculate(request: Request):
    form_data = await request.form()
    origin = form_data.get("origin", "Bandra_West")
    destination = form_data.get("destination", "Lower_Parel")
    
    # Calculate dummy future deadline for demonstration based on the mode selected
    mode = form_data.get("mode", "fastest")
    deadline = time.time() + (3600 * 5) # 5 hours from now
    
    options = await ui_optimiser.compute_pareto_frontier(
        origin=origin,
        destination=destination,
        deadline=deadline,
        hours=12
    )
    
    return templates.TemplateResponse("partials/trip_results.html", {
        "request": request, 
        "options": options,
        "origin": origin,
        "destination": destination
    })
