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

@router.get("/settings", response_class=HTMLResponse)
async def settings_view(request: Request):
    """Returns the partial HTML for User Preferences."""
    return templates.TemplateResponse("pages/settings.html", {"request": request})
