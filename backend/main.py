from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from backend.core.config import settings
from backend.core.logger import setup_logging
from loguru import logger
import time

setup_logging()


def _seed_database() -> None:
    """Ensure tables exist and the demo guest user is present."""
    from backend.db.sqlite import engine, SessionLocal, Base
    # Import all models so Base.metadata is fully populated before create_all
    import backend.models.user  # noqa: F401
    import backend.models.journey  # noqa: F401
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        from backend.services.user_service import get_user_by_email, create_user
        from backend.schemas.user import UserCreate
        if not get_user_by_email(db, email="guest@cronos.com"):
            create_user(
                db,
                UserCreate(
                    email="guest@cronos.com",
                    full_name="Demo User",
                    password="cronos-demo",
                ),
            )
            logger.info("Seeded demo user: guest@cronos.com")
        else:
            logger.info("Demo user already exists, skipping seed.")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before serving requests."""
    _seed_database()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="City-scale Route & Occupancy Network with Optimised Scheduling",
    version="1.0.0",
    lifespan=lifespan,
)

# Static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Templates
templates = Jinja2Templates(directory="frontend/templates")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - HTTP {response.status_code} - {process_time:.4f}s"
    )
    return response

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("base.html", {"request": request, "title": "CRONOS Dashboard"})

@app.get("/pages/{page_name}", response_class=HTMLResponse)
async def deep_link_pages(request: Request, page_name: str):
    """Deep links into UI pages. HTMX pushes these URLs."""
    return templates.TemplateResponse("base.html", {"request": request, "title": f"CRONOS | {page_name.capitalize()}"})

# Included external routes
from backend.api.users import router as users_router
from backend.api.forecast import router as forecast_router
from backend.api.scheduler import router as scheduler_router
from backend.api.frontend import router as frontend_router

app.include_router(users_router, prefix=f"{settings.API_V1_STR}/users", tags=["users"])
app.include_router(forecast_router, prefix=f"{settings.API_V1_STR}/forecast", tags=["forecast"])
app.include_router(scheduler_router, prefix=f"{settings.API_V1_STR}/scheduler", tags=["scheduler"])
app.include_router(frontend_router, prefix=f"{settings.API_V1_STR}/frontend", tags=["frontend"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
