from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from backend.core.config import settings
from backend.core.logger import setup_logging
from loguru import logger
import time

setup_logging()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="City-scale Route & Occupancy Network with Optimised Scheduling",
    version="1.0.0",
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

# Included external routes
from backend.api.users import router as users_router
app.include_router(users_router, prefix=f"{settings.API_V1_STR}/users", tags=["users"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
