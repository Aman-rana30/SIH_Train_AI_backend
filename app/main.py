"""
Main FastAPI application for train traffic control system.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager
import os

from app.core.config import settings
from app.api.routes import schedule, metrics, websocket, map_simulation, settings as settings_routes
from app.db.session import engine
from app.db.base import Base

# Import all models to ensure they are registered with SQLAlchemy
from app.models.user import User
from app.models.settings import UserSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting train traffic control system...")

    # Only create database tables if not in test mode
    if not os.getenv("TESTING"):
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {str(e)}")
    else:
        logger.info("Skipping database table creation in test mode")

    yield

    logger.info("Shutting down train traffic control system...")


# Create FastAPI application
app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="AI-powered train traffic control decision-support system",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Global exception on {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "path": str(request.url),
            "timestamp": time.time()
        }
    )


# Include API routes
app.include_router(
    schedule.router,
    prefix=f"{settings.api_v1_prefix}/schedule",
    tags=["schedule"]
)

app.include_router(
    metrics.router,
    prefix=f"{settings.api_v1_prefix}/metrics",
    tags=["metrics"]
)

app.include_router(
    websocket.router,
    prefix="/ws",
    tags=["websocket"]
)

app.include_router(
    map_simulation.router,
    prefix=f"{settings.api_v1_prefix}/map",
    tags=["map-simulation"]
)

app.include_router(
    settings_routes.router,
    prefix=f"{settings.api_v1_prefix}/settings",
    tags=["settings"]
)


# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.version
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Train Traffic Control Decision-Support System",
        "version": settings.version,
        "docs_url": "/docs",
        "health_url": "/health",
        "api_prefix": settings.api_v1_prefix
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
