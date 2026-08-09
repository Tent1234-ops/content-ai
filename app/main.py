import asyncio
import importlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.database.db import Base, DB_BOOTSTRAP_ERROR, SessionLocal, engine
from app.database.migrations import archive_phase10_notification_tables
from app.routes import admin, admin_scanner, analyze, auth, classification, clustering, contents, dashboard, datasets, nlp, recommendation, trends, notifications, follows
from app.services.trending_fetcher import start_trending_fetcher, stop_trending_fetcher
from app.services.live_trend_snapshots import get_live_provider_health

app = FastAPI(title=settings.app_name, version=settings.app_version)

def preload_ai_models():
    modules = [
        "models.speech_to_text",
        "models.keyword_ai",
        "models.keyword_ranker",
        "models.semantic_keyword",
        "models.blip_caption",
        "models.scene_detect",
        "models.frame_extract",
    ]
    print("Starting AI model preload...", flush=True)
    for module_name in modules:
        try:
            print(f"Preloading {module_name}...", flush=True)
            importlib.import_module(module_name)
        except Exception as exc:
            print(f"Warning: failed to preload {module_name}: {exc}", flush=True)
    print("AI model preload complete.", flush=True)


@app.on_event("startup")
async def startup_event():
    # Skipping AI model preload for faster testing
    # await asyncio.to_thread(preload_ai_models)
    start_trending_fetcher()


@app.on_event("shutdown")
async def shutdown_event():
    stop_trending_fetcher()


# Configure CORS for Flutter Web & Mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Flutter Web dev
        "http://127.0.0.1:3000",      # Flutter Web dev (localhost)
        "http://localhost:5000",      # Flutter Web alt port
        "http://127.0.0.1:5000",      # Flutter Web alt port
        "http://localhost",           # Flutter Web production
        "http://127.0.0.1",           # Flutter Web production
        "*",                          # Allow all origins (dev only - restrict in production)
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep startup simple for Phase 1 so the API is ready after boot.
db_init_status = "ok"
phase11_migration_status = {}
try:
    phase11_migration_status = archive_phase10_notification_tables(engine)
    Base.metadata.create_all(bind=engine)
except (SQLAlchemyError, Exception) as exc:
    db_init_status = f"degraded: {exc.__class__.__name__}"
if DB_BOOTSTRAP_ERROR:
    db_init_status = f"degraded: {DB_BOOTSTRAP_ERROR}"

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(classification.router)
app.include_router(clustering.router)
app.include_router(contents.router)
app.include_router(dashboard.router)
app.include_router(datasets.router)
app.include_router(nlp.router)
app.include_router(recommendation.router)
app.include_router(trends.router)
app.include_router(analyze.router)
app.include_router(notifications.router)
app.include_router(follows.router)
app.include_router(admin_scanner.router)
# jobs router for background job status
from app.routes import jobs
app.include_router(jobs.router)


@app.get("/", tags=["health"])
def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "status": "ok",
        "db_init": db_init_status,
    }


@app.get("/health", tags=["health"])
def health():
    db = SessionLocal()
    database = {
        "status": "ok",
        "init": db_init_status,
        "phase11_archived_tables": phase11_migration_status,
    }
    live_trends = {
        "run_id": None,
        "snapshot_status": "unavailable",
        "generated_at": None,
        "providers": {},
    }
    try:
        db.execute(text("SELECT 1"))
        live_trends = get_live_provider_health(db, region=settings.youtube_region)
    except Exception as exc:
        database = {
            "status": "error",
            "init": db_init_status,
            "phase11_archived_tables": phase11_migration_status,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    finally:
        db.close()

    provider_states = [
        str(item.get("status") or "pending")
        for item in live_trends.get("providers", {}).values()
    ]
    if database["status"] != "ok":
        status = "unhealthy"
    elif not provider_states or any(item in {"error", "pending"} for item in provider_states):
        status = "degraded"
    else:
        status = "ok"

    return {
        "status": status,
        "database": database,
        "live_trends": live_trends,
    }
