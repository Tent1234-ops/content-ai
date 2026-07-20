import asyncio
import importlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.database.db import Base, DB_BOOTSTRAP_ERROR, engine
from app.routes import admin, analyze, auth, classification, clustering, contents, dashboard, datasets, nlp, recommendation, trends

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
    await asyncio.to_thread(preload_ai_models)
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
try:
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


@app.get("/", tags=["health"])
def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "status": "ok",
        "db_init": db_init_status,
    }
