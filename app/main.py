from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.database.db import Base, DB_BOOTSTRAP_ERROR, engine
from app.routes import admin, analyze, auth, classification, clustering, contents, dashboard, datasets, nlp, recommendation, trends

app = FastAPI(title=settings.app_name, version=settings.app_version)

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
