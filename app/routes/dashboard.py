from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.database.db import get_db
from app.database.models import User
from app.schemas.dashboard import DashboardOverviewResponse
from app.services.dashboard import build_dashboard_overview

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/overview", response_model=DashboardOverviewResponse)
def dashboard_overview(
    region: str = Query(default=settings.youtube_region, min_length=2, max_length=2),
    trend_mode: str = Query(default="auto", pattern="^(auto|mock|live)$"),
    trend_limit: int = Query(default=5, ge=1, le=10),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return DashboardOverviewResponse.model_validate(
        build_dashboard_overview(
            db=db,
            current_user=current_user,
            region=region.upper(),
            trend_mode=trend_mode,
            trend_limit=trend_limit,
        )
    )
