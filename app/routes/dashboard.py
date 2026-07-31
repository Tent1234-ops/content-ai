from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.database.db import get_db
from app.database.models import User, UserContent
from app.schemas.dashboard import (
    DashboardEmergingTopicsResponse,
    DashboardOverviewResponse,
    DashboardRefreshResponse,
)
from app.services.dashboard import build_dashboard_overview, build_dashboard_topic_insights
from app.services.trending_fetcher import RateLimitedError, trigger_trending_refresh

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/overview", response_model=DashboardOverviewResponse)
def dashboard_overview(
    region: str = Query(default=settings.youtube_region, min_length=2, max_length=2),
    trend_mode: str = Query(default="auto", pattern="^(auto|mock|live)$"),
    trend_limit: int = Query(default=5, ge=1, le=20),
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


@router.get("/summary")
def dashboard_summary(
    region: str = Query(default=settings.youtube_region, min_length=2, max_length=2),
    trend_mode: str = Query(default="auto", pattern="^(auto|mock|live)$"),
    trend_limit: int = Query(default=3, ge=1, le=20),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Lightweight dashboard summary for frontend cards. Respects runtime.dashboard_live flag and caches result for 60s."""
    from app.runtime import get as runtime_get
    from app.services.simple_cache import get as cache_get, invalidate_prefix as cache_invalidate_prefix, set as cache_set

    # Feature-flag: force mock mode if dashboard_live disabled
    if not runtime_get("dashboard_live", False):
        effective_mode = "mock"
    else:
        effective_mode = trend_mode

    cache_key = f"dashboard_summary:{current_user.user_id}:{region}:{effective_mode}:{trend_limit}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    # Build a lightweight summary for dashboard cards so the dashboard does not wait on heavy dataset profiling.
    from app.services.dashboard import build_dashboard_summary

    summary = build_dashboard_summary(
        db=db,
        current_user=current_user,
        region=region.upper(),
        trend_mode=effective_mode,
        trend_limit=trend_limit,
    )

    # Top trends are already limited by requested trend_limit in summary builder.
    top_trends = summary.get("top_trends", [])

    # Recent analyses: latest 3 UserContent entries for user
    recent_analyses = []
    try:
        rows = (
            db.query(UserContent)
            .filter(UserContent.user_id == current_user.user_id)
            .order_by(UserContent.created_at.desc())
            .limit(3)
            .all()
        )
        for r in rows:
            recent_analyses.append({
                "content_id": r.content_id,
                "title": r.title,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "video_url": r.video_url,
            })
    except Exception:
        recent_analyses = []

    quick_recommendations = summary.get("quick_recommendations", [])
    recommended_duration = summary.get("recommended_duration")

    result = {
        "top_trends": top_trends,
        "quick_recommendations": quick_recommendations,
        "recommended_duration": recommended_duration,
        "recent_analyses": recent_analyses,
        # include a minimal set of overview fields to avoid front-end breaking
        "metrics": summary.get("metrics", {}),
        "platform_summaries": summary.get("platform_summaries", []),
        "source_distribution": summary.get("source_distribution", []),
        "youtube_trends": summary.get("youtube_trends", {}),
        "google_trends": summary.get("google_trends", {}),
        "tiktok_trends": summary.get("tiktok_trends", {}),
        "generated_at": datetime.utcnow().isoformat(),
    }

    # Cache for 60 seconds
    cache_set(cache_key, result, ttl_seconds=60)
    return result


@router.post("/refresh", response_model=DashboardRefreshResponse)
def dashboard_refresh(
    current_user: User = Depends(get_current_user),
):
    from app.services.simple_cache import invalidate_prefix as cache_invalidate_prefix
    try:
        result = trigger_trending_refresh(user_id=current_user.user_id)
        cache_invalidate_prefix(f"dashboard_summary:{current_user.user_id}:")
        return result
    except RateLimitedError as exc:
        raise HTTPException(status_code=429, detail=str(exc))


@router.get("/emerging-topics", response_model=DashboardEmergingTopicsResponse)
def dashboard_emerging_topics(
    region: str = Query(default=settings.youtube_region, min_length=2, max_length=2),
    trend_mode: str = Query(default="auto", pattern="^(auto|mock|live)$"),
    trend_limit: int = Query(default=5, ge=1, le=20),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return DashboardEmergingTopicsResponse.model_validate(
        build_dashboard_topic_insights(
            db=db,
            current_user=current_user,
            region=region.upper(),
            trend_mode=trend_mode,
            trend_limit=trend_limit,
        )
    )
