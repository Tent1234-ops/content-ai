from collections import Counter
from typing import Dict, List

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.models import (
    AnalysisResult,
    Cluster,
    ClusterMembership,
    ClusterRun,
    DatasetContent,
    SystemLog,
    User,
    UserContent,
)
from app.services.nlp import filter_tokens, tokenize_text
from app.services.persistence import save_youtube_trends
from app.services.recommendation import build_dataset_profiles, compare_dataset_profiles
from app.services.trends import get_youtube_trending


def _safe_scalar(query, default: int = 0) -> int:
    value = query.scalar()
    if value is None:
        return default
    return int(value)


def build_dashboard_overview(
    db: Session,
    current_user: User,
    region: str,
    trend_mode: str,
    trend_limit: int,
) -> Dict[str, object]:
    db_status = "ok"
    db_error = None

    metrics = {
        "total_users": 0,
        "active_users": 0,
        "total_user_contents": 0,
        "total_analysis_results": 0,
        "total_clusters": 0,
        "total_cluster_runs": 0,
        "total_cluster_memberships": 0,
        "total_dataset_contents": 0,
        "total_system_logs": 0,
        "my_contents": 0,
        "my_analysis_results": 0,
    }

    category_distribution: List[Dict[str, object]] = []
    cluster_distribution: List[Dict[str, object]] = []
    top_trends: List[Dict[str, object]] = []
    top_keywords: List[Dict[str, object]] = []
    source_distribution: List[Dict[str, object]] = []
    platform_summaries: List[Dict[str, object]] = []
    platform_comparison: List[Dict[str, object]] = []

    # Sync live YouTube trends into the saved dataset content whenever possible.
    trend_source_mode, trend_items = get_youtube_trending(region=region, limit=trend_limit, mode=trend_mode)
    if trend_source_mode == "live" and trend_items:
        try:
            save_youtube_trends(db=db, items=trend_items, user_id=current_user.user_id, source_mode=trend_source_mode)
        except Exception:
            # We do not want dashboard rendering to fail because sync failed.
            pass

    try:
        metrics["total_users"] = _safe_scalar(db.query(func.count(User.user_id)))
        metrics["active_users"] = _safe_scalar(db.query(func.count(User.user_id)).filter(User.is_active.is_(True)))
        metrics["total_user_contents"] = _safe_scalar(db.query(func.count(UserContent.content_id)))
        metrics["total_analysis_results"] = _safe_scalar(db.query(func.count(AnalysisResult.result_id)))
        metrics["total_clusters"] = _safe_scalar(db.query(func.count(Cluster.cluster_id)))
        metrics["total_cluster_runs"] = _safe_scalar(db.query(func.count(ClusterRun.run_id)))
        metrics["total_cluster_memberships"] = _safe_scalar(db.query(func.count(ClusterMembership.membership_id)))
        metrics["total_dataset_contents"] = _safe_scalar(db.query(func.count(DatasetContent.dataset_id)))
        metrics["total_system_logs"] = _safe_scalar(db.query(func.count(SystemLog.log_id)))
        metrics["my_contents"] = _safe_scalar(
            db.query(func.count(UserContent.content_id)).filter(UserContent.user_id == current_user.user_id)
        )
        metrics["my_analysis_results"] = _safe_scalar(
            db.query(func.count(AnalysisResult.result_id))
            .join(UserContent, UserContent.content_id == AnalysisResult.content_id)
            .filter(UserContent.user_id == current_user.user_id)
        )

        raw_categories = (
            db.query(DatasetContent.category, func.count(DatasetContent.dataset_id))
            .group_by(DatasetContent.category)
            .order_by(func.count(DatasetContent.dataset_id).desc())
            .limit(10)
            .all()
        )
        category_distribution = [
            {"category": category or "uncategorized", "count": int(count)}
            for category, count in raw_categories
        ]
        top_categories = list(category_distribution)

        raw_clusters = (
            db.query(Cluster.cluster_name, func.count(ClusterMembership.membership_id))
            .outerjoin(ClusterMembership, ClusterMembership.cluster_id == Cluster.cluster_id)
            .group_by(Cluster.cluster_id, Cluster.cluster_name)
            .order_by(func.count(ClusterMembership.membership_id).desc())
            .limit(10)
            .all()
        )
        cluster_distribution = [
            {"cluster_name": cluster_name, "count": int(count)}
            for cluster_name, count in raw_clusters
        ]

        raw_top_trends = (
            db.query(DatasetContent)
            .order_by(DatasetContent.trend_score.desc(), DatasetContent.views.desc(), DatasetContent.created_at.desc())
            .limit(10)
            .all()
        )
        top_trends = [
            {
                "dataset_id": row.dataset_id,
                "title": row.title,
                "category": row.category,
                "source_platform": row.source_platform,
                "video_url": row.video_url,
                "views": int(row.views or 0),
                "likes": int(row.likes or 0),
                "comments": int(row.comments or 0),
                "trend_score": float(row.trend_score or 0),
                "published_at": row.published_at.isoformat() if row.published_at else None,
            }
            for row in raw_top_trends
        ]

        raw_sources = (
            db.query(DatasetContent.source_platform, func.count(DatasetContent.dataset_id))
            .group_by(DatasetContent.source_platform)
            .order_by(func.count(DatasetContent.dataset_id).desc())
            .all()
        )
        source_distribution = [
            {"source_platform": source_platform or "unknown", "count": int(count)}
            for source_platform, count in raw_sources
        ]

        keyword_counts: Counter[str] = Counter()
        for row in raw_top_trends:
            text = " ".join(part for part in [row.title or "", row.transcript or "", row.category or ""] if part).strip()
            if not text:
                continue
            keyword_counts.update(filter_tokens(tokenize_text(text)))
        top_keywords = [
            {"keyword": keyword, "count": int(count)}
            for keyword, count in keyword_counts.most_common(10)
        ]

        youtube_profiles = build_dataset_profiles(db, source_prefix="youtube", limit=100)
        google_profiles = build_dataset_profiles(db, source_prefix="google", limit=100)
        platform_summaries = [
            {
                "source": "youtube",
                "dataset_count": _safe_scalar(
                    db.query(func.count(DatasetContent.dataset_id)).filter(DatasetContent.source_platform.like("youtube%"))
                ),
                "profile_count": len(youtube_profiles),
                "domains": [profile["domain"] for profile in youtube_profiles],
            },
            {
                "source": "google",
                "dataset_count": _safe_scalar(
                    db.query(func.count(DatasetContent.dataset_id)).filter(DatasetContent.source_platform.like("google%"))
                ),
                "profile_count": len(google_profiles),
                "domains": [profile["domain"] for profile in google_profiles],
            },
        ]

        comparison = compare_dataset_profiles(db, left_source="youtube", right_source="google", limit=100)
        platform_comparison = [
            {
                "domain": item["domain"],
                "youtube_sample_size": int(item["left_sample_size"]),
                "google_sample_size": int(item["right_sample_size"]),
                "youtube_duration": item["left_duration"]["recommended_range"],
                "google_duration": item["right_duration"]["recommended_range"],
            }
            for item in comparison["comparisons"]
        ]
    except SQLAlchemyError as exc:
        db_status = "degraded"
        db_error = exc.__class__.__name__
        top_categories = []
    else:
        top_categories = list(category_distribution)

    return {
        "db_status": db_status,
        "db_error": db_error,
        "user_role": current_user.role,
        "metrics": metrics,
        "category_distribution": category_distribution,
        "cluster_distribution": cluster_distribution,
        "top_trends": top_trends,
        "top_categories": top_categories,
        "top_keywords": top_keywords,
        "source_distribution": source_distribution,
        "platform_summaries": platform_summaries,
        "platform_comparison": platform_comparison,
        "youtube_trends": {
            "mode": trend_source_mode,
            "region": region,
            "total": len(trend_items),
            "items": trend_items,
        },
    }
