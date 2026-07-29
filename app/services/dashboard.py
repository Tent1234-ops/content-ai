from collections import Counter
from datetime import datetime, timedelta
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
from app.services.persistence import save_google_trends, save_tiktok_trends, save_youtube_trends
from app.services.recommendation import build_dataset_profiles, compare_dataset_profiles
from app.services.trends import get_google_trending, get_tiktok_trending, get_youtube_trending


def _safe_scalar(query, default: int = 0) -> int:
    value = query.scalar()
    if value is None:
        return default
    return int(value)


def _topic_keywords_from_text(text: str) -> list[str]:
    tokens = filter_tokens(tokenize_text(text))
    return [token for token in tokens if len(token) > 2]


def _normalize_trend_item(item: object) -> dict:
    if isinstance(item, dict):
        return item
    if hasattr(item, "dict") and callable(getattr(item, "dict")):
        return item.dict()
    return {}


def _build_topic_scores_from_trends(items: list[object]) -> Counter[str]:
    scores: Counter[str] = Counter()
    for item in items:
        data = _normalize_trend_item(item)
        title = str(data.get("title", "")) or ""
        category = str(data.get("category", "")) or ""
        trend_weight = float(data.get("trend_score", 1.0) or 1.0)
        text = " ".join([title, category]).strip()
        if not text:
            continue
        if trend_weight <= 0:
            trend_weight = 1.0
        for token in _topic_keywords_from_text(text):
            scores[token] += min(trend_weight / 1000.0, 10.0)
    return scores


def _build_item_novelty_score(item: object, historical_counts: Counter[str]) -> tuple[float, int]:
    data = _normalize_trend_item(item)
    title = str(data.get("title", "")) or ""
    category = str(data.get("category", "")) or ""
    text = " ".join([title, category]).strip()

    if not text:
        return 0.0, 0

    keywords = _topic_keywords_from_text(text)
    if not keywords:
        return 0.0, 0

    novelty = 0.0
    total_history = 0
    for keyword in set(keywords):
        history_count = historical_counts.get(keyword, 0)
        novelty += 1.0 / (1.0 + history_count)
        total_history += history_count

    return novelty, total_history


def _rank_priority_items(items: list[object], historical_counts: Counter[str], limit: int = 10) -> list[Dict[str, object]]:
    ranked: list[Dict[str, object]] = []
    for item in items:
        data = _normalize_trend_item(item)
        title = str(data.get("title", "")) or ""
        if not title:
            continue
        category = str(data.get("category", "")) if data.get("category") is not None else None
        source_platform = data.get("source") or data.get("source_platform")
        trend_score = float(data.get("trend_score", 0.0) or 0.0)
        novelty, total_history = _build_item_novelty_score(item, historical_counts)
        views = int(data.get("views", 0) or 0)
        likes = int(data.get("likes", 0) or 0)
        comments = int(data.get("comments", 0) or 0)
        published_at = data.get("published_at")
        video_url = data.get("video_url") or None

        priority_score = trend_score + (novelty * 120.0) + min(views / 20000.0, 10.0)
        ranked.append(
            {
                "title": title,
                "category": category,
                "source_platform": source_platform,
                "video_url": video_url,
                "trend_score": trend_score,
                "priority_score": round(priority_score, 3),
                "novelty_score": round(novelty, 3),
                "views": views,
                "likes": likes,
                "comments": comments,
                "published_at": published_at.isoformat() if hasattr(published_at, "isoformat") else str(published_at) if published_at else None,
                "history_count": int(total_history),
            }
        )
    ranked.sort(key=lambda item: (-item["priority_score"], -item["trend_score"], item["title"].lower()))
    return [
        {
            "title": item["title"],
            "category": item["category"],
            "source_platform": item["source_platform"],
            "video_url": item["video_url"],
            "trend_score": item["trend_score"],
            "priority_score": item["priority_score"],
            "novelty_score": item["novelty_score"],
            "views": item["views"],
            "likes": item["likes"],
            "comments": item["comments"],
            "published_at": item["published_at"],
        }
        for item in ranked[:limit]
    ]


def _build_historical_topic_counts(db: Session, days: int = 30) -> Counter[str]:
    cutoff = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(DatasetContent.title, DatasetContent.transcript, DatasetContent.category)
        .filter(DatasetContent.created_at >= cutoff)
        .all()
    )
    counts: Counter[str] = Counter()
    for title, transcript, category in rows:
        text = " ".join(part for part in [title or "", transcript or "", category or ""] if part).strip()
        for token in _topic_keywords_from_text(text):
            counts[token] += 1
    return counts


def _rank_priority_topics(live_scores: Counter[str], historical_counts: Counter[str], limit: int = 10) -> list[Dict[str, object]]:
    ranked: list[Dict[str, object]] = []
    for keyword, score in live_scores.items():
        history = historical_counts.get(keyword, 0)
        priority_score = score + (history * 0.05)
        ranked.append({"keyword": keyword, "score": round(priority_score, 3), "history_count": int(history)})
    ranked.sort(key=lambda item: (-item["score"], item["keyword"]))
    return [{"keyword": item["keyword"], "score": item["score"]} for item in ranked[:limit]]


def _rank_emerging_topics(live_scores: Counter[str], historical_counts: Counter[str], limit: int = 10) -> list[Dict[str, object]]:
    ranked: list[Dict[str, object]] = []
    for keyword, score in live_scores.items():
        history = historical_counts.get(keyword, 0)
        emerging_score = score * (2.0 / (1.0 + history))
        ranked.append({"keyword": keyword, "score": round(emerging_score, 3), "history_count": int(history)})
    ranked.sort(key=lambda item: (-item["score"], item["history_count"], item["keyword"]))
    return [{"keyword": item["keyword"], "score": item["score"]} for item in ranked[:limit]]


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
    priority_topics: List[Dict[str, object]] = []
    emerging_topics: List[Dict[str, object]] = []
    priority_items: List[Dict[str, object]] = []

    # Sync live YouTube trends into the saved dataset content whenever possible.
    trend_source_mode, trend_items = get_youtube_trending(region=region, limit=trend_limit, mode=trend_mode)
    if trend_source_mode == "live" and trend_items:
        try:
            save_youtube_trends(db=db, items=trend_items, user_id=current_user.user_id, source_mode=trend_source_mode)
        except Exception:
            pass

    # Sync live Google Trends into the saved dataset content whenever possible.
    google_source_mode, google_items = get_google_trending(region=region, limit=trend_limit, mode=trend_mode)
    if google_source_mode == "live" and google_items:
        try:
            save_google_trends(db=db, items=google_items, user_id=current_user.user_id, source_mode=google_source_mode)
        except Exception:
            pass

    # Sync live TikTok trends into the saved dataset content whenever possible.
    tiktok_source_mode, tiktok_items = get_tiktok_trending(region=region, limit=trend_limit, mode=trend_mode)
    if tiktok_source_mode == "live" and tiktok_items:
        try:
            save_tiktok_trends(db=db, items=tiktok_items, user_id=current_user.user_id, source_mode=tiktok_source_mode)
        except Exception:
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
        tiktok_profiles = build_dataset_profiles(db, source_prefix="tiktok", limit=100)
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
            {
                "source": "tiktok",
                "dataset_count": _safe_scalar(
                    db.query(func.count(DatasetContent.dataset_id)).filter(DatasetContent.source_platform.like("tiktok%"))
                ),
                "profile_count": len(tiktok_profiles),
                "domains": [profile["domain"] for profile in tiktok_profiles],
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

        live_topic_scores = _build_topic_scores_from_trends(trend_items + google_items + tiktok_items)
        historical_counts = _build_historical_topic_counts(db)
        priority_topics = _rank_priority_topics(live_topic_scores, historical_counts, limit=10)
        emerging_topics = _rank_emerging_topics(live_topic_scores, historical_counts, limit=10)
        priority_items = _rank_priority_items(trend_items + google_items + tiktok_items, historical_counts, limit=trend_limit)
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
        "priority_topics": priority_topics,
        "emerging_topics": emerging_topics,
        "priority_items": priority_items,
        "youtube_trends": {
            "mode": trend_source_mode,
            "region": region,
            "total": len(trend_items),
            "items": trend_items,
        },
        "google_trends": {
            "mode": google_source_mode,
            "region": region,
            "total": len(google_items),
            "items": google_items,
        },
        "tiktok_trends": {
            "mode": tiktok_source_mode,
            "region": region,
            "total": len(tiktok_items),
            "items": tiktok_items,
        },
    }


def build_dashboard_topic_insights(
    db: Session,
    current_user: User,
    region: str,
    trend_mode: str,
    trend_limit: int,
) -> Dict[str, object]:
    trend_source_mode, trend_items = get_youtube_trending(region=region, limit=trend_limit, mode=trend_mode)
    if trend_source_mode == "live" and trend_items:
        try:
            save_youtube_trends(db=db, items=trend_items, user_id=current_user.user_id, source_mode=trend_source_mode)
        except Exception:
            pass

    google_source_mode, google_items = get_google_trending(region=region, limit=trend_limit, mode=trend_mode)
    if google_source_mode == "live" and google_items:
        try:
            save_google_trends(db=db, items=google_items, user_id=current_user.user_id, source_mode=google_source_mode)
        except Exception:
            pass

    tiktok_source_mode, tiktok_items = get_tiktok_trending(region=region, limit=trend_limit, mode=trend_mode)
    if tiktok_source_mode == "live" and tiktok_items:
        try:
            save_tiktok_trends(db=db, items=tiktok_items, user_id=current_user.user_id, source_mode=tiktok_source_mode)
        except Exception:
            pass

    live_topic_scores = _build_topic_scores_from_trends(trend_items + google_items + tiktok_items)
    historical_counts = _build_historical_topic_counts(db)
    return {
        "priority_items": _rank_priority_items(trend_items + google_items + tiktok_items, historical_counts, limit=trend_limit),
        "emerging_topics": _rank_emerging_topics(live_topic_scores, historical_counts, limit=trend_limit),
        "youtube_trends": {
            "mode": trend_source_mode,
            "region": region,
            "total": len(trend_items),
            "items": trend_items,
        },
        "google_trends": {
            "mode": google_source_mode,
            "region": region,
            "total": len(google_items),
            "items": google_items,
        },
        "tiktok_trends": {
            "mode": tiktok_source_mode,
            "region": region,
            "total": len(tiktok_items),
            "items": tiktok_items,
        },
    }
