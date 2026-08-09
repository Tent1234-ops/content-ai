from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Mapping

from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.db import SessionLocal
from app.database.models import TrendSnapshotItem, TrendSnapshotRun
from app.services.trends import get_google_trending, get_tiktok_trending, get_youtube_trending


PLATFORMS = ("youtube", "google", "tiktok")
ProviderFetcher = Callable[[str, int], tuple[str, List[object]]]

_refresh_lock = threading.Lock()


def _item_to_dict(item: object) -> Dict[str, object]:
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump") and callable(getattr(item, "model_dump")):
        return item.model_dump()
    if hasattr(item, "dict") and callable(getattr(item, "dict")):
        return item.dict()
    return {}


def _stable_key(platform: str, data: Dict[str, object]) -> str:
    title = str(data.get("title") or data.get("query") or "").strip().lower()
    url = str(data.get("video_url") or data.get("url") or "").strip().lower()
    return hashlib.sha1(f"{platform}|{url or title}".encode("utf-8")).hexdigest()


def normalize_live_item(platform: str, item: object) -> Dict[str, object]:
    data = _item_to_dict(item)
    title = str(data.get("title") or data.get("query") or "").strip()
    category = str(data.get("category") or data.get("domain") or "general").strip() or "general"
    source_platform = str(data.get("source_platform") or data.get("source") or platform).strip()
    published_at = data.get("published_at")
    if hasattr(published_at, "isoformat"):
        published_at = published_at.isoformat()
    elif published_at is not None:
        published_at = str(published_at)

    views = int(data.get("views") or 0)
    likes = int(data.get("likes") or 0)
    comments = int(data.get("comments") or 0)
    trend_score = float(data.get("trend_score") or data.get("score") or 0.0)
    engagement_signal = float((views * 0.2) + (likes * 1.5) + (comments * 3.0))
    if engagement_signal <= 0:
        engagement_signal = trend_score

    return {
        "key": _stable_key(platform, data),
        "platform": platform,
        "title": title,
        "category": category,
        "source_platform": source_platform or platform,
        "video_url": str(data.get("video_url") or data.get("url") or "").strip(),
        "views": views,
        "likes": likes,
        "comments": comments,
        "trend_score": trend_score,
        "engagement_signal": engagement_signal,
        "published_at": published_at,
    }


def _provider_fetchers(region: str) -> Dict[str, ProviderFetcher]:
    return {
        "youtube": lambda _region, limit: get_youtube_trending(
            region=_region,
            limit=limit,
            mode="live",
        ),
        "google": lambda _region, limit: get_google_trending(
            region=_region,
            limit=limit,
            mode="live",
        ),
        "tiktok": lambda _region, limit: get_tiktok_trending(
            region=_region,
            limit=limit,
            mode="live",
        ),
    }


def _fetch_provider(
    platform: str,
    *,
    region: str,
    limit: int,
    fetcher: ProviderFetcher,
) -> Dict[str, object]:
    started = time.perf_counter()
    mode, raw_items = fetcher(region, limit)
    items = [
        normalize_live_item(platform, item)
        for item in raw_items
        if str(_item_to_dict(item).get("title") or _item_to_dict(item).get("query") or "").strip()
    ]
    return {
        "status": "ok" if items else "empty",
        "mode": mode,
        "total": len(items),
        "duration_ms": round((time.perf_counter() - started) * 1000),
        "items": items,
    }


def _cleanup_old_runs(db: Session, *, region: str) -> None:
    old_runs = (
        db.query(TrendSnapshotRun)
        .filter(TrendSnapshotRun.region == region)
        .order_by(TrendSnapshotRun.run_id.desc())
        .offset(settings.live_trend_snapshot_retention_runs)
        .all()
    )
    for run in old_runs:
        db.delete(run)


def refresh_global_live_trends(
    *,
    region: str | None = None,
    limit: int | None = None,
    platforms: Iterable[str] = PLATFORMS,
    fetchers: Mapping[str, ProviderFetcher] | None = None,
    db: Session | None = None,
) -> Dict[str, object]:
    region = (region or settings.youtube_region).upper()
    limit = min(100, max(1, int(limit or settings.live_trend_limit)))
    selected_platforms = [item.lower() for item in platforms if item.lower() in PLATFORMS]

    if not _refresh_lock.acquire(blocking=False):
        return {"status": "busy", "region": region, "total_items": 0, "providers": {}}

    owns_db = db is None
    session = db or SessionLocal()
    run: TrendSnapshotRun | None = None
    try:
        started_at = datetime.utcnow()
        run = TrendSnapshotRun(
            region=region,
            status="running",
            provider_status="{}",
            total_items=0,
            started_at=started_at,
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        configured_fetchers = dict(fetchers or _provider_fetchers(region))
        executor = ThreadPoolExecutor(max_workers=max(1, len(selected_platforms)))
        futures: Dict[Future[Dict[str, object]], str] = {}
        for platform in selected_platforms:
            fetcher = configured_fetchers.get(platform)
            if fetcher is None:
                continue
            futures[
                executor.submit(
                    _fetch_provider,
                    platform,
                    region=region,
                    limit=limit,
                    fetcher=fetcher,
                )
            ] = platform

        timeout_seconds = settings.live_trend_provider_timeout_seconds
        done, pending = wait(futures, timeout=timeout_seconds)
        provider_results: Dict[str, Dict[str, object]] = {}

        for future in done:
            platform = futures[future]
            try:
                provider_results[platform] = future.result()
            except Exception as exc:
                provider_results[platform] = {
                    "status": "error",
                    "mode": "live_error",
                    "total": 0,
                    "error": str(exc),
                    "items": [],
                }

        for future in pending:
            platform = futures[future]
            future.cancel()
            provider_results[platform] = {
                "status": "error",
                "mode": "live_error",
                "total": 0,
                "error": f"Provider exceeded {timeout_seconds:g}s timeout",
                "items": [],
            }
        executor.shutdown(wait=False, cancel_futures=True)

        for platform in selected_platforms:
            provider_results.setdefault(
                platform,
                {
                    "status": "error",
                    "mode": "live_error",
                    "total": 0,
                    "error": "Provider fetcher is not configured",
                    "items": [],
                },
            )

        total_items = 0
        for platform, result in provider_results.items():
            for item in result.pop("items", []):
                session.add(
                    TrendSnapshotItem(
                        run_id=run.run_id,
                        platform=platform,
                        trend_key=str(item["key"]),
                        title=str(item["title"])[:500],
                        category=str(item["category"])[:100],
                        source_platform=str(item["source_platform"])[:100],
                        video_url=str(item["video_url"])[:1024] or None,
                        views=int(item["views"]),
                        likes=int(item["likes"]),
                        comments=int(item["comments"]),
                        trend_score=float(item["trend_score"]),
                        engagement_signal=float(item["engagement_signal"]),
                        published_at=str(item["published_at"])[:64] if item["published_at"] else None,
                    )
                )
                total_items += 1

        error_count = sum(1 for result in provider_results.values() if result["status"] == "error")
        if error_count == len(provider_results):
            overall_status = "failed"
        elif error_count:
            overall_status = "partial"
        else:
            overall_status = "completed"

        completed_at = datetime.utcnow()
        run.status = overall_status
        run.provider_status = json.dumps(provider_results, ensure_ascii=False, default=str)
        run.total_items = total_items
        run.completed_at = completed_at
        _cleanup_old_runs(session, region=region)
        session.commit()

        return {
            "run_id": run.run_id,
            "status": overall_status,
            "region": region,
            "total_items": total_items,
            "providers": provider_results,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
        }
    except Exception as exc:
        session.rollback()
        if run is not None and run.run_id is not None:
            run = session.get(TrendSnapshotRun, run.run_id)
            if run is not None:
                run.status = "failed"
                run.provider_status = json.dumps({"scheduler": {"status": "error", "error": str(exc)}})
                run.completed_at = datetime.utcnow()
                session.commit()
        raise
    finally:
        if owns_db:
            session.close()
        _refresh_lock.release()


def _provider_status_for_run(run: TrendSnapshotRun) -> Dict[str, Dict[str, object]]:
    try:
        raw = json.loads(run.provider_status or "{}")
    except Exception:
        raw = {}
    return raw if isinstance(raw, dict) else {}


def load_latest_live_snapshot(
    db: Session,
    *,
    region: str,
    limit: int,
) -> Dict[str, object]:
    region = region.upper()
    latest = (
        db.query(TrendSnapshotRun)
        .filter(
            TrendSnapshotRun.region == region,
            TrendSnapshotRun.status.in_(("completed", "partial", "failed")),
        )
        .order_by(TrendSnapshotRun.run_id.desc())
        .first()
    )
    if latest is None:
        return {
            "run_id": None,
            "snapshot_status": "pending",
            "generated_at": None,
            "region": region,
            "platforms": {
                platform: {
                    "mode": "pending",
                    "provider_status": "pending",
                    "total": 0,
                    "items": [],
                    "new_items": [],
                    "error": "Waiting for the first background live-trend snapshot",
                }
                for platform in PLATFORMS
            },
        }

    provider_status = _provider_status_for_run(latest)
    rows = (
        db.query(TrendSnapshotItem)
        .filter(TrendSnapshotItem.run_id == latest.run_id)
        .order_by(TrendSnapshotItem.platform, TrendSnapshotItem.engagement_signal.desc())
        .all()
    )
    grouped: Dict[str, List[Dict[str, object]]] = {platform: [] for platform in PLATFORMS}
    for row in rows:
        if row.platform not in grouped or len(grouped[row.platform]) >= limit:
            continue
        grouped[row.platform].append(
            {
                "key": row.trend_key,
                "platform": row.platform,
                "title": row.title,
                "category": row.category,
                "source_platform": row.source_platform,
                "video_url": row.video_url or "",
                "views": row.views,
                "likes": row.likes,
                "comments": row.comments,
                "trend_score": row.trend_score,
                "engagement_signal": row.engagement_signal,
                "engagement_change_percent": 0.0,
                "status": "Stable",
                "is_new": False,
                "published_at": row.published_at,
            }
        )

    generated_at = latest.completed_at or latest.started_at
    platforms_payload: Dict[str, Dict[str, object]] = {}
    for platform in PLATFORMS:
        details = provider_status.get(platform, {})
        provider_state = str(details.get("status") or "error")
        platforms_payload[platform] = {
            "mode": "live" if provider_state in {"ok", "empty"} else "live_error",
            "provider_status": provider_state,
            "total": len(grouped[platform]),
            "items": grouped[platform],
            "new_items": [],
            "error": details.get("error"),
            "last_checked_at": generated_at.isoformat() if generated_at else None,
        }

    return {
        "run_id": latest.run_id,
        "snapshot_status": latest.status,
        "generated_at": generated_at.isoformat() if generated_at else None,
        "region": latest.region,
        "platforms": platforms_payload,
    }


def get_live_provider_health(db: Session, *, region: str) -> Dict[str, object]:
    snapshot = load_latest_live_snapshot(db, region=region, limit=1)
    providers = {
        platform: {
            "status": details.get("provider_status", "pending"),
            "mode": details.get("mode", "pending"),
            "last_checked_at": details.get("last_checked_at"),
            "error": details.get("error"),
        }
        for platform, details in snapshot["platforms"].items()
    }
    return {
        "run_id": snapshot.get("run_id"),
        "snapshot_status": snapshot.get("snapshot_status"),
        "generated_at": snapshot.get("generated_at"),
        "providers": providers,
    }
