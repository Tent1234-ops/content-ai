import json
import random
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.database.db import SessionLocal
from app.services.jobs import enqueue
from app.services.persistence import log_system_event, save_trending_items
from app.services.simple_cache import get as cache_get, set as cache_set
from app.services.trends import get_google_trending, get_youtube_trending


_RATE_LIMIT_SECONDS = {
    "youtube": 60,
    "google": 60,
}
_DEFAULT_SOURCES = ["youtube", "google"]
_CACHE_TTL_SECONDS = 60
_FETCH_INTERVAL_MIN = 60
_FETCH_INTERVAL_MAX = 120

_last_fetch: Dict[str, datetime] = {}
_lock = threading.Lock()
_fetch_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


class RateLimitedError(Exception):
    pass


def _now() -> datetime:
    return datetime.utcnow()


def _normalize_trend_item(item: object, source_name: str) -> Dict[str, object]:
    if hasattr(item, "dict") and callable(getattr(item, "dict")):
        data = item.dict()
    elif hasattr(item, "__dict__"):
        data = {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
    else:
        data = {}

    keyword = data.get("query") or data.get("title") or data.get("channel_title") or data.get("creator") or source_name
    keyword = str(keyword).strip()
    if not keyword:
        keyword = source_name

    domain = data.get("category") or data.get("source_platform") or data.get("channel_title") or "general"
    domain = str(domain).strip() if domain is not None else "general"

    score = float(data.get("trend_score") or data.get("score") or 0.0)
    fetched_at = _now()
    meta = json.dumps({**data, "source_name": source_name}, ensure_ascii=False, default=str)

    return {
        "keyword": keyword,
        "score": score,
        "source": source_name,
        "domain": domain,
        "fetched_at": fetched_at,
        "meta": meta,
    }


def _get_rate_limit_for_source(source: str) -> int:
    return _RATE_LIMIT_SECONDS.get(source, 60)


def _should_fetch(source: str) -> bool:
    with _lock:
        last = _last_fetch.get(source)
        if last is None:
            return True
        return (datetime.utcnow() - last).total_seconds() >= _get_rate_limit_for_source(source)


def _touch_source_fetch(source: str) -> None:
    with _lock:
        _last_fetch[source] = datetime.utcnow()


def _cache_fetch_result(result: Dict[str, object]) -> None:
    cache_set("trending_fetcher:last_result", result, ttl_seconds=_CACHE_TTL_SECONDS)


def get_cached_fetch_result() -> Optional[Dict[str, object]]:
    return cache_get("trending_fetcher:last_result")


def _fetch_source_items(source: str, mode: str, limit: int) -> Tuple[str, List[object]]:
    if source == "youtube":
        return get_youtube_trending(region=settings.youtube_region, limit=limit, mode=mode)
    if source == "google":
        return get_google_trending(region=settings.google_region, limit=limit, mode=mode)
    raise ValueError(f"Unsupported source: {source}")


def fetch_trending_items(
    limit: int = 10,
    mode: str = "auto",
    sources: Optional[List[str]] = None,
    user_id: Optional[int] = None,
) -> Dict[str, object]:
    sources = sources or _DEFAULT_SOURCES
    stats: Dict[str, object] = {"created": 0, "updated": 0, "skipped": 0, "sources": {}}
    normalized_items: List[Dict[str, object]] = []

    for source in sources:
        try:
            if not _should_fetch(source):
                stats["skipped"] += 1
                stats["sources"][source] = {"status": "rate_limited"}
                continue

            source_mode, items = _fetch_source_items(source, mode=mode, limit=limit)
            normalized = [_normalize_trend_item(item, source_name=source) for item in items]
            normalized_items.extend(normalized)
            stats["sources"][source] = {
                "mode": source_mode,
                "total_fetched": len(normalized),
            }
            _touch_source_fetch(source)
        except Exception as exc:
            stats["sources"][source] = {"status": "error", "error": str(exc)}
            log_system_event(
                db=SessionLocal(),
                user_id=user_id,
                action=f"trending_fetch_{source}",
                status="failed",
                detail=str(exc),
            )

    if normalized_items:
        db = SessionLocal()
        try:
            save_stats = save_trending_items(db=db, items=normalized_items, user_id=user_id)
            stats["created"] = save_stats.get("created", 0)
            stats["updated"] = save_stats.get("updated", 0)
        finally:
            db.close()

    stats["fetched_at"] = datetime.utcnow().isoformat()
    _cache_fetch_result(stats)
    return stats


def fetch_trending_items_job(
    limit: int = 10,
    mode: str = "auto",
    sources: Optional[List[str]] = None,
    user_id: Optional[int] = None,
) -> Dict[str, object]:
    return fetch_trending_items(limit=limit, mode=mode, sources=sources, user_id=user_id)


def trigger_trending_refresh(
    limit: int = 10,
    mode: str = "auto",
    sources: Optional[List[str]] = None,
    user_id: Optional[int] = None,
) -> Dict[str, object]:
    sources = sources or _DEFAULT_SOURCES
    now = _now()
    next_allowed = None
    for source in sources:
        with _lock:
            last = _last_fetch.get(source)
        if last is not None:
            elapsed = (now - last).total_seconds()
            wait = _get_rate_limit_for_source(source) - elapsed
            if wait > 0:
                next_allowed = (now + timedelta(seconds=wait)).isoformat()
                raise RateLimitedError(
                    f"Refresh for {source} is rate-limited. Try again after {round(wait)}s."
                )

    job_id = enqueue(fetch_trending_items_job, limit=limit, mode=mode, sources=sources, user_id=user_id)
    return {"job_id": job_id, "status": "queued", "rate_limited": False}


def _fetch_loop() -> None:
    while not _stop_event.is_set():
        try:
            fetch_trending_items(limit=10, mode="auto", sources=_DEFAULT_SOURCES)
        except Exception as exc:
            db = SessionLocal()
            try:
                log_system_event(
                    db=db,
                    user_id=None,
                    action="trending_fetcher_loop",
                    status="failed",
                    detail=str(exc),
                )
                db.commit()
            finally:
                db.close()
        wait_seconds = random.uniform(_FETCH_INTERVAL_MIN, _FETCH_INTERVAL_MAX)
        _stop_event.wait(wait_seconds)


def start_trending_fetcher() -> None:
    global _fetch_thread
    if _fetch_thread and _fetch_thread.is_alive():
        return
    _stop_event.clear()
    _fetch_thread = threading.Thread(target=_fetch_loop, daemon=True)
    _fetch_thread.start()


def stop_trending_fetcher() -> None:
    _stop_event.set()
    global _fetch_thread
    if _fetch_thread:
        _fetch_thread.join(timeout=5)
        _fetch_thread = None
