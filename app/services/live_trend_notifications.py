from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Dict, Iterable, List

from sqlalchemy.orm import Session

from app.database.models import Notification, User, UserTrendSnapshot
from app.services.notifications import create_notification
from app.services.trends import get_google_trending, get_tiktok_trending, get_youtube_trending


PLATFORMS = ("youtube", "google", "tiktok")


def _item_to_dict(item: object) -> Dict[str, object]:
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump") and callable(getattr(item, "model_dump")):
        return item.model_dump()
    if hasattr(item, "dict") and callable(getattr(item, "dict")):
        return item.dict()
    return {}


def _stable_key(platform: str, item: Dict[str, object]) -> str:
    title = str(item.get("title") or "").strip().lower()
    url = str(item.get("video_url") or item.get("url") or "").strip().lower()
    raw_key = f"{platform}|{url or title}"
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()


def _normalize_live_item(platform: str, item: object) -> Dict[str, object]:
    data = _item_to_dict(item)
    title = str(data.get("title") or "").strip()
    category = str(data.get("category") or data.get("domain") or "general").strip() or "general"
    source_platform = str(data.get("source_platform") or data.get("source") or platform).strip()
    video_url = str(data.get("video_url") or data.get("url") or "").strip()
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

    normalized = {
        "key": _stable_key(platform, data),
        "platform": platform,
        "title": title,
        "category": category,
        "source_platform": source_platform,
        "video_url": video_url,
        "trend_score": trend_score,
        "engagement_signal": engagement_signal,
        "engagement_change_percent": 0.0,
        "status": "Stable",
        "is_new": False,
        "views": views,
        "likes": likes,
        "comments": comments,
        "published_at": published_at,
    }
    return normalized


def _fetch_platform(platform: str, *, region: str, limit: int) -> tuple[str, List[Dict[str, object]]]:
    if platform == "youtube":
        mode, items = get_youtube_trending(region=region, limit=limit, mode="live")
    elif platform == "google":
        mode, items = get_google_trending(region=region, limit=limit, mode="live")
    elif platform == "tiktok":
        mode, items = get_tiktok_trending(region=region, limit=limit, mode="live")
    else:
        return "unsupported", []
    return mode, [_normalize_live_item(platform, item) for item in items if _item_to_dict(item).get("title")]


def _load_seen_keys(snapshot: UserTrendSnapshot | None) -> set[str]:
    if snapshot is None:
        return set()
    try:
        raw = json.loads(snapshot.item_keys or "[]")
    except Exception:
        return set()
    if not isinstance(raw, list):
        return set()
    return {str(item) for item in raw if item}


def _upsert_snapshot(
    db: Session,
    *,
    user_id: int,
    platform: str,
    items: List[Dict[str, object]],
    checked_at: datetime,
) -> UserTrendSnapshot:
    item_keys = [str(item["key"]) for item in items if item.get("key")]
    payload = [
        {
            "key": item.get("key"),
            "title": item.get("title"),
            "category": item.get("category"),
            "source_platform": item.get("source_platform"),
            "video_url": item.get("video_url"),
            "trend_score": item.get("trend_score"),
            "engagement_signal": item.get("engagement_signal"),
            "status": item.get("status"),
        }
        for item in items
    ]
    snapshot = (
        db.query(UserTrendSnapshot)
        .filter(
            UserTrendSnapshot.user_id == user_id,
            UserTrendSnapshot.platform == platform,
        )
        .first()
    )
    if snapshot is None:
        snapshot = UserTrendSnapshot(
            user_id=user_id,
            platform=platform,
            item_keys=json.dumps(item_keys, ensure_ascii=False),
            snapshot_payload=json.dumps(payload, ensure_ascii=False),
            last_checked_at=checked_at,
            created_at=checked_at,
            updated_at=checked_at,
        )
        db.add(snapshot)
    else:
        snapshot.item_keys = json.dumps(item_keys, ensure_ascii=False)
        snapshot.snapshot_payload = json.dumps(payload, ensure_ascii=False)
        snapshot.last_checked_at = checked_at
        snapshot.updated_at = checked_at
    db.flush()
    return snapshot


def _notification_exists(db: Session, *, user_id: int, platform: str, key: str) -> bool:
    pattern = f'"trend_key": "{key}"'
    return (
        db.query(Notification)
        .filter(
            Notification.user_id == user_id,
            Notification.type == "new_live_trend",
            Notification.source_platform.like(f"{platform}%"),
            Notification.payload.like(f"%{pattern}%"),
        )
        .first()
        is not None
    )


def _create_live_trend_notification(
    db: Session,
    *,
    user: User,
    platform: str,
    item: Dict[str, object],
    detected_at: datetime,
) -> Notification | None:
    key = str(item.get("key") or "")
    if not key or _notification_exists(db, user_id=user.user_id, platform=platform, key=key):
        return None

    title = str(item.get("title") or "Untitled trend")
    category = str(item.get("category") or "general")
    source_platform = str(item.get("source_platform") or platform)
    detected_text = detected_at.isoformat()
    return create_notification(
        db=db,
        user_id=user.user_id,
        title=f"New trend detected: {title[:140]}",
        body=f"{platform.title()} has a new live trend in {category}.",
        message=f"New trend detected: {title} ({platform.title()} / {category})",
        link=str(item.get("video_url") or ""),
        type="new_live_trend",
        topic=category,
        source_platform=source_platform,
        trend_score=float(item.get("trend_score") or 0.0),
        payload={
            "trend_key": key,
            "platform": platform,
            "title": title,
            "category": category,
            "detected_at": detected_text,
            "views": item.get("views", 0),
            "likes": item.get("likes", 0),
            "comments": item.get("comments", 0),
            "published_at": item.get("published_at"),
        },
    )


def _previous_payload_by_key(snapshot: UserTrendSnapshot | None) -> Dict[str, Dict[str, object]]:
    if snapshot is None:
        return {}
    try:
        raw = json.loads(snapshot.snapshot_payload or "[]")
    except Exception:
        return {}
    if not isinstance(raw, list):
        return {}
    result: Dict[str, Dict[str, object]] = {}
    for item in raw:
        if isinstance(item, dict) and item.get("key"):
            result[str(item["key"])] = item
    return result


def _apply_engagement_status(items: List[Dict[str, object]], previous: Dict[str, Dict[str, object]], new_keys: set[str]) -> None:
    signals = [float(item.get("engagement_signal") or 0.0) for item in items]
    max_signal = max(signals) if signals else 0.0
    for item in items:
        key = str(item.get("key") or "")
        current = float(item.get("engagement_signal") or 0.0)
        old = previous.get(key)
        old_signal = float(old.get("engagement_signal") or old.get("trend_score") or 0.0) if old else 0.0
        change = 0.0
        if old_signal > 0:
            change = ((current - old_signal) / old_signal) * 100.0
        elif key in new_keys:
            change = 100.0

        if key in new_keys:
            status = "Rising"
        elif change >= 15:
            status = "Rising"
        elif change <= -10:
            status = "Cooling"
        elif max_signal > 0 and current >= max_signal * 0.75:
            status = "Hot"
        else:
            status = "Stable"

        item["engagement_change_percent"] = round(change, 1)
        item["status"] = status
        item["is_new"] = key in new_keys


def compare_live_trend_snapshot(
    db: Session,
    *,
    user: User,
    region: str,
    limit: int,
    platforms: Iterable[str] = PLATFORMS,
) -> Dict[str, object]:
    detected_at = datetime.utcnow()
    platform_results: Dict[str, object] = {}
    new_notifications: List[Notification] = []

    for platform in platforms:
        platform = platform.strip().lower()
        if platform not in PLATFORMS:
            continue

        try:
            mode, items = _fetch_platform(platform, region=region, limit=limit)
        except Exception as exc:
            platform_results[platform] = {
                "mode": "live_error",
                "total": 0,
                "items": [],
                "new_items": [],
                "error": str(exc),
            }
            continue

        snapshot = (
            db.query(UserTrendSnapshot)
            .filter(
                UserTrendSnapshot.user_id == user.user_id,
                UserTrendSnapshot.platform == platform,
            )
            .first()
        )
        previous_keys = _load_seen_keys(snapshot)
        previous_payload = _previous_payload_by_key(snapshot)
        is_baseline = snapshot is None
        new_items = []
        if not is_baseline:
            new_items = [item for item in items if str(item.get("key") or "") not in previous_keys]
            new_keys = {str(item.get("key") or "") for item in new_items}
            _apply_engagement_status(items, previous_payload, new_keys)
            for item in new_items:
                notification = _create_live_trend_notification(
                    db=db,
                    user=user,
                    platform=platform,
                    item=item,
                    detected_at=detected_at,
                )
                if notification is not None:
                    new_notifications.append(notification)
        else:
            _apply_engagement_status(items, previous_payload, set())

        if mode == "live" and items:
            _upsert_snapshot(
                db=db,
                user_id=user.user_id,
                platform=platform,
                items=items,
                checked_at=detected_at,
            )
            db.commit()

        platform_results[platform] = {
            "mode": mode,
            "baseline": is_baseline,
            "total": len(items),
            "items": items,
            "new_items": new_items,
            "last_checked_at": detected_at.isoformat(),
        }

    return {
        "generated_at": detected_at.isoformat(),
        "region": region,
        "new_count": len(new_notifications),
        "new_notifications": new_notifications,
        "platforms": platform_results,
    }
